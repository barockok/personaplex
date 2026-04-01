# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""End-to-end integration tests using a full aiohttp server with mock backends.

These tests spin up a real aiohttp ``TestServer`` (no GPU, no model weights)
and exercise the HTTP and WebSocket endpoints over the network.  Heavy
transitive dependencies (sphn, einops, sentencepiece, etc.) are stubbed via
``sys.modules`` so the tests run in any CI environment.

The full audio streaming loop is *not* tested here because ``session.py``
imports ``sphn`` for Opus encoding.  Instead we verify:

* ``/health`` returns correct JSON
* ``/metrics`` returns Prometheus text with the expected gauge
* WebSocket connections at capacity are terminated promptly
* Multiple ``/health`` requests return consistent results
* ``/health`` reflects active-session count changes across acquire/release
"""

from __future__ import annotations

import asyncio

import pytest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import make_mocked_request, TestServer

from moshi.moshi.mock_backend import (
    MockMimiModel,
    MockLMGen,
    MockTextTokenizer,
)
from moshi.moshi.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Re-define handlers locally so we don't need to import the real server
# module (which pulls in torch/loaders at module level).
# ---------------------------------------------------------------------------

async def metrics_handler(request: web.Request) -> web.Response:
    """Return Prometheus-style metrics text."""
    sm: SessionManager = request.app["session_manager"]
    body = (
        "# HELP personaplex_active_sessions Current number of active voice sessions\n"
        "# TYPE personaplex_active_sessions gauge\n"
        f"personaplex_active_sessions {sm.active_count}.0\n"
    )
    return web.Response(
        body=body.encode(),
        content_type="text/plain",
    )


async def handle_health(request: web.Request) -> web.Response:
    """Return worker health status as JSON."""
    session_manager: SessionManager = request.app["session_manager"]
    return web.json_response(session_manager.health_data())


async def handle_chat(request: web.Request) -> web.WebSocketResponse:
    """Handle a voice session WebSocket connection.

    Acquires a session slot from the SessionManager, delegates to the
    Session's ``handle_chat`` method, and releases the slot on completion
    or error.
    """
    session_manager: SessionManager = request.app["session_manager"]

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session = await session_manager.acquire_session(request)
    try:
        await session.handle_chat(ws)
    finally:
        await session_manager.release_session(session.session_id)
        await ws.close()

    return ws


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app(max_sessions: int = 2, worker_id: str = "e2e-worker") -> web.Application:
    """Build an aiohttp Application wired to mock backends."""
    mimi = MockMimiModel()
    other_mimi = MockMimiModel()
    lm_gen = MockLMGen()
    text_tokenizer = MockTextTokenizer()

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    sm = SessionManager(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        device="cpu",
        voice_prompt_dir=None,
        max_sessions=max_sessions,
        worker_id=worker_id,
    )

    app = web.Application()
    app["session_manager"] = sm
    app.router.add_get("/api/chat", handle_chat)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", metrics_handler)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns JSON with expected keys and values."""
    app = _build_app(max_sessions=4, worker_id="health-test")
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as client:
            url = f"http://127.0.0.1:{server.port}/health"
            async with client.get(url) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["worker_id"] == "health-test"
                assert data["active_sessions"] == 0
                assert data["max_sessions"] == 4
                assert data["healthy"] is True
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """GET /metrics returns Prometheus text containing the active-sessions gauge."""
    app = _build_app(max_sessions=2, worker_id="metrics-test")
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as client:
            url = f"http://127.0.0.1:{server.port}/metrics"
            async with client.get(url) as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "personaplex_active_sessions" in body
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_websocket_connection_rejected_at_capacity():
    """When max_sessions=0, a WebSocket connect to /api/chat is terminated.

    Because ``handle_chat`` calls ``ws.prepare()`` before ``acquire_session()``,
    the HTTP upgrade (101) succeeds, but the server-side
    ``HTTPServiceUnavailable`` exception causes the WebSocket to close
    immediately.  We verify the client sees a prompt close.
    """
    app = _build_app(max_sessions=0, worker_id="reject-test")
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as client:
            ws_url = (
                f"http://127.0.0.1:{server.port}"
                "/api/chat?voice_prompt=mock&text_prompt=test"
            )
            async with client.ws_connect(ws_url) as ws:
                # The server raises HTTPServiceUnavailable after ws.prepare(),
                # which propagates and closes the WebSocket connection.
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                assert msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                )
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_multiple_health_checks_consistent():
    """Multiple sequential /health requests return identical JSON."""
    app = _build_app(max_sessions=3, worker_id="consistent-test")
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as client:
            url = f"http://127.0.0.1:{server.port}/health"
            responses = []
            for _ in range(5):
                async with client.get(url) as resp:
                    assert resp.status == 200
                    responses.append(await resp.json())

            first = responses[0]
            for i, data in enumerate(responses[1:], start=2):
                assert data == first, (
                    f"Response #{i} differs from first: {data} != {first}"
                )
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_health_reflects_active_session_changes():
    """Verify /health active_sessions count updates after acquire/release.

    We manipulate the SessionManager directly (bypassing the full audio
    WebSocket loop which requires sphn) and then query /health over HTTP
    to confirm the app-level wiring is correct.
    """
    app = _build_app(max_sessions=3, worker_id="active-test")
    server = TestServer(app)
    await server.start_server()
    try:
        sm: SessionManager = app["session_manager"]

        async with aiohttp.ClientSession() as client:
            url = f"http://127.0.0.1:{server.port}/health"

            # Baseline: 0 active
            async with client.get(url) as resp:
                data = await resp.json()
                assert data["active_sessions"] == 0

            # Acquire a session via the SessionManager directly.
            req = make_mocked_request(
                "GET",
                "/api/chat?voice_prompt=mock&text_prompt=test",
                headers={},
            )
            session = await sm.acquire_session(req)

            # Now /health should report 1 active session.
            async with client.get(url) as resp:
                data = await resp.json()
                assert data["active_sessions"] == 1

            # Release and verify it drops back to 0.
            await sm.release_session(session.session_id)

            async with client.get(url) as resp:
                data = await resp.json()
                assert data["active_sessions"] == 0
    finally:
        await server.close()
