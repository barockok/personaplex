# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for SessionManager session pool lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from moshi.moshi.mock_backend import MockMimiModel, MockLMGen, MockTextTokenizer
from moshi.moshi.session_manager import SessionManager

from aiohttp import web
from aiohttp.test_utils import make_mocked_request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(voice_prompt: str = "mock", text_prompt: str = "test") -> web.Request:
    """Build a minimal mocked aiohttp request with required query params."""
    path = f"/api/chat?voice_prompt={voice_prompt}&text_prompt={text_prompt}"
    return make_mocked_request("GET", path, headers={})


def _make_session_manager(max_sessions: int = 2, worker_id: str = "test-worker") -> SessionManager:
    """Build a SessionManager wired to mock backends."""
    return SessionManager(
        mimi=MockMimiModel(),
        other_mimi=MockMimiModel(),
        text_tokenizer=MockTextTokenizer(),
        lm_gen=MockLMGen(),
        device="cpu",
        voice_prompt_dir=None,
        max_sessions=max_sessions,
        worker_id=worker_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acquire_session():
    """Acquire one session and verify counters update."""
    mgr = _make_session_manager(max_sessions=2)

    session = await mgr.acquire_session(_make_request())

    assert mgr.active_count == 1
    assert mgr.available_slots == 1
    assert session.worker_id == "test-worker"


@pytest.mark.asyncio
async def test_acquire_multiple_sessions():
    """Acquire max sessions concurrently and verify active_count."""
    mgr = _make_session_manager(max_sessions=2)

    sessions = await asyncio.gather(
        mgr.acquire_session(_make_request()),
        mgr.acquire_session(_make_request()),
    )

    assert len(sessions) == 2
    assert mgr.active_count == 2
    assert mgr.available_slots == 0


@pytest.mark.asyncio
async def test_reject_over_capacity():
    """Third acquire should raise HTTPServiceUnavailable."""
    mgr = _make_session_manager(max_sessions=2)

    await mgr.acquire_session(_make_request())
    await mgr.acquire_session(_make_request())

    with pytest.raises(web.HTTPServiceUnavailable):
        await mgr.acquire_session(_make_request())


@pytest.mark.asyncio
async def test_release_session():
    """Release a session and verify the slot becomes available again."""
    mgr = _make_session_manager(max_sessions=2)

    session = await mgr.acquire_session(_make_request())
    assert mgr.active_count == 1

    await mgr.release_session(session.session_id)
    assert mgr.active_count == 0
    assert mgr.available_slots == 2

    # Slot is freed; acquiring again should succeed.
    session2 = await mgr.acquire_session(_make_request())
    assert mgr.active_count == 1
    assert session2.session_id != session.session_id


@pytest.mark.asyncio
async def test_concurrent_acquire_release():
    """Fill all slots, release one, then acquire again to verify slot reuse."""
    mgr = _make_session_manager(max_sessions=2)

    s1 = await mgr.acquire_session(_make_request())
    s2 = await mgr.acquire_session(_make_request())
    assert mgr.active_count == 2

    # At capacity -- next acquire must fail.
    with pytest.raises(web.HTTPServiceUnavailable):
        await mgr.acquire_session(_make_request())

    # Release one slot.
    await mgr.release_session(s1.session_id)
    assert mgr.active_count == 1

    # Now we can acquire again.
    s3 = await mgr.acquire_session(_make_request())
    assert mgr.active_count == 2
    assert s3.session_id != s1.session_id


@pytest.mark.asyncio
async def test_health_data():
    """health_data() returns correct dict reflecting current state."""
    mgr = _make_session_manager(max_sessions=4, worker_id="node-42")

    data = mgr.health_data()
    assert data == {
        "worker_id": "node-42",
        "active_sessions": 0,
        "max_sessions": 4,
        "healthy": True,
    }

    await mgr.acquire_session(_make_request())

    data = mgr.health_data()
    assert data["active_sessions"] == 1
    assert data["max_sessions"] == 4
    assert data["healthy"] is True
