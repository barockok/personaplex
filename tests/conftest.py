from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy transitive imports BEFORE any moshi module is loaded.
# moshi.__init__ eagerly imports modules, models, quantization which pull in
# einops, safetensors, huggingface_hub, sphn, sentencepiece, tqdm, etc.
# For tests that only need mock_backend/session/session_manager, we stub
# these so tests run without GPU libraries.
# ---------------------------------------------------------------------------
_HEAVY_STUBS = [
    "einops",
    "safetensors",
    "safetensors.torch",
    "huggingface_hub",
    "sounddevice",
    "sentencepiece",
    "sphn",
    "tqdm",
    "tqdm.auto",
]
for _mod_name in _HEAVY_STUBS:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Stub moshi.moshi.server to avoid importing the full server module
# (which has heavy top-level imports). Tests that need server functions
# should import them explicitly after this stub is in place.
if "moshi.moshi.server" not in sys.modules:
    _server_stub = types.ModuleType("moshi.moshi.server")
    _server_stub.wrap_with_system_tags = lambda t: t
    _server_stub.seed_all = lambda s: None
    sys.modules["moshi.moshi.server"] = _server_stub

import socket
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest.fixture()
def free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture()
async def mock_ws_client():
    """Factory fixture that creates an aiohttp WebSocket client for a given URL.

    Yields a callable that returns an async context manager. Usage::

        async with mock_ws_client("ws://127.0.0.1:1234/ws") as ws:
            await ws.send_str("hello")
            msg = await ws.receive()
    """
    sessions: list[aiohttp.ClientSession] = []

    @asynccontextmanager
    async def _connect(url: str) -> AsyncIterator[aiohttp.ClientWebSocketResponse]:
        session = aiohttp.ClientSession()
        sessions.append(session)
        async with session.ws_connect(url) as ws:
            yield ws

    yield _connect

    for session in sessions:
        await session.close()


@pytest_asyncio.fixture()
async def mock_server_app():
    """Factory fixture that starts an aiohttp TestServer for a given Application.

    Yields a callable that accepts a ``web.Application`` and returns the
    running ``TestServer`` instance. The server is closed automatically after
    the test. Usage::

        app = web.Application()
        server = await mock_server_app(app)
        url = f"http://127.0.0.1:{server.port}"
    """
    servers: list[TestServer] = []

    async def _start(app: web.Application) -> TestServer:
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return server

    yield _start

    for server in servers:
        await server.close()
