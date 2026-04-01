"""Tests for moshi.moshi.router — Router class and health endpoint."""

from __future__ import annotations

import pytest
import pytest_asyncio
import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from moshi.moshi.router import Router, WorkerInfo, handle_router_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worker(
    worker_id: str,
    *,
    max_sessions: int = 3,
    active_sessions: int = 0,
    healthy: bool = True,
) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        host="127.0.0.1",
        port=8000,
        max_sessions=max_sessions,
        active_sessions=active_sessions,
        healthy=healthy,
    )


# ---------------------------------------------------------------------------
# _select_worker tests
# ---------------------------------------------------------------------------


def test_select_worker_least_loaded():
    """Router picks the worker with the lowest utilization."""
    worker_a = _worker("worker-a", active_sessions=1, max_sessions=3)
    worker_b = _worker("worker-b", active_sessions=0, max_sessions=3)
    router = Router(workers=[worker_a, worker_b])

    selected = router._select_worker()

    assert selected is not None
    assert selected.worker_id == "worker-b"


def test_select_worker_no_capacity():
    """Router returns None when all workers are at full capacity."""
    worker = _worker("worker-a", active_sessions=3, max_sessions=3)
    router = Router(workers=[worker])

    assert router._select_worker() is None


def test_select_worker_unhealthy_skipped():
    """Router skips unhealthy workers and returns the healthy one."""
    unhealthy = _worker("worker-a", active_sessions=0, healthy=False)
    healthy = _worker("worker-b", active_sessions=1)
    router = Router(workers=[unhealthy, healthy])

    selected = router._select_worker()

    assert selected is not None
    assert selected.worker_id == "worker-b"


# ---------------------------------------------------------------------------
# health_data tests
# ---------------------------------------------------------------------------


def test_health_data():
    """health_data returns correct aggregated metrics for all workers."""
    worker_a = _worker("worker-a", max_sessions=3, active_sessions=1)
    worker_b = _worker("worker-b", max_sessions=5, active_sessions=2)
    router = Router(workers=[worker_a, worker_b])

    data = router.health_data()

    assert data == {
        "total_workers": 2,
        "healthy_workers": 2,
        "total_capacity": 8,
        "active_sessions": 3,
        "available_slots": 5,
    }


def test_health_data_unhealthy_excluded():
    """health_data excludes unhealthy workers from capacity and session counts."""
    healthy = _worker("worker-a", max_sessions=4, active_sessions=1)
    unhealthy = _worker("worker-b", max_sessions=4, active_sessions=2, healthy=False)
    router = Router(workers=[healthy, unhealthy])

    data = router.health_data()

    assert data == {
        "total_workers": 2,
        "healthy_workers": 1,
        "total_capacity": 4,
        "active_sessions": 1,
        "available_slots": 3,
    }


# ---------------------------------------------------------------------------
# HTTP endpoint test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_health_endpoint(mock_server_app):
    """GET /health returns JSON with router health data."""
    worker_a = _worker("worker-a", max_sessions=3, active_sessions=1)
    worker_b = _worker("worker-b", max_sessions=3, active_sessions=0)

    app = web.Application()
    router = Router(workers=[worker_a, worker_b])
    app["router"] = router
    app.router.add_get("/health", handle_router_health)

    server: TestServer = await mock_server_app(app)
    url = f"http://127.0.0.1:{server.port}/health"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            assert resp.status == 200
            body = await resp.json()

    assert body == {
        "total_workers": 2,
        "healthy_workers": 2,
        "total_capacity": 6,
        "active_sessions": 1,
        "available_slots": 5,
    }
