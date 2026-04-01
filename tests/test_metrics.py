"""Tests for Prometheus metrics endpoints."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from moshi.moshi.metrics import (
    active_sessions,
    connection_rejections_total,
    metrics_handler,
)


@pytest.fixture()
def metrics_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)
    return app


# ------------------------------------------------------------------
# 1. /metrics returns valid Prometheus exposition format
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_handler_returns_prometheus_format(metrics_app: web.Application):
    async with TestClient(TestServer(metrics_app)) as client:
        resp = await client.get("/metrics")

        assert resp.status == 200
        assert "text/plain" in resp.content_type
        body = await resp.text()
        assert "personaplex_active_sessions" in body


# ------------------------------------------------------------------
# 2. /metrics body contains all five expected metric names
# ------------------------------------------------------------------

EXPECTED_METRIC_NAMES = [
    "personaplex_active_sessions",
    "personaplex_session_duration_seconds",
    "personaplex_connection_rejections_total",
    "personaplex_worker_healthy",
    "personaplex_routed_connections_total",
]


@pytest.mark.asyncio
async def test_metrics_contains_all_expected_metrics(metrics_app: web.Application):
    async with TestClient(TestServer(metrics_app)) as client:
        resp = await client.get("/metrics")
        body = await resp.text()

        for name in EXPECTED_METRIC_NAMES:
            assert name in body, f"Expected metric {name!r} not found in /metrics output"


# ------------------------------------------------------------------
# 3. active_sessions Gauge can be incremented and decremented
# ------------------------------------------------------------------


def test_active_sessions_gauge_can_inc_dec():
    before = active_sessions._value.get()

    active_sessions.inc()
    after_inc = active_sessions._value.get()
    assert after_inc == before + 1

    active_sessions.dec()
    after_dec = active_sessions._value.get()
    assert after_dec == before


# ------------------------------------------------------------------
# 4. connection_rejections_total Counter increments
# ------------------------------------------------------------------


def test_connection_rejections_counter_increments():
    before = connection_rejections_total._value.get()

    connection_rejections_total.inc()
    after = connection_rejections_total._value.get()
    assert after == before + 1
