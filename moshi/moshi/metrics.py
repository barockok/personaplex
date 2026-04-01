"""Prometheus metrics for Personaplex voice session routing."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from aiohttp import web

# Active voice sessions
active_sessions = Gauge(
    "personaplex_active_sessions",
    "Current number of active voice sessions",
)

# Session duration
session_duration_seconds = Histogram(
    "personaplex_session_duration_seconds",
    "Duration of voice sessions in seconds",
    buckets=[10, 30, 60, 120, 300, 600, float("inf")],
)

# Rejected connections
connection_rejections_total = Counter(
    "personaplex_connection_rejections_total",
    "Total number of rejected connections due to capacity",
)

# Worker health
worker_healthy = Gauge(
    "personaplex_worker_healthy",
    "Worker health status (1=healthy, 0=unhealthy)",
    ["worker"],
)

# Routed connections per worker
routed_connections_total = Counter(
    "personaplex_routed_connections_total",
    "Total connections routed to each worker",
    ["worker"],
)


async def metrics_handler(request: web.Request) -> web.Response:
    """Return Prometheus metrics in exposition format."""
    resp = web.Response(body=generate_latest())
    resp.content_type = "text/plain"
    resp.charset = "utf-8"
    return resp
