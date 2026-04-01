# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Session router — transparent WebSocket proxy distributing connections
across multiple PersonaPlex worker processes.

Accepts client WebSocket connections, selects the least-loaded healthy
worker, and relays frames bidirectionally. Clients connect only to the
router address and have no knowledge of individual worker endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from aiohttp import web

from .metrics import (
    metrics_handler,
    routed_connections_total,
    worker_healthy,
)
from .utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class WorkerInfo:
    """Tracks a backend worker's address and reported capacity."""

    worker_id: str
    host: str
    port: int
    max_sessions: int = 0
    active_sessions: int = 0
    healthy: bool = True
    last_health_check: float = 0.0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def utilization(self) -> float:
        if self.max_sessions == 0:
            return 1.0
        return self.active_sessions / self.max_sessions


class Router:
    """Distributes WebSocket connections across backend workers.

    Uses least-connections routing: selects the healthy worker with the
    lowest utilization ratio (active_sessions / max_sessions).
    """

    def __init__(
        self,
        workers: list[WorkerInfo],
        health_check_interval: float = 10.0,
    ):
        self.workers = {w.worker_id: w for w in workers}
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def start(self, app: web.Application):
        """Start background health check loop."""
        self._http_session = aiohttp.ClientSession()
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self, app: web.Application):
        """Stop health check loop and close HTTP session."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        if self._http_session is not None:
            await self._http_session.close()

    def _select_worker(self) -> Optional[WorkerInfo]:
        """Select the healthy worker with the lowest utilization."""
        candidates = [
            w for w in self.workers.values()
            if w.healthy and w.active_sessions < w.max_sessions
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.utilization)

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        """Transparent WebSocket proxy to a backend worker."""
        worker = self._select_worker()
        if worker is None:
            raise web.HTTPServiceUnavailable(
                text="No healthy workers with available capacity"
            )

        # Build backend URL preserving query params
        query = request.query_string
        backend_url = f"ws://{worker.address}/api/chat"
        if query:
            backend_url += f"?{query}"

        # Accept client WebSocket
        client_ws = web.WebSocketResponse()
        await client_ws.prepare(request)

        routed_connections_total.labels(worker=worker.worker_id).inc()
        worker.active_sessions += 1
        logger.info(
            f"routing to {worker.worker_id} "
            f"({worker.active_sessions}/{worker.max_sessions})"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(backend_url) as backend_ws:
                    await self._relay(client_ws, backend_ws)
        except Exception as e:
            logger.error(f"relay error for {worker.worker_id}: {e}")
        finally:
            worker.active_sessions = max(0, worker.active_sessions - 1)
            if not client_ws.closed:
                await client_ws.close()

        return client_ws

    async def _relay(
        self,
        client_ws: web.WebSocketResponse,
        backend_ws: aiohttp.ClientWebSocketResponse,
    ):
        """Relay frames bidirectionally between client and backend."""

        async def client_to_backend():
            async for msg in client_ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await backend_ws.send_bytes(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await backend_ws.send_str(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break

        async def backend_to_client():
            async for msg in backend_ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await client_ws.send_bytes(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await client_ws.send_str(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break

        tasks = [
            asyncio.create_task(client_to_backend()),
            asyncio.create_task(backend_to_client()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        """Periodically poll each worker's /health endpoint."""
        while True:
            await asyncio.sleep(self.health_check_interval)
            for w in self.workers.values():
                try:
                    url = f"http://{w.address}/health"
                    async with self._http_session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            w.max_sessions = data.get("max_sessions", w.max_sessions)
                            w.active_sessions = data.get("active_sessions", w.active_sessions)
                            if not w.healthy:
                                logger.info(f"worker {w.worker_id} recovered")
                            w.healthy = True
                        else:
                            w.healthy = False
                            logger.warning(f"worker {w.worker_id} unhealthy: HTTP {resp.status}")
                except Exception as e:
                    if w.healthy:
                        logger.warning(f"worker {w.worker_id} health check failed: {e}")
                    w.healthy = False
                worker_healthy.labels(worker=w.worker_id).set(1 if w.healthy else 0)

    def health_data(self) -> dict:
        """Return aggregated health status for all workers."""
        workers = list(self.workers.values())
        healthy_count = sum(1 for w in workers if w.healthy)
        total_cap = sum(w.max_sessions for w in workers if w.healthy)
        total_active = sum(w.active_sessions for w in workers if w.healthy)
        return {
            "total_workers": len(workers),
            "healthy_workers": healthy_count,
            "total_capacity": total_cap,
            "active_sessions": total_active,
            "available_slots": total_cap - total_active,
        }


async def handle_router_health(request: web.Request) -> web.Response:
    """Return router-level health status."""
    router: Router = request.app["router"]
    return web.json_response(router.health_data())


def parse_worker_arg(arg: str) -> WorkerInfo:
    """Parse 'ID:HOST:PORT' into a WorkerInfo."""
    parts = arg.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected ID:HOST:PORT, got '{arg}'"
        )
    worker_id, host, port_str = parts
    return WorkerInfo(worker_id=worker_id, host=host, port=int(port_str))


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex Session Router")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument(
        "--workers", action="append", required=True, type=parse_worker_arg,
        help="Worker address in ID:HOST:PORT format (repeatable)",
    )
    parser.add_argument(
        "--health-check-interval", default=10.0, type=float,
        help="Seconds between worker health checks (default: 10)",
    )
    parser.add_argument(
        "--ssl", type=str,
        help="Directory with key.pem and cert.pem for HTTPS",
    )

    args = parser.parse_args()

    router = Router(
        workers=args.workers,
        health_check_interval=args.health_check_interval,
    )

    app = web.Application()
    app["router"] = router
    app.on_startup.append(router.start)
    app.on_cleanup.append(router.stop)
    app.router.add_get("/api/chat", router.handle_chat)
    app.router.add_get("/health", handle_router_health)
    app.router.add_get("/metrics", metrics_handler)

    ssl_context = None
    if args.ssl is not None:
        from .utils.connection import create_ssl_context
        ssl_context, _ = create_ssl_context(args.ssl)

    logger.info(
        f"router starting on {args.host}:{args.port} "
        f"with {len(args.workers)} workers"
    )
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
