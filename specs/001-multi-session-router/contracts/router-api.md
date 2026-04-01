# Router API Contract

**Version**: 1.0 | **Date**: 2026-04-02

## WebSocket Endpoint: `/api/chat`

Transparent proxy. Accepts the same query parameters and binary
protocol as the worker endpoint. The client connects to the router
exactly as it would connect to a worker — no client-side changes
needed.

### Routing Behavior

1. Client opens WebSocket to router at `/api/chat?...`
2. Router selects the healthy worker with lowest utilization ratio
   (`active_sessions / max_sessions`)
3. Router opens a backend WebSocket to the selected worker,
   forwarding all query parameters
4. All binary frames are relayed bidirectionally (client ↔ router ↔ worker)
5. When either side closes, router closes both connections and the
   worker frees the session slot

### Error Responses (from router)

| Code | Condition |
|------|-----------|
| 503 Service Unavailable | All workers at capacity or no healthy workers |
| 502 Bad Gateway | Selected worker failed to accept the connection |

---

## HTTP Endpoint: `GET /health`

### Response (200 OK)

```json
{
  "total_workers": 3,
  "healthy_workers": 2,
  "total_capacity": 15,
  "active_sessions": 8,
  "available_slots": 7
}
```

---

## HTTP Endpoint: `GET /metrics`

Prometheus-compatible metrics for the router itself.

### Response (200 OK, text/plain)

```
# HELP personaplex_worker_healthy Worker health status
# TYPE personaplex_worker_healthy gauge
personaplex_worker_healthy{worker="worker-1"} 1
personaplex_worker_healthy{worker="worker-2"} 1
personaplex_worker_healthy{worker="worker-3"} 0

# HELP personaplex_routed_connections_total Connections routed
# TYPE personaplex_routed_connections_total counter
personaplex_routed_connections_total{worker="worker-1"} 42
personaplex_routed_connections_total{worker="worker-2"} 38
personaplex_routed_connections_total{worker="worker-3"} 20
```

---

## CLI Configuration

### Worker

```bash
python -m moshi.server \
  --host 0.0.0.0 --port 8998 \
  --max-sessions 5 \
  --worker-id worker-1
```

New flags:
- `--max-sessions N`: Maximum concurrent sessions (default: 1 for
  backward compatibility)
- `--worker-id ID`: Identifier reported in health/metrics endpoints
  (default: hostname)
- `--mock`: Start with mock inference backend (no model loading)

### Router

```bash
python -m moshi.router \
  --host 0.0.0.0 --port 9000 \
  --workers worker-1:localhost:8998 \
  --workers worker-2:localhost:8999 \
  --health-check-interval 10
```

Flags:
- `--workers ID:HOST:PORT`: Worker address (repeatable)
- `--health-check-interval SECONDS`: Health check period (default: 10)
- `--host`, `--port`: Router listen address
