# Data Model: Multi-Session Router & Concurrent Inference

**Branch**: `001-multi-session-router` | **Date**: 2026-04-02

## Entities

### Session

Represents one active voice conversation within a worker process.

| Field | Type | Description |
|-------|------|-------------|
| session_id | string (UUID) | Unique identifier for this session |
| worker_id | string | ID of the worker hosting this session |
| voice_prompt | string | Path to the voice prompt file |
| text_prompt | string | System text prompt for the persona |
| seed | int or null | Random seed for reproducibility |
| status | enum | `connecting`, `active`, `disconnecting`, `closed` |
| created_at | float | Unix timestamp of session creation |
| client_ip | string | Remote IP address of the client |

**Relationships**: Belongs to one Worker. Has one WebSocket connection.

**Lifecycle**:

```
connecting → active → disconnecting → closed
     │                      ↑
     └──────────────────────┘ (error during connect)
```

- `connecting`: WebSocket accepted, streaming context being
  initialized, system prompts processing.
- `active`: Handshake sent, audio streaming bidirectionally.
- `disconnecting`: Client disconnected or error; cleaning up
  streaming state and releasing slot.
- `closed`: All resources freed, slot available for reuse.

**State isolation per session**:
- Mimi encoder streaming state (~200 KB)
- Mimi decoder streaming state (~200 KB)
- LMGen streaming state: token cache, provided mask, offset,
  CUDAGraphed objects (~400 KB)
- Opus reader/writer instances
- Per-session voice prompt embeddings (if loaded)

### Worker

A server process capable of hosting concurrent sessions.

| Field | Type | Description |
|-------|------|-------------|
| worker_id | string | Unique identifier (from config) |
| address | string | `host:port` for WebSocket connections |
| max_sessions | int | Maximum concurrent sessions allowed |
| active_sessions | int | Current number of active sessions |
| healthy | bool | Whether the worker responds to health checks |
| last_health_check | float | Unix timestamp of last successful check |

**Relationships**: Hosts 0..N Sessions. Registered with one Router.

**Uniqueness**: `worker_id` and `address` are both unique within a
Router's configuration.

### Router

The entry point that distributes client connections to workers.

| Field | Type | Description |
|-------|------|-------------|
| workers | list[Worker] | Statically configured worker pool |
| health_check_interval | float | Seconds between health checks |

**Relationships**: Manages 1..N Workers. Proxies 0..M client
connections.

**Routing algorithm**: Least-connections — route to the worker
with the lowest `active_sessions / max_sessions` ratio among
healthy workers.

## Metrics Entities

### Session Metrics (per worker)

| Metric | Type | Description |
|--------|------|-------------|
| personaplex_active_sessions | Gauge | Current active session count |
| personaplex_session_duration_seconds | Histogram | Session duration distribution |
| personaplex_connection_rejections_total | Counter | Rejected connections (at capacity) |

### Router Metrics

| Metric | Type | Description |
|--------|------|-------------|
| personaplex_worker_healthy | Gauge (labeled) | Per-worker health status (0/1) |
| personaplex_routed_connections_total | Counter (labeled) | Connections routed per worker |
