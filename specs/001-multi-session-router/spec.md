# Feature Specification: Multi-Session Router & Concurrent Inference

**Feature Branch**: `001-multi-session-router`
**Created**: 2026-04-02
**Status**: Draft
**Input**: User description: "building a session router/load balancer and removing the single-session lock, make one process can running multiple process, make it programmatically testable local without require heavy gpu to verify the multiple session per process"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multiple Users Connect Simultaneously (Priority: P1)

Two or more users open the PersonaPlex WebUI and start voice
conversations at the same time, served by the same server process.
Today the `asyncio.Lock` in `ServerState` serializes all sessions:
the second caller waits until the first hangs up. After this feature,
both callers MUST be served concurrently with independent audio
streams and independent model state.

**Why this priority**: This is the core constraint blocking
multi-user hosting. Without it, every concurrent user requires a
separate OS process and full model copy in VRAM.

**Independent Test**: Start the server, connect two WebSocket
clients simultaneously, send audio on both, and verify both receive
audio responses without one blocking the other.

**Acceptance Scenarios**:

1. **Given** a running server with capacity for 2+ sessions,
   **When** two clients connect via WebSocket at the same time,
   **Then** both receive the handshake byte (`0x00`) and begin
   streaming audio within their respective sessions without delay.

2. **Given** two active concurrent sessions,
   **When** one session disconnects,
   **Then** the other session continues uninterrupted and the freed
   slot becomes available for a new connection.

3. **Given** a server configured with a maximum session limit,
   **When** a client attempts to connect beyond that limit,
   **Then** the server rejects the connection with a clear error
   message indicating capacity is full.

---

### User Story 2 - Session Router Distributes Connections (Priority: P2)

An operator runs multiple GPU worker processes (or a single process
with multiple session slots). A session router sits in front of the
workers, accepts incoming WebSocket connections, and forwards each
to an available worker/slot. The router tracks which workers have
capacity and routes new connections accordingly.

**Why this priority**: This is the infrastructure layer that enables
the warm-pool hosting architecture described in the cost analysis.
It depends on US1 (concurrent sessions) being possible first.

**Independent Test**: Start a router and two backend workers (can be
mock workers), connect several clients through the router, and
verify connections are distributed across workers based on
availability.

**Acceptance Scenarios**:

1. **Given** a router with 2 registered workers each having 1 free
   slot, **When** 2 clients connect, **Then** each client is routed
   to a different worker.

2. **Given** a router where all workers are at capacity,
   **When** a new client connects,
   **Then** the router returns a "no capacity" response and does not
   drop existing sessions.

3. **Given** a worker that becomes unavailable (crash or disconnect),
   **When** the router performs a health check,
   **Then** the worker is removed from the pool and no new sessions
   are routed to it.

---

### User Story 3 - Local Testing Without GPU (Priority: P1)

A developer runs the full multi-session server locally on a CPU-only
machine to verify session routing, concurrency, connection lifecycle,
and load balancing logic. The system provides a mock/stub inference
backend that replaces the real model with a lightweight stand-in
(e.g., echoes audio back, returns fixed tokens) so that all
session management code can be exercised without loading the 7B model
or requiring a GPU.

**Why this priority**: Equal to P1 because without local testability,
the concurrency and routing logic cannot be verified during
development. This unblocks rapid iteration on all other stories.

**Independent Test**: Run the server in mock mode on a laptop without
a GPU, connect multiple WebSocket clients, and confirm sessions are
created, routed, and torn down correctly.

**Acceptance Scenarios**:

1. **Given** the server started with a mock inference flag,
   **When** a client connects and sends audio frames,
   **Then** the server responds with audio frames (mock output)
   without loading any model weights or requiring CUDA.

2. **Given** the server in mock mode,
   **When** multiple clients connect concurrently,
   **Then** each session operates independently and the concurrency
   behavior is identical to the real-model path.

3. **Given** the mock mode,
   **When** a developer writes an automated test that connects N
   clients and validates session lifecycle,
   **Then** the test completes in under 10 seconds on a standard
   laptop without GPU.

---

### Edge Cases

- What happens when a client sends malformed audio frames during a
  concurrent session? Other sessions MUST NOT be affected.
- What happens when the server process runs out of memory trying to
  allocate a new session? The server MUST reject the new connection
  gracefully without crashing existing sessions.
- What happens when a WebSocket connection drops mid-stream? The
  session's resources (model state, audio buffers) MUST be cleaned
  up and the slot freed for reuse.
- What happens when two sessions request the same voice prompt?
  Each session MUST have independent model streaming state.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The server MUST support multiple concurrent voice
  sessions within a single process, each with independent audio
  encoding/decoding and language model streaming state.
- **FR-002**: The server MUST expose a configurable maximum session
  limit (e.g., `--max-sessions N`) to control resource usage.
- **FR-003**: The server MUST reject new connections with a clear
  error when the session limit is reached, without affecting
  existing sessions.
- **FR-004**: A session router MUST accept incoming WebSocket
  connections and distribute them to available backend workers
  based on reported capacity. The router operates as a transparent
  proxy, relaying WebSocket frames bidirectionally so that clients
  connect only to the router address and have no knowledge of
  individual worker endpoints.
- **FR-005**: The session router MUST detect unhealthy or
  disconnected workers and stop routing sessions to them.
- **FR-006**: The system MUST provide a mock inference backend that
  can replace the real model for local development and testing,
  requiring no GPU and no model weight downloads.
- **FR-007**: Each session MUST maintain isolated state: voice
  prompt, text prompt, seed, streaming buffers, and model context
  MUST NOT leak between sessions.
- **FR-008**: When a session ends (client disconnect or error), all
  resources associated with that session MUST be released and the
  capacity slot MUST become available for new connections.
- **FR-009**: The session router MUST expose a health/status
  endpoint reporting the number of active sessions, total capacity,
  and worker availability.
- **FR-010**: The system MUST emit structured log lines for each
  session lifecycle event: connect, disconnect, error, with
  session ID, worker ID, duration, and error reason where
  applicable.
- **FR-011**: The system MUST expose a metrics endpoint
  (Prometheus-compatible) reporting: active session count per
  worker, session duration histogram, connection rejection count,
  and worker health status.

### Key Entities

- **Session**: Represents one active voice conversation. Holds
  its own lightweight streaming context (Mimi encoder/decoder
  state, LM generation state, audio buffers) on top of shared
  model weights, plus per-session configuration (voice prompt,
  text prompt, seed).
- **Worker**: A server process capable of hosting one or more
  sessions. Reports its current and maximum capacity to the router.
- **Router**: The entry point that accepts client connections and
  assigns them to workers with available capacity. Discovers
  workers via a static configuration list of addresses provided
  at startup.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single server process handles at least 2 concurrent
  voice sessions without one session blocking or degrading the other.
- **SC-002**: The session router correctly distributes connections
  across multiple workers, achieving even utilization (no worker
  exceeds 80% capacity while another is below 50%, when load
  allows).
- **SC-003**: Automated tests covering session concurrency, routing,
  and lifecycle run successfully on a machine without a GPU in under
  30 seconds.
- **SC-004**: When a session disconnects, its capacity slot is
  reclaimed and available for a new connection within 2 seconds.
- **SC-005**: The system rejects excess connections gracefully — zero
  crashes or existing-session disruptions during overload.

## Clarifications

### Session 2026-04-02

- Q: Should sessions share model weights or get full independent copies? → A: Shared model weights, isolated streaming state per session.
- Q: How should the router forward traffic to workers? → A: Transparent WebSocket proxy; router relays frames bidirectionally, client only sees router address.
- Q: How does the router discover available workers? → A: Static configuration; router reads a list of worker addresses at startup.
- Q: Behavior when all session slots are full? → A: Reject only; return "capacity full" error, no eviction or queuing.
- Q: What level of observability is needed? → A: Health endpoint + structured per-session logging + metrics export (Prometheus-style).

## Assumptions

- Sessions share the loaded Mimi and LM model weights in VRAM.
  Each session instantiates its own lightweight streaming context
  (encoder/decoder state, LM generation state, audio buffers) on
  top of the shared weights. This requires the model classes to
  support multiple independent streaming contexts without full
  model duplication.
- The mock inference backend does not need to produce acoustically
  meaningful output — it only needs to exercise the same session
  lifecycle and concurrency paths as the real backend.
- The session router and workers communicate over the local network
  (localhost or LAN); external/internet routing and TLS termination
  are handled by a separate reverse proxy.
- GPU memory constraints (~14 GB per model instance) limit the
  practical number of concurrent real-inference sessions per process;
  the `--max-sessions` flag allows the operator to tune this based
  on available hardware.
