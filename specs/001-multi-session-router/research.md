# Research: Multi-Session Router & Concurrent Inference

**Branch**: `001-multi-session-router` | **Date**: 2026-04-02

## R1: Streaming State Isolation

**Decision**: Sessions share model weights; each gets its own
streaming context (~200-500 KB per session vs ~14 GB model).

**Rationale**: The existing `StreamingModule` architecture in
`moshi/modules/streaming.py` cleanly separates model parameters
(read-only) from streaming state (per-instance `_streaming_state`
dataclass). `LMGen._init_streaming_state()` creates isolated
token caches, offset counters, and CUDAGraphed wrappers.
`reset_streaming()` clears per-session state without touching
weights.

**Alternatives considered**:
- Full model copy per session: ~14 GB × N sessions, infeasible for
  multi-session on a single GPU.
- Copy-on-write: Unnecessary complexity; streaming state is already
  small and independent.

**Key findings**:
- `_LMGenState` contains: cache tensor (~8 KB), provided tensor
  (~1 KB), initial tensor (<1 KB), 3× CUDAGraphed objects
  (~100-300 KB each), offset counter.
- `_MimiState` contains: 2× CUDAGraphed objects (~100-500 KB).
- Total per-session overhead: ~400 KB–1 MB.
- `streaming_forever(batch_size)` and `reset_streaming()` are the
  key lifecycle methods.

## R2: Concurrency Model

**Decision**: Replace the global `asyncio.Lock` with a
`SessionManager` that maintains a pool of session slots, each with
its own streaming context. Sessions run concurrently in the asyncio
event loop.

**Rationale**: The current lock serializes all WebSocket sessions.
Since PyTorch GPU operations release the GIL and the asyncio event
loop handles I/O concurrently, multiple sessions can interleave
their encode/decode/generate steps without true thread contention.
The GPU will naturally serialize CUDA kernel launches, but the
I/O (WebSocket read/write, Opus encode/decode) can overlap.

**Alternatives considered**:
- Multi-process with separate model copies: Higher memory usage,
  complex IPC. The router layer handles this at a higher level
  for scaling beyond one GPU.
- Thread pool: PyTorch + asyncio don't benefit from threads for
  GPU-bound work; adds complexity without throughput gain.

## R3: Session Context Creation

**Decision**: Create per-session Mimi and LMGen instances that
reference the same underlying model weights but hold independent
streaming state.

**Rationale**: The current server creates one `MimiModel` and one
`LMGen` instance, then uses `reset_streaming()` between sessions.
For concurrent sessions, each needs its own streaming state. The
approach is:
1. Load model weights once (shared `MimiModel.parameters()`,
   shared `LMModel.parameters()`).
2. For each session, create new `LMGen` and call
   `streaming_forever(1)` to initialize fresh streaming state.
3. Mimi encoder/decoder: need per-session instances that share
   the same weight tensors. This requires creating wrapper
   instances or using `streaming_forever` per session.

**Open consideration**: The Mimi model has internal streaming
state in its SEANet encoder/decoder and transformers. Creating
per-session Mimi instances that share weights may require
a lightweight wrapper that references the parent's parameters
but holds its own `_streaming_state`. This is the main
implementation challenge.

## R4: Router Architecture

**Decision**: Transparent WebSocket proxy using aiohttp, with
static worker configuration and least-connections routing.

**Rationale**: The router accepts client WebSocket connections,
selects a worker with available capacity, opens a backend
WebSocket to that worker, and relays frames bidirectionally.
Static config (list of `host:port` pairs) is sufficient for
the target scale of 2-10 workers. Least-connections routing
achieves even distribution without tracking session duration.

**Alternatives considered**:
- HTTP redirect: Requires client-side logic changes and exposes
  worker addresses.
- gRPC/custom protocol: Over-engineered for relaying binary
  WebSocket frames.
- Dynamic registration: Adds operational complexity; static
  config is simpler and debuggable.

## R5: Mock Inference Backend

**Decision**: Implement a `MockBackend` class that conforms to the
same interface as the real inference pipeline but returns fixed/echo
audio output without loading any model weights.

**Rationale**: The mock needs to:
1. Accept audio frames as input (same binary protocol).
2. Return audio frames as output (can echo input or return silence).
3. Return fixed text tokens (e.g., a repeated word).
4. Support the same session lifecycle: connect, handshake, stream,
   disconnect.
5. Run on CPU without torch.cuda.

This enables pytest-based integration tests that exercise the full
WebSocket session lifecycle, session manager concurrency, and router
distribution without any GPU.

**Alternatives considered**:
- Mocking at the WebSocket level: Too shallow; doesn't test session
  state management.
- Using CPU inference with a tiny model: Still requires model
  download and slow inference; not suitable for fast CI tests.

## R6: Metrics & Observability

**Decision**: Use `prometheus_client` library to expose a `/metrics`
endpoint on both workers and the router.

**Rationale**: `prometheus_client` is the standard Python library for
Prometheus-compatible metrics. It's lightweight (~100 KB), has no
transitive dependencies, and integrates easily with aiohttp. Metrics
include:
- `personaplex_active_sessions` (gauge, per worker)
- `personaplex_session_duration_seconds` (histogram)
- `personaplex_connection_rejections_total` (counter)
- `personaplex_worker_healthy` (gauge, per worker, on router)

Structured logging uses the existing `utils/logging.py` module with
added session_id and worker_id fields.

**Alternatives considered**:
- OpenTelemetry: Heavier dependency, more config. Prometheus is
  sufficient for the current scale.
- Custom metrics format: Non-standard, harder to integrate with
  existing monitoring.
