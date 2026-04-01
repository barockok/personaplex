# Tasks: Multi-Session Router & Concurrent Inference

**Input**: Design documents from `/specs/001-multi-session-router/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Test tasks are included — the spec explicitly requires GPU-free testability (US3, SC-003).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Source code**: `moshi/moshi/` (existing package)
- **Tests**: `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: Project initialization, test infrastructure, and shared dependencies

- [x] T001 Add `prometheus-client`, `pytest`, and `pytest-asyncio` as dev/optional dependencies in `moshi/pyproject.toml`
- [x] T002 Create `tests/` directory with `tests/conftest.py` containing shared pytest fixtures (aiohttp test client helper, mock WebSocket client factory, free port finder)
- [x] T003 [P] Create `moshi/moshi/metrics.py` with Prometheus metric definitions: `personaplex_active_sessions` (Gauge), `personaplex_session_duration_seconds` (Histogram), `personaplex_connection_rejections_total` (Counter), `personaplex_worker_healthy` (Gauge with worker label), `personaplex_routed_connections_total` (Counter with worker label)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Mock backend and Session class that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create `moshi/moshi/mock_backend.py` implementing a `MockBackend` class that conforms to the same interface as the real inference pipeline: accepts audio frames, returns silence/echo audio, returns fixed text tokens, supports `streaming_forever()`, `reset_streaming()`, and `step()` methods — runs on CPU with no model weight loading
- [x] T005 Create `moshi/moshi/session.py` with `Session` dataclass/class containing: `session_id` (UUID), `worker_id`, `voice_prompt`, `text_prompt`, `seed`, `status` enum (`connecting`/`active`/`disconnecting`/`closed`), `created_at`, `client_ip`, and references to per-session Mimi encoder/decoder streaming state and LMGen streaming state. Include `handle_chat(ws)` method extracted from current `ServerState.handle_chat`, operating on the session's own streaming context instead of the shared one
- [x] T006 Create `moshi/moshi/session_manager.py` with `SessionManager` class: holds shared model weights (MimiModel, LMModel, text_tokenizer), manages a pool of session slots up to `max_sessions`, provides `acquire_session(request) -> Session` (creates per-session streaming context on shared weights) and `release_session(session_id)` (cleans up streaming state, frees slot). Replace the `asyncio.Lock` pattern with slot-based concurrency control

**Checkpoint**: Foundation ready — mock backend, session, and session manager are the building blocks for all stories

---

## Phase 3: User Story 1 — Multiple Users Connect Simultaneously (Priority: P1) 🎯 MVP

**Goal**: Remove the single-session lock and enable concurrent voice sessions in one server process

**Independent Test**: Start mock server with `--max-sessions 3`, connect 3 WebSocket clients concurrently, verify all receive handshake and stream independently

### Tests for User Story 1

- [x] T007 [P] [US1] Write `tests/test_session.py` — test session lifecycle: create session via mock backend, verify status transitions (`connecting` → `active` → `disconnecting` → `closed`), verify resource cleanup on disconnect, verify session state isolation (two sessions with different voice_prompt/seed do not interfere)
- [x] T008 [P] [US1] Write `tests/test_session_manager.py` — test concurrent sessions: create SessionManager with mock backend and `max_sessions=3`, acquire 3 sessions concurrently via `asyncio.gather`, verify all 3 are active simultaneously, verify 4th acquisition is rejected with capacity error, verify releasing a session frees the slot for reuse

### Implementation for User Story 1

- [x] T009 [US1] Refactor `moshi/moshi/server.py`: replace `ServerState` with `SessionManager` integration. Remove `asyncio.Lock`. Update `handle_chat` to call `session_manager.acquire_session()` on connect, delegate to `session.handle_chat(ws)`, and call `session_manager.release_session()` on disconnect/error. Add `--max-sessions` and `--worker-id` CLI flags. Add `--mock` flag that uses `MockBackend` instead of loading real models
- [x] T010 [US1] Add `/health` endpoint to `moshi/moshi/server.py` returning JSON with `worker_id`, `active_sessions`, `max_sessions`, `healthy` status per worker-api contract
- [x] T011 [US1] Add `/metrics` endpoint to `moshi/moshi/server.py` serving Prometheus metrics from `moshi/moshi/metrics.py`: increment/decrement `active_sessions` gauge on session acquire/release, observe `session_duration_seconds` on session close, increment `connection_rejections_total` on capacity rejection
- [x] T012 [US1] Add structured logging to `moshi/moshi/session.py` and `moshi/moshi/session_manager.py`: log session connect/disconnect/error events with `session_id`, `worker_id`, duration, and error reason using the existing `utils/logging.py` module

**Checkpoint**: At this point, a single server process can handle multiple concurrent sessions. Verify with `tests/test_session.py` and `tests/test_session_manager.py`

---

## Phase 4: User Story 3 — Local Testing Without GPU (Priority: P1)

**Goal**: Ensure the entire multi-session system is testable on a CPU-only machine via mock backend

**Independent Test**: Run `pytest tests/ -v` on a machine without a GPU — all tests pass in under 30 seconds

### Tests for User Story 3

- [x] T013 [P] [US3] Write `tests/test_mock_backend.py` — test MockBackend: verify it implements the same interface as real backend (streaming_forever, reset_streaming, step, encode, decode), verify it runs on CPU without torch.cuda, verify it returns audio frames and text tokens, verify multiple MockBackend streaming contexts can run concurrently

### Implementation for User Story 3

- [x] T014 [US3] Add end-to-end integration test in `tests/test_e2e_mock.py`: start a full server with `--mock --max-sessions 3` using `aiohttp.test_utils`, connect N WebSocket clients, send audio frames, verify each receives handshake + audio responses, verify concurrent operation, verify disconnect cleanup — all without GPU. This validates the quickstart.md scenario programmatically
- [x] T015 [US3] Verify all tests pass on CPU-only: run full `pytest tests/` suite, confirm no test imports or exercises CUDA, confirm total runtime < 30 seconds

**Checkpoint**: All US1 + US3 tests pass on a CPU-only machine. The mock backend enables rapid development iteration for US2

---

## Phase 5: User Story 2 — Session Router Distributes Connections (Priority: P2)

**Goal**: Route incoming WebSocket connections across multiple worker processes via transparent proxy

**Independent Test**: Start router + 2 mock workers, connect clients through router, verify distribution and failover

### Tests for User Story 2

- [x] T016 [P] [US2] Write `tests/test_router.py` — test router logic: create router with 2 mock workers (using `aiohttp.test_utils` servers with mock backend), connect clients through router, verify least-connections distribution (2 clients → 2 different workers), verify capacity-full rejection when all workers full, verify unhealthy worker removal after failed health check, verify transparent WebSocket frame relay (client receives handshake + audio through router)

### Implementation for User Story 2

- [x] T017 [US2] Create `moshi/moshi/router.py` with `Router` class: parse `--workers ID:HOST:PORT` args into worker pool, implement `handle_chat` that selects least-loaded healthy worker, opens backend WebSocket, relays frames bidirectionally between client and worker WebSocket, closes both on disconnect
- [x] T018 [US2] Add health check loop to `moshi/moshi/router.py`: periodically poll each worker's `/health` endpoint at `--health-check-interval` rate, mark workers as unhealthy on failure, restore on recovery, update `personaplex_worker_healthy` metric
- [x] T019 [US2] Add `/health` and `/metrics` endpoints to `moshi/moshi/router.py` per router-api contract: health returns total/healthy workers, capacity, active sessions; metrics returns per-worker health gauge and routed connections counter
- [x] T020 [US2] Create `moshi/moshi/__main__` router entry point so `python -m moshi.router` works: argparse with `--host`, `--port`, `--workers` (repeatable), `--health-check-interval`, `--ssl`. Wire up aiohttp app with router handlers
- [x] T021 [P] [US2] Write `tests/test_metrics.py` — test Prometheus metrics: verify `/metrics` endpoint on both worker and router returns valid Prometheus text format, verify metric values update correctly on session acquire/release/reject

**Checkpoint**: Full system testable: router distributes across workers, all metrics/health endpoints work, all tested without GPU

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and validation across all stories

- [x] T022 [P] Update `README.md` with multi-session server usage: document `--max-sessions`, `--worker-id`, `--mock` flags and router usage (`python -m moshi.router`)
- [x] T023 [P] Update `Dockerfile` and `docker-compose.yaml` to support multi-worker deployment: add `prometheus-client` to container, expose `/health` and `/metrics` ports, add example docker-compose with router + 2 workers
- [x] T024 Run `quickstart.md` validation: execute each quickstart scenario end-to-end on mock mode, confirm all commands work as documented
- [x] T025 Run full `pytest tests/` suite and confirm all pass, runtime < 30 seconds, zero GPU dependency

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — core multi-session capability
- **US3 (Phase 4)**: Depends on US1 (Phase 3) — validates testability of US1
- **US2 (Phase 5)**: Depends on Foundational (Phase 2) — can start after Phase 2, but US1 should be stable first
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Phase 2. No dependency on other stories.
- **User Story 3 (P1)**: Depends on Phase 2 + US1 implementation (needs mock backend + session manager to test).
- **User Story 2 (P2)**: Depends on Phase 2. Can start after Phase 2 but benefits from US1 being stable. Independently testable against mock workers.

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Foundation modules (session, session_manager) before server refactor
- Server refactor before health/metrics endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- T002 and T003 can run in parallel (different files)
- T007 and T008 can run in parallel (different test files)
- T016 and T021 can run in parallel (different test files)
- T022 and T023 can run in parallel (different documentation files)

---

## Parallel Example: User Story 1

```bash
# Launch tests for US1 together:
Task: "Write tests/test_session.py"
Task: "Write tests/test_session_manager.py"

# Then implementation sequentially:
Task: "Refactor server.py with SessionManager"
Task: "Add /health endpoint"
Task: "Add /metrics endpoint"
Task: "Add structured logging"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (mock backend + session + session manager)
3. Complete Phase 3: User Story 1 (multi-session server)
4. **STOP and VALIDATE**: Test with mock backend, verify concurrent sessions
5. Deploy single multi-session worker if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Multi-session server works (MVP!)
3. Add User Story 3 → Validate GPU-free testing → CI-ready
4. Add User Story 2 → Test independently → Full router + workers
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (server refactor)
   - Developer B: User Story 2 (router) — can start T017-T020 in parallel
3. User Story 3 validates both after completion

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The mock backend (T004) is the linchpin — it enables all GPU-free testing
