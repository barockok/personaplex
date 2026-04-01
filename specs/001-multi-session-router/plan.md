# Implementation Plan: Multi-Session Router & Concurrent Inference

**Branch**: `001-multi-session-router` | **Date**: 2026-04-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-multi-session-router/spec.md`

## Summary

Remove the single-session `asyncio.Lock` bottleneck from PersonaPlex's
server, enabling multiple concurrent voice sessions per process with
shared model weights and isolated per-session streaming state. Add a
session router that distributes WebSocket connections across GPU worker
processes via transparent proxying, with static worker configuration
and Prometheus-compatible metrics. Provide a mock inference backend so
the entire session lifecycle can be tested locally without a GPU.

## Technical Context

**Language/Version**: Python 3.10+ (PyTorch 2.2-2.4)
**Primary Dependencies**: aiohttp 3.10.x, sphn 0.1.x, torch, sentencepiece, huggingface-hub, prometheus-client (new)
**Storage**: N/A (in-memory session state only)
**Testing**: pytest + pytest-asyncio (new dev dependencies)
**Target Platform**: Linux server (CUDA GPU for production; CPU for tests)
**Project Type**: Real-time WebSocket service + CLI
**Performance Goals**: ≥2 concurrent sessions per process; sub-100ms session establishment
**Constraints**: ~14 GB VRAM for model weights; ~200-500 KB per additional session streaming state
**Scale/Scope**: 2-50 concurrent sessions across 1-10 GPU workers

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model Integrity | ✅ Pass | Shared weights are read-only; per-session streaming state isolates all mutable context. Audio codec params unchanged. |
| II. Real-Time Performance | ✅ Pass | Removing asyncio.Lock eliminates blocking between sessions. Hot-path code unchanged; only session management added around it. |
| III. Reproducibility | ✅ Pass | Per-session seed isolation preserves determinism. Offline mode unaffected. |
| IV. Test Before Ship | ✅ Pass | Mock backend enables full test coverage without GPU. New CLI flags include smoke-testable examples. |
| V. Simplicity | ✅ Pass | One new dependency (prometheus-client). Router uses static config, not a service discovery system. Mock backend reuses existing interfaces. |

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-session-router/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
moshi/moshi/
├── server.py            # MODIFY: refactor ServerState → SessionManager + Session
├── session.py           # NEW: Session class (per-connection streaming context)
├── session_manager.py   # NEW: SessionManager (replaces asyncio.Lock with slot pool)
├── router.py            # NEW: Session router (transparent WebSocket proxy)
├── mock_backend.py      # NEW: Mock inference backend for GPU-free testing
├── metrics.py           # NEW: Prometheus metrics endpoint
├── models/
│   ├── lm.py            # NO CHANGE (streaming state already isolated)
│   ├── compression.py   # NO CHANGE (MimiModel streaming already isolated)
│   └── loaders.py       # NO CHANGE
├── modules/
│   └── streaming.py     # NO CHANGE (StreamingModule supports multi-context)
└── utils/
    └── ...              # NO CHANGE

tests/                   # NEW directory at repo root
├── conftest.py          # Shared fixtures (mock backend, test clients)
├── test_session.py      # Session lifecycle tests
├── test_session_manager.py  # Concurrent session tests
├── test_router.py       # Router distribution and health check tests
└── test_metrics.py      # Metrics endpoint tests
```

**Structure Decision**: Extend the existing `moshi/moshi/` package with
new modules. No new top-level packages. Tests in a new `tests/`
directory at repo root, using pytest with the mock backend.

## Complexity Tracking

No constitution violations to justify.
