<!--
  Sync Impact Report
  ==================
  Version change: (none) → 1.0.0
  Modified principles: N/A (initial ratification)
  Added sections:
    - Core Principles (5): Model Integrity, Real-Time Performance,
      Reproducibility, Test Before Ship, Simplicity
    - Technical Standards
    - Development Workflow
    - Governance
  Removed sections: N/A
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ no update needed
      (Constitution Check section is generic; gates derived at plan time)
    - .specify/templates/spec-template.md — ✅ no update needed
      (requirements and scenarios structure is compatible)
    - .specify/templates/tasks-template.md — ✅ no update needed
      (phase structure accommodates all principle-driven task types)
    - .specify/templates/checklist-template.md — ✅ no update needed
  Follow-up TODOs: none
-->

# PersonaPlex Constitution

## Core Principles

### I. Model Integrity

All changes to model architecture, weight loading, audio codecs, or
inference pipelines MUST preserve correctness of model outputs.

- Model weights MUST NOT be modified in-place; any fine-tuning or
  adaptation MUST produce a new artifact with a versioned identifier.
- Audio codec parameters (sample rate, frame size, Opus settings)
  MUST match the values the model was trained with unless an
  explicit migration is documented and tested.
- Voice prompt embeddings and text prompts MUST be validated against
  the supported set before inference begins.

**Rationale**: PersonaPlex is a research model whose outputs are
evaluated against published benchmarks (FullDuplexBench). Silent
regressions in model behavior undermine reproducibility and trust.

### II. Real-Time Performance

The server mode MUST maintain low-latency, full-duplex audio
streaming under normal operating conditions.

- Hot-path code (audio encode/decode, token generation, streaming
  I/O) MUST NOT introduce blocking operations or unbounded
  allocations.
- Memory-intensive operations (model loading, weight transfer) MUST
  support the `--cpu-offload` path for GPUs with limited VRAM.
- WebSocket and audio stream handling MUST be non-blocking; timeouts
  MUST be explicit and configurable.

**Rationale**: PersonaPlex is a real-time conversational system.
Latency spikes or OOM crashes during inference directly degrade the
user experience and invalidate latency benchmarks.

### III. Reproducibility

Given identical inputs (audio, prompts, seed, model weights), the
system MUST produce deterministic outputs in offline mode.

- The `--seed` flag MUST fully control all sources of randomness in
  offline evaluation.
- Inference configuration (voice prompt, text prompt, model version)
  MUST be logged or serializable so that any run can be reproduced.
- Docker and pip environments MUST pin dependency versions to avoid
  silent behavior changes across installs.

**Rationale**: Research credibility depends on reproducible results.
The paper and benchmarks reference specific seeds and configurations.

### IV. Test Before Ship

Every user-facing change MUST be verified before merge.

- Bug fixes MUST include a regression test or a manual verification
  procedure documented in the PR.
- New CLI flags or server endpoints MUST include usage examples that
  can be run as smoke tests.
- Offline evaluation scripts MUST produce consistent outputs when
  run against the reference test assets in `assets/test/`.

**Rationale**: The project serves both researchers and developers.
Broken CLI commands or silent output regressions erode adoption.

### V. Simplicity

Prefer the smallest change that solves the problem. Do not add
abstractions, configuration, or indirection without a concrete need.

- New dependencies MUST be justified; prefer the Python standard
  library or existing project dependencies when feasible.
- Configuration MUST use CLI arguments or environment variables;
  avoid config file proliferation.
- Code comments MUST explain *why*, not *what*. Self-evident code
  needs no annotation.

**Rationale**: PersonaPlex is a focused research artifact. Excess
complexity slows onboarding, increases maintenance burden, and
obscures the core model logic.

## Technical Standards

- **Language**: Python 3.10+ with PyTorch.
- **Audio**: Opus codec via `libopus`; sample rate and frame
  parameters MUST match model training configuration.
- **Packaging**: `pip install moshi/.` from repository root.
  `pyproject.toml` is the single source of dependency truth.
- **Containerization**: `Dockerfile` and `docker-compose.yaml` MUST
  remain buildable and consistent with the pip install path.
- **Secrets**: HuggingFace tokens (`HF_TOKEN`) MUST NOT be committed
  to the repository. `.env` files MUST be gitignored.

## Development Workflow

- All code changes MUST be submitted via pull request with at least
  one reviewer.
- Commits MUST have descriptive messages; prefer imperative mood
  (e.g., "fix OOM during model init").
- Breaking changes to CLI interface or server API MUST be documented
  in the PR description and README.
- The `main` branch MUST always be in a deployable state: server
  starts, offline evaluation completes on reference inputs.

## Governance

This constitution is the authoritative source of project principles.
It supersedes informal conventions when conflicts arise.

- **Amendments** require a pull request with rationale, review by at
  least one maintainer, and an updated version number.
- **Versioning** follows semantic versioning:
  - MAJOR: principle removed or fundamentally redefined.
  - MINOR: new principle or section added.
  - PATCH: clarification or wording fix.
- **Compliance** is verified during code review. Reviewers SHOULD
  reference specific principles when requesting changes.

**Version**: 1.0.0 | **Ratified**: 2026-04-01 | **Last Amended**: 2026-04-01
