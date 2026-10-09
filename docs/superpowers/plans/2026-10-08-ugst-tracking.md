# UGST Tracking Implementation Plan

**Goal:** Add optional, auditable UGST-inspired steering to the voice customer simulator.
**Architecture:** Immutable goal state plus an LLM tracker in one module; a small streaming integration; explicit config and CLI switch.
**Tech stack:** Python, Pydantic, pytest, existing LiteLLM adapter.

- [x] Write offline tests for extraction defaults, immutable requirements, evidence validation, atomic updates, malformed/provider failures, privacy, cost metadata, and isolation. Run before implementation and confirm missing feature.
- [x] Implement src/tau2/user/goal_tracking.py with models, prompts, validation, heard-text filtering, and call accounting. Run tests to green.
- [x] Write failing integration tests for baseline call count, private steering, async snapshot isolation/copy-back, and config defaults. Implement streaming/config/runner/CLI wiring.
- [x] Run targeted tests, lint, format and make check-all. Record environment limits; no network model calls.
- [x] Add docs/ugst.md with design, enablement, paper differences, failure modes, and validation record. Review diff and tests independently; fix findings.
- [ ] Commit only implementation/tests/docs. Publish under the user's account, provide a base-versus-feature comparison, and append verified details to the report.

Use test-driven development: observe failing tests, implement only what is needed, rerun. Test model calls with deterministic fakes; integration tests must use real streaming methods when dependencies permit. Do not present fake-model results as proof of better customer behavior.
