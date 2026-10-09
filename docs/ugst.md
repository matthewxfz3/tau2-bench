# UGST-inspired voice user tracking

This optional change gives the simulated customer a private record of its goals before each full response. It adapts inference-time steering from Mehri et al. (2025), *Goal Alignment in LLM-Based User Simulators for Conversational AI*, [arXiv:2507.20152v1, §§4–5.1](https://arxiv.org/abs/2507.20152v1).

## Enable it

```bash
uv sync --extra voice --extra dev
uv run tau2 run --domain retail --audio-native --user-goal-tracking --num-tasks 1
```

This command makes paid provider calls if run with configured credentials. It was **not run** for this implementation. Omit `--user-goal-tracking` for the unchanged baseline. Programmatic callers can set `AudioNativeConfig(user_goal_tracking=True)` or pass `user_goal_tracking=True` to `VoiceStreamingUserSimulator`.

## Algorithm

1. Extract fixed components from the task's customer scenario once: objectives, requirements, preferences, profile facts, and behavior rules.
2. Before each full response, ask the simulator's configured model to update statuses from the conversation heard so far. All statuses are updated in one call.
3. Validate the response. Reject unknown IDs, rewritten definitions, duplicate updates, invalid status/category combinations, and evidence not quoted from a single heard utterance. Apply valid updates atomically.
4. Supply the record privately to the response model alongside its original scenario and conversation history. The checklist is never passed directly to speech synthesis.

Objectives and requirements can be `incomplete`, `complete`, or `attempted` (sufficient attempts blocked by external factors). Profiles, policies, and preferences can be `aligned` or `misaligned`. Initial statuses follow the paper: objectives/requirements incomplete, profiles/policies aligned, preferences misaligned. The prompt distinguishes a proposal or promise from heard confirmation of completion and permits later corrections to reopen a request.

The initial extraction has one call, followed by one batched update when heard speech is available. Later full responses have one update call. Backchannels do not invoke tracking. Definitions never change after extraction. Updates are beliefs based on heard speech, not verified database outcomes. No task IDs, reference actions, evaluator outputs, or private agent tool results enter the tracker. The existing streaming linearizer defines the heard-history boundary; queued output is excluded.

## Failures and overhead

Malformed updates preserve the last valid state and mark the reminder as stale. Initial extraction failures leave the original scenario available and retry extraction on the next full response. There is no added validation/regeneration loop or forced stop restriction. Tracker calls disable provider retries and record only the exception class to avoid copying sensitive provider error text.

Per-response `raw_data.goal_tracking` records state, errors, calls, known cost, token usage, elapsed time, and cost/usage completeness flags. Known tracker overhead is added to the response totals once. A provider failure may have unknown charges; inspect the completeness flags rather than treating missing charges as zero. The metadata survives ordinary speech chunking, stop responses, tool calls, and silent responses. Canceled generations may incur provider charges without an emitted response; existing runtime call logs remain necessary for a full billing audit.

Frozen models and tuples isolate conversation state from background generation snapshots. Completed background generations copy back the new goal record without discarding newer input chunks.

## Differences from the paper and remaining risks

This implements an adaptation of inference-time steering, not the complete paper. It batches status updates instead of separately updating each component, and updates at voice-response boundaries rather than clean conversational turns. It does not implement supervised fine-tuning, GRPO, or the paper's evaluation pipeline.

Exact-quote checks establish where evidence came from, not whether it justifies the status. Extraction can omit a goal, updates can misinterpret speech, and the response model can ignore a correct record. Additional calls add delay and can alter interruption behavior. No improvement in simulator fidelity, naturalness, scores, or live latency has been demonstrated.

## Validation

Test-first sequence: 15 tracker tests failed against unimplemented functions, then passed. Streaming enablement and configuration tests failed before wiring, then passed. Independent review found that silent responses discarded overhead; a new regression test failed before the fix and passed afterward.

Offline command:

```bash
python -m pytest -q -c /dev/null -p no:cacheprovider \
  --confcutdir=tests/test_goal_tracking tests/test_goal_tracking
```

Result: **23 tests passed**. Repository-wide Ruff lint, changed-file formatting checks, and `git diff --check` passed. The full `make check-all` command was blocked by the same `uv` platform panic.

The offline suite needs pytest and Pydantic 2. It loads the tracker independently and executes selected real streaming methods with fake provider/audio boundaries. This is **not a full-package or audio-runtime integration test**. Local execution used Python 3.14 because that interpreter had these packages; the upstream supported range is Python 3.12–3.13. `uv sync` failed with a platform panic and dependency installation was unavailable. Runtime validation on a supported Python with voice dependencies remains necessary.

The main report specifies a paired baseline-versus-checklist experiment. That experiment and paid model calls were not run.

## Review scope

The base commit is `4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699`. The earlier prompt-only PR is excluded. Changes are limited to the tracker, streaming integration, default-off configuration, CLI/runner wiring, tests, and documentation. No domain tasks, agent policies, or scoring rules are changed.

AI disclosure: Codex assisted with design, implementation, tests, and independent code review.
