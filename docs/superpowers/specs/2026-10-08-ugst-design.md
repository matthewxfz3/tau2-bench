# UGST-inspired voice user tracking

Approved scope: optional inference-time goal tracking in the existing full-duplex simulator; no training, evaluator changes, hard-coded task IDs, or paid runs.

Base: 4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699. The earlier prompt-only PR is excluded.

Extract fixed goal components once from the scenario. Categories: objective, requirement, preference, profile, policy. Objective/requirement statuses: incomplete, complete, attempted; other statuses: aligned, misaligned. Preferences initially misaligned; profile/policy aligned; objectives/requirements incomplete, following the paper. Update all components in one call before a full customer response, using only the linearized heard text. Each update must reference an existing goal, retain its fixed text/category, and provide an exact nonempty evidence excerpt from an individual heard utterance. Validate all updates before applying any. Quoted evidence checks provenance, not semantic correctness. Preserve original requirements even after misalignment.

Inject a private reminder with current state; never synthesize the reminder as speech. The original scenario remains authoritative. Do not force successful completion or prohibit legitimate endings. Invalid output keeps the last valid state and records a stale/error flag. Failures of initial extraction leave the original response path available and are flagged. Retry extraction next response, at most once per response. No nested retries.

Store immutable Pydantic records in per-conversation state. Background generation snapshots can safely share immutable records; copy back the resulting record when a generation completes. Updates inspect only the same heard-history snapshot as generation, never future queued speech, private agent tools, or gold actions. Preserve incomplete-utterance markers. The record is a belief based on heard speech, not proof of database changes.

Use the same configured simulator model for extraction/updates. Tracker calls have distinct log names. Attach per-response tracking metadata (state, call usage/cost/duration, errors) to the generated message; include tracker cost/usage and duration in response accounting. Disabled by default via AudioNativeConfig.user_goal_tracking and constructor argument. Expose a CLI option so paired tests can toggle it.

This adapts §§4–5.1 of https://arxiv.org/abs/2507.20152v1. Batched updates and voice response-boundary updates differ from the paper; no SFT or GRPO. Behavior and latency benefits are unproven.
