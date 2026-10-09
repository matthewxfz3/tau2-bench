"""Offline tests of goal tracking; load the module without optional voice imports."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def extracted():
    return {
        "goals": [
            {
                "id": "return",
                "category": "objective",
                "text": "Return the more expensive tablet.",
            },
            {
                "id": "refund",
                "category": "requirement",
                "text": "Accept a gift-card refund if credit card is unavailable.",
            },
            {"id": "name", "category": "profile", "text": "Your name is Chen Silva."},
            {
                "id": "card",
                "category": "preference",
                "text": "Prefer a credit-card refund.",
            },
        ]
    }


def utterance(role, content, **kwargs):
    return SimpleNamespace(role=role, content=content, tool_calls=None, **kwargs)


def test_extract_initial_statuses_and_freeze_goals(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    assert [g.status for g in state.goals] == [
        "incomplete",
        "incomplete",
        "aligned",
        "misaligned",
    ]
    with pytest.raises(ValidationError):
        state.goals[0].text = "Return the cheaper tablet."


def test_progress_updates_preserve_requirements_and_previous_state(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    payload = {
        "updates": [
            {
                "id": "return",
                "status": "complete",
                "evidence": "Your return is submitted.",
            }
        ]
    }
    updated = tracking.apply_updates(
        state, json.dumps(payload), ["Your return is submitted."]
    )
    assert updated.goals[0].status == "complete"
    assert state.goals[0].status == "incomplete"
    assert updated.goals[0].text == state.goals[0].text


@pytest.mark.parametrize(
    "update",
    [
        {"id": "missing", "status": "complete", "evidence": "Done."},
        {"id": "return", "status": "aligned", "evidence": "Done."},
        {"id": "return", "status": "complete", "evidence": "Unheard future speech"},
        {"id": "return", "status": "complete", "evidence": ""},
        {
            "id": "return",
            "status": "complete",
            "evidence": "Done.",
            "text": "Change the requirement",
        },
    ],
)
def test_reject_invalid_updates_atomically(tracking, update):
    state = tracking.initial_state(json.dumps(extracted()))
    payload = {
        "updates": [
            {"id": "refund", "status": "attempted", "evidence": "Done."},
            update,
        ]
    }
    with pytest.raises(ValueError):
        tracking.apply_updates(state, json.dumps(payload), ["Done."])
    assert all(g.status == "incomplete" for g in state.goals[:2])


def test_reject_duplicate_ids(tracking):
    payload = extracted()
    payload["goals"].append(payload["goals"][0])
    with pytest.raises(ValueError):
        tracking.initial_state(json.dumps(payload))
    state = tracking.initial_state(json.dumps(extracted()))
    update = {"id": "return", "status": "complete", "evidence": "Done."}
    with pytest.raises(ValueError):
        tracking.apply_updates(
            state, json.dumps({"updates": [update, update]}), ["Done."]
        )


def test_evidence_cannot_span_utterances(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    with pytest.raises(ValueError):
        tracking.apply_updates(
            state,
            json.dumps(
                {
                    "updates": [
                        {
                            "id": "return",
                            "status": "complete",
                            "evidence": "Your return is submitted.",
                        }
                    ]
                }
            ),
            ["Your return", "is submitted."],
        )


def test_heard_history_excludes_tools_system_and_empty_text(tracking):
    messages = [
        utterance("assistant", "I can refund it to [incomplete]"),
        utterance("user", "Which card?"),
        utterance("tool", "SECRET DATABASE"),
        utterance("system", "PRIVATE"),
        utterance("assistant", None),
    ]
    heard = tracking.heard_dialogue(messages)
    assert heard == [
        {"role": "agent", "text": "I can refund it to [incomplete]"},
        {"role": "customer", "text": "Which card?"},
    ]


class FakeModel:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, purpose, instructions, payload):
        self.calls.append((purpose, instructions, payload))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(
            content=output,
            cost=0.02,
            usage={"prompt_tokens": 8, "completion_tokens": 4},
        )


def test_initial_extraction_then_update_and_private_reminder(tracking):
    fake = FakeModel(json.dumps(extracted()), json.dumps({"updates": []}))
    result = tracking.track_goals(
        "SCENARIO", None, [utterance("assistant", "Hello")], fake
    )
    assert len(fake.calls) == 2
    assert fake.calls[0][2] == {"scenario": "SCENARIO"}
    assert fake.calls[1][2]["conversation"] == [{"role": "agent", "text": "Hello"}]
    assert result.error is None
    assert result.cost == pytest.approx(0.04)
    assert result.usage == {"prompt_tokens": 16, "completion_tokens": 8}
    assert "Return the more expensive tablet." in result.reminder()
    assert "Do not read" in result.reminder()


def test_no_update_call_without_heard_speech(tracking):
    fake = FakeModel(json.dumps(extracted()))
    result = tracking.track_goals("SCENARIO", None, [], fake)
    assert len(fake.calls) == 1
    assert result.state.goals[0].status == "incomplete"


def test_failed_update_keeps_state_and_accounts_for_call(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    fake = FakeModel("{bad json")
    result = tracking.track_goals(
        "SCENARIO", state, [utterance("assistant", "Hello")], fake
    )
    assert result.state == state
    assert result.error is not None
    assert result.cost == pytest.approx(0.02)
    assert "stale" in result.reminder().lower()


def test_provider_failure_does_not_expose_exception_payload(tracking):
    fake = FakeModel(RuntimeError("SECRET API TOKEN"))
    result = tracking.track_goals("SCENARIO", None, [], fake)
    assert result.state is None
    assert result.error == "RuntimeError"
    assert "SECRET" not in result.model_dump_json()
    assert result.calls == 1


def test_attempted_does_not_rewrite_or_complete_requirement(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    updated = tracking.apply_updates(
        state,
        json.dumps(
            {
                "updates": [
                    {
                        "id": "refund",
                        "status": "attempted",
                        "evidence": "No refund methods are available.",
                    }
                ]
            }
        ),
        ["No refund methods are available."],
    )
    assert updated.goals[1].status == "attempted"
    assert updated.goals[1].text == state.goals[1].text


def test_extraction_retries_after_failure_then_recovers(tracking):
    fake = FakeModel("bad JSON", json.dumps(extracted()))
    failed = tracking.track_goals("SCENARIO", None, [], fake)
    assert failed.state is None and failed.error
    recovered = tracking.track_goals("SCENARIO", failed.state, [], fake)
    assert recovered.state is not None and recovered.error is None
    assert recovered.calls == 1


def test_update_can_recover_and_reopen_completed_goal(tracking):
    state = tracking.initial_state(json.dumps(extracted()))
    state = tracking.apply_updates(
        state,
        json.dumps(
            {"updates": [{"id": "return", "status": "complete", "evidence": "Done."}]}
        ),
        ["Done."],
    )
    fake = FakeModel(
        RuntimeError("unavailable"),
        json.dumps(
            {
                "updates": [
                    {
                        "id": "return",
                        "status": "incomplete",
                        "evidence": "Correction: it failed.",
                    }
                ]
            }
        ),
    )
    messages = [utterance("assistant", "Correction: it failed.")]
    failed = tracking.track_goals("SCENARIO", state, messages, fake)
    assert not failed.cost_complete and not failed.usage_complete
    recovered = tracking.track_goals("SCENARIO", failed.state, messages, fake)
    assert recovered.error is None
    assert recovered.state.goals[0].status == "incomplete"


def test_new_conversations_do_not_share_goal_state(tracking):
    first = tracking.track_goals("A", None, [], FakeModel(json.dumps(extracted())))
    other = {
        "goals": [{"id": "flight", "category": "objective", "text": "Book a flight."}]
    }
    second = tracking.track_goals("B", None, [], FakeModel(json.dumps(other)))
    assert first.state.goals[0].id == "return"
    assert second.state.goals[0].id == "flight"
    assert isinstance(first.state.goals, tuple)
