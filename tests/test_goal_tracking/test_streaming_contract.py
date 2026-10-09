"""Isolated execution of real streaming methods, with provider/audio boundaries faked.

These tests run without voice dependencies. They do not exercise the full runtime.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_state import extracted

ROOT = Path(__file__).parents[2]
STREAMING = ROOT / "src/tau2/user/user_simulator_streaming.py"


class Message:
    def __init__(self, role="assistant", content=None, **kwargs):
        self.role, self.content = role, content
        self.__dict__.update(kwargs)
        for key in [
            "cost",
            "usage",
            "raw_data",
            "tool_calls",
            "generation_time_seconds",
        ]:
            self.__dict__.setdefault(key, None)

    def is_tool_call(self):
        return bool(self.tool_calls)

    def has_text_content(self):
        return bool(self.content)


class State(SimpleNamespace):
    def get_linearized_messages(self, **kwargs):
        return self.heard

    def model_copy(self, update):
        return State(**(vars(self) | update))


def isolated_simulator(tracking, provider):
    tree = ast.parse(STREAMING.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VoiceStreamingUserSimulator"
    )
    names = {
        "_generate_full_duplex_voice_message",
        "_track_goal_state",
        "_generation_snapshot",
        "_perform_turn_taking_action",
        "_emit_waiting_chunk",
        "_generate_backchannel_message",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for method in methods:
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
    minimal = ast.Module(
        body=[
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "generate": provider,
        "SystemMessage": Message,
        "UserMessage": Message,
        "AssistantMessage": Message,
        "ValidUserInputMessage": Message,
        "LinearizationStrategy": SimpleNamespace(CONTAINMENT_AWARE="heard"),
        "logger": SimpleNamespace(
            info=lambda *a: None, warning=lambda *a: None, debug=lambda *a: None
        ),
        "track_goals": tracking.track_goals,
        "tracking_payload": tracking.tracking_payload,
    }
    exec(compile(ast.fix_missing_locations(minimal), str(STREAMING), "exec"), namespace)
    user = namespace["Subject"]()
    user.instructions = "Return the more expensive tablet."
    user.llm, user.llm_args, user.tools = "fake", {"temperature": 0}, None
    (
        user.integration_ticks,
        user.silence_annotation_threshold_ticks,
        user.tick_duration_seconds,
    ) = 1, None, 0.2
    user._flip_roles_for_llm = lambda messages: [
        Message("user", m.content) for m in messages
    ]
    user._apply_chunk_effects = lambda *a, **kw: Message(content=None, cost=0.0)
    user.is_stop = lambda message: message.content == "###STOP###"
    return user


def state():
    return State(
        heard=[Message(content="Your return is submitted.")],
        system_messages=[Message("system", "Original instructions")],
        goal_state=None,
        ticks=[],
        input_turn_taking_buffer=[],
        output_streaming_queue=[Message(content="UNHEARD")],
    )


def test_disabled_tracker_preserves_one_response_call(tracking):
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return Message(
            content="###STOP###",
            cost=0.1,
            usage={"prompt_tokens": 1, "completion_tokens": 2},
            generation_time_seconds=0.5,
        )

    user = isolated_simulator(tracking, provider)
    user.user_goal_tracking = False
    result, _ = user._generate_full_duplex_voice_message(None, state())
    assert len(calls) == 1
    assert result.cost == 0.1
    assert result.raw_data is None


def test_enabled_tracker_steers_privately_and_counts_overhead(tracking):
    import json

    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        if kwargs["call_name"] == "user_goal_extract":
            content = json.dumps(extracted())
        elif kwargs["call_name"] == "user_goal_update":
            content = json.dumps(
                {
                    "updates": [
                        {
                            "id": "return",
                            "status": "complete",
                            "evidence": "Your return is submitted.",
                        }
                    ]
                }
            )
        else:
            content = "###STOP###"
        return Message(
            content=content,
            cost=0.1,
            usage={"prompt_tokens": 1, "completion_tokens": 2},
            generation_time_seconds=0.5,
        )

    user = isolated_simulator(tracking, provider)
    user.user_goal_tracking = True
    live = state()
    result, updated = user._generate_full_duplex_voice_message(None, live)
    assert len(calls) == 3
    assert updated.goal_state.goals[0].status == "complete"
    assert result.content == "###STOP###"
    assert result.cost == pytest.approx(0.3)
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 6}
    assert result.raw_data["goal_tracking"]["calls"] == 2
    sent = "\n".join(m.content or "" for call in calls for m in call["messages"])
    assert "UNHEARD" not in sent
    assert "PRIVATE CUSTOMER GOAL CHECKLIST" in calls[-1]["messages"][-1].content
    assert calls[0]["num_retries"] == 0
    assert calls[0]["tools"] is None


def test_snapshot_goals_cannot_mutate_live_state(tracking):
    import json

    user = isolated_simulator(tracking, lambda **kw: None)
    live = state()
    live.goal_state = tracking.initial_state(json.dumps(extracted()))
    snapshot = user._generation_snapshot(live)
    snapshot.goal_state = tracking.apply_updates(
        snapshot.goal_state,
        json.dumps(
            {"updates": [{"id": "return", "status": "complete", "evidence": "Done"}]}
        ),
        ["Done"],
    )
    assert live.goal_state.goals[0].status == "incomplete"
    assert snapshot.goal_state.goals[0].status == "complete"


def test_config_and_async_copyback_are_wired():
    source = STREAMING.read_text()
    assert "state.goal_state = generated_state.goal_state" in source
    assert "user_goal_tracking: bool = False" in source
    config = (ROOT / "src/tau2/data_model/simulation.py").read_text()
    assert "user_goal_tracking: bool = Field(" in config
    builder = (ROOT / "src/tau2/runner/build.py").read_text()
    assert "user_goal_tracking=audio_native_config.user_goal_tracking" in builder
    cli = (ROOT / "src/tau2/cli.py").read_text()
    assert '"--user-goal-tracking"' in cli
    assert "user_goal_tracking=args.user_goal_tracking" in cli


def test_async_silent_response_keeps_tracking_metadata(tracking):
    import json
    from concurrent.futures import Future

    user = isolated_simulator(tracking, lambda **kw: None)
    user.user_goal_tracking = True
    user.realtime_generation = True
    live = state()
    live.input_turn_taking_buffer = [Message(content="Old"), Message(content="New")]
    live.time_since_last_talk = 0
    generated = user._generation_snapshot(live)
    generated.goal_state = tracking.initial_state(json.dumps(extracted()))
    generated.user_utterance_count = 1
    generated._llm_generation_seconds = 1.2
    response = Message(
        content=None,
        cost=0.2,
        usage={"prompt_tokens": 9},
        raw_data={"goal_tracking": {"error": "ValueError"}},
    )
    user._generation = Future()
    user._generation.set_result((response, generated))
    user._generation_input_count = 1
    user._generation_from_tool = False
    chunk, result_state = user._perform_turn_taking_action(
        live, SimpleNamespace(action="generate_message")
    )
    assert chunk.cost == 0.2
    assert chunk.usage == {"prompt_tokens": 9}
    assert chunk.raw_data["goal_tracking"]["error"] == "ValueError"
    assert result_state.goal_state == generated.goal_state
    assert [m.content for m in result_state.input_turn_taking_buffer] == ["New"]
