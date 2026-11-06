from types import SimpleNamespace as NS

from ptbrbench.adapters import anthropic_model, hf_model


class FakeClient:
    def __init__(self, blocks, stop_reason="end_turn", raises=None):
        self.calls = []
        self.messages = NS(create=self._create)
        self._blocks, self._stop, self._raises = blocks, stop_reason, raises

    def _create(self, **kw):
        self.calls.append(kw)
        if self._raises:
            raise self._raises
        return NS(content=self._blocks, stop_reason=self._stop)


def _text(t):
    return NS(type="text", text=t)


def test_anthropic_sends_system_and_user_and_joins_text_blocks():
    c = FakeClient([_text("ola "), NS(type="thinking", text="x"), _text("mundo")])
    m = anthropic_model(c, "claude-x", effort="low", max_tokens=99)
    assert m("sys", "oi") == "ola mundo"
    kw = c.calls[0]
    assert kw["model"] == "claude-x" and kw["max_tokens"] == 99
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "oi"}]
    assert kw["output_config"] == {"effort": "low"}


def test_anthropic_omits_effort_and_empty_system():
    c = FakeClient([_text("a")])
    anthropic_model(c, "m", effort=None)("", "oi")
    assert "output_config" not in c.calls[0] and "system" not in c.calls[0]


def test_anthropic_api_error_becomes_empty_answer():
    class Boom(Exception):
        pass

    m = anthropic_model(FakeClient([], raises=Boom()), "m", error_types=(Boom,))
    assert m("s", "p") == ""


class FakeTok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(f"{x['role']}:{x['content']}" for x in msgs)


def test_hf_model_builds_chat_and_returns_generated_text():
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return ["resp"] * len(prompts)

    m = hf_model(FakeTok(), generate)
    assert m("sys", "oi") == "resp"
    assert seen == ["system:sys|user:oi"]
    assert hf_model(FakeTok(), generate)("", "oi") == "resp"
    assert seen[-1] == "user:oi"
