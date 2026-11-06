"""Adapters que transformam um backend real em `Model = (system, prompt) -> str`.
Nenhum SDK é importado aqui no topo: o cliente/tokenizer é injetado por quem chama."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ptbrbench.core import Model


def anthropic_model(
    client,
    model: str,
    *,
    effort: str | None = "low",
    max_tokens: int = 1024,
    error_types: tuple[type[Exception], ...] | None = None,
) -> Model:
    """Erros da API viram resposta vazia (pontuam 0). `effort=None` omite `output_config`."""

    def api_errors() -> tuple[type[Exception], ...]:
        if error_types is not None:
            return error_types
        import anthropic

        return (anthropic.APIError,)

    def call(system: str, prompt: str) -> str:
        kw: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kw["system"] = system
        if effort:
            kw["output_config"] = {"effort": effort}
        try:
            resp = client.messages.create(**kw)
        except Exception as e:
            if isinstance(e, api_errors()):
                return ""
            raise
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


def hf_model(tok, generate: Callable[[Sequence[str]], list[str]]) -> Model:
    """`generate` recebe prompts já formatados pelo chat template e devolve só o texto novo."""

    def call(system: str, prompt: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return generate([text])[0]

    return call
