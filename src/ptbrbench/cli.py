"""CLI: `python -m ptbrbench --model anthropic:claude-haiku-5-5 --model hf:Qwen/X@user/lora`.
Cada modelo roda em todas as tarefas escolhidas; gera results.json e report.md em --out.
Atenção: modelos `anthropic:` gastam dinheiro real da chave usada."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from ptbrbench.core import Model, Task
from ptbrbench.report import render_report
from ptbrbench.runner import run, score_outputs
from ptbrbench.tasks import load_ambiguidade, load_extracao_pedido, load_intencao

TASKS: dict[str, Callable[[], Task]] = {
    "intencao": load_intencao,
    "ambiguidade": load_ambiguidade,
    "extracao_pedido": load_extracao_pedido,
}
BACKENDS = ("anthropic", "hf")


def parse_model_spec(spec: str) -> tuple[str, str, str | None]:
    backend, sep, rest = spec.partition(":")
    if not sep or backend not in BACKENDS or not rest:
        raise ValueError(f"modelo inválido {spec!r}: use anthropic:<id> ou hf:<id>[@adapter]")
    model_id, _, adapter = rest.partition("@")
    if not model_id:
        raise ValueError(f"modelo inválido {spec!r}: id vazio")
    return backend, model_id, adapter or None


def select_tasks(names: list[str], limit: int | None) -> list[Task]:
    out = []
    for n in names:
        if n not in TASKS:
            raise ValueError(f"tarefa desconhecida {n!r}; opções: {', '.join(TASKS)}")
        t = TASKS[n]()
        out.append(dataclasses.replace(t, examples=t.examples[:limit]) if limit else t)
    return out


def build_model(spec: str, effort: str | None = "low") -> Model:
    backend, model_id, adapter = parse_model_spec(spec)
    if backend == "anthropic":
        import anthropic

        from ptbrbench.adapters import anthropic_model

        return anthropic_model(anthropic.Anthropic(), model_id, effort=effort)
    from ptbrbench.adapters import hf_model
    from ptbrbench.hf_backend import load_generate

    tok, generate = load_generate(model_id, adapter)
    return hf_model(tok, generate)


def _write(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_report(results)
    (out / "report.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def rescore_main(argv: list[str]) -> None:
    """`ptbrbench rescore results.json --out DIR`: repontua respostas salvas, sem rodar modelos."""
    ap = argparse.ArgumentParser(prog="ptbrbench rescore")
    ap.add_argument("results", help="results.json de uma rodada anterior (com `outputs`)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    old = json.loads(Path(a.results).read_text(encoding="utf-8"))
    new: dict = {}
    for model, tasks in old.items():
        new[model] = {}
        for name, r in tasks.items():
            if "outputs" not in r:
                ap.error(f"{model}/{name}: sem respostas brutas ('outputs'); rode de novo")
            task = select_tasks([name], None)[0]
            new[model][name] = score_outputs(task, r["outputs"])
    _write(new, Path(a.out))


def main(argv: list[str] | None = None, model_factory: Callable[[str], Model] | None = None) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "rescore":
        return rescore_main(argv[1:])
    ap = argparse.ArgumentParser(prog="ptbrbench")
    ap.add_argument("--model", action="append", default=[], help="anthropic:<id> | hf:<id>[@adapter]")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="usa só os N primeiros de cada tarefa")
    ap.add_argument("--effort", default="low", help="esforço (Anthropic); 'none' omite")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    if not a.model:
        ap.error("informe ao menos um --model")

    effort = None if a.effort == "none" else a.effort
    factory = model_factory or (lambda s: build_model(s, effort))
    tasks = select_tasks(a.tasks, a.limit)
    results = run({s: factory(s) for s in a.model}, tasks)

    _write(results, Path(a.out))
