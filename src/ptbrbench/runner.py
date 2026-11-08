from __future__ import annotations

from collections import defaultdict

from ptbrbench.core import Model, Task


def score_outputs(task: Task, outputs: dict[str, str]) -> dict:
    """Pontua respostas já geradas (só os ids presentes em `outputs`)."""
    exs = [ex for ex in task.examples if ex.id in outputs]
    if not exs:
        raise ValueError(f"nenhuma resposta salva corresponde à tarefa {task.name!r}")
    per_example: dict[str, float] = {}
    by_tag: dict[str, list[float]] = defaultdict(list)
    for ex in exs:
        s = task.scorer(outputs[ex.id], ex.gold)
        per_example[ex.id] = s
        by_tag[ex.tag].append(s)
    res = {
        "score": sum(per_example.values()) / len(per_example),
        "by_tag": {t: sum(v) / len(v) for t, v in sorted(by_tag.items())},
        "per_example": per_example,
        "outputs": {ex.id: outputs[ex.id] for ex in exs},
    }
    if task.lenient_scorer:
        lenient = {ex.id: task.lenient_scorer(outputs[ex.id], ex.gold) for ex in exs}
        res["score_lenient"] = sum(lenient.values()) / len(lenient)
        res["per_example_lenient"] = lenient
    return res


def run(models: dict[str, Model], tasks: list[Task]) -> dict:
    results: dict = {}
    for mname, model in models.items():
        results[mname] = {}
        for task in tasks:
            outputs = {ex.id: model(task.system, ex.prompt) for ex in task.examples}
            results[mname][task.name] = score_outputs(task, outputs)
    return results
