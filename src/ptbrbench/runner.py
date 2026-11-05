from __future__ import annotations

from collections import defaultdict

from ptbrbench.core import Model, Task


def run(models: dict[str, Model], tasks: list[Task]) -> dict:
    results: dict = {}
    for mname, model in models.items():
        results[mname] = {}
        for task in tasks:
            per_example: dict[str, float] = {}
            by_tag: dict[str, list[float]] = defaultdict(list)
            for ex in task.examples:
                s = task.scorer(model(task.system, ex.prompt), ex.gold)
                per_example[ex.id] = s
                by_tag[ex.tag].append(s)
            results[mname][task.name] = {
                "score": sum(per_example.values()) / len(per_example),
                "by_tag": {t: sum(v) / len(v) for t, v in sorted(by_tag.items())},
                "per_example": per_example,
            }
    return results
