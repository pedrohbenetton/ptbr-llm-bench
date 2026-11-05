from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(results: dict) -> str:
    tasks = sorted({t for r in results.values() for t in r})
    avg = {m: sum(r[t]["score"] for t in tasks if t in r) / len(tasks) for m, r in results.items()}
    rows = ["| Modelo | Média | " + " | ".join(tasks) + " |", "|---|---|" + "---|" * len(tasks)]
    for m in sorted(results, key=lambda k: -avg[k]):
        cells = [_pct(results[m][t]["score"]) if t in results[m] else "-" for t in tasks]
        rows.append(f"| {m} | {_pct(avg[m])} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
