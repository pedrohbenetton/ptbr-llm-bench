from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(results: dict) -> str:
    """Média e ordenação usam a nota rígida; a nota tolerante (quando existe) vira coluna extra."""
    tasks = sorted({t for r in results.values() for t in r})
    lenient = [t for t in tasks if any("score_lenient" in r.get(t, {}) for r in results.values())]
    avg = {m: sum(r[t]["score"] for t in tasks if t in r) / len(tasks) for m, r in results.items()}
    heads = tasks + [f"{t} (tol.)" for t in lenient]
    rows = ["| Modelo | Média | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
    for m in sorted(results, key=lambda k: -avg[k]):
        cells = [_pct(results[m][t]["score"]) if t in results[m] else "-" for t in tasks]
        cells += [
            _pct(results[m][t]["score_lenient"]) if "score_lenient" in results[m].get(t, {}) else "-"
            for t in lenient
        ]
        rows.append(f"| {m} | {_pct(avg[m])} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
