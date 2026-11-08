from ptbrbench.core import Example, Task
from ptbrbench.report import render_report
from ptbrbench.runner import run
from ptbrbench.scorers import label_scorer


def _task():
    ex = [
        Example("1", "q1", "a", "giria"),
        Example("2", "q2", "b", "giria"),
        Example("3", "q3", "a", "typo"),
    ]
    return Task("intencao", ex, label_scorer, system="responda")


def test_run_scores_per_model_task_and_tag():
    perfect = lambda system, prompt: {"q1": "a", "q2": "b", "q3": "a"}[prompt]
    half = lambda system, prompt: "a"
    res = run({"perfect": perfect, "half": half}, [_task()])
    assert res["perfect"]["intencao"]["score"] == 1.0
    assert abs(res["half"]["intencao"]["score"] - 2 / 3) < 1e-9
    assert res["half"]["intencao"]["by_tag"] == {"giria": 0.5, "typo": 1.0}
    assert res["half"]["intencao"]["per_example"] == {"1": 1.0, "2": 0.0, "3": 1.0}


def test_run_passes_system_prompt_to_model():
    seen = []
    run({"m": lambda system, prompt: seen.append(system) or "a"}, [_task()])
    assert set(seen) == {"responda"}


def test_task_requires_examples():
    import pytest

    with pytest.raises(ValueError):
        Task("vazia", [], label_scorer)


def test_report_sorted_by_average_desc():
    res = {
        "fraco": {"t1": {"score": 0.2}, "t2": {"score": 0.4}},
        "forte": {"t1": {"score": 0.9}, "t2": {"score": 0.7}},
    }
    md = render_report(res)
    assert md.index("forte") < md.index("fraco")
    assert "| t1 | t2 |" in md or "t1" in md and "t2" in md
    assert "80.0%" in md  # média do forte


def test_run_guarda_resposta_bruta_por_exemplo():
    res = run({"m": lambda system, prompt: f"resp-{prompt}"}, [_task()])
    assert res["m"]["intencao"]["outputs"] == {"1": "resp-q1", "2": "resp-q2", "3": "resp-q3"}
