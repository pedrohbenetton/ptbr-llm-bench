import json

import pytest

from ptbrbench.cli import main
from ptbrbench.core import Example, Task
from ptbrbench.report import render_report
from ptbrbench.runner import run, score_outputs
from ptbrbench.scorers import choice_scorer, label_scorer
from ptbrbench.tasks import load_ambiguidade


def _task(lenient=True):
    ex = [Example("1", "q1", "a", "x"), Example("2", "q2", "b", "x")]
    return Task("t", ex, label_scorer, lenient_scorer=choice_scorer if lenient else None)


def test_run_calcula_nota_tolerante_so_quando_a_tarefa_define():
    m = lambda system, prompt: {"q1": "a) texto", "q2": "b"}[prompt]
    r = run({"m": m}, [_task()])["m"]["t"]
    assert r["score"] == 0.5 and r["score_lenient"] == 1.0
    assert r["per_example_lenient"] == {"1": 1.0, "2": 1.0}
    assert "score_lenient" not in run({"m": m}, [_task(lenient=False)])["m"]["t"]


def test_score_outputs_usa_apenas_ids_presentes():
    r = score_outputs(_task(), {"1": "a"})
    assert list(r["per_example"]) == ["1"] and r["score"] == 1.0
    with pytest.raises(ValueError):
        score_outputs(_task(), {"zzz": "a"})


def test_relatorio_mostra_coluna_tolerante():
    md = render_report({"m": {"t": {"score": 0.5, "score_lenient": 1.0}}})
    assert "t (tol.)" in md and "100.0%" in md
    assert "(tol.)" not in render_report({"m": {"t": {"score": 0.5}}})


def test_ambiguidade_tem_scorer_tolerante():
    assert load_ambiguidade().lenient_scorer is choice_scorer


def test_rescore_recalcula_a_partir_das_respostas_salvas(tmp_path):
    t = load_ambiguidade()
    outs = {e.id: f"{e.gold}) texto da alternativa" for e in t.examples}
    src = tmp_path / "results.json"
    src.write_text(json.dumps({"m": {"ambiguidade": {"score": 0.0, "outputs": outs}}}), "utf-8")
    out = tmp_path / "novo"
    main(["rescore", str(src), "--out", str(out)])
    r = json.loads((out / "results.json").read_text("utf-8"))["m"]["ambiguidade"]
    assert r["score"] == 0.0 and r["score_lenient"] == 1.0
    assert "ambiguidade (tol.)" in (out / "report.md").read_text("utf-8")


def test_rescore_recusa_resultado_sem_respostas_brutas(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps({"m": {"intencao": {"score": 0.5}}}), "utf-8")
    with pytest.raises(SystemExit):
        main(["rescore", str(src), "--out", str(tmp_path / "o")])
