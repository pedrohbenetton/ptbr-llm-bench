import json

import pytest

from ptbrbench.cli import main, parse_model_spec, select_tasks


def test_parse_model_spec():
    assert parse_model_spec("anthropic:claude-x") == ("anthropic", "claude-x", None)
    assert parse_model_spec("hf:Qwen/Qwen2.5-1.5B-Instruct") == (
        "hf", "Qwen/Qwen2.5-1.5B-Instruct", None,
    )
    assert parse_model_spec("hf:Qwen/Q@user/lora") == ("hf", "Qwen/Q", "user/lora")
    for bad in ("claude-x", "openai:gpt", "hf:", ":x"):
        with pytest.raises(ValueError):
            parse_model_spec(bad)


def test_select_tasks_limit_and_unknown():
    ts = select_tasks(["intencao", "ambiguidade"], limit=5)
    assert [t.name for t in ts] == ["intencao", "ambiguidade"]
    assert all(len(t.examples) == 5 for t in ts)
    assert len(select_tasks(["extracao_pedido"], None)[0].examples) == 60
    with pytest.raises(ValueError):
        select_tasks(["nao_existe"], None)


def test_main_escreve_resultados_e_relatorio(tmp_path, capsys):
    modelos = []

    def fabrica(spec):
        modelos.append(spec)
        return lambda system, prompt: "a"

    main(
        ["--model", "anthropic:fake", "--tasks", "ambiguidade", "--limit", "4", "--out", str(tmp_path)],
        model_factory=fabrica,
    )
    assert modelos == ["anthropic:fake"]
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert set(res["anthropic:fake"]["ambiguidade"]["per_example"]) == {
        "amb-001", "amb-002", "amb-003", "amb-004",
    }
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "anthropic:fake" in md and "ambiguidade" in md
    assert "anthropic:fake" in capsys.readouterr().out


def test_main_exige_ao_menos_um_modelo(tmp_path):
    with pytest.raises(SystemExit):
        main(["--tasks", "intencao", "--out", str(tmp_path)], model_factory=lambda s: None)
