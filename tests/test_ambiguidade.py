from collections import Counter

from ptbrbench.tasks import load_ambiguidade


def test_ambiguidade_estrutura():
    t = load_ambiguidade()
    assert t.name == "ambiguidade" and len(t.examples) == 40
    assert len({e.id for e in t.examples}) == 40
    assert {e.gold for e in t.examples} == {"a", "b"}
    assert Counter(e.tag for e in t.examples) == {
        "giria": 10, "ironia": 10, "regionalismo": 10, "contexto_delivery": 10,
    }
    assert "letra" in t.system


def test_ambiguidade_balanceado_e_prompts_completos():
    t = load_ambiguidade()
    n_a = sum(e.gold == "a" for e in t.examples)
    assert 16 <= n_a <= 24
    for e in t.examples:
        assert "\na) " in e.prompt and "\nb) " in e.prompt, e.id
    por_tag = Counter((e.tag, e.gold) for e in t.examples)
    assert all(por_tag[(tag, g)] >= 3 for tag in {e.tag for e in t.examples} for g in "ab")


def test_scorer_aceita_letra_com_pontuacao():
    t = load_ambiguidade()
    assert t.scorer("A)", "a") == 1.0 and t.scorer("b", "a") == 0.0
