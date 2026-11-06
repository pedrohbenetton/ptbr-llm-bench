from ptbrbench.scorers import json_fields_scorer
from ptbrbench.tasks import load_extracao_pedido


def test_extracao_pedido_carrega_60_casos_com_gold_dict():
    t = load_extracao_pedido()
    assert t.name == "extracao_pedido" and len(t.examples) == 60
    assert len({e.id for e in t.examples}) == 60
    assert all(isinstance(e.gold, dict) and "itens" in e.gold for e in t.examples)
    assert "JSON" in t.system


def test_gold_pontua_1_contra_si_mesmo():
    import json

    t = load_extracao_pedido()
    for e in t.examples:
        assert t.scorer(json.dumps(e.gold, ensure_ascii=False), e.gold) == 1.0
    assert json_fields_scorer("sem json", t.examples[0].gold) == 0.0
