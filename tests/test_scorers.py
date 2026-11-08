from ptbrbench.scorers import extract_json, json_fields_scorer, label_scorer


def test_label_ignores_case_accent_and_punctuation():
    assert label_scorer("  Reclamação.\n", "reclamacao") == 1.0


def test_label_wrong():
    assert label_scorer("duvida", "reclamacao") == 0.0


def test_label_uses_first_line_only():
    assert label_scorer("duvida\nporque o cliente pergunta", "duvida") == 1.0


def test_extract_json_tolerates_surrounding_text():
    assert extract_json('claro! {"a": {"b": 1}} pronto') == {"a": {"b": 1}}


def test_extract_json_invalid_returns_none():
    assert extract_json("sem json aqui") is None
    assert extract_json('{"a": ') is None


def test_json_fields_full_match():
    assert json_fields_scorer('{"x": 1, "y": "a"}', {"x": 1, "y": "a"}) == 1.0


def test_json_fields_partial_and_extra_keys_ignored():
    assert json_fields_scorer('{"x": 1, "y": "b", "z": 9}', {"x": 1, "y": "a"}) == 0.5


def test_json_fields_invalid_json_is_zero():
    assert json_fields_scorer("nada", {"x": 1}) == 0.0


def test_json_fields_compares_strings_normalized():
    assert json_fields_scorer('{"n": "Açaí"}', {"n": "acai"}) == 1.0


def test_choice_scorer_le_a_letra_inicial():
    from ptbrbench.scorers import choice_scorer

    for pred in ("a", "A", "a)", "(a)", "a.", "a) elogio sincero", " (a) texto\noutra linha"):
        assert choice_scorer(pred, "a") == 1.0, pred
        assert choice_scorer(pred, "b") == 0.0, pred
    for pred in ("", "a resposta é b", "talvez", "ab", "elogio sincero"):
        assert choice_scorer(pred, "a") == 0.0, pred
        assert choice_scorer(pred, "b") == 0.0, pred
