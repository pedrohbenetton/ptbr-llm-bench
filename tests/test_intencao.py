from collections import Counter

from ptbrbench.tasks import LABELS, load_intencao

TAGS = {"giria", "abreviacao", "typo", "formal", "ambigua"}


def test_intencao_loads_valid_task():
    t = load_intencao()
    assert t.name == "intencao"
    assert all(e.gold in LABELS for e in t.examples)
    assert {e.tag for e in t.examples} <= TAGS


def test_ids_unique_and_prompts_distinct():
    t = load_intencao()
    ids = [e.id for e in t.examples]
    assert len(ids) == len(set(ids))
    prompts = [e.prompt for e in t.examples]
    assert len(prompts) == len(set(prompts))


def test_every_label_has_at_least_eight_examples():
    counts = Counter(e.gold for e in load_intencao().examples)
    assert all(counts[label] >= 8 for label in LABELS)


def test_system_prompt_lists_all_labels():
    t = load_intencao()
    assert all(label in t.system for label in LABELS)


def test_scorer_accepts_label_answer():
    t = load_intencao()
    assert t.scorer("reclamacao", "reclamacao") == 1.0
