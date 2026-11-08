from __future__ import annotations

import json
from pathlib import Path

from ptbrbench.core import Example, Task
from ptbrbench.scorers import choice_scorer, json_fields_scorer, label_scorer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

LABELS = (
    "novo_pedido",
    "reclamacao",
    "duvida",
    "cancelamento",
    "elogio",
    "acompanhamento",
)

INTENCAO_SYSTEM = (
    "Você classifica mensagens de clientes de um app de delivery brasileiro. "
    "Responda apenas com uma destas etiquetas, sem explicação: " + ", ".join(LABELS) + "."
)


def load_intencao(path: Path | None = None) -> Task:
    path = path or DATA_DIR / "intencao.jsonl"
    exs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            exs.append(Example(d["id"], d["texto"], d["label"], d["tag"]))
    return Task("intencao", exs, label_scorer, system=INTENCAO_SYSTEM)


EXTRACAO_SYSTEM = (
    "Você extrai pedidos de delivery em português do Brasil. Responda somente com um JSON "
    'no formato {"itens": [{"nome": str, "quantidade": int, "observacoes": [str]}], '
    '"restricoes": [str], "pagamento": "pix"|"dinheiro"|"credito"|"debito"|"vale_refeicao"|null}.'
)


def load_extracao_pedido(path: Path | None = None) -> Task:
    path = path or DATA_DIR / "extracao_pedido.jsonl"
    exs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            exs.append(Example(d["id"], d["texto"], d["gold"], d["tag"]))
    return Task("extracao_pedido", exs, json_fields_scorer, system=EXTRACAO_SYSTEM)


AMBIGUIDADE_SYSTEM = (
    "Você entende português do Brasil informal, com gírias, regionalismos e ironia. "
    "Leia a mensagem e a pergunta e responda apenas com a letra da alternativa correta (a ou b), "
    "sem explicação."
)


def load_ambiguidade(path: Path | None = None) -> Task:
    path = path or DATA_DIR / "ambiguidade.jsonl"
    exs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            exs.append(Example(d["id"], d["prompt"], d["gold"], d["tag"]))
    return Task(
        "ambiguidade", exs, label_scorer, system=AMBIGUIDADE_SYSTEM, lenient_scorer=choice_scorer
    )
