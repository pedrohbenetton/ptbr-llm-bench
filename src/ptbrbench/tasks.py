from __future__ import annotations

import json
from pathlib import Path

from ptbrbench.core import Example, Task
from ptbrbench.scorers import label_scorer

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
