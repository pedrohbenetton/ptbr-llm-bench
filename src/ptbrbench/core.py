from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Scorer = Callable[[str, Any], float]
Model = Callable[[str, str], str]  # (system, prompt) -> texto


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    gold: Any
    tag: str = "all"


@dataclass
class Task:
    name: str
    examples: list[Example]
    scorer: Scorer
    system: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError(f"tarefa {self.name!r} sem exemplos")
