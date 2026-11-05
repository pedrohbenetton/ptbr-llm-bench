# ptbr-llm-bench

Benchmark pequeno e escrito à mão para comparar LLMs em PT-BR informal (gírias, abreviações, erros de digitação, correções no meio da frase).

**Status:** núcleo (tarefas, scorers, runner, relatório) pronto e testado. Datasets, adapters de modelos e resultados ainda não existem, então este repo **não reporta nenhum número ainda**. Veja [SPEC.md](SPEC.md).

```bash
uv sync
uv run pytest
```
