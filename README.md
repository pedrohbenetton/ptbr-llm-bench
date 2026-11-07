# ptbr-llm-bench

Benchmark pequeno e escrito à mão para comparar LLMs em PT-BR informal (gírias, abreviações, erros de digitação, correções no meio da frase).

**Status:** núcleo (tarefas, scorers, runner, relatório), adapters (Anthropic e Hugging Face) e três datasets escritos à mão (`intencao` 48, `ambiguidade` 40, `extracao_pedido` 60) prontos e testados. Nenhuma rodada de modelos foi executada, então este repo **não reporta nenhum número ainda**. Veja [SPEC.md](SPEC.md).

```bash
uv sync
uv run pytest
```

## Limitações conhecidas dos datasets

- `intencao`: os itens com tag `ambigua` têm rótulo escolhido por julgamento. `int-048` ("ainda n chegou, ta chegando?") é `acompanhamento`, mas poderia ser lido como `reclamacao`.
- `ambiguidade`: rascunho revisado uma vez; pode ainda ser fácil para modelos fortes (efeito teto). `amb-035` admite outra leitura. Os regionalismos foram escritos de memória.
- Amostras pequenas (40 a 60 itens por tarefa): diferenças de poucos pontos não são conclusivas.
