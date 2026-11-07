# ptbr-llm-bench

Benchmark pequeno e escrito à mão para comparar LLMs em PT-BR informal (gírias, abreviações, erros de digitação, correções no meio da frase).

**Status:** núcleo (tarefas, scorers, runner, relatório), adapters (Anthropic e Hugging Face) e três datasets escritos à mão (`intencao` 48, `ambiguidade` 40, `extracao_pedido` 60) prontos e testados. Nenhuma rodada de modelos foi executada, então este repo **não reporta nenhum número ainda**. Veja [SPEC.md](SPEC.md).

```bash
uv sync
uv run pytest
```

## Rodar o benchmark

```bash
# modelo aberto + LoRA (precisa de GPU; use o extra hf)
uv run --extra hf python -m ptbrbench --model hf:Qwen/Qwen2.5-1.5B-Instruct --model hf:Qwen/Qwen2.5-1.5B-Instruct@pedrohbenetton/pedido-extractor-ptbr-lora

# API da Anthropic (gasta dinheiro real da chave em ANTHROPIC_API_KEY; teste antes com --limit 5)
uv run --extra api python -m ptbrbench --model anthropic:claude-haiku-5-5 --limit 5
```

Gera `results/results.json` (score por exemplo) e `results/report.md` (tabela por tarefa).

## Limitações conhecidas dos datasets

- `intencao`: os itens com tag `ambigua` têm rótulo escolhido por julgamento. `int-048` ("ainda n chegou, ta chegando?") é `acompanhamento`, mas poderia ser lido como `reclamacao`.
- `ambiguidade`: rascunho revisado uma vez; pode ainda ser fácil para modelos fortes (efeito teto). `amb-035` admite outra leitura. Os regionalismos foram escritos de memória.
- Amostras pequenas (40 a 60 itens por tarefa): diferenças de poucos pontos não são conclusivas.
