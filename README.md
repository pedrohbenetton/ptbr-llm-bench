# ptbr-llm-bench

[![CI](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml)

Benchmark pequeno e escrito à mão para comparar LLMs em PT-BR informal (gírias, abreviações, erros de digitação, correções no meio da frase). A tarefa `extracao_pedido` reaproveita o conjunto difícil do projeto de fine-tuning [pedido-extractor-ptbr](https://github.com/pedrohbenetton/pedido-extractor-ptbr).

**Status:** núcleo (tarefas, scorers, runner, relatório), adapters (Anthropic e Hugging Face) e três datasets escritos à mão (`intencao` 48, `ambiguidade` 40, `extracao_pedido` 60) prontos e testados. Uma primeira rodada com dois modelos abertos foi executada (veja Resultados); modelos de API ainda não. Veja [SPEC.md](SPEC.md).

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

Gera `results/results.json` (nota e resposta bruta por exemplo) e `results/report.md` (tabela por tarefa).

### Nota rígida e nota tolerante

Cada resposta é pontuada de forma **rígida** (a resposta inteira tem que ser exatamente a etiqueta ou a letra pedida: mede seguir a instrução). Na `ambiguidade` há também a nota **tolerante** (`ambiguidade (tol.)`), que lê só a letra inicial (`b) texto` conta como `b`) e mede se o modelo entendeu a frase. A média e a ordenação da tabela usam a nota rígida.

A nota tolerante foi acrescentada **depois** da primeira rodada, ao ver que respostas como `b) elogio sincero` pontuavam zero com a letra certa. Para repontuar uma rodada salva, sem rodar os modelos de novo:

```bash
uv run python -m ptbrbench rescore results/results.json --out results/rescored
```

## Resultados (primeira rodada)

Qwen2.5-1.5B-Instruct em 4 bits, base contra base + LoRA [`pedido-extractor-ptbr-lora`](https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora) (treinado só em extração de pedido). Decodificação determinística, **uma execução** (a repetição da rodada deu exatamente a mesma tabela), T4. Nota em %, sobre 48 (`intencao`), 40 (`ambiguidade`) e 60 (`extracao_pedido`) exemplos.

| Modelo | Média (rígida) | extracao_pedido | intencao | ambiguidade (rígida) | ambiguidade (tolerante) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B + LoRA | 62,3 | **89,4** | 50,0 | 47,5 | 65,0 |
| Qwen2.5-1.5B base | 50,2 | 40,6 | **62,5** | 47,5 | **77,5** |

**Leitura honesta**

- O LoRA ganha muito na tarefa para a qual foi treinado (`extracao_pedido`, 40,6 para 89,4; nota por campo, não é o exact match do projeto de fine-tuning) e **perde nas outras duas**: cerca de 12 pontos na `intencao` e na `ambiguidade` tolerante. Os dois sentidos coincidem com um custo de especialização, mas são 5 e 6 itens em amostras pequenas e uma só execução: é indício, não prova.
- Na `ambiguidade` a nota rígida (47,5 para os dois) escondia a diferença: vários `b) elogio sincero` pontuavam zero com a letra certa. A nota tolerante foi **criada depois de ver esse resultado**, e as duas são mostradas por isso. Com duas opções o acaso é 50%: o modelo base (77,5) está bem acima dele, o LoRA (65,0) só moderadamente.
- A hipótese de que o LoRA responderia JSON no lugar das etiquetas foi testada e **descartada** (0 de 88 respostas começam com `{`). A causa da queda na `intencao` não foi identificada; há respostas com duas etiquetas (`elogio, acompanhamento`) que contam como erro.
- A média mistura uma tarefa em que o LoRA é especialista com duas em que não é; não é uma "nota geral".
- Faltam modelos de API e modelos maiores: nenhuma conclusão sobre eles.

## Limitações conhecidas dos datasets

- `intencao`: os itens com tag `ambigua` têm rótulo escolhido por julgamento. `int-048` ("ainda n chegou, ta chegando?") é `acompanhamento`, mas poderia ser lido como `reclamacao`.
- `ambiguidade`: rascunho revisado uma vez; pode ainda ser fácil para modelos fortes (efeito teto). `amb-035` admite outra leitura. Os regionalismos foram escritos de memória.
- Amostras pequenas (40 a 60 itens por tarefa): diferenças de poucos pontos não são conclusivas.
