# ptbr-llm-bench

[![CI](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml)

Benchmark pequeno e escrito à mão para comparar LLMs em PT-BR informal (gírias, abreviações, erros de digitação, correções no meio da frase). A tarefa `extracao_pedido` reaproveita o conjunto difícil do projeto de fine-tuning [pedido-extractor-ptbr](https://github.com/pedrohbenetton/pedido-extractor-ptbr).

**Status:** núcleo (tarefas, scorers, runner, relatório), adapters (Anthropic e Hugging Face) e três datasets escritos à mão (`intencao` 48, `ambiguidade` 40, `extracao_pedido` 60) prontos e testados. Uma primeira rodada (dois modelos abertos e o Claude Haiku 5.5) foi executada; veja Resultados. Veja [SPEC.md](SPEC.md).

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

Três modelos: Qwen2.5-1.5B-Instruct em 4 bits, base e com o LoRA [`pedido-extractor-ptbr-lora`](https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora) (treinado só em extração de pedido), e Claude Haiku 5.5 pela API (zero-shot, esforço `low`). Decodificação determinística nos Qwen, **uma execução** (repetir a rodada deu exatamente a mesma tabela), T4. Nota em %, sobre 48 (`intencao`), 40 (`ambiguidade`) e 60 (`extracao_pedido`) exemplos. Respostas brutas e notas por exemplo em [`results/published/`](results/published/).

| Modelo | Média (rígida) | extracao_pedido | intencao | ambiguidade (rígida) | ambiguidade (tolerante) |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (API) | **95,6** | 89,4 | **100,0** | **97,5** | **97,5** |
| Qwen2.5-1.5B + LoRA | 62,3 | **89,4** | 50,0 | 47,5 | 65,0 |
| Qwen2.5-1.5B base | 50,2 | 40,6 | 62,5 | 47,5 | 77,5 |

**Leitura honesta**

- **`intencao` e `ambiguidade` não separam modelos fortes:** o Haiku tira 100,0 e 97,5, efeito teto. Hoje elas só distinguem modelos pequenos. O único erro do Haiku na `ambiguidade` (`amb-018`, "Muito bom, repito com certeza.") tem gabarito discutível, sem marcador de ironia ou elogio.
- **`extracao_pedido` é a tarefa que discrimina:** o LoRA de 1,5B empata com o Haiku em nota (89,4) e em itens com nota máxima (43 de 60 cada). Por categoria os dois divergem (o Haiku vai melhor em `correcao`, 100 contra 70,8, e `quantidade_extenso`; o LoRA em `negacao` e `typo`), mas com poucos itens por categoria.
- **Isso não quer dizer que o LoRA é melhor que o Haiku.** O Haiku foi avaliado sem exemplos e sem as convenções do gabarito (parte do que conta como erro é convenção, como "ao ponto" contra "no ponto" no `hard-011`, já marcado como ambíguo). O LoRA foi treinado exatamente nesse formato.
- O LoRA ganha muito na tarefa treinada (40,6 para 89,4 contra o base) e **perde nas outras duas**: cerca de 12 pontos na `intencao` e na `ambiguidade` tolerante. Coincide com um custo de especialização, mas são 5 e 6 itens, uma só execução: indício, não prova. A causa da queda na `intencao` não foi identificada (a hipótese de o LoRA responder JSON foi testada e descartada: 0 de 88).
- Na `ambiguidade` a nota rígida escondia a diferença entre os Qwen (vários `b) elogio sincero` pontuavam zero com a letra certa). A nota tolerante foi **criada depois de ver esse resultado**; as duas aparecem por isso.
- A média mistura tarefas muito diferentes e não é uma "nota geral". O custo da rodada com API não foi medido.

## Limitações conhecidas dos datasets

- `intencao`: os itens com tag `ambigua` têm rótulo escolhido por julgamento. `int-048` ("ainda n chegou, ta chegando?") é `acompanhamento`, mas poderia ser lido como `reclamacao`.
- `ambiguidade`: rascunho revisado uma vez; pode ainda ser fácil para modelos fortes (efeito teto). `amb-035` admite outra leitura. Os regionalismos foram escritos de memória.
- Amostras pequenas (40 a 60 itens por tarefa): diferenças de poucos pontos não são conclusivas.
