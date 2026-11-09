# ptbr-llm-bench

[![CI](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/ptbr-llm-bench/actions/workflows/ci.yml)

Benchmark pequeno para comparar LLMs em PT-BR informal, com datasets escritos com ajuda de IA e revisados com apoio externo (veja Limitações) (gírias, abreviações, erros de digitação, correções no meio da frase). A tarefa `extracao_pedido` reaproveita o conjunto difícil do projeto de fine-tuning [pedido-extractor-ptbr](https://github.com/pedrohbenetton/pedido-extractor-ptbr).

**Status:** núcleo (tarefas, scorers, runner, relatório), adapters (Anthropic e Hugging Face) e três datasets escritos com ajuda de IA (`intencao` 48, `ambiguidade` 40, `extracao_pedido` 60) prontos e testados. Uma primeira rodada (dois modelos abertos e o Claude Haiku 5.5) foi executada; veja Resultados. Veja [SPEC.md](SPEC.md).

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

Quatro modelos: Qwen2.5-1.5B-Instruct em 4 bits, base, com o LoRA **v1** e com o LoRA **v2** (v2 publicado em [`pedido-extractor-ptbr-lora`](https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora); ambos treinados só em extração de pedido, o v2 com dados mais variados: correções, typos, quantidade por extenso), e Claude Haiku 5.5 pela API (zero-shot, esforço `low`). Decodificação determinística nos Qwen, **uma execução** (repetir deu respostas idênticas), T4. Nota em %, sobre 48 (`intencao`), 40 (`ambiguidade`) e 60 (`extracao_pedido`) exemplos. Respostas brutas e notas por exemplo em [`results/published/`](results/published/).

| Modelo | Média (rígida) | extracao_pedido | intencao | ambiguidade (rígida) | ambiguidade (tolerante) |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (API) | **95,6** | 89,4 | **100,0** | **97,5** | **97,5** |
| Qwen2.5-1.5B + LoRA v2 | 66,9 | **96,1** | 64,6 | 40,0 | 70,0 |
| Qwen2.5-1.5B + LoRA v1 | 62,3 | 89,4 | 50,0 | 47,5 | 65,0 |
| Qwen2.5-1.5B base | 50,2 | 40,6 | 62,5 | 47,5 | 77,5 |

Na `extracao_pedido` a nota é por campo do JSON e compara listas estritamente (ordem, acento). Com a métrica do projeto de fine-tuning (exact match normalizado, sobre as mesmas respostas): base 3,3 · v1 71,7 · **v2 90,0** · Haiku 75,0 (o mesmo Haiku deu 68,3 em outra execução: a API não permite fixar a aleatoriedade).

**Leitura honesta**

- **Extração:** o v2 é o melhor nas duas métricas (96,1 e 90,0 de exact match, contra 89,4 e 75,0 do Haiku). Isso **não quer dizer que o v2 é melhor que o Haiku em geral**: o Haiku foi avaliado sem exemplos e sem as convenções do gabarito (parte do erro é convenção, como "ao ponto" contra "no ponto" no `hard-011`, já marcado como ambíguo), e o v2 foi treinado em dados sintéticos de um gerador que compartilha essas convenções. O conjunto tem 60 itens e 6 a 8 por categoria.
- **Custo de especialização:** o v1 perdia cerca de 12 pontos na `intencao` e na `ambiguidade` tolerante contra o base. O v2 **recupera a `intencao`** (64,6 contra 62,5 do base, uma diferença de 1 item) e ainda perde 7,5 pontos na `ambiguidade` tolerante (3 itens). Portanto o custo não é uniforme, e eu não sei explicar por que o v2 preserva mais; é uma execução só, em amostras pequenas.
- **`intencao` e `ambiguidade` não separam modelos fortes:** o Haiku tira 100,0 e 97,5 (efeito teto). O único erro dele na `ambiguidade` (`amb-018`) tem gabarito discutível.
- **Formato:** na `ambiguidade` a nota rígida (v2: 40,0) mede seguir a instrução "só a letra"; vários modelos respondem `b) elogio sincero`. A nota tolerante, **criada depois de ver esse resultado**, mede a compreensão. Na `intencao`, o v2 devolveu 4 respostas fora das etiquetas (como "sem resposta adequada").
- A média mistura tarefas muito diferentes e não é uma "nota geral". O custo da rodada com API não foi medido.

**Histórico de correção:** a primeira versão desta tabela chamava de "LoRA" o adapter v1 por engano (o repositório do Hugging Face ainda tinha os pesos do v1). Foi detectado ao repontuar as respostas com a métrica do projeto de fine-tuning, os pesos do v2 foram reenviados e a rodada foi repetida.

## Limitações conhecidas dos datasets

- **Autoria:** os rascunhos de `intencao` e `ambiguidade` foram gerados com ajuda de IA e passaram por uma revisão externa também feita com IA; não houve revisão humana independente. O conjunto difícil de extração foi escrito pelo autor com ajuda de IA, sem revisão externa.

- `intencao`: os itens com tag `ambigua` têm rótulo escolhido por julgamento. `int-048` ("ainda n chegou, ta chegando?") é `acompanhamento`, mas poderia ser lido como `reclamacao`.
- `ambiguidade`: rascunho revisado uma vez; pode ainda ser fácil para modelos fortes (efeito teto). `amb-035` admite outra leitura. Os regionalismos foram escritos de memória.
- Amostras pequenas (40 a 60 itens por tarefa): diferenças de poucos pontos não são conclusivas.
