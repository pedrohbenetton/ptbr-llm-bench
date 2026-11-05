# SPEC: ptbr-llm-bench

Benchmark pequeno, reprodutível e escrito à mão para comparar LLMs em tarefas de **português do Brasil informal** (gírias, abreviações, erros de digitação, correções no meio da frase).

## Objetivo
Responder com números: "qual modelo (API ou aberto, base ou fine-tunado) lida melhor com PT-BR do dia a dia, a que custo e latência?"

## Requisitos
- R1. Uma tarefa é: nome, exemplos (`id`, `prompt`, `gold`, `tag`), prompt de sistema opcional e uma função de score em [0, 1].
- R2. Um modelo é qualquer função `(system, prompt) -> str`. O núcleo não depende de SDK nenhum.
- R3. `run` avalia todos os modelos em todas as tarefas e devolve score médio, score por tag e score por exemplo.
- R4. Scorers v1: `label` (classificação, tolerante a caixa/acento/pontuação) e `json_fields` (fração dos campos do gold presentes e iguais no JSON previsto; JSON inválido = 0).
- R5. O relatório em markdown ordena os modelos pela média das tarefas e mostra uma coluna por tarefa.
- R6. Datasets em JSONL escritos à mão, versionados em `data/`, nunca usados para treino.
- R7. Adapters (API Anthropic, Hugging Face + LoRA) ficam fora do núcleo e reaproveitam o código do `pedido-extractor-ptbr`.

## Tarefas planejadas (v1)
1. `extracao_pedido`: reaproveita o teste difícil do `pedido-extractor-ptbr` (scorer `json_fields`).
2. `intencao`: classificar a intenção de mensagens de cliente (pedido, reclamação, dúvida, cancelamento...) com gírias (scorer `label`).
3. `ambiguidade`: perguntas de resposta curta cuja interpretação muda com o contexto brasileiro (scorer `label`).

## Decisões em aberto
- Quais modelos entram na primeira rodada e o orçamento de API.
- Tamanho final de cada dataset (meta: 50 a 100 exemplos por tarefa).

## Fora de escopo
Leaderboard online, avaliação por LLM-juiz, tarefas longas de geração livre.
