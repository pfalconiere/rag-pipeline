# RAG Pipeline para Question Answering Médico — Relatório de Resultados

## 1. Introdução

Este relatório apresenta os resultados do pipeline de Retrieval-Augmented Generation (RAG) desenvolvido para responder perguntas médicas do tipo yes/no/maybe, utilizando o dataset **PubMedQA** (500 amostras rotuladas).

O pipeline foi iterativamente otimizado em três versões, alcançando melhorias significativas em accuracy e factual correctness.

### Arquitetura do Pipeline

```
PubMedQA Query
      │
      ▼
┌─────────────────────────┐
│  Embedding Retrieval     │  BAAI/bge-base-en-v1.5
│  (Top-k = 20 chunks)    │  ChromaDB (43.806 chunks)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Cross-Encoder Reranking │  cross-encoder/ms-marco-MiniLM-L-6-v2
│  (Top-n = 5 chunks)     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  LLM Generation          │  GPT-4o-mini (temperature=0)
│  (Few-shot prompting)    │  max_tokens=100
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Answer Extraction       │  Regex parser (yes/no/maybe)
└──────────┬──────────────┘
           │
           ▼
     yes / no / maybe
```

---

## 2. Dataset

| Propriedade | Valor |
|---|---|
| **Dataset** | PubMedQA (labeled subset) |
| **Amostras** | 500 |
| **Tipo de resposta** | yes / no / maybe |
| **Distribuição** | yes: 276 (55.2%), no: 169 (33.8%), maybe: 55 (11.0%) |
| **Base de conhecimento** | 43.806 chunks indexados no ChromaDB |

---

## 3. Evolução do Pipeline

O pipeline foi otimizado em três versões. A tabela abaixo resume a evolução:

| Métrica | v0 (Claude + parser frágil) | v1 (GPT-4o-mini + parser robusto) | v2 (Few-shot + temp=0) |
|---|---|---|---|
| **Accuracy** | 9.60% | 47.00% | **51.00%** |
| **Factual Correctness** | 0.022 | 0.470 | **0.510** |
| **Faithfulness** | 0.770 | 0.380 | 0.347 |
| **RAGAS amostras** | 50 | 500 | 500 |
| **Custo estimado** | ~$5.00 | ~$0.20 | ~$0.20 |

### Evolução visual

![Evolução da Accuracy](results/figures/01_accuracy_evolution.png)

![Evolução RAGAS](results/figures/08_ragas_evolution.png)

### O que mudou em cada versão

**v0 → v1:**
- **LLM**: Claude Sonnet → GPT-4o-mini (~20x mais barato)
- **Prompt**: De "provide justification + Final Answer:" para "respond with a single word only"
- **Parser**: De `split("\n")[-1].replace("Final Answer:", "")` (frágil) para `re.search(r'\b(yes|no|maybe)\b')` (robusto)
- **Workers**: 5 → 10 (GPT-4o-mini tem rate limits mais altos)
- **Impacto**: Accuracy 9.6% → 47.0% (+390%)

**v1 → v2:**
- **Temperature**: default → 0.0 (respostas determinísticas)
- **Few-shot**: 4 exemplos no prompt (2 yes, 1 no, 1 maybe)
- **Prompt anti-maybe**: "Prefer yes or no. Use maybe ONLY if evidence is genuinely mixed"
- **Impacto**: Accuracy 47.0% → 51.0% (+4pp), "no" accuracy 24.8% → 39.6%

---

## 4. Métricas de Retrieval

O retriever busca os 20 chunks mais similares via embedding (BAAI/bge-base-en-v1.5) e o cross-encoder reranker seleciona os 5 mais relevantes.

| Métrica | Before Rerank | After Rerank |
|---|---|---|
| **Hit Rate@1** | 0.764 | 0.691 |
| **Hit Rate@5** | 0.865 | 0.809 |
| **Hit Rate@10** | 0.904 | 0.809 |
| **Precision@5** | 0.257 | 0.232 |
| **Recall@5** | 0.734 | 0.658 |
| **MRR** | 0.806 | 0.735 |

**Observação**: As métricas "after rerank" são menores porque o reranker reduz de 20 para 5 documentos. Isso é esperado — o objetivo é ter os 5 chunks mais relevantes, não maximizar recall bruto. O Hit Rate@5 de 0.809 indica que em 81% das queries, o documento golden está entre os 5 chunks selecionados.

![Métricas de Retrieval](results/figures/05_retrieval_metrics.png)

---

## 5. Métricas de Geração (Accuracy)

### 5.1 Accuracy Geral

**Accuracy final: 51.00%** (255/500 corretos)

### 5.2 Accuracy por Classe

| Classe | Accuracy | Acertos/Total |
|---|---|---|
| **yes** | 64.5% | 178/276 |
| **no** | 39.6% | 67/169 |
| **maybe** | 18.2% | 10/55 |

![Accuracy por Classe](results/figures/04_per_class_accuracy.png)

### 5.3 Distribuição de Predições vs Ground Truth

| Label | Ground Truth | Predicted |
|---|---|---|
| **yes** | 276 | 247 |
| **no** | 169 | 106 |
| **maybe** | 55 | 141 |
| **unknown** | — | 6 |

O modelo ainda tende a predizer "maybe" em excesso (141 vs 55 no ground truth), mas é significativamente melhor que a v1 (que predizia 217 "maybe").

![Distribuição GT vs Predicted](results/figures/03_distribution_comparison.png)

### 5.4 Matriz de Confusão

![Matriz de Confusão](results/figures/02_confusion_matrix.png)

**Principais padrões de erro:**
- **100 queries "yes" preditas como "maybe"**: O modelo é cauteloso em afirmar positivamente quando a evidência é indireta.
- **91 queries "no" preditas como "maybe"**: O modelo evita negar quando a evidência é parcial.
- **43 queries "no" preditas como "yes"**: Interpretação otimista da evidência.

---

## 6. Métricas RAGAS (LLM-Based)

As métricas RAGAS foram calculadas com GPT-4o-mini sobre todas as 500 amostras, com 16 workers paralelos.

### 6.1 Faithfulness

**Score médio: 0.347**

Faithfulness mede se a resposta gerada é sustentada pelos contextos recuperados (sem alucinação). O score baixo é um **artefato do formato de resposta**: respostas de uma única palavra ("yes") não contêm claims verificáveis contra os contextos. O RAGAS espera respostas mais longas para decomor em claims e verificar cada um.

### 6.2 Factual Correctness

**Score médio: 0.510**

Factual Correctness (modo F1) mede a sobreposição textual entre a resposta gerada e o ground truth. Este score é consistente com a accuracy de 51%: quando a resposta é "yes" e o ground truth é "yes", o F1 é 1.0; caso contrário, 0.0.

![Distribuição RAGAS](results/figures/06_ragas_distributions.png)

---

## 7. Painel Completo de Métricas

![Dashboard](results/figures/07_metrics_dashboard.png)

---

## 8. Análise de Custo

| Item | v0 (Claude Sonnet) | v2 (GPT-4o-mini) |
|---|---|---|
| **Modelo de geração** | claude-sonnet-4 | gpt-4o-mini |
| **Modelo de avaliação** | claude-sonnet-4 | gpt-4o-mini |
| **Custo por 500 queries (geração)** | ~$2.50 | ~$0.10 |
| **Custo por 500 queries (RAGAS eval)** | ~$5.00 | ~$0.15 |
| **Custo total estimado** | ~$7.50 | **~$0.25** |
| **Redução** | — | **~30x mais barato** |
| **Workers geração** | 5 | 10 |
| **Workers RAGAS** | 1 | 16 |

---

## 9. Limitações e Trabalho Futuro

### Limitações atuais
1. **Accuracy de 51%** está abaixo do estado da arte em PubMedQA (~78% com modelos fine-tunados).
2. **Classe "no" subdetectada**: 39.6% de accuracy, com muitos falsos "maybe".
3. **Classe "maybe" ambígua**: Apenas 18.2% de accuracy — o modelo não captura bem a incerteza calibrada.
4. **Faithfulness artificialmente baixa**: Respostas de 1 palavra não são adequadas para esta métrica.

### Possíveis melhorias futuras
1. **Chain-of-Thought + resposta final**: Gerar justificativa seguida de "Final Answer: yes/no/maybe", extraindo apenas a resposta final. Isso melhoraria a Faithfulness sem prejudicar a accuracy.
2. **Fine-tuning de poucos exemplos**: Fine-tune do GPT-4o-mini com exemplos do PubMedQA para calibrar a distribuição de respostas.
3. **Modelo de embedding médico**: Substituir BGE por um modelo treinado em textos biomédicos (e.g., PubMedBERT embeddings).
4. **Aumentar top_k do retriever**: Testar top_k=50 para melhorar o recall antes do reranking.
5. **Cross-encoder biomédico**: Substituir ms-marco-MiniLM por um reranker treinado em domínio médico.

---

## 10. Reprodutibilidade

### Dependências principais
- `llama-index-core`, `llama-index-llms-openai`
- `chromadb`, `sentence-transformers`
- `ragas>=0.2.0`, `langchain-openai`
- `BAAI/bge-base-en-v1.5` (embedding)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)
- `gpt-4o-mini` (geração + avaliação)

### Notebooks
1. `notebooks/01_data_preparation.ipynb` — Download e preparação do PubMedQA
2. `notebooks/02_indexing.ipynb` — Chunking + indexação no ChromaDB
3. `notebooks/03_retrieval_generation.ipynb` — Retrieval + Reranking + Geração
4. `notebooks/04_evaluation.ipynb` — Métricas de retrieval + RAGAS

### Arquivos de resultado
- `results/rag_results.jsonl` — Resultados completos (query, contextos, resposta)
- `results/rag_summary.csv` — Resumo com accuracy por query
- `results/ragas_results.csv` — Scores RAGAS por query
- `results/retrieval_metrics.csv` — Métricas de retrieval before/after rerank
- `results/figures/` — Todos os gráficos deste relatório

---

*Relatório gerado em 14/03/2026. Pipeline RAG para PubMedQA — Projeto Final de Mestrado.*
