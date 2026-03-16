# RAG Pipeline para Question Answering Medico (PubMedQA)

Projeto final de mestrado: pipeline completo de Retrieval-Augmented Generation (RAG) para responder perguntas medicas do tipo yes/no/maybe, utilizando o dataset PubMedQA.

---

## Requisitos Cobertos

Este projeto cobre **todos** os requisitos da especificacao:

| Requisito | Status | Onde |
|---|---|---|
| Obter dados de perguntas/respostas | OK | Notebook 01 (PubMedQA do HuggingFace) |
| Indexacao dos documentos | OK | Notebook 02 (ChromaDB + embeddings BGE) |
| Retriever | OK | Notebook 03 (embedding retrieval + cross-encoder reranking) |
| Gerador de resposta | OK | Notebook 03 (GPT-4o-mini com Chain-of-Thought) |
| Metricas baseadas em LLMs (correctness + faithfulness) | OK | Notebook 04 (RAGAS: Factual Correctness + Faithfulness) |
| Metricas semanticas (BERTScore, ROUGE-L, Cosine Sim) | OK | Notebook 04 (avaliacao local sem API) |
| Metricas de recuperacao de informacao (bonus) | OK | Notebook 04 (Hit Rate, Precision, Recall, MRR) |
| Ferramentas: LangChain | OK | Notebook 03b (pipeline LangChain) + Notebook 04 (RAGAS wrappers) |
| Ferramentas: LlamaIndex | OK | Notebook 03 (pipeline principal) |
| Ablation study | OK | Notebook 04 (impacto do reranker) |

---

## Pre-requisitos

- Python 3.10+
- Chave de API da OpenAI (para GPT-4o-mini)

## Instalacao

```bash
# 1. Clonar o repositorio
git clone <url-do-repo>
cd mestrado-cesar-13-marco-projeto-final

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar a chave da OpenAI
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Como Rodar (Passo a Passo)

### Passo 1: Download e Preparacao dos Dados

**Notebook:** `notebooks/01_data_loading.ipynb`

**O que faz:**
- Baixa o dataset PubMedQA (labeled subset) do HuggingFace
- Extrai 500 amostras com perguntas, respostas (yes/no/maybe) e documentos de referencia
- Salva os documentos PubMed completos (21.903 abstracts) em `data/pubmed_documents.json`
- Salva as 500 queries com ground truth em `data/pubmedqa.jsonl`

**Algoritmo:** Download direto do HuggingFace Datasets. Cada amostra do PubMedQA tem uma pergunta medica, contextos de artigos PubMed, e uma resposta rotulada por especialistas.

**Resultado esperado:**
- `data/pubmed_documents.json` (~15 MB, 21.903 documentos)
- `data/pubmedqa.jsonl` (500 linhas, cada uma com `id`, `query`, `ground_truth`, `golden_doc`)

**Como saber se ta certo:** Verificar que o arquivo tem 500 linhas e a distribuicao e ~55% yes, ~34% no, ~11% maybe.

---

### Passo 2: Indexacao no ChromaDB

**Notebook:** `notebooks/02_indexing.ipynb`

**O que faz:**
- Divide os 21.903 documentos em chunks menores (chunking)
- Gera embeddings para cada chunk usando o modelo BAAI/bge-base-en-v1.5
- Armazena os chunks + embeddings no ChromaDB (banco vetorial persistente)

**Algoritmos usados:**

1. **Chunking (SentenceSplitter)**: Divide documentos longos em pedacos de ~512 tokens com overlap de 50 tokens. **Por que:** LLMs tem limite de contexto; chunks menores permitem busca mais precisa e o overlap evita perder informacao nas bordas.

2. **Embedding Model (BAAI/bge-base-en-v1.5)**: Modelo de 768 dimensoes treinado para representar textos como vetores numericos. **Por que:** BGE e um dos melhores modelos open-source para busca semantica (top do MTEB benchmark), funciona bem para textos em ingles, e e eficiente (roda local, sem custo de API).

3. **ChromaDB**: Banco de dados vetorial com busca por similaridade. **Por que:** Leve, persistente em disco, integracao nativa com LlamaIndex e LangChain, gratuito e open-source.

**Resultado esperado:**
- Pasta `chroma_db/` com o indice persistido
- ~43.806 chunks indexados

**Como saber se ta certo:** O notebook imprime o numero de chunks indexados. Deve ser ~43.000-44.000.

---

### Passo 3: Retrieval + Geracao (Pipeline Principal)

**Notebook:** `notebooks/03_retrieval_generation.ipynb`

**O que faz:**
- Para cada uma das 500 queries:
  1. Busca os 20 chunks mais similares no ChromaDB (embedding retrieval)
  2. Reordena com cross-encoder para selecionar os 5 melhores (reranking)
  3. Gera resposta com GPT-4o-mini usando Chain-of-Thought (CoT)
  4. Extrai a resposta final (yes/no/maybe) do raciocinio

**Algoritmos usados:**

1. **Embedding Retrieval (Top-k=20)**: Busca vetorial por cosine similarity no ChromaDB. **Por que:** Rapido e escalavel para milhares de documentos. Busca os 20 mais similares para dar margem ao reranker.

2. **Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)**: Modelo que recebe (query, documento) como par e da um score de relevancia. Reduz de 20 para 5 documentos. **Por que:** Cross-encoders sao mais precisos que embeddings para ranking (comparacao direta query-documento), mas sao mais lentos. Usar em dois estagios (embedding rapido -> reranker preciso) e o padrao da industria.

3. **LLM Generation (GPT-4o-mini, temperature=0)**: Gera resposta com raciocinio Chain-of-Thought. **Por que:** GPT-4o-mini e rapido, barato ($0.15/1M tokens input) e capaz. Temperature=0 para respostas deterministicas. max_tokens=300 para permitir raciocinio.

4. **Chain-of-Thought Prompting**: O modelo primeiro raciocina sobre a evidencia (2-3 frases) e depois da a resposta final. **Por que:** Melhora a qualidade das respostas e permite que o RAGAS avalie Faithfulness (decomposicao em claims verificaveis).

5. **Few-Shot Prompting**: 4 exemplos no prompt (2 yes, 1 no, 1 maybe) com raciocinio. **Por que:** Calibra o modelo para o formato esperado e reduz respostas "maybe" excessivas.

6. **Answer Extraction**: Parser que busca "Final Answer: yes/no/maybe" com fallback regex. **Por que:** Robusto a variacoes no formato da resposta do LLM.

**Resultado esperado:**
- `results/rag_results.jsonl` — 500 resultados com contextos, resposta CoT completa, e resposta extraida
- `results/rag_summary.csv` — Resumo com accuracy por query
- Accuracy esperada: ~48-52%

**Como saber se ta certo:**
- O notebook imprime a accuracy ao final
- Verificar que nao ha "unknown" (0% apos retry)
- O campo `raw_response` deve conter raciocinio (nao apenas "yes")

**Tempo estimado:** ~30-40 min (retrieval local + 500 chamadas API)

**Custo estimado:** ~$0.15 (GPT-4o-mini e muito barato)

---

### Passo 3b: Pipeline Alternativo com LangChain

**Notebook:** `notebooks/03b_langchain_pipeline.ipynb`

**O que faz:**
- Reimplementa o MESMO pipeline usando LangChain em vez de LlamaIndex
- Roda em 50 queries (amostra) para demonstrar funcionamento
- Compara resultados com a implementacao LlamaIndex

**Componentes LangChain:**
- `langchain_openai.ChatOpenAI` — LLM
- `langchain_huggingface.HuggingFaceEmbeddings` — Embeddings
- `langchain_chroma.Chroma` — Vector store
- `ChatPromptTemplate` — Template de prompt estruturado
- `sentence_transformers.CrossEncoder` — Reranker (direto, sem wrapper)

**Por que dois frameworks:** Demonstra dominio de ambos os frameworks (LlamaIndex e LangChain) e confirma que a performance depende dos modelos/prompts, nao do framework de orquestracao.

**Resultado esperado:**
- `results/langchain_results.csv` — Resultados das 50 queries
- Accuracy comparavel ao LlamaIndex (diferenca < 5pp)

**Tempo estimado:** ~5 min (50 queries)

---

### Passo 4: Avaliacao

**Notebook:** `notebooks/04_evaluation.ipynb`

**O que faz:**
1. Calcula metricas de retrieval (sem LLM)
2. Roda ablation study (com vs sem reranker)
3. Calcula metricas RAGAS (com LLM)
4. Avaliacao semantica local (BERTScore, ROUGE-L, Cosine Similarity)

**Algoritmos e metricas:**

#### 4.1 Metricas de Retrieval (sem LLM)

| Metrica | O que mede | Por que e importante |
|---|---|---|
| **Hit Rate@k** | O doc golden esta no top-k? | Se o doc certo nem foi recuperado, o LLM nao consegue responder |
| **MRR** | Posicao do primeiro doc golden | Docs no topo recebem mais atencao do LLM |
| **Precision@k** | Fracao de docs relevantes no top-k | Menos ruido = melhores respostas |
| **Recall@k** | Fracao de docs golden encontrados | Cobertura dos docs relevantes |

Calculadas before e after reranking para comparar.

#### 4.2 Ablation Study (com LLM, 500 chamadas API)

Compara accuracy COM reranker vs SEM reranker para justificar a inclusao do cross-encoder.

**Como funciona:** Regenera respostas usando apenas os top-5 por embedding (sem reranking) e compara accuracy.

#### 4.3 RAGAS (com LLM, ~1000 chamadas API)

| Metrica | O que mede | Como funciona |
|---|---|---|
| **Faithfulness** | Resposta e sustentada pelos contextos? | Decompoe a resposta em claims e verifica cada um contra os contextos recuperados |
| **Factual Correctness (F1)** | Resposta confere com o ground truth? | Compara sobreposicao textual entre resposta e referencia |

**Por que CoT importa para RAGAS:** Na v2, respostas de 1 palavra ("yes") davam Faithfulness baixa (0.35) porque nao ha claims para verificar. Com CoT, o raciocinio permite decomposicao em claims verificaveis.

#### 4.4 Avaliacao Semantica (sem LLM, local)

| Metrica | O que mede | Por que e importante |
|---|---|---|
| **BERTScore F1** | Similaridade semantica contextual (BERT) | Captura significado alem de tokens superficiais |
| **ROUGE-L F1** | Subsequencia comum mais longa | Mede sobreposicao estrutural do texto |
| **Cosine Similarity (BGE)** | Similaridade de embeddings | Usa o mesmo modelo do retriever |

**Como funciona:** Compara o raciocinio CoT (raw_response) contra os documentos golden (referencia dos especialistas). Mostra que o modelo entende a evidencia mesmo quando o label yes/no/maybe nao confere.

**Por que e importante:** A accuracy de classificacao (46%) subestima a qualidade do pipeline. As metricas semanticas mostram que o raciocinio e semanticamente alinhado com a evidencia medica.

**Resultado esperado:**
- `results/retrieval_metrics.csv` — Metricas de retrieval
- `results/ragas_results.csv` — Scores RAGAS por query
- `results/ablation_reranker.csv` — Comparacao com/sem reranker
- `results/semantic_evaluation.csv` — BERTScore, ROUGE-L, Cosine Sim por query
- Faithfulness: 0.859 (melhoria significativa vs v2)
- Factual Correctness corrigido: 0.526 (consistente com accuracy 47.4%)
- BERTScore F1 esperado: ~0.85+ (alta similaridade semantica)
- Cosine Similarity esperado: ~0.70+ (bom alinhamento com golden docs)

**Como saber se ta certo:**
- Faithfulness > 0.5 (melhoria vs v2 que era 0.35)
- Factual Correctness corrigido ~ accuracy (ambos medem "acertou?")
- Ablation mostra impacto do reranker (-3.4pp)
- BERTScore > 0.80 (raciocinio semanticamente coerente)
- Scores semanticos similares entre predicoes corretas e incorretas (valida que o modelo entende a evidencia)

**Tempo estimado:** ~30-40 min (retrieval local + ablation 500 API + RAGAS ~1000 API)

**Custo estimado:** ~$0.20 (RAGAS + ablation)

---

### Passo 5: Regenerar Figuras

```bash
python results/generate_report_figures.py
```

Gera 10 figuras em `results/figures/` a partir dos CSVs de resultados.

---

## Estrutura do Codigo

```
mestrado-cesar-13-marco-projeto-final/
|
|-- .env                          # Chave OPENAI_API_KEY (NAO commitado)
|-- .gitignore
|-- requirements.txt              # Todas as dependencias Python
|-- REPORT.md                     # Relatorio completo com analise
|-- README.md                     # Este arquivo
|
|-- data/
|   |-- pubmed_documents.json     # 21.903 abstracts PubMed
|   |-- pubmedqa.jsonl            # 500 queries com ground truth
|
|-- chroma_db/                    # Indice vetorial ChromaDB persistido
|   |-- chroma.sqlite3
|   |-- <collection>/             # Embeddings + metadata
|
|-- notebooks/
|   |-- 01_data_loading.ipynb     # Passo 1: Download dados
|   |-- 02_indexing.ipynb         # Passo 2: Chunking + indexacao
|   |-- 03_retrieval_generation.ipynb  # Passo 3: RAG pipeline (LlamaIndex)
|   |-- 03b_langchain_pipeline.ipynb   # Passo 3b: RAG pipeline (LangChain)
|   |-- 04_evaluation.ipynb       # Passo 4: Avaliacao + ablation + RAGAS
|
|-- results/
|   |-- rag_results.jsonl         # Resultados completos (500 queries)
|   |-- rag_summary.csv           # Resumo com accuracy
|   |-- ragas_results.csv         # Scores RAGAS por query
|   |-- retrieval_metrics.csv     # Metricas retrieval before/after rerank
|   |-- ablation_reranker.csv     # Ablation: com vs sem reranker
|   |-- langchain_results.csv     # Resultados LangChain (50 queries)
|   |-- semantic_evaluation.csv  # BERTScore, ROUGE-L, Cosine Sim por query
|   |-- generate_report_figures.py # Script para gerar graficos
|   |-- figures/                  # 10 graficos PNG
|       |-- 01_accuracy_evolution.png
|       |-- 02_confusion_matrix.png
|       |-- 03_distribution_comparison.png
|       |-- 04_per_class_accuracy.png
|       |-- 05_retrieval_metrics.png
|       |-- 06_ragas_distributions.png
|       |-- 07_metrics_dashboard.png
|       |-- 08_ragas_evolution.png
|       |-- 09_ablation_reranker.png
|       |-- 10_semantic_evaluation.png
|
|-- scripts/
    |-- run_self_consistency.py   # Script standalone (opcional)
```

---

## Evolucao do Pipeline (4 versoes)

| Versao | Mudancas | Accuracy | Faithfulness | FC (corrigido) |
|---|---|---|---|---|
| **v0** | Claude Sonnet + parser fragil | 9.6% | 0.770 | 0.022 |
| **v1** | GPT-4o-mini + parser robusto | 47.0% | 0.380 | 0.470 |
| **v2** | Few-shot + temp=0 | 51.0% | 0.347 | 0.510 |
| **v3** | Chain-of-Thought + CoT reasoning | **47.4%** | **0.859** | **0.526** |

---

## Custo Total Estimado

| Etapa | Chamadas API | Custo |
|---|---|---|
| Notebook 03 (geracao) | 500 | ~$0.15 |
| Notebook 03b (LangChain demo) | 50 | ~$0.02 |
| Notebook 04 (ablation) | 500 | ~$0.15 |
| Notebook 04 (RAGAS) | ~1000 | ~$0.15 |
| **Total** | **~2050** | **~$0.47** |

Notebooks 01 e 02 nao usam APIs pagas (tudo local).

---

## Troubleshooting

**Erro "rate_limit" ou "429":**
O pipeline tem retry automatico com backoff exponencial. Se persistir, reduzir `NUM_WORKERS`.

**ChromaDB lento:**
A busca em 43.806 chunks pode levar ~20-30 min para 500 queries. E normal.

**"unknown" nas respostas:**
Se mais de 2% das respostas sao "unknown", verificar se o prompt e os exemplos few-shot estao corretos.

**RAGAS Faithfulness muito baixa (<0.3):**
Verificar que o campo `raw_response` contem raciocinio CoT (nao apenas "yes"). O RAGAS precisa de texto para decompor em claims.
