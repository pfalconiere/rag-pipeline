"""Run corrected Factual Correctness using extracted answer only (not full CoT)."""

import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Load results
results = []
with open(RESULTS_DIR / "rag_results.jsonl", "r") as f:
    for line in f:
        results.append(json.loads(line))

print(f"Loaded {len(results)} results")


def extract_answer(text):
    text_clean = text.strip().lower()
    match = re.search(r'final answer:\s*(yes|no|maybe)\b', text_clean, re.I)
    if match:
        return match.group(1).lower()
    if text_clean in ("yes", "no", "maybe"):
        return text_clean
    match = re.search(r'\b(yes|no|maybe)\b', text_clean, re.I)
    return match.group(1).lower() if match else "unknown"


# Setup RAGAS
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

eval_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)
)
eval_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
)
run_config = RunConfig(max_workers=16, max_wait=180)

# Build samples with extracted answer only
samples_extracted = []
for r in results:
    raw_response = r.get("raw_response", r["generated_answer"])
    extracted = extract_answer(raw_response)
    sample = SingleTurnSample(
        user_input=r["query"],
        response=extracted,
        retrieved_contexts=r["contexts"],
        reference=r["ground_truth"].lower(),
    )
    samples_extracted.append(sample)

eval_dataset = EvaluationDataset(samples=samples_extracted)
print(f"Running RAGAS FC on {len(eval_dataset)} samples (extracted answers only)...")

ragas_result = evaluate(
    dataset=eval_dataset,
    metrics=[FactualCorrectness()],
    llm=eval_llm,
    embeddings=eval_embeddings,
    run_config=run_config,
)

print(f"\n=== Corrected FC Result ===")
print(ragas_result)

# Get per-sample results
df_corrected = ragas_result.to_pandas()
fc_col = [c for c in df_corrected.columns if 'factual_correctness' in c][0]

# Merge into existing RAGAS results
df_ragas = pd.read_csv(RESULTS_DIR / "ragas_results.csv")
df_ragas['fc_corrected'] = df_corrected[fc_col].values
df_ragas.to_csv(RESULTS_DIR / "ragas_results.csv", index=False)

# Find the original FC column
fc_original_col = [c for c in df_ragas.columns if 'factual_correctness' in c and c != 'fc_corrected'][0]

print(f"\n=== Comparison ===")
print(f"FC (full CoT):       {df_ragas[fc_original_col].mean():.4f}")
print(f"FC (extracted only): {df_ragas['fc_corrected'].mean():.4f}")
print(f"Accuracy:            {pd.read_csv(RESULTS_DIR / 'rag_summary.csv')['correct'].mean():.4f}")
print(f"\nSaved to ragas_results.csv (column 'fc_corrected')")
