"""Retry queries that returned 'unknown' due to rate limits."""

import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Load all results
results = []
with open(RESULTS_DIR / "rag_results.jsonl", "r") as f:
    for line in f:
        results.append(json.loads(line))

# Find unknowns
unknown_indices = [i for i, r in enumerate(results) if r["generated_answer"] == "unknown"]
print(f"Found {len(unknown_indices)} unknown entries to retry")

if not unknown_indices:
    print("Nothing to retry!")
    exit(0)

# Setup LLM
from openai import OpenAI
client = OpenAI()

SYSTEM_PROMPT = """You are a medical expert answering yes/no/maybe questions based on PubMed abstracts.
Rules:
- First, reason step-by-step about what the evidence says (2-3 sentences).
- Then give your final answer on a new line in the format: "Final Answer: yes" or "Final Answer: no" or "Final Answer: maybe"
- Answer "yes" if the evidence supports the claim.
- Answer "no" if the evidence contradicts or does not support the claim.
- Answer "maybe" ONLY if the evidence is genuinely mixed or insufficient to decide.
- Most questions have a definitive answer. Prefer "yes" or "no" over "maybe"."""

FEW_SHOT_EXAMPLES = """Example 1:
Question: Is increased gravitational stress beneficial for bone density?
Context: Studies show weight-bearing exercise increases bone mineral density by 2-8% in postmenopausal women...
Reasoning: The evidence indicates that weight-bearing exercise, which increases gravitational stress on bones, leads to measurable increases in bone mineral density (2-8%). This supports the claim that gravitational stress benefits bone density.
Final Answer: yes

Example 2:
Question: Does smoking cessation reduce cardiovascular risk?
Context: After 1 year of cessation, coronary heart disease risk drops by 50%. After 15 years, risk equals that of a non-smoker...
Reasoning: The evidence clearly shows that quitting smoking leads to substantial cardiovascular risk reduction - 50% within one year and full normalization after 15 years. This strongly supports the claim.
Final Answer: yes

Example 3:
Question: Is homeopathy effective for treating asthma?
Context: A systematic review of 6 RCTs found no significant difference between homeopathic treatments and placebo in lung function or symptom scores...
Reasoning: A systematic review of 6 randomized controlled trials found no significant difference between homeopathy and placebo. This is strong evidence against effectiveness, as systematic reviews of RCTs are high-quality evidence.
Final Answer: no

Example 4:
Question: Can MRI replace biopsy for diagnosing prostate cancer?
Context: MRI showed sensitivity of 91% but specificity of only 37%. While useful for risk stratification, results were inconsistent across centers...
Reasoning: While MRI has high sensitivity (91%), its low specificity (37%) means many false positives. Additionally, results are inconsistent across centers, making it unreliable as a biopsy replacement. It may complement but cannot replace biopsy.
Final Answer: maybe

"""


def build_prompt(query_text, contexts):
    context_parts = [f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    context = "\n\n".join(context_parts)
    return f"""{FEW_SHOT_EXAMPLES}Now answer this question:

Context:
{context}

Question: {query_text}

Reasoning:"""


def extract_answer(text):
    text_clean = text.strip().lower()
    match = re.search(r'final answer:\s*(yes|no|maybe)\b', text_clean, re.I)
    if match:
        return match.group(1).lower()
    if text_clean in ("yes", "no", "maybe"):
        return text_clean
    match = re.search(r'\b(yes|no|maybe)\b', text_clean, re.I)
    return match.group(1).lower() if match else "unknown"


# Retry each unknown query
retried = 0
still_unknown = 0
for idx in unknown_indices:
    r = results[idx]
    prompt = build_prompt(r["query"], r["contexts"])

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content
            answer = extract_answer(raw)

            results[idx]["raw_response"] = raw
            results[idx]["generated_answer"] = answer
            results[idx]["votes"] = [answer]

            status = "OK" if answer != "unknown" else "STILL UNKNOWN"
            print(f"  [{status}] Query {r['id']}: {answer} (GT: {r['ground_truth']})")

            if answer != "unknown":
                retried += 1
            else:
                still_unknown += 1
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {e}")
                still_unknown += 1
                break

    time.sleep(0.5)  # Small delay between queries

print(f"\nRetried: {retried}, Still unknown: {still_unknown}")

# Save updated results
with open(RESULTS_DIR / "rag_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Updated rag_results.jsonl")

# Rebuild summary CSV
import pandas as pd
summary = []
for r in results:
    predicted = r["generated_answer"]
    summary.append({
        "id": r["id"],
        "query": r["query"],
        "ground_truth": r["ground_truth"],
        "predicted": predicted,
        "correct": predicted == r["ground_truth"].lower(),
        "votes": str(r["votes"]),
        "num_docs_retrieved": len(r.get("docs_after_rerank", [])),
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(RESULTS_DIR / "rag_summary.csv", index=False)

accuracy = df_summary["correct"].mean()
unknowns = (df_summary["predicted"] == "unknown").sum()
print(f"Updated rag_summary.csv")
print(f"New accuracy: {accuracy:.2%}")
print(f"Remaining unknowns: {unknowns}")
print(f"\nPrediction distribution:")
print(df_summary["predicted"].value_counts().to_string())
