"""Run self-consistency voting for RAG pipeline (standalone script)."""

import json
import os
import re
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

RESULTS_DIR = Path(__file__).parent.parent / "results"
client = OpenAI()

NUM_VOTES = 5
NUM_WORKERS = 5
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are a medical expert answering yes/no/maybe questions based on PubMed abstracts.
Rules:
- Answer "yes" if the evidence supports the claim.
- Answer "no" if the evidence contradicts or does not support the claim.
- Answer "maybe" ONLY if the evidence is genuinely mixed or insufficient to decide.
- Most questions have a definitive answer. Prefer "yes" or "no" over "maybe".
- Respond with exactly one word: yes, no, or maybe."""

FEW_SHOT_EXAMPLES = """Example 1:
Question: Is increased gravitational stress beneficial for bone density?
Context: Studies show weight-bearing exercise increases bone mineral density by 2-8% in postmenopausal women...
Answer: yes

Example 2:
Question: Does smoking cessation reduce cardiovascular risk?
Context: After 1 year of cessation, coronary heart disease risk drops by 50%. After 15 years, risk equals that of a non-smoker...
Answer: yes

Example 3:
Question: Is homeopathy effective for treating asthma?
Context: A systematic review of 6 RCTs found no significant difference between homeopathic treatments and placebo in lung function or symptom scores...
Answer: no

Example 4:
Question: Can MRI replace biopsy for diagnosing prostate cancer?
Context: MRI showed sensitivity of 91% but specificity of only 37%. While useful for risk stratification, results were inconsistent across centers...
Answer: maybe

"""


def extract_answer(text):
    text = text.strip().lower()
    if text in ("yes", "no", "maybe"):
        return text
    match = re.search(r'\b(yes|no|maybe)\b', text, re.I)
    return match.group(1).lower() if match else "unknown"


def majority_vote(answers):
    valid = [a for a in answers if a in ("yes", "no", "maybe")]
    if not valid:
        return "unknown"
    return Counter(valid).most_common(1)[0][0]


def build_prompt(query_text, contexts):
    context_parts = [f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    context = "\n\n".join(context_parts)
    return f"""{FEW_SHOT_EXAMPLES}Now answer this question:

Context:
{context}

Question: {query_text}

Answer:"""


def call_openai(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = min(2 ** attempt, 30)
                print(f"    Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    # Final attempt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    # Load existing results (has retrieval data)
    print("Loading results...")
    results = []
    with open(RESULTS_DIR / "rag_results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    print(f"Loaded {len(results)} results")

    # Build prompts
    prompts = []
    for r in results:
        prompt = build_prompt(r["query"], r["contexts"])
        prompts.append(prompt)

    # Generate votes with rate-limited batching
    total_calls = len(results) * NUM_VOTES
    print(f"\nSelf-consistency: {NUM_VOTES} votes x {len(results)} queries = {total_calls} API calls")
    print(f"Workers: {NUM_WORKERS}")

    all_votes = [[] for _ in range(len(results))]

    # Process in batches to respect rate limits (450 per minute)
    BATCH_SIZE = 400
    tasks = [(i, v, prompts[i]) for i in range(len(results)) for v in range(NUM_VOTES)]

    completed = 0
    errors = 0
    batch_start = time.time()

    for batch_idx in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE

        # Rate limit: wait if needed to stay under 500 RPM
        if batch_idx > 0:
            elapsed = time.time() - batch_start
            if elapsed < 65:  # Wait at least 65s between batches
                wait = 65 - elapsed
                print(f"  Rate limit pause: {wait:.0f}s...")
                time.sleep(wait)
            batch_start = time.time()

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} calls)...")

        def do_call(task):
            idx, vote_idx, prompt = task
            answer = call_openai(prompt)
            return idx, answer

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(do_call, t): t for t in batch}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    idx, answer = future.result()
                    all_votes[idx].append(answer)
                except Exception as e:
                    idx = task[0]
                    all_votes[idx].append(f"ERROR: {str(e)}")
                    errors += 1

                completed += 1
                if completed % 500 == 0:
                    print(f"    {completed}/{total_calls} done...")

    print(f"\n{total_calls - errors} succeeded, {errors} errors")

    # Apply majority voting and save
    new_results = []
    for i, r in enumerate(results):
        votes_extracted = [extract_answer(v) for v in all_votes[i]]
        winner = majority_vote(votes_extracted)
        new_results.append({
            "id": r["id"],
            "query": r["query"],
            "ground_truth": r["ground_truth"],
            "golden_doc": r["golden_doc"],
            "generated_answer": winner,
            "votes": votes_extracted,
            "docs_before_rerank": r["docs_before_rerank"],
            "docs_after_rerank": r["docs_after_rerank"],
            "contexts": r["contexts"],
        })

    # Save
    with open(RESULTS_DIR / "rag_results.jsonl", "w") as f:
        for r in new_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    correct = sum(1 for r in new_results if r["generated_answer"] == r["ground_truth"].lower())
    total = len(new_results)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")

    # Distribution
    from collections import Counter
    pred_dist = Counter(r["generated_answer"] for r in new_results)
    gt_dist = Counter(r["ground_truth"].lower() for r in new_results)
    print(f"\nPredicted: {dict(pred_dist)}")
    print(f"GT:        {dict(gt_dist)}")

    # Unanimity
    unanimity = sum(1 for r in new_results
                    if len(set(v for v in r["votes"] if v in ("yes","no","maybe"))) == 1)
    print(f"Unanimous: {unanimity}/{total} ({unanimity/total:.1%})")

    # Save summary CSV
    import pandas as pd
    summary = []
    for r in new_results:
        summary.append({
            "id": r["id"],
            "query": r["query"],
            "ground_truth": r["ground_truth"],
            "predicted": r["generated_answer"],
            "correct": r["generated_answer"] == r["ground_truth"].lower(),
            "votes": str(r["votes"]),
            "num_docs_retrieved": len(r["docs_after_rerank"]),
        })
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "rag_summary.csv", index=False)
    print("Saved rag_results.jsonl and rag_summary.csv")


if __name__ == "__main__":
    main()
