"""Generate all figures for the RAG pipeline evaluation report (v3 CoT)."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

RESULTS_DIR = Path(__file__).parent
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
df_summary = pd.read_csv(RESULTS_DIR / "rag_summary.csv")
df_ragas = pd.read_csv(RESULTS_DIR / "ragas_results.csv")
df_retrieval = pd.read_csv(RESULTS_DIR / "retrieval_metrics.csv")
fc_col = [c for c in df_ragas.columns if 'factual_correctness' in c][0]

# v3 accuracy
v3_accuracy = df_summary['correct'].mean() * 100

# Load ablation if available
try:
    df_ablation = pd.read_csv(RESULTS_DIR / "ablation_reranker.csv")
    has_ablation = True
except FileNotFoundError:
    has_ablation = False


# ============================================================
# Figure 1: Evolution of Accuracy across versions (v0-v3)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
versions = ['v0\nClaude +\nparser fragil', 'v1\nGPT-4o-mini +\nparser robusto',
            'v2\nFew-shot +\nself-consistency', 'v3\nChain-of-Thought\n+ CoT']
accuracies = [9.6, 47.0, 51.0, v3_accuracy]
colors = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60']
bars = ax.bar(versions, accuracies, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Evolucao da Accuracy ao Longo das Otimizacoes')
ax.set_ylim(0, 70)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_accuracy_evolution.png")
plt.close()
print("Figure 1: accuracy evolution")


# ============================================================
# Figure 2: Confusion Matrix
# ============================================================
labels = ['yes', 'no', 'maybe']
cm = np.zeros((3, 3), dtype=int)
for i, gt in enumerate(labels):
    subset = df_summary[df_summary.ground_truth == gt]
    for j, pred in enumerate(labels):
        cm[i, j] = (subset.predicted == pred).sum()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=13)
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Ground Truth', fontsize=13)
ax.set_title('Matriz de Confusao (500 amostras)')

for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_confusion_matrix.png")
plt.close()
print("Figure 2: confusion matrix")


# ============================================================
# Figure 3: Prediction vs Ground Truth Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ground truth
gt_counts = df_summary.ground_truth.value_counts()
axes[0].bar(['yes', 'no', 'maybe'],
            [gt_counts.get('yes', 0), gt_counts.get('no', 0), gt_counts.get('maybe', 0)],
            color=['#27AE60', '#E74C3C', '#F39C12'], edgecolor='white')
axes[0].set_title('Distribuicao Ground Truth')
axes[0].set_ylabel('Contagem')
for i, v in enumerate([gt_counts.get('yes', 0), gt_counts.get('no', 0), gt_counts.get('maybe', 0)]):
    axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# Predicted
pred_counts = df_summary.predicted.value_counts()
axes[1].bar(['yes', 'no', 'maybe'],
            [pred_counts.get('yes', 0), pred_counts.get('no', 0), pred_counts.get('maybe', 0)],
            color=['#27AE60', '#E74C3C', '#F39C12'], edgecolor='white')
axes[1].set_title('Distribuicao Predicted')
axes[1].set_ylabel('Contagem')
for i, v in enumerate([pred_counts.get('yes', 0), pred_counts.get('no', 0), pred_counts.get('maybe', 0)]):
    axes[1].text(i, v + 3, str(v), ha='center', fontweight='bold')

for ax in axes:
    ax.set_ylim(0, 310)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Ground Truth vs Predicted (500 amostras)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_distribution_comparison.png")
plt.close()
print("Figure 3: distribution comparison")


# ============================================================
# Figure 4: Per-class Accuracy
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
class_acc = {}
for gt in ['yes', 'no', 'maybe']:
    subset = df_summary[df_summary.ground_truth == gt]
    class_acc[gt] = subset.correct.mean() * 100

bars = ax.bar(class_acc.keys(), class_acc.values(),
              color=['#27AE60', '#E74C3C', '#F39C12'], edgecolor='white', width=0.5)
for bar, val in zip(bars, class_acc.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=13)
ax.axhline(y=v3_accuracy, color='gray', linestyle='--', alpha=0.7,
           label=f'Accuracy geral ({v3_accuracy:.1f}%)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy por Classe (v3 CoT)')
ax.set_ylim(0, 80)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_per_class_accuracy.png")
plt.close()
print("Figure 4: per-class accuracy")


# ============================================================
# Figure 5: Retrieval Metrics - Before vs After Rerank
# ============================================================
k_values = [1, 3, 5, 10, 20]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hit Rate
hit_before = [df_retrieval[df_retrieval.metric == f'hit_rate@{k}']['before_rerank'].values[0] for k in k_values]
hit_after = [df_retrieval[df_retrieval.metric == f'hit_rate@{k}']['after_rerank'].values[0] for k in k_values]
x = np.arange(len(k_values))
w = 0.35
axes[0].bar(x - w/2, hit_before, w, label='Before Rerank', color='#4C72B0')
axes[0].bar(x + w/2, hit_after, w, label='After Rerank', color='#DD8452')
axes[0].set_xticks(x)
axes[0].set_xticklabels(k_values)
axes[0].set_xlabel('k')
axes[0].set_ylabel('Hit Rate')
axes[0].set_title('Hit Rate @ k')
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# Recall
recall_before = [df_retrieval[df_retrieval.metric == f'recall@{k}']['before_rerank'].values[0] for k in k_values]
recall_after = [df_retrieval[df_retrieval.metric == f'recall@{k}']['after_rerank'].values[0] for k in k_values]
axes[1].bar(x - w/2, recall_before, w, label='Before Rerank', color='#4C72B0')
axes[1].bar(x + w/2, recall_after, w, label='After Rerank', color='#DD8452')
axes[1].set_xticks(x)
axes[1].set_xticklabels(k_values)
axes[1].set_xlabel('k')
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall @ k')
axes[1].legend()
axes[1].set_ylim(0, 1.05)

# Key metrics at k=5
key_metrics = ['hit_rate@5', 'precision@5', 'recall@5']
key_labels = ['Hit Rate@5', 'Precision@5', 'Recall@5']
mrr_before = df_retrieval[df_retrieval.metric == 'mrr']['before_rerank'].values[0]
mrr_after = df_retrieval[df_retrieval.metric == 'mrr']['after_rerank'].values[0]
before_vals = [df_retrieval[df_retrieval.metric == m]['before_rerank'].values[0] for m in key_metrics] + [mrr_before]
after_vals = [df_retrieval[df_retrieval.metric == m]['after_rerank'].values[0] for m in key_metrics] + [mrr_after]
key_labels.append('MRR')

x3 = np.arange(len(key_labels))
axes[2].bar(x3 - w/2, before_vals, w, label='Before Rerank', color='#4C72B0')
axes[2].bar(x3 + w/2, after_vals, w, label='After Rerank', color='#DD8452')
axes[2].set_xticks(x3)
axes[2].set_xticklabels(key_labels, rotation=15)
axes[2].set_ylabel('Score')
axes[2].set_title('Metricas-chave de Retrieval')
axes[2].legend()
axes[2].set_ylim(0, 1.05)

plt.suptitle('Metricas de Retrieval: Before vs After Reranking', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_retrieval_metrics.png")
plt.close()
print("Figure 5: retrieval metrics")


# ============================================================
# Figure 6: RAGAS Score Distributions
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df_ragas['faithfulness'].dropna(), bins=25, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].axvline(df_ragas['faithfulness'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {df_ragas["faithfulness"].mean():.3f}')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Contagem')
axes[0].set_title('Distribuicao de Faithfulness (v3 CoT)')
axes[0].legend(fontsize=11)

axes[1].hist(df_ragas[fc_col].dropna(), bins=25, color='#DD8452', edgecolor='white', alpha=0.85)
axes[1].axvline(df_ragas[fc_col].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {df_ragas[fc_col].mean():.3f}')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('Contagem')
axes[1].set_title('Distribuicao de Factual Correctness')
axes[1].legend(fontsize=11)

plt.suptitle('RAGAS Scores (500 amostras, v3 CoT)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_ragas_distributions.png")
plt.close()
print("Figure 6: RAGAS distributions")


# ============================================================
# Figure 7: All Metrics Summary Dashboard
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

metrics = {
    'Accuracy': v3_accuracy / 100,
    'Factual\nCorrectness': df_ragas[fc_col].mean(),
    'Faithfulness': df_ragas['faithfulness'].mean(),
    'Hit Rate@5': df_retrieval[df_retrieval.metric == 'hit_rate@5']['after_rerank'].values[0],
    'Precision@5': df_retrieval[df_retrieval.metric == 'precision@5']['after_rerank'].values[0],
    'Recall@5': df_retrieval[df_retrieval.metric == 'recall@5']['after_rerank'].values[0],
    'MRR': df_retrieval[df_retrieval.metric == 'mrr']['after_rerank'].values[0],
}

colors = ['#27AE60', '#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#1ABC9C']
bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Score')
ax.set_title('Painel Completo de Metricas do Pipeline RAG (v3 CoT)')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_metrics_dashboard.png")
plt.close()
print("Figure 7: metrics dashboard")


# ============================================================
# Figure 8: Factual Correctness + Faithfulness Evolution (v0-v3)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Factual Correctness
versions_short = ['v0\n(Claude)', 'v1\n(GPT-4o-mini)', 'v2\n(Few-shot)', 'v3\n(CoT)']
fc_vals = [0.022, 0.470, 0.510, df_ragas[fc_col].mean()]
colors_fc = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60']
bars = axes[0].bar(versions_short, fc_vals, color=colors_fc, edgecolor='white', width=0.5)
for bar, val in zip(bars, fc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Score')
axes[0].set_title('Factual Correctness (F1)')
axes[0].set_ylim(0, 0.8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Faithfulness
faith_vals = [0.770, 0.380, 0.347, df_ragas['faithfulness'].mean()]
bars2 = axes[1].bar(versions_short, faith_vals, color=colors_fc, edgecolor='white', width=0.5)
for bar, val in zip(bars2, faith_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Score')
axes[1].set_title('Faithfulness')
axes[1].set_ylim(0, 1.0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Evolucao das Metricas RAGAS', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_ragas_evolution.png")
plt.close()
print("Figure 8: RAGAS evolution")


# ============================================================
# Figure 9: Ablation Study - Reranker Impact
# ============================================================
if has_ablation:
    fig, ax = plt.subplots(figsize=(8, 5))
    configs = df_ablation['configuration'].tolist()
    accs = (df_ablation['accuracy'] * 100).tolist()
    colors_abl = ['#27AE60', '#E74C3C']
    bars = ax.bar(configs, accs, color=colors_abl, edgecolor='white', width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study: Impacto do Reranker na Accuracy')
    ax.set_ylim(0, max(accs) + 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_ablation_reranker.png")
    plt.close()
    print("Figure 9: ablation reranker")
else:
    print("Figure 9: SKIPPED (no ablation data)")


# ============================================================
# Figure 10: Semantic Evaluation
# ============================================================
semantic_path = RESULTS_DIR / "semantic_evaluation.csv"
if semantic_path.exists():
    df_sem = pd.read_csv(semantic_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sem_metrics = [
        ("bertscore_f1", "BERTScore F1", "#4C72B0"),
        ("rougeL_f1", "ROUGE-L F1", "#DD8452"),
        ("cosine_similarity", "Cosine Similarity (BGE)", "#55A868"),
    ]

    for ax, (col, title, color) in zip(axes, sem_metrics):
        ax.hist(df_sem[col].dropna(), bins=25, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(df_sem[col].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media: {df_sem[col].mean():.3f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Contagem')
        ax.set_title(title)
        ax.legend(fontsize=11)

    plt.suptitle('Avaliacao Semantica: CoT Reasoning vs Golden Documents', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_semantic_evaluation.png")
    plt.close()
    print("Figure 10: semantic evaluation")
else:
    print("Figure 10: SKIPPED (no semantic evaluation data)")


print(f"\nAll figures saved to {FIGURES_DIR}/")
