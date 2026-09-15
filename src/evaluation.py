"""
Retrieval Evaluation
Precision@k, Recall@k, mAP, per-class breakdown, and a pretrained-vs-
fine-tuned comparison harness. Used offline (in the notebook / a script)
to measure retrieval quality — not called by the live app.
"""

import numpy as np
import pandas as pd
import faiss
from collections import defaultdict


def precision_at_k(retrieved_labels, query_label, k):
    """Fraction of the top-k retrieved items that share the query's label."""
    top_k = retrieved_labels[:k]
    hits = np.sum(top_k == query_label)
    return hits / k


def recall_at_k(retrieved_labels, query_label, k, total_relevant):
    """Fraction of ALL same-label items in the database found in the top-k."""
    top_k = retrieved_labels[:k]
    hits = np.sum(top_k == query_label)
    return hits / total_relevant if total_relevant > 0 else 0.0


def average_precision(retrieved_labels, query_label):
    """
    Average Precision for one query: precision computed at each position
    where a correct (same-label) item appears, averaged over the number
    of correct items found in the retrieved list.
    """
    relevant_flags = (retrieved_labels == query_label).astype(int)
    if relevant_flags.sum() == 0:
        return 0.0

    cumulative_hits = np.cumsum(relevant_flags)
    precisions_at_hits = cumulative_hits / (np.arange(len(relevant_flags)) + 1)
    ap = np.sum(precisions_at_hits * relevant_flags) / relevant_flags.sum()
    return ap


def evaluate_retrieval_system(
    train_emb_norm, test_emb_norm, train_meta, test_meta, k_eval=10
):
    """
    Builds a FAISS index from train_emb_norm, searches it with every
    embedding in test_emb_norm, and returns overall + per-class metrics.
    Reusable for comparing any two embedding sets (e.g. pretrained vs
    fine-tuned, or a future metric-learning model).

    Returns:
        overall (dict): Precision@1, Precision@5, Recall@5, Recall@10, mAP
        per_class_table (DataFrame): same metrics broken down by category
    """
    dim = train_emb_norm.shape[1]
    local_index = faiss.IndexFlatIP(dim)
    local_index.add(train_emb_norm.astype(np.float32))

    sims, idxs = local_index.search(test_emb_norm.astype(np.float32), k_eval)

    train_class_counts = train_meta["category"].value_counts().to_dict()

    p1_list, p5_list, r5_list, r10_list, ap_list = [], [], [], [], []
    per_class = defaultdict(lambda: defaultdict(list))

    for i in range(len(test_meta)):
        query_label = test_meta.iloc[i]["category"]
        retrieved_labels = train_meta.iloc[idxs[i]]["category"].values
        total_relevant = train_class_counts.get(query_label, 0)

        p1 = precision_at_k(retrieved_labels, query_label, k=1)
        p5 = precision_at_k(retrieved_labels, query_label, k=5)
        r5 = recall_at_k(
            retrieved_labels, query_label, k=5, total_relevant=total_relevant
        )
        r10 = recall_at_k(
            retrieved_labels, query_label, k=10, total_relevant=total_relevant
        )
        ap = average_precision(retrieved_labels, query_label)

        p1_list.append(p1)
        p5_list.append(p5)
        r5_list.append(r5)
        r10_list.append(r10)
        ap_list.append(ap)

        per_class["Precision@1"][query_label].append(p1)
        per_class["Precision@5"][query_label].append(p5)
        per_class["Recall@5"][query_label].append(r5)
        per_class["Recall@10"][query_label].append(r10)
        per_class["mAP"][query_label].append(ap)

    overall = {
        "Precision@1": np.mean(p1_list),
        "Precision@5": np.mean(p5_list),
        "Recall@5": np.mean(r5_list),
        "Recall@10": np.mean(r10_list),
        "mAP": np.mean(ap_list),
    }

    per_class_table = pd.DataFrame(
        {
            metric: {cat: np.mean(vals) for cat, vals in per_class[metric].items()}
            for metric in per_class
        }
    ).round(4)

    return overall, per_class_table


def majority_vote_label(retrieved_labels, k=5):
    """Assigns a query the majority label among its top-k retrieved neighbors."""
    top_k = retrieved_labels[:k]
    vals, counts = np.unique(top_k, return_counts=True)
    return vals[np.argmax(counts)]
