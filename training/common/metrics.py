"""
Metrics used during training and (copied) in evaluation/evaluate.py -- keep the two in sync.
"""

import numpy as np
from sklearn.metrics import f1_score, recall_score

BENIGN = 0
ATTACK_LABELS = {1: "prompt_injection", 2: "jailbreak"}


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics = {"macro_f1": float(f1_score(labels, preds, average="macro"))}

    recalls = recall_score(labels, preds, labels=list(ATTACK_LABELS), average=None, zero_division=0)
    for label_id, recall in zip(ATTACK_LABELS, recalls):
        metrics[f"recall_{ATTACK_LABELS[label_id]}"] = float(recall)

    benign_mask = labels == BENIGN
    false_positive_rate = float((preds[benign_mask] != BENIGN).mean()) if benign_mask.any() else 0.0
    metrics["false_positive_rate"] = false_positive_rate

    return metrics
