"""
Common metric utilities for consistent binary/macro metric reporting.
All functions compute the positive-class (malignant) binary precision/recall/f1
and also provide macro/balanced variants where useful. This centralizes
metric computation so different modules report consistent numbers.
"""
from typing import Optional, Dict
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score, roc_auc_score
)


def classification_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None,
                           pos_label: int = 1) -> Dict:
    """Compute a consistent set of classification metrics.

    Returns a dict with:
      - confusion_matrix: 2x2 list [ [tn, fp], [fn, tp] ]
      - precision: positive-class precision (binary)
      - recall: positive-class recall (binary)
      - f1_score: positive-class F1 (binary)
      - f1_macro: macro-averaged F1
      - accuracy: plain accuracy
      - balanced_accuracy: balanced accuracy
      - auc_roc: ROC AUC (if y_score provided), otherwise None
    """
    cm = confusion_matrix(y_true, y_pred)

    # Binary (positive-class) metrics
    precision = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    # Macro / balanced metrics for additional context
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    auc = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None

    return {
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f1_macro': f1_macro,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc_roc': auc
    }
