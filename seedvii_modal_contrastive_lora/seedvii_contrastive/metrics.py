from __future__ import annotations

import numpy as np


def accuracy_macro_f1(y_true, y_pred, num_classes: int = 3) -> dict:
    """Compute accuracy and macro F1 score.
    
    FIX: Ensures numpy array inputs for numerical stability.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    if len(y_true) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "f1_per_class": [0.0] * num_classes}
    
    acc = float((y_true == y_pred).mean())
    
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(float(f1))
    
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "f1_per_class": f1s}
