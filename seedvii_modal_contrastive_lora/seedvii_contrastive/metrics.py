from __future__ import annotations

import numpy as np


def accuracy_macro_f1(y_true, y_pred, num_classes: int = 3) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(float(f1))
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "f1_per_class": f1s}
