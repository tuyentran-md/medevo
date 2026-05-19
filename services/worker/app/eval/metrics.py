from __future__ import annotations


def confusion_matrix(expected: list[bool], observed: list[bool]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(expected, observed, strict=True):
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def false_negative_rate(matrix: dict[str, int]) -> float:
    denom = matrix["tp"] + matrix["fn"]
    return 0.0 if denom == 0 else matrix["fn"] / denom


def false_positive_rate(matrix: dict[str, int]) -> float:
    denom = matrix["fp"] + matrix["tn"]
    return 0.0 if denom == 0 else matrix["fp"] / denom
