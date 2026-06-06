import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """Top-k accuracy."""
    if top_k == 1:
        preds = logits.argmax(dim=1)
    else:
        _, preds = logits.topk(top_k, dim=1)
        correct = (preds == targets.view(-1, 1)).any(dim=1)
        return correct.float().mean().item()
    return (preds == targets).float().mean().item()


def precision_recall_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    per_class = {}
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    if average == "macro":
        return {
            "precision": sum(v["precision"] for v in per_class.values()) / num_classes,
            "recall": sum(v["recall"] for v in per_class.values()) / num_classes,
            "f1": sum(v["f1"] for v in per_class.values()) / num_classes,
        }
    return per_class[0]
