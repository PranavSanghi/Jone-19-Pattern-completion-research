import os

import torch
from torch.utils.data import DataLoader
from dataset1 import Candidatecreator
from model1 import model as PatchMatcher


def evaluate():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

    net = PatchMatcher().to(device)
    net.load_state_dict(torch.load("best_model1.pth", map_location=device))
    net.eval()

    test_set = Candidatecreator(
        "../Data/processed/test.jsonl",
        "../Data/processed",
    )
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, **loader_kwargs
    )

    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            outputs = net(imgs).squeeze(1)

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_scores.extend(outputs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    binary_acc = correct / total
    print(f"Binary Accuracy: {binary_acc:.4f} ({correct}/{total})")

    hole_correct = 0
    hole_total = 0
    offset = 0
    for n in test_set.hole_sizes:
        chunk_scores = all_scores[offset : offset + n]
        chunk_labels = all_labels[offset : offset + n]
        offset += n
        if not chunk_scores:
            continue
        pred_idx = chunk_scores.index(max(chunk_scores))
        if chunk_labels[pred_idx] == 1:
            hole_correct += 1
        hole_total += 1

    hole_acc = hole_correct / hole_total if hole_total > 0 else 0
    print(f"Per-Hole Top-1 Accuracy: {hole_acc:.4f} ({hole_correct}/{hole_total})")
    print(f"Random baseline would be: {1/32:.4f} (1/32)")


if __name__ == "__main__":
    evaluate()
