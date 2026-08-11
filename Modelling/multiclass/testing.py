from pathlib import Path
from unittest import loader

import torch
from torch.utils.data import DataLoader

from dataset2 import ds as HoleDataset
from model2 import MultiClassMatcher

DATA_ROOT = Path(__file__).resolve().parents[2] / "Data" / "processed"


def main():
    train_jsonl = DATA_ROOT / "train.jsonl"
    val_jsonl = DATA_ROOT / "val.jsonl"
    print(f"data_root: {DATA_ROOT}")
    print(f"train jsonl exists: {train_jsonl.is_file()}")
    print(f"val jsonl exists: {val_jsonl.is_file()}")

    train_set = HoleDataset(str(train_jsonl), str(DATA_ROOT))
    val_set = HoleDataset(str(val_jsonl), str(DATA_ROOT))
    print(f"\nlen(train_set) = {len(train_set)}")
    print(f"len(val_set) = {len(val_set)}")

    n_cands = {}
    for h in train_set.holes:
        k = len(h["candidate_paths"])
        n_cands[k] = n_cands.get(k, 0) + 1
    print(f"train candidate-count distribution: {dict(sorted(n_cands.items()))}")

    sample = train_set.holes[0]
    print("\n--- first hole metadata ---")
    print(f"context_path: {sample['context_path']}")
    print(f"hole_x, hole_y: {sample['hole_x']}, {sample['hole_y']}")
    print(f"num candidates: {len(sample['candidate_paths'])}")
    print(f"correct_idx: {sample['correct_idx']}")
    print(f"first candidate: {sample['candidate_paths'][0]}")
    print(f"correct candidate: {sample['candidate_paths'][sample['correct_idx']]}")

    print("\n--- __getitem__(0) ---")
    composites, label = train_set[0]
    print(f"composites type: {type(composites)}")
    print(f"composites shape: {tuple(composites.shape)}")
    print(f"composites dtype: {composites.dtype}")
    print(f"composites min/max/mean: {composites.min():.3f} / {composites.max():.3f} / {composites.mean():.3f}")
    print(f"label: {label} (type={type(label).__name__})")
    print(f"label in range [0, num_cands): {0 <= int(label) < composites.shape[0]}")

    print("\n--- DataLoader batch ---")
    loader = DataLoader(train_set, batch_size=2, shuffle=False, num_workers=0)
    try:
        batch_comp, batch_labels = next(iter(loader))
        print(f"batch composites shape: {tuple(batch_comp.shape)}")
        print(f"batch labels shape: {tuple(batch_labels.shape)}")
        print(f"batch labels: {batch_labels.tolist()}")
    except Exception as e:
        print(f"DataLoader failed: {type(e).__name__}: {e}")
        print("Likely cause: variable number of candidates across holes (can't stack).")
        return

    print("\n--- model forward ---")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    net = MultiClassMatcher().to(device)
    net.eval()
    with torch.no_grad():
        scores = net(batch_comp.to(device))
    print(f"scores shape: {tuple(scores.shape)}")
    print(f"scores[0] (first {min(8, scores.shape[1])}): {scores[0, :8].tolist()}")
    preds = scores.argmax(dim=1)
    print(f"argmax preds: {preds.tolist()}")
    print(f"labels:       {batch_labels.tolist()}")
    print("\nOK — dataset / loader / model shapes look consistent.")
    comp, label = train_set[0]
    print(comp.shape, label)
    batch_comp, batch_labels = next(iter(loader))
    print(batch_comp.shape, batch_labels)


if __name__ == "__main__":
    main()
