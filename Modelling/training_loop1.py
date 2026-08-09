import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset1 import Candidatecreator
from model1 import model as lol
from tqdm import tqdm


def train():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    epochs = 50
    batch_size = 256
    lr = 1e-4
    patience = 5
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    print(f"DataLoader workers: {num_workers}", flush=True)

    train_set = Candidatecreator(
        jsonl_path="../Data/processed/train.jsonl",
        data_root="../Data/processed",
        
    )
    val_set = Candidatecreator(
        jsonl_path="../Data/processed/val.jsonl",
        data_root="../Data/processed",
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    net = lol().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best = -1.0
    stale = 0
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = batch
            imgs = imgs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = net(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        net.eval()
        with torch.no_grad():
            val_total_loss = 0
            all_scores = []
            all_labels = []
            for batch in val_loader:
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory).float()
                outputs = net(imgs).squeeze(1)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                all_scores.extend(outputs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            val_loss = val_total_loss / len(val_loader)

            hole_correct = 0
            hole_total = 0
            offset = 0
            for n in val_set.hole_sizes:
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
            print(
                f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
                f"Hole Acc: {hole_acc:.4f} ({hole_correct}/{hole_total})"
            )
        if hole_acc > best:
            best = hole_acc
            stale = 0
            torch.save(net.state_dict(), "best_model1.pth")
            print(f"Saved best model (hole_acc={best:.4f})")
        else:
            stale += 1
            print(f"No improvement ({stale}/{patience})")
            if stale >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()
