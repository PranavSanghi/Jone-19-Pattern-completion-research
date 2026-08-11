import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset2 import ds as HoleDataset
from model2 import MultiClassMatcher
from tqdm import tqdm


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    epochs = 50
    batch_size = 8
    lr = 1e-5
    patience = 5

    train_set = HoleDataset("../../Data/processed/train.jsonl", "../../Data/processed")
    val_set = HoleDataset("../../Data/processed/val.jsonl", "../../Data/processed")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    net = MultiClassMatcher().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    stale = 0

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for composites, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            composites = composites.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = net(composites)
            loss = criterion(scores, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            preds = scores.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}")

        net.eval()
        val_total_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for composites, labels in val_loader:
                composites = composites.to(device)
                labels = labels.to(device)

                scores = net(composites)
                loss = criterion(scores, labels)
                val_total_loss += loss.item()

                preds = scores.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_total_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f} ({val_correct}/{val_total})")

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            torch.save(net.state_dict(), "best_model_mc.pth")
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            stale += 1
            print(f"  No improvement ({stale}/{patience})")
            if stale >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()