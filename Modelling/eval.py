import torch
import json
from torch.utils.data import DataLoader
from dataset import PatchMatchDataset
from model import PatchMatcher


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    
    net = PatchMatcher().to(device)
    net.load_state_dict(torch.load("best_model.pth", map_location=device))
    net.eval()

    
    test_set = PatchMatchDataset("../Data/processed/test.jsonl", "../Data/processed")
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

   
    correct = 0
    total = 0

    
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for context, candidate, labels in test_loader:
            context = context.to(device)
            candidate = candidate.to(device)
            labels = labels.to(device)

            outputs = net(context, candidate).squeeze(1)

            
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_scores.extend(outputs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    binary_acc = correct / total
    print(f"Binary Accuracy: {binary_acc:.4f} ({correct}/{total})")

    
    hole_correct = 0
    hole_total = 0

    for i in range(0, len(all_scores), 32):
        chunk_scores = all_scores[i:i+32]
        chunk_labels = all_labels[i:i+32]

        if len(chunk_scores) < 32:
            break

        
        pred_idx = chunk_scores.index(max(chunk_scores))
        
        if chunk_labels[pred_idx] == 1:
            hole_correct += 1
        hole_total += 1

    hole_acc = hole_correct / hole_total if hole_total > 0 else 0
    print(f"Per-Hole Top-1 Accuracy: {hole_acc:.4f} ({hole_correct}/{hole_total})")
    print(f"Random baseline would be: {1/32:.4f} (1/32)")#1/32 is the random baseline


if __name__ == "__main__":
    evaluate()