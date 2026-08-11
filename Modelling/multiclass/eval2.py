import torch
from torch.utils.data import DataLoader
from dataset2 import ds as  HoleDataset
from model2 import MultiClassMatcher


def evaluate():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    net = MultiClassMatcher().to(device)
    net.load_state_dict(torch.load("best_model_mc.pth", map_location=device))
    net.eval()

    test_set = HoleDataset("../../Data/processed/test.jsonl", "../../Data/processed")
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for composites, labels in test_loader:
            composites = composites.to(device)
            labels = labels.to(device)

            scores = net(composites)
            preds = scores.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.4f} ({correct}/{total})")
    print(f"Random baseline: {1/32:.4f}")


if __name__ == "__main__":
    evaluate()