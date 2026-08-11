import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


class ds(Dataset):
    def __init__(self,jsonl_path,data_root):
        self.data_root = Path(data_root)
        self.holes=[]
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                self._extract_holes(sample)
        print(f"Loaded {len(self.holes)} holes from {jsonl_path}")

    def _extract_holes(self, sample):
        if len(sample["candidates"]) != 32:
            return
        context_path = self.data_root / "contexts" / f"{sample['sample_id']}.png"
        candidates = sorted(sample["candidates"], key=lambda c: c["idx"])
        for hole in sample["holes"]:
            correct_idx = None
            for i, c in enumerate(candidates):
                if c["type"] == "correct" and c["for_hole"] == hole["hole_idx"]:
                    correct_idx = i
                    break
            if correct_idx is None:
                continue
            self.holes.append(
                {
                    "context_path": str(context_path),
                    "hole_x": hole["x"],
                    "hole_y": hole["y"],
                    "candidate_paths": [
                        str(self.data_root / c["path"]) for c in candidates
                    ],
                    "correct_idx": correct_idx,
                }
            )
    def __len__(self):
        return len(self.holes)

    def __getitem__(self, idx):
        hole = self.holes[idx]

        context = Image.open(hole["context_path"]).convert("RGB")
        hx = hole["hole_x"]
        hy = hole["hole_y"]

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        composites = []
        for path in hole["candidate_paths"]:
            img = context.copy()
            patch = Image.open(path).convert("RGB")
            img.paste(patch, (hx + 16, hy + 16))
            img = img.resize((448, 448), Image.BILINEAR)#remember to experiment if low 
            composites.append(to_tensor(img))

        composites = torch.stack(composites)  # 32 3 224 224 
        label = hole["correct_idx"]

        return composites, label