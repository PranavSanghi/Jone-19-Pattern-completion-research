import random

import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


class Candidatecreator(Dataset):
    def __init__(self, jsonl_path, data_root, sample_negatives=None):
        self.data_root = Path(data_root)
        self.sample_negatives = sample_negatives
        self.pairs = []
        self.hole_sizes = []
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                self._extract_pairs(sample)
        print(f"Loaded {len(self.pairs)} pairs from {jsonl_path}")

    def _extract_pairs(self, sample):
        context_path = self.data_root / "contexts" / f"{sample['sample_id']}.png"
        for hole in sample["holes"]:
            hole_idx = hole["hole_idx"]
            correct = []
            negatives = []
            for candidate in sample["candidates"]:
                label = (
                    1
                    if candidate["type"] == "correct"
                    and candidate["for_hole"] == hole_idx
                    else 0
                )
                pair = {
                    "context_path": str(context_path),
                    "hole_x": hole["x"],
                    "hole_y": hole["y"],
                    "candidate_path": str(self.data_root / candidate["path"]),
                    "label": label,
                }
                if label == 1:
                    correct.append(pair)
                else:
                    negatives.append(pair)
            self.pairs.extend(correct)
            if self.sample_negatives is not None:
                chosen = random.sample(
                    negatives, min(self.sample_negatives, len(negatives))
                )
                self.pairs.extend(chosen)
                self.hole_sizes.append(len(correct) + len(chosen))
            else:
                self.pairs.extend(negatives)
                self.hole_sizes.append(len(correct) + len(negatives))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img = Image.open(pair["context_path"]).convert("RGB")
        patch = Image.open(pair["candidate_path"]).convert("RGB")
        hx = pair["hole_x"]
        hy = pair["hole_y"]
        img.paste(patch, (hx + 16, hy + 16))
        cx = hx + 48
        cy = hy + 48
        left = cx - 96
        right = cx + 96
        up = cy - 96
        down = cy + 96
        img_w, img_h = img.size
        pad_l = max(0, -left)
        pad_up = max(0, -up)
        pad_r = max(0, right - img_w)
        pad_down = max(0, down - img_h)
        left = max(0, left)
        up = max(0, up)
        right = min(img_w, right)
        down = min(img_h, down)
        img = img.crop((left, up, right, down))
        if pad_l or pad_up or pad_r or pad_down:
            temp = Image.new("RGB", (192, 192), (0, 0, 0))
            temp.paste(img, (pad_l, pad_up))
            img = temp
        to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_t = to_tensor(img)
        label = torch.tensor(pair["label"], dtype=torch.float32)
        return img_t, label
