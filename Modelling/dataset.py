from calendar import c
import torch 
import json 
from torch.utils.data import Dataset 
from PIL import Image 
from pathlib import Path 
from torchvision import transforms 

class Candidatecreator (Dataset):
    def __init__(self,jsonl_path,data_root):
        self.data_root = data_root
        self.pairs =  []
        with open(jsonl_path) as f:
            for line in f:
                sample1 = json.loads(line)
                self._extract_pairs(sample1)
        print (f"#1 len(self.pairs) from {jsonl_path}")
    def _extract_pairs(self,sample1):
        contextimg  = self.data_root / "contexts" / f"{sample1['sample_id']}_context.png"#path to conext image
        for hole in sample1['holes']:
            idx = hole['idx']
            x = hole['x']#leftmost x coordinate  
            y = hole['y']#topmost y coordinate

            for candidate in hole['candidates']:
                if candidate['type'] == "correct" and candidate["for_hole"] == idx:#Correct context patch pair
                    label = 1
                else:
                    label = 0#this is an incorrect pair 
                self.pairs.append({
                    "context_path":contextimg,
                    "hole_x":x,
                    "hole_y":y,
                    "candidate_path":candidate["path"],
                    "label":label 

                })

    def _len_(self):
        return len(self.pairs)
    def __getitem__(self,idx):
        pair = self.pairs[idx]
        img = Image.open(pair["context_path"]).convert("RGB")
        patch = Image.open(pair["candidate_path"]).convert("RGB")
        cx = pair["hole_x"]+48 
        cy = pair["hole_y"]+48 
        left = cx - 96
        right = cx + 96
        up = cy - 96
        down= cy + 96
        img_w, img_h = img.size
        pad_l = max(0, -left)
        pad_up = max(0, -up)
        pad_r = max(0, right - img_w)
        pad_down = max(0, down- img_h)

        left = max(0,left)
        up = max(0,up)
        right = min(img_w,right)
        down = min(img_h,down)
        img = img.crop((left,up,right,down))
        if pad_l or pad_up or pad_r or pad_down:
         temp = Image.new("RGB", (192, 192), (0, 0, 0))
         temp.paste(img, (pad_l, pad_up))
         img = temp
        to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        img_t = to_tensor(img)
        patch_t = to_tensor(patch)
        label = torch.tensor(pair["label"], dtype = torch.float32)
        return img_t, patch_t, label
        


#Prepare hole-patch pairs for training
#32 * 4 pairs per image
#For each pair we return the patch tensor and a tensor of the surrounding 192*192 context , if it exceeds the image we patch 


