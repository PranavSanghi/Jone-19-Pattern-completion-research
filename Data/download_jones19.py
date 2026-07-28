

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
import os


def main():
    output_dir = Path(__file__).parent / "raw" / "JONES-19"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading JONES-19 from HuggingFace...")
    ds = load_dataset("harvardseas-cultural-ornaments/JONES-19", split="train")
    print(f"Loaded {len(ds)} images")

    
    for row in tqdm(ds, desc="Saving images"):
        cat_dir = output_dir / row["category"]
        cat_dir.mkdir(exist_ok=True)
        row["image"].save(cat_dir / row["image_file"])

    
    meta_path = output_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for row in ds:
            f.write(json.dumps({
                "image_file": row["image_file"],
                "name": row["name"],
                "category": row["category"],
                "subgroup": row["subgroup"],
                "width": row["width"],
                "height": row["height"],
                "page": row["page"],
            }) + "\n")

    print(f"\nDone! {len(ds)} images saved to {output_dir}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
