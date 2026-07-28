
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = Path(__file__).parent / "Data" / "raw" / "JONES-19"
IMAGE_EXTS = {".png"}
N_PER_CLASS = 5


def collect_samples(raw_dir: Path, n_per_class: int):
    samples = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if images:
            samples.append((class_dir.name, images[:n_per_class]))
    return samples


def main():
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Raw data not found: {RAW_DIR}")

    samples = collect_samples(RAW_DIR, N_PER_CLASS)
    if not samples:
        raise SystemExit(f"No class images found under {RAW_DIR}")

    rows = len(samples)
    cols = N_PER_CLASS
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))

    for row, (label, paths) in enumerate(samples):
        for col in range(cols):
            ax = axes[row, col]
            if col < len(paths):
                ax.imshow(Image.open(paths[col]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col == 0:
                ax.set_ylabel(
                    label,
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                )

            # Label each image with class name + index
            ax.set_title(f"{label} ({col + 1})", fontsize=7, pad=2)

    fig.suptitle(f"JONES-19 — {N_PER_CLASS} images per class", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.12, 0, 1, 0.98])
    plt.show()


if __name__ == "__main__":
    main()
#Celtic and Moresque are taken as test classes because they seem to be the most distinct classes
