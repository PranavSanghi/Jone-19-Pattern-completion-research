#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

SEED = 42
HOLE_SIZE = 96
PATCH_SIZE = 64
HOLE_MARGIN = 16
MIN_HOLE_GAP = 16
MAX_HOLES = 4
GRID_STRIDE = 16
CANDIDATES_PER_HOLE = 8
TEST_CATEGORIES = {"Celtic", "Moresque"}
VAL_FRACTION = 0.1
MAX_WORKERS = 4
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def shannon_entropy(patch: Image.Image) -> float:
    gray = np.asarray(patch.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log(p)).sum())


def color_stats(patch: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(patch.convert("RGB"), dtype=np.float32)
    return arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))


def color_distance(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    return float(np.linalg.norm(a[0] - b[0]) + 0.5 * np.linalg.norm(a[1] - b[1]))


def boxes_conflict(a: tuple[int, int], b: tuple[int, int], size: int, gap: int) -> bool:
    ax, ay = a
    bx, by = b
    return not (
        ax + size + gap <= bx
        or bx + size + gap <= ax
        or ay + size + gap <= by
        or by + size + gap <= ay
    )


def candidate_hole_regions(width: int, height: int) -> list[tuple[int, int]]:
    if width < HOLE_SIZE or height < HOLE_SIZE:
        return []
    xs = list(range(0, width - HOLE_SIZE + 1, GRID_STRIDE))
    ys = list(range(0, height - HOLE_SIZE + 1, GRID_STRIDE))
    if xs[-1] != width - HOLE_SIZE:
        xs.append(width - HOLE_SIZE)
    if ys[-1] != height - HOLE_SIZE:
        ys.append(height - HOLE_SIZE)
    return [(x, y) for y in ys for x in xs]


def select_entropy_holes(
    image: Image.Image, rng: random.Random, max_holes: int = MAX_HOLES
) -> list[dict]:
    width, height = image.size
    positions = candidate_hole_regions(width, height)
    if not positions:
        return []

    scored = []
    for x, y in positions:
        region = image.crop((x, y, x + HOLE_SIZE, y + HOLE_SIZE))
        scored.append({"x": x, "y": y, "entropy": shannon_entropy(region)})

    entropies = np.array([s["entropy"] for s in scored], dtype=np.float64)
    order = np.argsort(entropies)
    n = len(scored)

    def percentile_slice(lo_pct: float, hi_pct: float) -> list[dict]:
        lo = int(math.floor((lo_pct / 100.0) * (n - 1)))
        hi = int(math.ceil((hi_pct / 100.0) * (n - 1)))
        idx = order[lo : hi + 1]
        return [scored[i] for i in idx]

    lo_pct, hi_pct = 60.0, 90.0
    selected: list[dict] = []

    while True:
        pool = percentile_slice(lo_pct, hi_pct)
        rng.shuffle(pool)
        selected = []
        for cand in pool:
            if all(
                not boxes_conflict(
                    (cand["x"], cand["y"]),
                    (s["x"], s["y"]),
                    HOLE_SIZE,
                    MIN_HOLE_GAP,
                )
                for s in selected
            ):
                selected.append(cand)
                if len(selected) >= max_holes:
                    break
        if len(selected) >= max_holes:
            break
        if lo_pct <= 0.0 and hi_pct >= 100.0:
            break
        lo_pct = max(0.0, lo_pct - 5.0)
        hi_pct = min(100.0, hi_pct + 5.0)

    if len(selected) < max_holes:
        remaining = [s for s in scored if s not in selected]
        remaining.sort(key=lambda s: abs(s["entropy"] - float(np.median(entropies))))
        for cand in remaining:
            if all(
                not boxes_conflict(
                    (cand["x"], cand["y"]),
                    (s["x"], s["y"]),
                    HOLE_SIZE,
                    MIN_HOLE_GAP,
                )
                for s in selected
            ):
                selected.append(cand)
                if len(selected) >= max_holes:
                    break

    return selected[:max_holes]


def extract_patch(image: Image.Image, x: int, y: int) -> Image.Image:
    return image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE)).convert("RGB")


def clamp_patch_xy(image: Image.Image, x: int, y: int) -> tuple[int, int]:
    w, h = image.size
    x = max(0, min(x, w - PATCH_SIZE))
    y = max(0, min(y, h - PATCH_SIZE))
    return x, y


def hole_center_xy(hole: dict) -> tuple[int, int]:
    return hole["x"] + HOLE_MARGIN, hole["y"] + HOLE_MARGIN


def make_shifted_patch(
    image: Image.Image, hole: dict, rng: random.Random
) -> tuple[Image.Image, list[int]]:
    cx, cy = hole_center_xy(hole)
    mag = rng.randint(8, 24)
    angle = rng.uniform(0, 2 * math.pi)
    dx = int(round(mag * math.cos(angle)))
    dy = int(round(mag * math.sin(angle)))
    if dx == 0 and dy == 0:
        dx = mag
    sx, sy = clamp_patch_xy(image, cx + dx, cy + dy)
    return extract_patch(image, sx, sy), [sx - cx, sy - cy]


def make_color_aug(patch: Image.Image, rng: random.Random) -> Image.Image:
    out = patch.convert("RGB")
    brightness = rng.uniform(0.75, 1.25)
    contrast = rng.uniform(0.75, 1.25)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    hue_shift = rng.randint(-18, 18)
    if hue_shift != 0:
        hsv = np.asarray(out.convert("HSV"), dtype=np.uint8).copy()
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 256
        out = Image.merge(
            "HSV",
            (
                Image.fromarray(hsv[:, :, 0]),
                Image.fromarray(hsv[:, :, 1]),
                Image.fromarray(hsv[:, :, 2]),
            ),
        ).convert("RGB")
    return out


def make_rotated_flipped(patch: Image.Image, rng: random.Random) -> Image.Image:
    ops = [
        lambda im: im.transpose(Image.Transpose.ROTATE_90),
        lambda im: im.transpose(Image.Transpose.ROTATE_180),
        lambda im: im.transpose(Image.Transpose.ROTATE_270),
        lambda im: ImageOps.mirror(im),
        lambda im: ImageOps.flip(im),
        lambda im: ImageOps.mirror(im.transpose(Image.Transpose.ROTATE_90)),
    ]
    return ops[rng.randrange(len(ops))](patch.convert("RGB"))


def iter_valid_crop_origins(
    width: int, height: int, forbidden: list[tuple[int, int, int, int]], rng: random.Random
) -> list[tuple[int, int]]:
    if width < PATCH_SIZE or height < PATCH_SIZE:
        return []
    xs = list(range(0, width - PATCH_SIZE + 1, GRID_STRIDE))
    ys = list(range(0, height - PATCH_SIZE + 1, GRID_STRIDE))
    if not xs or not ys:
        return []
    if xs[-1] != width - PATCH_SIZE:
        xs.append(width - PATCH_SIZE)
    if ys[-1] != height - PATCH_SIZE:
        ys.append(height - PATCH_SIZE)
    coords = [(x, y) for y in ys for x in xs]
    rng.shuffle(coords)
    valid = []
    for x, y in coords:
        box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
        overlaps = False
        for fx1, fy1, fx2, fy2 in forbidden:
            if not (box[2] <= fx1 or box[0] >= fx2 or box[3] <= fy1 or box[1] >= fy2):
                overlaps = True
                break
        if not overlaps:
            valid.append((x, y))
    return valid


def make_similar_color_crops(
    image: Image.Image,
    targets: list[Image.Image],
    forbidden: list[tuple[int, int, int, int]],
    n: int,
    rng: random.Random,
) -> list[Image.Image]:
    origins = iter_valid_crop_origins(image.size[0], image.size[1], forbidden, rng)
    if not origins:
        return [
            extract_patch(image, *clamp_patch_xy(image, rng.randint(0, max(0, image.size[0] - PATCH_SIZE)), rng.randint(0, max(0, image.size[1] - PATCH_SIZE))))
            for _ in range(n)
        ]
    target_stats = [color_stats(t) for t in targets]
    scored = []
    for x, y in origins[: min(len(origins), 400)]:
        patch = extract_patch(image, x, y)
        stats = color_stats(patch)
        dist = min(color_distance(stats, ts) for ts in target_stats)
        scored.append((dist, patch))
    scored.sort(key=lambda t: t[0])
    picks = [p for _, p in scored[: max(n * 3, n)]]
    rng.shuffle(picks)
    while len(picks) < n:
        x, y = origins[rng.randrange(len(origins))]
        picks.append(extract_patch(image, x, y))
    return picks[:n]


def make_blended_patches(
    image: Image.Image,
    forbidden: list[tuple[int, int, int, int]],
    n: int,
    rng: random.Random,
) -> list[Image.Image]:
    origins = iter_valid_crop_origins(image.size[0], image.size[1], forbidden, rng)
    if len(origins) < 2:
        w, h = image.size
        origins = [
            clamp_patch_xy(image, rng.randint(0, max(0, w - PATCH_SIZE)), rng.randint(0, max(0, h - PATCH_SIZE)))
            for _ in range(max(2, n * 2))
        ]
    out = []
    for _ in range(n):
        (x1, y1), (x2, y2) = rng.sample(origins, 2) if len(origins) >= 2 else (origins[0], origins[0])
        a = np.asarray(extract_patch(image, x1, y1), dtype=np.float32)
        b = np.asarray(extract_patch(image, x2, y2), dtype=np.float32)
        alpha = rng.uniform(0.3, 0.7)
        blend = np.clip(alpha * a + (1.0 - alpha) * b, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(blend))
    return out


def make_distractors(
    category_images: list[Path],
    source_path: Path,
    n: int,
    rng: random.Random,
) -> list[Image.Image]:
    others = [p for p in category_images if p != source_path]
    if not others:
        others = list(category_images)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        path = others[rng.randrange(len(others))]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w < PATCH_SIZE or h < PATCH_SIZE:
                    continue
                x = rng.randint(0, w - PATCH_SIZE)
                y = rng.randint(0, h - PATCH_SIZE)
                out.append(extract_patch(im, x, y))
        except Exception:
            continue
    while len(out) < n:
        out.append(Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (0, 0, 0)))
    return out


def punch_holes(image: Image.Image, holes: list[dict]) -> Image.Image:
    ctx = image.convert("RGB").copy()
    black = Image.new("RGB", (HOLE_SIZE, HOLE_SIZE), (0, 0, 0))
    for hole in holes:
        ctx.paste(black, (hole["x"], hole["y"]))
    return ctx


def build_candidates(
    image: Image.Image,
    holes: list[dict],
    category_images: list[Path],
    source_path: Path,
    rng: random.Random,
) -> tuple[list[dict], list[Image.Image], list[int]]:
    n = len(holes)
    correct_patches = []
    forbidden = []
    for hole in holes:
        cx, cy = hole_center_xy(hole)
        correct_patches.append(extract_patch(image, cx, cy))
        forbidden.append((cx, cy, cx + PATCH_SIZE, cy + PATCH_SIZE))

    items: list[tuple[dict, Image.Image]] = []

    for i, patch in enumerate(correct_patches):
        items.append(({"type": "correct", "for_hole": i}, patch))

    for i, hole in enumerate(holes):
        patch, shift_px = make_shifted_patch(image, hole, rng)
        items.append(
            ({"type": "shifted", "for_hole": i, "shift_px": shift_px}, patch)
        )

    for i, patch in enumerate(correct_patches):
        items.append(
            ({"type": "color_aug", "for_hole": i}, make_color_aug(patch, rng))
        )

    similar = make_similar_color_crops(image, correct_patches, forbidden, n, rng)
    for i, patch in enumerate(similar):
        items.append(({"type": "same_image", "for_hole": i}, patch))

    for i, patch in enumerate(correct_patches):
        items.append(
            ({"type": "rotated", "for_hole": i}, make_rotated_flipped(patch, rng))
        )

    blended = make_blended_patches(image, forbidden, n, rng)
    for i, patch in enumerate(blended):
        items.append(({"type": "blended", "for_hole": i}, patch))

    distractors = make_distractors(category_images, source_path, 2 * n, rng)
    for i, patch in enumerate(distractors):
        items.append(({"type": "distractor", "for_hole": i % n}, patch))

    order = list(range(len(items)))
    rng.shuffle(order)
    candidates = []
    images = []
    correct_indices = [-1] * n
    for new_idx, old_idx in enumerate(order):
        meta, patch = items[old_idx]
        entry = {"idx": new_idx, "type": meta["type"], "for_hole": meta["for_hole"]}
        if "shift_px" in meta:
            entry["shift_px"] = meta["shift_px"]
        candidates.append(entry)
        images.append(patch)
        if meta["type"] == "correct":
            correct_indices[meta["for_hole"]] = new_idx

    return candidates, images, correct_indices


def process_one(args: tuple) -> dict | None:
    (
        sample_id,
        image_path,
        category,
        split,
        category_images,
        out_dir,
        dry_run,
        seed,
    ) = args
    rng = random.Random(seed)
    try:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            holes = select_entropy_holes(image, rng, MAX_HOLES)
            if not holes:
                return None

            candidates, cand_images, correct_indices = build_candidates(
                image, holes, category_images, Path(image_path), rng
            )
            context = punch_holes(image, holes)

            hole_records = []
            for i, hole in enumerate(holes):
                hole_records.append(
                    {
                        "hole_idx": i,
                        "x": hole["x"],
                        "y": hole["y"],
                        "entropy": round(float(hole["entropy"]), 6),
                        "correct_candidate_idx": correct_indices[i],
                    }
                )

            candidate_records = []
            for entry, patch in zip(candidates, cand_images):
                rel = f"patches/{sample_id}_c{entry['idx']:02d}.png"
                rec = dict(entry)
                rec["path"] = rel
                candidate_records.append(rec)
                if not dry_run:
                    patch_path = Path(out_dir) / rel
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    patch.save(patch_path)

            if not dry_run:
                ctx_path = Path(out_dir) / "contexts" / f"{sample_id}.png"
                ctx_path.parent.mkdir(parents=True, exist_ok=True)
                context.save(ctx_path)

            return {
                "sample_id": sample_id,
                "source_image": Path(image_path).name,
                "category": category,
                "split": split,
                "image_size": [width, height],
                "num_holes": len(holes),
                "holes": hole_records,
                "candidates": candidate_records,
                "correct_indices": correct_indices,
            }
    except Exception as exc:
        return {
            "_error": True,
            "path": str(image_path),
            "error": str(exc),
        }


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    by_cat: dict[str, list[Path]] = {}
    for cat_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in cat_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if images:
            by_cat[cat_dir.name] = images
    return by_cat


def assign_splits(
    by_cat: dict[str, list[Path]], seed: int
) -> list[tuple[Path, str, str]]:
    rng = random.Random(seed)
    assignments: list[tuple[Path, str, str]] = []
    for category, images in sorted(by_cat.items()):
        imgs = list(images)
        if category in TEST_CATEGORIES:
            for path in imgs:
                assignments.append((path, category, "test"))
            continue
        rng.shuffle(imgs)
        n_val = max(1, int(round(len(imgs) * VAL_FRACTION))) if len(imgs) > 1 else 0
        val_set = set(imgs[:n_val])
        for path in imgs:
            split = "val" if path in val_set else "train"
            assignments.append((path, category, split))
    split_order = {"train": 0, "val": 1, "test": 2}
    assignments.sort(key=lambda t: (split_order[t[2]], t[1], t[0].name))
    return assignments


def summarize(samples: list[dict]) -> None:
    split_counts = Counter(s["split"] for s in samples)
    cat_split = defaultdict(Counter)
    type_counts = Counter()
    entropy_by_split: dict[str, list[float]] = defaultdict(list)
    hole_counts = Counter()

    for s in samples:
        cat_split[s["split"]][s["category"]] += 1
        hole_counts[s["num_holes"]] += 1
        for hole in s["holes"]:
            entropy_by_split[s["split"]].append(float(hole["entropy"]))
        for c in s["candidates"]:
            type_counts[c["type"]] += 1

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(samples)}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {split_counts.get(split, 0)} images")
        for cat, n in sorted(cat_split[split].items()):
            print(f"    {cat}: {n}")

    print("\nHoles per image:")
    for k in sorted(hole_counts):
        print(f"  {k} hole(s): {hole_counts[k]} images")

    print("\nEntropy by split (min / mean / max):")
    for split in ("train", "val", "test"):
        vals = entropy_by_split.get(split, [])
        if not vals:
            print(f"  {split}: n/a")
            continue
        arr = np.array(vals, dtype=np.float64)
        print(
            f"  {split}: {arr.min():.4f} / {arr.mean():.4f} / {arr.max():.4f} "
            f"(n={len(arr)})"
        )

    print("\nCandidates by type:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")

    expected = sum(s["num_holes"] * CANDIDATES_PER_HOLE for s in samples)
    actual = sum(len(s["candidates"]) for s in samples)
    print(f"\nTotal candidates: {actual} (expected {expected})")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build pattern completion benchmark from JONES-19 ornaments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "raw" / "JONES-19",
        help="Path to JONES-19 root with culture folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "processed",
        help="Output directory for contexts/, patches/, and jsonl splits",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process 5 images, save preview under processed/_dry_run/, print stats",
    )
    return parser.parse_args()


def save_dry_run_preview(samples: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 1 + MAX_HOLES, figsize=(3.2 * (1 + MAX_HOLES), 3.0 * n))
    if n == 1:
        axes = np.array([axes])
    for row, sample in enumerate(samples):
        ctx = Image.open(out_dir / "contexts" / f"{sample['sample_id']}.png")
        ax = axes[row, 0]
        ax.imshow(ctx)
        ax.set_title(
            f"{sample['sample_id']} | {sample['category']} | {sample['split']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for col in range(MAX_HOLES):
            ax = axes[row, col + 1]
            if col < sample["num_holes"]:
                cidx = sample["correct_indices"][col]
                patch = Image.open(out_dir / sample["candidates"][cidx]["path"])
                ax.imshow(patch)
                ent = sample["holes"][col]["entropy"]
                ax.set_title(f"hole {col} H={ent:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Dry-run preview: context (left) + correct 64x64 patches", fontsize=12)
    fig.tight_layout()
    preview_path = out_dir / "preview.png"
    fig.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return preview_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = args.input
    out_dir = args.output
    if not raw_dir.is_dir():
        raise SystemExit(f"Input directory not found: {raw_dir}")

    by_cat = collect_images(raw_dir)
    if not by_cat:
        raise SystemExit(f"No images found under {raw_dir}")

    assignments = assign_splits(by_cat, args.seed)
    if args.dry_run:
        out_dir = out_dir / "_dry_run"
        dry = []
        by_split: dict[str, list] = {"train": [], "val": [], "test": []}
        for item in assignments:
            by_split[item[2]].append(item)
        for split in ("train", "val", "test"):
            if by_split[split]:
                dry.append(by_split[split][0])
        for item in assignments:
            if len(dry) >= 5:
                break
            if item not in dry:
                dry.append(item)
        assignments = dry[:5]

    (out_dir / "contexts").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)

    category_image_map = {cat: [str(p) for p in paths] for cat, paths in by_cat.items()}

    tasks = []
    for i, (path, category, split) in enumerate(assignments, start=1):
        sample_id = f"{i:06d}"
        sample_seed = args.seed + i * 10007
        tasks.append(
            (
                sample_id,
                str(path),
                category,
                split,
                category_image_map[category],
                str(out_dir),
                False,
                sample_seed,
            )
        )

    workers = 1 if args.dry_run else max(1, min(args.workers, MAX_WORKERS))
    samples: list[dict] = []
    errors = []

    if workers == 1:
        iterator = (process_one(t) for t in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Processing"):
            if result is None:
                continue
            if result.get("_error"):
                errors.append(result)
                continue
            samples.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = fut.result()
                if result is None:
                    continue
                if result.get("_error"):
                    errors.append(result)
                    continue
                samples.append(result)

    samples.sort(key=lambda s: s["sample_id"])

    split_files = {
        "train": out_dir / "train.jsonl",
        "val": out_dir / "val.jsonl",
        "test": out_dir / "test.jsonl",
    }
    handles = {k: open(v, "w", encoding="utf-8") for k, v in split_files.items()}
    try:
        for sample in samples:
            line = json.dumps(sample, ensure_ascii=False)
            handles[sample["split"]].write(line + "\n")
    finally:
        for fh in handles.values():
            fh.close()

    summarize(samples)
    if errors:
        print(f"\nSkipped {len(errors)} images due to errors:")
        for err in errors[:10]:
            print(f"  {err['path']}: {err['error']}")

    print(f"\nWrote dataset to {out_dir}")
    if args.dry_run:
        preview = save_dry_run_preview(samples, out_dir)
        print(f"Preview grid: {preview}")
        print("Open contexts/ to browse the punched images.")


if __name__ == "__main__":
    main()

#entropy based hole and candidate selection
#for every image we create 4 holes and 32 candidates 
#4 correct+ 20 hard negatives(augmentations of correct patches)+8 distractors 
#python3 Data/prepare_dataset.py --dry-run to run a demo and obtain 5 samples
#python3 Data/prepare_dataset.py full build
#script is non stochastic with random seed 42

