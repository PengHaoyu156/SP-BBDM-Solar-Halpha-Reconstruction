#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ROI patches for bright active-region-like areas and filament-like dark areas.

ROI locations are detected from the ground-truth H-alpha images only.
The same ROI coordinates are then applied to the fake/generated images.

Output:
  out_dir/
    bright/real/
    bright/fake/
    bright/overlay/        optional
    filament/real/
    filament/fake/
    filament/overlay/      optional
    roi_records.csv

Example:
python extract_bright_filament_rois.py \
  --real_dir /path/to/real_B \
  --fake_dir /path/to/fake_B \
  --out_dir /path/to/roi_eval \
  --roi_size 48 \
  --max_per_type 50 \
  --match_mode name \
  --save_overlays \
  --overwrite
"""

import csv
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name
    )


def build_pairs(real_dir: Path, fake_dir: Path, match_mode: str) -> List[Tuple[Path, Path, str]]:
    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)

    if not real_files:
        raise RuntimeError(f"No images found in real_dir: {real_dir}")
    if not fake_files:
        raise RuntimeError(f"No images found in fake_dir: {fake_dir}")

    if match_mode == "name":
        fake_by_stem = {p.stem: p for p in fake_files}
        pairs = []
        missing = []
        for r in real_files:
            f = fake_by_stem.get(r.stem)
            if f is None:
                missing.append(r.name)
                continue
            pairs.append((r, f, r.stem))

        if not pairs:
            raise RuntimeError("No pairs matched by filename stem. Try --match_mode sorted.")
        if missing:
            print(f"[Warning] {len(missing)} real images have no fake match. First few: {missing[:5]}")
        return pairs

    if match_mode == "sorted":
        n = min(len(real_files), len(fake_files))
        if len(real_files) != len(fake_files):
            print(f"[Warning] real count={len(real_files)}, fake count={len(fake_files)}; pairing first {n}.")
        return [(r, f, f"{i:06d}") for i, (r, f) in enumerate(zip(real_files[:n], fake_files[:n]))]

    raise ValueError(f"Unknown match_mode: {match_mode}")


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def gray01(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if g.max() > 1.5:
        g /= 255.0
    return np.clip(g, 0.0, 1.0)


def disk_mask(h: int, w: int, radius_ratio: float = 0.47, inner_ratio: float = 0.92) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = min(h, w) * radius_ratio * inner_ratio
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def sobel_mag(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def normalize_inside(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    if hi <= lo + 1e-8:
        return out
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def q_inside(x: np.ndarray, mask: np.ndarray, q: float) -> float:
    vals = x[mask]
    if vals.size == 0:
        return 0.0
    return float(np.quantile(vals, q))


def best_component(mask: np.ndarray, score: np.ndarray, min_area: int) -> Optional[Tuple[int, int, float, int]]:
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best = None
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == lab
        mean_score = float(score[comp].mean())
        max_score = float(score[comp].max())
        cx, cy = cents[lab]
        if best is None or mean_score > best[0]:
            best = (mean_score, int(round(cx)), int(round(cy)), max_score, area)
    if best is None:
        return None
    _, cx, cy, max_score, area = best
    return cx, cy, max_score, area


def detect_bright(g: np.ndarray, mask: np.ndarray, bright_q: float, grad_q: float, min_area: int):
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    grad = sobel_mag(blur)

    t_bright = q_inside(blur, mask, bright_q)
    t_grad = q_inside(grad, mask, grad_q)

    cand = (blur >= t_bright) & mask & ((grad >= t_grad) | (blur >= q_inside(blur, mask, min(0.995, bright_q + 0.03))))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)

    score = normalize_inside(blur, mask) + 0.5 * normalize_inside(grad, mask)
    return best_component(cand, score, min_area)


def detect_filament(g: np.ndarray, mask: np.ndarray, dark_q: float, contrast_q: float, min_area: int):
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    local_mean = cv2.GaussianBlur(blur, (0, 0), sigmaX=7, sigmaY=7)
    dark_contrast = np.maximum(local_mean - blur, 0.0)

    t_dark = q_inside(blur, mask, dark_q)
    t_contrast = q_inside(dark_contrast, mask, contrast_q)

    cand = (blur <= t_dark) & (dark_contrast >= t_contrast) & mask
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)

    score = normalize_inside(dark_contrast, mask)
    return best_component(cand, score, min_area)


def clamp_center(cx: int, cy: int, h: int, w: int, size: int) -> Tuple[int, int]:
    half = size // 2
    cx = int(np.clip(cx, half, w - (size - half)))
    cy = int(np.clip(cy, half, h - (size - half)))
    return cx, cy


def center_to_box(cx: int, cy: int, h: int, w: int, size: int) -> Tuple[int, int, int, int]:
    cx, cy = clamp_center(cx, cy, h, w, size)
    half = size // 2
    x1, y1 = cx - half, cy - half
    return x1, y1, x1 + size, y1 + size


def crop_by_box(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].copy()


def draw_overlay(rgb: np.ndarray, box: Tuple[int, int, int, int], roi_type: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box
    color = (0, 255, 255) if roi_type == "bright" else (255, 0, 255)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(bgr, roi_type, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_roi(out_dir: Path, roi_type: str, idx: int, stem: str, real_rgb: np.ndarray, fake_rgb: np.ndarray,
             box: Tuple[int, int, int, int], save_overlay: bool, records: list, score: float, area: int,
             real_name: str, fake_name: str):
    out_name = f"{idx:04d}_{stem}.png"

    real_patch = crop_by_box(real_rgb, box)
    fake_patch = crop_by_box(fake_rgb, box)

    save_rgb(out_dir / roi_type / "real" / out_name, real_patch)
    save_rgb(out_dir / roi_type / "fake" / out_name, fake_patch)

    if save_overlay:
        save_rgb(out_dir / roi_type / "overlay" / out_name, draw_overlay(real_rgb, box, roi_type))

    x1, y1, x2, y2 = box
    records.append({
        "roi_type": roi_type,
        "out_name": out_name,
        "real_filename": real_name,
        "fake_filename": fake_name,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
        "score": score,
        "area": area,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="Ground-truth H-alpha image folder.")
    ap.add_argument("--fake_dir", required=True, help="Generated/fake H-alpha image folder.")
    ap.add_argument("--out_dir", required=True, help="Output ROI folder.")
    ap.add_argument("--roi_size", type=int, default=48)
    ap.add_argument("--max_per_type", type=int, default=50)
    ap.add_argument("--match_mode", choices=["name", "sorted"], default="name")

    ap.add_argument("--disk_radius_ratio", type=float, default=0.47)
    ap.add_argument("--inner_disk_ratio", type=float, default=0.92)

    ap.add_argument("--bright_quantile", type=float, default=0.95)
    ap.add_argument("--bright_grad_quantile", type=float, default=0.70)

    ap.add_argument("--dark_quantile", type=float, default=0.18)
    ap.add_argument("--dark_contrast_quantile", type=float, default=0.85)

    ap.add_argument("--min_area", type=int, default=8)
    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    out_dir = Path(args.out_dir)

    if not real_dir.is_dir():
        raise NotADirectoryError(f"real_dir is not valid: {real_dir}")
    if not fake_dir.is_dir():
        raise NotADirectoryError(f"fake_dir is not valid: {fake_dir}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for roi_type in ["bright", "filament"]:
        (out_dir / roi_type / "real").mkdir(parents=True, exist_ok=True)
        (out_dir / roi_type / "fake").mkdir(parents=True, exist_ok=True)
        if args.save_overlays:
            (out_dir / roi_type / "overlay").mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(real_dir, fake_dir, args.match_mode)

    print("========== ROI extraction ==========")
    print(f"real_dir     : {real_dir}")
    print(f"fake_dir     : {fake_dir}")
    print(f"out_dir      : {out_dir}")
    print(f"pairs        : {len(pairs)}")
    print(f"roi_size     : {args.roi_size}")
    print(f"max_per_type : {args.max_per_type}")
    print("Detection source: ground-truth H-alpha")
    print("====================================")

    counts = {"bright": 0, "filament": 0}
    records = []

    for real_path, fake_path, stem in pairs:
        if counts["bright"] >= args.max_per_type and counts["filament"] >= args.max_per_type:
            break

        real_rgb = read_rgb(real_path)
        fake_rgb = read_rgb(fake_path)

        if real_rgb.shape != fake_rgb.shape:
            h = min(real_rgb.shape[0], fake_rgb.shape[0])
            w = min(real_rgb.shape[1], fake_rgb.shape[1])
            real_rgb = real_rgb[:h, :w]
            fake_rgb = fake_rgb[:h, :w]

        h, w = real_rgb.shape[:2]
        g = gray01(real_rgb)
        m = disk_mask(h, w, args.disk_radius_ratio, args.inner_disk_ratio)

        if counts["bright"] < args.max_per_type:
            det = detect_bright(g, m, args.bright_quantile, args.bright_grad_quantile, args.min_area)
            if det is not None:
                cx, cy, score, area = det
                box = center_to_box(cx, cy, h, w, args.roi_size)
                save_roi(out_dir, "bright", counts["bright"], stem, real_rgb, fake_rgb, box,
                         args.save_overlays, records, score, area, real_path.name, fake_path.name)
                counts["bright"] += 1

        if counts["filament"] < args.max_per_type:
            det = detect_filament(g, m, args.dark_quantile, args.dark_contrast_quantile, args.min_area)
            if det is not None:
                cx, cy, score, area = det
                box = center_to_box(cx, cy, h, w, args.roi_size)
                save_roi(out_dir, "filament", counts["filament"], stem, real_rgb, fake_rgb, box,
                         args.save_overlays, records, score, area, real_path.name, fake_path.name)
                counts["filament"] += 1

    csv_path = out_dir / "roi_records.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["roi_type", "out_name", "real_filename", "fake_filename",
                  "x1", "y1", "x2", "y2", "cx", "cy", "score", "area"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print("========== Finished ==========")
    print(f"bright ROIs   : {counts['bright']}")
    print(f"filament ROIs : {counts['filament']}")
    print(f"CSV saved     : {csv_path}")
    print("")
    print("Metric commands:")
    print(f"python print_psnr_ssim_lpips_mean_std.py --pred {out_dir/'bright'/'fake'} --gt {out_dir/'bright'/'real'} --device cuda --csv bright_roi_metrics.csv")
    print(f"python print_psnr_ssim_lpips_mean_std.py --pred {out_dir/'filament'/'fake'} --gt {out_dir/'filament'/'real'} --device cuda --csv filament_roi_metrics.csv")


if __name__ == "__main__":
    main()
