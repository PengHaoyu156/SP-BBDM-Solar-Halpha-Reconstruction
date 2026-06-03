#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ROI-specific metrics for bright-region ROIs and filament ROIs.

Bright ROI metrics:
  1. MAE
  2. Grad-MAE
  3. Bright-region intensity error

Filament ROI metrics:
  1. MAE
  2. Grad-MAE
  3. Filament contrast error

All metrics are computed on grayscale images normalized to [0, 1].
"""

import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name
    )


def build_pairs(real_dir: Path, fake_dir: Path, match_mode: str) -> List[Tuple[Path, Path, str]]:
    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)

    if len(real_files) == 0:
        raise RuntimeError(f"No images found in real_dir: {real_dir}")
    if len(fake_files) == 0:
        raise RuntimeError(f"No images found in fake_dir: {fake_dir}")

    if match_mode == "name":
        fake_by_name = {p.name: p for p in fake_files}
        fake_by_stem = {p.stem: p for p in fake_files}
        pairs = []
        missing = []

        for r in real_files:
            f = fake_by_name.get(r.name)
            if f is None:
                f = fake_by_stem.get(r.stem)
            if f is None:
                missing.append(r.name)
                continue
            pairs.append((r, f, r.name))

        if len(pairs) == 0:
            raise RuntimeError("No pairs matched by name/stem. Try --match_mode sorted.")
        if missing:
            print(f"[Warning] {len(missing)} real files had no fake match. First few: {missing[:5]}")
        return pairs

    if match_mode == "sorted":
        n = min(len(real_files), len(fake_files))
        if len(real_files) != len(fake_files):
            print(f"[Warning] real count={len(real_files)}, fake count={len(fake_files)}. Pairing first {n}.")
        return [(r, f, f"{i:06d}.png") for i, (r, f) in enumerate(zip(real_files[:n], fake_files[:n]))]

    raise ValueError(f"Unsupported match_mode: {match_mode}")


def read_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0)


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x):
        y0 = (x.shape[0] - h) // 2
        x0 = (x.shape[1] - w) // 2
        return x[y0:y0 + h, x0:x0 + w]

    return crop(a), crop(b)


def mae(fake: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(fake - real)))


def grad_mae(fake: np.ndarray, real: np.ndarray) -> float:
    dx_f = fake[:, 1:] - fake[:, :-1]
    dx_r = real[:, 1:] - real[:, :-1]
    dy_f = fake[1:, :] - fake[:-1, :]
    dy_r = real[1:, :] - real[:-1, :]
    return float(np.mean(np.abs(dx_f - dx_r)) + np.mean(np.abs(dy_f - dy_r)))


def bright_intensity_error(fake: np.ndarray, real: np.ndarray, bright_ratio: float):
    q = 1.0 - bright_ratio
    thr = float(np.quantile(real, q))
    mask = real >= thr

    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0

    real_mean = float(real[mask].mean())
    fake_mean = float(fake[mask].mean())
    err = abs(fake_mean - real_mean)
    return err, real_mean, fake_mean, int(mask.sum())


def filament_contrast_error(fake: np.ndarray, real: np.ndarray, dark_ratio: float, bg_quantile: float):
    dark_thr = float(np.quantile(real, dark_ratio))
    bg_thr = float(np.quantile(real, bg_quantile))

    dark_mask = real <= dark_thr
    bg_mask = real >= bg_thr

    if dark_mask.sum() == 0 or bg_mask.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), int(dark_mask.sum()), int(bg_mask.sum())

    real_dark = float(real[dark_mask].mean())
    real_bg = float(real[bg_mask].mean())
    fake_dark = float(fake[dark_mask].mean())
    fake_bg = float(fake[bg_mask].mean())

    c_real = real_bg - real_dark
    c_fake = fake_bg - fake_dark
    err = abs(c_fake - c_real)

    return err, c_real, c_fake, real_dark, fake_dark, int(dark_mask.sum()), int(bg_mask.sum())


def safe_values(xs):
    return np.array([x for x in xs if np.isfinite(x)], dtype=np.float64)


def safe_mean(xs):
    vals = safe_values(xs)
    return float(vals.mean()) if vals.size else float("nan")


def safe_median(xs):
    vals = safe_values(xs)
    return float(np.median(vals)) if vals.size else float("nan")


def safe_std(xs):
    vals = safe_values(xs)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=1))


def summarize(rows: List[Dict[str, float]], roi_type: str, metric_keys: List[str]) -> List[str]:
    sub = [r for r in rows if r["roi_type"] == roi_type]
    lines = []
    lines.append(f"========== {roi_type} ROI summary ==========")
    lines.append(f"COUNT: {len(sub)}")

    for key in metric_keys:
        vals = [float(r[key]) for r in sub if key in r]
        lines.append(
            f"{key}: mean={safe_mean(vals):.6f}, std={safe_std(vals):.6f}, median={safe_median(vals):.6f}"
        )

    return lines


def evaluate_bright_pairs(pairs: List[Tuple[Path, Path, str]], bright_ratio: float) -> List[Dict[str, float]]:
    rows = []
    for real_path, fake_path, name in pairs:
        real = read_gray01(real_path)
        fake = read_gray01(fake_path)

        if real.shape != fake.shape:
            real, fake = center_crop_to_common(real, fake)

        b_err, real_b_mean, fake_b_mean, n_bright = bright_intensity_error(fake, real, bright_ratio)

        rows.append({
            "roi_type": "bright",
            "filename": name,
            "mae": mae(fake, real),
            "grad_mae": grad_mae(fake, real),
            "bright_intensity_error": b_err,
            "real_bright_mean": real_b_mean,
            "fake_bright_mean": fake_b_mean,
            "n_bright_pixels": n_bright,
        })
    return rows


def evaluate_filament_pairs(pairs: List[Tuple[Path, Path, str]], dark_ratio: float, bg_quantile: float) -> List[Dict[str, float]]:
    rows = []
    for real_path, fake_path, name in pairs:
        real = read_gray01(real_path)
        fake = read_gray01(fake_path)

        if real.shape != fake.shape:
            real, fake = center_crop_to_common(real, fake)

        f_err, c_real, c_fake, real_dark, fake_dark, n_dark, n_bg = filament_contrast_error(
            fake, real, dark_ratio=dark_ratio, bg_quantile=bg_quantile
        )

        rows.append({
            "roi_type": "filament",
            "filename": name,
            "mae": mae(fake, real),
            "grad_mae": grad_mae(fake, real),
            "filament_contrast_error": f_err,
            "real_filament_contrast": c_real,
            "fake_filament_contrast": c_fake,
            "real_dark_mean": real_dark,
            "fake_dark_mean": fake_dark,
            "n_dark_pixels": n_dark,
            "n_bg_pixels": n_bg,
        })
    return rows


def save_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    if not rows:
        return

    preferred = [
        "roi_type", "filename",
        "mae", "grad_mae",
        "bright_intensity_error",
        "filament_contrast_error",
        "real_bright_mean", "fake_bright_mean",
        "real_filament_contrast", "fake_filament_contrast",
        "real_dark_mean", "fake_dark_mean",
        "n_bright_pixels", "n_dark_pixels", "n_bg_pixels",
    ]
    keys = [k for k in preferred if any(k in r for r in rows)]
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bright_real", type=str, default=None, help="Folder of real bright ROI patches.")
    parser.add_argument("--bright_fake", type=str, default=None, help="Folder of fake bright ROI patches.")
    parser.add_argument("--filament_real", type=str, default=None, help="Folder of real filament ROI patches.")
    parser.add_argument("--filament_fake", type=str, default=None, help="Folder of fake filament ROI patches.")
    parser.add_argument("--match_mode", choices=["name", "sorted"], default="name")

    parser.add_argument("--bright_ratio", type=float, default=0.20,
                        help="Top bright_ratio pixels in real bright ROI are used for bright intensity error.")
    parser.add_argument("--dark_ratio", type=float, default=0.20,
                        help="Bottom dark_ratio pixels in real filament ROI are used as filament pixels.")
    parser.add_argument("--bg_quantile", type=float, default=0.50,
                        help="Pixels above this real ROI quantile are used as filament local background.")

    parser.add_argument("--out_csv", type=str, default="roi_specific_metrics.csv")
    parser.add_argument("--summary_txt", type=str, default="roi_specific_metrics_summary.txt")
    args = parser.parse_args()

    rows = []
    summary_lines = []

    if args.bright_real and args.bright_fake:
        bright_pairs = build_pairs(Path(args.bright_real), Path(args.bright_fake), args.match_mode)
        bright_rows = evaluate_bright_pairs(bright_pairs, bright_ratio=args.bright_ratio)
        rows.extend(bright_rows)
        summary_lines.extend(summarize(
            bright_rows,
            "bright",
            ["mae", "grad_mae", "bright_intensity_error", "real_bright_mean", "fake_bright_mean"]
        ))
        summary_lines.append("")

    if args.filament_real and args.filament_fake:
        filament_pairs = build_pairs(Path(args.filament_real), Path(args.filament_fake), args.match_mode)
        filament_rows = evaluate_filament_pairs(filament_pairs, dark_ratio=args.dark_ratio, bg_quantile=args.bg_quantile)
        rows.extend(filament_rows)
        summary_lines.extend(summarize(
            filament_rows,
            "filament",
            ["mae", "grad_mae", "filament_contrast_error", "real_filament_contrast", "fake_filament_contrast"]
        ))
        summary_lines.append("")

    if not rows:
        raise RuntimeError("No ROI folders were provided. Please provide bright and/or filament ROI folders.")

    out_csv = Path(args.out_csv)
    summary_txt = Path(args.summary_txt)

    save_csv(rows, out_csv)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Per-ROI CSV saved to: {out_csv}")
    print(f"Summary saved to: {summary_txt}")


if __name__ == "__main__":
    main()
