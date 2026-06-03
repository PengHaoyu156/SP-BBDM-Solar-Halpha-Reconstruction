#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directly print PSNR/SSIM/LPIPS for paired images (folder or single file).

- If given two folders, it matches by filename and prints mean ± std and median.
- If given two files, it prints the metrics for that single pair.
- std means per-image standard deviation across the test set.
- LPIPS is computed using the lpips package, with images normalized to [-1, 1].

Usage:
  python3 tools/print_psnr_ssim_lpips_mean_std.py \
      --pred <pred_folder> \
      --gt <gt_folder> \
      --per_image \
      --csv metrics.csv

Example:
  python3 tools/print_psnr_ssim_lpips_mean_std.py \
      --pred results/spbbdm/test/output \
      --gt results/spbbdm/test/ground_truth \
      --device cuda \
      --csv spbbdm_metrics.csv
"""

import os
import cv2
import math
import argparse
import numpy as np
from typing import Tuple, List, Dict

ALLOWED = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def is_file(p):
    return os.path.isfile(p)


def is_dir(p):
    return os.path.isdir(p)


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center crop two images to their common minimum height and width."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x):
        dh = (x.shape[0] - h) // 2
        dw = (x.shape[1] - w) // 2
        return x[dh:dh + h, dw:dw + w]

    return crop(a), crop(b)


def crop_border(x: np.ndarray, border: int) -> np.ndarray:
    """Crop image border if border > 0."""
    if border <= 0:
        return x

    h, w = x.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(
            f"Border {border} is too large for image size {h}x{w}."
        )

    return x[border:h - border, border:w - border]


def to_u8(x: np.ndarray) -> np.ndarray:
    """Convert image to uint8 in [0, 255]."""
    return x if x.dtype == np.uint8 else np.clip(x, 0, 255).astype(np.uint8)


def psnr(x: np.ndarray, y: np.ndarray, border: int = 0) -> float:
    """Compute PSNR with MAX = 255."""
    if x.shape != y.shape:
        raise ValueError("PSNR: shape mismatch after cropping.")

    x = crop_border(x, border)
    y = crop_border(y, border)

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return float('inf')

    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    """Compute SSIM for single-channel images, following the standard Gaussian-window form."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.T)

    mux = cv2.filter2D(x, -1, win)[5:-5, 5:-5]
    muy = cv2.filter2D(y, -1, win)[5:-5, 5:-5]

    mux2 = mux * mux
    muy2 = muy * muy
    muxy = mux * muy

    sigx2 = cv2.filter2D(x * x, -1, win)[5:-5, 5:-5] - mux2
    sigy2 = cv2.filter2D(y * y, -1, win)[5:-5, 5:-5] - muy2
    sigxy = cv2.filter2D(x * y, -1, win)[5:-5, 5:-5] - muxy

    s = ((2 * muxy + C1) * (2 * sigxy + C2)) / (
        (mux2 + muy2 + C1) * (sigx2 + sigy2 + C2)
    )

    return float(s.mean())


def ssim(x: np.ndarray, y: np.ndarray, border: int = 0) -> float:
    """Compute SSIM. For RGB/BGR images, average SSIM over three channels."""
    if x.shape != y.shape:
        raise ValueError("SSIM: shape mismatch after cropping.")

    x = crop_border(x, border)
    y = crop_border(y, border)

    if x.ndim == 2:
        return ssim_gray(x, y)

    if x.ndim == 3:
        if x.shape[2] == 3:
            return float(np.mean([
                ssim_gray(x[:, :, c], y[:, :, c]) for c in range(3)
            ]))

        if x.shape[2] == 1:
            return ssim_gray(x[:, :, 0], y[:, :, 0])

        return ssim_gray(
            cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        )

    raise ValueError("SSIM: wrong ndim.")


def list_imgs(d: str) -> List[str]:
    """List image files in a directory."""
    fs = [
        f for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(ALLOWED)
    ]
    fs.sort()
    return fs


def read_pair(p_path: str, g_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read predicted and ground-truth images with OpenCV."""
    p = cv2.imread(p_path, cv2.IMREAD_COLOR)
    g = cv2.imread(g_path, cv2.IMREAD_COLOR)

    if p is None or g is None:
        raise FileNotFoundError(f"Read failed: {p_path} or {g_path}")

    if p.shape != g.shape:
        p, g = center_crop_to_common(p, g)

    p = to_u8(p)
    g = to_u8(g)

    return p, g


class LPIPSEvaluator:
    """LPIPS evaluator. Images are converted from BGR uint8 to RGB tensor in [-1, 1]."""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        import torch
        import lpips

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()

    def image_to_tensor(self, img_bgr: np.ndarray):
        """Convert BGR uint8 image to torch tensor with shape [1, 3, H, W] in [-1, 1]."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = np.transpose(img, (2, 0, 1))
        tensor = self.torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor

    def __call__(self, pred_bgr: np.ndarray, gt_bgr: np.ndarray, border: int = 0) -> float:
        pred_bgr = crop_border(pred_bgr, border)
        gt_bgr = crop_border(gt_bgr, border)

        with self.torch.no_grad():
            pred_tensor = self.image_to_tensor(pred_bgr)
            gt_tensor = self.image_to_tensor(gt_bgr)
            value = self.loss_fn(pred_tensor, gt_tensor)

        return float(value.item())


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def safe_median(values: List[float]) -> float:
    return float(np.median(values)) if len(values) > 0 else float("nan")


def safe_std(values: List[float]) -> float:
    """
    Sample standard deviation.
    If there is only one image, std is set to 0.0.
    """
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def format_mean_std(values: List[float], digits: int = 4) -> str:
    mean = safe_mean(values)
    std = safe_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def eval_pair(
    p_path: str,
    g_path: str,
    border: int = 0,
    lpips_eval: LPIPSEvaluator = None
) -> Dict[str, float]:
    """Evaluate one image pair."""
    p, g = read_pair(p_path, g_path)

    result = {
        "psnr": psnr(p, g, border),
        "ssim": ssim(p, g, border),
    }

    if lpips_eval is not None:
        result["lpips"] = lpips_eval(p, g, border)

    return result


def save_csv(csv_path: str, rows: List[Dict[str, float]]):
    """Save per-image metrics to CSV."""
    import csv

    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--pred', required=True, help='Predicted image file or folder.')
    ap.add_argument('--gt', required=True, help='Ground-truth image file or folder.')
    ap.add_argument('--border', type=int, default=0, help='Crop border pixels before metric calculation.')
    ap.add_argument('--per_image', action='store_true', help='Print per-image metrics when evaluating folders.')
    ap.add_argument('--csv', type=str, default=None, help='Save per-image metrics to CSV.')

    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for LPIPS.')
    ap.add_argument('--lpips_net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'], help='LPIPS backbone.')
    ap.add_argument('--no_lpips', action='store_true', help='Disable LPIPS calculation.')

    args = ap.parse_args()

    lpips_eval = None
    if not args.no_lpips:
        print(f"Loading LPIPS model: net={args.lpips_net}, device={args.device}")
        lpips_eval = LPIPSEvaluator(net=args.lpips_net, device=args.device)

    # single file vs file
    if is_file(args.pred) and is_file(args.gt):
        metrics = eval_pair(args.pred, args.gt, args.border, lpips_eval)

        print("Single image pair metrics:")
        print(f"PSNR : {metrics['psnr']:.4f}")
        print(f"SSIM : {metrics['ssim']:.4f}")

        if "lpips" in metrics:
            print(f"LPIPS: {metrics['lpips']:.4f}")

        return

    # folder vs folder
    if not (is_dir(args.pred) and is_dir(args.gt)):
        raise ValueError("Both --pred and --gt must be directories, or both must be files.")

    pred_files = list_imgs(args.pred)
    gt_set = set(list_imgs(args.gt))

    ps_list = []
    ss_list = []
    lp_list = []
    rows = []

    matched = 0
    missing_in_gt = 0

    for fn in pred_files:
        if fn not in gt_set:
            missing_in_gt += 1
            continue

        p_path = os.path.join(args.pred, fn)
        g_path = os.path.join(args.gt, fn)

        metrics = eval_pair(p_path, g_path, args.border, lpips_eval)

        PS = metrics["psnr"]
        SS = metrics["ssim"]

        ps_list.append(PS)
        ss_list.append(SS)

        row = {
            "filename": fn,
            "psnr": PS,
            "ssim": SS,
        }

        if "lpips" in metrics:
            LP = metrics["lpips"]
            lp_list.append(LP)
            row["lpips"] = LP

        rows.append(row)

        matched += 1

        if args.per_image:
            if "lpips" in metrics:
                print(f"{fn}, PSNR={PS:.4f}, SSIM={SS:.4f}, LPIPS={metrics['lpips']:.4f}")
            else:
                print(f"{fn}, PSNR={PS:.4f}, SSIM={SS:.4f}")

    if matched == 0:
        print("No matched image pairs.")
        return

    if args.csv is not None:
        save_csv(args.csv, rows)
        print(f"Per-image metrics saved to: {args.csv}")

    print("")
    print("========== Evaluation Summary ==========")
    print(f"COUNT: {matched}")

    if missing_in_gt > 0:
        print(f"WARNING: {missing_in_gt} predicted files were skipped because no same-name GT file was found.")

    print("")
    print("Mean ± Std across test images:")
    print(f"PSNR : {format_mean_std(ps_list, digits=4)}")
    print(f"SSIM : {format_mean_std(ss_list, digits=4)}")

    if len(lp_list) > 0:
        print(f"LPIPS: {format_mean_std(lp_list, digits=4)}")

    print("")
    print("Median:")
    print(f"PSNR : {safe_median(ps_list):.4f}")
    print(f"SSIM : {safe_median(ss_list):.4f}")

    if len(lp_list) > 0:
        print(f"LPIPS: {safe_median(lp_list):.4f}")

    print("========================================")


if __name__ == '__main__':
    main()