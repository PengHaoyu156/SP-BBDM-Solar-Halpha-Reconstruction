import os
from PIL import Image, ImageOps
import torch
import numpy as np
import torchvision.transforms as transforms


def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    paths = []
    for name in flist:
        p = os.path.join(fdir, name)
        if os.path.isdir(p):
            paths.extend(get_image_paths_from_dir(p))
        else:
            paths.append(p)
    return paths


def _estimate_disk_center_radius_stable(
    pil_img,
    thr_ratio=0.2,
    sat_thr_core=0.18,
):
    """
    估计太阳盘中心 (cx, cy) 和半径 r （在原始图坐标系里）。
    思路：
    - 用“亮 AND 高饱和”的像素估计太阳盘的位置，避免把白色坐标轴/白文字当太阳。
      太阳盘 / 日珥：亮 + 高饱和(红橙)。
      坐标轴 / 时间戳：亮 + 低饱和(接近白/灰)。
    - 估计半径时忽略图像最底部10%，防止底部白色时间戳把 r 拉大。
    """

    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = rgb.shape

    # 灰度亮度
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    # 饱和度：颜色强度。高饱和=红橙日珥，低饱和=白字/白轴
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = (cmax - cmin) / (cmax + 1e-6)

    maxv = float(gray.max()) if gray.size > 0 else 0.0
    if maxv <= 0.0:
        # 极端兜底：整张图黑掉或是奇怪格式
        cx = W / 2.0
        cy = H / 2.0
        r = min(H, W) / 2.2
        return float(cx), float(cy), float(r)

    # 1) 中心点估计：亮 AND 高饱和
    core_mask = (gray > (thr_ratio * maxv)) & (sat > sat_thr_core)
    ys_c, xs_c = np.where(core_mask)

    if ys_c.size < 50:
        # 如果太阳比较“平”，饱和度不算很高，就退化为亮度阈值
        bright_mask = (gray > (thr_ratio * maxv))
        ys_c, xs_c = np.where(bright_mask)

    if ys_c.size < 10:
        # 还不行 → 用图像中心兜底
        cx = W / 2.0
        cy = H / 2.0
        r = min(H, W) / 2.2
        return float(cx), float(cy), float(r)

    cx = xs_c.mean()
    cy = ys_c.mean()

    # 2) 半径估计：稍微放宽亮度+饱和度，且忽略最底部10%
    bottom_cut = int(H * 0.90)
    radius_mask = (
        (gray > (thr_ratio * maxv * 0.8)) &
        (sat > (sat_thr_core * 0.5))
    )
    radius_mask[bottom_cut:, :] = False

    ys_r, xs_r = np.where(radius_mask)
    if ys_r.size < 10:
        ys_r, xs_r = ys_c, xs_c

    dists = np.sqrt((xs_r - cx) ** 2 + (ys_r - cy) ** 2)
    r = np.percentile(dists, 99) * 1.02  # 给一点冗余，防止低估太阳边缘/日珥

    # 不允许半径大到离谱
    r_cap = min(H, W) * 0.5
    if r > r_cap:
        r = r_cap

    return float(cx), float(cy), float(r)


def crop_and_normalize_disk(
    pil_img,
    out_size,
    flip=False,
    to_normal=False,
    thr_ratio=0.2,
    radius_margin=1.05,
):
    """
    预处理单张太阳图像，返回一个张量:
      - 太阳盘居中、缩放一致
      - 坐标轴/标尺/时间戳（包括阴影）都被清理
      - 日珥和盘内结构保留
    步骤：
      1. 稳定估计 (cx,cy,r)。
      2. 以 (cx,cy) 为中心裁出边长 = 2*r*radius_margin 的正方形，pad 防越界。
      3. 可选水平翻转。
      4. resize -> (out_size, out_size)。
      5. 转 [0,1] tensor。
      6. 根据 r_norm 构建太阳盘几何掩膜。
      7. 清理：
         - 太阳盘外(>1.05*r_norm)的白轴/白刻度等
         - 底部区域里的时间戳+阴影，但仅当它几何上不在太阳圆盘内
           (避免误杀盘内耀斑)
         - 对时间戳mask做3x3膨胀后再限制到盘外
      8. 可选把范围归一到[-1,1]。
    """

    # ---------- (1) 稳定估计太阳盘几何 ----------
    cx, cy, r = _estimate_disk_center_radius_stable(
        pil_img,
        thr_ratio=thr_ratio,
        sat_thr_core=0.18,
    )

    # ---------- (2) 计算裁剪窗口 ----------
    side = int(np.ceil(2.0 * r * radius_margin))
    half = side // 2

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + side
    bottom = top + side

    W, H = pil_img.size

    # ---------- (3) pad 防止越界 ----------
    pad_left   = max(0, -left)
    pad_top    = max(0, -top)
    pad_right  = max(0, right - W)
    pad_bottom = max(0, bottom - H)

    if pad_left or pad_top or pad_right or pad_bottom:
        newW = W + pad_left + pad_right
        newH = H + pad_top + pad_bottom
        canvas = Image.new('RGB', (newW, newH), (0, 0, 0))
        canvas.paste(pil_img, (pad_left, pad_top))

        left   += pad_left
        right  += pad_left
        top    += pad_top
        bottom += pad_top
        pil_img = canvas

    crop = pil_img.crop((left, top, right, bottom))

    # ---------- (4) 可选水平翻转 ----------
    if flip:
        crop = ImageOps.mirror(crop)

    # ---------- (5) resize 到固定大小 ----------
    crop = crop.resize((out_size, out_size), resample=Image.BICUBIC)

    # ---------- (6) ToTensor -> [0,1] ----------
    tensor = transforms.ToTensor()(crop)  # [3,H,W] (CPU)
    C, Ht, Wt = tensor.shape

    # ---------- (7) 计算统一尺度下的太阳半径 r_norm ----------
    # 对任意输入，裁剪窗口边长是 2*r*radius_margin
    # resize 后，对应的太阳半径就变成常数：
    #   r_norm = out_size / (2*radius_margin)
    # 这样保证所有样本里太阳盘大小一致、居中。
    r_norm = out_size / (2.0 * radius_margin)

    yy = torch.arange(Ht, dtype=torch.float32).view(Ht, 1)
    xx = torch.arange(Wt, dtype=torch.float32).view(1, Wt)
    cy_n = (Ht - 1) / 2.0
    cx_n = (Wt - 1) / 2.0

    dist2 = (yy - cy_n) ** 2 + (xx - cx_n) ** 2
    dist = torch.sqrt(dist2 + 1e-9)

    # outside_disk:
    #   比太阳半径大很多的地方，基本就是背景（+日珥末端）
    #   用在“白色坐标轴/白刻度线”的清理
    outside_disk = dist > (r_norm * 1.05)

    # safe_outside_for_timestamp:
    #   至少不是太阳盘本体。我们用 1.00 而不是 1.05，
    #   允许贴着太阳脚边的时间戳被干掉，
    #   但保证盘内结构(耀斑)不会被误杀。
    safe_outside_for_timestamp = dist > (r_norm * 1.00)

    # ---------- (8) 颜色特征，识别“像文字/坐标轴的东西” ----------
    R = tensor[0]
    G = tensor[1]
    B = tensor[2]

    gray = (0.299 * R + 0.587 * G + 0.114 * B)

    cmax = torch.maximum(torch.maximum(R, G), B)
    cmin = torch.minimum(torch.minimum(R, G), B)
    sat  = (cmax - cmin) / (cmax + 1e-6)

    # 比较严格的白字 / 白刻度线判据
    bright_text_strict = gray > 0.4
    low_sat_strict     = sat  < 0.3
    rgb_close_strict   = (
        (torch.abs(R - G) < 0.15) &
        (torch.abs(R - B) < 0.15) &
        (torch.abs(G - B) < 0.15)
    )

    # 宽松版本，用来抓时间戳的半透明边缘
    bright_text_loose = gray > 0.30
    low_sat_loose     = sat  < 0.40
    rgb_close_loose   = (
        (torch.abs(R - G) < 0.20) &
        (torch.abs(R - B) < 0.20) &
        (torch.abs(G - B) < 0.20)
    )

    # ---------- (9) 清理盘外的白色坐标轴 / 刻度线 ----------
    mask_outside_white = (
        outside_disk &
        bright_text_strict &
        low_sat_strict &
        rgb_close_strict
    )
    # 这一类通常是图像左/下/右边那种白坐标轴、白刻度条

    # ---------- (10) 清理底部时间戳（含残影） ----------
    # 只在图像底部大约40%高度里找（时间戳几乎总在下部）
    bottom_band = yy > (Ht * 0.60)

    mask_bottom_raw = bottom_band & (
        (bright_text_strict & low_sat_strict & rgb_close_strict) |
        (bright_text_loose  & low_sat_loose  & rgb_close_loose)
    )

    # 不要误杀太阳盘内部结构：要求它至少在盘外(>= r_norm*1.00)
    mask_bottom_raw = mask_bottom_raw & safe_outside_for_timestamp

    # 3x3 膨胀，把时间戳阴影/锯齿小点也吃掉
    if mask_bottom_raw.any():
        m = mask_bottom_raw.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        kernel = torch.ones((1, 1, 3, 3), dtype=m.dtype)
        dilated = torch.nn.functional.conv2d(m, kernel, padding=1)
        mask_bottom_dilated = (dilated[0, 0] > 0)

        # dilation 可能“往太阳盘方向长回来”，再裁一刀：仍然必须是盘外
        mask_bottom_any = mask_bottom_dilated & safe_outside_for_timestamp
    else:
        mask_bottom_any = mask_bottom_raw  # 直接 False mask

    # ---------- (11) 合并要清除的区域，抹黑 ----------
    kill_mask = mask_outside_white | mask_bottom_any
    if kill_mask.any():
        tensor[:, kill_mask] = 0.0

    # ---------- (12) 可选：把范围归一到[-1,1] ----------
    if to_normal:
        tensor = (tensor - 0.5) * 2.0
        tensor = tensor.clamp_(-1., 1.)

    return tensor
