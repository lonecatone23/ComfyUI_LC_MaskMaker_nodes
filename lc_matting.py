"""
Matting helpers shared by the LC MaskMaker nodes: VITMatte and the
one-call refine pipeline:

    grow -> fix gaps -> trimap (edge erode / edge dilate) -> matte -> levels -> feather

The trimap is the heart of it. Sure foreground is the mask eroded by `edge_erode`,
sure background is everything beyond the mask dilated by `edge_dilate`, and the band
between them is the unknown zone the matting step decides. A tighter band means a
tighter mask.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .lc_refine_core import blur, fix_gaps, guided_filter, levels, trimap_from
from .lc_refine_core import grow as grow_mask

METHODS = ["guided_filter", "vitmatte", "none"]


def get_device():
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_vitmatte_cache = {}  # folder -> (processor, model); one model kept at a time


def _load_vitmatte(folder):
    if folder not in _vitmatte_cache:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        _vitmatte_cache.clear()
        proc = VitMatteImageProcessor.from_pretrained(folder)
        model = VitMatteForImageMatting.from_pretrained(folder).eval()
        _vitmatte_cache[folder] = (proc, model)
    return _vitmatte_cache[folder]


@torch.no_grad()
def _vitmatte(image_chw, tri, max_mp, folder, device):
    """image_chw (3,H,W), tri (1,H,W) trimap. Returns a (1,H,W) alpha on the same device."""
    from PIL import Image

    proc, model = _load_vitmatte(folder)
    _, h, w = image_chw.shape
    scale = min(1.0, (max_mp * 1_000_000 / float(h * w)) ** 0.5)
    ph, pw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
    img = F.interpolate(image_chw[None], size=(ph, pw), mode="bilinear", align_corners=False)
    small_tri = F.interpolate(tri[None], size=(ph, pw), mode="nearest")

    pil_img = (img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    pil_tri = (small_tri[0, 0].cpu().numpy() * 255).round().astype(np.uint8)
    inputs = proc(images=Image.fromarray(pil_img), trimaps=Image.fromarray(pil_tri, mode="L"), return_tensors="pt")

    model.to(device)
    try:
        alpha = model(inputs["pixel_values"].to(device)).alphas[:, :, :ph, :pw]
    finally:
        model.to("cpu")
    alpha = alpha.float().clamp(0, 1).to(tri.device)
    alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)[0]
    return torch.where(tri >= 0.99, torch.ones_like(alpha), torch.where(tri <= 0.01, torch.zeros_like(alpha), alpha))


def _matte(img, m, method, folder, erode, dilate, softness, threshold, max_megapixels, dev):
    """One trimap + matte pass. Returns (alpha, trimap), both (1,1,H,W)."""
    tri = trimap_from(m, erode, dilate, threshold)
    if method == "guided_filter":
        radius = max(1, int(round((float(erode) + float(dilate)) / 2.0)))
        alpha = guided_filter(img, m, radius, softness).clamp(0, 1)
        alpha = torch.where(tri >= 0.99, torch.ones_like(alpha), torch.where(tri <= 0.01, torch.zeros_like(alpha), alpha))
    elif method == "vitmatte":
        alpha = _vitmatte(img[0], tri[0], max_megapixels, folder, dev).unsqueeze(0)
    else:
        alpha = m
    return alpha, tri


def refine_mask(img, m, method, folder, grow, fix_gap, fix_threshold, edge_erode, edge_dilate,
                softness, threshold, black_point, white_point, feather, max_megapixels, dev,
                detail_region=None, detail_erode=72, detail_dilate=64, detail_blur=4):
    """img (1,3,H,W), m (1,1,H,W), optional detail_region (1,1,H,W) on dev.

    Returns (mask, trimap), both (1,1,H,W). Where `detail_region` is white, a second,
    wider trimap (detail_erode / detail_dilate) is used, for hair and other fine areas.
    """
    m = grow_mask(m, grow)
    m = fix_gaps(m, fix_gap, fix_threshold)

    alpha, tri = _matte(img, m, method, folder, edge_erode, edge_dilate, softness, threshold, max_megapixels, dev)
    if detail_region is not None and method != "none":
        alpha2, tri2 = _matte(img, m, method, folder, detail_erode, detail_dilate, softness, threshold, max_megapixels, dev)
        w = blur(detail_region.clamp(0, 1), detail_blur)
        alpha = alpha * (1.0 - w) + alpha2 * w
        tri = torch.where(detail_region > 0.5, tri2, tri)

    alpha = levels(alpha, black_point, white_point)
    alpha = blur(alpha, feather)
    return alpha.clamp(0, 1), tri
