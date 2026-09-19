"""
Mask helpers shared by the LC MaskMaker nodes.

Grow, shrink, trimap and gap fixing work on true Euclidean distances, so radii can be
fractional (0.1 px steps) and every edge is a real circle instead of a blocky
approximation. That uses OpenCV's distance transform when it is importable (it almost
always is in ComfyUI); without it everything still works with whole-pixel radii.

All masks are (B, 1, H, W) float in 0-1 unless a function says otherwise.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2

    _HAS_CV2 = hasattr(cv2, "distanceTransform")
except ImportError:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


# --------------------------------------------------------------------------
# box mean (O(1) per pixel, any radius)
# --------------------------------------------------------------------------
def _window_sum(t, dim, r):
    n = t.shape[dim]
    c = torch.cumsum(t, dim)
    c = torch.cat([torch.zeros_like(t.narrow(dim, 0, 1)), c], dim)
    idx = torch.arange(n, device=t.device)
    hi = (idx + r + 1).clamp(max=n)
    lo = (idx - r).clamp(min=0)
    return c.index_select(dim, hi) - c.index_select(dim, lo)


def box_mean(x, r):
    """Mean over a (2r+1) square window, clipped at the borders."""
    r = int(r)
    if r <= 0:
        return x
    _, _, h, w = x.shape
    ones = torch.ones(1, 1, h, w, device=x.device, dtype=x.dtype)
    cnt = _window_sum(_window_sum(ones, 2, r), 3, r)
    return _window_sum(_window_sum(x, 2, r), 3, r) / cnt


# --------------------------------------------------------------------------
# guided filter (color guide, He et al.)
# --------------------------------------------------------------------------
def guided_filter(guide, p, radius, eps):
    """guide: (B,3,H,W) 0-1, p: (B,1,H,W). Returns filtered p."""
    r = int(max(1, radius))
    mean_i = box_mean(guide, r)
    mean_p = box_mean(p, r)
    cov_ip = box_mean(guide * p, r) - mean_i * mean_p  # (B,3,H,W)

    cols = []
    for i in range(3):
        for j in range(3):
            cols.append(box_mean(guide[:, i:i + 1] * guide[:, j:j + 1], r) - mean_i[:, i:i + 1] * mean_i[:, j:j + 1])
    var = torch.cat(cols, 1)  # (B,9,H,W)
    b, _, h, w = var.shape
    var = var.permute(0, 2, 3, 1).reshape(b, h, w, 3, 3)
    var = var + eps * torch.eye(3, device=var.device, dtype=var.dtype)
    rhs = cov_ip.permute(0, 2, 3, 1).unsqueeze(-1)  # (B,H,W,3,1)
    a = torch.linalg.solve(var, rhs).squeeze(-1).permute(0, 3, 1, 2)  # (B,3,H,W)
    bias = mean_p - (a * mean_i).sum(1, keepdim=True)
    return (box_mean(a, r) * guide).sum(1, keepdim=True) + box_mean(bias, r)


# --------------------------------------------------------------------------
# distance-based morphology
# --------------------------------------------------------------------------
def _dist(inside):
    """Distance of every True pixel to the nearest False pixel. inside: bool (B,1,H,W)."""
    arr = inside[:, 0].cpu().numpy().astype(np.uint8)
    out = np.stack([cv2.distanceTransform(a, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for a in arr])
    return torch.from_numpy(out).unsqueeze(1).to(inside.device)


def _grow_octagon(m, px):
    """Whole-pixel fallback without OpenCV: alternating cross / square 3x3 steps."""
    px = int(px)
    if px == 0:
        return m
    invert = px < 0
    x = 1.0 - m if invert else m
    for i in range(abs(px)):
        if i % 2 == 0:
            x = torch.maximum(F.max_pool2d(x, (3, 1), 1, (1, 0)), F.max_pool2d(x, (1, 3), 1, (0, 1)))
        else:
            x = F.max_pool2d(x, 3, 1, 1)
    return 1.0 - x if invert else x


def grow(m, px):
    """Dilate (px > 0) or erode (px < 0) by |px| pixels; px may be fractional."""
    px = float(px)
    if abs(px) < 1e-6:
        return m
    if not _HAS_CV2:
        return _grow_octagon(m, round(px))
    binary = m > 0.5
    if px > 0:
        cover = (px - _dist(~binary) + 1.0).clamp(0.0, 1.0)  # distance outward from the edge
        return torch.maximum(m, cover)
    cover = (_dist(binary) + px).clamp(0.0, 1.0)  # px is negative
    return torch.minimum(m, cover)


def trimap_from(m, erode, dilate, threshold=0.5):
    """(B,1,H,W) trimap: 1 = sure foreground, 0 = sure background, 0.5 = unknown.

    erode: how far the sure-foreground area is pulled in from the mask edge (px, fractional ok).
    dilate: how far out the unknown band reaches beyond the mask edge (px, fractional ok).
    """
    binary = m > threshold
    erode, dilate = float(erode), float(dilate)
    if _HAS_CV2:
        fg = _dist(binary) >= erode + 0.5
        bg = _dist(~binary) > dilate + 0.5
        bg = bg & ~binary
    else:
        b = binary.to(m.dtype)
        fg = (_grow_octagon(b, -round(erode)) > 0.5) if erode > 0 else binary
        bg = (1.0 - (_grow_octagon(b, round(dilate)) if dilate > 0 else b)) > 0.5
    tri = torch.full_like(m, 0.5)
    tri = torch.where(fg, torch.ones_like(tri), tri)
    tri = torch.where(bg, torch.zeros_like(tri), tri)
    return tri


def fix_gaps(m, gap, threshold):
    """Close small holes and cracks (binary closing by `gap` px), merged back over the mask."""
    gap = float(gap)
    if gap <= 0:
        return m
    binary = m > threshold
    if _HAS_CV2:
        dilated = (_dist(~binary) <= gap + 0.5) | binary
        closed = (_dist(dilated) >= gap + 0.5).to(m.dtype)
    else:
        b = binary.to(m.dtype)
        closed = _grow_octagon(_grow_octagon(b, round(gap)), -round(gap))
    return torch.maximum(m, closed)


# --------------------------------------------------------------------------
# levels, blur, color spread
# --------------------------------------------------------------------------
def levels(m, black, white):
    white = max(float(white), float(black) + 1e-4)
    return ((m - float(black)) / (white - float(black))).clamp(0.0, 1.0)


def blur(m, sigma):
    """Gaussian blur, sigma in pixels (fractional ok). Large sigmas use three box passes."""
    sigma = float(sigma)
    if sigma <= 0.05:
        return m
    if sigma > 24.0:
        w = math.sqrt(12.0 * sigma * sigma / 3.0 + 1.0)
        r = max(1, int(round((w - 1.0) / 2.0)))
        for _ in range(3):
            m = box_mean(m, r)
        return m
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=m.device, dtype=m.dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = m.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    m = F.pad(m, (radius, radius, 0, 0), mode="replicate")
    m = F.conv2d(m, kh, groups=c)
    m = F.pad(m, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(m, kv, groups=c)


def spread_colors(img, alpha, px):
    """Push clean foreground colors outward over the soft edge (less background halo).

    img (B,3,H,W), alpha (B,1,H,W). Pixels that are not fully opaque take the average
    color of nearby solid foreground pixels.
    """
    if px <= 0:
        return img
    solid = (grow(alpha, -1) > 0.98).to(img.dtype)
    r = int(px)
    num = box_mean(img * solid, r)
    den = box_mean(solid, r)
    est = num / den.clamp_min(1e-4)
    use = ((den > 1e-3) & (alpha < 0.99)).to(img.dtype)
    return img * (1.0 - use) + est * use
