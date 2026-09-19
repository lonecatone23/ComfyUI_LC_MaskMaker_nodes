"""
LC Image Blend Advance
----------------------
Place a layer on a background (position, scale, aspect, rotate, mirror) and blend it with one of
29 blend modes, through an optional mask. The background is optional: without one the layer goes on a
transparent canvas. Pure torch, no extra dependencies.
"""

import math

import torch
import torch.nn.functional as F

from .lc_blend_modes import MODE_NAMES, blend_rgba
from .lc_matting import get_device

METHODS = ["lanczos", "bicubic", "bilinear", "nearest"]


def _lanczos_matrix(n_in, n_out, device, dtype, a=3):
    """(n_out, n_in) Lanczos-3 resampling weights (support widens when shrinking, like PIL)."""
    scale = n_in / float(n_out)
    stretch = max(scale, 1.0)
    centers = (torch.arange(n_out, device=device, dtype=dtype) + 0.5) * scale
    x = ((torch.arange(n_in, device=device, dtype=dtype) + 0.5)[None, :] - centers[:, None]) / stretch
    w = torch.where(x.abs() < a, torch.sinc(x) * torch.sinc(x / a), torch.zeros_like(x))
    return w / w.sum(1, keepdim=True).clamp_min(1e-8)


def _lanczos(prem, th, tw):
    """prem (1,C,h,w) -> (1,C,th,tw) with Lanczos-3, separable."""
    _, _, h, w = prem.shape
    if (h, w) == (th, tw):
        return prem
    wy = _lanczos_matrix(h, th, prem.device, prem.dtype)
    wx = _lanczos_matrix(w, tw, prem.device, prem.dtype)
    out = torch.matmul(wy, prem)                       # (1,C,th,w)
    out = torch.matmul(out, wx.T)                      # (1,C,th,tw)
    return out.clamp(0, 1)


def _resize(prem, tw, th, method):
    """prem (1,4,h,w) premultiplied RGBA -> (1,4,th,tw)."""
    _, _, h, w = prem.shape
    if (w, h) == (tw, th):
        return prem
    if method == "lanczos":
        return _lanczos(prem, th, tw)
    if method == "nearest":
        return F.interpolate(prem, size=(th, tw), mode="nearest")
    down = tw < w or th < h
    return F.interpolate(prem, size=(th, tw), mode=method, align_corners=False, antialias=down)


def _rotate(prem, angle, method, ssaa):
    """Rotate counter-clockwise by `angle` degrees, growing the canvas so nothing is cut off."""
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    _, _, h, w = prem.shape
    nw = max(1, int(math.ceil(abs(w * c) + abs(h * s) - 1e-6)))
    nh = max(1, int(math.ceil(abs(w * s) + abs(h * c) - 1e-6)))
    k = max(1, int(ssaa))
    ow, oh = nw * k, nh * k
    xs = (torch.arange(ow, device=prem.device, dtype=prem.dtype) + 0.5) / k - nw / 2.0
    ys = (torch.arange(oh, device=prem.device, dtype=prem.dtype) + 0.5) / k - nh / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    src_x = xx * c - yy * s
    src_y = xx * s + yy * c
    grid = torch.stack([src_x / (w / 2.0), src_y / (h / 2.0)], dim=-1)[None]
    sampler = "bicubic" if method == "lanczos" else method   # like PIL: lanczos resizes, rotation samples bicubic
    out = F.grid_sample(prem, grid, mode=sampler, padding_mode="zeros", align_corners=False)
    if k > 1:
        out = _lanczos(out, nh, nw) if method == "lanczos" else F.avg_pool2d(out, k)
    return out


def _transform_layer(rgb, alpha, mirror, scale, aspect, rotate, method, ssaa):
    """rgb (h,w,3), alpha (h,w,1). Returns transformed (rgb (H,W,3), alpha (H,W,1))."""
    h, w = rgb.shape[:2]
    prem = torch.cat([rgb * alpha, alpha], dim=-1).permute(2, 0, 1)[None]  # 1,4,h,w
    if mirror == "horizontal":
        prem = prem.flip(3)
    elif mirror == "vertical":
        prem = prem.flip(2)

    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale * aspect)))
    prem = _resize(prem, tw, th, method)

    if abs(rotate % 360.0) > 1e-6:
        prem = _rotate(prem, rotate, method, ssaa)

    prem = prem[0].permute(1, 2, 0)  # H,W,4
    a = prem[..., 3:4].clamp(0, 1)
    color = torch.where(a > 1e-5, prem[..., :3] / a.clamp_min(1e-5), torch.zeros_like(prem[..., :3]))
    return color.clamp(0, 1), a


class LCImageBlendAdvance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE", {"tooltip": "The layer. An alpha channel, if it has one, is used."}),
                "blend_mode": (MODE_NAMES, {
                    "default": "normal",
                    "tooltip": (
                        "Standard blend modes. overlay looks at the background, hard_light looks at the layer.\n"
                        "LayerStyle's older ImageBlend nodes had these two swapped; see swap_roles."
                    ),
                }),
                "opacity": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Layer strength in percent.",
                }),
                "swap_roles": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Swap which image is the base and which is the blend for the mode's formula.\n"
                        "On + overlay reproduces LayerStyle ImageBlend's old 'overlay' (a standard hard_light)."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Flip layer_mask (black becomes visible)."}),
                "x_percent": ("FLOAT", {
                    "default": 50.0, "min": -999.0, "max": 999.0, "step": 0.01,
                    "tooltip": "Layer center, as a percent of the background width. 50 = centered.",
                }),
                "y_percent": ("FLOAT", {
                    "default": 50.0, "min": -999.0, "max": 999.0, "step": 0.01,
                    "tooltip": "Layer center, as a percent of the background height. 50 = centered.",
                }),
                "mirror": (["None", "horizontal", "vertical"], {"default": "None"}),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.001,
                    "tooltip": "Layer size multiplier.",
                }),
                "aspect_ratio": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.001,
                    "tooltip": "Stretches the layer height. 1 = keep proportions.",
                }),
                "rotate": ("FLOAT", {
                    "default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.01,
                    "tooltip": "Degrees, counter-clockwise.",
                }),
                "transform_method": (METHODS, {
                    "default": "lanczos",
                    "tooltip": "How the layer is resampled when scaled or rotated. lanczos is the sharpest for resizing; rotation itself always samples with bicubic, like PIL.",
                }),
                "anti_aliasing": ("INT", {
                    "default": 2, "min": 0, "max": 4, "step": 1,
                    "tooltip": "Smooths rotated edges by drawing them larger and shrinking. 0 = off. Higher is slower.",
                }),
            },
            "optional": {
                "background_image": ("IMAGE", {
                    "tooltip": (
                        "The image the layer is placed on. Leave it unconnected to place the layer on a transparent canvas "
                        "the size of the layer (the output is then RGBA, with the mask as its alpha)."
                    ),
                }),
                "layer_mask": ("MASK", {"tooltip": "White = the layer shows. Multiplied with the layer's alpha, if any."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "blend_layer"
    CATEGORY = "LC MaskMaker/blend"
    DESCRIPTION = (
        "Place a layer on a background (position, scale, aspect, rotate, mirror) and blend it with "
        "one of 29 blend modes, through an optional mask."
    )

    def blend_layer(self, layer_image, blend_mode, opacity, swap_roles, invert_mask,
                    x_percent, y_percent, mirror, scale, aspect_ratio, rotate, transform_method,
                    anti_aliasing, background_image=None, layer_mask=None):
        dev = get_device()
        nl = layer_image.shape[0]
        nb = background_image.shape[0] if background_image is not None else nl
        masks_in = None
        if layer_mask is not None:
            masks_in = layer_mask.unsqueeze(0) if layer_mask.dim() == 2 else layer_mask
        n = max(nb, nl, masks_in.shape[0] if masks_in is not None else 1)

        outs, out_masks = [], []
        for i in range(n):
            lay = layer_image[min(i, nl - 1)].to(dev).float()
            lh, lw = lay.shape[:2]
            if background_image is None:      # transparent canvas the size of the layer
                bg = torch.zeros(lh, lw, 3, device=dev)
                bg_a = torch.zeros(lh, lw, 1, device=dev)
                rgba_out = True
            else:
                raw = background_image[min(i, nb - 1)].to(dev).float()
                bg = raw[..., :3]
                bg_a = raw[..., 3:4] if raw.shape[-1] == 4 else torch.ones(*raw.shape[:2], 1, device=dev)
                rgba_out = raw.shape[-1] == 4
            rgb = lay[..., :3]
            alpha = lay[..., 3:4] if lay.shape[-1] == 4 else torch.ones(lh, lw, 1, device=dev)
            if masks_in is not None:
                m = masks_in[min(i, masks_in.shape[0] - 1)].to(dev).float()
                if m.shape != (lh, lw):
                    m = F.interpolate(m[None, None], size=(lh, lw), mode="bilinear", align_corners=False)[0, 0]
                if invert_mask:
                    m = 1.0 - m
                alpha = alpha * m.clamp(0, 1).unsqueeze(-1)

            rgb, alpha = _transform_layer(rgb, alpha, mirror, scale, aspect_ratio, rotate,
                                          transform_method, anti_aliasing)
            ph, pw = rgb.shape[:2]
            hc, wc = bg.shape[:2]

            x0 = int(round(wc * x_percent / 100.0 - pw / 2.0))
            y0 = int(round(hc * y_percent / 100.0 - ph / 2.0))
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(wc, x0 + pw), min(hc, y0 + ph)

            result, result_a = bg.clone(), bg_a.clone()
            full_alpha = torch.zeros(hc, wc, 1, device=dev)
            if cx1 > cx0 and cy1 > cy0:
                sx0, sy0 = cx0 - x0, cy0 - y0
                sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
                src = rgb[sy0:sy1, sx0:sx1][None]
                a = alpha[sy0:sy1, sx0:sx1][None]
                color, ar = blend_rgba(bg[cy0:cy1, cx0:cx1][None], bg_a[cy0:cy1, cx0:cx1][None], src, a,
                                       blend_mode, opacity / 100.0, swap_roles, seed=i)
                result[cy0:cy1, cx0:cx1] = color[0]
                result_a[cy0:cy1, cx0:cx1] = ar[0]
                full_alpha[cy0:cy1, cx0:cx1] = a[0]

            out = torch.cat([result, result_a], dim=-1) if rgba_out else result
            outs.append(out.cpu())
            out_masks.append(full_alpha[..., 0].cpu())

        return (torch.stack(outs, 0), torch.stack(out_masks, 0))


NODE_CLASS_MAPPINGS = {
    "LCImageBlendAdvance": LCImageBlendAdvance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCImageBlendAdvance": "LC Image Blend Advance 🎚️",
}
