"""
LC 360 View 🧭: flat camera views out of a 360 panorama (a 2:1 equirectangular picture from any model).

Aim on the node: drag the view to look around, Shift + drag up / down to zoom, click the strip under it to jump there.
width / height set the aspect. Every line in `views` ("window side: 0, -4, 60" = name: yaw, pitch, fov) adds one more
view to the output batch, so one panorama gives a whole set of matching backgrounds for a scene.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from nodes import PreviewImage

NODE_ID = "LC360View"
NODE_DISPLAY_NAME = "LC 360 View 🧭"
PREVIEW_W = 2048  # the panorama the node shows for aiming (the output always samples the full-size one)
_LINE = re.compile(r"^\s*(?:(.+?)\s*:)?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)(?:\s*[, ]\s*(\d+(?:\.\d+)?))?\s*$")


def close_seam(pano: torch.Tensor, band: float = 0.02) -> torch.Tensor:
    """Blend the left and right edges into each other, so a panorama whose ends don't quite meet has no hard line
    where they join. pano (B, H, W, C)."""
    w = pano.shape[2]
    n = max(2, int(w * band))
    out = pano.clone()
    left, right = pano[:, :, :n], pano[:, :, -n:]
    t = torch.linspace(0, 1, n, device=pano.device).view(1, 1, n, 1)
    mid = (left + right.flip(2)) / 2  # the colour both edges meet at
    out[:, :, :n] = left * t + mid * (1 - t)
    out[:, :, -n:] = right * t.flip(2) + mid.flip(2) * (1 - t.flip(2))
    return out


def view(pano: torch.Tensor, yaw: float, pitch: float, fov: float, width: int, height: int) -> torch.Tensor:
    """One flat view (1, height, width, C) from an equirectangular panorama (1, H, W, C). yaw: degrees, 0 = the
    panorama's centre, + = right. pitch: + = up. fov: horizontal field of view in degrees."""
    dev = pano.device
    f = (width / 2) / math.tan(math.radians(fov) / 2)
    ys, xs = torch.meshgrid(torch.arange(height, device=dev, dtype=torch.float32) - height / 2 + 0.5,
                            torch.arange(width, device=dev, dtype=torch.float32) - width / 2 + 0.5, indexing="ij")
    d = torch.stack([xs, -ys, torch.full_like(xs, f)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    p, t = math.radians(pitch), math.radians(yaw)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]], device=dev)  # + pitch looks up
    ry = torch.tensor([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]], device=dev)
    d = d @ (ry @ rx).T
    lon = torch.atan2(d[..., 0], d[..., 2])
    lat = torch.asin(d[..., 1].clamp(-1, 1))
    src = pano.permute(0, 3, 1, 2)
    pad = 2  # a few columns from the other end on each side: views across the seam wrap instead of smearing
    src = torch.cat([src[..., -pad:], src, src[..., :pad]], -1)
    w_in = pano.shape[2]
    gx = (lon / math.pi) * w_in / (w_in + 2 * pad)  # the panorama's [-1, 1] inside the padded width
    gy = -lat / (math.pi / 2)
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    out = F.grid_sample(src, grid, mode="bicubic", padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1).clamp(0, 1)


def parse_views(text: str, fov: float) -> tuple:
    """[(name, yaw, pitch, fov)] from the views box; lines that don't read are reported, not fatal."""
    out, bad = [], []
    for k, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            bad.append(k)
            continue
        out.append(((m.group(1) or f"view {len(out) + 1}").strip(), float(m.group(2)), float(m.group(3)),
                    float(m.group(4)) if m.group(4) else fov))
    return out, bad


class LC360View(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE", {"tooltip": "A 360 panorama: a 2:1 equirectangular picture (e.g. 2048 x 1024), from any model or camera."}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                                  "tooltip": "Turn left / right in degrees (0 = the middle of the panorama). Or drag the view on the node."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -85.0, "max": 85.0, "step": 0.5,
                                    "tooltip": "Look up (+) or down (-). Slightly down (-3 to -6) feels like a seated camera."}),
                "fov": ("FLOAT", {"default": 70.0, "min": 10.0, "max": 150.0, "step": 0.5,
                                  "tooltip": "How wide the lens is, in degrees. 40-60 for a close-up background, 70-90 for a room. Shift + drag on the node."}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8, "tooltip": "Output width. Width and height set the aspect."}),
                "height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "views": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "More views in one go, one per line: name: yaw, pitch, fov (fov is optional). "
                                                "e.g. window side: 0, -4, 60. ➕ Add this view writes the current one here. "
                                                "Empty: just the view above."}),
                "close_seam": ("BOOLEAN", {"default": False, "tooltip": "Blend the panorama's left and right edges, for a picture whose ends don't quite meet."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "names")
    OUTPUT_TOOLTIPS = ("The view, or one per line of views (all the same size).", "The view names, one per line, in batch order.")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker"
    DESCRIPTION = ("Flat camera views out of a 360 panorama. Drag on the node to look around, Shift + drag to zoom; "
                   "width and height set the aspect; views makes a whole set at once.")

    def run(self, panorama, yaw, pitch, fov, width, height, views="", close_seam=False):
        pano = panorama[:1, ..., :3].float()
        if close_seam:
            pano = close_seam_fn(pano)
        h, w = pano.shape[1], pano.shape[2]
        if abs(w / h - 2.0) > 0.15:
            print(f"[LC MaskMaker] 360 View: the panorama is {w} x {h}, not 2:1. Views will look stretched; a 360 picture is twice as wide as tall.")
        dev = comfy_device()
        pano_d = pano.to(dev)
        want, bad = parse_views(views, fov)
        if bad:
            print(f"[LC MaskMaker] 360 View: views line(s) {', '.join(map(str, bad))} not understood (name: yaw, pitch, fov).")
        shots = want or [("view", yaw, pitch, fov)]
        imgs = [view(pano_d, y, p, f, int(width), int(height)).cpu() for _, y, p, f in shots]
        out = torch.cat(imgs, 0)
        ui = {}
        try:
            small = pano
            if w > PREVIEW_W:
                small = F.interpolate(pano.permute(0, 3, 1, 2), size=(int(h * PREVIEW_W / w), PREVIEW_W), mode="area").permute(0, 2, 3, 1)
            ui["lc_preview"] = self.save_images(small, filename_prefix="lc_360_pano")["ui"]["images"]
            ui["src_size"] = [{"width": int(w), "height": int(h)}]
        except Exception as exc:
            print(f"[LC MaskMaker] 360 View: preview skipped ({exc})")
        return {"ui": ui, "result": (out, "\n".join(n for n, *_ in shots))}


close_seam_fn = close_seam


def comfy_device():
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cpu")


NODE_CLASS_MAPPINGS = {NODE_ID: LC360View}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: NODE_DISPLAY_NAME}
