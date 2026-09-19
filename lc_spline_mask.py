"""
LC Create Mask
--------------
Draw a mask on the image right on the node: click to add points (as many as you
like), drag them to reshape, or switch to the pencil and draw freehand.
Points are stored normalized (0-1) in a hidden widget, so the shape follows the
image if its resolution changes.
"""

import json

import numpy as np
import torch
from nodes import PreviewImage
from PIL import Image, ImageDraw, ImageFilter


def _parse_points(text):
    try:
        raw = json.loads(text or "[]")
    except (TypeError, ValueError):
        return []
    pts = []
    if not isinstance(raw, list):
        return pts
    for p in raw:
        try:
            if isinstance(p, dict):
                x, y = float(p["x"]), float(p["y"])
            else:
                x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return pts


def _catmull_rom_closed(pts, samples=16):
    """Closed uniform Catmull-Rom through pts (Nx2 array)."""
    n = len(pts)
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t ** 3
        )
        out.append(seg)
    return np.concatenate(out, axis=0)


def _render_mask(points, w, h, smooth, feather):
    """Return an (h, w) float32 mask in 0-1; empty when fewer than 3 points."""
    if len(points) < 3:
        return np.zeros((h, w), dtype=np.float32)

    ss = 2 if max(w, h) <= 2048 else 1
    px = np.array([[x * w * ss, y * h * ss] for x, y in points], dtype=np.float64)
    if smooth:
        px = _catmull_rom_closed(px)
        px[:, 0] = np.clip(px[:, 0], 0, w * ss)
        px[:, 1] = np.clip(px[:, 1], 0, h * ss)

    img = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in px], fill=255)
    if ss > 1:
        img = img.resize((w, h), Image.LANCZOS)
    if feather > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return np.asarray(img, dtype=np.float32) / 255.0


class LCSplineMask(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to draw the mask on."}),
                "block": ("BOOLEAN", {
                    "default": False,
                    "label_on": "if_empty_mask",
                    "label_off": "never",
                    "tooltip": (
                        "if_empty_mask: when nothing is drawn (no shape, or an all-black mask), everything "
                        "downstream of this node is stopped so you can draw, then queue again.\n"
                        "never: always passes the image and mask through."
                    ),
                }),
                "pencil": ("BOOLEAN", {
                    "default": False,
                    "label_on": "pencil",
                    "label_off": "points",
                    "tooltip": "points: click to add, drag to move, shift+click removes. pencil: draw freehand.",
                }),
                "smooth": ("BOOLEAN", {
                    "default": False,
                    "label_on": "smooth",
                    "label_off": "straight",
                    "tooltip": "smooth curves through the points instead of straight edges.",
                }),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the mask (outside becomes white)."}),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Edge softness in pixels (gaussian blur radius).",
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask strength. Scales the mask values (1.0 = solid).",
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How strongly the mask shows over the preview on the node. Display only.",
                }),
                "points": ("STRING", {"default": "[]", "tooltip": "Internal: normalized point list."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "make"
    CATEGORY = "LC MaskMaker/mask"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Draw a mask on the node: click to add points (no limit), drag to reshape, "
        "or use the pencil for freehand. Feather, invert, and opacity built in."
    )

    def make(self, image, block=False, pencil=False, smooth=False, invert=False, feather=0.0,
             opacity=1.0, overlay_opacity=0.45, points="[]"):
        b, h, w, _c = image.shape
        pts = _parse_points(points)
        mask = _render_mask(pts, w, h, bool(smooth), float(feather))
        if invert:
            mask = 1.0 - mask
        mask = np.clip(mask * float(opacity), 0.0, 1.0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).repeat(b, 1, 1).to(image.device)

        out = (image, mask_t)
        is_empty = len(pts) < 3 or not bool(mask.any())
        if block and is_empty:
            try:
                from comfy_execution.graph import ExecutionBlocker

                out = (ExecutionBlocker(None), ExecutionBlocker(None))
            except ImportError:
                print("[LC Create Mask] ComfyUI is too old for ExecutionBlocker - block is disabled.")

        result = {"ui": {}, "result": out}
        try:
            saved = self.save_images(image[:1], filename_prefix="lc_spline_src")
            result["ui"]["lc_preview"] = saved["ui"]["images"]
            result["ui"]["src_size"] = [{"width": int(w), "height": int(h)}]
        except Exception:
            pass
        return result


NODE_CLASS_MAPPINGS = {
    "LCSplineMask": LCSplineMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSplineMask": "LC Create Mask 🖼️✏️",
}
