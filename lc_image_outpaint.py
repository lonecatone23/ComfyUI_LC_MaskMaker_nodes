"""
LC Outpaint
-----------
Interactive canvas expansion (drag the edges of the canvas outward on the node).
The source is placed on a grey canvas; the mask marks the new area (white = new).
left/top/right/bottom store the expansion as a percentage of the source
(left/right of its width, top/bottom of its height).
"""

import math

import torch
from nodes import PreviewImage

ASPECT_OPTIONS = [
    "free",
    "original",
    "1:1",
    "4:3",
    "3:2",
    "16:9",
    "3:4",
    "2:3",
    "9:16",
]

SNAP_OPTIONS = ["1", "8", "16", "32", "64"]

MAX_PCT = 400.0
FILL = 0.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _pct(tip):
    return (
        "FLOAT",
        {"default": 0.0, "min": 0.0, "max": MAX_PCT, "step": 0.01, "tooltip": tip},
    )


class LCImageOutpaint(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to extend."}),
                "block": ("BOOLEAN", {
                    "default": False,
                    "label_on": "if_empty_mask",
                    "label_off": "never",
                    "tooltip": (
                        "if_empty_mask: until you have pulled at least one edge out (all four expansions are 0), everything "
                        "downstream of this node is stopped so you can set up the canvas, then queue again.\n"
                        "never: always passes the canvas and mask through."
                    ),
                }),
                "hold_mask": ("BOOLEAN", {
                    "default": False,
                    "label_on": "held",
                    "label_off": "snap_to_image",
                    "tooltip": (
                        "held: keep the canvas you set up (the expansions below) and pass the image through with it.\n"
                        "snap_to_image: the canvas snaps back to the image size on every run and the image passes "
                        "through as is, so a new image always starts from a clean box. With block set to if_empty_mask "
                        "the run also stops there, so you can drag the edges out, switch to held, and run again."
                    ),
                }),
                "left": _pct("Expansion on the left, % of source width."),
                "top": _pct("Expansion on the top, % of source height."),
                "right": _pct("Expansion on the right, % of source width."),
                "bottom": _pct("Expansion on the bottom, % of source height."),
                "aspect": (ASPECT_OPTIONS, {
                    "default": "free",
                    "tooltip": "Lock the final canvas aspect ratio. 'original' matches the source image.",
                }),
                "snap_to": (SNAP_OPTIONS, {
                    "default": "8",
                    "tooltip": "Round the final width/height up to a multiple of this (extra goes to the right/bottom). 1 = off.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("control_image", "control_mask", "mask_image", "width", "height")
    FUNCTION = "outpaint"
    CATEGORY = "LC MaskMaker/image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Interactive outpaint prep: drag the canvas edges outward on the node. "
        "Source sits on a grey canvas; the mask is white where the new area is. "
        "Optional aspect lock (free / original / common ratios). block set to if_empty_mask stops everything "
        "downstream until you have pulled an edge out."
    )

    def outpaint(self, image, left, top, right, bottom, aspect="free", snap_to="8", block=False, hold_mask=False, unique_id=None):
        b, h, w, c = image.shape

        if not hold_mask:
            # Not held: the canvas snaps to the image. Nothing is expanded, so nothing is snap-padded either.
            mask = torch.zeros((b, h, w), dtype=torch.float32, device=image.device)
            out = (image, mask, mask.unsqueeze(-1).repeat(1, 1, 1, 3), int(w), int(h))
            if block:
                out = self._blocked(out)
            result = {"ui": {"lc_reset": [True]}, "result": out}
            self._add_preview(result, image, w, h, w, h)
            return result

        l = int(round(w * _clamp(float(left), 0.0, MAX_PCT) / 100.0))
        r = int(round(w * _clamp(float(right), 0.0, MAX_PCT) / 100.0))
        t = int(round(h * _clamp(float(top), 0.0, MAX_PCT) / 100.0))
        bt = int(round(h * _clamp(float(bottom), 0.0, MAX_PCT) / 100.0))

        is_empty = l == 0 and r == 0 and t == 0 and bt == 0

        try:
            m = max(1, int(snap_to))
        except (TypeError, ValueError):
            m = 1
        if m > 1:
            r += (-(w + l + r)) % m
            bt += (-(h + t + bt)) % m

        out_w = w + l + r
        out_h = h + t + bt

        canvas = torch.full((b, out_h, out_w, c), FILL, dtype=image.dtype, device=image.device)
        if c == 4:
            canvas[..., 3] = 1.0
        canvas[:, t:t + h, l:l + w, :] = image

        mask = torch.ones((b, out_h, out_w), dtype=torch.float32, device=image.device)
        mask[:, t:t + h, l:l + w] = 0.0
        mask_image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        out = (canvas, mask, mask_image, int(out_w), int(out_h))
        if block and is_empty:
            out = self._blocked(out)

        result = {"ui": {}, "result": out}
        self._add_preview(result, image, w, h, out_w, out_h)
        return result

    @staticmethod
    def _blocked(out):
        try:
            from comfy_execution.graph import ExecutionBlocker

            return tuple(ExecutionBlocker(None) for _ in out)
        except ImportError:
            print("[LC Outpaint] ComfyUI is too old for ExecutionBlocker - block is disabled.")
            return out

    def _add_preview(self, result, image, w, h, out_w, out_h):
        try:
            saved = self.save_images(image[:1], filename_prefix="lc_outpaint_src")
            result["ui"]["lc_preview"] = saved["ui"]["images"]
            result["ui"]["src_size"] = [{"width": int(w), "height": int(h)}]
            result["ui"]["out_size"] = [{"width": int(out_w), "height": int(out_h)}]
        except Exception:
            pass


NODE_CLASS_MAPPINGS = {
    "LCImageOutpaint": LCImageOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCImageOutpaint": "LC Outpaint 🖌️➕",
}
