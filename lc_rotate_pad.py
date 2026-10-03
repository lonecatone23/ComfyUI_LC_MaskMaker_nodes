"""
LC Image Rotate + Pad
---------------------
Turn, crop and pad a picture by dragging on the node. The picture turns around its centre, then one frame box
decides what comes out: drag an edge out to pad, in to crop. The mask is white wherever the model has to paint
(the padding and the corners a turn opens up), so it goes straight into LC Krea2 AnyPaint or any inpaint node.

left/top/right/bottom store the frame edges as a percentage of the source (left/right of its width, top/bottom of
its height), measured from the base box: positive = pad outward, negative = crop inward. The base box is the whole
turned picture (fit = expand) or the largest box with the source's shape that has no empty corners (fit = crop).
Inspired by AusBoss's Image Crop + Rotate + Pad (ausboss-nodes, MIT); written from scratch.
"""

import math

import torch
import torch.nn.functional as F
from nodes import PreviewImage

ASPECT_OPTIONS = ["free", "original", "1:1", "4:3", "3:2", "16:9", "3:4", "2:3", "9:16"]
SNAP_OPTIONS = ["1", "8", "16", "32", "64"]
FIT_OPTIONS = ["expand", "crop"]
FILL_OPTIONS = ["gray", "edge", "white", "black"]
FILL_VALUE = {"gray": 0.5, "white": 1.0, "black": 0.0}

MIN_PCT = -100.0
MAX_PCT = 400.0
MIN_SIDE = 16


def _pct(tip):
    return ("FLOAT", {"default": 0.0, "min": MIN_PCT, "max": MAX_PCT, "step": 0.01, "tooltip": tip})


def base_box(w, h, angle, fit):
    """Size of the base box around the picture centre after a turn of `angle` degrees (clockwise)."""
    a = math.radians(angle)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    bw, bh = w * c + h * s, w * s + h * c
    if fit == "crop":
        k = min(w / bw, h / bh) if bw > 0 and bh > 0 else 1.0
        m = 0 if abs(angle / 90.0 - round(angle / 90.0)) < 1e-9 else 2
        return max(1.0, w * k - m), max(1.0, h * k - m)  # 1 px inside, so rounding never shows a corner
    return bw, bh


def frame_rect(w, h, angle, fit, left, top, right, bottom):
    """Frame edges in pixels around the picture centre: (x0, y0, x1, y1), at least MIN_SIDE wide and high."""
    bw, bh = base_box(w, h, angle, fit)
    x0 = -bw / 2 - left / 100.0 * w
    x1 = bw / 2 + right / 100.0 * w
    y0 = -bh / 2 - top / 100.0 * h
    y1 = bh / 2 + bottom / 100.0 * h
    if x1 - x0 < MIN_SIDE:
        mid = (x0 + x1) / 2
        x0, x1 = mid - MIN_SIDE / 2, mid + MIN_SIDE / 2
    if y1 - y0 < MIN_SIDE:
        mid = (y0 + y1) / 2
        y0, y1 = mid - MIN_SIDE / 2, mid + MIN_SIDE / 2
    return x0, y0, x1, y1


class LCImageRotatePad(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Picture to turn, crop and pad."}),
                "keep_frame": ("BOOLEAN", {
                    "default": True,
                    "label_on": "keep (same image)",
                    "label_off": "snap back after run",
                    "tooltip": (
                        "keep: the turn and frame stay as you set them, run after run. Use it when you rerun the "
                        "same picture (post processing, same seed).\nsnap back after run: once a run finishes, the "
                        "turn and frame go back to the plain picture, so the next image starts clean."
                    ),
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1,
                    "tooltip": "Turn in degrees, clockwise. Or drag the round handle on the node (hold Shift for 15° steps).",
                }),
                "fit": (FIT_OPTIONS, {
                    "default": "expand",
                    "tooltip": (
                        "expand: the frame starts around the whole turned picture, and the empty corners are masked "
                        "for the model to paint.\ncrop: the frame starts at the largest box with no empty corners."
                    ),
                }),
                "fill": (FILL_OPTIONS, {
                    "default": "gray",
                    "tooltip": (
                        "What goes in the new area before the model paints it. gray is what Krea 2 and AnyPaint expect. "
                        "edge stretches the outer pixels out (for models that copy what they see)."
                    ),
                }),
                "left": _pct("Frame left edge, % of source width. Positive pads, negative crops."),
                "top": _pct("Frame top edge, % of source height. Positive pads, negative crops."),
                "right": _pct("Frame right edge, % of source width. Positive pads, negative crops."),
                "bottom": _pct("Frame bottom edge, % of source height. Positive pads, negative crops."),
                "aspect": (ASPECT_OPTIONS, {
                    "default": "free",
                    "tooltip": "Lock the output shape. 'original' matches the source picture.",
                }),
                "snap_to": (SNAP_OPTIONS, {
                    "default": "16",
                    "tooltip": "Round the output width and height up to a multiple of this (extra goes right and bottom). 1 = off.",
                }),
                "block": ("BOOLEAN", {
                    "default": True,
                    "label_on": "until turned or framed",
                    "label_off": "never",
                    "tooltip": (
                        "until turned or framed: while the picture is untouched (no turn, no crop, no pad), the run "
                        "stops after this node. Run once to see the picture here, set it up, then run again." + "\n"
                        "never: always passes the picture on, even untouched."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("control_image", "control_mask", "width", "height")
    OUTPUT_TOOLTIPS = (
        "The turned, cropped and padded picture.",
        "White where the model paints: the padding and the corners a turn opens up.",
        "Output width.",
        "Output height.",
    )
    FUNCTION = "rotate_pad"
    CATEGORY = "LC MaskMaker/image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Turn, crop and pad a picture by dragging on the node: the round handle turns it, the frame edges pad "
        "(drag out) or crop (drag in). The mask covers the new area and the corners a turn opens up."
    )

    def rotate_pad(self, image, angle, fit, fill, left, top, right, bottom, aspect="free", snap_to="16", keep_frame=True, block=True):
        b, h, w, c = image.shape
        angle = float(angle)
        clamp = lambda v: max(MIN_PCT, min(MAX_PCT, float(v)))
        x0, y0, x1, y1 = frame_rect(w, h, angle, fit, clamp(left), clamp(top), clamp(right), clamp(bottom))

        try:
            m = max(1, int(snap_to))
        except (TypeError, ValueError):
            m = 1
        out_w = max(MIN_SIDE, int(round(x1 - x0)))
        out_h = max(MIN_SIDE, int(round(y1 - y0)))
        out_w += (-out_w) % m
        out_h += (-out_h) % m
        x0, y0 = round(x0), round(y0)

        quarter = angle / 90.0
        if abs(quarter - round(quarter)) < 1e-9:
            out, mask = self._exact(image, int(round(quarter)) % 4, x0, y0, out_w, out_h, fill)
        else:
            out, mask = self._turned(image, angle, x0, y0, out_w, out_h, fill)

        # snap back: the browser resets the turn and frame once the whole run has finished
        outputs = (out, mask, int(out_w), int(out_h))
        untouched = angle % 360 == 0 and all(float(v) == 0 for v in (left, top, right, bottom))
        if block and untouched:
            outputs = self._blocked(outputs)
        result = {"ui": {} if keep_frame else {"lc_reset_after": [True]}, "result": outputs}
        try:
            saved = self.save_images(image[:1], filename_prefix="lc_rotate_pad_src")
            result["ui"]["lc_preview"] = saved["ui"]["images"]
            result["ui"]["src_size"] = [{"width": int(w), "height": int(h)}]
        except Exception:
            pass
        return result

    @staticmethod
    def _blocked(outputs):
        try:
            from comfy_execution.graph import ExecutionBlocker

            return tuple(ExecutionBlocker(None) for _ in outputs)
        except ImportError:
            print("[LC Image Rotate + Pad] ComfyUI is too old for ExecutionBlocker - block is disabled.")
            return outputs

    @staticmethod
    def _exact(image, k, x0, y0, out_w, out_h, fill):
        """Quarter turns: pixels are copied, never resampled, so a straight picture comes back bit-exact."""
        src = torch.rot90(image, k=-k, dims=(1, 2)) if k else image  # clockwise
        b, h, w, c = src.shape
        # picture's top-left in output pixels (frame coordinates are around the centre)
        px, py = int(round(-w / 2 - x0)), int(round(-h / 2 - y0))
        mask = torch.ones((b, out_h, out_w), dtype=torch.float32, device=image.device)
        sx0, sy0 = max(0, -px), max(0, -py)
        dx0, dy0 = max(0, px), max(0, py)
        cw = min(w - sx0, out_w - dx0)
        ch = min(h - sy0, out_h - dy0)
        if fill == "edge":
            ys = (torch.arange(out_h, device=image.device) - py).clamp(0, h - 1)
            xs = (torch.arange(out_w, device=image.device) - px).clamp(0, w - 1)
            out = src[:, ys][:, :, xs].clone()
        else:
            out = torch.full((b, out_h, out_w, c), FILL_VALUE[fill], dtype=image.dtype, device=image.device)
            if c == 4:
                out[..., 3] = 1.0
        if cw > 0 and ch > 0:
            out[:, dy0:dy0 + ch, dx0:dx0 + cw] = src[:, sy0:sy0 + ch, sx0:sx0 + cw]
            mask[:, dy0:dy0 + ch, dx0:dx0 + cw] = 0.0
        return out, mask

    @staticmethod
    def _turned(image, angle, x0, y0, out_w, out_h, fill):
        b, h, w, c = image.shape
        dev = image.device
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        X = torch.arange(out_w, device=dev, dtype=torch.float32) + 0.5 + x0
        Y = torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5 + y0
        Y, X = torch.meshgrid(Y, X, indexing="ij")
        # undo the clockwise turn (screen y points down) to find each output pixel in the source
        u = X * ca + Y * sa + w / 2
        v = -X * sa + Y * ca + h / 2
        grid = torch.stack((2 * u / w - 1, 2 * v / h - 1), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        src = image.permute(0, 3, 1, 2).float()
        out = F.grid_sample(src, grid, mode="bicubic", padding_mode="border", align_corners=False)
        out = out.clamp(0, 1).permute(0, 2, 3, 1).to(image.dtype)
        # a pixel is kept only when it sits fully inside the turned picture; edge pixels get painted too
        r = 0.5 * (abs(ca) + abs(sa))
        inside = (u >= r) & (u <= w - r) & (v >= r) & (v <= h - r)
        mask = (~inside).float().unsqueeze(0).expand(b, -1, -1).contiguous()
        if fill != "edge":
            val = torch.full((c,), FILL_VALUE[fill], dtype=out.dtype, device=dev)
            if c == 4:
                val[3] = 1.0
            out = torch.where(mask.unsqueeze(-1) > 0, val, out)
        return out, mask


NODE_CLASS_MAPPINGS = {"LCImageRotatePad": LCImageRotatePad}
NODE_DISPLAY_NAME_MAPPINGS = {"LCImageRotatePad": "LC Image Rotate + Pad 🔄"}
