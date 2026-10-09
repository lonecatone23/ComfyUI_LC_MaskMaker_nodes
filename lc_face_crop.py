"""
LC Face Crop 📐
---------------
Crops a picture to a format (the same presets as LC Aspect Ratio Simplifier) with the faces placed well instead of
a blind centre crop: one render, every social format, nobody's head cut off. Framing from the widest crop the
picture allows down to a tight face crop (character datasets). SAM 3 finds the faces, or wire your own mask.
"""

import importlib.util
import os

import comfy.utils
import torch
from nodes import PreviewImage

from . import lc_models, lc_sam3

_FALLBACK = {
    "custom": None,
    "Instagram Portrait (4:5) - 1080x1350": (1080, 1350),
    "Instagram Square (1:1) - 1080x1080": (1080, 1080),
    "Widescreen (16:9) - 1344x768": (1344, 768),
    "TikTok (9:16) - 1080x1920": (1080, 1920),
    "CivitAI Cover (4:1) - 1600x400": (1600, 400),
    "2:3 (Portrait Photo) - 832x1248": (832, 1248),
    "3:2 (Photo) - 1248x832": (1248, 832),
    "3:4 (Portrait Standard) - 896x1152": (896, 1152),
    "4:3 (Standard) - 1152x896": (1152, 896),
    "21:9 (Ultrawide) - 1536x640": (1536, 640),
}


def _presets():
    """LC Aspect Ratio Simplifier's own list, so both nodes always offer the same formats."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI_LC123_nodes", "aspect_ratio.py")
    try:
        spec = importlib.util.spec_from_file_location("lc_face_crop_aspect_presets", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if isinstance(getattr(mod, "ASPECT_PRESETS", None), dict) and mod.ASPECT_PRESETS:
            return dict(mod.ASPECT_PRESETS)
    except Exception:
        pass
    return dict(_FALLBACK)


PRESETS = _presets()
# framing -> (crop height in face heights, where the face centre sits from the top of the crop)
FRAMING = {
    "widest": (None, 0.33),
    "upper body": (5.5, 0.28),
    "head & shoulders": (3.2, 0.38),
    "face": (1.7, 0.47),
}


def _box(m):
    ys, xs = torch.where(m > 0.5)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]


def _crop_box(W, H, ar, faces, framing, which):
    """faces: list of [x0,y0,x1,y1]. Returns integer (x0, y0, w, h) of aspect ar (w/h) inside W x H."""
    max_h = min(H, W / ar)
    if not faces:  # nobody found: plain centre crop at the widest size
        cw, ch = max_h * ar, max_h
        return int(round((W - cw) / 2)), int(round((H - ch) / 2)), int(round(cw)), int(round(ch))
    faces = sorted(faces, key=lambda f: -(f[2] - f[0]) * (f[3] - f[1]))
    if which == "largest":
        faces = faces[:1]
    ux0, uy0 = min(f[0] for f in faces), min(f[1] for f in faces)
    ux1, uy1 = max(f[2] for f in faces), max(f[3] for f in faces)
    fh = faces[0][3] - faces[0][1]  # size from the main face
    mult, place = FRAMING[framing]
    ch = max_h if mult is None else min(max_h, fh * mult)
    # every chosen face has to fit
    ch = min(max_h, max(ch, (uy1 - uy0) * 1.15, (ux1 - ux0) * 1.15 / ar))
    cw = ch * ar
    cx, cy = (ux0 + ux1) / 2, (uy0 + uy1) / 2
    x0 = min(max(0.0, cx - cw / 2), W - cw)
    y0 = min(max(0.0, cy - place * ch), H - ch)
    return int(round(x0)), int(round(y0)), int(round(cw)), int(round(ch))


def _resize(img, w, h):
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)


class LCFaceCrop(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(PRESETS.keys()), {"default": next(k for k in PRESETS if k != "custom"),
                                 "tooltip": "The format to crop to (the same list as LC Aspect Ratio Simplifier)."}),
                "custom_width": ("INT", {"default": 1080, "min": 64, "max": 16384, "step": 8,
                                         "tooltip": "Used when aspect_ratio is custom."}),
                "custom_height": ("INT", {"default": 1350, "min": 64, "max": 16384, "step": 8,
                                          "tooltip": "Used when aspect_ratio is custom."}),
                "framing": (list(FRAMING.keys()), {"default": "widest",
                            "tooltip": "widest: as much of the picture as the format allows, faces placed well. "
                                       "upper body / head & shoulders / face: tighter, sized from the face (datasets)."}),
                "faces": (["all", "largest"], {"default": "all",
                          "tooltip": "all: every face found stays in the crop. largest: frame the main face only."}),
                "output_size": (["preset size", "keep pixels"], {"default": "preset size",
                                "tooltip": "preset size: resized to the format's size (e.g. 1080x1350). "
                                           "keep pixels: the crop at the picture's own resolution."}),
                "sam3_model": (lc_models.sam3_choices(), {"tooltip": "SAM 3 model that finds the faces."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                                        "tooltip": "How sure SAM 3 has to be that it is a face."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional. Frame this mask instead of searching for faces "
                                             "(any subject: a car, a product, a pet)."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, sam3_model=None):
        # a '⬇ Download' entry saved on another machine is still valid here once the model is installed
        return lc_models.validate_models(sam3_model=("sam3", sam3_model))

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Crops to a format (LC Aspect Ratio Simplifier's presets) with the faces placed well, not a blind centre "
        "crop. widest keeps as much picture as the format allows; upper body, head & shoulders and face frame "
        "tighter from the face size (character datasets). SAM 3 finds the faces, or wire a mask to frame anything."
    )

    def run(self, image, aspect_ratio, custom_width, custom_height, framing, faces, output_size, sam3_model, threshold,
            mask=None):
        size = PRESETS.get(aspect_ratio) or (int(custom_width), int(custom_height))
        ar = size[0] / float(size[1])
        B, H, W, _ = image.shape
        path = None if mask is not None else lc_models.resolve_sam3(sam3_model)
        crops, found = [], 0
        for i in range(B):
            frame = image[i:i + 1]
            if mask is not None:
                m = mask[min(i, mask.shape[0] - 1)] if mask.dim() == 3 else mask
                if m.shape != (H, W):
                    m = torch.nn.functional.interpolate(m[None, None].float(), size=(H, W), mode="bilinear")[0, 0]
                boxes = [b for b in [_box(m)] if b]
            else:
                boxes = [b for b in (_box(m) for m in lc_sam3.objects(frame, ["face"], path, threshold)) if b]
            found += bool(boxes)
            x0, y0, cw, ch = _crop_box(W, H, ar, boxes, framing, faces)
            crop = frame[:, y0:y0 + ch, x0:x0 + cw, :]
            if output_size == "preset size":
                crop = _resize(crop, int(size[0]), int(size[1]))
            elif crops and crop.shape[1:3] != crops[0].shape[1:3]:
                crop = _resize(crop, crops[0].shape[2], crops[0].shape[1])
            crops.append(crop)
        out = torch.cat(crops, 0)
        if not found:
            print(f"[LC Face Crop] No {'mask' if mask is not None else 'face'} found: centre crop.")
        ui = self.save_images(out[:4], filename_prefix="lc_face_crop")["ui"]
        return {"ui": ui, "result": (out, int(out.shape[2]), int(out.shape[1]))}


NODE_CLASS_MAPPINGS = {"LCFaceCrop": LCFaceCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFaceCrop": "LC Face Crop 📐"}
