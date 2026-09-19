"""
LC Remove Background
--------------------
Cut the subject out of an image with a BiRefNet-family model. Uses ComfyUI's
built-in background-removal loader, so there is no extra Python dependency and
ComfyUI manages the model's memory. Optional refine step (guided filter or
VITMatte) plus grow, levels and feather. No preview on the node; add LC Mask Refine after it if you want one.
"""

import torch

from . import lc_models
from .lc_matting import METHODS, get_device, refine_mask

BACKGROUNDS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.0, 0.69, 0.25),
    "blue": (0.0, 0.28, 1.0),
    "magenta": (1.0, 0.0, 1.0),
}

_model_cache = {}  # (path, size) -> model; one kept at a time (ComfyUI offloads it after use)


def _load_bg_model(path, size):
    key = (path, size)
    if key in _model_cache:
        return _model_cache[key]

    from comfy.bg_removal_model import load

    model = load(path)
    if model is None:
        raise RuntimeError(
            "[LC Remove Background] This file is not a BiRefNet-format model that ComfyUI can load: "
            f"{path}"
        )
    model.image_size = size
    _model_cache.clear()
    _model_cache[key] = model
    return model


class LCRemBG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to cut the subject out of."}),
                "model": (lc_models.bgremoval_choices(), {
                    "tooltip": (
                        "Background removal model. The license is in the label.\n"
                        "Entries marked 'Download' are fetched the first time you run."
                    ),
                }),
                "refine": (METHODS, {
                    "default": "none",
                    "tooltip": (
                        "none: use the model's mask as is.\n"
                        "guided_filter: fast, snaps the edge to the image.\n"
                        "vitmatte: matting model for hair and fine detail."
                    ),
                }),
                "vitmatte_model": (lc_models.vitmatte_choices(), {
                    "tooltip": "Only used when refine is vitmatte.",
                }),
                "grow": ("FLOAT", {
                    "default": 0.0, "min": -512.0, "max": 512.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Pixels to grow (positive) or shrink (negative) the mask before refining.",
                }),
                "edge_erode": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Refine trimap: how far the sure-foreground area is pulled in from the mask edge. Smaller = tighter.",
                }),
                "edge_dilate": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Refine trimap: how far the unknown band reaches past the mask edge. Smaller = tighter.",
                }),
                "black_point": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.999, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or below this become 0. Raise it to clear faint background haze.",
                }),
                "white_point": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or above this become 1. Lower it to solidify the subject.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Final edge softness in pixels.",
                }),
                "max_megapixels": ("FLOAT", {
                    "default": 2.0, "min": 0.25, "max": 16.0, "step": 0.25,
                    "tooltip": "vitmatte only. Larger images are matted at this size, then scaled back.",
                }),
                "background": (list(BACKGROUNDS.keys()), {
                    "default": "black",
                    "tooltip": "Color behind the subject in the on_background output.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("cutout", "mask", "on_background")
    FUNCTION = "remove"
    CATEGORY = "LC MaskMaker/mask"
    DESCRIPTION = (
        "Remove the background with BiRefNet. Outputs an RGBA cutout, the mask, and the subject on a "
        "solid color. Optional guided filter or VITMatte refine."
    )

    def remove(self, image, model, refine, vitmatte_model, grow, edge_erode, edge_dilate, black_point,
               white_point, feather, max_megapixels, background):
        path, size = lc_models.resolve_bgremoval(model)
        folder = lc_models.resolve_vitmatte(vitmatte_model) if refine == "vitmatte" else None
        bg_model = _load_bg_model(path, size)

        rgb = image[..., :3]
        raw = bg_model.encode_image(rgb).float().cpu().clamp(0, 1)  # (B,H,W)
        b, h, w = raw.shape

        dev = get_device()
        masks = []
        for i in range(b):
            img = rgb[i].permute(2, 0, 1).unsqueeze(0).to(dev).float()
            m = raw[i].unsqueeze(0).unsqueeze(0).to(dev)
            m, _tri = refine_mask(img, m, refine, folder, grow, 0, 0.75, edge_erode, edge_dilate, 0.001, 0.5,
                                 black_point, white_point, feather, max_megapixels, dev)
            masks.append(m[0, 0].cpu())
        mask = torch.stack(masks, 0)  # (B,H,W)

        rgb_cpu = rgb.cpu().float()
        cutout = torch.cat([rgb_cpu, mask.unsqueeze(-1)], dim=-1)
        color = torch.tensor(BACKGROUNDS[background], dtype=torch.float32).view(1, 1, 1, 3)
        on_bg = rgb_cpu * mask.unsqueeze(-1) + color * (1.0 - mask.unsqueeze(-1))

        return (cutout, mask, on_bg)


NODE_CLASS_MAPPINGS = {
    "LCRemBG": LCRemBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCRemBG": "LC Remove Background ✂️",
}
