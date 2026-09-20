"""
LC Depth Anything V2
--------------------
Relative depth from a single image with Depth Anything V2. Reads the original weight files
(.pth / .safetensors), so a copy you already have in models/depthanything works as is.
The model code is vendored (Apache-2.0, see THIRD_PARTY.md); no extra Python package needed.
"""

import math

import torch
import torch.nn.functional as F

from . import lc_models
from .lc_matting import get_device, no_cudnn_autotune

CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
_cache = {}  # path -> model on cpu; one kept at a time


def _load(path, encoder):
    if path in _cache:
        return _cache[path]
    from .vendor.depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder=encoder, **CONFIGS[encoder])
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[len("module."):] if k.startswith("module.") else k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    _cache.clear()
    _cache[path] = model
    return model


def _model_size(h, w, target):
    """Original Depth Anything V2 sizing: shorter side reaches `target`, both sides a multiple of 14."""
    scale = max(target / h, target / w)

    def fit(x):
        v = int(round(x * scale / 14.0)) * 14
        return v if v >= target else int(math.ceil(x * scale / 14.0)) * 14

    return fit(h), fit(w)


@torch.no_grad()
def estimate_depth(model, frame_hwc, resolution, dev):
    """frame (H,W,3) 0-1 -> (H,W) depth, larger = nearer (not yet normalized)."""
    h, w = frame_hwc.shape[:2]
    mh, mw = _model_size(h, w, resolution)
    x = frame_hwc.permute(2, 0, 1).unsqueeze(0).to(dev).float()
    x = F.interpolate(x, size=(mh, mw), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    depth = model((x - mean) / std, 1.0)  # (1,mh,mw)
    depth = F.interpolate(depth[:, None], size=(h, w), mode="bilinear", align_corners=True)[0, 0]
    return depth.float().cpu()


class LCDepthAnythingV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to estimate depth for."}),
                "model": (lc_models.depth_choices(), {
                    "tooltip": (
                        "Depth Anything V2 weights. The license is in the label: Small is Apache-2.0, "
                        "Base / Large / Giant are CC-BY-NC-4.0 (non-commercial).\n"
                        "Entries marked 'Download' are fetched the first time you run."
                    ),
                }),
                "resolution": ("INT", {
                    "default": 518, "min": 224, "max": 2044, "step": 14,
                    "tooltip": "Size the shorter side is processed at (rounded to a multiple of 14). 518 is what the model was trained at; higher can add detail on big images.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: bright = near (what Depth Anything and depth ControlNets use). On: bright = far.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth", "depth_mask")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/depth"
    DESCRIPTION = (
        "Depth Anything V2 depth map, normalized 0-1 per image. Bright = near unless invert is on. "
        "Outputs a 3-channel image and a MASK of the same map."
    )

    def run(self, image, model, resolution, invert):
        path, encoder = lc_models.resolve_depth(model)
        net = _load(path, encoder)
        dev = get_device()
        net.to(dev)
        maps = []
        try:
            with no_cudnn_autotune():
                for i in range(image.shape[0]):
                    d = estimate_depth(net, image[i, ..., :3], resolution, dev)
                    lo, hi = d.min(), d.max()
                    d = (d - lo) / (hi - lo).clamp_min(1e-8)
                    maps.append(1.0 - d if invert else d)
        finally:
            net.to("cpu")
        depth = torch.stack(maps, 0)  # (B,H,W)
        return (depth.unsqueeze(-1).repeat(1, 1, 1, 3), depth)


NODE_CLASS_MAPPINGS = {
    "LCDepthAnythingV2": LCDepthAnythingV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCDepthAnythingV2": "LC Depth Anything V2 🌊",
}
