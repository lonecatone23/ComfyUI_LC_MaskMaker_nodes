"""
LC Normal Map (BAE)
-------------------
Surface normals from a single image with Bae et al.'s "Estimating and Exploiting the Aleatoric
Uncertainty in Surface Normal Estimation" network (the same one ControlNet's normal_bae uses).
The network code is vendored (MIT, see THIRD_PARTY.md). It needs the `timm` package.
"""

import types

import torch
import torch.nn.functional as F

from . import lc_models
from .lc_matting import get_device, no_cudnn_autotune

_cache = {}  # path -> model on cpu; one kept at a time


def _load(path):
    if path in _cache:
        return _cache[path]
    try:
        import timm  # noqa: F401
    except ImportError as e:
        raise RuntimeError("[LC Normal Map (BAE)] This node needs the 'timm' package (python -m pip install timm).") from e
    from .vendor.normalbae.NNET import NNET

    args = types.SimpleNamespace(mode="client", architecture="BN", pretrained="scannet",
                                 sampling_ratio=0.4, importance_ratio=0.7)
    model = NNET(args)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({(k[len("module."):] if k.startswith("module.") else k): v for k, v in ckpt.items()})
    model.eval()
    _cache.clear()
    _cache[path] = model
    return model


@torch.no_grad()
def estimate_normals(model, frame_hwc, resolution, dev):
    """frame (H,W,3) 0-1 -> (H,W,3) unit normals in -1..1 (RGB = XYZ as the model outputs them)."""
    h, w = frame_hwc.shape[:2]
    k = float(resolution) / float(min(h, w))
    nh, nw = max(8, int(round(h * k))), max(8, int(round(w * k)))
    x = frame_hwc.permute(2, 0, 1).unsqueeze(0).to(dev).float()
    x = F.interpolate(x, size=(nh, nw), mode="bicubic" if k > 1 else "area", **({"align_corners": False} if k > 1 else {})).clamp(0, 1)
    ph, pw = (-nh) % 64, (-nw) % 64
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = model((x - mean) / std)[0][-1][:, :3]  # (1,3,H',W')
    out = out[:, :, :nh, :nw]
    out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return out[0].permute(1, 2, 0).float().cpu()


def _native_blur(image, blur_radius, sigma):
    """ComfyUI's own Blur Image (ImageBlur) node, so the result matches chaining that node after this one."""
    from comfy_extras.nodes_post_processing import Blur

    out = Blur.execute(image, int(blur_radius), float(sigma))
    return out.args[0] if hasattr(out, "args") else out[0]


class LCNormalBAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to estimate surface normals for."}),
                "model": (lc_models.bae_choices(), {
                    "tooltip": (
                        "BAE weights (scannet.pt). Trained on ScanNet, whose terms are not commercial-friendly, "
                        "and the weights' Hugging Face card just says 'other'. Verify before commercial use.\n"
                        "Uses the copy you already have (models/normalbae or comfyui_controlnet_aux), "
                        "or downloads it the first time you run."
                    ),
                }),
                "resolution": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Size the shorter side is processed at. 512 is what ControlNet uses; higher is slower and can add detail.",
                }),
                "flip_y": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip the green channel (OpenGL vs DirectX style normal maps).",
                }),
                "blur_radius": ("INT", {
                    "default": 0, "min": 0, "max": 31, "step": 1,
                    "tooltip": (
                        "Blur the finished normal map with ComfyUI's Blur Image. The radius is in pixels, the blur kernel is "
                        "2 x radius + 1 wide. Smooths the fine noise BAE leaves on flat surfaces. 0 = no blur."
                    ),
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "How evenly the blur spreads across its radius (same meaning as in Blur Image). Higher = stronger, "
                        "closer to a plain box blur. Lower = gentler, weighted toward the center. Only used when blur_radius is above 0."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        # Workflows saved before 0.12 hold a machine-specific label; any scannet label still runs.
        return True if "scannet" in str(model).lower() else f"Unknown BAE model: {model}"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/depth"
    DESCRIPTION = "BAE surface normal map (RGB = XYZ), the same style ControlNet's normal_bae makes. Optional blur (blur_radius, sigma) uses ComfyUI's Blur Image."

    def run(self, image, model, resolution, flip_y, blur_radius=0, sigma=1.0):
        path = lc_models.resolve_bae(model)
        net = _load(path)
        dev = get_device()
        net.to(dev)
        outs = []
        try:
            with no_cudnn_autotune():
                for i in range(image.shape[0]):
                    n = estimate_normals(net, image[i, ..., :3], resolution, dev)
                    if flip_y:
                        n = torch.stack([n[..., 0], -n[..., 1], n[..., 2]], -1)
                    outs.append(((n + 1.0) * 0.5).clamp(0, 1))
        finally:
            net.to("cpu")
        result = torch.stack(outs, 0)
        if blur_radius > 0:
            result = _native_blur(result, blur_radius, sigma).clamp(0, 1)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "LCNormalBAE": LCNormalBAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCNormalBAE": "LC Normal Map (BAE) 🗺️",
}
