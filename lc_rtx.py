"""
NVIDIA RTX Video Super Resolution as a resize method for LC nodes.

rtx_resize(bchw, width, height) enlarges with RTX VSR when it can and falls back to lanczos when it can't
(no nvidia-vfx, no CUDA, or the target is smaller than the source: RTX VSR only upscales). Works with both the
old nvidia-vfx API (0.1.x: nvvfx.effects.QualityLevel) and the newer one (nvvfx.QualityLevel).
"""

from __future__ import annotations

import torch

import comfy.model_management
import comfy.utils

RTX_METHOD = "nvidia_rtx_vsr"
MAX_BATCH_PIXELS = 1024 * 1024 * 16
_WARNED = set()


def _warn(key: str, msg: str):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[LC MaskMaker] {msg}")


def _nvvfx():
    try:
        import nvvfx
    except Exception as e:
        _warn("import", f"RTX upscale unavailable ({e}). Install nvidia-vfx on an RTX card. Using lanczos.")
        return None, None
    quality = getattr(nvvfx, "QualityLevel", None) or getattr(getattr(nvvfx, "effects", None), "QualityLevel", None)
    if quality is None:
        _warn("api", "RTX upscale: unknown nvidia-vfx version. Using lanczos.")
        return None, None
    return nvvfx, quality


def available() -> bool:
    nv, _ = _nvvfx()
    return nv is not None and comfy.model_management.get_torch_device().type == "cuda"


def rtx_resize(samples: torch.Tensor, width: int, height: int, quality: str = "ULTRA") -> torch.Tensor:
    """BCHW float 0..1 (3 or 4 channels) -> BCHW at width x height. Alpha / extra channels are resized bilinear."""
    b, c, h, w = samples.shape
    if width < w or height < h:
        _warn("down", "RTX upscale only enlarges; smaller targets use lanczos.")
        return comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    nv, levels = _nvvfx()
    device = comfy.model_management.get_torch_device()
    if nv is None or device.type != "cuda" or c < 3:
        return comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    # RTX VSR wants sizes in steps of 8: run it at the nearest multiple of 8, then snap to the exact size
    ow, oh = max(8, round(width / 8) * 8), max(8, round(height / 8) * 8)
    index = device.index if device.index is not None else torch.cuda.current_device()
    rgb = samples[:, :3].to(device=device, dtype=torch.float32).clamp(0, 1)
    out = torch.empty((b, 3, oh, ow), device=device, dtype=torch.float32)
    try:
        level = levels[quality]
        try:
            ctx = nv.VideoSuperRes(quality=level, device=index)
        except TypeError:
            ctx = nv.VideoSuperRes(level)
        with torch.cuda.device(device), ctx as effect:
            effect.output_width, effect.output_height = ow, oh
            effect.load()
            for k in range(b):
                out[k] = torch.from_dlpack(effect.run(rgb[k].contiguous()).image).clone()
    except Exception as e:
        _warn("run", f"RTX upscale failed ({e}). Using lanczos.")
        return comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    if (ow, oh) != (width, height):
        out = comfy.utils.common_upscale(out, width, height, "bicubic", "disabled")
    out = out.clamp(0, 1).to(samples.device, samples.dtype)
    if c > 3:
        extra = torch.nn.functional.interpolate(samples[:, 3:].float(), size=(height, width), mode="bilinear",
                                                align_corners=False).to(samples.dtype)
        out = torch.cat([out, extra], dim=1)
    return out


def resize(samples: torch.Tensor, width: int, height: int, method: str, crop: str = "disabled") -> torch.Tensor:
    """Drop-in for comfy.utils.common_upscale that also understands nvidia_rtx_vsr."""
    if method != RTX_METHOD:
        return comfy.utils.common_upscale(samples, width, height, method, crop)
    if crop == "center":  # same crop as common_upscale
        ow, oh = samples.shape[-1], samples.shape[-2]
        old_aspect, new_aspect = ow / oh, width / height
        if old_aspect > new_aspect:
            x = round((ow - ow * (new_aspect / old_aspect)) / 2)
            samples = samples[..., :, x:ow - x]
        elif old_aspect < new_aspect:
            y = round((oh - oh * (old_aspect / new_aspect)) / 2)
            samples = samples[..., y:oh - y, :]
    return rtx_resize(samples, width, height)
