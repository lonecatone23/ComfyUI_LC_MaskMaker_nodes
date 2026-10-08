"""
LC VOSR2 Upscale: one-step VOSR 2.0 super-resolution, built in (no other node pack needed).

VOSR 2.0 (Rongyuan Wu et al., Apache-2.0) is a dedicated upscale model: it sharpens and adds fine detail to the
picture it is given instead of redrawing it with your image model, so faces and hands the detailers fixed come
through as they are. The code lives in vendor/vosr2 (from ComfyUI-VOSR2, Apache-2.0, see vendor/vosr2/NOTICE); the weights
download from huggingface.co/CSWRY/VOSR on first run into models/vosr2.
"""

import torch
import torch.nn.functional as F

from . import lc_pipe_keys
from .vendor.vosr2 import loader as _vl
from .vendor.vosr2.inference import run_vosr2

PIPE_TYPE = "LC_PIPE"
_BUNDLE = {}  # dtype -> loaded VOSR2Model, kept for the session


def _bundle(dtype):
    if dtype not in _BUNDLE:
        _BUNDLE.clear()
        _BUNDLE[dtype] = _vl.load_vosr2(_vl.KNOWN_MODEL, dtype)
    return _BUNDLE[dtype]


# Tested on the bench: one pass up to 4x is clean; a single 6x / 8x pass breaks the skin up into patches, while
# 3x + 2x and 4x + 2x stay clean.
SCALES = {"1.5x": (1.5,), "2x": (2,), "3x": (3,), "4x": (4,), "6x (3x + 2x)": (3, 2), "8x (4x + 2x)": (4, 2)}


def upscale(image, scale, seed, color_alignment, tile_size, vae_tile_size, dtype="default"):
    """image (B,H,W,3 or 4) in [0,1] -> upscaled by SCALES[scale], one VOSR2 pass per factor."""
    overlap = max(16, tile_size // 8)
    vae_overlap = max(16, vae_tile_size // 8)
    out = image[..., :3].float()
    for k in SCALES[scale]:
        out = run_vosr2(_bundle(dtype), out, k, int(seed), color_alignment,
                        int(tile_size), overlap, int(vae_tile_size), vae_overlap).float().cpu()
    return out.clamp(0, 1)


def _scale_mask(mask, image_out):
    """(B,h,w) mask -> the upscaled picture's size, soft edges kept (bilinear)."""
    _, H, W, _ = image_out.shape
    m = mask if mask.ndim == 3 else mask[None]
    m = F.interpolate(m[:, None].float().cpu(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return m.clamp(0, 1)


PROTECT_TIP = ("The faces and hands the detailers redrew, at the new size. Wire it into LC Skin Texture / LC Skin Upscale "
               "protect_mask so they do not add skin detail on top of the detailers' work.")


def _inputs():
    return {
        "upscale_by": (list(SCALES), {"default": "2x",
                       "tooltip": "Final size. 1.5x to 4x are one pass (1.5x is about a third faster than 2x). 6x and 8x run "
                                  "two passes (3x + 2x, 4x + 2x): one big pass breaks the skin up into patches."}),
        "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed",
                         "tooltip": "VOSR2 is one step from noise: the seed only changes the finest grain."}),
        "color_alignment": (["wavelet", "adain", "none"], {"default": "wavelet",
                            "tooltip": "Keeps the result's colours on the input's. wavelet = best match."}),
        "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                              "tooltip": "Pixel tiles for the upscale model. VOSR2 was trained at 512: keep it there."}),
        "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64,
                                  "tooltip": "Pixel tiles for its VAE. Lower = less VRAM, a little slower."}),
    }


class LCVOSRUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **_inputs()},
                "optional": {"protect_mask": ("MASK", {"tooltip": "Optional. LC Smart Detailer's mask (several combined "
                                                                  "with a max). Comes out at the new size as 'protected'."})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "protected")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/upscale"
    DESCRIPTION = ("VOSR 2.0 one-step upscale, built in. Sharpens and adds fine detail without redrawing the picture "
                   "with your model, so the detailers' work comes through. Run it after the detailers. Weights "
                   "download on first run (CSWRY/VOSR, Apache-2.0).")

    def run(self, image, upscale_by, seed, color_alignment, tile_size, vae_tile_size, protect_mask=None):
        out = upscale(image, upscale_by, seed, color_alignment, tile_size, vae_tile_size)
        prot = _scale_mask(protect_mask, out) if protect_mask is not None else torch.zeros(out.shape[0], out.shape[1], out.shape[2])
        return (out, prot)


class LCVOSRUpscalePipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pipe": (PIPE_TYPE, {"tooltip": "LC pipe: the picture comes from its image."}), **_inputs()},
            "optional": {"image": ("IMAGE", {"tooltip": "Optional. Empty = the pipe's image. Wire one when a node "
                                                        "without a pipe changed the picture in between."})},
        }

    RETURN_TYPES = (PIPE_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "image", "protected")
    OUTPUT_TOOLTIPS = ("The pipe with the upscaled picture, and the detailers' areas resized to it.",
                       "The upscaled picture.", PROTECT_TIP)
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/upscale"
    DESCRIPTION = ("LC VOSR2 Upscale on the LC pipe: upscales the pipe's picture (what the detailers left in it) and "
                   "puts the result back in the pipe, with the detailers' areas resized to it ('protected' output).")

    def run(self, pipe, upscale_by, seed, color_alignment, tile_size, vae_tile_size, image=None):
        if not isinstance(pipe, dict):
            raise ValueError("[LC VOSR2 Upscale (pipe)] The pipe input is not an LC pipe.")
        src = image if image is not None else pipe.get("image")
        if src is None:
            raise ValueError("[LC VOSR2 Upscale (pipe)] No image: wire one, or put it in the pipe.")
        protect = lc_pipe_keys.protect_for(pipe, src)  # only if it was made for this very picture
        out = upscale(src, upscale_by, seed, color_alignment, tile_size, vae_tile_size)
        prot = _scale_mask(protect, out) if protect is not None else None
        # a copy (the incoming pipe is never edited in place), with the protect mask resized and tagged to the new
        # picture; VOSR2 does not move anything, so it still lines up. SAM finds are per pixel of the small picture: dropped.
        new = lc_pipe_keys.updated(pipe, out, protect=prot)
        new.pop(lc_pipe_keys.FOUND, None)
        if prot is None:
            new.pop(lc_pipe_keys.PROTECT, None)
            prot = torch.zeros(out.shape[0], out.shape[1], out.shape[2])
        return (new, out, prot)


NODE_CLASS_MAPPINGS = {"LCVOSRUpscale": LCVOSRUpscale, "LCVOSRUpscalePipe": LCVOSRUpscalePipe}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVOSRUpscale": "LC VOSR2 Upscale 🧩", "LCVOSRUpscalePipe": "LC VOSR2 Upscale (pipe) 🧩"}
