"""
LC Smart Inpaint (+ pipe version)
---------------------------------
Finds things by name with SAM 3 ("hands, face") and redraws them at full detail:

  every object gets its own crop, with some context around it, scaled up to inpaint_resolution,
  sampled with a noise mask, scaled back down and blended in.

Objects whose crops overlap are done together in one crop. Works with 4-D and 5-D latents
(SDXL, Flux, Krea 2, Anima, Qwen, etc.). No negative wired = a zeroed positive, which is what
models that run at cfg 1 expect.
"""

import math
import re

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from nodes import PreviewImage

from . import lc_models, lc_sam3
from .lc_refine_core import blur, grow

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PIPE_TYPE = "LC_PIPE"
SNAP = 32  # crop sizes are multiples of this, safe for 8x and 16x VAEs with a 2x2 patch
OUTLINE = (0.25, 0.9, 1.0)

DESC_SMART = (
    "Finds what you describe (hands, face, eyes, etc.) and redraws it at full detail. "
    "Each hand or face gets its own crop, scaled up to inpaint_resolution, redrawn, then blended back in. "
    "Nothing found? The image passes through untouched and the node tells you.\n"
    "No negative wired: fine for Flux, Krea 2, Anima and other cfg 1 models. Wire one for SDXL."
)

TIP = {
    "prompt": "What to inpaint, as simple words separated by commas: hands, face.\n"
              "Every match is found (both hands, every face). Add :N to cap it, e.g. face:1.",
    "sam3_model": "SAM 3 model that does the finding. Pick the SAM 3.1 download, or use your own sam3.pt / sam3.safetensors in models/sam3.",
    "threshold": "How sure SAM 3 has to be. Lower finds more (and more wrong things), higher is pickier. 0.5 is a good start.\n"
                 "Colors are only a hint: 'red dress' can still pick a green dress. Things that are not there are not found.",
    "grow": "Pixels to grow (positive) or shrink (negative) the mask before inpainting. Growing a little covers the edges.",
    "feather": "Softness of the mask edge in pixels, so the redrawn area blends into the rest. 0 = hard edge.",
    "blend": "How much of the redraw is pasted in. 1 = all of it, 0.5 = half redraw, half original. Lower tames a redraw that went too far.",
    "padding": "Pixels of the surrounding image included with each crop, so the model sees the context.",
    "inpaint_resolution": "Each crop is scaled so its long side is this size before it is redrawn. Higher = more detail, slower.",
    "denoise": "How much is redrawn. Low keeps the shape and fixes details, high redraws from scratch.",
    "seed": "Seed for the redraw.",
    "steps": "Sampling steps.",
    "cfg": "CFG. 1 for Flux, Krea 2, Anima and other cfg 1 models; around 5 to 7 for SDXL.",
    "sampler_name": "Sampler.",
    "scheduler": "Scheduler.",
    "positive": "What the inpainted area should look like.",
    "negative": "Optional. Leave empty for cfg 1 models; wire it for SDXL and other models that use a negative.",
    "pipe": "LC pipe: model, VAE, seed, steps, cfg, sampler and scheduler come from here.",
    "model": "Optional. Wire a model here to use it instead of the pipe's (with LoRAs on it, or a different model).",
    "vae": "Optional. Wire a VAE here to use it instead of the pipe's.",
}


def _tip(name, **kw):
    return dict(kw, tooltip=TIP[name])


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def _find_inputs():
    return {
        "prompt": ("STRING", _tip("prompt", default="hands", multiline=True)),
        "sam3_model": (lc_models.sam3_choices(), _tip("sam3_model")),
        "threshold": ("FLOAT", _tip("threshold", default=0.5, min=0.05, max=0.95, step=0.01)),
    }


def _mask_inputs():
    return {
        "grow": ("FLOAT", _tip("grow", default=4.0, min=-64.0, max=256.0, step=0.5)),
        "feather": ("FLOAT", _tip("feather", default=8.0, min=0.0, max=128.0, step=0.5)),
        "blend": ("FLOAT", _tip("blend", default=0.8, min=0.0, max=1.0, step=0.01)),
        "padding": ("INT", _tip("padding", default=64, min=0, max=1024, step=8)),
        "inpaint_resolution": ("INT", _tip("inpaint_resolution", default=1024, min=256, max=4096, step=64)),
    }


def _sampler_inputs():
    return {
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, _tip("sampler_name")),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, _tip("scheduler")),
        "steps": ("INT", _tip("steps", default=20, min=1, max=150)),
        "cfg": ("FLOAT", _tip("cfg", default=1.0, min=0.0, max=30.0, step=0.1, round=0.01)),
        "denoise": ("FLOAT", _tip("denoise", default=0.3, min=0.0, max=1.0, step=0.01)),
        "seed": ("INT", _tip("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True)),
    }


def _denoise_input():
    return {"denoise": _sampler_inputs()["denoise"]}


def _cond_inputs():
    return ("CONDITIONING", _tip("positive")), ("CONDITIONING", _tip("negative"))


def _optional(neg=True, pos=False):
    out = {}
    if pos:  # pipe version: model / VAE wired here win over the pipe (a LoRA'd or swapped model)
        out["model"] = ("MODEL", _tip("model"))
        out["vae"] = ("VAE", _tip("vae"))
        out["positive"] = ("CONDITIONING", _tip("positive"))
    if neg:
        out["negative"] = ("CONDITIONING", _tip("negative"))
    return out


def _words(prompt):
    return [p.strip() for p in re.split(r"[,\n]+", prompt or "") if p.strip()]


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
def _zero_out(conditioning):
    out = []
    for t, d in conditioning:
        d = d.copy()
        for k in ("pooled_output", "conditioning_lyrics"):
            if d.get(k) is not None:
                d[k] = torch.zeros_like(d[k])
        out.append([torch.zeros_like(t), d])
    return out


def _box(m):
    ys, xs = torch.where(m > 0.5)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _expand(b, margin, w, h):
    return [max(0, b[0] - margin), max(0, b[1] - margin), min(w, b[2] + margin), min(h, b[3] + margin)]


def _touch(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _components(mask_hw):
    """(labels, areas) of the separate areas of a mask, or None without OpenCV."""
    if cv2 is None:
        return None
    binary = (mask_hw > 0.5).cpu().numpy().astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return torch.from_numpy(labels), stats[:, cv2.CC_STAT_AREA]


def _min_area(w, h):
    return max(64, int(w * h * 0.0002))


def _clean(mask_hw):
    """Drop stray specks from one object's mask: keep areas at least 5% the size of its biggest one."""
    comp = _components(mask_hw)
    if comp is None:
        return mask_hw
    labels, areas = comp
    if len(areas) <= 2:
        return mask_hw
    keep_min = max(_min_area(*mask_hw.shape[::-1]), 0.05 * areas[1:].max())
    keep = [i for i in range(1, len(areas)) if areas[i] >= keep_min]
    return mask_hw * torch.isin(labels, torch.tensor(keep)).to(mask_hw.dtype)


def _regions(masks, grow_px, feather, padding, w, h):
    """Grow every object, merge objects that overlap each other, then add the padding.
    Returns [(mask (H,W), crop box)]. Crops of separate objects may overlap: they are done one after another."""
    edge = int(math.ceil(feather * 2))
    items = []
    for m in masks:
        g = grow(m.float().unsqueeze(0).unsqueeze(0), grow_px)[0, 0]
        b = _box(g)
        if b is not None:
            items.append([g, _expand(b, edge, w, h)])
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _touch(items[i][1], items[j][1]):
                    a, b = items[i], items.pop(j)
                    a[0] = torch.maximum(a[0], b[0])
                    a[1] = [min(a[1][0], b[1][0]), min(a[1][1], b[1][1]), max(a[1][2], b[1][2]), max(a[1][3], b[1][3])]
                    merged = True
                    break
            if merged:
                break
    return [(m, _expand(b, int(padding), w, h)) for m, b in items]


def _target(cw, ch, res):
    s = res / float(max(cw, ch))
    tw = max(SNAP, int(round(cw * s / SNAP)) * SNAP)
    th = max(SNAP, int(round(ch * s / SNAP)) * SNAP)
    return tw, th


def _inpaint_region(frame, region, box, s):
    """frame (1,H,W,3); region (H,W). Returns (frame with the region redrawn, soft blend mask (H,W))."""
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    soft = blur(region[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0), s["feather"])[0, 0].clamp(0, 1)
    crop = frame[:, y1:y2, x1:x2, :3]
    tw, th = _target(cw, ch, s["res"])
    up = comfy.utils.common_upscale(crop.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
    noise_mask = F.interpolate(soft[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0]  # (1,th,tw)

    model, vae = s["model"], s["vae"]
    latent = vae.encode(up)
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    noise = comfy.sample.prepare_noise(latent, s["seed"])
    callback = latent_preview.prepare_callback(model, s["steps"])
    samples = comfy.sample.sample(
        model, noise, s["steps"], s["cfg"], s["sampler_name"], s["scheduler"], s["positive"], s["negative"], latent,
        denoise=s["denoise"], noise_mask=noise_mask, callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=s["seed"],
    )
    patch = vae.decode(samples)
    if patch.ndim == 5:  # 5-D latents decode to (B, T, H, W, C)
        patch = patch.reshape(-1, *patch.shape[-3:])
    patch = patch[:1, ..., :3].float().cpu()
    down = comfy.utils.common_upscale(patch.movedim(-1, 1), cw, ch, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)

    out = frame.clone()
    a = (soft * s["blend"])[None, ..., None]
    out[:, y1:y2, x1:x2, :3] = crop * (1.0 - a) + down * a
    full = torch.zeros(frame.shape[1], frame.shape[2])
    full[y1:y2, x1:x2] = soft
    return out, full


def _settings(kw, model, vae, positive, negative, sampler_name, scheduler, steps, cfg, seed):
    """kw: the node's own widgets (grow, feather, blend, padding, inpaint_resolution, denoise)."""
    if negative is None:
        negative = _zero_out(positive)
    return dict(model=model, vae=vae, positive=positive, negative=negative, sampler_name=sampler_name,
                scheduler=scheduler, steps=int(steps), cfg=float(cfg), denoise=float(kw["denoise"]), seed=int(seed),
                grow=float(kw["grow"]), feather=float(kw["feather"]), padding=int(kw["padding"]),
                res=int(kw["inpaint_resolution"]), blend=float(kw["blend"]))


def _run(image, masks_per_frame, s):
    """masks_per_frame: list (one per frame) of lists of (H,W) masks. Returns (image, mask, done count)."""
    b, h, w, _ = image.shape
    frames, fmasks, done = [], [], 0
    for i in range(b):
        frame = image[i:i + 1].float().cpu()
        fmask = torch.zeros(h, w)
        for region, box in _regions(masks_per_frame[i], s["grow"], s["feather"], s["padding"], w, h):
            comfy.model_management.throw_exception_if_processing_interrupted()
            frame, soft = _inpaint_region(frame, region, box, s)
            fmask = torch.maximum(fmask, soft)
            done += 1
        frames.append(frame)
        fmasks.append(fmask)
    return torch.cat(frames, 0), torch.stack(fmasks, 0), done


def _preview(node, before, after, mask, note):
    """Before (with the inpainted areas outlined) and after, small, for the wipe on the node."""
    ui = {"lc_inpaint_note": [note]}
    try:
        h, w = before.shape[1], before.shape[2]
        s = min(1.0, 768.0 / max(h, w))
        size = (max(1, int(h * s)), max(1, int(w * s)))
        src = F.interpolate(before[:1, ..., :3].permute(0, 3, 1, 2).float(), size=size, mode="area")
        res = F.interpolate(after[:1, ..., :3].permute(0, 3, 1, 2).float(), size=size, mode="area")
        m = F.interpolate(mask[:1].unsqueeze(1).float(), size=size, mode="bilinear", align_corners=False)
        inside = (m > 0.1).float()
        edge = (F.max_pool2d(inside, 5, 1, 2) - inside).clamp(0, 1)
        color = torch.tensor(OUTLINE).view(1, 3, 1, 1)
        src = src * (1 - edge) + color * edge
        saved = node.save_images(torch.cat([src, res], 0).permute(0, 2, 3, 1), filename_prefix="lc_inpaint")
        ui["lc_inpaint"] = saved["ui"]["images"]
    except Exception as e:
        print(f"[LC MaskMaker] inpaint preview skipped: {e}")
    return ui


def _note(done, what):
    if done:
        return f"{done} area{'s' if done != 1 else ''} inpainted"
    return f"Nothing found for '{what}': image passed through"


# --------------------------------------------------------------------------
# pipe
# --------------------------------------------------------------------------
def _from_pipe(pipe, name, have=()):
    if not isinstance(pipe, dict):
        raise ValueError(f"[{name}] The pipe input is not an LC pipe.")

    def get(*keys):
        for k in keys:
            v = pipe.get(k)
            if v is not None and not (isinstance(v, str) and v == ""):
                return v
        return None

    vals = {
        "model": get("model_1"), "vae": get("vae_1"), "seed": get("seed"),
        "steps": get("detailer_steps", "total_steps"), "cfg": get("cfg_1"),
        "sampler_name": get("sampler_name"), "scheduler": get("scheduler"),
        "positive": get("positive"), "negative": get("negative"),
    }
    missing = [k for k in ("model", "vae", "seed", "steps", "cfg", "sampler_name", "scheduler") if vals[k] is None and k not in have]
    if missing:
        raise ValueError(f"[{name}] The pipe has no {', '.join(missing)}. Add it with LC Pipe In / LC Pipe Edit.")
    return vals


def _pipe_settings(pipe, kw, name):
    have = [k for k in ("model", "vae") if kw.get(k) is not None]
    p = _from_pipe(pipe, name, have)
    for k in have:
        p[k] = kw[k]
    positive = kw.get("positive") if kw.get("positive") is not None else p["positive"]
    if positive is None:
        raise ValueError(f"[{name}] No positive: wire one, or put it in the pipe.")
    negative = kw.get("negative") if kw.get("negative") is not None else p["negative"]
    return _settings(kw, p["model"], p["vae"], positive, negative, p["sampler_name"], p["scheduler"], p["steps"],
                     p["cfg"], p["seed"])


def _direct_settings(kw):
    return _settings(kw, kw["model"], kw["vae"], kw["positive"], kw.get("negative"), kw["sampler_name"],
                     kw["scheduler"], kw["steps"], kw["cfg"], kw["seed"])


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
class _Base(PreviewImage):
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "LC MaskMaker/inpaint"
    OUTPUT_NODE = True
    FUNCTION = "run"

    def __init__(self):
        super().__init__()

    def _smart(self, image, kw, s):
        prompt = kw["prompt"]
        words = _words(prompt)
        if not words:
            raise ValueError("[LC Smart Inpaint] The prompt is empty. Type what to inpaint, e.g. hands.")
        path = lc_models.resolve_sam3(kw["sam3_model"])
        masks = [[_clean(m) for m in lc_sam3.objects(image[i:i + 1], words, path, kw["threshold"])]
                 for i in range(image.shape[0])]
        out, mask, done = _run(image, masks, s)
        return out, mask, _note(done, prompt.strip())

class LCSmartInpaint(_Base):
    DESCRIPTION = DESC_SMART

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), "model": ("MODEL",), "vae": ("VAE",), "positive": _cond_inputs()[0],
                **_find_inputs(), **_mask_inputs(), **_sampler_inputs(),
            },
            "optional": _optional(),
        }

    def run(self, image, **kw):
        s = _direct_settings(kw)
        out, mask, note = self._smart(image, kw, s)
        return {"ui": _preview(self, image, out, mask, note), "result": (out, mask)}


class LCSmartInpaintPipe(_Base):
    DESCRIPTION = DESC_SMART
    RETURN_TYPES = (PIPE_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (PIPE_TYPE, {"tooltip": TIP["pipe"]}), "image": ("IMAGE",),
                **_find_inputs(), **_mask_inputs(), **_denoise_input(),
            },
            "optional": _optional(pos=True),
        }

    def run(self, pipe, image, **kw):
        s = _pipe_settings(pipe, kw, "LC Smart Inpaint (pipe)")
        out, mask, note = self._smart(image, kw, s)
        return {"ui": _preview(self, image, out, mask, note), "result": (pipe, out, mask)}


NODE_CLASS_MAPPINGS = {
    "LCSmartInpaint": LCSmartInpaint,
    "LCSmartInpaintPipe": LCSmartInpaintPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSmartInpaint": "LC Smart Inpaint 🩹",
    "LCSmartInpaintPipe": "LC Smart Inpaint (pipe) 🩹",
}
