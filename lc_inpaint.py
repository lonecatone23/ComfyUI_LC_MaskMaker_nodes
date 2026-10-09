"""
LC Smart Detailer (+ pipe version)
----------------------------------
(node ID LCSmartInpaint / LCSmartInpaintPipe, kept so saved workflows still load)
A SEGS / Impact FaceDetailer replacement: finds things by name with SAM 3 ("hands, face") and redraws them at full detail:

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

from . import lc_models, lc_pipe_keys, lc_sam3
from .lc_refine_core import blur, grow

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PIPE_TYPE = "LC_PIPE"
SNAP = 32  # crop sizes are multiples of this, safe for 8x and 16x VAEs with a 2x2 patch
SEAM_CAP = 0.08  # the seam colour fix never shifts a channel by more than this


def _ring(feather):
    """How far the repainted area reaches past the pasted area (px). The model redraws that ring too, so the paste
    edge lands on pixels it already made consistent, and the ring's known colours measure any colour drift."""
    return max(8, int(round(float(feather) * 1.5)))
OUTLINE = (0.25, 0.9, 1.0)

DESC_SMART = (
    "Finds what you describe (hands, face, eyes, etc.) and redraws it at full detail. "
    "Each hand or face gets its own crop, scaled up to inpaint_resolution, redrawn, then blended back in. "
    "Nothing found? The image passes through untouched and the node tells you.\n"
    "No negative wired: fine for Flux, Krea 2, Anima and other cfg 1 models. Wire one for SDXL."
)

TIP = {
    "prompt": "What to detail, as simple words separated by commas: hands, face.\n"
              "Every match is found (both hands, every face). Add :N to cap it, e.g. face:1.",
    "sam3_model": "SAM 3 model that does the finding. Pick the SAM 3.1 download, or use your own sam3.pt / sam3.safetensors in models/sam3.",
    "threshold": "How sure SAM 3 has to be. Lower finds more (and more wrong things), higher is pickier. 0.5 is a good start.\n"
                 "Colors are only a hint: 'red dress' can still pick a green dress. Things that are not there are not found.",
    "grow": "Pixels to grow (positive) or shrink (negative) the mask before inpainting. Growing a little covers the edges.",
    "feather": "Softness of the mask edge in pixels, so the redrawn area blends into the rest. 0 = hard edge.",
    "blend": "How much of the redraw is pasted in: 1 = all of it. Below 1 the old and new pixels are mixed, and features "
             "the redraw moved a little (lashes, eyelids) can show twice. For a lighter redraw, lower denoise instead.",
    "padding": "Pixels of the surrounding image included with each crop, so the model sees the context.",
    "inpaint_resolution": "Each crop is redrawn at about this many pixels squared (1024 = 1 megapixel), whatever its shape. "
                          "Higher = more detail, slower.",
    "denoise": "How much is redrawn. Low keeps the shape and fixes details, high redraws from scratch.",
    "seed": "Seed for the redraw.",
    "steps": "Sampling steps.",
    "cfg": "CFG for the redraw. 1 works for Krea 2, Flux, Z-Image, Anima and, at detailer denoise (up to about 0.4), "
           "SDXL too (tested: SDXL looked the same at cfg 1 to 7). Never goes below 1: below 1 the redraw comes out speckled.",
    "sampler_name": "Sampler.",
    "scheduler": "Scheduler.",
    "positive": "The image's prompt. A prompt describing the whole scene can get drawn into every crop at denoise 0.3 "
                "and up (a whole tiny scene in the face): wire inpaint_positive to prevent that.",
    "inpaint_positive": "Optional. A prompt for the crops only, e.g. 'close-up of a woman's face, natural skin'. "
                        "Used instead of positive for the redraw. Recommended whenever positive describes the whole scene.",
    "negative": "Optional. Leave empty for cfg 1 models; wire it for SDXL and other models that use a negative.",
    "pipe": "LC pipe: model, VAE, seed, steps, cfg, sampler and scheduler come from here.",
    "model": "Optional. Wire a model here to use it instead of the pipe's (with LoRAs on it, or a different model).",
    "vae": "Optional. Wire a VAE here to use it instead of the pipe's.",
    "pipe_cfg": "CFG for the redraw. 1 works for Krea 2, Flux, Z-Image and, at detailer denoise (up to about 0.4), SDXL too "
                "(tested: SDXL looked the same at cfg 1 to 7). Used instead of the pipe's cfg_1, which is usually the main "
                "sampler's.",
    "tone_match": "On: the redraw gets the original's brightness, contrast and colour back where it is pasted, keeping the "
                  "new detail. A light redraw comes back a little flat and dull without it. Turn off when the redraw is "
                  "meant to change colours (a new shirt colour, a different object).",
}


def _tip(name, **kw):
    return dict(kw, tooltip=TIP[name])


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def _find_inputs():
    return {
        "prompt": ("STRING", _tip("prompt", default="Face", multiline=True)),
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
        "cfg": ("FLOAT", _tip("cfg", default=1.0, min=1.0, max=30.0, step=0.1, round=0.01)),
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
    out["inpaint_positive"] = ("CONDITIONING", _tip("inpaint_positive"))
    if neg:
        out["negative"] = ("CONDITIONING", _tip("negative"))
    out["tone_match"] = ("BOOLEAN", _tip("tone_match", default=True))  # last: saved workflows keep their widget order
    if pos:  # pipe version only, after tone_match for the same reason
        out["cfg"] = ("FLOAT", _tip("pipe_cfg", default=1.0, min=1.0, max=30.0, step=0.1, round=0.01))
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
    edge = int(math.ceil(feather * 2)) + _ring(feather)
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
    # tested: a crop reaching out half the object's size made the face a smaller part of the redraw, and at 4x
    # upscales that left blotches. Padding alone keeps the object large in the crop.
    return [(m, _expand(b, int(padding), w, h)) for m, b in items]


def _target(cw, ch, res):
    # sized by area (res x res pixels), not by the long side: a wide crop keeps as much detail as a square one
    s = math.sqrt(float(res) * float(res) / float(max(1, cw * ch)))
    tw = max(SNAP, int(round(cw * s / SNAP)) * SNAP)
    th = max(SNAP, int(round(ch * s / SNAP)) * SNAP)
    return tw, th


def _inpaint_region(frame, region, box, s):
    """frame (1,H,W,3); region (H,W). Returns (frame with the region redrawn, soft blend mask (H,W))."""
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    rc = region[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
    soft = blur(rc, s["feather"])[0, 0].clamp(0, 1)  # what gets pasted back
    # what gets redrawn: a ring wider than the paste, so the seam sits on pixels the model already blended
    repaint = blur(grow(rc, _ring(s["feather"])), max(1.0, s["feather"] * 0.5))[0, 0].clamp(0, 1)
    repaint = torch.maximum(repaint, soft)
    crop = frame[:, y1:y2, x1:x2, :3]
    tw, th = _target(cw, ch, s["res"])
    up = comfy.utils.common_upscale(crop.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
    noise_mask = F.interpolate(repaint[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0]  # (1,th,tw)

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
    if s.get("tone_match", True):
        down = _match_tones(down, crop, soft)
    down = _seam_colour(down, crop, soft, repaint)

    out = frame.clone()
    a = (soft * s["blend"])[None, ..., None]
    out[:, y1:y2, x1:x2, :3] = crop * (1.0 - a) + down * a
    full = torch.zeros(frame.shape[1], frame.shape[2])
    full[y1:y2, x1:x2] = soft
    return out, full


def _seam_colour(down, crop, alpha, repaint):
    """Shift the redraw's colour to match the original along the seam. In the ring that was redrawn but is not pasted
    back, the true colour is known, so (original - redraw) there is the colour drift. It is spread as a smooth field,
    capped, and added to the redraw. down / crop: (1,H,W,3); alpha / repaint: (H,W)."""
    w = (repaint * (1.0 - alpha)).clamp(0, 1)
    if float(w.sum()) < 50.0:
        return down
    h, wd = w.shape
    diff = (crop - down)[0].permute(2, 0, 1)  # (3,H,W)
    k = max(1, min(h, wd) // 64)  # work small: the field is smooth anyway
    small = lambda t: F.interpolate(t[None], size=(max(1, h // k), max(1, wd // k)), mode="area")[0]
    ws, ds = small(w[None]), small(diff * w[None])
    sigma = max(2.0, max(ws.shape[-2:]) / 6.0)
    num = blur(ds[:, None], sigma)[:, 0]
    den = blur(ws[:, None], sigma)[:, 0]
    glob = (diff * w[None]).sum((1, 2)) / w.sum()  # fallback far from the ring
    field = torch.where(den > 1e-3, num / den.clamp(min=1e-3), glob.view(3, 1, 1))
    field = F.interpolate(field[None], size=(h, wd), mode="bilinear", align_corners=False)[0].clamp(-SEAM_CAP, SEAM_CAP)
    return (down + field.permute(1, 2, 0)[None]).clamp(0, 1)


def _lowfreq(img, size=32):
    """The broad colour of (1,H,W,3): shrunk to about 32 px, softened, scaled back up."""
    h, w = img.shape[1], img.shape[2]
    t = img.movedim(-1, 1)
    sh, sw = max(2, round(size * h / max(h, w))), max(2, round(size * w / max(h, w)))
    t = blur(F.interpolate(t, size=(sh, sw), mode="area"), 1.5)
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)


def _match_tones(down, crop, weight):
    """Give the redraw the original's tones where it is pasted: brightness, contrast and colour per channel, then the
    original's broad colour. A light redraw (and the VAE round trip, and the resize) comes back flatter and duller;
    the new fine detail stays. down / crop: (1,H,W,3); weight: (H,W)."""
    w = weight.clamp(0, 1)[None, ..., None]
    ws = float(w.sum())
    if ws < 50.0:
        return down
    mean = lambda t: (t * w).sum((0, 1, 2)) / ws
    md, mo = mean(down), mean(crop)
    sd = ((((down - md) ** 2) * w).sum((0, 1, 2)) / ws).sqrt().clamp(min=1e-4)
    so = ((((crop - mo) ** 2) * w).sum((0, 1, 2)) / ws).sqrt()
    x = (down - md) * (so / sd).clamp(0.8, 1.25) + mo
    return (x - _lowfreq(x) + _lowfreq(crop)).clamp(0, 1)


def _settings(kw, model, vae, positive, negative, sampler_name, scheduler, steps, cfg, seed):
    """kw: the node's own widgets (grow, feather, blend, padding, inpaint_resolution, denoise, inpaint_positive)."""
    if kw.get("inpaint_positive") is not None:
        positive = kw["inpaint_positive"]
    if negative is None:
        negative = _zero_out(positive)
    return dict(model=model, vae=vae, positive=positive, negative=negative, sampler_name=sampler_name,
                scheduler=scheduler, steps=int(steps), cfg=max(1.0, float(cfg)), denoise=float(kw["denoise"]), seed=int(seed),
                grow=float(kw["grow"]), feather=float(kw["feather"]), padding=int(kw["padding"]),
                res=int(kw["inpaint_resolution"]), blend=float(kw["blend"]), tone_match=kw.get("tone_match") is not False)


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
    missing = [k for k in ("model", "vae", "seed", "steps", "sampler_name", "scheduler") if vals[k] is None and k not in have]
    if missing:
        raise ValueError(f"[{name}] The pipe has no {', '.join(missing)}. Add it with LC Pipe In / LC Pipe Edit.")
    return vals


def _pipe_settings(pipe, kw, name):
    have = [k for k in ("model", "vae") if kw.get(k) is not None]
    p = _from_pipe(pipe, name, have)
    for k in have:
        p[k] = kw[k]
    positive = kw.get("positive") if kw.get("positive") is not None else p["positive"]
    if positive is None and kw.get("inpaint_positive") is None:
        raise ValueError(f"[{name}] No positive: wire one, or put it in the pipe.")
    negative = kw.get("negative") if kw.get("negative") is not None else p["negative"]
    cfg = kw.get("cfg") if kw.get("cfg") is not None else 1.0  # the detailer's own cfg, not the main sampler's cfg_1
    return _settings(kw, p["model"], p["vae"], positive, negative, p["sampler_name"], p["scheduler"], p["steps"],
                     cfg, p["seed"])


def _direct_settings(kw):
    return _settings(kw, kw["model"], kw["vae"], kw["positive"], kw.get("negative"), kw["sampler_name"],
                     kw["scheduler"], kw["steps"], kw["cfg"], kw["seed"])


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
class _Base(PreviewImage):
    @classmethod
    def VALIDATE_INPUTS(cls, sam3_model=None):
        # a '⬇ Download' entry saved on another machine is still valid here once the model is installed
        return lc_models.validate_models(sam3_model=("sam3", sam3_model))

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "LC MaskMaker/detailer"
    OUTPUT_NODE = True
    FUNCTION = "run"

    def __init__(self):
        super().__init__()

    def _smart(self, image, kw, s, hits=None):
        """hits: SAM 3 results already found for this image (from the pipe), keyed by word / model / threshold.
        Returns (image, mask, note, hits with this node's finds added)."""
        prompt = kw["prompt"]
        words = _words(prompt)
        if not words:
            raise ValueError("[LC Smart Detailer] The prompt is empty. Type what to detail, e.g. hands.")
        path = lc_models.resolve_sam3(kw["sam3_model"])
        hits = dict(hits or {})
        # the finding is kept on the node too: changing denoise, steps, feather, etc. redraws without searching again
        img_key = lc_pipe_keys.tag(image)
        if getattr(self, "_found", (None,))[0] != img_key:
            self._found = (img_key, {})
        reused = 0
        masks = [[] for _ in range(image.shape[0])]
        for word in words:
            key = f"{word}|{path}|{float(kw['threshold']):.3f}"
            per_frame = hits.get(key) or self._found[1].get(key)
            if per_frame is None:
                per_frame = [[_clean(m) for m in lc_sam3.objects(image[i:i + 1], [word], path, kw["threshold"])]
                             for i in range(image.shape[0])]
            else:
                reused += 1
            hits[key] = per_frame
            self._found[1][key] = per_frame
            for i in range(image.shape[0]):
                masks[i].extend(per_frame[i])
        if reused:
            print(f"[LC Smart Detailer] Reused what was already found for {reused} of {len(words)} word(s): no new search.")
        out, mask, done = _run(image, masks, s)
        return out, mask, _note(done, prompt.strip()), hits

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
        out, mask, note, _ = self._smart(image, kw, s)
        return {"ui": _preview(self, image, out, mask, note), "result": (out, mask)}


class LCSmartInpaintPipe(_Base):
    DESCRIPTION = DESC_SMART
    RETURN_TYPES = (PIPE_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (PIPE_TYPE, {"tooltip": TIP["pipe"]}),
                **_find_inputs(), **_mask_inputs(), **_denoise_input(),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional. Empty = the pipe's image. Wire one when a node without a pipe "
                                               "changed the picture in between."}),
                **_optional(pos=True),
            },
        }

    def run(self, pipe, image=None, **kw):
        name = "LC Smart Detailer (pipe)"
        image = lc_pipe_keys.pipe_image(pipe, image)
        if image is None:
            raise ValueError(f"[{name}] No image: wire one, or put it in the pipe (LC Pipe In / Aspect Ratio pipe).")
        s = _pipe_settings(pipe, kw, name)
        out, mask, note, hits = self._smart(image, kw, s, lc_pipe_keys.found_hits(pipe, image))
        # what goes on down the pipe: the result as the pipe's image, every area redrawn so far, what was found
        protect = lc_pipe_keys.merge_masks(lc_pipe_keys.protect_for(pipe, image), mask)
        out_pipe = lc_pipe_keys.updated(pipe, out, protect=protect, hits=hits)
        return {"ui": _preview(self, image, out, mask, note), "result": (out_pipe, out, mask)}


NODE_CLASS_MAPPINGS = {
    "LCSmartInpaint": LCSmartInpaint,
    "LCSmartInpaintPipe": LCSmartInpaintPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSmartInpaint": "LC Smart Detailer 🩹",
    "LCSmartInpaintPipe": "LC Smart Detailer (pipe) 🩹",
}
