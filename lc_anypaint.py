"""
LC Krea2 AnyPaint
-----------------
Inpaint and outpaint for Krea 2 with yijunwang2's AnyPaint LoRA (krea2_anypaint_rank32), done the way the LoRA
was trained (https://huggingface.co/yijunwang2/krea2-anypaint). The LoRA on its own does little: it needs this
runtime.

  reference  : the whole canvas, with the area to paint filled with the median colour of the kept pixels (a grey
               fill is read as content and painted back), shrunk to a 384 px edge and given to the model as a
               reference at timestep 0, also shown to the text encoder (VLM). Its tokens are spread over the
               full canvas (registered coordinates) and only see each other (isolated), as in training.
  keep       : everything outside the mask grown by a 32 px border is put back after every step (token-exact),
               so the kept pixels come back unchanged; the border is left for the model to blend.
  composite  : the redraw is pasted over the original with a soft edge inside the border, and the colour drift
               measured in the outer border is corrected (LC Smart Detailer's seam fix).
  crop       : a small mask is done in its own crop, scaled up to inpaint_resolution (more detail, faster).

The reference path runs in this node's own model wrapper, so it works even when another pack replaces Krea 2's
forward (ComfyUI-RedNodeStudio does, which made the first version copy the flat fill). LoRAs and attention / speed-up
options still apply. Method after yijunwang2's reference pipeline (Krea 2 Community License); the reference forward
is adapted from alexw5702-afk/krea2-anypaint (MIT) and ostris ComfyUI-Krea2-Ostris-Edit (MIT).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension as pe
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from einops import rearrange

from .lc_inpaint import _Base, _box, _expand, _preview, _seam_colour, _target, _zero_out
from .lc_refine_core import blur, grow

REF_EDGE = 384
TOKEN = 16  # Krea 2: 8x VAE, 2x2 patch
VLM_PREFIX = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
_LORA_CACHE = {}

TIP = {
    "prompt": "Describe the WHOLE finished picture, not just the change: \"a woman in a red coat on a snowy street\". "
              "Removal-only prompts (\"remove the car\") tend to leave a flat blob.",
    "mask": "White = paint, black = keep. Optional: no mask with LC Outpaint's canvas uses its mask. Wire LC Outpaint's "
            "control_image as the image and control_mask as the mask to outpaint.",
    "lora": "yijunwang2's krea2_anypaint_rank32 (models/loras). Applied at lora_strength for this node only.",
    "lora_strength": "1.0 is what it was trained for.",
    "steps": "8 (Krea 2 Turbo). The keep is exact with euler.",
    "sampler_name": "euler is what AnyPaint was trained and tested with; the keep step is exact only for euler.",
    "scheduler": "simple, as in the reference.",
    "crop_to_mask": "On: a small mask is redrawn in its own crop, scaled up to inpaint_resolution (more detail). "
                    "Off: the whole picture at inpaint_resolution.",
    "padding": "Context around the mask when cropping (px). More = the model sees more of the scene.",
    "inpaint_resolution": "Size of the redraw (as an area: 1024 = about 1 MP). AusBoss found above about 1.6 MP Krea 2 "
                          "starts to crease limbs and darken skies.",
    "boundary_px": "Border around the mask the model redraws to blend the seam (in redraw pixels). 32 = the reference.",
    "vlm_reference": "Show the reference to the text encoder too. On = the reference recipe.",
    "composite": "keep original = only the painted area (soft edge, colour fixed) is pasted back: everything else is "
                 "exactly your picture. raw = the model's whole redraw of the crop.",
}


def _tip(k, **kw):
    kw["tooltip"] = TIP[k]
    return kw


def _lora_choices():
    loras = folder_paths.get_filename_list("loras")
    hits = [l for l in loras if "anypaint" in l.lower()]
    return hits + [l for l in loras if l not in hits], (hits[0] if hits else (loras[0] if loras else ""))


# --------------------------------------------------------------------------------------- the model runtime
# The reference path runs in our own DIFFUSION_MODEL wrapper rather than in core's Krea 2 forward, because other
# packs may replace that forward for the whole class (e.g. ComfyUI-RedNodeStudio puts references "in context" on
# their own grid, which makes AnyPaint copy its flat fill instead of painting). Weight patches (LoRAs) and
# attention / speed-up options still apply. Adapted from alexw5702-afk/krea2-anypaint (MIT) and ostris
# ComfyUI-Krea2-Ostris-Edit (MIT), following yijunwang2's reference pipeline.
def _pack_refs(dit, refs, bs, device, dtype, th, tw):
    """Reference latents -> tokens on frame i+1, their y/x spread over the whole target grid (registered,
    centre-sampled, fractional), as in training."""
    patch = dit.patch
    toks, poss = [], []
    for i, ref in enumerate(refs):
        if ref.ndim == 5:
            rb, rc, rt, rh5, rw5 = ref.shape
            ref = ref.reshape(rb * rt, rc, rh5, rw5)
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref.to(device, dtype), (patch, patch))
        ref = comfy.utils.repeat_to_batch_size(ref, bs)
        rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
        toks.append(rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
        rid = torch.zeros(rh, rw, 3, device=device)
        rid[..., 0] = i + 1.0
        rid[..., 1] = ((torch.arange(rh, device=device) + 0.5) * (th / rh) - 0.5)[:, None]
        rid[..., 2] = ((torch.arange(rw, device=device) + 0.5) * (tw / rw) - 0.5)[None, :]
        poss.append(rid.reshape(1, rh * rw, 3).repeat(bs, 1, 1))
    return torch.cat(toks, 1), torch.cat(poss, 1)


def _attn(attn, x, freqs, capture=None, cache=None, to=None):
    """Krea 2 attention with the reference K/V captured (reference pass) or appended as extra keys (image pass)."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    q, k = apply_rope(q, k, freqs)
    if capture is not None:
        capture.append((k, v))
    if cache is not None:
        k = torch.cat((k, cache[0].to(k.dtype)), 2)
        v = torch.cat((v, cache[1].to(v.dtype)), 2)
    if attn.kvheads != attn.heads:
        r = attn.heads // attn.kvheads
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    out = optimized_attention_masked(q, k, v, attn.heads, mask=None, skip_reshape=True, transformer_options=to or {})
    return attn.wo(out * F.sigmoid(gate))


def _block(block, x, vec, freqs, capture=None, cache=None, to=None):
    ps, psh, pg, qs, qsh, qg = block.mod(vec)
    x = x + pg * _attn(block.attn, (1 + ps) * block.prenorm(x) + psh, freqs, capture, cache, to)
    return x + qg * block.mlp((1 + qs) * block.postnorm(x) + qsh)


def _ref_kv(dit, refs, timesteps, bs, device, dtype, th, tw, to):
    """The reference tokens alone (isolated: they only see each other), at timestep 0: each block's K/V."""
    tok, pos = _pack_refs(dit, refs, bs, device, dtype, th, tw)
    h = dit.first(tok)
    t0 = dit.tproj(dit.tmlp(timestep_embedding(torch.zeros_like(timesteps), dit.tdim).unsqueeze(1).to(h.dtype)))
    freqs = dit.pe_embedder(pos)
    kvs = []
    for block in dit.blocks:
        cap = []
        h = _block(block, h, t0, freqs, capture=cap, to=to)
        kvs.append(cap[0])
    return kvs


def _anypaint_forward(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={},
                      **kwargs):
    dit = executor.class_obj
    if not ref_latents:
        return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
    to = transformer_options
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, _, Ho, Wo = x.shape
    p = dit.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (p, p))
    h_, w_ = x.shape[-2] // p, x.shape[-1] // p
    dev = x.device
    kvs = _ref_kv(dit, ref_latents, timesteps, bs, dev, x.dtype, h_, w_, to)
    ctx = dit.txtmlp(dit.txtfusion(dit._unpack_context(context), mask=None, transformer_options=to))
    img = dit.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p))
    t = dit.tmlp(timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(img.dtype))
    tvec = dit.tproj(t)
    tl, il = ctx.shape[1], img.shape[1]
    comb = torch.cat((ctx, img), 1)
    ids = torch.zeros(h_, w_, 3, device=dev)
    ids[..., 1] = torch.arange(h_, device=dev)[:, None]
    ids[..., 2] = torch.arange(w_, device=dev)[None, :]
    pos = torch.cat((torch.zeros(bs, tl, 3, device=dev), ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)), 1)
    freqs = dit.pe_embedder(pos)
    for block, kv in zip(dit.blocks, kvs):
        comb = _block(block, comb, tvec, freqs, cache=kv, to=to)
    out = dit.last(comb, t)[:, tl:tl + il]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=p, pw=p, c=dit.channels)[:, :, :Ho, :Wo]
    if temporal:
        out = out.reshape(b5, t5, dit.channels, Ho, Wo).movedim(1, 2)
    return out


def _patched_model(model, lora_name, strength):
    m = model
    if lora_name and strength:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = _LORA_CACHE.get(path)
        if sd is None:
            _LORA_CACHE.clear()
            sd = _LORA_CACHE[path] = comfy.utils.load_torch_file(path, safe_load=True)
        m, _ = comfy.sd.load_lora_for_models(m, None, sd, float(strength), 0.0)
    m = m.clone()
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "lc_krea2_anypaint", _anypaint_forward)
    return m


# --------------------------------------------------------------------------------------- preparation
def _median(pixels):
    """(N,3) -> (3,) median colour, or mid grey when nothing is kept."""
    if pixels.numel() == 0:
        return torch.full((3,), 0.5)
    return pixels.median(dim=0).values


def _resize(img, w, h, mode="lanczos"):
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, mode, "disabled").movedim(1, -1).clamp(0, 1)


def _reference(known, gen):
    """The semantic reference: kept pixels as they are, painted pixels = median of the kept ones, max edge 384."""
    ref = known.clone()
    g = gen > 0.5
    ref[0][g] = _median(known[0][~g])
    h, w = ref.shape[1:3]
    s = min(1.0, REF_EDGE / max(h, w))
    rw = max(TOKEN, int(round(w * s)) // TOKEN * TOKEN)
    rh = max(TOKEN, int(round(h * s)) // TOKEN * TOKEN)
    return _resize(ref, rw, rh)


def _keep_tokens(gen, boundary_px, lh, lw):
    """Pixel mask of what is painted -> latent-size noise mask (1 = generate) on the exact token grid: a token is
    kept only if all its pixels are outside the painted area grown by the border (square dilation)."""
    r = int(boundary_px)
    g = gen[None, None].float()
    if r > 0:
        g = F.max_pool2d(g, 2 * r + 1, stride=1, padding=r)
    keep = 1.0 - g.clamp(0, 1)
    keep = F.interpolate(keep, size=(lh, lw), mode="nearest")
    keep = (keep > 0.5).float()
    p = 2
    hb, wb = lh // p, lw // p
    kb = keep[..., : hb * p, : wb * p].reshape(1, 1, hb, p, wb, p).amin(dim=(3, 5))
    kt = kb.repeat_interleave(p, 2).repeat_interleave(p, 3)
    full = torch.zeros((1, 1, lh, lw))
    full[..., : hb * p, : wb * p] = kt
    return (1.0 - full)[:, 0]  # (1, lh, lw)


def _encode(clip, vae, prompt, ref, vlm):
    if vlm:
        try:
            from comfy.text_encoders.krea2 import KREA2_TEMPLATE

            tokens = clip.tokenize(VLM_PREFIX + prompt, images=[ref], llama_template=KREA2_TEMPLATE)
        except Exception:
            tokens = clip.tokenize(VLM_PREFIX + prompt, images=[ref])
    else:
        tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    ref_latent = vae.encode(ref[..., :3])
    cond = node_helpers.conditioning_set_values(cond, {"reference_latents": [ref_latent]}, append=True)
    return node_helpers.conditioning_set_values(cond, {"reference_latents_method": "index_timestep_zero"})


# --------------------------------------------------------------------------------------- one frame
def _paint(frame, gen_full, s):
    """frame (1,H,W,3), gen_full (H,W) binary. Returns (frame, paste alpha (H,W))."""
    H, W = frame.shape[1], frame.shape[2]
    if s["crop"]:
        b = _box(gen_full)
        b = _expand(b, int(s["padding"]) + int(s["boundary"]) * 2, W, H)
    else:
        b = [0, 0, W, H]
    x1, y1, x2, y2 = b
    cw, ch = x2 - x1, y2 - y1
    crop = frame[:, y1:y2, x1:x2, :3]
    gen_c = gen_full[y1:y2, x1:x2]
    tw, th = _target(cw, ch, s["res"])
    up = _resize(crop, tw, th)
    gen = (F.interpolate(gen_c[None, None].float(), size=(th, tw), mode="nearest")[0, 0] > 0.5).float()

    ref = _reference(up, gen)
    pos = _encode(s["clip"], s["vae"], s["prompt"], ref, s["vlm"])
    neg = _zero_out(pos)

    model, vae = s["model"], s["vae"]
    latent = vae.encode(up)
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    lh, lw = latent.shape[-2], latent.shape[-1]
    noise_mask = _keep_tokens(gen, s["boundary"], lh, lw)
    noise = comfy.sample.prepare_noise(latent, s["seed"])
    callback = latent_preview.prepare_callback(model, s["steps"])
    samples = comfy.sample.sample(
        model, noise, s["steps"], 1.0, s["sampler_name"], s["scheduler"], pos, neg, latent,
        denoise=1.0, noise_mask=noise_mask, callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=s["seed"],
    )
    patch = vae.decode(samples)
    if patch.ndim == 5:
        patch = patch.reshape(-1, *patch.shape[-3:])
    patch = patch[:1, ..., :3].float().cpu()
    down = _resize(patch, cw, ch)

    out = frame.clone()
    full = torch.zeros(H, W)
    if s["composite"] == "raw":
        out[:, y1:y2, x1:x2, :3] = down
        full[y1:y2, x1:x2] = 1.0
        return out, full
    # the border in crop pixels: paste the painted area plus half the border with a soft edge; the outer half of
    # the border was redrawn too, so it measures the colour drift for the seam fix
    bpx = max(1.0, float(s["boundary"]) * cw / float(tw))
    alpha = blur(grow(gen_c[None, None].float(), bpx * 0.5), bpx * 0.25)[0, 0].clamp(0, 1)
    alpha = torch.maximum(alpha, gen_c.float())
    repaint = grow(gen_c[None, None].float(), bpx)[0, 0].clamp(0, 1)
    down = _seam_colour(down, crop, alpha, repaint)
    a = alpha[None, ..., None]
    out[:, y1:y2, x1:x2, :3] = crop * (1.0 - a) + down * a
    full[y1:y2, x1:x2] = alpha
    return out, full


# --------------------------------------------------------------------------------------- the node
class LCKrea2AnyPaint(_Base):
    DESCRIPTION = (
        "Inpaint and outpaint for Krea 2 with yijunwang2's AnyPaint LoRA, done the way it was trained: a whole-canvas "
        "reference, the kept pixels put back after every step, and a border the model blends. Your picture outside the "
        "mask comes back unchanged. Describe the whole finished picture in the prompt.\n"
        "Outpaint: wire LC Outpaint's control_image and control_mask in."
    )

    @classmethod
    def INPUT_TYPES(cls):
        loras, default = _lora_choices()
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea 2 (Turbo). The LoRA and patches are added for this node only."}),
                "clip": ("CLIP", {"tooltip": "Krea 2 text encoder (Qwen3-VL 4B, CLIP type krea2)."}),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK", _tip("mask")),
                "prompt": ("STRING", _tip("prompt", default="", multiline=True)),
                "lora": (loras, _tip("lora", default=default)),
                "lora_strength": ("FLOAT", _tip("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05)),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", _tip("steps", default=8, min=1, max=100)),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, _tip("sampler_name", default="euler")),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, _tip("scheduler", default="simple")),
                "crop_to_mask": ("BOOLEAN", _tip("crop_to_mask", default=True)),
                "padding": ("INT", _tip("padding", default=128, min=0, max=2048, step=8)),
                "inpaint_resolution": ("INT", _tip("inpaint_resolution", default=1024, min=256, max=2048, step=64)),
                "boundary_px": ("INT", _tip("boundary_px", default=32, min=0, max=128, step=4)),
                "vlm_reference": ("BOOLEAN", _tip("vlm_reference", default=True)),
                "composite": (["keep original", "raw"], _tip("composite", default="keep original")),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = ("Your picture with the masked area painted.", "Where it was pasted back (soft edge).")
    FUNCTION = "paint"
    CATEGORY = "LC MaskMaker/inpaint"
    OUTPUT_NODE = True

    def paint(self, model, clip, vae, image, mask, prompt, lora, lora_strength, seed, steps, sampler_name, scheduler,
              crop_to_mask, padding, inpaint_resolution, boundary_px, vlm_reference, composite):
        if not (prompt or "").strip():
            raise ValueError("[LC Krea2 AnyPaint] Describe the whole finished picture in the prompt.")
        s = dict(model=_patched_model(model, lora, lora_strength), clip=clip, vae=vae, prompt=prompt.strip(),
                 seed=int(seed), steps=int(steps), sampler_name=sampler_name, scheduler=scheduler, crop=bool(crop_to_mask),
                 padding=int(padding), res=int(inpaint_resolution), boundary=int(boundary_px), vlm=bool(vlm_reference),
                 composite=composite)
        b, h, w, _ = image.shape
        m = mask
        if m.ndim == 2:
            m = m[None]
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m[:, None].float(), size=(h, w), mode="nearest")[:, 0]
        frames, masks, done = [], [], 0
        for i in range(b):
            comfy.model_management.throw_exception_if_processing_interrupted()
            frame = image[i:i + 1].float().cpu()
            gen = (m[min(i, m.shape[0] - 1)].float().cpu() > 0.5).float()
            if gen.sum() < 1:
                frames.append(frame)
                masks.append(torch.zeros(h, w))
                continue
            frame, alpha = _paint(frame, gen, s)
            frames.append(frame)
            masks.append(alpha)
            done += 1
        out, outmask = torch.cat(frames, 0), torch.stack(masks, 0)
        note = f"painted {done} of {b} image{'s' if b > 1 else ''}" if done else "the mask is empty: nothing painted"
        return {"ui": _preview(self, image, out, outmask, note), "result": (out, outmask)}


NODE_CLASS_MAPPINGS = {"LCKrea2AnyPaint": LCKrea2AnyPaint}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2AnyPaint": "LC Krea2 AnyPaint 🩹"}
