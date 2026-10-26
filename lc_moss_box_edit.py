"""
LC MOSS Box Edit 🖌️ (BETA)
--------------------------
Redraws only the boxes drawn on a picture with MOSS Image; everything outside them stays pixel-identical.

How (the method of artfat-creator/ComfyUI-Artfat-MOSS-BBox, MIT):
- the WHOLE picture is VAE-encoded and a soft mask made from the boxes is its noise mask, so MOSS sees all of the
  surroundings while it redraws inside the boxes (the old LC MOSS AnyPaint redrew a crop blind);
- the prompt is MOSS's own layout JSON: a caption of the whole picture as it must look after the change, plus each
  box's description at its exact place;
- Differential Diffusion turns the soft mask edge into a soft edit, and the result is pasted back through the same
  mask.
Boxes come from LC Bounding Boxes (its `elements`); a "move" box masks both where the thing was and where it goes,
a "remove" box asks for the background to continue.
"""
from __future__ import annotations

import json
import math

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
import torch


def _centred(bbox, width, height):
    """0-1000 [y0, x0, y1, x1] -> pixel box centred on the canvas, MOSS layout rounding (same as LC MOSS Layout Prompt)."""
    y0, x0, y1, x1 = (float(v) for v in bbox)
    return [math.floor(round(y0 * height / 1000, 9)) - height // 2, math.floor(round(x0 * width / 1000, 9)) - width // 2,
            math.ceil(round(y1 * height / 1000, 9)) - height // 2, math.ceil(round(x1 * width / 1000, 9)) - width // 2]


def _box_mask(boxes, width, height, grow, feather):
    """White inside the boxes (0-1000 yxyx), grown by `grow` px, then a linear fade over `feather` px."""
    ys = torch.arange(height, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, -1)
    m = torch.zeros((height, width), dtype=torch.float32)
    for y0, x0, y1, x1 in boxes:
        x0, x1 = x0 * width / 1000 - grow, x1 * width / 1000 + grow
        y0, y1 = y0 * height / 1000 - grow, y1 * height / 1000 + grow
        if x1 <= x0 or y1 <= y0:
            continue
        dx = (x0 - xs).clamp(min=0) + (xs - x1).clamp(min=0)
        dy = (y0 - ys).clamp(min=0) + (ys - y1).clamp(min=0)
        d = torch.sqrt(dx * dx + dy * dy)
        m = torch.maximum(m, (d == 0).float() if feather <= 0 else (1.0 - d / feather).clamp(0.0, 1.0))
    return m


def _layout(caption, els, width, height):
    items = []
    for e in els:
        kind = e.get("kind", "new")
        desc = str(e.get("desc") or "").strip()
        if kind == "remove":
            desc = "nothing here: the background around it continues seamlessly"
        label = "text" if e.get("type") == "text" else ("character" if e.get("label") == "character" else "object")
        item = {"label": label, "caption": desc or label}
        if label == "text":
            item["text"] = str(e.get("text") or "")
        pal = e.get("color_palette") or e.get("palette")
        if isinstance(pal, list) and pal:
            item["palette"] = pal
        item["bbox"] = _centred(e.get("tgt_bbox") or e["bbox"], width, height)
        items.append(item)
    return json.dumps({"summary": {"canvas": "single", "caption": caption.strip() or "A photograph."}, "items": items,
                       "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}, ensure_ascii=False, separators=(",", ":"))


class LCMOSSBoxEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MOSS Image (SFT or a merge of it), LoRAs already applied."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The FLUX.2 VAE (flux2-vae)."}),
                "image": ("IMAGE", {"tooltip": "The picture to edit (the same picture the boxes were drawn on)."}),
                "elements": ("ARRAY", {"tooltip": "The boxes from LC Bounding Boxes: they make the mask (and the layout, when the prompt is a plain sentence)."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Wire LC MOSS Layout Prompt's prompt here (fed the same boxes), or type the whole picture as it must look AFTER the change (the scene, the people, the light). "
                                                 "Describe the result, not the instruction: 'a woman in a red puffer jacket by a rainy window', "
                                                 "not 'make her jacket red'."}),
                "negative": ("STRING", {"multiline": True, "default": "censored, worst quality, bad anatomy, watermark, jpeg artifact, blurry",
                                        "tooltip": "MOSS has a real negative."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "tooltip": "25 for MOSS SFT; 8 with the DMD 8-step LoRA."}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "2.5 for MOSS SFT; 1 with the DMD 8-step LoRA."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "grow": ("INT", {"default": 16, "min": 0, "max": 512, "tooltip": "The redrawn area reaches this many pixels past each box."}),
                "feather": ("INT", {"default": 64, "min": 0, "max": 1024,
                                    "tooltip": "Soft edge in pixels past the grown box. 0 = a hard edge (a visible seam)."}),
                "keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                                   "tooltip": "How much of what is inside the boxes to keep. 0 = redraw it fully (a new thing). "
                                              "0.3 - 0.5 = keep the shape and pose, change the look."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "layout_prompt")
    OUTPUT_TOOLTIPS = ("The edited picture: outside the boxes it is pixel-identical.",
                       "The soft mask that was redrawn.",
                       "The layout prompt MOSS got.")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/MOSS"
    DESCRIPTION = ("Redraws only the boxes you drew on a picture with MOSS Image, seeing the whole picture while it works; "
                   "everything outside the boxes stays pixel-identical. Draw the boxes with LC Bounding Boxes and describe "
                   "what goes in each one; the prompt describes the whole picture after the change.")

    def run(self, model, clip, vae, image, elements, prompt, negative, seed, steps, cfg, sampler_name, scheduler, grow, feather, keep):
        from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

        image = image[:1, :, :, :3]
        h0, w0 = int(image.shape[1]), int(image.shape[2])
        W, H = max(16, w0 // 16 * 16), max(16, h0 // 16 * 16)
        src = image if (W, H) == (w0, h0) else image[:, :H, :W, :]
        els = [e for e in (elements if isinstance(elements, list) else []) if isinstance(e, dict) and e.get("bbox")]
        if not els:
            return (image, torch.zeros((1, h0, w0)), prompt)
        boxes = []
        for e in els:
            for k in ("src_bbox", "tgt_bbox", "bbox"):
                if e.get(k):
                    boxes.append([float(v) for v in e[k]])
        mask = _box_mask(boxes, W, H, grow, feather)[None]  # [1,H,W]
        # a finished layout from LC MOSS Layout Prompt (or the Prompt Enhancer after it) goes in as it is; a plain
        # sentence becomes the caption and the boxes are added here
        p = (prompt or "").strip()
        text = p if p.startswith("{") and '"items"' in p else _layout(p, els, W, H)
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative or ""))

        latent = {"samples": vae.encode(src), "noise_mask": mask.reshape(1, 1, H, W)}
        m = model.clone()
        m.set_model_denoise_mask_function(lambda *a, **k: DifferentialDiffusion.forward(*a, **k, strength=1.0))
        start = min(int(round(keep * steps)), steps - 1)
        out = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent,
                                    start_step=start, last_step=steps, force_full_denoise=True)[0]
        dec = vae.decode(out["samples"])
        if dec.ndim == 5:
            dec = dec.reshape(-1, dec.shape[-3], dec.shape[-2], dec.shape[-1])
        dec = dec[:1].to(src.device)
        if dec.shape[1:3] != src.shape[1:3]:
            dec = comfy.utils.common_upscale(dec.movedim(-1, 1), W, H, "bilinear", "disabled").movedim(1, -1)
        a = mask.reshape(1, H, W, 1).to(dec.device)
        edited = src.to(dec.device) * (1.0 - a) + dec * a
        result = image.clone().to(dec.device)
        result[:, :H, :W, :] = edited
        full = torch.zeros((1, h0, w0))
        full[:, :H, :W] = mask
        return (result.cpu(), full, text)


NODE_CLASS_MAPPINGS = {"LCMOSSBoxEdit": LCMOSSBoxEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"LCMOSSBoxEdit": "LC MOSS Box Edit 🖌️ (BETA)"}
