"""
LC Character Detailer 🩹
------------------------
LC Smart Detailer for one character: SAM 3 finds every match (every face), and only ONE of them is redrawn: the Nth
from the left / right / top, the Nth largest, or the one inside box N of LC Bounding Boxes. One per character, each on
its own model branch (its own LoRA): two characters, two of these.

Built on LC Smart Detailer's engine (lc_inpaint): the finding, the crop, the redraw and the blend are the same code.
LC Smart Detailer itself is not changed.
"""
from __future__ import annotations

import torch

from . import lc_inpaint as I
from . import lc_models, lc_pipe_keys, lc_sam3

PICKS = ["left to right", "right to left", "top to bottom", "largest first", "in box"]
TIPS = {
    "pick": "Which match this detailer redraws (SAM 3 finds them all). With several words (face, hair) the first one "
            "picks the person and the others come along when they belong to them.\n"
            "left to right / right to left / top to bottom: number 1 = the first one that way.\n"
            "largest first: number 1 = the biggest.\n"
            "in box: the one inside box 'number' of LC Bounding Boxes (wire its elements): the face stays with its "
            "character even when the pose changes.",
    "number": "Which one: 1 = the first. With pick on in box, the box's number as shown on LC Bounding Boxes.",
    "elements": "LC Bounding Boxes' elements, for pick = in box.",
}
DESC = ("LC Smart Detailer for one character. SAM 3 finds every match (every face, every person) and only one of them "
        "is redrawn: the Nth from the left, right or top, the Nth largest, or the one inside a box of LC Bounding "
        "Boxes. Give each character its own: branch the model, add that character's LoRA, and pick their face here. "
        "Same settings and the same engine as LC Smart Detailer.")


def _stats(m):
    """(centre x, centre y, area) of a (H,W) mask, in 0-1 of the picture."""
    m = m.float()
    h, w = m.shape
    total = float(m.sum())
    if total <= 0:
        return None
    ys = torch.arange(h, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, w)
    return float((m * xs).sum()) / total / w, float((m * ys).sum()) / total / h, total / (h * w)


def _belongs(extra, anchor):
    """An extra word's match (hair, neck) belongs to the picked one when most of it lies within the picked one's area
    grown by its own size (hair around a face, not the hair of the person next to her)."""
    a = anchor > 0.5
    if not bool(a.any()):
        return False
    ys, xs = torch.nonzero(a, as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = y1 - y0 + 1, x1 - x0 + 1
    H, W = a.shape
    Y0, Y1, X0, X1 = max(0, y0 - h), min(H, y1 + h + 1), max(0, x0 - w), min(W, x1 + w + 1)
    e = extra > 0.5
    total = float(e.sum())
    return total > 0 and float(e[Y0:Y1, X0:X1].sum()) / total >= 0.6


def _choose(masks, pick, number, elements):
    """The one mask this node redraws, or None. Returns (mask or None, why-not text)."""
    st = [(m, _stats(m)) for m in masks]
    st = [(m, s) for m, s in st if s]
    if not st:
        return None, "nothing found"
    n = max(1, int(number))
    if pick == "in box":
        els = [e for e in (elements if isinstance(elements, list) else []) if isinstance(e, dict) and e.get("bbox")]
        if not els:
            return None, "pick is in box but no LC Bounding Boxes elements are wired"
        if n > len(els):
            return None, f"there is no box {n} (the layout has {len(els)})"
        y0, x0, y1, x1 = (float(v) / 1000 for v in els[n - 1]["bbox"])
        inside = [(m, s) for m, s in st if x0 <= s[0] <= x1 and y0 <= s[1] <= y1]
        if not inside:
            return None, f"none of the {len(st)} found is inside box {n}"
        return max(inside, key=lambda t: t[1][2])[0], ""
    key = {"left to right": lambda t: t[1][0], "right to left": lambda t: -t[1][0],
           "top to bottom": lambda t: t[1][1], "largest first": lambda t: -t[1][2]}[pick]
    order = sorted(st, key=key)
    if n > len(order):
        return None, f"only {len(order)} found, there is no number {n}"
    return order[n - 1][0], ""


def _numbers(image, found, pick, elements, chosen):
    """The before picture with every match numbered the way pick counts them (in box: the box number, "-" outside the
    boxes), the redrawn one in yellow, like Impact's SEGS Picker: read the number off the preview."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    arr = (image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, W // 28))
    except OSError:
        font = ImageFont.load_default()
    st = [(m, _stats(m)) for m in found]
    st = [(m, s) for m, s in st if s]
    labels = []
    if pick == "in box":
        els = [e for e in (elements if isinstance(elements, list) else []) if isinstance(e, dict) and e.get("bbox")]
        for m, s in st:
            lab = "-"
            for k, e in enumerate(els):
                y0, x0, y1, x1 = (float(v) / 1000 for v in e["bbox"])
                if x0 <= s[0] <= x1 and y0 <= s[1] <= y1:
                    lab = f"box {k + 1}"
                    break
            labels.append((m, s, lab))
    else:
        key = {"left to right": lambda t: t[1][0], "right to left": lambda t: -t[1][0],
               "top to bottom": lambda t: t[1][1], "largest first": lambda t: -t[1][2]}[pick]
        labels = [(m, s, str(k + 1)) for k, (m, s) in enumerate(sorted(st, key=key))]
    for m, s, lab in labels:
        hit = chosen is not None and m is chosen
        x, y = s[0] * W, s[1] * H
        bb = d.textbbox((0, 0), lab, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 6
        d.rounded_rectangle([x - tw / 2 - pad, y - th / 2 - pad, x + tw / 2 + pad, y + th / 2 + pad], radius=6,
                            fill=(255, 210, 40) if hit else (30, 30, 30), outline=(255, 255, 255), width=2)
        d.text((x - tw / 2, y - th / 2 - bb[1]), lab, fill=(0, 0, 0) if hit else (255, 255, 255), font=font)
    out = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]
    return torch.cat([out, image[:1, ..., 3:]], -1) if image.shape[-1] == 4 else out


class _CharBase(I._Base):
    DESCRIPTION = DESC

    def _one(self, image, kw, s, hits=None):
        """LC Smart Detailer's finding (same cache), then only the picked match per frame."""
        words = I._words(kw["prompt"])
        if not words:
            raise ValueError("[LC Character Detailer] The prompt is empty. Type what to find, e.g. face.")
        path = lc_models.resolve_sam3(kw["sam3_model"])
        hits = dict(hits or {})
        img_key = lc_pipe_keys.tag(image)
        if getattr(self, "_found", (None,))[0] != img_key:
            self._found = (img_key, {})
        # the first word picks the character (face); the other words (hair, neck) come along when they belong to them
        per_word = []
        for word in words:
            key = f"{word}|{path}|{float(kw['threshold']):.3f}"
            per_frame = hits.get(key) or self._found[1].get(key)
            if per_frame is None:
                per_frame = [[I._clean(m) for m in lc_sam3.objects(image[i:i + 1], [word], path, kw["threshold"])]
                             for i in range(image.shape[0])]
            hits[key] = per_frame
            self._found[1][key] = per_frame
            per_word.append(per_frame)
        masks, why, first = [], "", None
        for i in range(image.shape[0]):
            m, why_i = _choose(per_word[0][i], kw["pick"], kw["number"], kw.get("elements"))
            picked = [m] if m is not None else []
            if m is not None:
                for extra in per_word[1:]:
                    picked += [e for e in extra[i] if _belongs(e, m)]
            masks.append(picked)
            why = why or why_i
            if i == 0:
                first = m
        self._numbered = _numbers(image, per_word[0][0], kw["pick"], kw.get("elements"), first)
        out, mask, done = I._run(image, masks, s)
        if done:
            note = f"{kw['prompt'].strip()} {kw['number']} ({kw['pick']}) redrawn"
        else:
            note = f"Nothing redrawn: {why}. Image passed through."
            print(f"[LC Character Detailer] {note}")
        return out, mask, note, hits


def _pick_inputs():
    return {"pick": (PICKS, {"default": "left to right", "tooltip": TIPS["pick"]}),
            "number": ("INT", {"default": 1, "min": 1, "max": 32, "tooltip": TIPS["number"]})}


class LCCharacterDetailer(_CharBase):
    @classmethod
    def INPUT_TYPES(cls):
        opt = I._optional()
        opt["elements"] = ("ARRAY", {"tooltip": TIPS["elements"]})
        return {
            "required": {
                "image": ("IMAGE",), "model": ("MODEL",), "vae": ("VAE",), "positive": I._cond_inputs()[0],
                **I._find_inputs(), **_pick_inputs(), **I._mask_inputs(), **I._sampler_inputs(),
            },
            "optional": opt,
        }

    def run(self, image, **kw):
        s = I._direct_settings(kw)
        out, mask, note, _ = self._one(image, kw, s)
        return {"ui": I._preview(self, self._numbered, out, mask, note), "result": (out, mask)}


class LCCharacterDetailerPipe(_CharBase):
    RETURN_TYPES = (I.PIPE_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (I.PIPE_TYPE, {"tooltip": I.TIP["pipe"]}),
                **I._find_inputs(), **_pick_inputs(), **I._mask_inputs(), **I._denoise_input(),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional. Empty = the pipe's image. Wire one when a node without a pipe "
                                               "changed the picture in between."}),
                **I._optional(pos=True),
                "elements": ("ARRAY", {"tooltip": TIPS["elements"]}),
            },
        }

    def run(self, pipe, image=None, **kw):
        name = "LC Character Detailer (pipe)"
        image = lc_pipe_keys.pipe_image(pipe, image)
        if image is None:
            raise ValueError(f"[{name}] No image: wire one, or put it in the pipe (LC Pipe In / Aspect Ratio pipe).")
        s = I._pipe_settings(pipe, kw, name)
        out, mask, note, hits = self._one(image, kw, s, lc_pipe_keys.found_hits(pipe, image))
        protect = lc_pipe_keys.merge_masks(lc_pipe_keys.protect_for(pipe, image), mask)
        out_pipe = lc_pipe_keys.updated(pipe, out, protect=protect, hits=hits)
        return {"ui": I._preview(self, self._numbered, out, mask, note), "result": (out_pipe, out, mask)}


NODE_CLASS_MAPPINGS = {"LCCharacterDetailer": LCCharacterDetailer, "LCCharacterDetailerPipe": LCCharacterDetailerPipe}
NODE_DISPLAY_NAME_MAPPINGS = {"LCCharacterDetailer": "LC Character Detailer (BETA) 🩹", "LCCharacterDetailerPipe": "LC Character Detailer (pipe) (BETA) 🩹"}
