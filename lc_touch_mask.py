"""
LC Touch Mask
-------------
Mask one thing exactly, by hand, on the node:

  draw around it   : a lasso. SAM 3 gets the lasso's box plus a point inside it and snaps to the object's edges
                     (clipped to the lasso, so it never runs off to something you did not circle).
  ctrl + draw      : the same, but cuts the area out of the mask.
  tap it           : grows to the next bigger thing that holds it (face -> + hair -> head -> whole person, or
                     handle -> cup -> cup and saucer). On a person the steps come from the Segformer parts
                     LC Person Mask uses; everything else grows with SAM 3 boxes around the current selection.
  double-tap it    : one step back (at the first step it removes that selection).
  tap empty space  : picks the object under the tap (shift + tap cuts it out).

Each selection is saved as a recipe (the lasso or tap, plus the chosen growth steps), so the run rebuilds exactly
what the node showed, at full resolution. The live preview needs the picture: it reads a Load Image upstream
straight away, anything else after one run.

The stroke-to-prompt idea (a box plus an inner point from a hand-drawn shape) follows AusBoss' Mask by Touch
(ausboss-nodes, MIT).
"""

import asyncio
import base64
import io
import json
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import lc_cache, lc_hold, lc_models
from .lc_matting import METHODS

DECIDE_MAX = 768  # longer side the growth steps are decided at (the same at preview and run)
SEG_MAX = 1536
MAX_STEPS = 7
GROW_MIN = 1.15  # a step must be at least this much bigger
COVER_MIN = 0.6  # and hold at least this much of the current selection
SAM_SCALES = (1.5, 2.2, 3.2)
LASSO_SLACK = 0.02  # the lasso clip is grown by this much of the image diagonal

# Segformer (segformer_b2_clothes) parts for growing on a person, smallest first
PERSON_STEPS = [
    ("face", {"face"}),
    ("face + hair", {"face", "hair"}),
    ("head", {"face", "hair", "hat", "sunglasses"}),
    ("top", {"upper-clothes", "scarf"}),
    ("clothes", {"upper-clothes", "skirt", "pants", "dress", "belt", "scarf"}),
    ("person", None),  # everything but background
]

_lock = threading.Lock()
_enc_cache = lc_cache.register("touch_mask_sam3", {})  # image key -> features
_seg_cache = lc_cache.register("touch_mask_parts", {})  # image key -> (probs, names)
_ladder_cache = lc_cache.register("touch_mask_ladders", {})  # (image key, seed) -> ladder


# ---------- models ----------

def _sam3_path(choice):
    return lc_models.resolve_sam3(choice)


def _encode(key, image_hw3, sam3_choice):
    """SAM 3 image features for the interactive decoder, computed once per picture."""
    if key in _enc_cache:
        return _enc_cache[key]
    import comfy.model_management as mm
    import comfy.utils

    from . import lc_sam3

    model, _clip = lc_sam3.load(_sam3_path(sam3_choice))
    mm.load_model_gpu(model)
    dev = mm.get_torch_device()
    dtype = model.model.get_dtype()
    net = model.model.diffusion_model
    x = comfy.utils.common_upscale(image_hw3[None].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
    x = x.to(device=dev, dtype=dtype)
    with torch.no_grad():
        bb = net.detector.backbone["vision_backbone"]
        if bb.multiplex:
            _, _, feats, _pos = bb(x, tracker_mode="interactive")
        else:
            _, _, feats, _pos = bb(x, need_tracker=True)
            if net.detector.scalp > 0:
                feats = feats[:-net.detector.scalp]
        high_res = list(feats[:-1])
        feat = feats[-1]
        b, c, h, w = feat.shape
        no_mem = getattr(net.tracker, "interactivity_no_mem_embed", None)
        if no_mem is None:
            no_mem = getattr(net.tracker, "no_mem_embed", None)
        if no_mem is not None:
            flat = feat.flatten(2).permute(0, 2, 1)
            flat = flat + no_mem.to(flat)
            feat = flat.view(b, h, w, c).permute(0, 3, 1, 2)
    enc = {"model": model, "net": net, "high_res": high_res, "feat": feat, "dev": dev, "dtype": dtype}
    _enc_cache.clear()
    _enc_cache[key] = enc
    return enc


def _sam(enc, h, w, points=None, box=None):
    """One SAM 3 mask at (h, w). points = [(x, y, label)] and box = (x0, y0, x1, y1), all normalized 0-1."""
    import comfy.model_management as mm

    mm.load_model_gpu(enc["model"])
    dev, dtype = enc["dev"], enc["dtype"]
    pi = bi = None
    if points:
        pi = {"point_coords": torch.tensor([[[p[0] * 1008, p[1] * 1008] for p in points]], device=dev, dtype=dtype),
              "point_labels": torch.tensor([[int(p[2]) for p in points]], device=dev, dtype=torch.int32)}
    if box is not None:
        x0, y0, x1, y1 = box
        bi = torch.tensor([[[x0 * 1008, y0 * 1008], [x1 * 1008, y1 * 1008]]], device=dev, dtype=dtype)
    heads = enc["net"].tracker._forward_sam_heads
    with torch.no_grad():
        n = 0 if pi is None else pi["point_labels"].size(1)
        _, logit, _, _ = heads(backbone_features=enc["feat"], point_inputs=pi, mask_inputs=None, box_inputs=bi,
                               high_res_features=enc["high_res"], multimask_output=(0 < n <= 1))
        _, logit, _, _ = heads(backbone_features=enc["feat"], point_inputs=None, mask_inputs=logit, box_inputs=None,
                               high_res_features=enc["high_res"], multimask_output=False)
        m = F.interpolate(logit.float(), size=(h, w), mode="bilinear", align_corners=False)
    return torch.sigmoid(m[0, 0]).cpu()


def _sam_multi(enc, h, w, pt):
    """SAM's three answers for one tap (a part, the middle, the whole thing) as soft masks, each refined once.
    Same decoder call as ComfyUI's forward_sam_heads, minus its pick of the 'best' one."""
    import comfy.model_management as mm

    from comfy.ops import cast_to_input

    mm.load_model_gpu(enc["model"])
    dev, dtype = enc["dev"], enc["dtype"]
    tr = enc["net"].tracker
    feat = enc["feat"]
    # SAM 3.1 (multiplex) keeps its click heads under interactive_*, SAM 3 under the plain names
    penc = getattr(tr, "interactive_sam_prompt_encoder", None) or tr.sam_prompt_encoder
    pdec = getattr(tr, "interactive_sam_mask_decoder", None) or tr.sam_mask_decoder
    coords = torch.tensor([[[pt[0] * 1008, pt[1] * 1008]]], device=dev, dtype=dtype)
    labels = torch.tensor([[1]], device=dev, dtype=torch.int32)
    with torch.no_grad():
        sparse, dense = penc(points=(coords, labels), boxes=None, masks=None)
        pe = cast_to_input(penc.get_dense_pe(), feat)
        low, _ious, _tok, _obj = pdec(
            image_embeddings=feat, image_pe=pe, sparse_prompt_embeddings=cast_to_input(sparse, feat),
            dense_prompt_embeddings=cast_to_input(dense, feat), high_res_features=enc["high_res"],
            multimask_output=True, return_all=True)
        out = []
        for i in range(low.shape[1]):
            up = F.interpolate(low[:, i:i + 1].float(), size=(1008, 1008), mode="bilinear", align_corners=False)
            _, logit, _, _ = tr._forward_sam_heads(backbone_features=feat, point_inputs=None, mask_inputs=up.to(dtype),
                                                   box_inputs=None, high_res_features=enc["high_res"],
                                                   multimask_output=False)
            m = F.interpolate(logit.float(), size=(h, w), mode="bilinear", align_corners=False)
            out.append(torch.sigmoid(m[0, 0]).cpu())
    return out


def _parts(key, image_hw3, seg_choice):
    """Segformer class probabilities (C, h, w) at up to SEG_MAX, plus the class names."""
    if key in _seg_cache:
        return _seg_cache[key]
    from PIL import Image

    from . import lc_person
    from .lc_matting import get_device

    folder = lc_models.resolve_hf("segformer_b2_clothes", seg_choice)
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if folder not in lc_person._seg_cache:
        lc_person._seg_cache.clear()
        lc_person._seg_cache[folder] = (SegformerImageProcessor.from_pretrained(folder),
                                        SegformerForSemanticSegmentation.from_pretrained(folder).eval())
    proc, model = lc_person._seg_cache[folder]
    h, w = image_hw3.shape[:2]
    s = min(1.0, SEG_MAX / float(max(h, w)))
    wh, ww = max(1, round(h * s)), max(1, round(w * s))
    pil = Image.fromarray((image_hw3.numpy() * 255).round().astype(np.uint8))
    dev = get_device()
    model.to(dev)
    try:
        with torch.no_grad():
            logits = model(pixel_values=proc(images=pil, return_tensors="pt")["pixel_values"].to(dev)).logits
            prob = F.interpolate(logits, size=(wh, ww), mode="bilinear", align_corners=False).softmax(1)[0]
    finally:
        model.to("cpu")
    names = [model.config.id2label[i].lower() for i in range(prob.shape[0])]
    out = (prob.float().cpu(), names)
    _seg_cache.clear()
    _seg_cache[key] = out
    return out


def _group(parts, labels, h, w):
    probs, names = parts
    ids = [i for i, n in enumerate(names) if (n != "background" if labels is None else n in labels)]
    if not ids:
        return torch.zeros(h, w)
    p = probs[ids].sum(0)
    p = F.interpolate(p[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return p.clamp(0, 1)


# ---------- geometry ----------

def _poly_mask(pts, h, w, slack):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([[p[0] * w, p[1] * h] for p in pts], np.int32)], 1)
    r = int(round(slack * (h * h + w * w) ** 0.5))
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return torch.from_numpy(m).float()


def _inner_point(m):
    """The point deepest inside the mask, normalized."""
    a = m.numpy().astype(np.uint8)
    if not a.any():
        return None
    d = cv2.distanceTransform(np.pad(a, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    h, w = a.shape
    return ((x + 0.5) / w, (y + 0.5) / h)


def _bbox(m):
    ys, xs = torch.nonzero(m, as_tuple=True)
    h, w = m.shape
    return (xs.min().item() / w, ys.min().item() / h, (xs.max().item() + 1) / w, (ys.max().item() + 1) / h)


def _scaled_box(b, s):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw, hh = (b[2] - b[0]) * s / 2, (b[3] - b[1]) * s / 2
    return (max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh))


def _lasso_prompt(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    box = (min(xs), min(ys), max(xs), max(ys))
    probe = _poly_mask(pts, 256, 256, 0.0)
    ip = _inner_point(probe) or ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return box, ip


# ---------- recipes ----------

def _seed(sel):
    if sel.get("t") == "lasso":
        return {"t": "lasso", "pts": sel["pts"]}
    return {"t": "point", "pt": sel["pt"]}


def _render(step, ctx, h, w):
    """Mask of one recipe step at (h, w): cut at the node's threshold, then cleaned (smooth)."""
    return _tidy(_render_soft(step, ctx, h, w) > ctx["threshold"], ctx["smooth"], h, w)


def _tidy(hard, smooth, h, w):
    """Pinholes filled, stray specks dropped, jagged edges rounded. Measured against the picture size so the preview
    and the full-size run clean the same amount. Nothing is eroded: thin strands that belong stay."""
    a = hard.numpy().astype(np.uint8)
    if smooth <= 0 or not a.any():
        return torch.from_numpy(a.astype(np.float32))
    r = max(1, int(round(smooth * max(h, w) / DECIDE_MAX)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k)  # pinholes and cracks
    small = (2 * r + 1) ** 2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= small  # stray specks
    a = keep[lab].astype(np.uint8)
    inv = (1 - a).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    limit = max(small * 4, 0.03 * float(a.sum()))  # gaps between hair strands, dark folds
    for i in range(1, n):  # holes inside the selection (not the outside, which touches the border)
        x, y, bw, bh, area = stats[i]
        if area < limit and x > 0 and y > 0 and x + bw < w and y + bh < h:
            a[lab == i] = 1
    soft = cv2.GaussianBlur(a.astype(np.float32), (0, 0), r * 0.6)
    return torch.from_numpy((soft > 0.5).astype(np.float32))


def _render_soft(step, ctx, h, w):
    """Soft (0-1) mask of one recipe step at (h, w)."""
    kind = step["k"]
    if kind == "lasso":
        box, ip = _lasso_prompt(step["pts"])
        m = _sam(ctx["enc"](), h, w, points=[(ip[0], ip[1], 1)], box=box)
        return m * _poly_mask(step["pts"], h, w, LASSO_SLACK)
    if kind == "point":
        return _sam(ctx["enc"](), h, w, points=[(step["pt"][0], step["pt"][1], 1)])
    if kind == "pt3":  # one of SAM's three answers for the tap
        return _sam_multi(ctx["enc"](), h, w, step["pt"])[int(step["i"])]
    if kind == "sam":
        return _sam(ctx["enc"](), h, w, points=[tuple(step["pt"]) + (1,)], box=tuple(step["box"]))
    if kind == "seg":
        labels = None if step["labels"] is None else set(step["labels"])
        m = _group(ctx["parts"](), labels, h, w)
        if step.get("near"):  # only the person the selection is on
            m = m * _component((_group(ctx["parts"](), None, h, w) > 0.5).float(), step["near"])
        return m
    return torch.zeros(h, w)


def _component(person, near):
    """Connected parts of the person mask that touch the normalized point."""
    a = person.numpy().astype(np.uint8)
    n, lab = cv2.connectedComponents(a)
    h, w = a.shape
    x, y = min(w - 1, int(near[0] * w)), min(h - 1, int(near[1] * h))
    k = lab[y, x]
    if k == 0:  # the point fell on a gap: take the component closest to it
        ys, xs = np.nonzero(lab)
        if not len(xs):
            return torch.zeros(h, w)
        i = int(((xs - x) ** 2 + (ys - y) ** 2).argmin())
        k = lab[ys[i], xs[i]]
    return torch.from_numpy((lab == k).astype(np.float32))


def _ladder(seed, ctx, use_person):
    """The growth steps for one selection, decided at DECIDE_MAX: [(name, recipe step)]."""
    h, w = ctx["dh"], ctx["dw"]
    first = {"k": seed["t"], **({"pts": seed["pts"]} if seed["t"] == "lasso" else {"pt": seed["pt"]})}
    tap_cands = []
    if seed["t"] == "point":
        steps = [{"k": "pt3", "pt": seed["pt"], "i": i} for i in range(3)]
        rendered = [(st, _render(st, ctx, h, w)) for st in steps]
        rendered = [(st, m) for st, m in rendered if bool(m.any())]
        if rendered:
            rendered.sort(key=lambda t: float(t[1].sum()))
            first = rendered[0][0]
            tap_cands = [(None, st, m) for st, m in rendered[1:]]
    cur = _render(first, ctx, h, w)
    if not bool(cur.any()):
        return [("nothing found", first)]
    total = float(h * w)
    person = (_group(ctx["parts"](), None, h, w) > 0.5).float() if use_person else None
    on_person = person is not None and float((cur * person).sum()) >= 0.3 * float(cur.sum())
    seg_cands = []
    if on_person:
        near = _inner_point(cur * person) or _inner_point(cur)
        for name, labels in PERSON_STEPS:
            step = {"k": "seg", "labels": None if labels is None else sorted(labels), "near": list(near)}
            seg_cands.append((name, step, _render(step, ctx, h, w)))
    out = [(_name(cur, ctx, on_person) or "picked", first)]
    for _ in range(MAX_STEPS - 1):
        area = float(cur.sum())
        cands = list(seg_cands) + list(tap_cands)
        b, ip = _bbox(cur), _inner_point(cur)
        for s in SAM_SCALES:
            step = {"k": "sam", "box": list(_scaled_box(b, s)), "pt": list(ip)}
            cands.append((None, step, _render(step, ctx, h, w)))
        best = None
        for name, step, m in cands:
            if float((m * cur).sum()) < COVER_MIN * area:
                continue
            new = _attached(torch.maximum(cur, m), cur)
            na = float(new.sum())
            if na < GROW_MIN * area or na > 0.92 * total:
                continue
            if best is None or na < best[0]:
                best = (na, name, step, new)
        if best is None:
            break
        _, name, step, cur = best
        out.append((name or _name(cur, ctx, on_person) or "bigger", step))
        if step["k"] == "seg" and step["labels"] is None:  # the whole person is the top
            break
    return out


def _attached(new, cur):
    """Only the parts of the grown mask that touch the current selection: a step never jumps to a separate object."""
    a = new.numpy().astype(np.uint8)
    n, lab = cv2.connectedComponents(a)
    near = cv2.dilate(cur.numpy().astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    keep = np.unique(lab[near & (a > 0)])
    return torch.from_numpy(np.isin(lab, keep[keep > 0]).astype(np.float32))


def _name(m, ctx, on_person):
    if not on_person:
        return None
    probs, names = ctx["parts"]()
    h, w = m.shape
    p = F.interpolate(probs[None], size=(h, w), mode="bilinear", align_corners=False)[0].argmax(0)
    picked = p[m > 0.5]
    if not picked.numel():
        return None
    counts = torch.bincount(picked, minlength=len(names)).float()
    i = int(counts.argmax())
    return names[i] if counts[i] >= 0.5 * picked.numel() and names[i] != "background" else None


def _sel_mask(sel, ladder, ctx, h, w):
    level = max(0, min(int(sel.get("level", 0)), len(ladder) - 1))
    m = torch.zeros(h, w)
    for i, (_name_, step) in enumerate(ladder[:level + 1]):
        r = torch.maximum(m, _render(step, ctx, h, w))
        m = r if i == 0 or not bool(m.any()) else _attached(r, m)
    return m


def _combine(sels, masks, h, w):
    out = torch.zeros(h, w)
    for sel, m in zip(sels, masks):
        out = out * (1.0 - m) if sel.get("op") == "sub" else torch.maximum(out, m)
    return out


class _Ctx:
    """Lazy SAM 3 features and Segformer parts for one picture."""

    def __init__(self, key, image_hw3, sam3_choice, seg_choice, threshold=0.5, smooth=2):
        h, w = image_hw3.shape[:2]
        s = min(1.0, DECIDE_MAX / float(max(h, w)))
        self.d = {"dh": max(1, round(h * s)), "dw": max(1, round(w * s)),
                  "threshold": float(threshold), "smooth": int(smooth),
                  "enc": lambda: _encode(key, image_hw3, sam3_choice),
                  "parts": lambda: _parts(key, image_hw3, seg_choice)}
        self.key = key
        self.person = seg_choice

    def __getitem__(self, k):
        return self.d[k]

    def ladder(self, sel, use_person):
        seed = _seed(sel)
        ck = (self.key, json.dumps(seed, sort_keys=True), bool(use_person), self.d["threshold"], self.d["smooth"])
        if ck not in _ladder_cache:
            if len(_ladder_cache) > 64:
                _ladder_cache.clear()
            _ladder_cache[ck] = _ladder(seed, self, use_person)
        return _ladder_cache[ck]


def _parse(text):
    try:
        sels = json.loads(text or "[]")
    except Exception:
        return []
    out = []
    for s in sels if isinstance(sels, list) else []:
        if not isinstance(s, dict):
            continue
        if s.get("t") == "lasso" and isinstance(s.get("pts"), list) and len(s["pts"]) >= 3:
            out.append(s)
        elif s.get("t") == "point" and isinstance(s.get("pt"), list) and len(s["pt"]) == 2:
            out.append(s)
    return out


def _quantize(image_hw3):
    """8-bit, so the run decides on exactly the picture the preview saw."""
    return (image_hw3.clamp(0, 1) * 255).round() / 255


# ---------- node ----------

def _person_choices():
    return ["off"] + lc_models.hf_choices("segformer_b2_clothes")


class LCTouchMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Picture to mask. A Load Image upstream shows up right away, anything else after one run."}),
                "sam3_model": (lc_models.sam3_choices(), {
                    "tooltip": "Finds the edges. The same SAM 3 model LC Segment Anything and LC Smart Detailer use."}),
                "person_parts": (_person_choices(), {
                    "tooltip": "On a person, taps grow by body part (face, hair, head, clothes, person) with the Segformer "
                               "model LC Person Mask uses. off: SAM 3 decides every step."}),
                "block": ("BOOLEAN", {
                    "default": False, "label_on": "if_empty_mask", "label_off": "never",
                    "tooltip": "if_empty_mask: when nothing is selected, or a new picture comes in from a run, everything "
                               "downstream stops so you can select on the picture shown, then Run (with LC123 it carries "
                               "on with the same seeds). A new picture in a Load Image shows here right away, so it doesn't stop."}),
                "grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                                 "tooltip": "Grow (or shrink, below 0) the finished mask by this many pixels."}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                                      "tooltip": "Edge softness in pixels."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the mask (everything but the selection)."}),
                "selections": ("STRING", {"default": "[]", "tooltip": "Internal: what you drew and tapped."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": (
                    "How sure a pixel must be to join. Lower takes in more (wispy hair, soft edges), higher is tighter.")}),
                "smooth": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1, "tooltip": (
                    "Cleans the edge: fills pinholes, drops stray specks and rounds jagged edges, without shrinking the "
                    "selection. 0 = raw.")}),
                "refine": (METHODS, {"default": "guided_filter", "tooltip": (
                    "Edge finish on the run (the preview shows the selection before it). guided_filter: fast, follows "
                    "the picture's edges. vitmatte: best for hair, slower. none: the hard selection.")}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        return lc_hold.changed(unique_id)

    @classmethod
    def VALIDATE_INPUTS(cls, person_parts=None):
        if person_parts in (None, "off"):
            return True
        return lc_models.validate_models(person_parts=("segformer_b2_clothes", person_parts))

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("control_image", "control_mask")
    FUNCTION = "make"
    CATEGORY = "LC MaskMaker/mask"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Mask one thing exactly: draw around it and SAM 3 finds its edges. Tap it to grow to the next bigger thing "
        "(face, + hair, head, person), double-tap to step back. Ctrl + draw cuts out."
    )

    def make(self, image, sam3_model, person_parts, block, grow, feather, invert, selections, threshold=0.5, smooth=2,
             refine="guided_filter", unique_id=None, prompt=None):
        b, h, w, _c = image.shape
        sels = _parse(selections)
        # the selections were made on the last picture (a Load Image's new picture shows on the node before the run)
        new_picture = lc_hold.new_picture(unique_id, image, prompt)
        if block and new_picture and sels:
            print("[LC Touch Mask] New picture: stopped so you can select on it, then Run.")
            sels = []  # blocked either way: skip SAM
        use_person = person_parts != "off"
        frame = _quantize(image[0, ..., :3].cpu().float())
        mask = torch.zeros(h, w)
        if sels:
            key = ("run", h, w, float(frame.sum()), float(frame[::7, ::7].std()))
            with _lock:
                ctx = _Ctx(key, frame, sam3_model, person_parts if use_person else None, threshold, smooth)
                masks = [_sel_mask(s, ctx.ladder(s, use_person), ctx, h, w) for s in sels]
            mask = _combine(sels, masks, h, w)
        empty = not bool(mask.any())
        if not empty and refine != "none":
            mask = _refine(image[:1, ..., :3], mask, refine)
        mask = _finish(mask, grow, feather)
        if invert:
            mask = 1.0 - mask
        mask_b = mask[None].repeat(b, 1, 1)
        out = (image, mask_b)
        ui = {"src_size": [{"width": int(w), "height": int(h)}]}
        if block and (empty or new_picture):
            from comfy_execution.graph import ExecutionBlocker

            out = (ExecutionBlocker(None),) * 2
            lc_hold.hold(unique_id, ui)
        try:
            from nodes import PreviewImage

            ui["lc_preview"] = PreviewImage().save_images(image[:1], filename_prefix="lc_touch_src")["ui"]["images"]
        except Exception:
            pass
        return {"ui": ui, "result": out}


def _refine(image, mask, method):
    """LC Mask Refine's edge finish (guided filter or VITMatte on the edge band only), as LC Person Mask does it."""
    from .lc_matting import get_device, refine_mask

    dev = get_device()
    folder = lc_models.resolve_vitmatte(lc_models.vitmatte_choices()[0]) if method == "vitmatte" else None
    img = image[0].permute(2, 0, 1).unsqueeze(0).to(dev).float()
    m = mask.to(dev)[None, None].float()
    out, _tri = refine_mask(img, m, method, folder, 0.0, 0.0, 0.75, 10.0, 10.0, 0.001, 0.5, 0.01, 0.99, 0.0, 2.0, dev)
    return out[0, 0].float().cpu().clamp(0, 1)


def _finish(mask, grow, feather):
    if grow:
        r = abs(int(grow))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        a = mask.numpy().astype(np.float32)
        a = cv2.dilate(a, k) if grow > 0 else cv2.erode(a, k)
        mask = torch.from_numpy(a).float()
    if feather > 0:
        s = float(feather)
        mask = torch.from_numpy(cv2.GaussianBlur(mask.numpy(), (0, 0), s / 2))
    return mask.clamp(0, 1)


# ---------- live preview route ----------

def _load_view(meta):
    import folder_paths
    from PIL import Image

    base = folder_paths.get_directory_by_type(meta.get("type") or "input")
    if base is None:
        raise ValueError("unknown image folder")
    path = os.path.abspath(os.path.join(base, meta.get("subfolder") or "", meta.get("filename") or ""))
    if os.path.commonpath([path, os.path.abspath(base)]) != os.path.abspath(base) or not os.path.isfile(path):
        raise ValueError("picture not found")
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    return ("view", path, os.path.getmtime(path)), t


def _overlay(final, active, dh, dw):
    a = np.zeros((dh, dw, 4), np.uint8)
    f = final.numpy() > 0.5
    a[f] = (255, 40, 70, 150)
    if active is not None:
        m = active.numpy().astype(np.uint8)
        edge = cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8))
        a[edge > 0] = (255, 210, 74, 255)
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _preview(body):
    key, frame = _load_view(body.get("image") or {})
    use_person = (body.get("person_parts") or "off") != "off"
    ctx = _Ctx(key, frame, body.get("sam3_model"), body.get("person_parts") if use_person else None,
               float(body.get("threshold", 0.5)), int(body.get("smooth", 2)))
    dh, dw = ctx["dh"], ctx["dw"]
    sels = _parse(json.dumps(body.get("selections") or []))
    ladders = [ctx.ladder(s, use_person) for s in sels]
    masks = [_sel_mask(s, l, ctx, dh, dw) for s, l in zip(sels, ladders)]
    active = len(sels) - 1 if sels else None
    tap = body.get("tap")
    if tap:
        x, y = min(dw - 1, int(tap["x"] * dw)), min(dh - 1, int(tap["y"] * dh))
        hit = next((i for i in range(len(sels) - 1, -1, -1) if masks[i][y, x] > 0.5), None)
        d = int(tap.get("dir", 1))
        if hit is not None:
            lvl = int(sels[hit].get("level", 0)) + d
            if lvl < 0:
                sels.pop(hit)
                ladders.pop(hit)
                masks.pop(hit)
                active = len(sels) - 1 if sels else None
            else:
                sels[hit]["level"] = min(lvl, len(ladders[hit]) - 1)
                masks[hit] = _sel_mask(sels[hit], ladders[hit], ctx, dh, dw)
                active = hit
        elif d > 0:
            s = {"t": "point", "pt": [round(float(tap["x"]), 4), round(float(tap["y"]), 4)],
                 "op": "sub" if tap.get("sub") else "add", "level": 0}
            sels.append(s)
            ladders.append(ctx.ladder(s, use_person))
            masks.append(_sel_mask(s, ladders[-1], ctx, dh, dw))
            active = len(sels) - 1
    final = _combine(sels, masks, dh, dw)
    status = ""
    if active is not None:
        lad = ladders[active]
        lvl = max(0, min(int(sels[active].get("level", 0)), len(lad) - 1))
        status = f"{lad[lvl][0]} · {lvl + 1} of {len(lad)}"
        if sels[active].get("op") == "sub":
            status = "cut: " + status
    return {"selections": sels, "overlay": _overlay(final, masks[active] if active is not None else None, dh, dw),
            "status": status}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/lc_maskmaker/touch_mask")
    async def _touch_mask_route(request):
        body = await request.json()

        def work():
            with _lock:
                return _preview(body)

        try:
            res = await asyncio.get_event_loop().run_in_executor(None, work)
        except Exception as e:
            print(f"[LC Touch Mask] preview failed: {e}")
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(res)
except Exception as _route_e:  # no server (tests)
    print(f"[LC Touch Mask] live preview off: {_route_e}")


NODE_CLASS_MAPPINGS = {"LCTouchMask": LCTouchMask}
NODE_DISPLAY_NAME_MAPPINGS = {"LCTouchMask": "LC Touch Mask (BETA) 👆"}
