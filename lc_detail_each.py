"""
LC Detail Finder / LC Detail Sorter / LC Detailer For Each (pipe)
-----------------------------------------------------------------
A detailer per character, the way Impact's SEGS chain does it (detector -> ordered filter -> DetailerForEach), on
LC Smart Detailer's engine:

  LC Detail Finder        SAM 3 finds every match once (every face). With several words (face, hair) the first one is
                          what gets counted and the others join the match they belong to (each face gets its own hair).
  LC Detail Sorter        puts them in order (left to right, right to left, top to bottom, largest first) and breaks
                          them out: output 1 = the first, 2 = the second ...
  LC Detailer For Each    redraws what it is given with that character's model (LoRA), prompt and seed, on its
                          own or on an LC pipe. Chain one per character.

The crop, the redraw and the blend are LC Smart Detailer's code (lc_inpaint). LC Smart Detailer itself is not changed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import lc_character_detailer as C
from . import lc_inpaint as I
from . import lc_models, lc_pipe_keys, lc_sam3

FINDS = "LC_SAM3_MASKS"
ORDERS = ["left to right", "right to left", "top to bottom", "largest first"]
SLOTS = 8
COLORS = [(255, 210, 40), (80, 200, 255), (255, 110, 150), (140, 230, 110), (200, 150, 255), (255, 160, 60),
          (90, 230, 210), (240, 240, 240)]


def _draw(image, items, labels, colors):
    """The first frame with each match outlined and labelled at its centre."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    arr = (image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    h, w = arr.shape[:2]
    for it, col in zip(items, colors):
        m = it["mask"][None, None].float()
        edge = ((F.max_pool2d(m, 7, 1, 3) - m) > 0)[0, 0].cpu().numpy()
        arr[edge] = col
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, w // 28))
    except OSError:
        font = ImageFont.load_default()
    for it, lab, col in zip(items, labels, colors):
        x, y = it["stats"][0] * w, it["stats"][1] * h
        bb = d.textbbox((0, 0), lab, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 6
        d.rounded_rectangle([x - tw / 2 - pad, y - th / 2 - pad, x + tw / 2 + pad, y + th / 2 + pad], radius=6,
                            fill=col, outline=(0, 0, 0), width=2)
        d.text((x - tw / 2, y - th / 2 - bb[1]), lab, fill=(0, 0, 0), font=font)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


def _sorted(items, order):
    key = {"left to right": lambda t: t["stats"][0], "right to left": lambda t: -t["stats"][0],
           "top to bottom": lambda t: t["stats"][1], "largest first": lambda t: -t["stats"][2]}[order]
    return sorted(items, key=key)


def _check(masks, name):
    if not isinstance(masks, dict) or masks.get("_type") != FINDS:
        raise ValueError(f"[{name}] The SAM 3 masks must come from LC Detail Finder or LC Detail Sorter.")
    return masks


class LCDetailFinder(I._Base):
    RETURN_TYPES = (FINDS, "INT")
    RETURN_NAMES = ("sam3_masks", "count")
    OUTPUT_TOOLTIPS = ("The SAM 3 mask of every match, for LC Detail Sorter (or straight into LC Detailer For Each to redraw them all).",
                       "How many were found (first image).")
    FUNCTION = "find"
    DESCRIPTION = ("Finds every match once with SAM 3 (every face) for a detailer per character. Then LC Detail Sorter "
                   "puts them in order and breaks them out, one LC Detailer For Each per character.")

    @classmethod
    def INPUT_TYPES(cls):
        f = I._find_inputs()
        f["prompt"] = ("STRING", {"default": "face", "multiline": False, "tooltip": (
            "What to find, e.g. face. With several words (face, hair) the first is what gets counted and the others "
            "join the match they belong to: each face gets its own hair.")})
        return {"required": {"image": ("IMAGE",), **f}}

    def find(self, image, prompt, sam3_model, threshold):
        words = I._words(prompt)
        if not words:
            raise ValueError("[LC Detail Finder] The prompt is empty. Type what to find, e.g. face.")
        path = lc_models.resolve_sam3(sam3_model)
        frames = []
        for i in range(image.shape[0]):
            per_word = [[I._clean(m) for m in lc_sam3.objects(image[i:i + 1], [w], path, threshold)] for w in words]
            items = []
            for m in per_word[0]:
                st = C._stats(m)
                if not st:
                    continue
                merged = m.clone()
                for extra in per_word[1:]:
                    for e in extra:
                        if C._belongs(e, m):
                            merged = torch.maximum(merged, e)
                items.append({"mask": merged.cpu(), "stats": st})
            frames.append(_sorted(items, "left to right"))
        masks = {"_type": FINDS, "frames": frames, "prompt": prompt.strip(), "size": tuple(image.shape[1:3])}
        n = len(frames[0]) if frames else 0
        ui = {}
        try:
            prev = _draw(image, frames[0], [""] * n, [(80, 200, 255)] * n) if n else image[:1, ..., :3]
            ui = self.save_images(prev, filename_prefix="lc_detail_find")["ui"]
        except Exception as e:
            print(f"[LC Detail Finder] preview skipped: {e}")
        print(f"[LC Detail Finder] {n} found for '{prompt.strip()}'.")
        return {"ui": ui, "result": (masks, n)}

    @classmethod
    def VALIDATE_INPUTS(cls, sam3_model=None):
        return lc_models.validate_models(sam3_model=("sam3", sam3_model))


class LCDetailSorter(I._Base):
    RETURN_TYPES = tuple([FINDS] * SLOTS)
    RETURN_NAMES = tuple(str(k + 1) for k in range(SLOTS))
    OUTPUT_TOOLTIPS = tuple(f"The SAM 3 mask of match {k + 1} in the chosen order (empty when there are fewer). Wire it "
                            f"to that character's LC Detailer For Each." for k in range(SLOTS))
    FUNCTION = "sort"
    DESCRIPTION = ("Puts LC Detail Finder's matches in order and breaks them out: output 1 = the first, 2 = the second. "
                   "The preview numbers them the same way.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sam3_masks": (FINDS, {"tooltip": "From LC Detail Finder."}),
            "pick": (ORDERS, {"default": "left to right", "tooltip": (
                "The order the outputs count in. left to right: 1 = the leftmost. largest first: 1 = the biggest "
                "(usually the closest).")}),
        }, "optional": {"image": ("IMAGE", {"tooltip": "Optional: the picture, for the numbered preview."})}}

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    def sort(self, sam3_masks, pick, image=None):
        _check(sam3_masks, "LC Detail Sorter")
        frames = [_sorted(items, pick) for items in sam3_masks["frames"]]
        outs = [dict(sam3_masks, frames=[f[k:k + 1] for f in frames]) for k in range(SLOTS)]
        ui = {}
        if image is not None and frames and frames[0]:
            try:
                n = len(frames[0])
                prev = _draw(image, frames[0], [str(k + 1) for k in range(n)], [COLORS[k % len(COLORS)] for k in range(n)])
                ui = self.save_images(prev, filename_prefix="lc_detail_sort")["ui"]
            except Exception as e:
                print(f"[LC Detail Sorter] preview skipped: {e}")
        if frames and len(frames[0]) > SLOTS:
            print(f"[LC Detail Sorter] {len(frames[0])} found: only the first {SLOTS} have an output.")
        return {"ui": ui, "result": tuple(outs)}


def _with_pipe_tips(inputs):
    """The sampler widgets, with a line on how a wired pipe takes over (unless something is plugged in here)."""
    out = {}
    for k, (t, opts) in inputs.items():
        if k in ("sampler_name", "scheduler", "steps", "seed"):
            opts = dict(opts, tooltip=opts.get("tooltip", "") + "\nWith a pipe wired, the pipe's value is used unless "
                                                               "something is plugged in here.")
        out[k] = (t, opts)
    return out


class LCDetailerForEach(I._Base):
    RETURN_TYPES = (I.PIPE_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "image", "mask")
    DESCRIPTION = ("Redraws what it is given (one character from LC Detail Sorter, or everything from LC Detail Finder) "
                   "with that character's model (its LoRA branch), their own conditioned prompt and their own seed. "
                   "Works on its own, or on an LC pipe: a wired pipe fills in the model, VAE, prompts and picture, and "
                   "anything wired here wins. Same engine as LC Smart Detailer.")

    @classmethod
    def INPUT_TYPES(cls):
        opt = I._optional(pos=True)
        opt.pop("cfg", None)  # the node's own cfg widget is used
        return {
            "required": {
                "sam3_mask": (FINDS, {"tooltip": "What to redraw: one output of LC Detail Sorter (one character), or LC "
                                                 "Detail Finder's sam3_masks (all of them)."}),
                **{k: v for k, v in I._mask_inputs().items() if k != "blend"},
                **_with_pipe_tips(I._sampler_inputs()),
            },
            "optional": {
                "pipe": (I.PIPE_TYPE, {"tooltip": "Optional LC pipe: model, VAE, positive, negative and picture come from "
                                                  "here unless wired on this node. Chain the detailers through it."}),
                "image": ("IMAGE", {"tooltip": "The picture. Empty = the pipe's picture."}),
                **opt,
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    def run(self, sam3_mask, pipe=None, image=None, prompt=None, unique_id=None, **kw):
        name = "LC Detailer For Each"
        finds = _check(sam3_mask, name)
        p = pipe if isinstance(pipe, dict) else {}
        image = lc_pipe_keys.pipe_image(p, image) if p else image
        if image is None:
            raise ValueError(f"[{name}] No picture: wire image, or a pipe that carries one.")
        if tuple(image.shape[1:3]) != tuple(finds["size"]):
            raise ValueError(f"[{name}] The picture is {image.shape[2]}x{image.shape[1]} but the SAM 3 masks were made on "
                             f"{finds['size'][1]}x{finds['size'][0]}: find on the same picture.")

        def pick(key, pipe_key):
            v = kw.get(key)
            return v if v is not None else p.get(pipe_key)

        # sampler / scheduler / steps / seed: with a pipe, the pipe's value, unless something is plugged into that
        # widget on this node (a plugged-in value always beats the pipe). No pipe: the node's own values.
        me = (prompt or {}).get(str(unique_id), {}).get("inputs", {}) if prompt else {}
        plugged = {k for k, v in me.items() if isinstance(v, list)}
        for key, pipe_keys in (("sampler_name", ("sampler_name",)), ("scheduler", ("scheduler",)),
                               ("steps", ("detailer_steps", "total_steps")), ("seed", ("seed",))):
            if p and key not in plugged:
                v = next((p.get(k) for k in pipe_keys if p.get(k) not in (None, "")), None)
                if v is not None:
                    kw[key] = v

        model, vae = pick("model", "model_1"), pick("vae", "vae_1")
        positive, negative = pick("positive", "positive"), pick("negative", "negative")
        missing = [n for n, v in (("model", model), ("vae", vae)) if v is None]
        if missing:
            raise ValueError(f"[{name}] No {' or '.join(missing)}: wire it, or a pipe that carries it.")
        if positive is None and kw.get("inpaint_positive") is None:
            raise ValueError(f"[{name}] No prompt: wire inpaint_positive (or positive, or a pipe that carries one).")
        kw["blend"] = 1.0  # the whole redraw is pasted in: lower denoise for a lighter touch
        s = I._settings(kw, model, vae, positive, negative, kw["sampler_name"], kw["scheduler"], kw["steps"],
                        kw["cfg"], kw["seed"])
        frames = finds["frames"]
        masks = [[it["mask"] for it in frames[min(i, len(frames) - 1)]] for i in range(image.shape[0])]
        out, mask, done = I._run(image, masks, s)
        note = f"{done} redrawn" if done else "Nothing to redraw (this output is empty). Image passed through."
        protect = lc_pipe_keys.merge_masks(lc_pipe_keys.protect_for(p, image), mask) if p else mask
        out_pipe = lc_pipe_keys.updated(p if p else None, out, protect=protect)
        return {"ui": I._preview(self, image, out, mask, note), "result": (out_pipe, out, mask)}


NODE_CLASS_MAPPINGS = {
    "LCDetailFinder": LCDetailFinder,
    "LCDetailSorter": LCDetailSorter,
    "LCDetailerForEach": LCDetailerForEach,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LCDetailFinder": "LC Detail Finder (BETA) 🎯",
    "LCDetailSorter": "LC Detail Sorter (BETA) 🔢",
    "LCDetailerForEach": "LC Detailer For Each (BETA) 🩹",
}
