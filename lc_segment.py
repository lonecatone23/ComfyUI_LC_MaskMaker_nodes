"""
LC Segment Anything
-------------------
Cut things out of an image by describing them ("hair, eyes, bikini").

Two engines:
  grounding_dino + sam : GroundingDINO finds a box for each word, SAM turns each box into a mask
                         (Hugging Face `transformers`, Apache-2.0 models, downloaded on demand).
  sam3                 : SAM 3 finds and masks them in one step, using ComfyUI's own SAM 3 support
                         (the SAM 3.1 checkpoint is one click, or use your own sam3.safetensors in models/sam3).

The raw mask can then be refined (guided filter or VITMatte, with a trimap) and is shown on the node
as a before/after wipe.
"""

import re

import numpy as np
import torch
from nodes import PreviewImage

from . import lc_models
from .lc_matting import METHODS, get_device, refine_mask
from .lc_preview_util import wipe_preview

ENGINES = ["grounding_dino + sam", "sam3"]

_hf_cache = {}    # kind -> (folder, processor, model); one model per kind kept
_sam3_cache = {}  # path -> (model patcher, clip)


def _tokens(prompt):
    parts = [p.strip() for p in re.split(r"[,\n]+", prompt or "")]
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# GroundingDINO + SAM (transformers)
# --------------------------------------------------------------------------
def _hf_load(kind, folder):
    cached = _hf_cache.get(kind)
    if cached and cached[0] == folder:
        return cached[1], cached[2]
    if kind == "grounding-dino":
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        proc = AutoProcessor.from_pretrained(folder)
        model = GroundingDinoForObjectDetection.from_pretrained(folder).eval()
    else:
        from transformers import SamModel, SamProcessor

        proc = SamProcessor.from_pretrained(folder)
        model = SamModel.from_pretrained(folder).eval()
    _hf_cache[kind] = (folder, proc, model)
    return proc, model


@torch.no_grad()
def _dino_boxes(pil, tokens, folder, threshold, max_objects, dev):
    proc, model = _hf_load("grounding-dino", folder)
    model.to(dev)
    boxes = []
    try:
        for tok in tokens:
            text = tok.lower().strip().rstrip(".") + "."
            inputs = proc(images=pil, text=text, return_tensors="pt").to(dev)
            out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=float(threshold), text_threshold=0.25,
                target_sizes=[(pil.height, pil.width)],
            )[0]
            scores, bxs = res["scores"].cpu(), res["boxes"].cpu()
            order = scores.argsort(descending=True)
            if max_objects > 0:
                order = order[:max_objects]
            boxes.extend(bxs[order].tolist())
    finally:
        model.to("cpu")
    return boxes


@torch.no_grad()
def _sam_masks(pil, boxes, folder, dev):
    """Union of the SAM mask for every box. Returns (H,W) float."""
    proc, model = _hf_load("sam", folder)
    model.to(dev)
    try:
        inputs = proc(images=pil, input_boxes=[boxes], return_tensors="pt")
        sizes = (inputs["original_sizes"], inputs["reshaped_input_sizes"])
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), sizes[0].cpu(), sizes[1].cpu())[0]  # (N,1,H,W) bool
    finally:
        model.to("cpu")
    return masks.any(dim=0)[0].float()


# --------------------------------------------------------------------------
# SAM 3 (ComfyUI core)
# --------------------------------------------------------------------------
def _sam3_load(path):
    if path not in _sam3_cache:
        import comfy.sd
        import folder_paths

        out = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=False, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip = out[0], out[1]
        if clip is None:
            raise RuntimeError(f"[LC Segment Anything] {path} does not look like a SAM 3 checkpoint.")
        _sam3_cache.clear()
        _sam3_cache[path] = (model, clip)
    return _sam3_cache[path]


def _sam3_mask(image_1hwc, tokens, path, threshold):
    """Union of the SAM 3 mask for every word. Returns (H,W) float."""
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except Exception as e:
        raise RuntimeError("[LC Segment Anything] This ComfyUI has no built-in SAM 3 support. Update ComfyUI.") from e
    model, clip = _sam3_load(path)
    h, w = image_1hwc.shape[1:3]
    union = torch.zeros(h, w)
    for tok in tokens:
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(tok))
        res = SAM3_Detect.execute(model, image_1hwc[..., :3], conditioning=cond, threshold=float(threshold),
                                  refine_iterations=2, individual_masks=False)
        union = torch.maximum(union, res.args[0][0].float().cpu())
    return union


# --------------------------------------------------------------------------
class LCSegmentAnything(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to cut things out of."}),
                "prompt": ("STRING", {
                    "default": "subject", "multiline": True,
                    "tooltip": "What to select, as simple words separated by commas: hair, eyes, bikini.\nEach word is found on its own and the results are combined.",
                }),
                "engine": (ENGINES, {
                    "default": "grounding_dino + sam",
                    "tooltip": (
                        "grounding_dino + sam: GroundingDINO finds boxes, SAM makes the masks. Apache-2.0 models, downloaded when first used.\n"
                        "sam3: SAM 3 does both in one step (SAM License). Pick the SAM 3.1 download, or use your own sam3.safetensors in models/sam3."
                    ),
                }),
                "dino_model": (lc_models.hf_choices("grounding-dino"), {
                    "tooltip": "Used by grounding_dino + sam. Entries marked 'Download' are fetched the first time you run.",
                }),
                "sam_model": (lc_models.hf_choices("sam"), {
                    "tooltip": "Used by grounding_dino + sam. base is fastest, huge is the most accurate.",
                }),
                "sam3_model": (lc_models.sam3_choices(), {"tooltip": "Used by the sam3 engine."}),
                "threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "grounding_dino + sam: detection confidence. Lower finds more (and more wrong things), higher is pickier.",
                }),
                "sam3_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "sam3: detection confidence. SAM 3 scores run higher, so 0.5 is a good start. Around 0.3 it starts inventing matches for words that are not in the image.",
                }),
                "max_objects": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "grounding_dino + sam only. Keep at most this many boxes per word (best first). 0 = all.",
                }),
                "refine": (METHODS, {
                    "default": "vitmatte",
                    "tooltip": "Edge refinement of the raw mask. vitmatte is best for hair, guided_filter is fast, none keeps the raw mask.",
                }),
                "vitmatte_model": (lc_models.vitmatte_choices(), {"tooltip": "Only used when refine is vitmatte."}),
                "edge_erode": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Trimap: how far the sure-foreground is pulled in from the mask edge (pixels).",
                }),
                "edge_dilate": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Trimap: how far the unknown band reaches out past the mask edge (pixels).",
                }),
                "black_point": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.999, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or below this become 0.",
                }),
                "white_point": ("FLOAT", {
                    "default": 0.99, "min": 0.001, "max": 1.0, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or above this become 1.",
                }),
                "max_megapixels": ("FLOAT", {
                    "default": 2.0, "min": 0.25, "max": 16.0, "step": 0.25,
                    "tooltip": "vitmatte only. Larger images are matted at this size, then scaled back.",
                }),
                "preview_view": (["cutout", "mask", "trimap"], {
                    "default": "cutout",
                    "tooltip": "What the wipe on the node shows. Before is the raw mask.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cutout", "mask", "raw_mask")
    FUNCTION = "segment"
    CATEGORY = "LC MaskMaker/mask"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Select things by describing them. GroundingDINO + SAM (Apache-2.0, downloaded on demand) or SAM 3, "
        "then optional guided filter / VITMatte refine, with a before/after wipe on the node."
    )

    def segment(self, image, prompt, engine, dino_model, sam_model, sam3_model, threshold, sam3_threshold, max_objects,
                refine, vitmatte_model, edge_erode, edge_dilate, black_point, white_point, max_megapixels,
                preview_view):
        from PIL import Image

        tokens = _tokens(prompt)
        if not tokens:
            raise ValueError("[LC Segment Anything] The prompt is empty. Type what to select, e.g. hair, eyes.")
        dev = get_device()
        b, h, w, _c = image.shape

        if engine == "sam3":
            sam3_path = lc_models.resolve_sam3(sam3_model)
        else:
            dino_folder = lc_models.resolve_hf("grounding-dino", dino_model)
            sam_folder = lc_models.resolve_hf("sam", sam_model)
        folder = lc_models.resolve_vitmatte(vitmatte_model) if refine == "vitmatte" else None

        raws, masks, tris, cutouts = [], [], [], []
        for i in range(b):
            frame = image[i:i + 1]
            if engine == "sam3":
                raw = _sam3_mask(frame, tokens, sam3_path, sam3_threshold)
            else:
                pil = Image.fromarray((frame[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
                boxes = _dino_boxes(pil, tokens, dino_folder, threshold, max_objects, dev)
                raw = _sam_masks(pil, boxes, sam_folder, dev) if boxes else torch.zeros(h, w)
            if not bool(raw.any()):
                print(f"[LC Segment Anything] nothing found for '{prompt}' (frame {i}). Try a lower threshold or other words.")

            img = frame[0, ..., :3].permute(2, 0, 1).unsqueeze(0).to(dev).float()
            m = raw.to(dev).float().unsqueeze(0).unsqueeze(0)
            out, tri = refine_mask(img, m, refine, folder, 0.0, 0.0, 0.75, edge_erode, edge_dilate, 0.001, 0.5,
                                   black_point, white_point, 0.0, max_megapixels, dev)
            raws.append(raw)
            masks.append(out[0, 0].cpu())
            tris.append(tri[0, 0].cpu())
            cutouts.append(torch.cat([frame[0, ..., :3].cpu().float(), out[0, 0].cpu().unsqueeze(-1)], dim=-1))

        mask_out = torch.stack(masks, 0)
        ui = wipe_preview(self, image, raws[0], mask_out[0], trimap=tris[0], prefix="lc_segment")
        return {"ui": ui, "result": (torch.stack(cutouts, 0), mask_out, torch.stack(raws, 0))}


NODE_CLASS_MAPPINGS = {
    "LCSegmentAnything": LCSegmentAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSegmentAnything": "LC Segment Anything 🎯",
}
