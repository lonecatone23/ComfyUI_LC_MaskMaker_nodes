"""
LC Person Mask
--------------
Pick the parts of a person you want: face, hair, body skin, clothes, accessories, or the
background. Two engines:

  segformer : a Segformer human-parsing model (Hugging Face `transformers`). Works at a higher
              resolution. Its body class covers arms and legs only, so bare torso and neck
              come out as clothes.
  mediapipe : Google's selfie multiclass model, the one LayerStyle's PersonMaskUltra uses. Its body
              class includes torso and neck skin. Needs the `mediapipe` package (not installed
              by this pack).

Skin mask: mediapipe with face + body is face skin plus body skin. remove_features then cuts out the eyes, brows,
lips and teeth with SAM 3 (the model LC Segment Anything and LC Smart Detailer already use), which leaves skin only,
ready for LC Skin Beauty, LC Skin Upscale or LC Sharpen Pro.

The raw mask can then be refined with the same guided filter / VITMatte trimap as LC Mask Refine.
No preview on the node: use LC Mask Refine after it if you want the before/after wipe.
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import lc_models
from .lc_matting import METHODS, get_device, refine_mask

ENGINES = ["segformer", "mediapipe"]
WORK_MAX = 1536

# Segformer (mattmdjaga/segformer_b2_clothes) class names per part
SEGFORMER_PARTS = {
    "face": {"face"},
    "hair": {"hair"},
    "body": {"left-leg", "right-leg", "left-arm", "right-arm"},
    "clothes": {"upper-clothes", "skirt", "pants", "dress", "belt", "scarf"},
    "accessories": {"hat", "sunglasses", "left-shoe", "right-shoe", "bag"},
    "background": {"background"},
}
# MediaPipe selfie multiclass output channels
MEDIAPIPE_PARTS = {"background": 0, "hair": 1, "body": 2, "face": 3, "clothes": 4, "accessories": 5}

_seg_cache = {}  # folder -> (processor, model)

# Facial features cut out by remove_features. Tested: "nostrils" swallows the whole nose, so it is left out.
FEATURE_WORDS = ["eyes", "eyebrows", "lips", "teeth"]
FEATURE_THRESHOLD = 0.45


def _features(frame_1hwc):
    """Eyes, brows, lips and teeth as a soft (H,W) mask, grown a little so no lash or lip edge is left behind."""
    from . import lc_sam3

    choices = lc_models.sam3_choices()
    path = lc_models.resolve_sam3(choices[0])
    hard = lc_sam3.union(frame_1hwc[..., :3].cpu().float(), FEATURE_WORDS, path, FEATURE_THRESHOLD)
    if not bool(hard.any()):
        return hard
    m = F.max_pool2d(hard[None, None], 5, stride=1, padding=2)  # grow about 2 px
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    k = (k / k.sum())
    m = F.conv2d(F.pad(m, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5))
    m = F.conv2d(F.pad(m, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1))
    return m[0, 0].clamp(0, 1)


def _segformer_prob(pil, folder, parts, dev):
    """Summed probability of the chosen parts, as an (H,W) float tensor on cpu."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if folder not in _seg_cache:
        _seg_cache.clear()
        _seg_cache[folder] = (
            SegformerImageProcessor.from_pretrained(folder),
            SegformerForSemanticSegmentation.from_pretrained(folder).eval(),
        )
    proc, model = _seg_cache[folder]
    wanted = set()
    for p in parts:
        wanted |= SEGFORMER_PARTS[p]
    ids = [int(i) for i, name in model.config.id2label.items() if name.lower() in wanted]

    h, w = pil.height, pil.width
    scale = min(1.0, WORK_MAX / float(max(h, w)))
    wh, ww = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    inputs = proc(images=pil, return_tensors="pt")
    model.to(dev)
    try:
        with torch.no_grad():
            logits = model(pixel_values=inputs["pixel_values"].to(dev)).logits
            logits = F.interpolate(logits, size=(wh, ww), mode="bilinear", align_corners=False)
            prob = logits.softmax(dim=1)[0, ids].sum(0)
    finally:
        model.to("cpu")
    prob = prob.float().cpu()
    if (wh, ww) != (h, w):
        prob = F.interpolate(prob[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return prob


def _mediapipe_prob(rgb_uint8, model_path, parts):
    """Strongest confidence over the chosen parts, as an (H,W) float tensor."""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError(
            "[LC Person Mask] The mediapipe engine needs the 'mediapipe' package "
            "(python -m pip install mediapipe), or use the segformer engine."
        ) from e
    with open(model_path, "rb") as f:
        buf = f.read()
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=buf),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_uint8))
        result = segmenter.segment(image)
        best = None
        for p in parts:
            m = np.array(result.confidence_masks[MEDIAPIPE_PARTS[p]].numpy_view(), dtype=np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            best = m if best is None else np.maximum(best, m)
    return torch.from_numpy(best)


class LCPersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image with a person in it."}),
                "engine": (ENGINES, {
                    "default": "segformer",
                    "tooltip": (
                        "segformer: higher resolution. Body = arms and legs only (bare torso and neck read as clothes).\n"
                        "mediapipe: body includes torso and neck skin. Needs the mediapipe package."
                    ),
                }),
                "segformer_model": (lc_models.hf_choices("segformer_b2_clothes"), {
                    "tooltip": "Used by the segformer engine. Entries marked 'Download' are fetched the first time you run.",
                }),
                "face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                     "tooltip": "Skin below the face. See engine for what counts."}),
                "clothes": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "accessories": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                            "tooltip": "Hats, glasses, shoes, bags (and whatever else the model calls an accessory)."}),
                "background": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "confidence": ("FLOAT", {
                    "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "How sure the model must be. Lower grows the selection, higher shrinks it.",
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
                "remove_features": ("BOOLEAN", {
                    "default": False, "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Cut the eyes, eyebrows, lips and teeth out of the face, using SAM 3 (the model LC Segment "
                               "Anything and LC Smart Detailer use). With engine mediapipe and face + body on, this is a "
                               "skin-only mask for LC Skin Beauty, LC Skin Upscale or LC Sharpen Pro. Only runs when face is on.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, segformer_model=None, vitmatte_model=None):
        # a '⬇ Download' entry saved on another machine is still valid here once the model is installed
        return lc_models.validate_models(segformer_model=("segformer_b2_clothes", segformer_model), vitmatte_model=("vitmatte", vitmatte_model))

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cutout", "mask", "raw_mask")
    FUNCTION = "segment"
    CATEGORY = "LC MaskMaker/mask"
    DESCRIPTION = (
        "Select parts of a person: face, hair, body, clothes, accessories, background. "
        "Segformer or MediaPipe, with optional guided filter / VITMatte refine. "
        "Skin only: mediapipe, face + body, remove_features on (cuts eyes, brows, lips and teeth with SAM 3)."
    )

    def segment(self, image, engine, segformer_model, face, hair, body, clothes, accessories, background,
                confidence, refine, vitmatte_model, edge_erode, edge_dilate, black_point, white_point,
                max_megapixels, remove_features=False):
        from PIL import Image

        parts = [n for n, on in (("face", face), ("hair", hair), ("body", body), ("clothes", clothes),
                                 ("accessories", accessories), ("background", background)) if on]
        dev = get_device()
        b, h, w, _c = image.shape

        if not parts:
            print("[LC Person Mask] No part is enabled, so the mask is empty.")
        if engine == "mediapipe":
            model_path = lc_models.resolve_mediapipe() if parts else None
        else:
            folder_seg = lc_models.resolve_hf("segformer_b2_clothes", segformer_model) if parts else None
        folder = lc_models.resolve_vitmatte(vitmatte_model) if refine == "vitmatte" else None

        raws, masks, cutouts = [], [], []
        for i in range(b):
            frame = image[i:i + 1]
            rgb8 = (frame[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            if not parts:
                raw = torch.zeros(h, w)
            elif engine == "mediapipe":
                raw = (_mediapipe_prob(rgb8, model_path, parts) > confidence).float()
            else:
                raw = (_segformer_prob(Image.fromarray(rgb8), folder_seg, parts, dev) > confidence).float()

            img = frame[0, ..., :3].permute(2, 0, 1).unsqueeze(0).to(dev).float()
            m = raw.to(dev).unsqueeze(0).unsqueeze(0)
            out, _tri = refine_mask(img, m, refine, folder, 0.0, 0.0, 0.75, edge_erode, edge_dilate, 0.001, 0.5,
                                   black_point, white_point, 0.0, max_megapixels, dev)
            final = out[0, 0].cpu()
            if remove_features and face and bool(raw.any()):
                # after the edge refine, so the matting can not grow back over an eye or a lip
                feat = _features(frame)
                final = final * (1.0 - feat)
                raw = raw * (feat < 0.5).float()
            raws.append(raw)
            masks.append(final)
            cutouts.append(torch.cat([frame[0, ..., :3].cpu().float(), final.unsqueeze(-1)], dim=-1))

        return (torch.stack(cutouts, 0), torch.stack(masks, 0), torch.stack(raws, 0))


NODE_CLASS_MAPPINGS = {
    "LCPersonMask": LCPersonMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCPersonMask": "LC Person Mask 🧍",
}
