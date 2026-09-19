"""
LC Mask Refine
--------------
Tighten a mask against its image. A trimap is built from the mask (sure foreground
shrunk by edge_erode, sure background starting edge_dilate beyond it), then a guided
filter or VITMatte decides only the band in between. Optional detail_region gets a
wider trimap for hair. Levels and feather finish it, and the node shows a before/after
wipe, including the trimap.
"""

import torch
import torch.nn.functional as F
from nodes import PreviewImage

from . import lc_models
from .lc_matting import METHODS, get_device, refine_mask
from .lc_preview_util import wipe_preview
from .lc_refine_core import spread_colors


class LCMaskRefine(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image the mask belongs to. Used as the guide."}),
                "mask": ("MASK", {"tooltip": "Mask to refine. Any size; it is resized to the image."}),
                "method": (METHODS, {
                    "default": "vitmatte",
                    "tooltip": (
                        "vitmatte: matting model, best for hair and fine detail.\n"
                        "guided_filter: fast, no model, snaps the unknown band to image edges.\n"
                        "none: skip matting, only grow, fix gaps, levels and feather."
                    ),
                }),
                "vitmatte_model": (lc_models.vitmatte_choices(), {
                    "tooltip": "Only used by vitmatte. Entries marked 'Download' are fetched the first time you run.",
                }),
                "grow": ("FLOAT", {
                    "default": 0.0, "min": -512.0, "max": 512.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Pixels to grow (positive) or shrink (negative) the mask before the trimap is built.",
                }),
                "fix_gap": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Close small holes and cracks up to about this many pixels wide. 0 = off.",
                }),
                "fix_threshold": ("FLOAT", {
                    "default": 0.75, "min": 0.01, "max": 0.999, "step": 0.001,
                    "tooltip": "How white a mask pixel must be to count as solid when fixing gaps.",
                }),
                "edge_erode": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": (
                        "TRIMAP. How far the sure-foreground area is pulled in from the mask edge, in pixels.\n"
                        "Everything inside this is locked white. Smaller = tighter, less to decide."
                    ),
                }),
                "edge_dilate": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": (
                        "TRIMAP. How far the unknown band reaches out past the mask edge, in pixels.\n"
                        "Everything beyond this is locked black. Smaller = tighter, less to decide."
                    ),
                }),
                "softness": ("FLOAT", {
                    "default": 0.001, "min": 0.00001, "max": 0.1, "step": 0.0001, "round": 0.00001,
                    "tooltip": "guided_filter only. Lower follows image edges tighter, higher is smoother.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.999, "step": 0.001,
                    "tooltip": "Where the input mask is split into inside and outside when the trimap is built. Raise toward 1.0 to use only the solid core of a soft mask.",
                }),
                "black_point": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.999, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or below this become 0. Clears faint haze.",
                }),
                "white_point": ("FLOAT", {
                    "default": 0.99, "min": 0.001, "max": 1.0, "step": 0.001, "round": 0.0001,
                    "tooltip": "Mask values at or above this become 1. Solidifies the subject.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Final edge softness in pixels.",
                }),
                "edge_color_spread": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Cutout only. Pushes solid subject colors outward this many pixels to cut background halos. 0 = off.",
                }),
                "max_megapixels": ("FLOAT", {
                    "default": 3.0, "min": 0.25, "max": 16.0, "step": 0.25,
                    "tooltip": "vitmatte only. Larger images are matted at this size, then scaled back.",
                }),
                "detail_erode": ("FLOAT", {
                    "default": 72.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Same as edge_erode, but for the detail_region area (hair). Wider gives fine strands room.",
                }),
                "detail_dilate": ("FLOAT", {
                    "default": 64.0, "min": 0.0, "max": 1024.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Same as edge_dilate, but for the detail_region area.",
                }),
                "detail_blur": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 512.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Softens the seam where the detail_region result blends into the rest.",
                }),
                "preview_view": (["cutout", "mask", "trimap"], {
                    "default": "cutout",
                    "tooltip": (
                        "What the wipe on the node shows.\n"
                        "trimap: left is the trimap (black = sure background, gray = unknown, white = sure foreground), right is the result."
                    ),
                }),
            },
            "optional": {
                "detail_region": ("MASK", {
                    "tooltip": "Optional. White areas (like hair) use detail_erode / detail_dilate instead of edge_erode / edge_dilate.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cutout", "mask", "trimap")
    FUNCTION = "refine"
    CATEGORY = "LC MaskMaker/mask"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Tighten a mask with a trimap. edge_erode and edge_dilate set how wide the unknown band is; "
        "a guided filter or VITMatte decides only that band. Before/after wipe and trimap view on the node."
    )

    def refine(self, image, mask, method, vitmatte_model, grow, fix_gap, fix_threshold, edge_erode,
               edge_dilate, softness, threshold, black_point, white_point, feather, edge_color_spread,
               max_megapixels, detail_erode, detail_dilate, detail_blur, preview_view, detail_region=None):
        b, h, w, _c = image.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        n = max(b, mask.shape[0])
        dev = get_device()
        folder = lc_models.resolve_vitmatte(vitmatte_model) if method == "vitmatte" else None

        def fit(t, i):
            t = t[min(i, t.shape[0] - 1)].unsqueeze(0).unsqueeze(0).to(dev).float()
            if t.shape[-2:] != (h, w):
                t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            return t.clamp(0, 1)

        if detail_region is not None and detail_region.dim() == 2:
            detail_region = detail_region.unsqueeze(0)

        masks, tris, cutouts, first_before = [], [], [], None
        for i in range(n):
            img = image[min(i, b - 1)][..., :3].permute(2, 0, 1).unsqueeze(0).to(dev).float()
            m = fit(mask, i)
            if first_before is None:
                first_before = m[0, 0].cpu()
            region = fit(detail_region, i) if detail_region is not None else None
            out, tri = refine_mask(
                img, m, method, folder, grow, fix_gap, fix_threshold, edge_erode, edge_dilate,
                softness, threshold, black_point, white_point, feather, max_megapixels, dev,
                detail_region=region, detail_erode=detail_erode, detail_dilate=detail_dilate,
                detail_blur=detail_blur,
            )
            rgb = spread_colors(img, out, edge_color_spread)
            cutouts.append(torch.cat([rgb, out], dim=1)[0].permute(1, 2, 0).cpu())
            masks.append(out[0, 0].cpu())
            tris.append(tri[0, 0].cpu())

        mask_out = torch.stack(masks, 0)
        tri_out = torch.stack(tris, 0)
        cutout = torch.stack(cutouts, 0)
        ui = wipe_preview(self, image, first_before, mask_out[0], trimap=tri_out[0])
        return {"ui": ui, "result": (cutout, mask_out, tri_out)}


NODE_CLASS_MAPPINGS = {
    "LCMaskRefine": LCMaskRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCMaskRefine": "LC Mask Refine ✨",
}
