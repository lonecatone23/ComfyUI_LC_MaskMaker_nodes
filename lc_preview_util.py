"""Preview helper: source, before mask, after mask (and trimap) for the on-node wipe."""

import torch
import torch.nn.functional as F

PREVIEW_MAX = 768


def _small(t_bchw):
    _, _, h, w = t_bchw.shape
    s = min(1.0, PREVIEW_MAX / float(max(h, w)))
    if s < 1.0:
        t_bchw = F.interpolate(t_bchw, size=(max(1, int(h * s)), max(1, int(w * s))), mode="area")
    return t_bchw


def _mask3(mask, like):
    """First (H,W) mask of a batch as a (1,3,h,w) preview image the same size as `like`."""
    t = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[:1].unsqueeze(1).float().cpu()
    t = _small(t)
    if t.shape[-2:] != like.shape[-2:]:
        t = F.interpolate(t, size=like.shape[-2:], mode="nearest")
    return t.repeat(1, 3, 1, 1)


def wipe_preview(node, image, before, after, trimap=None, prefix="lc_refine"):
    """image (B,H,W,C); before/after/trimap are (H,W)-ish masks. Returns the ui dict.

    Saves 3 or 4 small images in the order: source, before, after, trimap.
    """
    ui = {}
    try:
        h, w = image.shape[1], image.shape[2]
        src = _small(image[:1][..., :3].permute(0, 3, 1, 2).float().cpu())
        parts = [src, _mask3(before, src), _mask3(after, src)]
        if trimap is not None:
            parts.append(_mask3(trimap, src))
        batch = torch.cat(parts, 0).permute(0, 2, 3, 1)
        saved = node.save_images(batch, filename_prefix=prefix)
        ui["lc_refine"] = saved["ui"]["images"]
        ui["src_size"] = [{"width": int(w), "height": int(h)}]
    except Exception as e:
        print(f"[LC MaskMaker] preview skipped: {e}")
    return ui
