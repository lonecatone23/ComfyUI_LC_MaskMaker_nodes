"""
What LC nodes leave in the LC_PIPE for the nodes after them (on top of the usual settings):

  image       the working image (an LC_PIPE slot already): pipe nodes put their result back in it
  lc_protect  {"mask": (B,H,W), "tag": tag}: every area a detailer redrew, merged; LC VOSR2 Upscale (pipe) resizes it to the new picture
  lc_found    {"tag": tag, "hits": {key: masks per frame}}: what SAM 3 found, so the same word is not searched twice

Every entry carries the tag of the image it belongs to (size + a pixel checksum). A reader checks the tag against its
own image and ignores the entry if the picture changed in between. The pipe is never edited in place: nodes copy it,
then set their keys, so nothing leaks into ComfyUI's cached pipes, between runs or between branches.
"""

import torch
import torch.nn.functional as F

PROTECT = "lc_protect"
FOUND = "lc_found"


def tag(image):
    """Identity of an IMAGE: its shape and a checksum of a pixel sample (fast at any size)."""
    if image is None:
        return None
    return (tuple(image.shape), round(float(image[:, ::37, ::37, :3].float().sum()), 3))


def pipe_image(pipe, wired=None):
    """The image a pipe node works on: a wired image wins, else the pipe's."""
    if wired is not None:
        return wired
    return pipe.get("image") if isinstance(pipe, dict) else None


def found_hits(pipe, image):
    """lc_found hits for this image, or {} if there are none or the image changed."""
    f = pipe.get(FOUND) if isinstance(pipe, dict) else None
    if isinstance(f, dict) and f.get("tag") == tag(image):
        return dict(f.get("hits", {}))
    return {}


def protect_for(pipe, image):
    """The protect mask (B,H,W) that belongs to this image, or None."""
    p = pipe.get(PROTECT) if isinstance(pipe, dict) else None
    if isinstance(p, dict) and p.get("tag") == tag(image) and p.get("mask") is not None:
        return p["mask"]
    if isinstance(p, dict) and p.get("mask") is not None:
        print("[LC pipe] The image changed since the protect mask was made: it is not used.")
    return None


def merge_masks(a, b):
    """Max of two (B,H,W) masks, as a new tensor. b is resized to a when sizes differ."""
    if a is None:
        return None if b is None else b.float().cpu().clone()
    if b is None:
        return a.float().cpu().clone()
    a, b = a.float().cpu(), b.float().cpu()
    if b.shape[-2:] != a.shape[-2:]:
        b = F.interpolate(b[:, None], size=a.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    if b.shape[0] != a.shape[0]:
        b = b[:1].expand(a.shape[0], -1, -1)
    return torch.maximum(a, b)


def updated(pipe, image, protect=None, hits=None):
    """A copy of the pipe with image, protect mask and found hits set for `image` (all tagged to it)."""
    out = dict(pipe) if isinstance(pipe, dict) else {"_type": "LC_PIPE"}
    out["_type"] = "LC_PIPE"
    t = tag(image)
    out["image"] = image
    if protect is not None:
        out[PROTECT] = {"mask": protect, "tag": t}
    if hits is not None:
        out[FOUND] = {"tag": t, "hits": hits}
    return out
