"""
Holds for the nodes with a block switch (a frame, a mask or boxes set up on the node).

A picture that comes out of a run is first seen on the node when that run reaches it, so a frame or mask still set
from the last picture would run straight through on the new one. block also stops on a new picture: set it up on the
picture you see, then Run. The stop is reported as a hold (ui lc_hold), so with LC123 that Run carries on with the
same seeds (web/lc_continue.js) and only what comes after runs. On a node whose face shows a Load Image's picture
before any run (pass the prompt), a picture straight from a Load Image doesn't count as new.

The block count goes into the node's IS_CHANGED: the run after a block runs the node again instead of handing back
the cached block.
"""

import hashlib
import time

import torch

_SHOWN: dict[str, str] = {}  # node id -> fingerprint of the picture last shown on it
_BLOCKS: dict[str, int] = {}  # node id -> blocks so far


def fingerprint(image) -> str:
    """A cheap fingerprint of the first picture: its size and a ~64 px sample of its pixels."""
    img = image[:1]
    h, w = int(img.shape[1]), int(img.shape[2])
    small = img[:, :: max(1, h // 64), :: max(1, w // 64), :].to(torch.float16).cpu().contiguous()
    return f"{w}x{h}:" + hashlib.sha1(small.numpy().tobytes()).hexdigest()


def _from_load_image(prompt, uid, name) -> bool:
    """The input is wired straight from a node with an image file widget (Load Image): the face shows it live."""
    try:
        src = (prompt or {})[str(uid)]["inputs"][name]
        return isinstance((prompt[str(src[0])]["inputs"]).get("image"), str)
    except (KeyError, TypeError, IndexError):
        return False


def new_picture(uid, image, prompt=None, name="image") -> bool:
    """True when this picture wasn't on the node yet (also after a restart). Remembers it as shown."""
    key = str(uid)
    shot = fingerprint(image)
    new = _SHOWN.get(key) != shot
    _SHOWN[key] = shot
    return new and not _from_load_image(prompt, uid, name)


def hold(uid, ui: dict) -> dict:
    """Count a block and mark the node's ui as a hold. Returns ui."""
    key = str(uid)
    _BLOCKS[key] = _BLOCKS.get(key, 0) + 1
    ui["lc_hold"] = [time.time()]
    return ui


def changed(uid) -> int:
    """For IS_CHANGED: moves after every block."""
    return _BLOCKS.get(str(uid), 0)
