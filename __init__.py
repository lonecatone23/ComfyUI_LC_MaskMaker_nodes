"""
ComfyUI_LC_MaskMaker_nodes — LC MaskMaker by lonecatone23

Mask, matting, segmentation and image-adjustment tools. Separate repo,
separate Registry name. Shares no code with ComfyUI_LC123_nodes (which stays
import-free); heavier, model-backed nodes live here and load their models on
demand.

https://github.com/lonecatone23
https://ko-fi.com/lonecatone
"""

import os as _os

_PACK_DIR = _os.path.dirname(_os.path.abspath(__file__))
_nested = _os.path.join(_PACK_DIR, "ComfyUI_LC_MaskMaker_nodes", "__init__.py")
if _os.path.isfile(_nested):
    print(
        "[LC MaskMaker] WARNING: nested pack folder detected. "
        f"{_nested} will be ignored. Unzip so __init__.py sits in {_PACK_DIR}"
    )

NODE_CLASS_MAPPINGS = {}
_FAILED = []  # modules that did not load (shown in full, and counted on the banner line)
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(module_name: str) -> None:
    """Import a submodule and merge its mappings. Log and skip on failure."""
    import importlib
    import traceback

    try:
        mod = importlib.import_module(f".{module_name}", __name__)
        maps = getattr(mod, "NODE_CLASS_MAPPINGS", None) or {}
        disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None) or {}
        NODE_CLASS_MAPPINGS.update(maps)
        NODE_DISPLAY_NAME_MAPPINGS.update(disp)
    except Exception as e:
        _FAILED.append(module_name)
        print(f"[LC MaskMaker] ! failed to load {module_name}: {e}")
        traceback.print_exc()


_load("lc_image_outpaint")
_load("lc_rotate_pad")
_load("lc_spline_mask")
_load("lc_mask_refine")
_load("lc_rembg")
_load("lc_segment")
_load("lc_inpaint")
_load("lc_anypaint")
_load("lc_person")
_load("lc_image_blend")
_load("lc_auto_adjust")
_load("lc_depth_anything")
_load("lc_normal_bae")
_load("lc_face_crop")
_load("lc_vosr_upscale")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:  # the Lonecat banner (first LC pack to load) or this pack's one line under it
    from .lc_banner import pack_version as _pv, show as _banner

    _banner("LC MaskMaker", _pv(_PACK_DIR), len(NODE_CLASS_MAPPINGS), len(_FAILED))
except Exception as _banner_e:
    print(f"[LC MaskMaker] {len(NODE_CLASS_MAPPINGS)} nodes ({_banner_e})")
