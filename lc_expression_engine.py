"""LivePortrait for LC Face Expression: load the five networks once, read a face crop, re-pose it, render it.

Pipeline (the paper's, from vendor/liveportrait, MIT):
  appearance extractor F : 256 px crop -> 3D feature volume
  motion extractor M     : 256 px crop -> 21 implicit 3D keypoints, head pose, expression offsets, scale, shift
  stitching S            : keeps the re-posed keypoints attached to the shoulders / crop edge
  warping W + SPADE G    : warps the feature volume from source to driving keypoints, renders a 512 px face
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from . import lc_cache

REPO = "Kijai/LivePortrait_safetensors"
FILES = {
    "F": ("appearance_feature_extractor.safetensors", "3 MB"),
    "M": ("motion_extractor.safetensors", "112 MB"),
    "W": ("warping_module.safetensors", "182 MB"),
    "G": ("spade_generator.safetensors", "222 MB"),
    "S": ("stitching_retargeting_module.safetensors", "1 MB"),
}
# models.yaml, as plain values (no yaml needed at run time)
PARAMS = {
    "F": dict(image_channel=3, block_expansion=64, num_down_blocks=2, max_features=512, reshape_channel=32,
              reshape_depth=16, num_resblocks=6),
    "M": dict(num_kp=21, backbone="convnextv2_tiny"),
    "W": dict(num_kp=21, block_expansion=64, max_features=512, num_down_blocks=2, reshape_channel=32,
              estimate_occlusion_map=True,
              dense_motion_params=dict(block_expansion=32, max_features=1024, num_blocks=5, reshape_depth=16,
                                       compress=4)),
    "G": dict(upscale=2, block_expansion=64, max_features=512, num_down_blocks=2),
    "S": dict(input_size=126, hidden_sizes=[128, 128, 64], output_size=65),
}
NUM_KP = 21

_cache = lc_cache.register("liveportrait", {})
LOCK = threading.Lock()  # the node and the drag editor's route can render at the same time


def model_dir():
    import folder_paths

    return os.path.join(folder_paths.models_dir, "liveportrait")


def _ensure_files():
    from . import lc_first_download

    d = model_dir()
    for name, size in FILES.values():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            lc_first_download.download(lc_first_download.hf_url(REPO, name), p, "LivePortrait " + name, size)
    return d


def _load_sd(path, prefix=""):
    import comfy.utils

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def load():
    """The five networks on the torch device, cached until LC's cache is cleared."""
    if "nets" in _cache:
        return _cache["nets"]
    import comfy.model_management as mm

    from .vendor.liveportrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
    from .vendor.liveportrait.modules.motion_extractor import MotionExtractor
    from .vendor.liveportrait.modules.spade_generator import SPADEDecoder
    from .vendor.liveportrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
    from .vendor.liveportrait.modules.warping_network import WarpingNetwork

    d = _ensure_files()
    dev = mm.get_torch_device()
    nets = {
        "F": AppearanceFeatureExtractor(**PARAMS["F"]),
        "M": MotionExtractor(**PARAMS["M"]),
        "W": WarpingNetwork(**PARAMS["W"]),
        "G": SPADEDecoder(**PARAMS["G"]),
        "S": StitchingRetargetingNetwork(**PARAMS["S"]),
    }
    for k, net in nets.items():
        path = os.path.join(d, FILES[k][0])
        sd = _load_sd(path, "retarget_shoulder_module." if k == "S" else "")
        net.load_state_dict(sd)
        net.to(dev).eval()
    _cache["nets"] = (nets, dev)
    return _cache["nets"]


def _degrees(pred):
    from .vendor.liveportrait.utils.camera import headpose_pred_to_degree

    return headpose_pred_to_degree(pred)


def _rot(pitch, yaw, roll):
    from .vendor.liveportrait.utils.camera import get_rotation_matrix

    return get_rotation_matrix(pitch, yaw, roll)


def _transform(kp, rot, exp, scale, t):
    """Canonical keypoints -> posed keypoints: (kp @ R + exp) * scale + t (x and y only)."""
    x = kp @ rot + exp
    x = x * scale[..., None]
    x[:, :, 0:2] = x[:, :, 0:2] + t[:, None, 0:2]
    return x


class Source:
    """One face read: everything a re-pose needs, so moving a lever never re-runs F or M."""

    def __init__(self, feat, kp, exp, scale, t, pitch, yaw, roll, x_s):
        self.feat, self.kp, self.exp, self.scale, self.t = feat, kp, exp, scale, t
        self.pitch, self.yaw, self.roll, self.x_s = pitch, yaw, roll, x_s


@torch.no_grad()
def read(crop_256):
    """crop_256: (256, 256, 3) float32 0..1 numpy. Returns a Source."""
    nets, dev = load()
    x = torch.from_numpy(crop_256).permute(2, 0, 1)[None].float().to(dev)
    feat = nets["F"](x)
    info = nets["M"](x)
    b = 1
    kp = info["kp"].view(b, NUM_KP, 3)
    exp = info["exp"].view(b, NUM_KP, 3)
    scale = info["scale"].view(b, 1)
    t = info["t"].view(b, 3)
    pitch, yaw, roll = _degrees(info["pitch"]), _degrees(info["yaw"]), _degrees(info["roll"])
    x_s = _transform(kp, _rot(pitch, yaw, roll), exp, scale, t)
    return Source(feat, kp, exp, scale, t, pitch, yaw, roll, x_s)


@torch.no_grad()
def render(src: Source, d_exp: torch.Tensor, d_rot=(0.0, 0.0, 0.0), keep=1.0):
    """Re-pose and render. d_exp: (21, 3) keypoint offsets; d_rot: pitch, yaw, roll in degrees.
    keep: how much of the photo's own expression stays (1 = all, 0 = a neutral face before the offsets).
    Returns (512, 512, 3) float32 0..1 numpy."""
    nets, dev = load()
    exp = src.exp * float(keep) + d_exp.to(dev).view(1, NUM_KP, 3)
    rot = _rot(src.pitch + d_rot[0], src.yaw + d_rot[1], src.roll + d_rot[2])
    x_d = _transform(src.kp, rot, exp, src.scale, src.t)
    # stitching: a small correction so the re-posed face stays attached to the rest of the crop
    delta = nets["S"](torch.cat([src.x_s.view(1, -1), x_d.view(1, -1)], dim=1))
    x_d = x_d + delta[:, :NUM_KP * 3].view(1, NUM_KP, 3)
    x_d[:, :, 0:2] = x_d[:, :, 0:2] + delta[:, NUM_KP * 3:NUM_KP * 3 + 2].view(1, 1, 2)
    warped = nets["W"](src.feat, kp_driving=x_d, kp_source=src.x_s)["out"]
    out = nets["G"](feature=warped)
    return out[0].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()


def resize(img, h, w):
    """Area / bicubic resize of an (H, W, 3) float numpy image."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
    down = h < t.shape[2]
    t = F.interpolate(t, size=(h, w), mode="area" if down else "bicubic", align_corners=None if down else False)
    return t[0].permute(1, 2, 0).clamp(0, 1).numpy()
