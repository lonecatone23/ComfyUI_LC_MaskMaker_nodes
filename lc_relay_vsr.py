"""LC Relay VSR Upscale: ComfyUI's native RelayVSR (UNET + Flash VAE, 1 step euler simple, cfg 1) in one node.

Same graph as RelayVSRConditioning -> KSampler -> VSRDecode. On top of it:
- the 2x / 4x model and Flash VAE are built in: found wherever they already are, else downloaded on first use
  (Comfy-Org/RelayVSR, Apache-2.0) into models/diffusion_models and models/vae, with the LC toast,
- the model and VAE stay cached between runs,
- keyframes always go through in small chunks with the model's own cache. The native shortcut runs a short clip in
  one pass, which overflows 8 GB cards at 4x (5 s clip on an RTX 5060 8 GB: 22.6 min native, 104 s here). Output
  differs from the one-pass result by a mean 0.2/255, and it is no slower on a 5090.
"""
from __future__ import annotations

import os

import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes

from .lc_first_download import download, hf_url

try:
    import comfy.ldm.wan.model_vsr as _vsr
except Exception:  # ComfyUI older than 0.44 has no RelayVSR
    _vsr = None

REPO = "Comfy-Org/RelayVSR"
FILES = {  # factor: (model, vae, model size)
    "2x": ("relayvsr_x2_lightx2v_int8_convrot.safetensors", "relayvsr_flash_vae_x2_fp32.safetensors", "5.2 GB"),
    "4x": ("relayvsr_x4_int8_convrot.safetensors", "relayvsr_flash_vae_x4_fp32.safetensors", "5.2 GB"),
}
CHUNK_TOKENS = 8192  # about one 4x keyframe of a 960x544 clip per chunk
_CACHE = {"model": (None, None), "vae": (None, None)}


def _find(kind, filename):
    """Full path of filename anywhere ComfyUI looks for this kind (any subfolder), else None."""
    want = filename.lower()
    for name in folder_paths.get_filename_list(kind):
        if os.path.basename(name.replace("\\", "/")).lower() == want:
            return folder_paths.get_full_path(kind, name)
    return None


def _ensure(kind, hf_folder, filename, label, size):
    path = _find(kind, filename)
    if path:
        return path
    dest = os.path.join(folder_paths.get_folder_paths(kind)[0], filename)
    if not os.path.isfile(dest):
        download(hf_url(REPO, f"{hf_folder}/{filename}"), dest, label, size)
    return dest


def _model(path):
    key, model = _CACHE["model"]
    if key != path:
        if model is not None:  # switching 2x <-> 4x: let the old 5 GB go before the new one loads
            _CACHE["model"] = (None, None)
            del model
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        model = comfy.sd.load_diffusion_model(path)
        _CACHE["model"] = (path, model)
    return model


def _vae(path):
    key, vae = _CACHE["vae"]
    if key != path:
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        _CACHE["vae"] = (path, vae)
    return vae


class LCRelayVSRUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "A picture, a batch of pictures, or the frames of one video (see source)."}),
            "source": (["video", "image"], {"default": "video",
                       "tooltip": "video: the frames are matched to each other, so they stay steady. "
                                  "image: each picture is upscaled on its own, so nothing carries over between them."}),
            "upscale": (list(FILES), {"default": "2x",
                        "tooltip": "2x for video, 4x for pictures. Each is its own 5.2 GB model, downloaded on first use."}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed",
                             "tooltip": "One step from noise: the seed only changes the finest grain."}),
            "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "How much of the added detail is kept. 1.0 = full model output, lower moves toward a plain bicubic upscale."}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "LC_RELAY")
    RETURN_NAMES = ("image", "relay")
    OUTPUT_TOOLTIPS = ("The upscaled frames. Held in RAM at full size: a 5 s clip at 4x is about 12 GB.",
                       "For LC Save Video's relay input: it decodes and saves the video a few frames at a time, so it never sits "
                       "in RAM. With only this output connected, the full-size image is never built.")
    FUNCTION = "run"
    CATEGORY = "LC MaskMaker/upscale"
    DESCRIPTION = ("RelayVSR upscale in one node (ComfyUI 0.44 or newer). Fast and sharp, adds fine skin and hair detail. "
                   "2x for video, 4x for pictures. Models download on first use (Comfy-Org/RelayVSR, Apache-2.0). "
                   "Keeps VRAM flat on 8 GB cards.")

    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        # A cached run that skipped the full-size decode has no image: once something is plugged into `image` it must
        # run again. ComfyUI does not hand the prompt to IS_CHANGED, so read it from the running check.
        prompt = _queued_prompt()
        if prompt is None or unique_id is None:
            return float("nan")
        return "image linked" if linked(prompt, unique_id, 0) else "relay only"

    def run(self, image, source, upscale, seed, detail, prompt=None, unique_id=None):
        if _vsr is None:
            raise RuntimeError("[LC Relay VSR] Needs ComfyUI 0.44 or newer (RelayVSR is built into ComfyUI from 0.44).")
        model_file, vae_file, size = FILES[upscale]
        m = _model(_ensure("diffusion_models", "diffusion_models", model_file, f"Relay VSR {upscale} model", size))
        v = _vae(_ensure("vae", "vae", vae_file, f"Relay VSR {upscale} VAE", "128 MB"))
        parts = [image[i:i + 1] for i in range(image.shape[0])] if source == "image" and image.shape[0] > 1 else [image]
        jobs = [(self._sample(m, p, seed), p) for p in parts]
        up = v.first_stage_model.upscale_factor
        relay = {"vae": v, "jobs": jobs, "detail": detail, "count": sum(int(p.shape[0]) for p in parts),
                 "size": (int(image.shape[2]) * up, int(image.shape[1]) * up)}
        relay["frames"] = lambda batch=8: iter_relay_frames(relay, batch)
        if prompt is not None and unique_id is not None and not linked(prompt, unique_id, 0):
            return (None, relay)  # only a relay saver is connected: skip the full-size decode
        from comfy_extras.nodes_wan import VSRDecode

        out = None
        for k, (samples, p) in enumerate(jobs):  # pictures go straight into one buffer, no second copy at the end
            dec = VSRDecode.execute(v, samples, p, detail).result[0]
            if len(jobs) == 1:
                return (dec, relay)
            if out is None:
                out = torch.empty((len(jobs),) + tuple(dec.shape[1:]), dtype=dec.dtype)
            out[k:k + 1] = dec
            del dec
        return (out, relay)

    @staticmethod
    def _sample(m, image, seed):
        from comfy_extras.nodes_wan import RelayVSRConditioning

        old = (_vsr.RELAY_WAN_CACHE_KEYFRAMES, _vsr.RELAY_WAN_CHUNK_TOKENS)
        try:
            _vsr.RELAY_WAN_CACHE_KEYFRAMES = 0
            _vsr.RELAY_WAN_CHUNK_TOKENS = CHUNK_TOKENS
            positive, latent = RelayVSRConditioning.execute(m, image).result
            return nodes.common_ksampler(m, seed, 1, 1.0, "euler", "simple", positive, positive, latent, denoise=1.0)[0]
        finally:
            _vsr.RELAY_WAN_CACHE_KEYFRAMES, _vsr.RELAY_WAN_CHUNK_TOKENS = old


def _decode_frames(vae, samples, lq_images, detail):
    """The RelayVSR Flash decoder's own frame loop (comfy/ldm/wan/model_vsr.py, decode()), yielding each finished frame
    (uint8 H x W x 3 on the GPU) instead of filling a full-size buffer. The decoder's history runs across the frames
    exactly as in one-pass decoding, so the frames are the same."""
    import torch.nn.functional as F

    fsm = vae.first_stage_model
    x = samples["samples"]
    mm = comfy.model_management
    mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(x.shape, vae.vae_dtype),
                       force_full_load=getattr(vae, "disable_offload", False))
    x = x.to(device=vae.device, dtype=vae.vae_dtype)
    _, _, _, height, width = x.shape
    frames, lq_h, lq_w = lq_images.shape[:3]
    out_h, out_w = lq_h * fsm.upscale_factor, lq_w * fsm.upscale_factor
    depth = len(fsm.backbone.blocks)
    keyframes = _vsr.relay_keyframes(frames)
    dev = x.device
    pbar = comfy.utils.ProgressBar(frames)
    with torch.inference_mode():
        freqs, freqs_past = fsm.grid_rope(0, height, width, dev), fsm.grid_rope(-1, height, width, dev)
        start = fsm.prepare_condition(x[:, :, 0], freqs)
    for window, (first, last) in enumerate(zip(keyframes, keyframes[1:] or keyframes)):
        mm.throw_exception_if_processing_interrupted()
        with torch.inference_mode():
            end = fsm.prepare_condition(x[:, :, min(window + 1, len(keyframes) - 1)], freqs)
            condition = list(zip(start, end))
            history = [None] * depth
            lr = lq_images[first:last + 1].to(device=dev)[..., :3].movedim(-1, 1) * 2.0 - 1.0
            lr = F.pad(lr, (0, width * fsm.latent_stride - lq_w, 0, height * fsm.latent_stride - lq_h), mode="replicate")
        for i, frame in enumerate(range(first, last + 1)):
            with torch.inference_mode():
                decoded, history = fsm.decode_frame(lr[i:i + 1], x.dtype, freqs, freqs_past, condition, history, detail)
                px = None
                if window == 0 or frame > first:  # the shared keyframe was already output by the previous window
                    px = decoded[0, :, :out_h, :out_w].add_(1.0).mul_(127.5).clamp_(0.0, 255.0).round_().to(torch.uint8).permute(1, 2, 0)
            if px is not None:
                pbar.update(1)
                yield px
        start = end


def iter_relay_frames(relay, batch=8):
    """The upscaled video of a relay output, a few frames at a time: uint8 numpy arrays [k, H, W, 3]. Used by
    LC Save Video (LC AV), so the whole video never sits in RAM."""
    pending = []
    for samples, lq in relay["jobs"]:
        for px in _decode_frames(relay["vae"], samples, lq, relay["detail"]):
            pending.append(px)
            if len(pending) >= batch:
                yield torch.stack(pending).cpu().numpy()
                pending = []
    if pending:
        yield torch.stack(pending).cpu().numpy()


def _queued_prompt():
    """The prompt being checked: ComfyUI's IS_CHANGED cache holds it as dynprompt one or more frames up."""
    import inspect

    frame = inspect.currentframe()
    try:
        while frame is not None:
            dp = getattr(frame.f_locals.get("self"), "dynprompt", None)
            if dp is not None and hasattr(dp, "get_original_prompt"):
                return dp.get_original_prompt()
            frame = frame.f_back
    finally:
        del frame
    return None


def linked(prompt, unique_id, slot):
    """True when any node in the queued prompt reads output `slot` of node `unique_id`."""
    uid = str(unique_id)
    for node in (prompt or {}).values():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, (list, tuple)) and len(v) == 2 and str(v[0]) == uid and v[1] == slot:
                return True
    return False


NODE_CLASS_MAPPINGS = {"LCRelayVSRUpscale": LCRelayVSRUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"LCRelayVSRUpscale": "LC Relay VSR Upscale 🧩"}
