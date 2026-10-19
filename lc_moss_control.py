"""
LC MOSS Control 🕹️
-------------------
Runs darask0's MOSS-Image Color-Depth Control adapter (Apache-2.0, huggingface.co/darask0/MOSS-Image-color-depth-control)
inside ComfyUI. The picture follows a color region map (where each thing goes) and / or a depth map (its shape and
front / back order, near = white).

How the adapter works (from its own training code): MOSS's input layer video_embedder Linear(128, 3072) becomes
Linear(384, 3072) over [noisy latent | color-map latent | depth-map latent], all plain FLUX.2 VAE latents; a zero
latent means "no control". A rank-64 LoRA rides along on the blocks and the text embedder.

Here: the 384-wide input layer goes in as an object patch, the LoRA through ComfyUI's own LoRA patching, and a
diffusion-model wrapper adds the two control latents to the noisy latent on every step (positive and negative
pass, like training), zeroed after color_end / depth_end of the steps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import folder_paths

_CACHE = {}


def _load_adapter(name):
    path = folder_paths.get_full_path_or_raise("model_patches", name)
    if path not in _CACHE:
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if "video_embedder.weight" not in sd or (meta or {}).get("control_channels") != "color,depth":
            raise ValueError(f"{name} is not a MOSS color / depth control adapter (no video_embedder or control_channels).")
        _CACHE.clear()
        _CACHE[path] = (sd, meta or {})
    return _CACHE[path]


class _ControlEmbedder(torch.nn.Module):
    """The adapter's 384-wide input layer. Weights stay on the CPU and are cast to the input's device and dtype."""

    def __init__(self, weight, bias):
        super().__init__()
        self.w_cpu, self.b_cpu = weight, bias
        self._dev = None

    def forward(self, x):
        key = (x.device, x.dtype)
        if self._dev != key:
            self.w, self.b = self.w_cpu.to(*key), self.b_cpu.to(*key)
            self._dev = key
        return F.linear(x, self.w, self.b)


def _encode(vae, image):
    if image is None:
        return None
    img = image[:1, :, :, :3]
    h, w = img.shape[1] // 16 * 16, img.shape[2] // 16 * 16
    if (h, w) != tuple(img.shape[1:3]):
        img = comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "lanczos", "center").movedim(1, -1)
    return vae.encode(img).float().cpu()


def _step_index(transformer_options):
    sig = transformer_options.get("sigmas")
    ss = transformer_options.get("sample_sigmas")
    if sig is None or ss is None or len(ss) < 2:
        return 0.0
    i = int(torch.argmin((ss.to(sig.device).float() - sig.float().max()).abs()).item())
    return i / max(len(ss) - 1, 1)


class LCMOSSControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MOSS Image SFT (or Base). Works with the 8-step turbo LoRA."}),
                "vae": ("VAE", {"tooltip": "The Flux.2 VAE: the maps are encoded with it."}),
                "control_file": (folder_paths.get_filename_list("model_patches"),
                                 {"tooltip": "darask0's moss-control.safetensors, in models/model_patches."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Strength of the adapter's LoRA. The input layer always applies in full."}),
                "color_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Stop the color map after this part of the steps: layout first, then the model "
                                                   "finishes the surfaces. Maps made from a real photo: 1.0. Mannequins and painted maps: "
                                                   "0.4 - 0.5."}),
                "depth_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Stop the depth map after this part of the steps. Real photo: 1.0. Mannequins: 0.15 "
                                                   "(0.25 with the 8-step turbo LoRA), or leave depth out. Later and the mannequin "
                                                   "shows through."}),
            },
            "optional": {
                "color_map": ("IMAGE", {"tooltip": "Flat colors, one per object or region (the colors are only IDs). Black = "
                                                   "unspecified. Same size as the picture. Pair each color with a region prompt."}),
                "depth_map": ("IMAGE", {"tooltip": "Grayscale depth, near = white (LC Depth Anything V2 gives that). "
                                                   "Same size as the picture."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "LC MaskMaker/MOSS Image"
    DESCRIPTION = ("ControlNet for MOSS Image: the picture follows a color region map and / or a depth map "
                   "(darask0's Color-Depth Control adapter). Either map can be left out.")

    def apply(self, model, vae, control_file, strength, color_end, depth_end, color_map=None, depth_map=None):
        if type(model.model).__name__ != "MOSSImage":
            raise ValueError("LC MOSS Control needs a MOSS Image model.")
        if color_map is None and depth_map is None:
            return (model,)
        sd, meta = _load_adapter(control_file)
        m = model.clone()
        emb = _ControlEmbedder(sd["video_embedder.weight"].float(), sd["video_embedder.bias"].float())
        m.add_object_patch("diffusion_model.video_embedder", emb)
        if strength:
            key_map = comfy.lora.model_lora_keys_unet(m.model, {})
            lora = {("diffusion_model." + k): v for k, v in sd.items() if ".lora_" in k}
            patches = comfy.lora.load_lora(lora, key_map, log_missing=False)
            if not patches:
                raise ValueError("The adapter's LoRA matched no layer of this model.")
            m.add_patches(patches, strength)
        color, depth = _encode(vae, color_map), _encode(vae, depth_map)
        if depth is not None and color is None:
            color = torch.zeros_like(depth)
        if color is not None and depth is None:
            depth = torch.zeros_like(color)
        state = {"color": color, "depth": depth, "has_color": color_map is not None, "has_depth": depth_map is not None,
                 "cache": {}}

        def fit(lat, x):
            key = (id(lat), x.shape[-2:], x.device, x.dtype)
            c = state["cache"].get(key)
            if c is None:
                c = lat
                if tuple(c.shape[-2:]) != tuple(x.shape[-2:]):
                    c = F.interpolate(c, size=x.shape[-2:], mode="bilinear", align_corners=False)
                c = c.to(x.device, x.dtype)
                state["cache"] = {key: c} if len(state["cache"]) > 8 else {**state["cache"], key: c}
            return c.expand(x.shape[0], -1, -1, -1)

        def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
            frac = _step_index(transformer_options)
            c = fit(state["color"], x)
            d = fit(state["depth"], x)
            if not state["has_color"] or frac >= color_end:
                c = torch.zeros_like(c)
            if not state["has_depth"] or frac >= depth_end:
                d = torch.zeros_like(d)
            return executor(torch.cat((x, c, d), dim=1), timestep, context, transformer_options, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "lc_moss_control", wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"LCMOSSControl": LCMOSSControl}
NODE_DISPLAY_NAME_MAPPINGS = {"LCMOSSControl": "LC MOSS Control 🕹️"}
