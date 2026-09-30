"""
SAM 3 through ComfyUI's own SAM 3 support, shared by LC Segment Anything and LC Smart Inpaint.

Two fixes over calling ComfyUI directly:
  - The original Meta checkpoints (sam3.pt / sam3.safetensors) carry a 512-wide text projection that
    ComfyUI's text model does not use and cannot load. It is dropped before loading.
  - ComfyUI keeps one detection per prompt unless the prompt says "word:N". Every word here asks for
    up to MAX_PER_WORD, so "hands" finds both hands.
"""

import torch

MAX_PER_WORD = 16

_cache = {}  # path -> (model patcher, clip)


def load(path):
    if path not in _cache:
        import comfy.sd
        import comfy.utils
        import folder_paths

        sd = comfy.utils.load_torch_file(path)
        for k in [k for k in sd if k.endswith("language_backbone.encoder.text_projection")]:
            sd.pop(k)
        out = comfy.sd.load_state_dict_guess_config(
            sd, output_vae=False, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip = out[0], out[1]
        if model is None or clip is None:
            raise RuntimeError(f"[LC MaskMaker] {path} does not look like a SAM 3 checkpoint.")
        _cache.clear()
        _cache[path] = (model, clip)
    return _cache[path]


def _detect_cls():
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except Exception as e:
        raise RuntimeError("[LC MaskMaker] This ComfyUI has no built-in SAM 3 support. Update ComfyUI.") from e
    return SAM3_Detect


def objects(image_1hwc, words, path, threshold):
    """Every object SAM 3 finds for any of the words, as a list of (H, W) float masks (0/1)."""
    detect = _detect_cls()
    model, clip = load(path)
    found = []
    for word in words:
        text = word.replace("(", "").replace(")", "")
        if ":" not in text:
            text = f"{text}:{MAX_PER_WORD}"
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        res = detect.execute(model, image_1hwc[..., :3], conditioning=cond, threshold=float(threshold),
                             refine_iterations=2, individual_masks=True)
        masks = res.args[0].float().cpu()  # (N, H, W)
        found.extend(m for m in masks if bool(m.any()))
    return found


def union(image_1hwc, words, path, threshold):
    """All objects for all words as one (H, W) mask."""
    h, w = image_1hwc.shape[1:3]
    out = torch.zeros(h, w)
    for m in objects(image_1hwc, words, path, threshold):
        out = torch.maximum(out, m)
    return out
