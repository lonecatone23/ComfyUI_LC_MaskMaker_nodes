"""
Model discovery and on-demand download for the LC MaskMaker nodes.
Models are never bundled or redistributed: each dropdown lists what is already
on disk plus a curated "⬇ Download" list, labelled with the license.
"""

import os
import shutil

import folder_paths

DOWNLOAD_PREFIX = "⬇ Download: "

# label -> (repo id, subfolder name under models/vitmatte)
VITMATTE_DOWNLOADS = {
    "vitmatte-small-composition-1k (Apache-2.0)": ("hustvl/vitmatte-small-composition-1k", "vitmatte-small-composition-1k"),
    "vitmatte-base-composition-1k (Apache-2.0)": ("hustvl/vitmatte-base-composition-1k", "vitmatte-base-composition-1k"),
}
VITMATTE_ROOT_LABEL = "models/vitmatte (local)"


def models_dir(*parts):
    return os.path.join(folder_paths.models_dir, *parts)


def _has_hf_model(path):
    return os.path.isfile(os.path.join(path, "config.json")) and any(
        os.path.isfile(os.path.join(path, f)) for f in ("model.safetensors", "pytorch_model.bin")
    )


def scan_hf_models(kind):
    """Local Hugging Face-format model folders under models/<kind>. Returns {label: path}."""
    base = models_dir(kind)
    found = {}
    if not os.path.isdir(base):
        return found
    if _has_hf_model(base):
        found[f"models/{kind} (local)"] = base
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and _has_hf_model(p):
            found[name] = p
    return found


def vitmatte_choices():
    local = scan_hf_models("vitmatte")
    downloads = []
    for label, (_repo, sub) in VITMATTE_DOWNLOADS.items():
        if sub not in local:
            downloads.append(DOWNLOAD_PREFIX + label)
    return list(local.keys()) + downloads


def resolve_vitmatte(choice):
    """Return a local folder for the choice, downloading first if needed."""
    local = scan_hf_models("vitmatte")
    if choice in local:
        return local[choice]
    label = choice[len(DOWNLOAD_PREFIX):] if choice.startswith(DOWNLOAD_PREFIX) else choice
    if label not in VITMATTE_DOWNLOADS:
        raise ValueError(f"[LC MaskMaker] Unknown VITMatte model: {choice}")
    repo, sub = VITMATTE_DOWNLOADS[label]
    target = models_dir("vitmatte", sub)
    if _has_hf_model(target):
        return target
    return download_hf(repo, target)


def download_hf(repo_id, target_dir):
    """Download the weights + config files of a HF repo into target_dir."""
    from huggingface_hub import list_repo_files, snapshot_download

    print(f"[LC MaskMaker] downloading {repo_id} -> {target_dir}")
    files = list_repo_files(repo_id)
    patterns = ["config.json", "preprocessor_config.json", "*.txt", "*.json"]
    patterns.append("model.safetensors" if "model.safetensors" in files else "pytorch_model.bin")
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=target_dir, allow_patterns=patterns)
    print(f"[LC MaskMaker] finished {repo_id}")
    return target_dir


# ---------------------------------------------------------------------------
# Background removal (ComfyUI's built-in BiRefNet loader reads these files)
# ---------------------------------------------------------------------------
# label -> (repo id, saved file name, processing size)
BGREM_DOWNLOADS = {
    "BiRefNet general (MIT)": ("ZhengPeng7/BiRefNet", "BiRefNet-general.safetensors", 1024),
    "BiRefNet matting (MIT)": ("ZhengPeng7/BiRefNet-matting", "BiRefNet-matting.safetensors", 1024),
    "BiRefNet portrait (MIT)": ("ZhengPeng7/BiRefNet-portrait", "BiRefNet-portrait.safetensors", 1024),
    "BiRefNet HR 2048 (MIT)": ("ZhengPeng7/BiRefNet_HR", "BiRefNet-HR.safetensors", 2048),
}
_WEIGHT_EXT = (".safetensors", ".pth", ".pt", ".ckpt")


def _bgrem_local():
    """{label: (path, size)} for every usable background-removal file on disk."""
    found = {}
    by_file = {v[1]: (k, v[2]) for k, v in BGREM_DOWNLOADS.items()}

    core = models_dir("background_removal")
    if os.path.isdir(core):
        for f in sorted(os.listdir(core)):
            if f.endswith(_WEIGHT_EXT):
                label, size = by_file.get(f, (f"{f} (local)", 1024))
                found[label] = (os.path.join(core, f), size)

    rmbg = os.path.join(models_dir("RMBG"), "RMBG-2.0", "model.safetensors")
    if os.path.isfile(rmbg):
        found["RMBG-2.0 (BRIA, non-commercial)"] = (rmbg, 1024)
    return found


def bgremoval_choices():
    local = _bgrem_local()
    downloads = [DOWNLOAD_PREFIX + k for k in BGREM_DOWNLOADS if k not in local]
    # BiRefNet (MIT) first so it is the default; the non-commercial one goes last.
    mit = [k for k in local if "non-commercial" not in k]
    restricted = [k for k in local if "non-commercial" in k]
    return mit + downloads + restricted


def resolve_bgremoval(choice):
    """Return (path, size) for the choice, downloading first if needed."""
    local = _bgrem_local()
    if choice in local:
        return local[choice]
    label = choice[len(DOWNLOAD_PREFIX):] if choice.startswith(DOWNLOAD_PREFIX) else choice
    if label not in BGREM_DOWNLOADS:
        raise ValueError(f"[LC MaskMaker] Unknown background removal model: {choice}")
    repo, fname, size = BGREM_DOWNLOADS[label]
    target_dir = models_dir("background_removal")
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, fname)
    if not os.path.isfile(dest):
        from huggingface_hub import hf_hub_download

        print(f"[LC MaskMaker] downloading {repo} -> {dest}")
        tmp_dir = os.path.join(target_dir, ".lc_tmp")
        tmp = hf_hub_download(repo_id=repo, filename="model.safetensors", local_dir=tmp_dir)
        os.replace(tmp, dest)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[LC MaskMaker] finished {repo}")
    return dest, size


# ---------------------------------------------------------------------------
# GroundingDINO + SAM (Hugging Face format) and SAM 3 (local file)
# ---------------------------------------------------------------------------
# kind -> {label: (repo id, folder name under models/<kind>)}
HF_DOWNLOADS = {
    "grounding-dino": {
        "grounding-dino-tiny (Apache-2.0)": ("IDEA-Research/grounding-dino-tiny", "grounding-dino-tiny"),
        "grounding-dino-base (Apache-2.0)": ("IDEA-Research/grounding-dino-base", "grounding-dino-base"),
    },
    "sam": {
        "sam-vit-base (Apache-2.0)": ("facebook/sam-vit-base", "sam-vit-base"),
        "sam-vit-large (Apache-2.0)": ("facebook/sam-vit-large", "sam-vit-large"),
        "sam-vit-huge (Apache-2.0)": ("facebook/sam-vit-huge", "sam-vit-huge"),
    },
    # an empty folder name means the model lives directly in models/<kind>
    "segformer_b2_clothes": {
        "segformer_b2_clothes (license: other, verify before commercial use)": ("mattmdjaga/segformer_b2_clothes", ""),
    },
}


def _hf_installed(kind, sub, local):
    return (f"models/{kind} (local)" in local) if not sub else (sub in local)


def hf_choices(kind):
    local = scan_hf_models(kind)
    downloads = [DOWNLOAD_PREFIX + label for label, (_repo, sub) in HF_DOWNLOADS[kind].items()
                 if not _hf_installed(kind, sub, local)]
    return list(local.keys()) + downloads


def resolve_hf(kind, choice):
    """Return a local folder for the choice, downloading first if needed."""
    local = scan_hf_models(kind)
    if choice in local:
        return local[choice]
    label = choice[len(DOWNLOAD_PREFIX):] if choice.startswith(DOWNLOAD_PREFIX) else choice
    table = HF_DOWNLOADS[kind]
    if label not in table:
        raise ValueError(f"[LC MaskMaker] Unknown {kind} model: {choice}")
    repo, sub = table[label]
    target = models_dir(kind, sub) if sub else models_dir(kind)
    if _has_hf_model(target):
        return target
    return download_hf(repo, target)


SAM3_DOWNLOAD_FILE = "sam3.1_multiplex_fp16.safetensors"
SAM3_DOWNLOAD_LABEL = f"{SAM3_DOWNLOAD_FILE} (SAM 3.1, SAM License, hosted by Comfy-Org)"


def _sam3_kind(fname):
    return "SAM 3.1" if "3.1" in fname or "sam31" in fname.lower() else "SAM 3"


def _sam3_local():
    """{label: path} for SAM 3 files in models/sam3 and in ComfyUI's models/detection folder."""
    found = {}
    for folder in (models_dir("sam3"), models_dir("detection")):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith("sam3") and f.endswith((".safetensors", ".pt", ".pth", ".ckpt")):
                found.setdefault(f"{f} ({_sam3_kind(f)}, SAM License)", os.path.join(folder, f))
    return found


def sam3_choices():
    """SAM 3 checkpoints. The original facebook/sam3 is gated by Meta so it is never downloaded for you;
    the ungated SAM 3.1 copy that ComfyUI's own template uses (Comfy-Org/sam3.1) is offered as a download."""
    local = _sam3_local()
    names = sorted(local.keys(), key=lambda k: (not k.split(" (")[0].endswith(".safetensors"), k))
    have = {os.path.basename(p) for p in local.values()}
    downloads = [] if SAM3_DOWNLOAD_FILE in have else [DOWNLOAD_PREFIX + SAM3_DOWNLOAD_LABEL]
    return names + downloads


def resolve_sam3(choice):
    local = _sam3_local()
    if choice in local:
        return local[choice]
    label = choice[len(DOWNLOAD_PREFIX):] if choice.startswith(DOWNLOAD_PREFIX) else choice
    if label == SAM3_DOWNLOAD_LABEL:
        dest = os.path.join(models_dir("sam3"), SAM3_DOWNLOAD_FILE)
        if not os.path.isfile(dest):
            from huggingface_hub import hf_hub_download

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"[LC MaskMaker] downloading Comfy-Org/sam3.1 -> {dest} (1.7 GB)")
            tmp_dir = os.path.join(models_dir("sam3"), ".lc_tmp")
            tmp = hf_hub_download(repo_id="Comfy-Org/sam3.1", filename="checkpoints/" + SAM3_DOWNLOAD_FILE, local_dir=tmp_dir)
            os.replace(tmp, dest)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print("[LC MaskMaker] finished sam3.1")
        return dest
    raise RuntimeError(
        "[LC Segment Anything] No SAM 3 file found. Choose the SAM 3.1 download, or put your own "
        "sam3.safetensors in ComfyUI/models/sam3."
    )


# ---------------------------------------------------------------------------
# MediaPipe selfie multiclass (person parts), a single tflite file
# ---------------------------------------------------------------------------
MEDIAPIPE_FILE = "selfie_multiclass_256x256.tflite"
MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/"
    + MEDIAPIPE_FILE
)


def resolve_mediapipe():
    path = os.path.join(models_dir("mediapipe"), MEDIAPIPE_FILE)
    if not os.path.isfile(path):
        import urllib.request

        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[LC MaskMaker] downloading {MEDIAPIPE_FILE} -> {path}")
        tmp = path + ".part"
        urllib.request.urlretrieve(MEDIAPIPE_URL, tmp)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Depth Anything V2 (original .pth / .safetensors) and BAE normal maps
# ---------------------------------------------------------------------------
# encoder key -> (nice name, license)
DEPTH_VARIANTS = {
    "vits": ("Small", "Apache-2.0"),
    "vitb": ("Base", "CC-BY-NC-4.0"),
    "vitl": ("Large", "CC-BY-NC-4.0"),
    "vitg": ("Giant", "CC-BY-NC-4.0"),
}
# file name -> (repo id, encoder)
DEPTH_DOWNLOADS = {
    "depth_anything_v2_vits.pth": ("depth-anything/Depth-Anything-V2-Small", "vits"),
    "depth_anything_v2_vitb.pth": ("depth-anything/Depth-Anything-V2-Base", "vitb"),
    "depth_anything_v2_vitl.pth": ("depth-anything/Depth-Anything-V2-Large", "vitl"),
}
_WEIGHT_EXTS = (".safetensors", ".pth", ".pt")


def _base_path():
    return getattr(folder_paths, "base_path", os.path.dirname(folder_paths.models_dir))


def _label_for_depth(fname, encoder, extra=""):
    name, lic = DEPTH_VARIANTS[encoder]
    return f"{fname} (Depth Anything V2 {name}, {lic}{extra})"


def _encoder_from_name(fname):
    low = fname.lower()
    for key in ("vitg", "vitl", "vitb", "vits"):
        if key in low:
            return key
    return None


def _depth_local():
    """{label: (path, encoder)} of Depth Anything V2 weights we can find."""
    found = {}
    folders = [(models_dir("depthanything"), ""),
               (os.path.join(_base_path(), "custom_nodes", "comfyui_controlnet_aux", "ckpts", "depth-anything"), ", from comfyui_controlnet_aux")]
    for folder, extra in folders:
        if not os.path.isdir(folder):
            continue
        for dp, _dn, fns in os.walk(folder):
            if ".cache" in dp:
                continue
            for f in sorted(fns):
                enc = _encoder_from_name(f)
                if f.endswith(_WEIGHT_EXTS) and enc and "metric" not in f.lower():
                    found.setdefault(_label_for_depth(f, enc, extra), (os.path.join(dp, f), enc))
    return found


def _depth_label(encoder):
    """The one label a model gets on every machine, wherever its file lives (or before it is downloaded)."""
    name, lic = DEPTH_VARIANTS[encoder]
    return f"Depth Anything V2 {name} ({lic})"


def depth_choices():
    # Same list everywhere, so a shared workflow never shows a missing model. Giant has no public
    # download, so it only shows up when you have the file. Apache-2.0 Small comes first.
    local_encs = {enc for _p, enc in _depth_local().values()}
    return [_depth_label(e) for e in ("vits", "vitb", "vitl", "vitg") if e != "vitg" or "vitg" in local_encs]


def _encoder_from_choice(choice):
    for enc in DEPTH_VARIANTS:
        if choice == _depth_label(enc):
            return enc
    # labels saved by older versions: file name + location, or a "Download:" entry
    label = choice[len(DOWNLOAD_PREFIX):] if choice.startswith(DOWNLOAD_PREFIX) else choice
    return _encoder_from_name(label.split(" (")[0])


def resolve_depth(choice):
    """Return (path, encoder), downloading first if needed. Accepts current and older labels."""
    local = _depth_local()
    if choice in local:  # an old label naming one exact local file
        return local[choice]
    enc = _encoder_from_choice(choice)
    if enc is None:
        raise ValueError(f"[LC MaskMaker] Unknown Depth Anything model: {choice}")
    # any local file of that size; models/depthanything wins over the controlnet_aux copy
    for _label, (path, e) in local.items():
        if e == enc:
            return path, enc
    for f, (repo, e) in DEPTH_DOWNLOADS.items():
        if e != enc:
            continue
        dest = os.path.join(models_dir("depthanything"), f)
        if not os.path.isfile(dest):
            from huggingface_hub import hf_hub_download

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"[LC MaskMaker] downloading {repo}/{f} -> {dest}")
            tmp_dir = os.path.join(models_dir("depthanything"), ".lc_tmp")
            tmp = hf_hub_download(repo_id=repo, filename=f, local_dir=tmp_dir)
            os.replace(tmp, dest)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[LC MaskMaker] finished {repo}")
        return dest, enc
    raise ValueError(f"[LC MaskMaker] {choice} is not on this machine and has no public download.")


BAE_FILE = "scannet.pt"
BAE_LABEL = "scannet.pt (BAE normals, ScanNet-trained; license unclear, verify before commercial use)"


def _bae_local():
    found = {}
    for path, extra in ((os.path.join(models_dir("normalbae"), BAE_FILE), ""),
                        (os.path.join(_base_path(), "custom_nodes", "comfyui_controlnet_aux", "ckpts", "lllyasviel", "Annotators", BAE_FILE),
                         " [from comfyui_controlnet_aux]")):
        if os.path.isfile(path):
            found[BAE_LABEL + extra] = path
    return found


def bae_choices():
    # One label on every machine, wherever the file lives (or before it is downloaded).
    return [BAE_LABEL]


def resolve_bae(choice):
    """Accepts the current label and the per-machine labels older versions saved."""
    local = _bae_local()
    if local:
        return next(iter(local.values()))  # models/normalbae wins over the controlnet_aux copy
    dest = os.path.join(models_dir("normalbae"), BAE_FILE)
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[LC MaskMaker] downloading lllyasviel/Annotators/{BAE_FILE} -> {dest}")
    tmp_dir = os.path.join(models_dir("normalbae"), ".lc_tmp")
    tmp = hf_hub_download(repo_id="lllyasviel/Annotators", filename=BAE_FILE, local_dir=tmp_dir)
    os.replace(tmp, dest)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[LC MaskMaker] finished scannet.pt")
    return dest
