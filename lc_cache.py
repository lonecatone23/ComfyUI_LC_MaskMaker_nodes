"""
Shared model-cache release for the LC MaskMaker nodes.

The nodes keep their models in small module dicts so a second run does not load them again (the LoRA state dict,
ControlNet / model patches, SAM 3, VITMatte, Segformer, GroundingDINO + SAM, BiRefNet, Depth Anything, BAE, VOSR2).
ComfyUI does not know about these dicts, so "Unload models" / "Free model and node cache" left them in RAM.

Each cache registers its dict here. `comfy.model_management.unload_all_models` is wrapped (the original still runs
first, so ComfyUI offloads its own ModelPatchers as before) to empty the registered dicts afterwards, move any torch
module still on the GPU to the CPU, run gc and `soft_empty_cache`. Nothing else changes: same keys, same reuse
between runs until ComfyUI is asked to free memory.

When ComfyUI's unload runs:
  - the /free route (the "Unload models" and "Free model and node cache" buttons): main.py reads the flags with
    PromptQueue.get_flags, then calls unload_all_models
  - after an out-of-memory error (execution.py)
  - after every prompt when ComfyUI runs with --disable-smart-memory: there the caches are kept (ComfyUI itself only
    moves models off the GPU in that case) unless the /free flags were just read, which get_flags tells us.
"""

import gc
import logging

_CACHES = {}  # name -> dict owned by a node module
_STATE = {"free_requested": False}
_MARK = "_lc_maskmaker_cache_hook"


def register(name, cache):
    """Let `cache` (a dict) be emptied when ComfyUI frees memory. Returns the dict, unchanged."""
    _CACHES[name] = cache
    return cache


def _to_cpu(obj, depth=0):
    """Move torch modules found in a cached value to the CPU. ModelPatchers are left to ComfyUI."""
    if depth > 3:
        return
    try:
        import torch
    except Exception:
        return
    if isinstance(obj, torch.nn.Module):
        try:
            obj.to("cpu")
        except Exception:
            pass
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _to_cpu(item, depth + 1)


def clear(reason=""):
    """Empty every registered cache. Returns the names that held something."""
    freed = []
    for name, cache in list(_CACHES.items()):
        if not cache:
            continue
        try:
            for value in list(cache.values()):
                _to_cpu(value)
            cache.clear()
            freed.append(name)
        except Exception as e:
            logging.warning(f"[LC MaskMaker] could not release the {name} cache: {e}")
    if freed:
        gc.collect()
        try:
            import comfy.model_management as mm

            mm.soft_empty_cache()
        except Exception:
            pass
        logging.info("[LC MaskMaker] released cached models%s: %s", f" ({reason})" if reason else "", ", ".join(freed))
    return freed


def _wrap_unload(original):
    def unload_all_models(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            import comfy.model_management as mm

            requested = _STATE["free_requested"]
            _STATE["free_requested"] = False
            if requested or not getattr(mm, "DISABLE_SMART_MEMORY", False):
                clear("free memory" if requested else "unload models")
        except Exception as e:
            logging.warning(f"[LC MaskMaker] could not release cached models: {e}")
        return result

    setattr(unload_all_models, _MARK, True)
    unload_all_models.__wrapped__ = original
    unload_all_models.__doc__ = getattr(original, "__doc__", None)
    return unload_all_models


def _wrap_get_flags(original):
    def get_flags(self, *args, **kwargs):
        flags = original(self, *args, **kwargs)
        try:
            reset = kwargs.get("reset", args[0] if args else True)
            if reset and isinstance(flags, dict) and (flags.get("unload_models") or flags.get("free_memory")):
                _STATE["free_requested"] = True
        except Exception:
            pass
        return flags

    setattr(get_flags, _MARK, True)
    get_flags.__wrapped__ = original
    return get_flags


def install():
    """Install the hooks once (safe to call again)."""
    try:
        import comfy.model_management as mm

        if not getattr(mm.unload_all_models, _MARK, False):
            mm.unload_all_models = _wrap_unload(mm.unload_all_models)
    except Exception as e:
        logging.warning(f"[LC MaskMaker] model cache release hook not installed: {e}")
    try:
        import execution

        queue_cls = getattr(execution, "PromptQueue", None)
        if queue_cls is not None and not getattr(queue_cls.get_flags, _MARK, False):
            queue_cls.get_flags = _wrap_get_flags(queue_cls.get_flags)
    except Exception:
        pass  # only needed with --disable-smart-memory


install()
