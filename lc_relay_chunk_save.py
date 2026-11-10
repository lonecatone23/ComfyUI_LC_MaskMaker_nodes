"""LC Relay Video Chunk Save: decodes LC Relay VSR Upscale's `relay` output a few frames at a time and pipes them
straight into ffmpeg, so the upscaled video never sits in RAM (a 5 s clip at 4x is about 12 GB as an IMAGE).

The decode is the RelayVSR Flash decoder's own frame loop (comfy/ldm/wan/model_vsr.py, decode()), with each finished
frame sent to the encoder instead of a full-size buffer. The decoder's history runs across the frames exactly as in
one-pass decoding, so the frames are the same.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths

from .lc_relay_vsr import _vsr, linked

BATCH = 8  # frames gathered on the GPU before one copy to the encoder
PRESETS = {"Quality": "slow", "Optimal": "medium", "Fast": "veryfast"}  # same goals as LC Model Optimizer


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_wav(path, audio):
    w = audio["waveform"]
    w = w[0] if w.ndim == 3 else w
    pcm = (np.clip(w.float().cpu().numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(int(pcm.shape[0]))
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes(pcm.T.reshape(-1).tobytes())


def _stream_decode(vae, samples, lq_images, detail, emit):
    """Flash decode, one frame at a time, each finished frame (uint8 HxWx3 on the GPU) handed to emit()."""
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
    freqs, freqs_past = fsm.grid_rope(0, height, width, dev), fsm.grid_rope(-1, height, width, dev)
    pbar = comfy.utils.ProgressBar(frames)
    with torch.inference_mode():
        start = fsm.prepare_condition(x[:, :, 0], freqs)
        for window, (first, last) in enumerate(zip(keyframes, keyframes[1:] or keyframes)):
            mm.throw_exception_if_processing_interrupted()
            end = fsm.prepare_condition(x[:, :, min(window + 1, len(keyframes) - 1)], freqs)
            condition = list(zip(start, end))
            history = [None] * depth
            lr = lq_images[first:last + 1].to(device=dev)[..., :3].movedim(-1, 1) * 2.0 - 1.0
            lr = F.pad(lr, (0, width * fsm.latent_stride - lq_w, 0, height * fsm.latent_stride - lq_h), mode="replicate")
            for i, frame in enumerate(range(first, last + 1)):
                decoded, history = fsm.decode_frame(lr[i:i + 1], x.dtype, freqs, freqs_past, condition, history, detail)
                if window == 0 or frame > first:  # the shared keyframe was already output by the previous window
                    px = decoded[0, :, :out_h, :out_w].add_(1.0).mul_(127.5).clamp_(0.0, 255.0).round_()
                    emit(px.to(torch.uint8).permute(1, 2, 0))
                    pbar.update(1)
            start = end


class LCRelayVideoChunkSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "relay": ("LC_RELAY", {"tooltip": "The relay output of LC Relay VSR Upscale."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Frame rate of the saved video."}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "tooltip": "Quality. Lower = better and bigger. 19 looks like the source."}),
                "goal": (list(PRESETS), {"default": "Optimal", "tooltip": "Quality = smallest file, slowest save. Optimal = balanced. Fast = quickest save, biggest file. The picture looks the same at the same crf."}),
                "filename_prefix": ("STRING", {"default": "LC_Relay", "tooltip": "Name under ComfyUI's output folder. A subfolder/name works."}),
            },
            "optional": {"audio": ("AUDIO", {"tooltip": "Optional soundtrack."})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    OUTPUT_TOOLTIPS = ("The saved video read back at the source size (not the upscaled size), for a preview node. "
                       "Only built when something is connected.",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LC MaskMaker/upscale"
    DESCRIPTION = ("Saves LC Relay VSR Upscale's video a few frames at a time, so the upscaled video never sits in RAM. "
                   "Connect the upscaler's relay output. The video plays on the node when it's done.")

    def save(self, relay, fps, crf, goal, filename_prefix, audio=None, prompt=None, unique_id=None):
        ff = _ffmpeg()
        if ff is None:
            raise RuntimeError("[LC Relay Chunk Save] ffmpeg not found (install ffmpeg or imageio-ffmpeg).")
        vae, jobs, detail = relay["vae"], relay["jobs"], relay["detail"]
        up = vae.first_stage_model.upscale_factor
        lq_h, lq_w = jobs[0][1].shape[1:3]
        w, h = lq_w * up, lq_h * up
        if any(p.shape[1:3] != jobs[0][1].shape[1:3] for _, p in jobs):
            raise ValueError("[LC Relay Chunk Save] All pictures in a batch need the same size to go in one video.")
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, w, h)
        os.makedirs(full_dir, exist_ok=True)
        fname = f"{name}_{counter:05}_.mp4"
        dest = os.path.join(full_dir, fname)
        tmp = tempfile.mkdtemp(prefix="lc_relay_")
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio is not None and audio.get("waveform") is not None and audio["waveform"].shape[-1] > 0:
            wav = os.path.join(tmp, "a.wav")
            _write_wav(wav, audio)
            # cut the sound at the last frame (-shortest drops the last frame when the sound is a hair shorter)
            n = sum(int(p.shape[0]) for _, p in jobs)
            cmd += ["-i", wav, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-t", f"{n / fps:.4f}"]
        # yuv420p needs even sides: pad by one pixel when the upscaled size is odd
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", PRESETS[goal], "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", dest]
        log = open(os.path.join(tmp, "ffmpeg.log"), "w+b")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        pending = []

        def flush():
            if pending:
                proc.stdin.write(torch.stack(pending).cpu().numpy().tobytes())
                pending.clear()

        def emit(px):
            pending.append(px)
            if len(pending) >= BATCH:
                flush()

        ok = False
        try:
            for samples, lq in jobs:
                _stream_decode(vae, samples, lq, detail, emit)
            flush()
            proc.stdin.close()
            ok = proc.wait() == 0
        except (BrokenPipeError, OSError):
            pass
        finally:
            if not ok:
                proc.kill()
                proc.wait()
                log.seek(0)
                err = log.read().decode("utf-8", "replace")[-800:]
                log.close()
                shutil.rmtree(tmp, ignore_errors=True)
                if os.path.exists(dest):
                    os.remove(dest)
                if err.strip():
                    raise RuntimeError(f"[LC Relay Chunk Save] ffmpeg failed: {err}")
            else:
                log.close()
                shutil.rmtree(tmp, ignore_errors=True)
        video = None
        if prompt is None or unique_id is None or linked(prompt, unique_id, 0):
            raw = subprocess.run([ff, "-v", "error", "-i", dest, "-vf", f"scale={lq_w}:{lq_h}:flags=area", "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            video = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(-1, lq_h, lq_w, 3).astype(np.float32) / 255.0)
        preview = {"filename": fname, "subfolder": subfolder, "type": "output"}
        return {"ui": {"images": [preview], "animated": (True,)}, "result": (video,)}


NODE_CLASS_MAPPINGS = {"LCRelayVideoChunkSave": LCRelayVideoChunkSave}
NODE_DISPLAY_NAME_MAPPINGS = {"LCRelayVideoChunkSave": "LC Relay Video Chunk Save 💾"}
