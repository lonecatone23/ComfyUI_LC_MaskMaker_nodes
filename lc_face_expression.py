"""
LC Expression Editor 😉: change a finished face's expression with levers (smile, jaw, eyes, gaze, brows, head), after
the detailers.

How: SAM 3 finds the face. LivePortrait (MIT, vendor/liveportrait) reads it from a square crop, moves its implicit
keypoints by the lever offsets (lc_expression_levers) and renders the face again at 512 px. Only the face goes back:
SAM 3 masks the face on the photo and on the posed render, the two are joined (a jaw that opens reaches below the old
chin), grown, feathered and pasted over the original. Hair, hood and background stay the original pixels.

protect_mask is that pasted area, ready for any later node that should leave the new face alone.
"""

from __future__ import annotations

import contextlib
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from . import lc_cache, lc_models, lc_sam3
from . import lc_expression_engine as E
from . import lc_expression_levers as L
from .lc_character_detailer import _choose
from .lc_inpaint import _clean

CROP = 1.75  # crop side = face mask width x this (LivePortrait was trained on loose crops with the shoulders in view)
LEVER_NAMES = ["smile", "smirk", "mouth_open", "mouth_wide", "eye_left", "eye_right", "gaze_x", "gaze_y", "brows", "furrow", "brow_tilt",
               "head_turn", "head_nod", "head_tilt"]
_reads = lc_cache.register("face_expression", {})  # picture + face -> (face mask, LivePortrait read, crop box)
_last = {}  # node id -> the last picture it saw (a new picture resets the levers)
STEPS = 9  # progress units per picture: find the face 3, read it 1, render 1, find it again 3, paste 1

TIPS = {
    "smile": "+ smiles (teeth show past about 0.5). - turns the mouth corners down.",
    "smirk": "One mouth corner up and the other down: + lifts the corner on the right of the picture, - the left. With smile, a one-sided smile.",
    "mouth_open": "Opens the jaw (+) or closes a mouth that is open in the picture (-). 0.3 to 0.5 reads as talking.",
    "mouth_wide": "+ stretches the mouth wide (teeth, grimace). - purses the lips (pucker, kiss).",
    "eye_left": "The eye on the left of the picture. - closes it (a wink), + opens it.",
    "eye_right": "The eye on the right of the picture. - closes it (a wink), + opens it.",
    "gaze_x": "Where the eyes look sideways: + toward the right of the picture, - toward the left.",
    "gaze_y": "Where the eyes look up (+) or down (-). With gaze_x the eyes reach any clock position.",
    "brows": "+ raises both brows (surprise, interest). - lowers them (anger, focus).",
    "furrow": "Pulls the inner brows together and up (worry, concentration), - relaxes them. With brows at -0.5 it reads as a scowl.",
    "brow_tilt": "One brow up and the other down (skeptical). Subtle: brows move as a pair in this model.",
    "head_turn": "Turns the head, in degrees: + toward the right of the picture. Past about 12 the face edge starts to stretch.",
    "head_nod": "Tips the head down (+) or up (-), in degrees.",
    "head_tilt": "Tilts the head toward a shoulder, in degrees: + toward the right of the picture.",
    "keep_expression": "How much of the expression already in the picture stays. 1 keeps it and adds the levers. "
                       "Lower it to calm a face that is already smiling before posing it.",
    "preset": "Custom: the levers as set. A named preset moves this face to that expression from wherever it starts (a face that already smiles is not pushed further), writes the levers to match and applies again to every new picture. Drag a handle or change a lever and it becomes Custom, keeping those values.",
    "preset_strength": "How far toward the preset: 1 all the way, 0.5 halfway from the picture's own expression.",
    "pick": "Which face to change when there are several.",
    "number": "Which one: 1 = the first, counted the way pick says.",
    "grow": "Grows the pasted area past the face edge, in pixels. Raise it if a sliver of the old jaw shows.",
    "feather": "Softens the edge of the pasted area, in pixels.",
}
DESC = (
    "Changes a finished face's expression with levers: smile, jaw, mouth width, each eye, gaze, brows and head "
    "angle. Put it after the detailers. LivePortrait re-poses the face, then only the face is pasted back, so hair and "
    "background stay untouched. protect_mask is the pasted area for any later step that should leave the face alone. "
    "Warping of the image on the preview node is normal. The final face is cut & pasted back in. All levers at 0 passes the picture through. Made for small adjustments on a clear face: a hand or anything else in front of the face warps with it."
)


def _lever(lo, hi, step=0.05, key=""):
    return ("FLOAT", {"default": 0.0, "min": lo, "max": hi, "step": step, "tooltip": TIPS[key]})


def _key(frame, extra):
    t = frame[0, ::8, ::8, :3].float().cpu().numpy()
    return hashlib.sha1(t.tobytes() + repr(extra).encode()).hexdigest()


def _bbox(m):
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_box(box, W, H):
    x0, y0, x1, y1 = box
    side = int(round((x1 - x0) * CROP))
    side = max(side, int((y1 - y0) * 1.2), 64)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def _cut(img, x, y, side):
    """Square crop that may reach outside the picture (edges repeated)."""
    H, W, _ = img.shape
    pad = max(0, -x, -y, x + side - W, y + side - H)
    if pad:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return img[y + pad:y + pad + side, x + pad:x + pad + side]


def _grow_feather(m, grow, feather):
    m = m[None, None].float()
    g = int(round(grow))
    if g > 0:
        m = F.max_pool2d(m, 2 * g + 1, stride=1, padding=g)
    f = int(round(feather))
    if f > 0:
        k = 2 * f + 1
        x = torch.arange(k, dtype=torch.float32) - f
        w = torch.exp(-(x ** 2) / (2 * (f / 2.0) ** 2))
        w = (w / w.sum()).to(m)
        m = F.conv2d(F.pad(m, (f, f, 0, 0), mode="replicate"), w.view(1, 1, 1, k))
        m = F.conv2d(F.pad(m, (0, 0, f, f), mode="replicate"), w.view(1, 1, k, 1))
    return m[0, 0].clamp(0, 1)


class LCFaceExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam3_model": (lc_models.sam3_choices(), {"tooltip": "SAM 3 model that finds the face."}),
                "pick": (["largest first", "left to right", "right to left", "top to bottom"],
                         {"default": "largest first", "tooltip": TIPS["pick"]}),
                "number": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": TIPS["number"]}),
                "preset": (["Custom"] + list(L.PRESETS), {"default": "Custom", "tooltip": TIPS["preset"]}),
                "preset_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": TIPS["preset_strength"]}),
                "smile": _lever(-1.0, 1.0, key="smile"),
                "smirk": _lever(-1.0, 1.0, key="smirk"),
                "mouth_open": _lever(-1.0, 1.0, key="mouth_open"),
                "mouth_wide": _lever(-1.0, 1.0, key="mouth_wide"),
                "eye_left": _lever(-1.0, 1.0, key="eye_left"),
                "eye_right": _lever(-1.0, 1.0, key="eye_right"),
                "gaze_x": _lever(-1.0, 1.0, key="gaze_x"),
                "gaze_y": _lever(-1.0, 1.0, key="gaze_y"),
                "brows": _lever(-1.0, 1.0, key="brows"),
                "furrow": _lever(-1.0, 1.0, key="furrow"),
                "brow_tilt": _lever(-1.0, 1.0, key="brow_tilt"),
                "head_turn": _lever(-20.0, 20.0, 1.0, key="head_turn"),
                "head_nod": _lever(-20.0, 20.0, 1.0, key="head_nod"),
                "head_tilt": _lever(-20.0, 20.0, 1.0, key="head_tilt"),
                "keep_expression": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": TIPS["keep_expression"]}),
                "grow": ("INT", {"default": 8, "min": 0, "max": 128, "tooltip": TIPS["grow"]}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 128, "tooltip": TIPS["feather"]}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, sam3_model=None):
        return lc_models.validate_models(sam3_model=("sam3", sam3_model))

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "protect_mask")
    OUTPUT_TOOLTIPS = ("The picture with the new expression.",
                       "The face area that was changed: wire it to anything later that should leave the face alone.")
    FUNCTION = "run"
    # an output node: ComfyUI re-sends a cached output node's result, so a fresh browser tab still gets the face
    # for the drag editor when nothing upstream changed
    OUTPUT_NODE = True
    CATEGORY = "LC MaskMaker/detailer"
    DESCRIPTION = DESC

    def run(self, image, sam3_model, pick, number, keep_expression, grow, feather, preset="Custom",
            preset_strength=1.0, unique_id=None, **levers):
        values = {k: float(levers.get(k, 0.0)) for k in LEVER_NAMES}
        # a new picture: the levers were set for the last face, so this run changes nothing and the editor resets
        # them to 0. The first run after a workflow loads keeps its saved levers (nothing to compare with yet).
        picture = _key(image[0:1], ())
        reset = unique_id is not None and _last.get(unique_id) not in (None, picture)
        if unique_id is not None:
            _last[unique_id] = picture
        if reset:
            values = {k: 0.0 for k in values}
            keep_expression = 1.0
        named = preset in L.PRESETS
        path = lc_models.resolve_sam3(sam3_model)
        B, H, W, _ = image.shape
        out_imgs, out_masks = [], []
        editor_key = None  # the first face found: the one the drag editor shows
        editor_levers = None  # the levers a named preset gave the editor's face (the editor writes them)
        # the progress bar moves with the real work, weighted by how long each part takes (the two SAM 3 passes
        # are most of it): it used to jump to the end at the start and then sit there
        import comfy.utils

        pbar = comfy.utils.ProgressBar(B * STEPS)

        def at(i, k):
            pbar.update_absolute(min(B * STEPS, i * STEPS + k), B * STEPS)

        @contextlib.contextmanager
        def within(i, a, b):
            """SAM 3 reports its own progress ("1 of 1") on this node, which flashed the bar full. While it runs, its
            progress is mapped into this node's stage a..b instead (a progress bar takes the hook when it is made)."""
            orig = comfy.utils.PROGRESS_BAR_HOOK

            def hook(v, t, preview=None, node_id=None):
                frac = min(1.0, v / t) if t else 1.0
                if orig is not None:
                    orig(i * STEPS + a + (b - a) * frac, B * STEPS, preview, node_id=node_id)

            comfy.utils.PROGRESS_BAR_HOOK = hook
            try:
                yield
            finally:
                comfy.utils.PROGRESS_BAR_HOOK = orig

        for i in range(B):
            at(i, 0)
            frame = image[i:i + 1]
            key = _key(frame, (pick, number, path))
            if key not in _reads:
                while len(_reads) >= 8:  # newest pictures only: a long batch or many runs don't pile up
                    _reads.pop(next(iter(_reads)))
                with within(i, 0, 3):
                    faces = [_clean(m) for m in lc_sam3.objects(frame, ["face"], path, 0.5)]
                at(i, 3)
                m0, why = _choose(faces, pick, number, None)
                if m0 is None:
                    print(f"[LC Face Expression] frame {i + 1}: no face ({why}), passed through.")
                    _reads[key] = None
                else:
                    box = _bbox(m0)
                    x, y, side = _crop_box(box, W, H)
                    arr = frame[0, :, :, :3].float().cpu().numpy()
                    crop = _cut(arr, x, y, side)
                    with E.LOCK, within(i, 3, 4):  # a first-use download reports here too
                        src = E.read(E.resize(crop, 256, 256))
                    at(i, 4)
                    # for the drag editor: the crop as shown on the node, and the face box inside it (0..1)
                    fbox = [(box[0] - x) / side, (box[1] - y) / side, (box[2] - x) / side, (box[3] - y) / side]
                    # the photo's own expression as lever values: where the editor's handles start
                    _reads[key] = (m0, src, (x, y, side), (E.resize(crop, 512, 512) * 255).astype(np.uint8), fbox,
                                   L.read(src.exp),
                                   # LivePortrait's 21 keypoints on the crop (0..1): the handles sit on them
                                   [[round((float(k[0]) + 1) / 2, 4), round((float(k[1]) + 1) / 2, 4)]
                                    for k in src.x_s[0].float().cpu().tolist()])
            entry = _reads[key]
            if entry is None:
                out_imgs.append(frame)
                out_masks.append(torch.zeros(1, H, W))
                continue
            m0, src, (x, y, side) = entry[:3]
            vals = dict(values)
            if named:  # this face's own way to the preset (each frame of a batch starts somewhere else)
                vals.update(L.preset_levers(preset, entry[5], preset_strength))
            if editor_key is None:
                editor_key = key
                editor_levers = vals if named else None
            posing = any(abs(v) > 1e-6 for v in vals.values()) or keep_expression < 1.0
            d_exp, d_rot = L.offsets(vals)
            if not posing:
                out_imgs.append(frame)
                out_masks.append(_grow_feather(m0, grow, feather)[None].cpu())
                continue
            with E.LOCK:
                face = E.resize(E.render(src, d_exp, d_rot, keep=keep_expression), side, side)
            at(i, 5)
            posed = frame[0, :, :, :3].float().cpu().clone()
            # paste the render's square back where it was cut (the part inside the picture)
            fx0, fy0 = max(0, x), max(0, y)
            fx1, fy1 = min(W, x + side), min(H, y + side)
            posed[fy0:fy1, fx0:fx1] = torch.from_numpy(face[fy0 - y:fy1 - y, fx0 - x:fx1 - x])
            # the face on the posed picture: the one that overlaps the original face most
            with within(i, 5, 8):
                new = [_clean(m) for m in lc_sam3.objects(posed[None], ["face"], path, 0.5)]
            at(i, 8)
            m1 = max(new, key=lambda m: float((m * m0).sum()), default=None)
            area = m0 if m1 is None or float((m1 * m0).sum()) <= 0 else torch.maximum(m0, m1)
            # fade out toward the edges of the rendered square (and the picture's edge where the square ran past
            # it): a big face or one cut off by the frame reaches them, and a hard clip there left a straight seam
            # across the forehead, or pasted the stretched padding of the render
            # a side where the square ran past the picture fades wider: a head that moves away from that edge
            # drags the render's stretched padding into view there
            rw = float(max(2 * feather, side // 16, 4))
            wide = float(max(rw, side // 6))
            ys = torch.arange(H, dtype=torch.float32).view(H, 1)
            xs = torch.arange(W, dtype=torch.float32).view(1, W)
            sides = [((ys - fy0) / (wide if y < 0 else rw)), (((fy1 - 1) - ys) / (wide if y + side > H else rw)),
                     ((xs - fx0) / (wide if x < 0 else rw)), (((fx1 - 1) - xs) / (wide if x + side > W else rw))]
            inside = torch.ones(H, W)
            for s in sides:
                inside = torch.minimum(inside, s.clamp(0, 1).expand(H, W))
            mask = _grow_feather(area * (inside > 0).float(), grow, feather) * inside
            orig = frame[0, :, :, :3].float().cpu()
            res = orig * (1 - mask[..., None]) + posed * mask[..., None]
            if frame.shape[-1] == 4:
                res = torch.cat([res, frame[0, :, :, 3:].float().cpu()], -1)
            out_imgs.append(res[None])
            out_masks.append(mask[None])
        at(B, 0)
        result = (torch.cat(out_imgs, 0).to(image), torch.cat(out_masks, 0))
        ui = {"key": editor_key or "", "reset": reset and not named}
        if editor_levers is not None:
            ui["levers"] = editor_levers
        return {"ui": {"lc_face": [ui]}, "result": result}


# ---- drag editor routes: the node's face read is reused, so a drag re-renders only the face (no queue) ----

def _jpeg(arr_u8):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    _GONE = "Run the node once to edit the face here."

    @PromptServer.instance.routes.get("/lc_maskmaker/face_expression/source")
    async def _fe_source(request):
        entry = _reads.get(request.query.get("key", ""))
        if not entry:
            return web.json_response({"error": _GONE}, status=404)
        return web.json_response({"img": _jpeg(entry[3]), "box": entry[4], "read": entry[5], "kp": entry[6],
                                  "presets": L.PRESETS})

    @PromptServer.instance.routes.post("/lc_maskmaker/face_expression/render")
    async def _fe_render(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        entry = _reads.get(body.get("key", ""))
        if not entry:
            return web.json_response({"error": _GONE}, status=404)
        levers = body.get("levers") or {}
        values = {k: float(levers.get(k, 0.0) or 0.0) for k in LEVER_NAMES}
        keep = float(body.get("keep", 1.0))

        def work():
            d_exp, d_rot = L.offsets(values)
            with E.LOCK:
                face = E.render(entry[1], d_exp, d_rot, keep=keep)
            return _jpeg((face * 255).astype(np.uint8))

        try:
            img = await asyncio.get_event_loop().run_in_executor(None, work)
        except Exception as e:
            print(f"[LC Face Expression] preview failed: {e}")
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"img": img})
except Exception as _route_e:  # no server (tests)
    print(f"[LC Face Expression] drag editor off: {_route_e}")


NODE_CLASS_MAPPINGS = {"LCFaceExpression": LCFaceExpression}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFaceExpression": "LC Expression Editor 😉"}
