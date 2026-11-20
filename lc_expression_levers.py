"""The levers of LC Face Expression as LivePortrait keypoint offsets.

Each lever is a list of (keypoint, axis, offset at lever = 1). Keypoints are LivePortrait's 21 implicit points, named
here by what moving them does (found by nudging each one on a front-facing portrait):
  1, 2   brows (each pulls both brows a little: brows move as a pair)
  11, 13 the image-left eye: 11 gaze + upper lid, 13 lid
  15, 16 the image-right eye: 15 gaze + upper lid, 16 lid
  14     upper lip / mouth corners      17 lower lip       19 jaw      20 mouth opening
  3, 7   cheeks
Rotation levers are degrees added to the photo's own head pose.
"""

import torch

X, Y, Z = 0, 1, 2

LEVERS = {
    "smile": [(14, Y, -0.025), (20, Y, -0.012), (17, Y, 0.008), (17, Z, 0.004), (3, Y, -0.004), (7, Y, -0.004)],
    # one corner up, the other down: + lifts the corner on the image's right (keypoint 20 sideways, 17 against it)
    "smirk": [(20, X, 0.015), (17, X, -0.015)],
    "mouth_open": [(19, Y, 0.12), (19, Z, 0.012), (17, Y, -0.012)],
    "mouth_wide": [(20, Y, -0.015), (20, Z, -0.015), (14, Y, -0.015)],  # - purses the lips (pucker)
    "eye_left": [(11, Y, -0.02)],     # the image-left eye: + opens wide, - closes (one keypoint per eye: even)
    "eye_right": [(15, Y, -0.024)],   # the image-right eye (its keypoint is weaker: measured on three faces)
    "gaze_x": [(11, X, 0.015), (15, X, 0.015)],         # + looks toward the image's right side
    # + looks up. The lids are pulled back against the gaze keypoints, or they just open and close
    "gaze_y": [(11, Y, -0.015), (15, Y, -0.015), (13, Y, -0.008), (16, Y, -0.008)],
    "brows": [(1, Y, 0.015), (2, Y, -0.015)],         # + raise, - lower
    "furrow": [(1, X, 0.01), (2, X, -0.01), (1, Y, -0.003), (2, Y, 0.003)],
    "brow_tilt": [(1, Y, 0.018), (2, Y, 0.018)],      # one side up, the other down
}
ROT = {"head_nod": 0, "head_turn": 1, "head_tilt": 2}  # pitch, yaw, roll; the lever value is in degrees
ROT_SIGN = {"head_nod": -1.0, "head_turn": -1.0, "head_tilt": 1.0}  # nod + = down; turn and tilt + = toward the image right
# a jaw that opens drops the chin a little: the head nods with it
COUPLED_ROT = {"mouth_open": (0, -3.0)}


FACE = [n for n in LEVERS]  # the levers a photo's own expression can be read into (head angle is read separately)
ABS_RANGE = {"smile": (-1, 1), "smirk": (-1, 1), "mouth_open": (0, 1), "mouth_wide": (-1, 1), "eye_left": (-1, 0.5),
             "eye_right": (-1, 0.5), "gaze_x": (-1, 1), "gaze_y": (-1, 1), "brows": (-1, 1), "furrow": (0, 1),
             "brow_tilt": (-1, 1)}


def read(exp):
    """The photo's own expression as lever values (least squares of LivePortrait's 21 x 3 offsets on the lever
    recipes): where the drag editor's handles start. exp: (1, 21, 3) tensor."""
    import numpy as np

    A = np.zeros((63, len(FACE)), np.float32)
    for j, n in enumerate(FACE):
        for kp, ax, c in LEVERS[n]:
            A[kp * 3 + ax, j] += c
    e = exp.reshape(-1).float().cpu().numpy()
    sol = np.linalg.lstsq(A, e, rcond=None)[0]
    return {n: round(float(min(max(v, ABS_RANGE[n][0]), ABS_RANGE[n][1])), 2) for n, v in zip(FACE, sol)}


# Presets are targets: the face the expression should end up as, in lever values (anything not named is 0, so a
# preset also calms what it doesn't use). The node subtracts the photo's own expression (read) to get the levers,
# so "Happy" on a face that already smiles doesn't over-smile, and an open mouth closes toward the target.
PRESETS = {
    "Neutral": {},
    "Happy": {"smile": 0.6, "eye_left": -0.15, "eye_right": -0.15, "brows": 0.1},
    "Big smile": {"smile": 1.0, "mouth_open": 0.25, "eye_left": -0.25, "eye_right": -0.25, "brows": 0.15},
    "Laugh": {"smile": 1.0, "mouth_open": 0.65, "eye_left": -0.35, "eye_right": -0.35, "brows": 0.3},
    "Surprised": {"brows": 1.0, "mouth_open": 0.5, "eye_left": 0.35, "eye_right": 0.35},
    "Angry": {"furrow": 1.0, "brows": -0.6, "smile": -0.4, "mouth_wide": -0.2, "eye_left": -0.2, "eye_right": -0.2},
    "Sad": {"furrow": 0.8, "brows": 0.2, "smile": -0.6, "eye_left": -0.35, "eye_right": -0.35, "gaze_y": -0.4},
    "Worried": {"furrow": 1.0, "brows": 0.35, "smile": -0.2, "mouth_wide": 0.2},
    "Skeptical": {"brow_tilt": 0.8, "mouth_wide": -0.3, "smile": -0.1, "eye_right": -0.2},
    "Wink": {"eye_left": -1.0, "smile": 0.5},
}


def symmetric(read_values):
    """The read as presets use it: mouth open, smile and the eyes as a pair (their average). One-sided readings
    pick up head tilt (a neutral face read brow_tilt 0.68), and presets then 'fixed' an asymmetry that isn't there."""
    e = (float(read_values.get("eye_left", 0.0)) + float(read_values.get("eye_right", 0.0))) / 2
    # only what the read gets right is subtracted: how open the mouth and eyes are, and the smile. The rest of the
    # fit is noise (a calm face read brows -0.75, mouth_wide -0.96) and "correcting" it bent the face
    return {"mouth_open": float(read_values.get("mouth_open", 0.0)), "smile": float(read_values.get("smile", 0.0)),
            "eye_left": e, "eye_right": e}


def preset_levers(name, read_values, strength=1.0):
    """Lever values that move this face (read_values) toward the preset target, strength 0..1 of the way."""
    target = PRESETS.get(name)
    if target is None:
        return None
    read_values = symmetric(read_values)
    out = {}
    for n in FACE:
        v = (float(target.get(n, 0.0)) - float(read_values.get(n, 0.0))) * float(strength)
        out[n] = round(min(1.0, max(-1.0, v)), 2)
    return out


def offsets(values):
    """values: lever name -> float. Returns ((21, 3) keypoint offsets, (pitch, yaw, roll) degrees)."""
    d = torch.zeros(21, 3)
    rot = [0.0, 0.0, 0.0]
    for name, v in values.items():
        v = float(v or 0.0)
        if not v:
            continue
        for kp, ax, c in LEVERS.get(name, ()):
            d[kp, ax] += c * v
        if name in ROT:
            rot[ROT[name]] += ROT_SIGN[name] * v
        if name in COUPLED_ROT:
            i, g = COUPLED_ROT[name]
            rot[i] += g * v
    return d, tuple(rot)
