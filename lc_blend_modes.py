"""
Blend modes in pure torch. Standard (Photoshop / W3C compositing) formulas.

Every function takes the backdrop `b` and the source (layer) `s` as (..., 3) float tensors
in 0-1 and returns the blended color, before opacity / mask compositing.
"""

import torch

EPS = 1e-6


# ---- helpers for the non-separable modes (W3C compositing spec) -----------------
def _lum(c):
    return (0.3 * c[..., 0:1] + 0.59 * c[..., 1:2] + 0.11 * c[..., 2:3])


def _clip_color(c):
    l = _lum(c)
    n = c.amin(-1, keepdim=True)
    x = c.amax(-1, keepdim=True)
    c = torch.where(n < 0, l + (c - l) * l / (l - n).clamp_min(EPS), c)
    c = torch.where(x > 1, l + (c - l) * (1 - l) / (x - l).clamp_min(EPS), c)
    return c


def _set_lum(c, l):
    return _clip_color(c + (l - _lum(c)))


def _sat(c):
    return c.amax(-1, keepdim=True) - c.amin(-1, keepdim=True)


def _set_sat(c, s):
    mn = c.amin(-1, keepdim=True)
    rng = c.amax(-1, keepdim=True) - mn
    return torch.where(rng > EPS, (c - mn) * s / rng.clamp_min(EPS), torch.zeros_like(c))


# ---- separable modes ------------------------------------------------------------
def normal(b, s):
    return s


def darken(b, s):
    return torch.minimum(b, s)


def multiply(b, s):
    return b * s


def color_burn(b, s):
    return torch.where(b >= 1.0, torch.ones_like(b), (1.0 - (1.0 - b) / s.clamp_min(EPS)).clamp(0, 1))


def linear_burn(b, s):
    return (b + s - 1.0).clamp(0, 1)


def darker_color(b, s):
    return torch.where(_lum(s) < _lum(b), s, b)


def lighten(b, s):
    return torch.maximum(b, s)


def screen(b, s):
    return 1.0 - (1.0 - b) * (1.0 - s)


def color_dodge(b, s):
    return torch.where(b <= 0.0, torch.zeros_like(b), (b / (1.0 - s).clamp_min(EPS)).clamp(0, 1))


def linear_dodge(b, s):
    return (b + s).clamp(0, 1)


def lighter_color(b, s):
    return torch.where(_lum(s) > _lum(b), s, b)


def hard_light(b, s):
    return torch.where(s <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def overlay(b, s):
    return hard_light(s, b)


def soft_light(b, s):
    d = torch.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, b.clamp_min(0).sqrt())
    return torch.where(s <= 0.5, b - (1.0 - 2.0 * s) * b * (1.0 - b), b + (2.0 * s - 1.0) * (d - b))


def vivid_light(b, s):
    burn = 1.0 - (1.0 - b) / (2.0 * s).clamp_min(EPS)
    dodge = b / (2.0 * (1.0 - s)).clamp_min(EPS)
    return torch.where(s <= 0.5, burn, dodge).clamp(0, 1)


def linear_light(b, s):
    return (b + 2.0 * s - 1.0).clamp(0, 1)


def pin_light(b, s):
    return torch.where(s <= 0.5, torch.minimum(b, 2.0 * s), torch.maximum(b, 2.0 * s - 1.0))


def hard_mix(b, s):
    return (b + s >= 1.0).to(b.dtype)


def difference(b, s):
    return (b - s).abs()


def exclusion(b, s):
    return b + s - 2.0 * b * s


def subtract(b, s):
    return (b - s).clamp(0, 1)


def divide(b, s):
    return (b / s.clamp_min(EPS)).clamp(0, 1)


def grain_extract(b, s):
    return (b - s + 0.5).clamp(0, 1)


def grain_merge(b, s):
    return (b + s - 0.5).clamp(0, 1)


# ---- non-separable modes --------------------------------------------------------
def hue(b, s):
    return _set_lum(_set_sat(s, _sat(b)), _lum(b))


def saturation(b, s):
    return _set_lum(_set_sat(b, _sat(s)), _lum(b))


def color(b, s):
    return _set_lum(s, _lum(b))


def luminosity(b, s):
    return _set_lum(b, _lum(s))


# ---- registry -------------------------------------------------------------------
def dissolve(b, s, alpha, seed=0):
    """Handled by the compositor (needs alpha): returns (blend, alpha) with alpha made binary."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.rand(alpha.shape, generator=g).to(alpha.device)
    return s, (noise < alpha).to(alpha.dtype)


BLEND_MODES = {
    "normal": normal,
    "dissolve": None,  # special-cased in blend()
    "darken": darken,
    "multiply": multiply,
    "color_burn": color_burn,
    "linear_burn": linear_burn,
    "darker_color": darker_color,
    "lighten": lighten,
    "screen": screen,
    "color_dodge": color_dodge,
    "linear_dodge (add)": linear_dodge,
    "lighter_color": lighter_color,
    "overlay": overlay,
    "soft_light": soft_light,
    "hard_light": hard_light,
    "vivid_light": vivid_light,
    "linear_light": linear_light,
    "pin_light": pin_light,
    "hard_mix": hard_mix,
    "difference": difference,
    "exclusion": exclusion,
    "subtract": subtract,
    "divide": divide,
    "hue": hue,
    "saturation": saturation,
    "color": color,
    "luminosity": luminosity,
    "grain_extract": grain_extract,
    "grain_merge": grain_merge,
}

MODE_NAMES = list(BLEND_MODES.keys())


def blend(backdrop, source, alpha, mode, opacity, swap_roles=False, seed=0):
    """Composite `source` over `backdrop` with a blend mode.

    backdrop, source: (B,H,W,3) float 0-1. alpha: (B,H,W,1) source coverage 0-1.
    opacity: 0-1 float. Returns the (B,H,W,3) result.
    """
    a = alpha * float(opacity)
    if mode == "dissolve":
        blended, a = dissolve(backdrop, source, a, seed)
    else:
        fn = BLEND_MODES[mode]
        blended = fn(source, backdrop) if swap_roles else fn(backdrop, source)
    return (backdrop * (1.0 - a) + blended.clamp(0, 1) * a).clamp(0, 1)


def blend_rgba(backdrop, backdrop_alpha, source, alpha, mode, opacity, swap_roles=False, seed=0):
    """Like blend(), but the backdrop may be (partly) transparent (W3C compositing).

    backdrop (B,H,W,3), backdrop_alpha (B,H,W,1), source (B,H,W,3), alpha (B,H,W,1) source coverage.
    Returns (color (B,H,W,3), alpha (B,H,W,1)).
    """
    a = alpha * float(opacity)
    if mode == "dissolve":
        blended, a = dissolve(backdrop, source, a, seed)
    else:
        fn = BLEND_MODES[mode]
        blended = (fn(source, backdrop) if swap_roles else fn(backdrop, source)).clamp(0, 1)
    ab = backdrop_alpha
    mixed = (1.0 - ab) * source + ab * blended          # where the backdrop is empty the layer shows as is
    ar = a + ab * (1.0 - a)
    ratio = torch.where(ar > 1e-6, a / ar.clamp_min(1e-6), torch.zeros_like(ar))
    color = (1.0 - ratio) * backdrop + ratio * mixed
    return color.clamp(0, 1), ar.clamp(0, 1)
