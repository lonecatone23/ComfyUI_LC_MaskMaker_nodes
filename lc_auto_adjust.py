"""
LC Auto Adjust
--------------
One-click auto levels with a few trims. Stretches the tonal range of the image (per channel, or
luminance and/or saturation only), then balance / brightness / contrast / saturation trims, then
blends the result back with the original by strength. An optional mask limits both where the
levels are measured and where the result is applied. Pure torch, no extra dependencies.
"""

import torch

from .lc_matting import get_device

MODES = ["RGB", "lum + sat", "mono", "luminance", "saturation"]


# ---- color spaces (all float 0-1, channels last) ---------------------------------
def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.amax(-1), rgb.amin(-1)
    d = mx - mn
    v = mx
    s = torch.where(mx > 0, d / mx.clamp_min(1e-8), torch.zeros_like(mx))
    dd = d.clamp_min(1e-8)
    h = torch.where(mx == r, ((g - b) / dd) % 6.0,
                    torch.where(mx == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0)) / 6.0
    h = torch.where(d > 0, h, torch.zeros_like(h))
    return torch.stack([h, s, v], -1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], -1).gather(-1, i.unsqueeze(-1))[..., 0]
    g = torch.stack([t, v, v, q, p, p], -1).gather(-1, i.unsqueeze(-1))[..., 0]
    b = torch.stack([p, p, t, v, v, q], -1).gather(-1, i.unsqueeze(-1))[..., 0]
    return torch.stack([r, g, b], -1)


_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def _rgb_to_lab(rgb):
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0) ** 2.4)
    xyz = lin @ _M.to(rgb).T / _WHITE.to(rgb)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-8) ** (1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    return torch.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx, fz = fy + lab[..., 1] / 500.0, fy - lab[..., 2] / 200.0
    f = torch.stack([fx, fy, fz], -1)
    xyz = torch.where(f ** 3 > 0.008856, f ** 3, (f - 16.0 / 116.0) / 7.787) * _WHITE.to(lab)
    lin = (xyz @ torch.linalg.inv(_M).to(lab).T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp_min(0) ** (1.0 / 2.4) - 0.055).clamp(0, 1)


# ---- auto level ------------------------------------------------------------------
def _stretch(ch, sel, lo_frac=0.0005):
    """Stretch a 0-1 channel so the used range fills 0-1.

    The used range comes from a 256-bin histogram of the selected pixels: the lowest and highest
    bins holding more than 0.05% of them.
    """
    vals = (ch[sel] * 255.0).round().clamp(0, 255)
    if vals.numel() == 0:
        return ch
    hist = torch.histc(vals.float(), bins=256, min=0, max=255)
    ok = torch.nonzero(hist > hist.sum() * lo_frac).flatten()
    if ok.numel() == 0:
        return ch
    lo, hi = float(ok.min()) / 255.0, float(ok.max()) / 255.0
    if hi - lo < 1e-6:
        return ch
    return ((ch - lo) / (hi - lo)).clamp(0, 1)


def _balance_gamma(b):
    return 0.00005 * b * b - 0.01 * b + 1.0


def _offset(v):
    return v / 100.0 + 1.0 if v < 0 else v / 50.0 + 1.0


def auto_adjust(img, sel, soft, mode, strength, brightness, contrast, saturation, red, green, blue):
    """img (H,W,3) 0-1, sel (H,W) bool used for the histogram, soft (H,W,1) apply mask 0-1."""
    if mode == "RGB":
        out = torch.stack([_stretch(img[..., c], sel) for c in range(3)], -1)
    elif mode == "lum + sat":
        hsv = _rgb_to_hsv(img)
        hsv = torch.cat([hsv[..., :1], _stretch(hsv[..., 1], sel).unsqueeze(-1), hsv[..., 2:]], -1)
        lab = _rgb_to_lab(_hsv_to_rgb(hsv))
        lab = torch.cat([(_stretch(lab[..., 0] / 100.0, sel) * 100.0).unsqueeze(-1), lab[..., 1:]], -1)
        out = _lab_to_rgb(lab)
    elif mode == "luminance":
        lab = _rgb_to_lab(img)
        lab = torch.cat([(_stretch(lab[..., 0] / 100.0, sel) * 100.0).unsqueeze(-1), lab[..., 1:]], -1)
        out = _lab_to_rgb(lab)
    elif mode == "saturation":
        hsv = _rgb_to_hsv(img)
        hsv = torch.cat([hsv[..., :1], _stretch(hsv[..., 1], sel).unsqueeze(-1), hsv[..., 2:]], -1)
        out = _hsv_to_rgb(hsv)
    else:  # mono
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
        out = _stretch(gray, sel).unsqueeze(-1).expand(-1, -1, 3)

    if mode != "mono":
        for c, bal in enumerate((red, green, blue)):
            if bal:
                out = out.clone()
                out[..., c] = out[..., c].clamp_min(0) ** _balance_gamma(bal)

    if brightness:
        out = (out * _offset(brightness)).clamp(0, 1)
    if contrast:
        mean = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]).mean()
        out = (mean + _offset(contrast) * (out - mean)).clamp(0, 1)
    if saturation:
        gray = (0.299 * out[..., 0:1] + 0.587 * out[..., 1:2] + 0.114 * out[..., 2:3])
        out = (gray + _offset(saturation) * (out - gray)).clamp(0, 1)

    out = img + (out - img) * (strength / 100.0)
    return img + (out - img) * soft


class LCAutoAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "An alpha channel, if there is one, is kept and used as the mask when no mask is connected."}),
                "mode": (MODES, {
                    "default": "RGB",
                    "tooltip": (
                        "RGB: stretch each color channel on its own (also removes color casts).\n"
                        "lum + sat: stretch saturation, then luminance.\n"
                        "luminance: stretch brightness only, colors stay put.\n"
                        "saturation: stretch saturation only.\n"
                        "mono: stretch a black and white version."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "How much of the result is used, in percent.",
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Trim after the auto levels.",
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Trim after the auto levels.",
                }),
                "saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Trim after the auto levels.",
                }),
                "red": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Color balance: positive brightens the red channel, negative darkens it. Not used in mono mode.",
                }),
                "green": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Color balance for the green channel.",
                }),
                "blue": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Color balance for the blue channel.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Limits both where the levels are measured and where the result is applied."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "LC MaskMaker/adjust"
    DESCRIPTION = (
        "Auto levels (per channel, luminance, saturation or mono) with balance, brightness, contrast and "
        "saturation trims, a strength blend, and an optional mask."
    )

    def adjust(self, image, mode, strength, brightness, contrast, saturation, red, green, blue, mask=None):
        dev = get_device()
        b, h, w, c = image.shape
        m_in = None
        if mask is not None:
            m_in = mask.unsqueeze(0) if mask.dim() == 2 else mask

        outs = []
        for i in range(b):
            frame = image[i].to(dev).float()
            rgb = frame[..., :3]
            if m_in is not None:
                m = m_in[min(i, m_in.shape[0] - 1)].to(dev).float()
                if m.shape != (h, w):
                    m = torch.nn.functional.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            elif c == 4:
                m = frame[..., 3]
            else:
                m = torch.ones(h, w, device=dev)
            m = m.clamp(0, 1)
            res = auto_adjust(rgb, m > 0.5, m.unsqueeze(-1), mode, float(strength), float(brightness),
                              float(contrast), float(saturation), float(red), float(green), float(blue))
            if c == 4:
                res = torch.cat([res, frame[..., 3:4]], -1)
            outs.append(res.clamp(0, 1).cpu())
        return (torch.stack(outs, 0),)


NODE_CLASS_MAPPINGS = {
    "LCAutoAdjust": LCAutoAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAutoAdjust": "LC Auto Adjust 🔆",
}
