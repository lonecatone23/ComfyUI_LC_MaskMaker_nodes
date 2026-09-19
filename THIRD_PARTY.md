# Third-party code

Model code in `vendor/` is copied from other projects, with the OpenCV / preprocessing parts removed
and imports adjusted. Weights are never included; see "Models and licensing" in the README.

| Folder | What | Origin | License |
|---|---|---|---|
| `vendor/depth_anything_v2/` | Depth Anything V2 network (DPT head) and DINOv2 backbone layers | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), via [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Apache-2.0 (`LICENSE-Apache-2.0.txt`) |
| `vendor/normalbae/` | NormalBAE surface normal network | [baegwangbin/surface_normal_uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty), via [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | MIT (`LICENSE`) |

The code licenses do not cover the pretrained weights. Depth Anything V2 Base / Large / Giant weights are
CC-BY-NC-4.0 (Small is Apache-2.0). The BAE `scannet.pt` weights were trained on ScanNet and carry no
clear license; check before commercial use.
