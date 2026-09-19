# ComfyUI LC MaskMaker Nodes

Mask, matting, segmentation and image-adjustment nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by [lonecatone23](https://github.com/lonecatone23).

- **Repo:** [https://github.com/lonecatone23/ComfyUI_LC_MaskMaker_nodes](https://github.com/lonecatone23/ComfyUI_LC_MaskMaker_nodes)
- **Civitai:** [lonecatone23](https://civitai.com/user/lonecatone23)
- **Support:** [Buy me a ☕](https://ko-fi.com/lonecatone)
- **Version:** 0.10.9 · **10 Python nodes**

> Companion to [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes). LC123 stays import-free. Anything that needs a model lives here and loads it on demand.

Release history lives in **git tags**. This page describes the pack **as it is right now**, not a changelog.

---

## Nodes

| Node | What it does |
|---|---|
| **LC Outpaint 🖌️➕** | Drag the canvas edges outward on the node. Drag inside the source to reposition it. **free** or locked to a common aspect ratio, grey fill. Outputs **control_image**, **control_mask** (white = new area), **mask_image**, **width**, **height**. **hold_mask**: **snap_to_image** puts the box back on the image every run and passes the image through as is (so a new image always starts clean); **held** keeps the canvas you set up and passes the image through with it. **block** set to **if_empty_mask** stops everything downstream when the mask is empty: with snap_to_image that means every run (drag the edges out, switch to held, run again), with held only if you have not pulled an edge out. Run once to get the preview. *note:* **snap_to** rounds the final size up to a multiple (8 by default), set it to 1 to turn that off. |
| **LC Create Mask 🖼️✏️** | Draw a mask right on the node. Click to add points (no limit), drag to move, shift+click removes one. Flip the toggle to **pencil** for freehand. **smooth**, **invert**, **feather**, and **opacity** are built in. Undo and Clear buttons on the node. **block** (top widget) set to **if_empty_mask** stops everything downstream until you have drawn something, same idea as Impact Pack's Preview Bridge. Run once to get the preview. |
| **LC Mask Refine ✨** | Tighten a mask with a **trimap**. **edge_erode** pulls the sure-foreground in from the mask edge, **edge_dilate** sets how far the unknown band reaches out. Only that band is decided by the **method** (**vitmatte** for hair and fine detail, **guided_filter** for speed, or **none**), so a narrower band means a tighter mask. Also **grow** (negative shrinks), **fix_gap**, **black_point** / **white_point**, **feather**, and **edge_color_spread** to cut background halos on the cutout. Plug a mask into **detail_region** (like hair) to give just that area a wider trimap (**detail_erode** / **detail_dilate**). Outputs **cutout** (RGBA), **mask**, **trimap**. Before/after wipe on the node, same as the LC123 previews: hover and the after image sits left of the pointer. **preview_view** switches between cutout, mask and trimap. Run once to get the preview. 💡 Start with 10 / 10, then shrink both toward 3 for a tight edge. **Micro adjust:** every pixel value (**grow**, **fix_gap**, **edge_erode**, **edge_dilate**, the **detail_** ones, **feather**) takes decimals in 0.1 px steps, the points step by 0.001, and the edges are true circles, not blocks. Wire sliders in as **FLOAT** (INT sockets will not plug into these). ⚠️ Numbers are real pixels. |
| **LC Remove Background ✂️** | Cut the subject out with a BiRefNet model. Outputs **cutout** (RGBA), **mask**, and **on_background** (subject on a solid color). Optional **refine** (guided_filter or vitmatte) with the same trimap controls (**edge_erode** / **edge_dilate**), plus **grow**, **black_point** / **white_point**, and **feather**. No preview on the node (add LC Mask Refine after it if you want one). Uses ComfyUI's built-in background-removal loader, so no extra dependency. 💡 Raise **black_point** to clear faint haze, use **vitmatte** for hair. |
| **LC Image Blend Advance 🎚️** | Place a layer on a background and blend it. **x_percent** / **y_percent** (0.01 steps), **scale**, **aspect_ratio**, **rotate**, **mirror**, and 29 blend modes, with **opacity** in 0.1 steps and an optional **layer_mask** (multiplied with the layer's own alpha). **background_image** is optional: leave it off and the layer goes on a transparent canvas the size of the layer (RGBA out, like LayerStyle's V3). **transform_method**: lanczos, bicubic, bilinear, nearest. Outputs **image** and **mask** (where the layer landed). Pure torch, no extra dependency.|
| **LC Segment Anything 🎯** | Select things by describing them: type **hair, eyes, bikini** and each word is found and combined. **engine**: **grounding_dino + sam** (GroundingDINO finds boxes, SAM makes masks) or **sam3** (SAM 3 does both, using ComfyUI's built-in SAM 3 support). Then the same refine as Mask Refine (**refine**, **edge_erode** / **edge_dilate**, **black_point** / **white_point**) and a before/after wipe on the node. Outputs **cutout** (RGBA), **mask**, **raw_mask**. 💡 For sam3, **sam3_threshold** 0.5 is a good start; near 0.3 it invents matches for words that are not in the image. |
| **LC Person Mask 🧍** | Pick the parts of a person: **face**, **hair**, **body**, **clothes**, **accessories**, **background**, combined into one mask. **engine**: **segformer** (higher resolution, body is arms and legs only) or **mediapipe** (body includes torso and neck skin). Then the same refine as Mask Refine. No preview on the node (add LC Mask Refine after it if you want one). Outputs **cutout** (RGBA), **mask**, **raw_mask**. 💡 For skin work, enable **face** and **body**. |
| **LC Auto Adjust 🔆** | One-click auto levels. **mode**: **RGB** (each channel on its own, also removes color casts), **lum + sat**, **luminance**, **saturation**, or **mono**. Then **red** / **green** / **blue** balance, **brightness**, **contrast**, **saturation** trims, and **strength** to blend back toward the original. All in 0.1 steps. An optional **mask** limits both where the levels are measured and where the result is applied (an alpha channel works as the mask when none is connected). Pure torch, no extra dependency. |
| **LC Depth Anything V2 🌊** | Depth map from one image, normalized 0-1 per image, **bright = near** unless **invert** is on. Reads the original Depth Anything V2 weights, so a copy you already have in `models/depthanything` works (it also finds the ones in comfyui_controlnet_aux). Outputs **depth** (3-channel image) and **depth_mask**. **resolution** in steps of 14. Matches comfyui_controlnet_aux's Depth Anything V2 (correlation 1.000). Model code is vendored, no extra package. |
| **LC Normal Map (BAE) 🗺️** | Surface normal map (RGB = XYZ) with the BAE network, the same style ControlNet's normal_bae makes. **resolution**, and **flip_y** for OpenGL vs DirectX. Matches comfyui_controlnet_aux's BAE (correlation 0.9999). Needs the `timm` package (the installer adds it if missing). 💡 Both maps are what LC Lighting Control in LC123 wants as its **depth_map** and **normal_map**. |

---

## Example workflows

| Workflow | What it does |
|---|---|
| [`workflows/LC Maskmaker nodes.json`](workflows/LC%20Maskmaker%20nodes.json) | Every node in the pack wired to one image, each with its own preview, so you can compare them side by side. Also uses **LC Image Desaturate** and **LC Skin Beauty** from LC123. Bilingual (English / Chinese) note on the canvas. |
| [`workflows/LC Maskmaker Krea2 Outpaint.json`](workflows/LC%20Maskmaker%20Krea2%20Outpaint.json) | Outpaint with Krea 2 Turbo and the Identity Edit LoRA. **LC Outpaint** sets the canvas, the grey area goes in as the reference image and the model repaints it to match the scene. Needs LC123, comfyui-krea2edit, KJNodes and cg-use-everywhere. Bilingual (English / Chinese) note on the canvas. |

---

## Models and licensing

- ✋ Models are **never bundled or redistributed**. Model-backed nodes download from the original source the first time you use them, or reuse a copy you already have.
- Each model in a dropdown is labelled with its license, so you can see what you are picking.
- **VITMatte** (LC Mask Refine): `vitmatte-small-composition-1k` and `vitmatte-base-composition-1k`, both Apache-2.0. Saved under `models/vitmatte`. A copy you already have in `models/vitmatte` is picked up automatically.
- **BiRefNet** (LC Remove Background): general, matting, portrait, and HR 2048 variants, all MIT. Saved under `models/background_removal`, the same folder ComfyUI's own loader reads.
- **RMBG-2.0** (BRIA): used if you already have it in `models/RMBG/RMBG-2.0`. Not downloaded for you. ⚠️ Non-commercial license.
- Older BiRefNet `.pth` files (like `BiRefNet-ep480.pth`) use an older layout that ComfyUI cannot load, so they are not listed.
- **GroundingDINO + SAM** (LC Segment Anything): `grounding-dino-tiny` / `-base` and `sam-vit-base` / `-large` / `-huge`, all Apache-2.0, Hugging Face format. Saved under `models/grounding-dino` and `models/sam`. Your original-format `.pth` files there are not used (different layout).
- **SAM 3** (LC Segment Anything, sam3 engine): Meta's original `facebook/sam3` is gated (manual approval), so it is never downloaded for you. Your own `sam3.safetensors` in `models/sam3` (or ComfyUI's `models/detection`) is picked up. For a one-click option, the dropdown offers **SAM 3.1** (`sam3.1_multiplex_fp16.safetensors`, 1.7 GB) from `Comfy-Org/sam3.1`, the same ungated copy ComfyUI's own SAM 3 template uses, saved to `models/sam3`. SAM License terms apply (Hugging Face lists it as *other*).
- **Segformer clothes** (LC Person Mask): `mattmdjaga/segformer_b2_clothes`, saved under `models/segformer_b2_clothes`. ⚠️ Its Hugging Face card says license *other* and it is built on NVIDIA's SegFormer, so treat it as **not cleared for commercial use** until you have checked.
- **MediaPipe selfie multiclass** (LC Person Mask): Google's `selfie_multiclass_256x256.tflite`, saved under `models/mediapipe`. Check its model card for terms. The `mediapipe` Python package is not installed by this pack.
- **Depth Anything V2** (LC Depth Anything V2): **Small is Apache-2.0**, **Base / Large / Giant are CC-BY-NC-4.0 (non-commercial)**. Saved under `models/depthanything`. Small is listed first so it is the default.
- **BAE normals** (LC Normal Map): `scannet.pt` from lllyasviel/Annotators, saved under `models/normalbae`. ⚠️ Trained on ScanNet and the Hugging Face card only says *other*. Treat as **not cleared for commercial use** until you have checked.
- ⚠️ Some models are **non-commercial**. The label tells you. Checking the license before commercial use is on you.

---

## License

MIT for the code in this pack.

Third-party model code in `vendor/` is credited in [THIRD_PARTY.md](THIRD_PARTY.md).
