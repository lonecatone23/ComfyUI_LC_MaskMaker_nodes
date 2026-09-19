/**
 * LC Outpaint 🖌️➕ — drag the canvas edges outward.
 * Same interaction model as LC Image Crop: hidden percent widgets, on-node
 * preview after a run, global mouseup so a drag always releases.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASS = "LCImageOutpaint";
const HANDLE = 12;
const HIT = 16;
const MAX_PCT = 400;
const HEADROOM = 0.12; // empty margin around the canvas so it has room to grow
const CANVAS_FILL = "#808080";

const ASPECT_MAP = {
  free: null,
  original: "original",
  "1:1": 1,
  "4:3": 4 / 3,
  "3:2": 3 / 2,
  "16:9": 16 / 9,
  "3:4": 3 / 4,
  "2:3": 2 / 3,
  "9:16": 9 / 16,
};

const SIDES = ["left", "top", "right", "bottom"];

let _activeNode = null;

function viewUrl(meta) {
  if (!meta) return null;
  const q = new URLSearchParams();
  q.set("filename", meta.filename || "");
  q.set("subfolder", meta.subfolder != null ? meta.subfolder : "");
  q.set("type", meta.type || "temp");
  return api.apiURL(`/view?${q.toString()}`);
}

function loadImg(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function getW(node, name) {
  return (node.widgets || []).find((x) => x && x.name === name);
}

function hideWidget(w) {
  if (!w || w._lcHidden) return;
  w._lcHidden = true;
  w.computeSize = () => [0, -4];
  w.draw = () => {};
  w.type = "hidden";
  if (w.element) {
    try { w.element.style.display = "none"; } catch (_) {}
  }
}

// ---- expansion state (percent of source; l/r of width, t/b of height) ----

function readExp(node) {
  if (node._lcOpLive) return { ...node._lcOpLive };
  const num = (n) => {
    const w = getW(node, n);
    const v = w ? Number(w.value) : 0;
    return Number.isFinite(v) ? v : 0;
  };
  return {
    l: num("left"),
    t: num("top"),
    r: num("right"),
    b: num("bottom"),
    aspect: getW(node, "aspect")?.value || "free",
  };
}

function writeExp(node, e) {
  node._lcOpLive = { ...e };
  const round = (v) => Math.round(v * 100) / 100;
  for (const [name, val] of [["left", e.l], ["top", e.t], ["right", e.r], ["bottom", e.b]]) {
    const w = getW(node, name);
    if (w) w.value = round(val);
  }
}

function clampExp(e) {
  for (const k of ["l", "t", "r", "b"]) e[k] = Math.max(0, Math.min(MAX_PCT, e[k]));
  return e;
}

// ---- layout ----

function widgetsBottom(node) {
  let bottom = 0;
  let any = false;
  for (const w of node.widgets || []) {
    if (!w || w.type === "hidden" || w._lcHidden) continue;
    if (typeof w.last_y === "number") {
      const h = typeof w.computeSize === "function" ? (w.computeSize(node.size[0])?.[1] ?? 20) : 20;
      bottom = Math.max(bottom, w.last_y + Math.max(20, h));
      any = true;
    }
  }
  if (any) return bottom;
  const rows = Math.max(node.inputs?.length || 0, node.outputs?.length || 0);
  let y = rows * 20 + 8;
  for (const w of node.widgets || []) {
    if (!w || w.type === "hidden" || w._lcHidden) continue;
    const h = typeof w.computeSize === "function" ? (w.computeSize(node.size[0])?.[1] ?? 24) : 24;
    y += Math.max(20, h) + 4;
  }
  return y;
}

function contentTop(node) {
  return widgetsBottom(node) + PAD;
}

function defaultHeight(node) {
  return contentTop(node) + lcPreviewHeight() + PAD;
}

function srcDims(node) {
  const img = node._lcOpImg;
  const w = node._lcSrcW || img?.naturalWidth || 0;
  const h = node._lcSrcH || img?.naturalHeight || 0;
  return w > 0 && h > 0 ? { w, h } : null;
}

// Layout maps source pixels -> node coordinates. `sx, sy` is the screen position
// of the source's top-left corner; the canvas rect hangs off it.
function computeLayout(node, e) {
  if (node._lcOpFrozen) return { ...node._lcOpFrozen };
  const dims = srcDims(node);
  if (!dims || !node._lcOpImg) return null;
  const top = contentTop(node);
  const box = { x: PAD, y: top, w: Math.max(1, node.size[0] - PAD * 2), h: Math.max(1, node.size[1] - top - PAD) };
  if (box.h < 40) return null;
  const CW = dims.w * (1 + (e.l + e.r) / 100);
  const CH = dims.h * (1 + (e.t + e.b) / 100);
  const scale = Math.min((box.w * (1 - 2 * HEADROOM)) / CW, (box.h * (1 - 2 * HEADROOM)) / CH);
  const cw = CW * scale;
  const ch = CH * scale;
  const cx = box.x + (box.w - cw) / 2;
  const cy = box.y + (box.h - ch) / 2;
  return {
    scale,
    srcW: dims.w,
    srcH: dims.h,
    sx: cx + (e.l / 100) * dims.w * scale,
    sy: cy + (e.t / 100) * dims.h * scale,
    box,
  };
}

function canvasRect(layout, e) {
  const { scale, srcW, srcH, sx, sy } = layout;
  return {
    x: sx - (e.l / 100) * srcW * scale,
    y: sy - (e.t / 100) * srcH * scale,
    w: srcW * (1 + (e.l + e.r) / 100) * scale,
    h: srcH * (1 + (e.t + e.b) / 100) * scale,
  };
}

function srcRect(layout) {
  return { x: layout.sx, y: layout.sy, w: layout.srcW * layout.scale, h: layout.srcH * layout.scale };
}

function aspectRatio(dims, key) {
  if (!key || key === "free") return null;
  if (key === "original") return dims.w / dims.h;
  return ASPECT_MAP[key] ?? null;
}

// Grow the expansion so the canvas matches `ar`, keeping the source contained.
// `anchor` is the dragged handle id ("" when nothing is being dragged).
function fixAspect(e, ar, srcW, srcH, anchor) {
  const hasE = anchor.includes("e");
  const hasW = anchor.includes("w");
  const hasN = anchor.includes("n");
  const hasS = anchor.includes("s");
  const horiz = hasE || hasW;
  const vert = hasN || hasS;
  let W = srcW * (1 + (e.l + e.r) / 100);
  let H = srcH * (1 + (e.t + e.b) / 100);

  if (horiz && !vert) H = W / ar;
  else if (vert && !horiz) W = H * ar;
  else if (horiz && vert) {
    const hFromW = W / ar;
    if (hFromW >= H) H = hFromW;
    else W = H * ar;
  } else if (W / H > ar) H = W / ar;
  else W = H * ar;

  if (W < srcW) { H *= srcW / W; W = srcW; }
  if (H < srcH) { W *= srcH / H; H = srcH; }

  const split = (total, loPx, hiPx, anchorLo, anchorHi) => {
    if (anchorLo && !anchorHi) loPx = total - hiPx;
    else if (anchorHi && !anchorLo) hiPx = total - loPx;
    else {
      const sum = loPx + hiPx;
      const f = sum > 0 ? loPx / sum : 0.5;
      loPx = total * f;
      hiPx = total - loPx;
    }
    if (loPx < 0) { hiPx += loPx; loPx = 0; }
    if (hiPx < 0) { loPx += hiPx; hiPx = 0; }
    return [loPx, hiPx];
  };

  const [lPx, rPx] = split(W - srcW, (e.l / 100) * srcW, (e.r / 100) * srcW, hasW, hasE);
  const [tPx, bPx] = split(H - srcH, (e.t / 100) * srcH, (e.b / 100) * srcH, hasN, hasS);
  e.l = (lPx / srcW) * 100;
  e.r = (rPx / srcW) * 100;
  e.t = (tPx / srcH) * 100;
  e.b = (bPx / srcH) * 100;
  return clampExp(e);
}

const HANDLES = [
  { id: "nw", x: 0, y: 0 }, { id: "n", x: 0.5, y: 0 }, { id: "ne", x: 1, y: 0 },
  { id: "e", x: 1, y: 0.5 }, { id: "se", x: 1, y: 1 }, { id: "s", x: 0.5, y: 1 },
  { id: "sw", x: 0, y: 1 }, { id: "w", x: 0, y: 0.5 },
];

function hitTest(layout, e, px, py) {
  const c = canvasRect(layout, e);
  for (const h of HANDLES) {
    const hx = c.x + h.x * c.w;
    const hy = c.y + h.y * c.h;
    if (Math.abs(px - hx) <= HIT && Math.abs(py - hy) <= HIT) return { type: "handle", id: h.id };
  }
  const s = srcRect(layout);
  if (px >= s.x && px <= s.x + s.w && py >= s.y && py <= s.y + s.h) return { type: "move" };
  return null;
}

// ---- drawing ----

function drawPreview(node, ctx) {
  const e = readExp(node);
  const layout = computeLayout(node, e);
  if (!layout) return;
  const c = canvasRect(layout, e);
  const s = srcRect(layout);
  const { box } = layout;

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x, box.y, box.w, box.h);
  ctx.clip();

  ctx.fillStyle = CANVAS_FILL;
  ctx.fillRect(c.x, c.y, c.w, c.h);
  ctx.drawImage(node._lcOpImg, s.x, s.y, s.w, s.h);

  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(s.x + 0.5, s.y + 0.5, Math.max(0, s.w - 1), Math.max(0, s.h - 1));
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(c.x + 0.5, c.y + 0.5, Math.max(0, c.w - 1), Math.max(0, c.h - 1));

  for (const h of HANDLES) {
    const hx = c.x + h.x * c.w;
    const hy = c.y + h.y * c.h;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  const outW = Math.round(layout.srcW * (1 + (e.l + e.r) / 100));
  const outH = Math.round(layout.srcH * (1 + (e.t + e.b) / 100));
  ctx.save();
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.textAlign = "center";
  ctx.fillText(`${layout.srcW}×${layout.srcH}  →  ${outW}×${outH}`, box.x + box.w / 2, box.y + box.h - 2);
  ctx.restore();
}

// ---- interaction ----

function processDrag(node, px, py) {
  const drag = node._lcDrag;
  if (!drag) return;
  const { layout, orig: o } = drag;
  const dxPct = ((px - drag.start.x) / layout.scale / layout.srcW) * 100;
  const dyPct = ((py - drag.start.y) / layout.scale / layout.srcH) * 100;
  let e = { ...o };

  if (drag.mode === "move") {
    const totX = o.l + o.r;
    const totY = o.t + o.b;
    e.l = Math.max(0, Math.min(totX, o.l + dxPct));
    e.r = totX - e.l;
    e.t = Math.max(0, Math.min(totY, o.t + dyPct));
    e.b = totY - e.t;
  } else {
    const id = drag.handle;
    if (id.includes("w")) e.l = o.l - dxPct;
    if (id.includes("e")) e.r = o.r + dxPct;
    if (id.includes("n")) e.t = o.t - dyPct;
    if (id.includes("s")) e.b = o.b + dyPct;
    e = clampExp(e);
    const ar = aspectRatio({ w: layout.srcW, h: layout.srcH }, o.aspect);
    if (ar) e = fixAspect(e, ar, layout.srcW, layout.srcH, id);
  }
  writeExp(node, e);
  node.setDirtyCanvas?.(true, true);
}

function endDrag() {
  if (_activeNode) {
    _activeNode._lcDrag = null;
    _activeNode._lcOpFrozen = null;
    _activeNode.setDirtyCanvas?.(true, true);
    _activeNode = null;
  }
}

function snapBox(node) {
  writeExp(node, { l: 0, t: 0, r: 0, b: 0, aspect: getW(node, "aspect")?.value || "free" });
  node.setDirtyCanvas?.(true, true);
}

function outpaintNodes() {
  return (app.graph?._nodes || []).filter((n) => n?.comfyClass === NODE_CLASS || n?.type === NODE_CLASS);
}

function installRunListeners() {
  if (window.__lcOutpaintRun) return;
  window.__lcOutpaintRun = true;
  api.addEventListener("execution_success", () => {
    for (const n of outpaintNodes()) {
      if (n._lcResetAfter) snapBox(n);
      n._lcResetAfter = false;
    }
  });
  const failed = () => { for (const n of outpaintNodes()) n._lcResetAfter = false; };
  api.addEventListener("execution_error", failed);
  api.addEventListener("execution_interrupted", failed);
}

function installGlobalMouseUp() {
  if (window.__lcOutpaintMouseUp) return;
  window.__lcOutpaintMouseUp = true;
  const end = () => endDrag();
  window.addEventListener("pointerup", end, true);
  window.addEventListener("mouseup", end, true);
  window.addEventListener("pointercancel", end, true);
  window.addEventListener("blur", end, true);
}

app.registerExtension({
  name: "LCMaskMaker.ImageOutpaint",

  async setup() {
    installGlobalMouseUp();
    installRunListeners();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this._lcOpImg = null;
      this._lcSrcW = 0;
      this._lcSrcH = 0;
      this._lcDrag = null;
      this._lcOpFrozen = null;
      this._lcOpLive = null;
      for (const name of SIDES) hideWidget(getW(this, name));

      const applySize = () => {
        if (this.size) this.size = [DEFAULT_W, defaultHeight(this)];
        this.setDirtyCanvas?.(true, true);
      };
      applySize();
      // ComfyUI's add-node search dialog re-assigns size right after this hook
      // returns; a node restored from a saved workflow (onConfigure runs in the
      // same tick) must keep its saved size.
      setTimeout(() => {
        if (!this._lcOpConfigured) applySize();
      }, 0);

      // Switching to snap_to_image puts the box back on the image right away, so a new image starts clean
      const hold = getW(this, "hold_mask");
      if (hold) {
        const prevHold = hold.callback;
        hold.callback = (val, ...rest) => {
          const out = prevHold?.call(this, val, ...rest);
          if (!val) {
            writeExp(this, { l: 0, t: 0, r: 0, b: 0, aspect: getW(this, "aspect")?.value || "free" });
            this.setDirtyCanvas?.(true, true);
          }
          return out;
        };
      }

      const aspect = getW(this, "aspect");
      if (aspect) {
        const prev = aspect.callback;
        aspect.callback = (val, ...rest) => {
          const out = prev?.call(this, val, ...rest);
          const dims = srcDims(this);
          const e = readExp(this);
          e.aspect = val;
          const ar = dims ? aspectRatio(dims, val) : null;
          writeExp(this, ar ? fixAspect(e, ar, dims.w, dims.h, "") : e);
          this.setDirtyCanvas?.(true, true);
          return out;
        };
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
      this._lcOpConfigured = true;
      this._lcOpLive = null;
      const r = onConfigure?.apply(this, arguments);

      // Older saves. Values are stored by position, so a widget added near the top shifts the rest:
      //  - 6 values, first is a number: saved before `block` and `hold_mask` existed
      //  - 7 values, first is a boolean: saved with `block` but before `hold_mask`
      // Those workflows had a canvas set up, so they load as held and behave as they did.
      const wv = data?.widgets_values;
      const setW = (name, value) => {
        const w = getW(this, name);
        if (w && value !== undefined) w.value = value;
      };
      if (Array.isArray(wv) && wv.length === 6 && typeof wv[0] === "number") {
        setW("block", false);
        setW("hold_mask", true);
        ["left", "top", "right", "bottom", "aspect", "snap_to"].forEach((name, i) => setW(name, wv[i]));
      } else if (Array.isArray(wv) && wv.length === 7 && typeof wv[0] === "boolean" && typeof wv[1] === "number") {
        setW("block", wv[0]);
        setW("hold_mask", true);
        ["left", "top", "right", "bottom", "aspect", "snap_to"].forEach((name, i) => setW(name, wv[i + 1]));
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const metas = message?.lc_preview || message?.images;
      if (metas?.length) {
        const url = viewUrl(metas[0]);
        if (url) {
          loadImg(url).then((img) => {
            this._lcOpImg = img;
            this.setDirtyCanvas?.(true, true);
          }).catch(() => {});
        }
      }
      const sz = message?.src_size?.[0];
      if (sz) {
        this._lcSrcW = sz.width || 0;
        this._lcSrcH = sz.height || 0;
      }
      // Not held and nothing pulled out: the run used the plain image, so the box is already on it
      if (message?.lc_reset?.[0]) snapBox(this);
      // Not held and edges pulled out: the box snaps back only after the whole run has finished
      this._lcResetAfter = !!message?.lc_reset_after?.[0];
      return r;
    };

    const onDrawFG = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      lcLaunchFit(this, this._lcOpConfigured);
      const r = onDrawFG?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      drawPreview(this, ctx);
      return r;
    };

    nodeType.prototype.onMouseDown = function (e, pos) {
      const cur = readExp(this);
      const layout = computeLayout(this, cur);
      if (!layout) return false;
      const hit = hitTest(layout, cur, pos[0], pos[1]);
      if (!hit) return false;
      this._lcOpFrozen = { ...layout };
      this._lcDrag = {
        mode: hit.type,
        handle: hit.id || null,
        start: { x: pos[0], y: pos[1] },
        orig: { ...cur },
        layout,
      };
      _activeNode = this;
      return true;
    };

    nodeType.prototype.onMouseMove = function (e, pos) {
      if (!this._lcDrag) return false;
      processDrag(this, pos[0], pos[1]);
      return true;
    };

    nodeType.prototype.onMouseUp = function () {
      if (this._lcDrag) {
        endDrag();
        return true;
      }
      return false;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size[0] < MIN_W) size[0] = MIN_W;
      return onResize?.apply(this, arguments);
    };
  },
});
