/**
 * LC Image Rotate + Pad 🔄: turn, crop and pad on the node.
 * Round handle turns the picture (Shift = 15° steps), the frame's square handles pad (drag out) or crop (drag in),
 * dragging inside the frame moves the picture in it. Same layout helpers as LC Outpaint; the frame maths mirrors
 * lc_rotate_pad.py (base_box / frame_rect), so the preview matches the output.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASS = "LCImageRotatePad";
const HANDLE = 12;
const HIT = 16;
const ROT_R = 7;
const ROT_OFF = 26;
const MIN_PCT = -100;
const MAX_PCT = 400;
const MIN_SIDE = 16;
const HEADROOM = 0.1;
const FILL_COLOR = { gray: "#808080", white: "#ffffff", black: "#000000", edge: "#808080" };
const ASPECT_MAP = { "1:1": 1, "4:3": 4 / 3, "3:2": 3 / 2, "16:9": 16 / 9, "3:4": 3 / 4, "2:3": 2 / 3, "9:16": 9 / 16 };
const SIDES = ["left", "top", "right", "bottom"];
const HANDLES = [
  { id: "nw", x: 0, y: 0 }, { id: "n", x: 0.5, y: 0 }, { id: "ne", x: 1, y: 0 },
  { id: "e", x: 1, y: 0.5 }, { id: "se", x: 1, y: 1 }, { id: "s", x: 0.5, y: 1 },
  { id: "sw", x: 0, y: 1 }, { id: "w", x: 0, y: 0.5 },
];

let _activeNode = null;

function viewUrl(meta) {
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

const getW = (node, name) => (node.widgets || []).find((x) => x && x.name === name);

function hideWidget(w) {
  if (!w || w._lcHidden) return;
  w._lcHidden = true;
  w.computeSize = () => [0, -4];
  w.draw = () => {};
  w.type = "hidden";
}

// ---- state ----

function readState(node) {
  if (node._lcRpLive) return { ...node._lcRpLive };
  const num = (n) => {
    const v = Number(getW(node, n)?.value);
    return Number.isFinite(v) ? v : 0;
  };
  return {
    angle: num("angle"),
    l: num("left"), t: num("top"), r: num("right"), b: num("bottom"),
    fit: getW(node, "fit")?.value || "expand",
    aspect: getW(node, "aspect")?.value || "free",
  };
}

function writeState(node, s) {
  node._lcRpLive = { ...s };
  const round = (v) => Math.round(v * 100) / 100;
  for (const [name, val] of [["left", s.l], ["top", s.t], ["right", s.r], ["bottom", s.b]]) {
    const w = getW(node, name);
    if (w) w.value = round(val);
  }
  const a = getW(node, "angle");
  if (a) a.value = Math.round(s.angle * 10) / 10;
}

// ---- frame maths (mirror of lc_rotate_pad.py) ----

function baseBox(w, h, angle, fit) {
  const a = (angle * Math.PI) / 180;
  const c = Math.abs(Math.cos(a));
  const s = Math.abs(Math.sin(a));
  const bw = w * c + h * s;
  const bh = w * s + h * c;
  if (fit === "crop") {
    const k = Math.min(w / bw, h / bh);
    const m = Math.abs(angle / 90 - Math.round(angle / 90)) < 1e-9 ? 0 : 2;
    return { w: Math.max(1, w * k - m), h: Math.max(1, h * k - m) };
  }
  return { w: bw, h: bh };
}

function frameRect(w, h, s) {
  const base = baseBox(w, h, s.angle, s.fit);
  let x0 = -base.w / 2 - (s.l / 100) * w;
  let x1 = base.w / 2 + (s.r / 100) * w;
  let y0 = -base.h / 2 - (s.t / 100) * h;
  let y1 = base.h / 2 + (s.b / 100) * h;
  if (x1 - x0 < MIN_SIDE) { const m = (x0 + x1) / 2; x0 = m - MIN_SIDE / 2; x1 = m + MIN_SIDE / 2; }
  if (y1 - y0 < MIN_SIDE) { const m = (y0 + y1) / 2; y0 = m - MIN_SIDE / 2; y1 = m + MIN_SIDE / 2; }
  return { x0, y0, x1, y1 };
}

function clampState(s, w, h) {
  for (const k of ["l", "t", "r", "b"]) s[k] = Math.max(MIN_PCT, Math.min(MAX_PCT, s[k]));
  // keep the frame at least MIN_SIDE: pull back the side being dragged
  const base = baseBox(w, h, s.angle, s.fit);
  const fw = base.w + ((s.l + s.r) / 100) * w;
  const fh = base.h + ((s.t + s.b) / 100) * h;
  if (fw < MIN_SIDE) { const d = ((MIN_SIDE - fw) / w) * 100; s.l += d / 2; s.r += d / 2; }
  if (fh < MIN_SIDE) { const d = ((MIN_SIDE - fh) / h) * 100; s.t += d / 2; s.b += d / 2; }
  return s;
}

function aspectRatio(w, h, key) {
  if (!key || key === "free") return null;
  if (key === "original") return w / h;
  return ASPECT_MAP[key] ?? null;
}

// Resize the frame to `ar`, moving the dragged side(s) (`anchor` = handle id, "" = both sides evenly).
function fixAspect(s, ar, w, h, anchor) {
  const base = baseBox(w, h, s.angle, s.fit);
  let W = base.w + ((s.l + s.r) / 100) * w;
  let H = base.h + ((s.t + s.b) / 100) * h;
  const horiz = /[ew]/.test(anchor);
  const vert = /[ns]/.test(anchor);
  if (horiz && !vert) H = W / ar;
  else if (vert && !horiz) W = H * ar;
  else if (W / H > ar) H = W / ar;
  else W = H * ar;
  const spread = (total, lo, hi, size, anchorLo, anchorHi) => {
    const loPx = (lo / 100) * size;
    const hiPx = (hi / 100) * size;
    const d = total - (loPx + hiPx);
    if (anchorLo && !anchorHi) return [((loPx + d) / size) * 100, hi];
    if (anchorHi && !anchorLo) return [lo, ((hiPx + d) / size) * 100];
    return [((loPx + d / 2) / size) * 100, ((hiPx + d / 2) / size) * 100];
  };
  [s.l, s.r] = spread(W - base.w, s.l, s.r, w, anchor.includes("w"), anchor.includes("e"));
  [s.t, s.b] = spread(H - base.h, s.t, s.b, h, anchor.includes("n"), anchor.includes("s"));
  return clampState(s, w, h);
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

const contentTop = (node) => widgetsBottom(node) + PAD;
const defaultHeight = (node) => contentTop(node) + lcPreviewHeight() + PAD;

function srcDims(node) {
  const w = node._lcSrcW || node._lcRpImg?.naturalWidth || 0;
  const h = node._lcSrcH || node._lcRpImg?.naturalHeight || 0;
  return w > 0 && h > 0 ? { w, h } : null;
}

// Maps picture pixels (around its centre) to node coordinates: screen = origin + p * scale.
function computeLayout(node, s) {
  if (node._lcRpFrozen) return { ...node._lcRpFrozen };
  const dims = srcDims(node);
  const top = contentTop(node);
  const box = { x: PAD, y: top, w: Math.max(1, node.size[0] - PAD * 2), h: Math.max(1, node.size[1] - top - PAD) };
  if (!dims || !node._lcRpImg || box.h < 40) return dims ? null : { box, empty: true };
  const turned = baseBox(dims.w, dims.h, s.angle, "expand");
  const f = frameRect(dims.w, dims.h, s);
  const ux0 = Math.min(-turned.w / 2, f.x0);
  const ux1 = Math.max(turned.w / 2, f.x1);
  const uy0 = Math.min(-turned.h / 2, f.y0);
  const uy1 = Math.max(turned.h / 2, f.y1);
  const room = ROT_OFF + ROT_R + 4; // the turn handle sits above the frame
  const scale = Math.min(
    (box.w * (1 - 2 * HEADROOM)) / (ux1 - ux0),
    (box.h * (1 - 2 * HEADROOM) - room) / (uy1 - uy0),
  );
  return {
    box,
    scale,
    srcW: dims.w,
    srcH: dims.h,
    ox: box.x + box.w / 2 - ((ux0 + ux1) / 2) * scale,
    oy: box.y + room / 2 + box.h / 2 - ((uy0 + uy1) / 2) * scale,
  };
}

function screenFrame(L, s) {
  const f = frameRect(L.srcW, L.srcH, s);
  return { x: L.ox + f.x0 * L.scale, y: L.oy + f.y0 * L.scale, w: (f.x1 - f.x0) * L.scale, h: (f.y1 - f.y0) * L.scale };
}

function rotHandle(fr) {
  return { x: fr.x + fr.w / 2, y: fr.y - ROT_OFF };
}

function hitTest(L, s, px, py) {
  const fr = screenFrame(L, s);
  const rh = rotHandle(fr);
  if (Math.hypot(px - rh.x, py - rh.y) <= ROT_R + 5) return { type: "rotate" };
  for (const h of HANDLES) {
    if (Math.abs(px - (fr.x + h.x * fr.w)) <= HIT && Math.abs(py - (fr.y + h.y * fr.h)) <= HIT) {
      return { type: "handle", id: h.id };
    }
  }
  if (px >= fr.x && px <= fr.x + fr.w && py >= fr.y && py <= fr.y + fr.h) return { type: "move" };
  return null;
}

// ---- drawing ----

function drawPreview(node, ctx) {
  const s = readState(node);
  const L = computeLayout(node, s);
  if (!L) return;
  if (L.empty) {
    ctx.save();
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.textAlign = "center";
    ctx.fillText("Run once to see the picture", L.box.x + L.box.w / 2, L.box.y + L.box.h / 2);
    ctx.restore();
    return;
  }
  const fr = screenFrame(L, s);
  const { box } = L;
  const fill = getW(node, "fill")?.value || "gray";

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x, box.y, box.w, box.h);
  ctx.clip();

  ctx.fillStyle = FILL_COLOR[fill] || "#808080";
  ctx.fillRect(fr.x, fr.y, fr.w, fr.h);

  ctx.save();
  ctx.translate(L.ox, L.oy);
  ctx.rotate((s.angle * Math.PI) / 180);
  ctx.drawImage(node._lcRpImg, (-L.srcW / 2) * L.scale, (-L.srcH / 2) * L.scale, L.srcW * L.scale, L.srcH * L.scale);
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1;
  ctx.strokeRect((-L.srcW / 2) * L.scale, (-L.srcH / 2) * L.scale, L.srcW * L.scale, L.srcH * L.scale);
  ctx.setLineDash([]);
  ctx.restore();

  // dim what the frame crops away
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.rect(box.x, box.y, box.w, box.h);
  ctx.rect(fr.x, fr.y, fr.w, fr.h);
  ctx.fill("evenodd");

  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(fr.x + 0.5, fr.y + 0.5, Math.max(0, fr.w - 1), Math.max(0, fr.h - 1));
  for (const h of HANDLES) {
    const hx = fr.x + h.x * fr.w;
    const hy = fr.y + h.y * fr.h;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
    ctx.fill();
    ctx.stroke();
  }

  const rh = rotHandle(fr);
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rh.x, fr.y);
  ctx.lineTo(rh.x, rh.y + ROT_R);
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#111";
  ctx.beginPath();
  ctx.arc(rh.x, rh.y, ROT_R, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  const f = frameRect(L.srcW, L.srcH, s);
  const outW = Math.round(f.x1 - f.x0);
  const outH = Math.round(f.y1 - f.y0);
  ctx.save();
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.textAlign = "center";
  const turn = Math.abs(s.angle) > 0.001 ? `  ↻ ${(Math.round(s.angle * 10) / 10).toFixed(1)}°` : "";
  ctx.fillText(`${L.srcW}×${L.srcH}  →  ${outW}×${outH}${turn}`, box.x + box.w / 2, box.y + box.h - 2);
  ctx.restore();
}

// ---- interaction ----

function processDrag(node, px, py, shift) {
  const drag = node._lcDrag;
  if (!drag) return;
  const { layout: L, orig: o } = drag;
  let s = { ...o };
  if (drag.mode === "rotate") {
    const a0 = Math.atan2(drag.start.y - L.oy, drag.start.x - L.ox);
    const a1 = Math.atan2(py - L.oy, px - L.ox);
    let ang = o.angle + ((a1 - a0) * 180) / Math.PI;
    ang = ((((ang + 180) % 360) + 360) % 360) - 180;
    s.angle = shift ? Math.round(ang / 15) * 15 : Math.round(ang * 10) / 10;
    if (s.angle === -180) s.angle = 180;
  } else {
    const dx = ((px - drag.start.x) / L.scale / L.srcW) * 100;
    const dy = ((py - drag.start.y) / L.scale / L.srcH) * 100;
    if (drag.mode === "move") {
      // the picture follows the pointer inside the frame
      s.l = o.l + dx; s.r = o.r - dx;
      s.t = o.t + dy; s.b = o.b - dy;
    } else {
      const id = drag.handle;
      if (id.includes("w")) s.l = o.l - dx;
      if (id.includes("e")) s.r = o.r + dx;
      if (id.includes("n")) s.t = o.t - dy;
      if (id.includes("s")) s.b = o.b + dy;
    }
    s = clampState(s, L.srcW, L.srcH);
    const ar = aspectRatio(L.srcW, L.srcH, o.aspect);
    if (ar && drag.mode !== "move") s = fixAspect(s, ar, L.srcW, L.srcH, drag.handle);
  }
  writeState(node, s);
  node.setDirtyCanvas?.(true, true);
}

function endDrag() {
  if (!_activeNode) return;
  _activeNode._lcDrag = null;
  _activeNode._lcRpFrozen = null;
  _activeNode.setDirtyCanvas?.(true, true);
  _activeNode = null;
}

function resetFrame(node) {
  const s = readState(node);
  writeState(node, { ...s, angle: 0, l: 0, t: 0, r: 0, b: 0 });
  node.setDirtyCanvas?.(true, true);
}

// snap back after run: reset once the run ends, finished or cancelled (this node already ran);
// a run that errors keeps the frame so it can be fixed and run again
function installRunListeners() {
  if (window.__lcRotatePadRun) return;
  window.__lcRotatePadRun = true;
  const nodes = () => (app.graph?._nodes || []).filter((n) => n?.comfyClass === NODE_CLASS || n?.type === NODE_CLASS);
  const ended = () => {
    for (const n of nodes()) {
      if (n._lcResetAfter) resetFrame(n);
      n._lcResetAfter = false;
    }
  };
  api.addEventListener("execution_success", ended);
  api.addEventListener("execution_interrupted", ended);
  api.addEventListener("execution_error", () => { for (const n of nodes()) n._lcResetAfter = false; });
}

function installGlobalMouseUp() {
  if (window.__lcRotatePadMouseUp) return;
  window.__lcRotatePadMouseUp = true;
  for (const ev of ["pointerup", "mouseup", "pointercancel", "blur"]) window.addEventListener(ev, endDrag, true);
}

app.registerExtension({
  name: "LCMaskMaker.ImageRotatePad",

  async setup() {
    installGlobalMouseUp();
    installRunListeners();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this._lcRpImg = null;
      this._lcSrcW = 0;
      this._lcSrcH = 0;
      this._lcDrag = null;
      this._lcRpFrozen = null;
      this._lcRpLive = null;
      for (const name of SIDES) hideWidget(getW(this, name));

      const applySize = () => {
        if (this.size) this.size = [DEFAULT_W, defaultHeight(this)];
        this.setDirtyCanvas?.(true, true);
      };
      applySize();
      setTimeout(() => {
        if (!this._lcRpConfigured) applySize();
      }, 0);

      // typed values: drop the live copy so the preview reads the widgets again
      for (const name of ["angle", "fit", "fill"]) {
        const w = getW(this, name);
        if (!w) continue;
        const prev = w.callback;
        w.callback = (val, ...rest) => {
          const out = prev?.call(this, val, ...rest);
          this._lcRpLive = null;
          this.setDirtyCanvas?.(true, true);
          return out;
        };
      }
      const aspect = getW(this, "aspect");
      if (aspect) {
        const prev = aspect.callback;
        aspect.callback = (val, ...rest) => {
          const out = prev?.call(this, val, ...rest);
          const dims = srcDims(this);
          const s = readState(this);
          s.aspect = val;
          const ar = dims ? aspectRatio(dims.w, dims.h, val) : null;
          writeState(this, ar ? fixAspect(s, ar, dims.w, dims.h, "") : s);
          this.setDirtyCanvas?.(true, true);
          return out;
        };
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      this._lcRpConfigured = true;
      this._lcRpLive = null;
      return onConfigure?.apply(this, arguments);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const metas = message?.lc_preview || message?.images;
      if (metas?.length) {
        loadImg(viewUrl(metas[0])).then((img) => {
          this._lcRpImg = img;
          this.setDirtyCanvas?.(true, true);
        }).catch(() => {});
      }
      const sz = message?.src_size?.[0];
      if (sz) {
        this._lcSrcW = sz.width || 0;
        this._lcSrcH = sz.height || 0;
      }
      this._lcResetAfter = !!message?.lc_reset_after?.[0];
      return r;
    };

    const onDrawFG = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      lcLaunchFit(this, this._lcRpConfigured);
      const r = onDrawFG?.apply(this, arguments);
      if (!this.flags?.collapsed) drawPreview(this, ctx);
      return r;
    };

    nodeType.prototype.onMouseDown = function (e, pos) {
      const cur = readState(this);
      const L = computeLayout(this, cur);
      if (!L || L.empty) return false;
      const hit = hitTest(L, cur, pos[0], pos[1]);
      if (!hit) return false;
      this._lcRpFrozen = { ...L };
      this._lcDrag = { mode: hit.type, handle: hit.id || "", start: { x: pos[0], y: pos[1] }, orig: { ...cur }, layout: L };
      _activeNode = this;
      return true;
    };

    nodeType.prototype.onMouseMove = function (e, pos) {
      if (!this._lcDrag) return false;
      processDrag(this, pos[0], pos[1], !!e?.shiftKey);
      return true;
    };

    nodeType.prototype.onMouseUp = function () {
      if (!this._lcDrag) return false;
      endDrag();
      return true;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size[0] < MIN_W) size[0] = MIN_W;
      return onResize?.apply(this, arguments);
    };
  },
});
