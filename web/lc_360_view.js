/**
 * LC 360 View 🧭: aim a flat camera view inside a 360 panorama, on the node.
 * The view (top) is drawn live from the panorama the last run sent back: drag to look around, the mouse wheel (or
 * Shift + drag) zooms. The strip under it is the whole panorama with the view's spot marked; click it to jump there.
 * Real HTML (one canvas in a DOM widget), so it works on the classic canvas and in Nodes 2.0. Same projection as
 * lc_360_view.py, so the preview matches the output.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W, LC_MIN_W, LC_PAD as PAD } from "./lc_standards.js";

const NODE_CLASS = "LC360View";
const PANO_MAX = 2048; // the pixels the preview samples from
const GAP = 8;
const INNER_W = LC_W - PAD * 2; // 268: the standard preview width

const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const num = (node, name, d) => Number(wv(node, name)?.value ?? d);

function setVal(node, name, v) {
  const w = wv(node, name);
  if (!w) return;
  const o = w.options || {};
  if (typeof o.min === "number") v = Math.max(o.min, v);
  if (typeof o.max === "number") v = Math.min(o.max, v);
  w.value = Math.round(v * 2) / 2; // the widgets step in halves
  w.callback?.(w.value);
}

function viewUrl(meta) {
  const q = new URLSearchParams({ filename: meta.filename || "", subfolder: meta.subfolder ?? "", type: meta.type || "temp" });
  return api.apiURL(`/view?${q}`);
}

/** The panorama's pixels, once, for the live view. */
function loadPano(node, meta) {
  const img = new Image();
  img.onload = () => {
    const s = Math.min(1, PANO_MAX / img.width);
    const c = document.createElement("canvas");
    c.width = Math.round(img.width * s);
    c.height = Math.round(img.height * s);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0, c.width, c.height);
    node._lc360 = { img, data: g.getImageData(0, 0, c.width, c.height), key: "" };
    paint(node);
  };
  img.src = viewUrl(meta);
}

/** View and strip rectangles inside the widget's canvas (CSS pixels). */
function boxes(node) {
  const W = node._lc360El?.clientWidth || INNER_W;
  const aspect = num(node, "height", 720) / Math.max(1, num(node, "width", 1280));
  const vh = Math.round(W * Math.min(aspect, 1.25));
  return { view: { x: 0, y: 0, w: W, h: vh }, strip: { x: 0, y: vh + GAP, w: W, h: Math.round(W / 2) }, total: vh + GAP + Math.round(W / 2) };
}

/** The flat view (nearest-neighbour: it only has to aim well). */
function renderView(node, W, H) {
  const st = node._lc360;
  const yaw = num(node, "yaw", 0), pitch = num(node, "pitch", 0), fov = num(node, "fov", 70);
  const key = `${yaw}|${pitch}|${fov}|${W}|${H}`;
  if (st.key === key && st.view) return st.view;
  const src = st.data, PW = src.width, PH = src.height, sd = src.data;
  const c = st.view || document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d");
  const out = g.createImageData(W, H), od = out.data;
  const f = W / 2 / Math.tan((fov * Math.PI) / 360);
  const p = (pitch * Math.PI) / 180, t = (yaw * Math.PI) / 180;
  const cp = Math.cos(p), sp = Math.sin(p), ct = Math.cos(t), stt = Math.sin(t);
  for (let j = 0; j < H; j++) {
    const y = -(j - H / 2 + 0.5);
    for (let i = 0; i < W; i++) {
      const x = i - W / 2 + 0.5;
      const n = Math.hypot(x, y, f);
      const dx = x / n, dy = y / n, dz = f / n;
      const dy2 = dy * cp + dz * sp, dz2 = -dy * sp + dz * cp; // pitch (about x)
      const dx3 = dx * ct + dz2 * stt, dz3 = -dx * stt + dz2 * ct; // then yaw (about y)
      const lon = Math.atan2(dx3, dz3), lat = Math.asin(Math.max(-1, Math.min(1, dy2)));
      let u = Math.floor((lon / (2 * Math.PI) + 0.5) * PW);
      const v = Math.min(PH - 1, Math.max(0, Math.floor((0.5 - lat / Math.PI) * PH)));
      u = ((u % PW) + PW) % PW;
      const si = (v * PW + u) * 4, oi = (j * W + i) * 4;
      od[oi] = sd[si];
      od[oi + 1] = sd[si + 1];
      od[oi + 2] = sd[si + 2];
      od[oi + 3] = 255;
    }
  }
  g.putImageData(out, 0, 0);
  st.view = c;
  st.key = key;
  return c;
}

function paint(node) {
  const el = node._lc360El;
  if (!el) return;
  const B = boxes(node);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (el.style.height !== `${B.total}px`) {
    el.style.height = `${B.total}px`;
    node.setDirtyCanvas?.(true, true);
  }
  el.width = Math.round(B.view.w * dpr);
  el.height = Math.round(B.total * dpr);
  const ctx = el.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { view: V, strip: S } = B;
  ctx.fillStyle = "#111";
  ctx.fillRect(V.x, V.y, V.w, V.h);
  ctx.fillRect(S.x, S.y, S.w, S.h);
  if (!node._lc360) {
    ctx.fillStyle = "#888";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Run once to load the panorama,", V.w / 2, V.h / 2 - 8);
    ctx.fillText("then drag here to aim.", V.w / 2, V.h / 2 + 10);
    return;
  }
  ctx.drawImage(renderView(node, Math.max(16, Math.round(V.w * dpr)), Math.max(16, Math.round(V.h * dpr))), V.x, V.y, V.w, V.h);
  ctx.drawImage(node._lc360.img, S.x, S.y, S.w, S.h);
  const yaw = num(node, "yaw", 0), pitch = num(node, "pitch", 0), fov = num(node, "fov", 70);
  const vfov = (2 * Math.atan(Math.tan((fov * Math.PI) / 360) * (V.h / V.w)) * 180) / Math.PI;
  const cx = S.x + (yaw / 360 + 0.5) * S.w, cy = S.y + (0.5 - pitch / 180) * S.h;
  const bw = (fov / 360) * S.w, bh = (vfov / 180) * S.h;
  ctx.save();
  ctx.beginPath();
  ctx.rect(S.x, S.y, S.w, S.h);
  ctx.clip();
  ctx.strokeStyle = "#eb4";
  ctx.lineWidth = 1.5;
  for (const off of [-S.w, 0, S.w]) ctx.strokeRect(cx - bw / 2 + off, cy - bh / 2, bw, bh); // wraps around the seam
  ctx.restore();
  ctx.fillStyle = "#000a";
  ctx.fillRect(V.x, V.y + V.h - 18, V.w, 18);
  ctx.fillStyle = "#ddd";
  ctx.font = "11px Arial";
  ctx.textAlign = "left";
  ctx.fillText(`yaw ${yaw}°  pitch ${pitch}°  fov ${fov}°   drag · wheel zooms`, V.x + 5, V.y + V.h - 5);
}

function addView(node) {
  const box = wv(node, "views");
  if (!box) return;
  const lines = String(box.value || "").split("\n").filter((l) => l.trim());
  const name = window.prompt("Name this view:", `view ${lines.length + 1}`);
  if (name == null) return;
  lines.push(`${name.replace(/:/g, " ").trim() || `view ${lines.length + 1}`}: ${num(node, "yaw", 0)}, ${num(node, "pitch", 0)}, ${num(node, "fov", 70)}`);
  box.value = lines.join("\n");
  box.callback?.(box.value);
  node.setDirtyCanvas?.(true, true);
}

function buildViewer(node) {
  const el = document.createElement("canvas");
  el.style.cssText = `width:100%;height:${INNER_W}px;display:block;cursor:grab;touch-action:none;border-radius:4px`;
  node._lc360El = el;
  const dom = node.addDOMWidget("lc_360_viewer", "LC_360_VIEWER", el, {
    serialize: false,
    getMinHeight: () => boxes(node).total + 4,
    getMaxHeight: () => boxes(node).total + 4,
  });
  dom.serializeValue = () => undefined;
  let drag = null;
  const local = (e) => {
    const r = el.getBoundingClientRect();
    const k = (node._lc360El.clientWidth || r.width) / r.width; // the canvas may be scaled with the graph zoom
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
  };
  el.addEventListener("pointerdown", (e) => {
    if (!node._lc360 || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation(); // the canvas would start dragging the node
    const B = boxes(node), p = local(e);
    if (p.y >= B.strip.y) {
      setVal(node, "yaw", (p.x / B.strip.w - 0.5) * 360);
      setVal(node, "pitch", (0.5 - (p.y - B.strip.y) / B.strip.h) * 180);
      paint(node);
      return;
    }
    el.setPointerCapture(e.pointerId);
    el.style.cursor = "grabbing";
    drag = { ...p, yaw: num(node, "yaw", 0), pitch: num(node, "pitch", 0), fov: num(node, "fov", 70), w: B.view.w };
  });
  el.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.stopPropagation();
    const p = local(e), dx = p.x - drag.x, dy = p.y - drag.y;
    if (e.shiftKey) setVal(node, "fov", drag.fov * Math.exp(dy * 0.01));
    else {
      let yaw = drag.yaw - (dx * drag.fov) / drag.w; // grab-and-pull: the picture follows the pointer
      yaw = ((((yaw + 180) % 360) + 360) % 360) - 180;
      setVal(node, "yaw", yaw);
      setVal(node, "pitch", drag.pitch + (dy * drag.fov) / drag.w);
    }
    paint(node);
  });
  const end = (e) => {
    if (!drag) return;
    drag = null;
    el.style.cursor = "grab";
    try {
      el.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("wheel", (e) => {
    if (!node._lc360 || local(e).y >= boxes(node).view.h) return; // over the strip: the canvas zooms as usual
    e.preventDefault();
    e.stopPropagation();
    setVal(node, "fov", num(node, "fov", 70) * Math.exp(e.deltaY * 0.001));
    paint(node);
  }, { passive: false });
  try {
    new ResizeObserver(() => paint(node)).observe(el);
  } catch (_) {}
  // typing a value (or the aspect) redraws the view
  for (const name of ["yaw", "pitch", "fov", "width", "height"]) {
    const w = wv(node, name);
    if (!w) continue;
    const cb = w.callback;
    w.callback = function () {
      const r = cb?.apply(this, arguments);
      requestAnimationFrame(() => paint(node));
      return r;
    };
  }
  paint(node);
}

app.registerExtension({
  name: "LC.MaskMaker.360View",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      // the views box stays small (a few lines, it scrolls): left alone it grows and squeezes the viewer
      const vbox = wv(this, "views");
      if (vbox) {
        vbox.options = vbox.options || {};
        vbox.options.getMinHeight = () => 56;
        vbox.options.getMaxHeight = () => 72;
      }
      const b = this.addWidget("button", "➕ Add this view to views", null, () => addView(this));
      if (b) b.serialize = false;
      buildViewer(this);
      if (!this._lc360Configured) this.setSize([LC_W, this.computeSize()[1]]);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      this._lc360Configured = true;
      const r = onConfigure?.apply(this, arguments);
      requestAnimationFrame(() => paint(this));
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const meta = message?.lc_preview?.[0];
      if (meta) loadPano(this, meta);
      return r;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size[0] < LC_MIN_W) size[0] = LC_MIN_W;
      const r = onResize?.apply(this, arguments);
      requestAnimationFrame(() => paint(this));
      return r;
    };
  },
});
