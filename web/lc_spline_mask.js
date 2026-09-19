/**
 * LC Create Mask 🖼️✏️ — draw a mask on the node.
 * points mode: click to add (as many as you like), drag to move, shift+click
 * removes. pencil mode: draw freehand. Points live in a hidden widget as
 * normalized 0-1 coordinates.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASS = "LCSplineMask";
const POINT_R = 5;
const HIT = 10;
const EDGE_HIT = 7;
const PENCIL_STEP = 3; // min screen px between freehand samples
const RDP_EPS = 1.2; // screen px, freehand simplification
const HISTORY_MAX = 60;
const SMOOTH_SAMPLES = 12;

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

function val(node, name, dflt) {
  const w = getW(node, name);
  return w && w.value !== undefined ? w.value : dflt;
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

// ---- points state ----

function readPts(node) {
  if (node._lcPts) return node._lcPts;
  let arr = [];
  try {
    const raw = JSON.parse(getW(node, "points")?.value || "[]");
    if (Array.isArray(raw)) {
      arr = raw
        .map((p) => (Array.isArray(p) ? [Number(p[0]), Number(p[1])] : [Number(p?.x), Number(p?.y)]))
        .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    }
  } catch (_) {}
  node._lcPts = arr;
  return arr;
}

function writePts(node, pts) {
  node._lcPts = pts;
  const w = getW(node, "points");
  if (w) {
    w.value = JSON.stringify(pts.map((p) => [Math.round(p[0] * 10000) / 10000, Math.round(p[1] * 10000) / 10000]));
  }
  node.setDirtyCanvas?.(true, true);
}

function pushHistory(node) {
  const h = (node._lcHist = node._lcHist || []);
  h.push(readPts(node).map((p) => [p[0], p[1]]));
  if (h.length > HISTORY_MAX) h.shift();
}

function undo(node) {
  const h = node._lcHist || [];
  if (!h.length) return;
  writePts(node, h.pop());
}

function clearPts(node) {
  if (!readPts(node).length) return;
  pushHistory(node);
  writePts(node, []);
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

function imageLayout(node) {
  const img = node._lcImg;
  if (!img) return null;
  const top = contentTop(node);
  const boxW = Math.max(1, node.size[0] - PAD * 2);
  const boxH = Math.max(1, node.size[1] - top - PAD);
  if (boxH < 40) return null;
  const srcW = node._lcSrcW || img.naturalWidth || 1;
  const srcH = node._lcSrcH || img.naturalHeight || 1;
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  return { dx: PAD + (boxW - dw) / 2, dy: top + (boxH - dh) / 2, dw, dh, srcW, srcH, scale };
}

const toScreen = (L, p) => [L.dx + p[0] * L.dw, L.dy + p[1] * L.dh];
const toNorm = (L, x, y) => [
  Math.max(0, Math.min(1, (x - L.dx) / L.dw)),
  Math.max(0, Math.min(1, (y - L.dy) / L.dh)),
];
const inImage = (L, x, y) => x >= L.dx && x <= L.dx + L.dw && y >= L.dy && y <= L.dy + L.dh;

// ---- geometry ----

function catmullRomClosed(pts) {
  const n = pts.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n];
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];
    for (let s = 0; s < SMOOTH_SAMPLES; s++) {
      const t = s / SMOOTH_SAMPLES;
      const t2 = t * t;
      const t3 = t2 * t;
      out.push([
        0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
      ]);
    }
  }
  return out;
}

function distToSegment(px, py, a, b) {
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const len2 = vx * vx + vy * vy;
  let t = len2 ? ((px - a[0]) * vx + (py - a[1]) * vy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a[0] + t * vx), py - (a[1] + t * vy));
}

function rdp(points, eps) {
  if (points.length < 3) return points;
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop();
    let maxD = 0;
    let idx = -1;
    for (let i = s + 1; i < e; i++) {
      const d = distToSegment(points[i][0], points[i][1], points[s], points[e]);
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (idx !== -1 && maxD > eps) {
      keep[idx] = true;
      stack.push([s, idx], [idx, e]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

// ---- drawing ----

function tracePath(ctx, sp, smooth) {
  const path = smooth && sp.length >= 3 ? catmullRomClosed(sp) : sp;
  ctx.moveTo(path[0][0], path[0][1]);
  for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
}

function drawPreview(node, ctx) {
  const L = imageLayout(node);
  if (!L) return;
  const pts = readPts(node);
  const smooth = !!val(node, "smooth", false);
  const invert = !!val(node, "invert", false);
  const feather = Number(val(node, "feather", 0)) || 0;
  const alpha = Math.max(0, Math.min(1, Number(val(node, "overlay_opacity", 0.45)))) *
    Math.max(0, Math.min(1, Number(val(node, "opacity", 1))));
  const sp = pts.map((p) => toScreen(L, p));

  ctx.save();
  ctx.drawImage(node._lcImg, L.dx, L.dy, L.dw, L.dh);

  ctx.save();
  ctx.beginPath();
  ctx.rect(L.dx, L.dy, L.dw, L.dh);
  ctx.clip();
  if (sp.length >= 3 && alpha > 0) {
    if (feather > 0) ctx.filter = `blur(${feather * L.scale}px)`;
    ctx.fillStyle = `rgba(255,40,70,${alpha})`;
    ctx.beginPath();
    if (invert) ctx.rect(L.dx - 50, L.dy - 50, L.dw + 100, L.dh + 100);
    tracePath(ctx, sp, smooth);
    ctx.closePath();
    ctx.fill(invert ? "evenodd" : "nonzero");
    ctx.filter = "none";
  } else if (invert && alpha > 0 && sp.length < 3) {
    ctx.fillStyle = `rgba(255,40,70,${alpha})`;
    ctx.fillRect(L.dx, L.dy, L.dw, L.dh);
  }
  if (sp.length >= 2) {
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    tracePath(ctx, sp, smooth);
    if (sp.length >= 3) ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();

  const dense = sp.length > 60;
  const r = dense ? 2 : POINT_R;
  sp.forEach((p, i) => {
    ctx.fillStyle = i === 0 ? "#ffd24a" : "#fff";
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fill();
    if (!dense) ctx.stroke();
  });

  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.textAlign = "left";
  ctx.fillText(`${val(node, "pencil", false) ? "PENCIL" : "POINTS"}  ${pts.length}`, L.dx + 4, L.dy + 12);
  ctx.restore();
}

// ---- interaction ----

function nearestPoint(L, pts, x, y) {
  let best = -1;
  let bestD = HIT;
  pts.forEach((p, i) => {
    const s = toScreen(L, p);
    const d = Math.hypot(x - s[0], y - s[1]);
    if (d <= bestD) { bestD = d; best = i; }
  });
  return best;
}

function nearestEdge(L, pts, x, y) {
  const n = pts.length;
  if (n < 2) return -1;
  const segs = n === 2 ? 1 : n;
  let best = -1;
  let bestD = EDGE_HIT;
  for (let i = 0; i < segs; i++) {
    const d = distToSegment(x, y, toScreen(L, pts[i]), toScreen(L, pts[(i + 1) % n]));
    if (d <= bestD) { bestD = d; best = i; }
  }
  return best;
}

function endDrag() {
  const node = _activeNode;
  if (!node) return;
  if (node._lcStroke) {
    const L = imageLayout(node);
    const { start } = node._lcStroke;
    const pts = readPts(node);
    if (L && pts.length - start > 2) {
      const stroke = pts.slice(start).map((p) => toScreen(L, p));
      const kept = rdp(stroke, RDP_EPS).map((s) => toNorm(L, s[0], s[1]));
      writePts(node, pts.slice(0, start).concat(kept));
    }
    node._lcStroke = null;
  }
  node._lcDrag = null;
  node.setDirtyCanvas?.(true, true);
  _activeNode = null;
}

function installGlobalMouseUp() {
  if (window.__lcSplineMouseUp) return;
  window.__lcSplineMouseUp = true;
  const end = () => endDrag();
  window.addEventListener("pointerup", end, true);
  window.addEventListener("mouseup", end, true);
  window.addEventListener("pointercancel", end, true);
  window.addEventListener("blur", end, true);
}

app.registerExtension({
  name: "LCMaskMaker.SplineMask",

  async setup() {
    installGlobalMouseUp();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this._lcImg = null;
      this._lcSrcW = 0;
      this._lcSrcH = 0;
      this._lcPts = null;
      this._lcHist = [];
      this._lcDrag = null;
      this._lcStroke = null;
      hideWidget(getW(this, "points"));

      this.addWidget("button", "Undo", null, () => undo(this), { serialize: false });
      this.addWidget("button", "Clear", null, () => clearPts(this), { serialize: false });

      const applySize = () => {
        if (this.size) this.size = [DEFAULT_W, defaultHeight(this)];
        this.setDirtyCanvas?.(true, true);
      };
      applySize();
      // ComfyUI's add-node search dialog re-assigns size right after this hook
      // returns; a node restored from a saved workflow (onConfigure runs in the
      // same tick) must keep its saved size.
      setTimeout(() => {
        if (!this._lcSplineConfigured) applySize();
      }, 0);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
      this._lcSplineConfigured = true;
      this._lcPts = null;
      this._lcHist = [];
      const r = onConfigure?.apply(this, arguments);

      // Saved before the `block` widget was added as the first widget: every
      // value sits one slot too early (points land in overlay_opacity, pencil
      // lands in block). Old layout is recognizable by the points string at
      // index 6, where the new layout has the overlay_opacity number.
      const wv = data?.widgets_values;
      if (Array.isArray(wv) && typeof wv[6] === "string") {
        const oldOrder = ["pencil", "smooth", "invert", "feather", "opacity", "overlay_opacity", "points"];
        const block = getW(this, "block");
        if (block) block.value = false;
        oldOrder.forEach((name, i) => {
          const w = getW(this, name);
          if (w && wv[i] !== undefined) w.value = wv[i];
        });
        this._lcPts = null;
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
            this._lcImg = img;
            this.setDirtyCanvas?.(true, true);
          }).catch(() => {});
        }
      }
      const sz = message?.src_size?.[0];
      if (sz) {
        this._lcSrcW = sz.width || 0;
        this._lcSrcH = sz.height || 0;
      }
      return r;
    };

    const onDrawFG = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      lcLaunchFit(this, this._lcSplineConfigured);
      const r = onDrawFG?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      drawPreview(this, ctx);
      return r;
    };

    nodeType.prototype.onMouseDown = function (e, pos) {
      const L = imageLayout(this);
      if (!L || !inImage(L, pos[0], pos[1])) return false;
      const pts = readPts(this);
      const idx = nearestPoint(L, pts, pos[0], pos[1]);

      if (e?.shiftKey) {
        if (idx === -1) return false;
        pushHistory(this);
        writePts(this, pts.filter((_, i) => i !== idx));
        return true;
      }

      _activeNode = this;
      if (val(this, "pencil", false)) {
        pushHistory(this);
        const start = pts.length;
        writePts(this, pts.concat([toNorm(L, pos[0], pos[1])]));
        this._lcStroke = { start, last: [pos[0], pos[1]] };
        this._lcDrag = { mode: "pencil" };
        return true;
      }

      if (idx !== -1) {
        this._lcDrag = { mode: "point", idx, pushed: false };
        return true;
      }
      const seg = nearestEdge(L, pts, pos[0], pos[1]);
      pushHistory(this);
      const np = toNorm(L, pos[0], pos[1]);
      if (seg !== -1) {
        const next = pts.slice();
        next.splice(seg + 1, 0, np);
        writePts(this, next);
        this._lcDrag = { mode: "point", idx: seg + 1, pushed: true };
      } else {
        writePts(this, pts.concat([np]));
        this._lcDrag = { mode: "point", idx: pts.length, pushed: true };
      }
      return true;
    };

    nodeType.prototype.onMouseMove = function (e, pos) {
      const drag = this._lcDrag;
      if (!drag) return false;
      const L = imageLayout(this);
      if (!L) return true;
      const pts = readPts(this);
      if (drag.mode === "point") {
        if (!drag.pushed) {
          pushHistory(this);
          drag.pushed = true;
        }
        const next = pts.slice();
        next[drag.idx] = toNorm(L, pos[0], pos[1]);
        writePts(this, next);
      } else if (drag.mode === "pencil" && this._lcStroke) {
        const last = this._lcStroke.last;
        if (Math.hypot(pos[0] - last[0], pos[1] - last[1]) >= PENCIL_STEP) {
          this._lcStroke.last = [pos[0], pos[1]];
          writePts(this, pts.concat([toNorm(L, pos[0], pos[1])]));
        }
      }
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
