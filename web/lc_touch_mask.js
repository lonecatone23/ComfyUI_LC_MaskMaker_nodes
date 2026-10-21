/**
 * LC Touch Mask 👆 — draw around a thing, SAM 3 finds its edges; tap grows, double-tap steps back.
 * The selections (lassos, taps and their growth step) live in a hidden widget; the server route
 * /lc_maskmaker/touch_mask turns them into the overlay shown on the node.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASS = "LCTouchMask";
const DRAG_MIN = 4; // screen px before a press becomes a lasso
const DOUBLE_MS = 280;
const PENCIL_STEP = 3;
const RDP_EPS = 1.5;
const HISTORY_MAX = 40;
const HINT = "draw around something";

let _activeNode = null;

function viewUrl(meta) {
  const q = new URLSearchParams();
  q.set("filename", meta.filename || "");
  q.set("subfolder", meta.subfolder || "");
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
const val = (node, name, dflt) => {
  const w = getW(node, name);
  return w && w.value !== undefined ? w.value : dflt;
};

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

// ---- selections state ----

function readSels(node) {
  try {
    const s = JSON.parse(getW(node, "selections")?.value || "[]");
    return Array.isArray(s) ? s : [];
  } catch (_) {
    return [];
  }
}

function writeSels(node, sels) {
  const w = getW(node, "selections");
  if (w) w.value = JSON.stringify(sels);
}

function pushHistory(node) {
  const h = (node._lcHist = node._lcHist || []);
  h.push(getW(node, "selections")?.value || "[]");
  if (h.length > HISTORY_MAX) h.shift();
}

function undo(node) {
  const h = node._lcHist || [];
  if (!h.length) return;
  getW(node, "selections").value = h.pop();
  refresh(node);
}

function clearAll(node) {
  if (!readSels(node).length) return;
  pushHistory(node);
  writeSels(node, []);
  refresh(node);
}

// ---- picture ----

/** A Load Image right above the node: its file, so the preview works before the first run. */
function upstreamFile(node) {
  const inp = (node.inputs || []).find((i) => i.name === "image");
  const link = inp?.link != null ? app.graph.links?.[inp.link] ?? app.graph.links?.get?.(inp.link) : null;
  const src = link ? app.graph.getNodeById(link.origin_id) : null;
  if (!src || src.mode === 2 || src.mode === 4) return null;
  const w = (src.widgets || []).find((x) => x.name === "image" && typeof x.value === "string");
  if (!w || !w.value) return null;
  let v = w.value.replace(/\s*\[(input|output|temp)\]$/, "");
  const type = (w.value.match(/\[(input|output|temp)\]$/) || [])[1] || "input";
  const i = v.lastIndexOf("/");
  return { filename: i >= 0 ? v.slice(i + 1) : v, subfolder: i >= 0 ? v.slice(0, i) : "", type };
}

function currentView(node) {
  return upstreamFile(node) || node._lcRunView || null;
}

function syncPicture(node) {
  const v = currentView(node);
  const key = v ? JSON.stringify(v) : "";
  if (key === node._lcViewKey) return false;
  node._lcViewKey = key;
  node._lcView = v;
  node._lcOverlay = null;
  if (!v) {
    node._lcImg = null;
    return true;
  }
  loadImg(viewUrl(v)).then((img) => {
    if (node._lcViewKey !== key) return;
    node._lcImg = img;
    node._lcSrcW = img.naturalWidth;
    node._lcSrcH = img.naturalHeight;
    refresh(node);
  }).catch(() => {});
  return true;
}

// ---- server ----

async function refresh(node, tap) {
  node.setDirtyCanvas?.(true, true);
  if (!node._lcView || !node._lcImg) return;
  const seq = (node._lcSeq = (node._lcSeq || 0) + 1);
  node._lcBusy = true;
  try {
    const r = await api.fetchApi("/lc_maskmaker/touch_mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: node._lcView,
        sam3_model: val(node, "sam3_model", ""),
        person_parts: val(node, "person_parts", "off"),
        threshold: Number(val(node, "threshold", 0.5)),
        smooth: Number(val(node, "smooth", 2)),
        selections: readSels(node),
        tap: tap || null,
      }),
    });
    const res = await r.json();
    if (seq !== node._lcSeq) return;
    if (res.error) {
      node._lcStatus = "⚠ " + res.error;
      node._lcOverlay = null;
      return;
    }
    writeSels(node, res.selections || []);
    node._lcStatus = res.status || "";
    node._lcOverlay = res.overlay ? await loadImg(res.overlay) : null;
  } catch (e) {
    if (seq === node._lcSeq) node._lcStatus = "⚠ preview failed";
  } finally {
    if (seq === node._lcSeq) node._lcBusy = false;
    node.setDirtyCanvas?.(true, true);
  }
}

// ---- layout ----

function widgetsBottom(node) {
  let bottom = 0;
  for (const w of node.widgets || []) {
    if (!w || w.type === "hidden" || w._lcHidden || typeof w.last_y !== "number") continue;
    const h = typeof w.computeSize === "function" ? (w.computeSize(node.size[0])?.[1] ?? 20) : 20;
    bottom = Math.max(bottom, w.last_y + Math.max(20, h));
  }
  return bottom || 200;
}

function imageLayout(node) {
  const top = widgetsBottom(node) + PAD;
  const boxW = Math.max(1, node.size[0] - PAD * 2);
  const boxH = Math.max(1, node.size[1] - top - PAD);
  if (boxH < 40) return null;
  const srcW = node._lcSrcW || 4;
  const srcH = node._lcSrcH || 5;
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  return { dx: PAD + (boxW - dw) / 2, dy: top + (boxH - dh) / 2, dw, dh, top, boxW, boxH };
}

const toNorm = (L, x, y) => [
  Math.max(0, Math.min(1, (x - L.dx) / L.dw)),
  Math.max(0, Math.min(1, (y - L.dy) / L.dh)),
];
const inImage = (L, x, y) => x >= L.dx && x <= L.dx + L.dw && y >= L.dy && y <= L.dy + L.dh;

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

function drawNode(node, ctx) {
  const L = imageLayout(node);
  if (!L) return;
  ctx.save();
  if (!node._lcImg) {
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(PAD, L.top, L.boxW, L.boxH);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Wire a Load Image, or run once", PAD + L.boxW / 2, L.top + L.boxH / 2);
    ctx.restore();
    return;
  }
  ctx.drawImage(node._lcImg, L.dx, L.dy, L.dw, L.dh);
  if (node._lcOverlay) ctx.drawImage(node._lcOverlay, L.dx, L.dy, L.dw, L.dh);

  const lasso = node._lcLasso;
  if (lasso && lasso.pts.length > 1) {
    ctx.strokeStyle = lasso.sub ? "rgba(120,200,255,0.95)" : "rgba(255,255,255,0.95)";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    lasso.pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const text = node._lcBusy ? "…" : (node._lcStatus || (readSels(node).length ? "" : HINT));
  if (text) {
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(L.dx + 3, L.dy + 3, tw + 10, 17);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, L.dx + 8, L.dy + 15);
  }
  ctx.restore();
}

// ---- interaction ----

function endPress(e) {
  const node = _activeNode;
  _activeNode = null;
  if (!node || !node._lcPress) return;
  const press = node._lcPress;
  node._lcPress = null;
  const L = imageLayout(node);
  if (!L) return;
  if (node._lcLasso) {
    const lasso = node._lcLasso;
    node._lcLasso = null;
    const pts = rdp(lasso.pts, RDP_EPS).map((s) => toNorm(L, s[0], s[1]));
    if (pts.length >= 3) {
      pushHistory(node);
      const sels = readSels(node);
      sels.push({ t: "lasso", pts: pts.map((p) => [Math.round(p[0] * 10000) / 10000, Math.round(p[1] * 10000) / 10000]),
        op: lasso.sub ? "sub" : "add", level: 0 });
      writeSels(node, sels);
      refresh(node);
    }
    node.setDirtyCanvas?.(true, true);
    return;
  }
  // a tap: wait a moment to see if it is a double-tap
  const [x, y] = toNorm(L, press.x, press.y);
  if (node._lcTapTimer) {
    clearTimeout(node._lcTapTimer);
    node._lcTapTimer = null;
    pushHistory(node);
    refresh(node, { x, y, dir: -1 });
    return;
  }
  const sub = !!press.shift;
  node._lcTapTimer = setTimeout(() => {
    node._lcTapTimer = null;
    pushHistory(node);
    refresh(node, { x, y, dir: 1, sub });
  }, DOUBLE_MS);
}

function installGlobalMouseUp() {
  if (window.__lcTouchMouseUp) return;
  window.__lcTouchMouseUp = true;
  const end = (e) => endPress(e);
  window.addEventListener("pointerup", end, true);
  window.addEventListener("mouseup", end, true);
  window.addEventListener("pointercancel", () => { if (_activeNode) { _activeNode._lcPress = null; _activeNode._lcLasso = null; _activeNode = null; } }, true);
}

app.registerExtension({
  name: "LCMaskMaker.TouchMask",

  async setup() {
    installGlobalMouseUp();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this._lcHist = [];
      hideWidget(getW(this, "selections"));
      for (const name of ["sam3_model", "person_parts", "threshold", "smooth"]) {
        const w = getW(this, name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = (...a) => {
          const out = cb?.apply(w, a);
          refresh(this);
          return out;
        };
      }
      this.addWidget("button", "Undo", null, () => undo(this), { serialize: false });
      this.addWidget("button", "Clear", null, () => clearAll(this), { serialize: false });
      if (this.size) this.size = [DEFAULT_W, Math.max(this.size[1], 360)];
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      this._lcTouchConfigured = true;
      this._lcHist = [];
      return onConfigure?.apply(this, arguments);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const meta = message?.lc_preview?.[0];
      if (meta) {
        this._lcRunView = { filename: meta.filename, subfolder: meta.subfolder || "", type: meta.type || "temp" };
        syncPicture(this);
      }
      return r;
    };

    const onDrawFG = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      lcLaunchFit(this, this._lcTouchConfigured);
      const r = onDrawFG?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      syncPicture(this);
      drawNode(this, ctx);
      return r;
    };

    nodeType.prototype.onMouseDown = function (e, pos) {
      const L = imageLayout(this);
      if (!L || !this._lcImg || !inImage(L, pos[0], pos[1])) return false;
      _activeNode = this;
      this._lcPress = { x: pos[0], y: pos[1], shift: !!e?.shiftKey, sub: !!(e?.ctrlKey || e?.metaKey) };
      return true;
    };

    nodeType.prototype.onMouseMove = function (e, pos) {
      const press = this._lcPress;
      if (!press) return false;
      const L = imageLayout(this);
      if (!L) return true;
      const x = Math.max(L.dx, Math.min(L.dx + L.dw, pos[0]));
      const y = Math.max(L.dy, Math.min(L.dy + L.dh, pos[1]));
      if (!this._lcLasso) {
        if (Math.hypot(pos[0] - press.x, pos[1] - press.y) < DRAG_MIN) return true;
        if (this._lcTapTimer) { clearTimeout(this._lcTapTimer); this._lcTapTimer = null; }
        this._lcLasso = { pts: [[press.x, press.y]], sub: press.sub };
      }
      const last = this._lcLasso.pts[this._lcLasso.pts.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) >= PENCIL_STEP) this._lcLasso.pts.push([x, y]);
      this.setDirtyCanvas?.(true, true);
      return true;
    };

    nodeType.prototype.onMouseUp = function (e) {
      if (this._lcPress) {
        endPress(e);
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
