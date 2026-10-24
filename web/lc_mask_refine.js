/**
 * On-node before/after preview for the mask, detailer and AnyPaint nodes.
 * Same behavior as the LC123 image previews: the after picture fitted to the node. Click a spot: it zooms to 100% (one
 * picture pixel per screen pixel) there and the pointer wipes before (left) | after (right). The wheel zooms in and out,
 * double-click goes back. It also goes back when the canvas is zoomed out too far or the node leaves the screen.
 *
 * The node sends up to four small images in message.lc_refine: source, before mask, after mask, trimap.
 * The inpaint nodes send message.lc_inpaint instead: before (areas outlined) and after, plus a one-line note.
 * preview_view: "cutout" puts the subject on a checkerboard, "mask" shows black/white,
 * "trimap" shows the trimap as the before side.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { lcApplyLaunchColor } from "./lc_color.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASSES = new Set([
  "LCMaskRefine", "LCSegmentAnything",
  "LCSmartInpaint", "LCSmartInpaintPipe", "LCCharacterDetailer", "LCCharacterDetailerPipe", "LCDetailerForEach", "LCKrea2AnyPaint", "LCZImageAnyPaint", "LCSDXLAnyPaint", "LCFlux3AnyPaint", "LCMOSSAnyPaint",
]);
// Multiline text widgets that must stay a fixed height (otherwise they grow to fill the node)
const FIXED_HEIGHT = { prompt: 64, negative: 48 };


function imageUrl(meta) {
  if (!meta) return null;
  const rand = typeof app.getRandParam === "function" ? app.getRandParam() : "";
  return api.apiURL(
    `/view?filename=${encodeURIComponent(meta.filename)}` +
      `&type=${meta.type || "temp"}` +
      `&subfolder=${encodeURIComponent(meta.subfolder || "")}${rand}`
  );
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

// ---- cutouts -------------------------------------------------------------

function toCanvas(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  c.getContext("2d").drawImage(img, 0, 0);
  return c;
}

function makeCutout(srcCanvas, maskImg) {
  const w = srcCanvas.width;
  const h = srcCanvas.height;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const octx = out.getContext("2d");
  octx.drawImage(srcCanvas, 0, 0);
  const mc = document.createElement("canvas");
  mc.width = w;
  mc.height = h;
  const mctx = mc.getContext("2d");
  mctx.drawImage(maskImg, 0, 0, w, h);
  const px = octx.getImageData(0, 0, w, h);
  const mp = mctx.getImageData(0, 0, w, h).data;
  for (let i = 0; i < px.data.length; i += 4) px.data[i + 3] = mp[i];
  octx.putImageData(px, 0, 0);
  return out;
}

let _checker = null;
function checkerPattern(ctx) {
  if (!_checker) {
    const c = document.createElement("canvas");
    c.width = 16;
    c.height = 16;
    const g = c.getContext("2d");
    g.fillStyle = "#555";
    g.fillRect(0, 0, 16, 16);
    g.fillStyle = "#777";
    g.fillRect(0, 0, 8, 8);
    g.fillRect(8, 8, 8, 8);
    _checker = c;
  }
  return ctx.createPattern(_checker, "repeat");
}

// ---- 100% zoom ----------------------------------------------------------

const ZOOMED = new Set();
let zoomTimer = null;
let wheelHooked = false;

function tooSmall(w, h) {
  const ds = app.canvas?.ds?.scale || 1;
  if (window.LC123Perf?.tooSmall) return window.LC123Perf.tooSmall(w, h); // LC123's "minimum preview size"
  return Math.sqrt(w * h) * ds < 40;
}

// while a preview is zoomed: end it once its node is off screen, and let the wheel magnify it
function watchZoomed() {
  if (!wheelHooked) {
    wheelHooked = true;
    window.addEventListener("wheel", (e) => {
      if (!ZOOMED.size || !app.canvas) return;
      const g = app.canvas.convertEventToCanvasOffset?.(e);
      if (!g) return;
      for (const c of ZOOMED) {
        if (c.wheel([g[0] - c.node.pos[0], g[1] - c.node.pos[1]], e.deltaY)) {
          e.preventDefault();
          e.stopImmediatePropagation();
          return;
        }
      }
    }, { capture: true, passive: false });
  }
  if (zoomTimer) return;
  zoomTimer = setInterval(() => {
    const va = app.canvas?.visible_area;
    for (const c of ZOOMED) {
      const n = c.node;
      if (!c.zoom || !app.graph?.getNodeById?.(n.id)) ZOOMED.delete(c);
      else if (va && va[2] > 0 && va[3] > 0 &&
               (n.pos[0] > va[0] + va[2] || n.pos[0] + n.size[0] < va[0] || n.pos[1] > va[1] + va[3] || n.pos[1] + n.size[1] < va[1])) {
        c.unzoom();
      }
    }
    if (!ZOOMED.size) {
      clearInterval(zoomTimer);
      zoomTimer = null;
    }
  }, 500);
}

class LCMaskPreview {
  constructor(node) {
    this.node = node;
    this.views = null;
    this.zoom = null; // {u, v, mag}: zoom centre as a fraction of the picture, magnification (1 = 100%)
    this.pointerOver = false;
    this.pointerPos = [0, 0];
    this._bind(node);
    queueMicrotask(() => this._restoreFromProps());
  }

  /** The preview box, the fitted picture in it, and (zoomed) the part of the picture shown and where. */
  _layout() {
    const v = this.views;
    const node = this.node;
    let top = widgetsBottom(node) + PAD;
    const x = PAD;
    const w = Math.max(1, node.size[0] - PAD * 2);
    let h = Math.max(1, node.size[1] - top - PAD);
    if (v?.note !== undefined) {
      top += 18;
      h -= 18;
    }
    if (!v?.w || h < 16) return null;
    const s = Math.min(w / v.w, h / v.h);
    const fit = { x: x + (w - v.w * s) / 2, y: top + (h - v.h * s) / 2, w: v.w * s, h: v.h * s };
    const out = { box: { x, y: top, w, h }, fit };
    if (this.zoom) {
      const ds = app.canvas?.ds?.scale || 1;
      const mag = this.zoom.mag || 1;
      const sw = Math.min(v.w, (w * ds) / mag), sh = Math.min(v.h, (h * ds) / mag);
      const dw = (sw * mag) / ds, dh = (sh * mag) / ds;
      const dx = x + (w - dw) / 2, dy = top + (h - dh) / 2;
      const u0 = Math.max(0, Math.min(v.w - sw, this.zoom.u * v.w - sw / 2));
      const v0 = Math.max(0, Math.min(v.h - sh, this.zoom.v * v.h - sh / 2));
      const k = dw / sw; // node units per picture pixel
      out.view = { dx, dy, dw, dh, sw, sh, u0, v0, full: { x: dx - u0 * k, y: dy - v0 * k, w: v.w * k, h: v.h * k } };
    }
    return out;
  }

  unzoom() {
    this.zoom = null;
    ZOOMED.delete(this);
    app.canvas?.setDirty?.(true, true);
  }

  wheel(pos, delta) {
    const L = this.zoom && this._layout();
    const z = L?.view;
    if (!z || pos[0] < z.dx || pos[0] > z.dx + z.dw || pos[1] < z.dy || pos[1] > z.dy + z.dh) return false;
    const v = this.views;
    const fx = (pos[0] - z.dx) / z.dw, fy = (pos[1] - z.dy) / z.dh;
    const pu = (z.u0 + fx * z.sw) / v.w, pv = (z.v0 + fy * z.sh) / v.h; // the picture point under the pointer
    const old = this.zoom.mag || 1;
    const step = Math.pow(1.25, Math.max(1, Math.min(4, Math.abs(delta) / 100)));
    const mag = Math.min(16, old * (delta > 0 ? 1 / step : step));
    const fitMag = (app.canvas?.ds?.scale || 1) * Math.min(L.box.w / v.w, L.box.h / v.h);
    if (delta > 0 && mag <= fitMag * 1.05) {
      this.unzoom(); // out past the whole picture: the fitted view again
      return true;
    }
    const vw = (z.sw * old) / mag / v.w, vh = (z.sh * old) / mag / v.h;
    this.zoom = { u: pu - fx * vw + vw / 2, v: pv - fy * vh + vh / 2, mag };
    app.canvas?.setDirty?.(true, true);
    return true;
  }

  _bind(node) {
    const self = this;

    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
      this._lcMaskConfigured = true;
      const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      queueMicrotask(() => self._restoreFromProps());
      return r;
    };

    const origDrawFG = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      lcLaunchFit(this, this._lcMaskConfigured);
      if (origDrawFG) origDrawFG.apply(this, arguments);
      if (this.flags?.collapsed) return;
      self.draw(ctx);
    };

    const origMouseMove = node.onMouseMove;
    node.onMouseMove = function (e, pos) {
      self.pointerPos = pos;
      if (self.zoom && self.pointerOver && self.views) app.canvas?.setDirty?.(true, true);
      return origMouseMove ? origMouseMove.apply(this, arguments) : false;
    };

    // the usage tooltip (lc_preview_tip.js) shows over the fitted picture
    node._lcPreviewHit = (pos) => {
      if (self.zoom) return false;
      const f = self._layout()?.fit;
      return !!f && pos[0] >= f.x && pos[0] <= f.x + f.w && pos[1] >= f.y && pos[1] <= f.y + f.h;
    };
    // click on the picture: 100% there (or the on-screen size when the canvas already shows it bigger);
    // double-click: back to the fitted picture
    const origMouseDown = node.onMouseDown;
    node.onMouseDown = function (e, pos) {
      if (!self.zoom && (e.button === 0 || e.button === undefined)) {
        const f = self._layout()?.fit;
        if (f && pos[0] >= f.x && pos[0] <= f.x + f.w && pos[1] >= f.y && pos[1] <= f.y + f.h) {
          const fit = ((app.canvas?.ds?.scale || 1) * f.w) / self.views.w; // screen pixels per picture pixel now
          self.zoom = { u: (pos[0] - f.x) / f.w, v: (pos[1] - f.y) / f.h, mag: Math.max(1, fit) };
          ZOOMED.add(self);
          watchZoomed();
          app.canvas?.setDirty?.(true, true);
          return true;
        }
      }
      return origMouseDown ? origMouseDown.apply(this, arguments) : false;
    };

    const origDblClick = node.onDblClick;
    node.onDblClick = function (e, pos) {
      if (self.zoom) {
        self.unzoom();
        return true;
      }
      return origDblClick ? origDblClick.apply(this, arguments) : false;
    };

    const origEnter = node.onMouseEnter;
    node.onMouseEnter = function () {
      self.pointerOver = true;
      app.canvas?.setDirty?.(true, true);
      if (origEnter) origEnter.apply(this, arguments);
    };

    const origLeave = node.onMouseLeave;
    node.onMouseLeave = function () {
      self.pointerOver = false;
      app.canvas?.setDirty?.(true, true);
      if (origLeave) origLeave.apply(this, arguments);
    };

    const origExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (origExecuted) origExecuted.apply(this, arguments);
      self.onExecuted(message);
    };

    const origResize = node.onResize;
    node.onResize = function (size) {
      if (size[0] < MIN_W) size[0] = MIN_W;
      if (origResize) origResize.apply(this, arguments);
    };

    const view = getW(node, "preview_view");
    if (view) {
      const prev = view.callback;
      view.callback = function () {
        const out = prev?.apply(this, arguments);
        app.canvas?.setDirty?.(true, true);
        return out;
      };
    }
  }

  async _load(metas) {
    if (!metas || metas.length < 3) return;
    try {
      const imgs = await Promise.all(metas.slice(0, 4).map((m) => loadImg(imageUrl(m))));
      const [src, before, after, trimap] = imgs;
      const sc = toCanvas(src);
      this.views = {
        w: src.naturalWidth,
        h: src.naturalHeight,
        cutBefore: makeCutout(sc, before),
        cutAfter: makeCutout(sc, after),
        maskBefore: before,
        maskAfter: after,
        trimap: trimap || null,
      };
      app.canvas?.setDirty?.(true, true);
    } catch (_) {}
  }

  async _loadPair(metas, note) {
    if (!metas || metas.length < 2) {
      this.views = note ? { note } : null;
      app.canvas?.setDirty?.(true, true);
      return;
    }
    try {
      const [before, after] = await Promise.all(metas.slice(0, 2).map((m) => loadImg(imageUrl(m))));
      this.views = { w: before.naturalWidth, h: before.naturalHeight, imgBefore: before, imgAfter: after, note };
      app.canvas?.setDirty?.(true, true);
    } catch (_) {}
  }

  // Keep the preview info on the node so undo / reload can bring the images back
  _restoreFromProps() {
    const p = this.node.properties || {};
    if (p.lc_inpaint_meta?.length || p.lc_inpaint_note) this._loadPair(p.lc_inpaint_meta, p.lc_inpaint_note);
    else if (p.lc_refine_meta?.length) this._load(p.lc_refine_meta);
  }

  onExecuted(message) {
    if (!this.node.properties) this.node.properties = {};
    if (message?.lc_inpaint_note) {
      const note = message.lc_inpaint_note[0] || "";
      this.node.properties.lc_inpaint_meta = message.lc_inpaint || [];
      this.node.properties.lc_inpaint_note = note;
      this._loadPair(message.lc_inpaint, note);
      return;
    }
    const metas = message?.lc_refine;
    if (!metas?.length) return;
    this.node.properties.lc_refine_meta = metas;
    this._load(metas);
  }

  _paintSide(ctx, which, r, view) {
    const v = this.views;
    if (v.imgBefore) {
      ctx.drawImage(which === "before" ? v.imgBefore : v.imgAfter, r.x, r.y, r.w, r.h);
      return;
    }
    if (view === "trimap" && which === "before" && v.trimap) {
      ctx.drawImage(v.trimap, r.x, r.y, r.w, r.h);
    } else if (view === "mask") {
      ctx.drawImage(which === "before" ? v.maskBefore : v.maskAfter, r.x, r.y, r.w, r.h);
    } else if (view === "trimap") {
      ctx.drawImage(v.maskAfter, r.x, r.y, r.w, r.h);
    } else {
      ctx.fillStyle = checkerPattern(ctx);
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.drawImage(which === "before" ? v.cutBefore : v.cutAfter, r.x, r.y, r.w, r.h);
    }
  }

  draw(ctx) {
    const v = this.views;
    if (!v) return;
    const node = this.node;
    let top = widgetsBottom(node) + PAD;
    const x = PAD;
    const w = Math.max(1, node.size[0] - PAD * 2);
    let h = Math.max(1, node.size[1] - top - PAD);
    if (h < 16) return;
    if (v.note !== undefined) {
      ctx.save();
      ctx.font = "12px sans-serif";
      ctx.fillStyle = /^Nothing|empty/.test(v.note) ? "#f0b35a" : "#9fd8c8";
      ctx.textBaseline = "top";
      ctx.fillText(v.note, x, top, w);
      ctx.restore();
      if (!v.imgBefore) return;
      top += 18;
      h -= 18;
      if (h < 16) return;
    }

    if (this.zoom && tooSmall(w, h)) this.unzoom(); // canvas zoomed out to where the preview is no longer drawn
    const L = this._layout();
    if (!L) return;
    const view = getW(node, "preview_view")?.value || "cutout";

    if (!L.view) {
      // the after picture, fitted
      ctx.save();
      ctx.beginPath();
      ctx.rect(L.fit.x, L.fit.y, L.fit.w, L.fit.h);
      ctx.clip();
      this._paintSide(ctx, "after", L.fit, view);
      ctx.restore();
      return;
    }

    // 100% (times the wheel's magnification): after underneath, before left of the pointer
    const z = L.view;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.beginPath();
    ctx.rect(z.dx, z.dy, z.dw, z.dh);
    ctx.clip();
    this._paintSide(ctx, "after", z.full, view);
    ctx.restore();
    const splitX = this.pointerOver ? Math.max(0, Math.min(z.dw, this.pointerPos[0] - z.dx)) : 0;
    if (splitX > 0) {
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.beginPath();
      ctx.rect(z.dx, z.dy, splitX, z.dh);
      ctx.clip();
      this._paintSide(ctx, "before", z.full, view);
      ctx.restore();
    }
    if (this.pointerOver) {
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(z.dx + splitX, z.dy);
      ctx.lineTo(z.dx + splitX, z.dy + z.dh);
      ctx.stroke();
    }
    ctx.save();
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    const label = `${Math.round((this.zoom.mag || 1) * 100)}% · before | after · double-click: back`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(z.dx + z.dw - tw - 10, z.dy + z.dh - 18, tw + 10, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, z.dx + z.dw - 5, z.dy + z.dh - 4);
    ctx.restore();
  }
}

function launchHeight(node) {
  const natural = node.computeSize?.([DEFAULT_W, 0])?.[1] ?? widgetsBottom(node);
  return natural + PAD + lcPreviewHeight() + PAD;
}

app.registerExtension({
  name: "LCMaskMaker.MaskPreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name || "";
    if (!NODE_CLASSES.has(name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      lcApplyLaunchColor(this, "#324B4B");

      // Fixed-height text boxes (same trick as LC123's Text Overlay)
      for (const [wname, height] of Object.entries(FIXED_HEIGHT)) {
        const tw = getW(this, wname);
        if (tw) {
          tw.computeSize = (width) => [width, height];
          if (tw.computeLayoutSize) tw.computeLayoutSize = () => ({ minHeight: height, maxHeight: height, minWidth: 0 });
        }
      }

      this.lcMaskPreview = new LCMaskPreview(this);

      const applySize = () => {
        if (!this.size) return;
        this.size = [DEFAULT_W, launchHeight(this)];
        this.setDirtyCanvas?.(true, true);
      };
      applySize();
      // The add-node search dialog re-assigns size right after this hook; a node restored from
      // a saved workflow (onConfigure, same tick) keeps its saved size.
      setTimeout(() => {
        if (!this._lcMaskConfigured) applySize();
      }, 0);
      return r;
    };
  },
});
