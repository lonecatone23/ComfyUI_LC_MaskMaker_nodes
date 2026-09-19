/**
 * On-node before/after preview for the mask nodes (Mask Refine, Segment Anything).
 * Same behavior as the LC123 image previews: hover the node and the after image sits on the left
 * of the pointer with the before image on the right; move away and it shows the full after image.
 *
 * The node sends up to four small images in message.lc_refine: source, before mask, after mask, trimap.
 * preview_view: "cutout" puts the subject on a checkerboard, "mask" shows black/white,
 * "trimap" shows the trimap as the before side.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { lcApplyLaunchColor } from "./lc_color.js";
import { LC_W as DEFAULT_W, LC_MIN_W as MIN_W, LC_PAD as PAD, lcPreviewHeight, lcLaunchFit } from "./lc_standards.js";

const NODE_CLASSES = new Set(["LCMaskRefine", "LCSegmentAnything"]);
// Multiline text widgets that must stay a fixed height (otherwise they grow to fill the node)
const FIXED_HEIGHT = { prompt: 64 };


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

class LCMaskPreview {
  constructor(node) {
    this.node = node;
    this.views = null;
    this.pointerOver = false;
    this.pointerPos = [0, 0];
    this._bind(node);
    queueMicrotask(() => this._restoreFromProps());
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
      if (self.pointerOver && self.views) app.canvas?.setDirty?.(true, true);
      return origMouseMove ? origMouseMove.apply(this, arguments) : false;
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

  // Keep the preview info on the node so undo / reload can bring the images back
  _restoreFromProps() {
    const metas = this.node.properties?.lc_refine_meta;
    if (metas?.length) this._load(metas);
  }

  onExecuted(message) {
    const metas = message?.lc_refine;
    if (!metas?.length) return;
    if (!this.node.properties) this.node.properties = {};
    this.node.properties.lc_refine_meta = metas;
    this._load(metas);
  }

  _paintSide(ctx, which, r, view) {
    const v = this.views;
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
    const top = widgetsBottom(node) + PAD;
    const x = PAD;
    const w = Math.max(1, node.size[0] - PAD * 2);
    const h = Math.max(1, node.size[1] - top - PAD);
    if (h < 16) return;

    // fit the image inside the box (letterbox)
    const s = Math.min(w / v.w, h / v.h);
    const iw = v.w * s;
    const ih = v.h * s;
    const r = { x: x + (w - iw) / 2, y: top + (h - ih) / 2, w: iw, h: ih };
    const view = getW(node, "preview_view")?.value || "cutout";

    // before underneath, after clipped to the left of the pointer (full after when not hovering)
    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x, r.y, r.w, r.h);
    ctx.clip();
    this._paintSide(ctx, "before", r, view);
    ctx.restore();

    let splitX = r.w;
    if (this.pointerOver) splitX = Math.max(0, Math.min(r.w, this.pointerPos[0] - r.x));
    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x, r.y, splitX, r.h);
    ctx.clip();
    this._paintSide(ctx, "after", r, view);
    ctx.restore();

    if (this.pointerOver) {
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(r.x + splitX, r.y);
      ctx.lineTo(r.x + splitX, r.y + r.h);
      ctx.stroke();
    }
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
