/**
 * LC Face Expression 🎭: drag editor on the node (a DOM widget, so it works with Nodes 2.0 on or off).
 * After one run the node keeps its read of the face; dragging a handle writes the lever widgets and asks
 * /lc_maskmaker/face_expression/render for a new 512 px face (no queue, no sampler). The sliders stay the saved
 * state, so dragging and typing a value are the same thing.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LC_W, LC_MIN_W, LC_PAD } from "./lc_standards.js";

const NODE_CLASS = "LCFaceExpression";
const LEVERS = ["smile", "smirk", "mouth_open", "mouth_wide", "eye_left", "eye_right", "gaze_x", "gaze_y", "brows", "furrow",
  "brow_tilt", "head_turn", "head_nod", "head_tilt"];
const RANGE = {
  smile: [-1, 1], smirk: [-1, 1], mouth_open: [-1, 1], mouth_wide: [-1, 1], eye_left: [-1, 1], eye_right: [-1, 1],
  gaze_x: [-1, 1], gaze_y: [-1, 1], brows: [-1, 1], furrow: [-1, 1], brow_tilt: [-1, 1],
  head_turn: [-20, 20], head_nod: [-20, 20], head_tilt: [-20, 20],
};
const SIZE = LC_W - LC_PAD * 2; // the square editor, CSS px
const EDITOR_H = SIZE + LC_PAD;
const HIT = 11; // px around a handle that still grabs it
const DOUBLE_MS = 300;
const AXIS_LOCK = 6; // px before a mouth / brow drag picks its direction
// how far (in face-box units) a handle travels for a full lever
const SB = 0.12, SF = 0.08, SL = 0.08, RG = 0.09, SS = 0.11, SW = 0.08, SO = 0.2;
const HINT = "Run the node once to edit the face here.";

const getW = (node, name) => (node.widgets || []).find((w) => w && w.name === name);
const lv = (node, name) => Number(getW(node, name)?.value ?? 0) || 0;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// where the face really is: the photo's own expression (read by the node) plus the lever change. The handles
// show this, and a drag stops at its real ends (an eye that is already half closed can only close half more)
const ABS = { mouth_open: [0, 1], eye_left: [-1, 0.5], eye_right: [-1, 0.5] };
const rd = (node, name) => Number(node._feRead?.[name] ?? 0) || 0;
const av = (node, name) => (name in ABS ? rd(node, name) : 0) + lv(node, name);

function setLever(node, name, v) {
  const w = getW(node, name);
  if (!w) return;
  const [lo, hi] = RANGE[name];
  v = clamp(v, lo, hi);
  w.value = name.startsWith("head_") ? Math.round(v) : Math.round(v * 100) / 100;
}

function setAbs(node, name, v) {
  if (!(name in ABS)) return setLever(node, name, v);
  setLever(node, name, clamp(v, ABS[name][0], ABS[name][1]) - rd(node, name));
}

function markWorkflowChanged() {
  try {
    const ct = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
    ct?.captureCanvasState ? ct.captureCanvasState() : ct?.checkState?.();
  } catch (_) {}
}

function loadImg(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// ---- server ----

async function fetchSource(node) {
  const key = node.properties?.lc_face_key;
  node._feSrc = null;
  node._feImg = null;
  node._feBox = null;
  if (!key) return paint(node);
  try {
    const r = await api.fetchApi(`/lc_maskmaker/face_expression/source?key=${encodeURIComponent(key)}`);
    const res = await r.json();
    if (res.error) {
      node._feStatus = res.error;
      return;
    }
    node._feBox = res.box;
    node._feRead = res.read || {};
    node._feKp = res.kp || null;
    node._fePresets = res.presets || null;
    node._feSrc = await loadImg(res.img);
    node._feImg = node._feSrc;
    node._feStatus = "";
    if (LEVERS.some((n) => lv(node, n)) || lv(node, "keep_expression") < 1) scheduleRender(node);
  } catch (e) {
    node._feStatus = HINT;
  } finally {
    paint(node);
  }
}

function scheduleRender(node) {
  paint(node);
  if (!node._feSrc) return;
  node._feWant = true;
  if (!node._feBusy) doRender(node);
}

async function doRender(node) {
  node._feBusy = true;
  node._feWant = false;
  const levers = {};
  for (const n of LEVERS) levers[n] = lv(node, n);
  try {
    const r = await api.fetchApi("/lc_maskmaker/face_expression/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: node.properties.lc_face_key, levers, keep: lv(node, "keep_expression") }),
    });
    const res = await r.json();
    if (res.error) {
      node._feStatus = res.error;
      node._feSrc = null;
      node._feImg = null;
    } else {
      node._feImg = await loadImg(res.img);
      node._feStatus = "";
    }
  } catch (e) {
    node._feStatus = "⚠ preview failed";
  } finally {
    node._feBusy = false;
    paint(node);
    if (node._feWant) doRender(node);
  }
}

// ---- handles ----

/** Handle positions (editor px). They sit on LivePortrait's own keypoints for this face (which already include the
 * photo's expression: an open mouth puts the jaw handle low) and move by the lever changes. */
function handles(node) {
  const b = node._feBox;
  const fw = b[2] - b[0], fh = b[3] - b[1];
  const kp = node._feKp;
  const mean = (ids) => [ids.reduce((s, i) => s + kp[i][0], 0) / ids.length, ids.reduce((s, i) => s + kp[i][1], 0) / ids.length];
  // the keypoints: 11-13 the image-left eye, 14-16 the image-right eye, 17 / 18 mouth corners, 19 lower lip and jaw
  const eyeL = kp ? mean([11, 12, 13]) : [(b[0] + b[2]) / 2 - 0.2 * fw, b[1] + 0.425 * fh];
  const eyeR = kp ? mean([14, 15, 16]) : [(b[0] + b[2]) / 2 + 0.2 * fw, b[1] + 0.425 * fh];
  const iod = Math.hypot(eyeR[0] - eyeL[0], eyeR[1] - eyeL[1]) || 0.4 * fw;
  const mL = kp ? kp[17] : [(b[0] + b[2]) / 2 - 0.14 * fw, b[1] + 0.79 * fh];
  const mR = kp ? kp[18] : [(b[0] + b[2]) / 2 + 0.14 * fw, b[1] + 0.79 * fh];
  const jaw = kp ? [kp[19][0], kp[19][1] + 0.08 * iod] : [(b[0] + b[2]) / 2, b[1] + 0.97 * fh];
  const P = (u, v) => [u * SIZE, v * SIZE];
  const brows = lv(node, "brows"), tilt = lv(node, "brow_tilt"), furrow = lv(node, "furrow");
  const vL = brows + tilt, vR = brows - tilt;
  const smile = lv(node, "smile"), smirk = lv(node, "smirk"), wide = lv(node, "mouth_wide");
  const gx = lv(node, "gaze_x"), gy = lv(node, "gaze_y");
  const browUp = 0.42 * iod, lidUp = 0.12 * iod;
  const gc = [(eyeL[0] + eyeR[0]) / 2, (eyeL[1] + eyeR[1]) / 2];
  const hr = 0.075 * SIZE;
  const hc = [SIZE - hr - 8, SIZE - hr - 8];
  return {
    browL: P(eyeL[0] + furrow * SF * fw, eyeL[1] - browUp - vL * SB * fh),
    browR: P(eyeR[0] - furrow * SF * fw, eyeR[1] - browUp - vR * SB * fh),
    lidL: P(eyeL[0], eyeL[1] - lidUp - lv(node, "eye_left") * SL * fh),
    lidR: P(eyeR[0], eyeR[1] - lidUp - lv(node, "eye_right") * SL * fh),
    gazeC: P(gc[0], gc[1]),
    gazeR: RG * fw * SIZE,
    gaze: P(gc[0] + gx * RG * fw, gc[1] - gy * RG * fw),
    mouthL: P(mL[0] - wide * SW * fw, mL[1] - (smile - smirk) * SS * fh),
    mouthR: P(mR[0] + wide * SW * fw, mR[1] - (smile + smirk) * SS * fh),
    chin: P(jaw[0], jaw[1] + lv(node, "mouth_open") * SO * fh),
    headC: hc,
    headR: hr,
    head: [hc[0] + (lv(node, "head_turn") / 20) * hr, hc[1] + (lv(node, "head_nod") / 20) * hr],
  };
}

const GRAB = ["gaze", "browL", "browR", "lidL", "lidR", "mouthL", "mouthR", "chin", "head"];
const RESET = {
  browL: ["brows", "brow_tilt", "furrow"], browR: ["brows", "brow_tilt", "furrow"], lidL: ["eye_left"],
  lidR: ["eye_right"], gaze: ["gaze_x", "gaze_y"], mouthL: ["smile", "smirk", "mouth_wide"], mouthR: ["smile", "smirk", "mouth_wide"],
  chin: ["mouth_open"], head: ["head_turn", "head_nod", "head_tilt"],
};

// ---- drawing ----

function dot(ctx, p, r, on) {
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fillStyle = on ? "#ffd27f" : "rgba(255,255,255,0.9)";
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(0,0,0,0.75)";
  ctx.stroke();
}

function bar(ctx, p, w, on) {
  ctx.lineCap = "round";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.beginPath();
  ctx.moveTo(p[0] - w, p[1]);
  ctx.lineTo(p[0] + w, p[1]);
  ctx.stroke();
  ctx.lineWidth = 3;
  ctx.strokeStyle = on ? "#ffd27f" : "rgba(255,255,255,0.9)";
  ctx.stroke();
}

function paint(node) {
  const ed = node._feEd;
  if (!ed) return;
  const { canvas, ctx } = ed;
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1)) * 2;
  if (canvas.width !== SIZE * dpr) {
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, SIZE, SIZE);
  if (!node._feImg) {
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(node._feStatus || HINT, SIZE / 2, SIZE / 2);
    return;
  }
  ctx.drawImage(node._feImg, 0, 0, SIZE, SIZE);
  if (node._feHide || !node._feBox) return;
  const H = handles(node);
  const on = (k) => node._fePress?.handle === k;
  ctx.beginPath();
  ctx.arc(H.gazeC[0], H.gazeC[1], H.gazeR, 0, Math.PI * 2);
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.stroke();
  ctx.setLineDash([]);
  dot(ctx, H.gaze, 4.5, on("gaze"));
  bar(ctx, H.browL, 9, on("browL"));
  bar(ctx, H.browR, 9, on("browR"));
  bar(ctx, H.lidL, 6, on("lidL"));
  bar(ctx, H.lidR, 6, on("lidR"));
  dot(ctx, H.mouthL, 4.5, on("mouthL"));
  dot(ctx, H.mouthR, 4.5, on("mouthR"));
  dot(ctx, H.chin, 4.5, on("chin"));
  // head pad: drag for turn / nod, Shift + drag sideways for tilt (the line shows the tilt)
  ctx.beginPath();
  ctx.arc(H.headC[0], H.headC[1], H.headR, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = 1;
  ctx.stroke();
  const t = (lv(node, "head_tilt") / 20) * (Math.PI / 4);
  ctx.beginPath();
  ctx.moveTo(H.headC[0] - Math.cos(t) * H.headR, H.headC[1] - Math.sin(t) * H.headR);
  ctx.lineTo(H.headC[0] + Math.cos(t) * H.headR, H.headC[1] + Math.sin(t) * H.headR);
  ctx.stroke();
  dot(ctx, H.head, 5, on("head"));
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("head", H.headC[0], H.headC[1] - H.headR - 3);
  const text = node._feBusy ? "…" : (node._feStatus || "");
  if (text) {
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(3, 3, tw + 10, 17);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, 8, 15);
  }
}

// ---- interaction ----

function toLocal(node, e) {
  const r = node._feEd.canvas.getBoundingClientRect();
  return [((e.clientX - r.left) * SIZE) / Math.max(1, r.width), ((e.clientY - r.top) * SIZE) / Math.max(1, r.height)];
}

function hit(node, p) {
  if (!node._feSrc || !node._feBox || node._feHide) return null;
  const H = handles(node);
  let best = null, bd = HIT;
  for (const k of GRAB) {
    const d = Math.hypot(p[0] - H[k][0], p[1] - H[k][1]);
    if (d < bd) { bd = d; best = k; }
  }
  if (!best && Math.hypot(p[0] - H.headC[0], p[1] - H.headC[1]) < H.headR) best = "head";
  return best;
}

function drag(node, press, p, shift) {
  const b = node._feBox;
  const fw = (b[2] - b[0]) * SIZE, fh = (b[3] - b[1]) * SIZE;
  let dx = p[0] - press.x, dy = p[1] - press.y;
  // mouth and brows do two things (up/down and in/out): the direction the drag starts in wins, so a smile
  // dragged a little crooked doesn't also stretch the mouth
  if (/^(mouth|brow)/.test(press.handle)) {
    if (!press.axis && Math.hypot(dx, dy) > AXIS_LOCK) press.axis = Math.abs(dx) > Math.abs(dy) ? "x" : "y";
    if (press.axis === "x") dy = 0;
    else dx = 0;
  }
  const v = press.start;
  switch (press.handle) {
    case "browL":
    case "browR": {
      const left = press.handle === "browL";
      let vL = v.brows + v.brow_tilt, vR = v.brows - v.brow_tilt;
      const nv = (left ? vL : vR) - dy / (SB * fh);
      if (shift) { vL = nv; vR = nv; } else if (left) vL = nv; else vR = nv;
      setAbs(node, "brows", (vL + vR) / 2);
      setAbs(node, "brow_tilt", (vL - vR) / 2);
      setAbs(node, "furrow", v.furrow + (left ? dx : -dx) / (SF * fw));
      break;
    }
    case "lidL":
    case "lidR": {
      const name = press.handle === "lidL" ? "eye_left" : "eye_right";
      const nv = v[name] - dy / (SL * fh);
      setAbs(node, name, nv);
      if (shift) setAbs(node, name === "eye_left" ? "eye_right" : "eye_left", nv);
      break;
    }
    case "gaze": {
      let gx = v.gaze_x + dx / (RG * fw), gy = v.gaze_y - dy / (RG * fw);
      const r = Math.hypot(gx, gy);
      if (r > 1) { gx /= r; gy /= r; }
      setAbs(node, "gaze_x", gx);
      setAbs(node, "gaze_y", gy);
      break;
    }
    case "mouthL":
    case "mouthR": {
      const left = press.handle === "mouthL";
      let cL = v.smile - v.smirk, cR = v.smile + v.smirk;
      const nv = (left ? cL : cR) - dy / (SS * fh);
      if (shift) { cL = nv; cR = nv; } else if (left) cL = nv; else cR = nv;
      setAbs(node, "smile", (cL + cR) / 2);
      setAbs(node, "smirk", (cR - cL) / 2);
      setAbs(node, "mouth_wide", v.mouth_wide + (left ? -dx : dx) / (SW * fw));
      break;
    }
    case "chin":
      setAbs(node, "mouth_open", v.mouth_open + dy / (SO * fh));
      break;
    case "head": {
      const r = handles(node).headR;
      if (shift) setAbs(node, "head_tilt", v.head_tilt + (dx / r) * 20);
      else {
        setAbs(node, "head_turn", v.head_turn + (dx / r) * 20);
        setAbs(node, "head_nod", v.head_nod + (dy / r) * 20);
      }
      break;
    }
  }
  node.setDirtyCanvas?.(true, false); // the sliders show the new values
  scheduleRender(node);
}

/** A named preset: the levers that take this face from its own expression toward the preset (as the node does). */
function applyPreset(node) {
  const name = getW(node, "preset")?.value;
  const target = node._fePresets?.[name];
  if (!target || !node._feRead) return false;
  const k = lv(node, "preset_strength");
  // the same read the node uses for presets: only mouth, smile and the eyes as a pair (the rest of the read is noise)
  const eyes = (rd(node, "eye_left") + rd(node, "eye_right")) / 2;
  const base = (n) => (n === "eye_left" || n === "eye_right" ? eyes
    : n === "mouth_open" || n === "smile" ? rd(node, n) : 0);
  for (const n of Object.keys(node._feRead)) setLever(node, n, ((target[n] || 0) - base(n)) * k);
  node.setDirtyCanvas?.(true, false);
  scheduleRender(node);
  return true;
}

/** A lever changed by hand (drag, slider, reset): the face is no longer the preset. */
function toCustom(node) {
  const w = getW(node, "preset");
  if (w && w.value !== "Custom") {
    w.value = "Custom";
    node.setDirtyCanvas?.(true, false);
  }
}

function resetAll(node) {
  toCustom(node);
  for (const n of LEVERS) setLever(node, n, 0);
  node.setDirtyCanvas?.(true, false);
  scheduleRender(node);
  markWorkflowChanged();
}

function attachEditor(node) {
  if (node._feEd) return;
  const wrap = document.createElement("div");
  wrap.style.cssText = `width:100%;height:${EDITOR_H}px;display:flex;justify-content:center;align-items:flex-start;`;
  const canvas = document.createElement("canvas");
  canvas.style.cssText = `width:${SIZE}px;height:${SIZE}px;border-radius:4px;touch-action:none;cursor:default;`;
  wrap.append(canvas);
  node._feEd = { wrap, canvas, ctx: canvas.getContext("2d") };

  // the drag keeps the pointer captured on this canvas: Nodes 2.0 stops move events from reaching the window
  const onUp = (e) => {
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    if (node._fePress) {
      node._fePress = null;
      paint(node);
      markWorkflowChanged();
    }
  };
  canvas.addEventListener("pointermove", (e) => {
    if (node._fePress) {
      e.preventDefault();
      e.stopPropagation();
      drag(node, node._fePress, toLocal(node, e), !!e.shiftKey);
      return;
    }
    canvas.style.cursor = hit(node, toLocal(node, e)) ? "grab" : "default";
  });
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("lostpointercapture", onUp);
  canvas.addEventListener("pointerdown", (e) => {
    const p = toLocal(node, e);
    const k = hit(node, p);
    if (!k) return; // empty spots: let the graph pan / select as usual
    e.preventDefault();
    e.stopPropagation();
    const now = Date.now();
    if (node._feLast && node._feLast.k === k && now - node._feLast.t < DOUBLE_MS) {
      toCustom(node);
      for (const n of RESET[k]) setLever(node, n, 0);
      node._feLast = null;
      node.setDirtyCanvas?.(true, false);
      scheduleRender(node);
      markWorkflowChanged();
      return;
    }
    node._feLast = { k, t: now };
    const start = {};
    for (const n of LEVERS) start[n] = av(node, n);
    node._fePress = { handle: k, x: p[0], y: p[1], start };
    toCustom(node);
    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    paint(node);
  });

  try {
    const w = node.addDOMWidget("lc_face_editor", "LC_FACE_EDITOR", wrap, {
      getMinHeight: () => EDITOR_H,
      getHeight: () => EDITOR_H,
      serialize: false,
    });
    w.computeSize = (width) => [width, EDITOR_H];
    // the Parameters side panel writes its own column width onto widgets: always follow the node
    Object.defineProperty(w, "width", { get: () => undefined, set: () => {}, configurable: true });
  } catch (e) {
    console.warn("LC Face Expression editor failed", e);
  }
  paint(node);
}

app.registerExtension({
  name: "LCMaskMaker.FaceExpression",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this.properties = this.properties || {};
      for (const name of [...LEVERS, "keep_expression"]) {
        const w = getW(this, name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = (...a) => {
          const out = cb?.apply(w, a);
          if (name !== "keep_expression") toCustom(this);
          scheduleRender(this);
          return out;
        };
      }
      for (const name of ["preset", "preset_strength"]) {
        const w = getW(this, name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = (...a) => {
          const out = cb?.apply(w, a);
          applyPreset(this);
          return out;
        };
      }
      this.addWidget("button", "Reset face", null, () => resetAll(this), { serialize: false });
      const toggle = this.addWidget("button", "Hide controls", null, () => {
        this._feHide = !this._feHide;
        toggle.name = toggle.label = this._feHide ? "Show controls" : "Hide controls";
        paint(this);
        this.setDirtyCanvas?.(true, true);
      }, { serialize: false });
      attachEditor(this);
      // a new node: grow once to fit the editor (a loaded workflow keeps its saved size)
      requestAnimationFrame(() => {
        if (this._feConfigured) return;
        try {
          this.setSize([Math.max(this.size[0], LC_W), Math.max(this.size[1], this.computeSize()[1])]);
          this.setDirtyCanvas?.(true, true);
        } catch (_) {}
      });
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this._feConfigured = true;
      if (this.properties?.lc_face_key) fetchSource(this);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const key = message?.lc_face?.[0]?.key;
      const sent = message?.lc_face?.[0]?.levers;
      if (sent) {
        // a named preset: the node worked out this face's levers, show them
        for (const n of Object.keys(sent)) setLever(this, n, sent[n]);
        this.setDirtyCanvas?.(true, true);
      }
      if (message?.lc_face?.[0]?.reset) {
        // a new picture came in: the levers were for the last face, start this one from its own expression
        for (const n of LEVERS) setLever(this, n, 0);
        const keep = getW(this, "keep_expression");
        if (keep) keep.value = 1;
        this._fePress = null;
        this.setDirtyCanvas?.(true, true);
        markWorkflowChanged();
      }
      if (key && (key !== this.properties?.lc_face_key || !this._feSrc)) {
        this.properties = this.properties || {};
        this.properties.lc_face_key = key;
        fetchSource(this);
      }
      return r;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size[0] < LC_MIN_W) size[0] = LC_MIN_W;
      return onResize?.apply(this, arguments);
    };
  },
});
