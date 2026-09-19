/**
 * Pack-wide launch look for every LC MaskMaker node:
 *  - default color (only on brand-new nodes; saved workflows and user-picked colors win)
 *  - the LC size standards (see lc_standards.js): 300 px wide, never narrower than 260.
 * Nodes with an on-node preview set their own launch height in their own script.
 */
import { app } from "../../scripts/app.js";
import { lcApplyLaunchColor } from "./lc_color.js";
import { LC_W, lcClampMinWidth } from "./lc_standards.js";

const COLOR = "#324b4b";
const BGCOLOR = "#324b4b";

// Nodes with a preview or a canvas (they size themselves)
const WITH_PREVIEW = new Set([
  "LCImageOutpaint",
  "LCSplineMask",
  "LCMaskRefine",
  "LCSegmentAnything",
]);
// Nodes without one: only the width standard applies, height stays natural
const PLAIN = new Set([
  "LCRemBG",
  "LCPersonMask",
  "LCAutoAdjust",
  "LCDepthAnythingV2",
  "LCNormalBAE",
  "LCImageBlendAdvance",
]);

app.registerExtension({
  name: "LCMaskMaker.Style",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name;
    if (!WITH_PREVIEW.has(name) && !PLAIN.has(name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      lcApplyLaunchColor(this, COLOR, BGCOLOR);

      if (PLAIN.has(name)) {
        const applyWidth = () => {
          if (!this.size) return;
          this.size = [LC_W, this.computeSize?.([LC_W, 0])?.[1] ?? this.size[1]];
          this.setDirtyCanvas?.(true, true);
        };
        applyWidth();
        // The add-node search dialog re-assigns size right after this hook; a node restored from
        // a saved workflow (onConfigure, same tick) keeps its saved size.
        setTimeout(() => {
          if (!this._lcStyleConfigured) applyWidth();
        }, 0);
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      this._lcStyleConfigured = true;
      return onConfigure?.apply(this, arguments);
    };

    if (PLAIN.has(name)) {
      const onResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function (size) {
        lcClampMinWidth(size);
        return onResize?.apply(this, arguments);
      };
    }
  },
});
