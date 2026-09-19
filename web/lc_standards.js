/**
 * Shared size standards for the LC image nodes. These are the numbers the LC123 image nodes use
 * (lc_image_preview.js, lc_image_crop.js, lc_tone_match.js): 300 px wide (never narrower than 260),
 * 16 px of padding on the sides, above the preview and below it, and a 4:5 preview area.
 */

export const LC_W = 300;
export const LC_MIN_W = 260;
export const LC_PAD = 16;
export const LC_PREVIEW_ASPECT = 5 / 4; // height / width of the preview area

/** Height of the preview area for a node LC_W wide. */
export function lcPreviewHeight() {
  return Math.round((LC_W - LC_PAD * 2) * LC_PREVIEW_ASPECT);
}

/** Never let the node get narrower than the standard minimum. */
export function lcClampMinWidth(size) {
  if (size && size[0] < LC_MIN_W) size[0] = LC_MIN_W;
  return size;
}

/** Bottom of the last widget, measured from where the widgets were actually drawn (null before the first draw). */
export function lcWidgetsBottom(node) {
  let bottom = null;
  for (const w of node.widgets || []) {
    if (!w || w.type === "hidden" || w._lcHidden) continue;
    if (typeof w.last_y === "number") {
      const h = typeof w.computeSize === "function" ? (w.computeSize(node.size[0])?.[1] ?? 20) : 20;
      bottom = Math.max(bottom ?? 0, w.last_y + Math.max(20, h));
    }
  }
  return bottom;
}

/**
 * Give a brand-new preview node its launch size once its widgets have been drawn, so the preview area
 * is exactly the standard size (268 x 335) with the standard padding below it. Call it from the draw hook.
 * Nodes restored from a saved workflow (`configured`) keep the size they were saved with.
 */
export function lcLaunchFit(node, configured) {
  if (node._lcLaunchFitted) return;
  const bottom = lcWidgetsBottom(node);
  if (bottom == null) return;
  node._lcLaunchFitted = true;
  if (configured) return;
  node.size = [LC_W, bottom + LC_PAD + lcPreviewHeight() + LC_PAD];
  node.setDirtyCanvas?.(true, true);
}
