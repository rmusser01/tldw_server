/**
 * Where the side panel's "Scroll to latest" pill sits (XS-09, G19).
 *
 * A pinned composer stays at the bottom of the panel, so the pill floats 24px
 * above it and never covers the input, Send or the composer's controls;
 * before the composer is measured it uses the side panel's default composer
 * height. An unpinned composer scrolls with the messages and is not under the
 * pill, so the pill sits just above the bottom edge (and the safe area).
 */
const PILL_GAP_PX = 24
const MIN_PILL_BOTTOM_PX = 128
const UNPINNED_PILL_BOTTOM = "calc(16px + env(safe-area-inset-bottom, 0px))"

export const sidepanelScrollToLatestBottom = (
  composerHeightPx: number,
  defaultComposerHeightPx: number,
  composerPinned: boolean
): string => {
  if (!composerPinned) return UNPINNED_PILL_BOTTOM
  const base =
    composerHeightPx > 0
      ? composerHeightPx + PILL_GAP_PX
      : defaultComposerHeightPx
  return `${Math.max(base, MIN_PILL_BOTTOM_PX)}px`
}
