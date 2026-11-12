/**
 * Where the side panel's "Scroll to latest" pill sits (XS-09).
 *
 * The pill floats 24px above the measured composer, whether the composer is
 * pinned to the bottom or scrolls with the messages, so it never covers the
 * input, Send or the composer's controls. Before the composer has been
 * measured it uses the side panel's default composer height.
 */
const PILL_GAP_PX = 24
const MIN_PILL_BOTTOM_PX = 128

export const sidepanelScrollToLatestBottom = (
  composerHeightPx: number,
  defaultComposerHeightPx: number
): string => {
  const base =
    composerHeightPx > 0
      ? composerHeightPx + PILL_GAP_PX
      : defaultComposerHeightPx
  return `${Math.max(base, MIN_PILL_BOTTOM_PX)}px`
}
