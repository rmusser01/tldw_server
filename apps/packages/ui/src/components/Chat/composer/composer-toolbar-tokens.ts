/**
 * Size grammar for composer toolbar controls (CC-07, #3101).
 *
 * The composer mixed control heights (16-44px), label sizes (10-16px) and
 * radii, so the primary Send action read as the weakest control. Toolbar
 * controls share one control height, one label size and one radius.
 *
 * Owner decision [3] (2026-11-19): controls are 32px on desktop and 44px on
 * touch. Below the `sm` breakpoint and on a coarse pointer (a touch screen at
 * any width) a control keeps its 44px target; otherwise it is 32px.
 *
 * Keep the values as literal class strings so Tailwind's content scan finds
 * them. Tailwind emits the `[@media(pointer:coarse)]` variant after `sm:`, so
 * touch wins over the desktop size.
 */
export const COMPOSER_TOOLBAR_TOKENS = {
  /** 32px square for an icon-only toolbar action on wider screens. */
  iconControl:
    "sm:h-8 sm:w-8 sm:min-h-0 sm:min-w-0 [@media(pointer:coarse)]:h-11 [@media(pointer:coarse)]:w-11",
  /** A complete icon-only toolbar button: 44px, 32px on desktop. */
  iconButton:
    "inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md sm:h-8 sm:w-8 [@media(pointer:coarse)]:h-11 [@media(pointer:coarse)]:w-11",
  /** A labelled toolbar control: 44px tall, 32px on desktop, 13px label. */
  textControl:
    "inline-flex min-h-11 items-center gap-1.5 rounded-md px-2.5 text-[13px] font-medium sm:min-h-8 [@media(pointer:coarse)]:min-h-11",
  /** 13px sentence-case label for a toolbar control. */
  label: "text-[13px] font-medium",
  /** The one corner radius toolbar controls share. */
  radius: "rounded-md"
} as const
