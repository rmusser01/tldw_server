/**
 * Size grammar for composer toolbar controls (CC-07, #3101).
 *
 * The composer mixed control heights (16-44px), label sizes (10-16px) and
 * radii, so the primary Send action read as the weakest control. Toolbar
 * controls share one control height, one label size and one radius; touch
 * layouts keep their 44px targets.
 *
 * Keep the values as literal class strings so Tailwind's content scan finds them.
 */
export const COMPOSER_TOOLBAR_TOKENS = {
  /** 32px square for an icon-only toolbar action on wider screens. */
  iconControl: "sm:h-8 sm:w-8 sm:min-h-0 sm:min-w-0",
  /** 13px sentence-case label for a toolbar control. */
  label: "text-[13px] font-medium",
  /** The one corner radius toolbar controls share. */
  radius: "rounded-md"
} as const
