/**
 * Layout Tokens
 *
 * Z-index scale, breakpoints, and container widths.
 */

/**
 * Z-index scale for layering elements
 * Use these instead of arbitrary z-* values
 */
export const zIndex = {
  /** Below everything */
  behind: "z-[-1]",
  /** Base level */
  base: "z-0",
  /** Dropdowns, popovers */
  dropdown: "z-10",
  /** Sticky headers */
  sticky: "z-20",
  /** Fixed elements */
  fixed: "z-30",
  /** Modal backdrop */
  modalBackdrop: "z-40",
  /** Modal content */
  modal: "z-50",
  /** Tooltips, popovers over modals */
  tooltip: "z-50",
  /** Toast notifications (highest) */
  toast: "z-[60]",
} as const

/**
 * Container max-widths
 */
export const container = {
  /** 384px - Small dialogs */
  sm: "max-w-sm",
  /** 448px - Medium dialogs */
  md: "max-w-md",
  /** 512px - Standard dialogs */
  lg: "max-w-lg",
  /** 576px - Large dialogs */
  xl: "max-w-xl",
  /** 672px - Extra large */
  "2xl": "max-w-2xl",
  /** 768px - Wide content */
  "3xl": "max-w-3xl",
  /** 896px - Settings content */
  "4xl": "max-w-4xl",
  /** 1024px - Full width content */
  "5xl": "max-w-5xl",
  /** Full width */
  full: "max-w-full",
} as const

/**
 * Sidebar widths
 *
 * The "themed" entry reads from the --sidebar-width CSS variable set by the
 * theme system (default 260 px).  Because Tailwind utility classes cannot
 * reference arbitrary CSS custom properties, components that need the themed
 * sidebar width should apply it via an inline `style` attribute:
 *
 *   style={{ width: "var(--sidebar-width)" }}
 *
 * The static entries below are kept for non-themed / fixed-width sidebars.
 */
export const sidebarWidth = {
  /** Theme-aware sidebar width – use via inline style, not className */
  themed: "var(--sidebar-width)",
  /** 256px - Compact sidebar (static) */
  compact: "w-64",
  /** 320px - Standard sidebar */
  standard: "w-80",
  /** 360px - Wide sidebar */
  wide: "w-[360px]",
  /** 400px - Extra wide */
  extraWide: "w-[400px]",
} as const

/**
 * Touch target sizes for accessibility
 * WCAG 2.1 recommends 44x44px minimum for touch targets
 */
export const touchTarget = {
  /** Minimum size for mobile touch targets */
  min: "min-h-[44px] min-w-[44px]",
  /** Mobile-only minimum (resets on desktop) */
  mobile: "min-h-[44px] sm:min-h-0",
  /** Smaller target for dense UIs (32px) */
  dense: "min-h-[32px] min-w-[32px]",
} as const

export type ZIndex = keyof typeof zIndex
export type Container = keyof typeof container
