import React from "react"

/**
 * Primary send button used in the composer bottom bar. Defaults to "Send"
 * with a ⌘↩ kbd hint; when `stopping` is true, renders as a danger-tinted
 * "Stop" button to cancel an in-flight generation.
 *
 * Shapes:
 *   - `rect`  — default rectangular button, "Send ⌘↩" desktop, ↩ compact.
 *               Used by V1 Terminal Stack and V3 Split Brief.
 *   - `round` — 36×36 circle with glyph only and a persistent glow.
 *               Used by V5 Radial Command.
 *
 * Compact density (sidepanel) hides the text label and kbd hint, leaving
 * just the return-arrow glyph in a smaller button.
 */

export type SendButtonShape = "rect" | "round"

export interface SendButtonProps {
  onClick?: () => void
  disabled?: boolean
  /** Show as stop/cancel button during streaming. */
  stopping?: boolean
  density?: "desktop" | "compact"
  /** Visual shape — rect (default) or round. */
  shape?: SendButtonShape
  /** Visible label — defaults to "Send". Ignored when shape="round". */
  label?: string
  /** Stop label — defaults to "Stop". Used as aria-label when shape="round". */
  stopLabel?: string
}

export const SendButton: React.FC<SendButtonProps> = ({
  onClick,
  disabled = false,
  stopping = false,
  density = "desktop",
  shape = "rect",
  label = "Send",
  stopLabel = "Stop",
}) => {
  const compact = density === "compact"
  const colors = stopping
    ? "bg-danger text-surface hover:brightness-110"
    : "bg-primary text-bg hover:[box-shadow:var(--glow-primary)]"

  if (shape === "round") {
    // V5's Radial Command send — 36×36 circle, glyph-only, persistent glow.
    // Compact shrinks to 32×32 but still exceeds WCAG 2.5.5 AAA floor.
    const roundSize = compact ? "w-8 h-8 text-xs" : "w-9 h-9 text-sm"
    const roundGlow = stopping ? "" : " [box-shadow:var(--glow-primary)]"
    const cls = `inline-flex items-center justify-center rounded-full ${roundSize} ${colors}${roundGlow} font-semibold transition-shadow${
      disabled ? " opacity-50 cursor-not-allowed" : ""
    }`
    return (
      <button
        type="button"
        className={cls}
        onClick={onClick}
        disabled={disabled}
        aria-label={stopping ? stopLabel : label}
      >
        <span aria-hidden="true">↩</span>
      </button>
    )
  }

  // Rectangular (default) — V1 and V3.
  // WCAG 2.5.5 Target Size (AAA): minimum 24×24 CSS px for non-standard
  // form controls. Compact density must not fall below this floor.
  const size = compact
    ? "min-w-[28px] min-h-[28px] px-2 py-1 text-xs"
    : "px-3.5 py-1.5 text-[13px]"
  const rectShape = compact ? "rounded-sm" : "rounded-md"
  const cls = `inline-flex items-center justify-center gap-1.5 ${size} ${rectShape} ${colors} font-semibold transition-shadow${
    disabled ? " opacity-50 cursor-not-allowed" : ""
  }`

  return (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={disabled}
      aria-label={stopping ? stopLabel : label}
    >
      {compact ? (
        <span aria-hidden="true">↩</span>
      ) : (
        <>
          <span>{stopping ? stopLabel : label}</span>
          <span
            className="font-mono text-[10px] opacity-75"
            aria-hidden="true"
          >
            ⌘↩
          </span>
        </>
      )}
    </button>
  )
}
