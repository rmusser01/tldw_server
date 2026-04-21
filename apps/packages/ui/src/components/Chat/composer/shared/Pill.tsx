import React from "react"

/**
 * Small inline pill used across the Primer composer variants. The designer's
 * `.pill` class (from composer-redesign.html) lives here in React form.
 *
 * Visual states:
 *   - default  — subtle surface-2 background, muted text
 *   - on       — cyan primary tint, primary border
 *   - accent   — amber accent tint (used in V1's RAG badge)
 */

export type PillVariant = "default" | "on" | "accent"

export interface PillProps {
  variant?: PillVariant
  onClick?: () => void
  className?: string
  children: React.ReactNode
  /** Optional aria-label for icon-only pills. */
  "aria-label"?: string
  /** Render as <button> (default) when onClick is provided, else <span>. */
  as?: "button" | "span"
  disabled?: boolean
}

const BASE_CLASS =
  "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[11px] whitespace-nowrap border transition-colors"

const VARIANT_CLASS: Record<PillVariant, string> = {
  default:
    "text-text-muted bg-surface2 border-border hover:text-text hover:border-border-strong",
  on:
    "text-primary border-primary/50 bg-primary/10 hover:bg-primary/15",
  accent:
    "text-accent border-accent/45 bg-accent/10 hover:bg-accent/15",
}

export const Pill: React.FC<PillProps> = ({
  variant = "default",
  onClick,
  className,
  children,
  "aria-label": ariaLabel,
  as,
  disabled,
}) => {
  const Element = as ?? (onClick ? "button" : "span")
  const cls = `${BASE_CLASS} ${VARIANT_CLASS[variant]}${
    disabled ? " opacity-50 cursor-not-allowed" : onClick ? " cursor-pointer" : ""
  }${className ? ` ${className}` : ""}`

  if (Element === "button") {
    return (
      <button
        type="button"
        className={cls}
        onClick={onClick}
        disabled={disabled}
        aria-label={ariaLabel}
      >
        {children}
      </button>
    )
  }
  // `as="span"` + `onClick` combination — honor the click via role="button"
  // and keyboard activation so a11y doesn't break. Callers that want a
  // non-interactive span should omit onClick entirely.
  if (onClick) {
    return (
      <span
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label={ariaLabel}
        aria-disabled={disabled || undefined}
        onClick={disabled ? undefined : onClick}
        onKeyDown={(e) => {
          if (disabled) return
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            onClick()
          }
        }}
        className={cls}
      >
        {children}
      </span>
    )
  }
  return (
    <span className={cls} aria-label={ariaLabel}>
      {children}
    </span>
  )
}
