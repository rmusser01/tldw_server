import React from "react"

/**
 * Small square icon button used in the composer bottom bar (attach, voice,
 * prompt library, slash commands). 28×28 by default; compact density picks
 * 24×24.
 */

export interface IconButtonProps {
  /** Required — short label for screen readers + title tooltip. */
  label: string
  onClick?: () => void
  children: React.ReactNode
  active?: boolean
  /** Explicit toggle semantics. Omit for buttons that are not true toggles. */
  pressed?: boolean
  disabled?: boolean
  density?: "desktop" | "compact"
}

export const IconButton: React.FC<IconButtonProps> = ({
  label,
  onClick,
  children,
  active = false,
  pressed,
  disabled = false,
  density = "desktop",
}) => {
  const size = density === "compact" ? "w-6 h-6 text-xs" : "w-7 h-7 text-sm"
  const baseCls =
    `inline-flex items-center justify-center ${size} rounded-sm font-mono transition-colors ` +
    "border border-transparent"
  const stateCls = active
    ? "text-primary border-primary/40 bg-primary/10"
    : "text-text-muted hover:text-text hover:border-border"
  const cls = `${baseCls} ${stateCls}${
    disabled ? " opacity-40 cursor-not-allowed" : ""
  }`

  return (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={typeof pressed === "boolean" ? pressed : undefined}
      title={label}
    >
      <span aria-hidden="true">{children}</span>
    </button>
  )
}
