import React from "react"

/**
 * Labeled key/value field used in V3's left brief panel. Desktop shows
 * both the key ("src", "mdl", "tmp") and the value ("▣ irb-archive · 14").
 * Compact density hides the key — the value chip stands on its own.
 *
 * Active state uses a primary-tint border + background so researchers can
 * tell at a glance which facets of the brief are wired up.
 */

export interface BriefFieldProps {
  /** Short key, e.g. "src", "mdl". Hidden when `hideKey` is true. */
  fieldKey?: string
  /** Value displayed on the right, e.g. "▣ irb-archive · 14". */
  value: React.ReactNode
  active?: boolean
  onClick?: () => void
  /** Compact sidepanel — drops the key column. */
  hideKey?: boolean
  /** Accessible label for icon-only values. */
  "aria-label"?: string
}

export const BriefField: React.FC<BriefFieldProps> = ({
  fieldKey,
  value,
  active = false,
  onClick,
  hideKey = false,
  "aria-label": ariaLabel,
}) => {
  const baseCls =
    "inline-flex items-center gap-2 px-2 py-1.5 rounded-md font-mono text-[11px] " +
    "border bg-surface text-text-muted transition-colors"
  const stateCls = active
    ? "border-primary/40 bg-primary/[0.06]"
    : "border-border hover:border-border-strong hover:text-text"
  const valueCls = active
    ? "text-primary overflow-hidden text-ellipsis whitespace-nowrap"
    : "text-text overflow-hidden text-ellipsis whitespace-nowrap"

  const cls = `${baseCls} ${stateCls}${onClick ? " cursor-pointer" : ""}`

  const inner = (
    <>
      {!hideKey && fieldKey && (
        <span className="text-text-subtle flex-shrink-0">{fieldKey}</span>
      )}
      <span className={valueCls}>{value}</span>
    </>
  )

  if (onClick) {
    return (
      <button
        type="button"
        className={`${cls} text-left`}
        onClick={onClick}
        aria-label={ariaLabel}
        aria-pressed={active}
      >
        {inner}
      </button>
    )
  }
  return (
    <span className={cls} aria-label={ariaLabel}>
      {inner}
    </span>
  )
}
