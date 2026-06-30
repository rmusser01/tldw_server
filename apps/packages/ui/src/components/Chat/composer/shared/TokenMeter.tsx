import React from "react"

/**
 * Token-count indicator with a tiny capacity bar. Designer's `.tok`:
 *
 *   127 [=====     ] 8K tok
 *
 * The bar auto-computes fill from `used / max`. Color escalates as the
 * ratio crosses thresholds so the indicator warns at a glance.
 */

export interface TokenMeterProps {
  used: number
  max: number
  /** If false, hides the "tok" unit suffix (used in compact sidepanel mode). */
  showUnit?: boolean
  className?: string
}

const formatCount = (n: number): string => {
  if (n >= 1000) {
    const k = n / 1000
    return k >= 10 ? `${Math.round(k)}K` : `${k.toFixed(1).replace(/\.0$/, "")}K`
  }
  return String(n)
}

const barColorFor = (ratio: number): string => {
  if (ratio >= 0.95) return "bg-danger"
  if (ratio >= 0.8) return "bg-warn"
  return "bg-primary"
}

export const TokenMeter: React.FC<TokenMeterProps> = ({
  used,
  max,
  showUnit = true,
  className,
}) => {
  const ratio = max > 0 ? Math.min(1, Math.max(0, used / max)) : 0
  const percent = ratio * 100
  const color = barColorFor(ratio)

  return (
    <span
      className={`inline-flex items-center font-mono text-[11px] text-text-subtle${
        className ? ` ${className}` : ""
      }`}
      aria-label={`${used} of ${formatCount(max)} tokens used`}
    >
      <span>{used}</span>
      <span
        className="relative inline-block w-16 h-[3px] mx-1.5 align-middle bg-border rounded-sm overflow-hidden"
        aria-hidden="true"
      >
        <span
          className={`absolute inset-y-0 left-0 ${color} rounded-sm`}
          style={{ width: `${percent}%` }}
        />
      </span>
      <span>
        {formatCount(max)}
        {showUnit ? " tok" : ""}
      </span>
    </span>
  )
}
