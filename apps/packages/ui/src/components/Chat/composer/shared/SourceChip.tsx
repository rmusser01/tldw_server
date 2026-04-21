import React from "react"

/**
 * Source chip used in the V1 composer's top rail. Shows an integer badge
 * (sources attached) plus a glyph + name. Clickable — clicking opens the
 * source picker.
 */

export interface SourceChipProps {
  /** Integer badge shown in the small square. */
  count: number
  /** Human-readable collection/source name. */
  label: string
  /** Optional leading glyph — designer uses "▣". */
  glyph?: string
  onClick?: () => void
  className?: string
}

export const SourceChip: React.FC<SourceChipProps> = ({
  count,
  label,
  glyph = "▣",
  onClick,
  className,
}) => {
  const cls =
    "inline-flex items-center gap-1.5 pl-1.5 pr-2.5 py-1 rounded-full font-mono text-[11px] " +
    "text-primary border border-primary/35 bg-primary/10 hover:bg-primary/15 " +
    "transition-colors" +
    (onClick ? " cursor-pointer" : "") +
    (className ? ` ${className}` : "")

  const inner = (
    <>
      <span className="inline-flex items-center justify-center w-4 h-4 rounded-sm bg-primary text-[10px] font-bold text-bg">
        {count}
      </span>
      <span aria-hidden="true">{glyph}</span>
      <span>{label}</span>
    </>
  )

  if (onClick) {
    return (
      <button type="button" className={cls} onClick={onClick}>
        {inner}
      </button>
    )
  }
  return <span className={cls}>{inner}</span>
}
