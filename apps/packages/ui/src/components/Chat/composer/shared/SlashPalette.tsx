import React from "react"

/**
 * Inline slash-command palette used by V5 Radial Command. Positioned
 * ABOVE the composer (absolute + bottom anchored) — NOT a centered modal.
 * Distinct from the global `CommandPalette` (⌘K modal) in two ways:
 *
 *   1. Always inline, attached visually to the composer surface.
 *   2. Richer rows: icon tile + cmd + hint + optional kbd hint, with
 *      grouped sections showing result counts ("Models · 6 results").
 *
 * This primitive is presentation-only. The parent owns:
 *   - The filtered command list (grouped)
 *   - The active-row index (keyboard navigation)
 *   - Open/close state
 *   - Query tracking (what the user has typed after `/`)
 *
 * Phase-0 spike 3 concluded the existing `SlashCommandMenu` (78 LOC,
 * simple dropdown) is the right evolution target for this shape. V5's
 * richer palette builds on the same mental model with extra props.
 */

export interface SlashPaletteRow {
  id: string
  /** Icon tile glyph (renders in a bordered square). */
  icon?: React.ReactNode
  /** The command itself, e.g. "/model haiku-4-5". */
  command: string
  /** Short contextual description. */
  hint?: string
  /** Optional key hint, e.g. "↩". Rendered right-aligned. */
  kbd?: string
  onSelect?: () => void
}

export interface SlashPaletteGroup {
  id: string
  /** Section header, e.g. "Models · 6 results". */
  label: string
  rows: SlashPaletteRow[]
}

export interface SlashPaletteProps {
  /** Controls whether the palette renders at all. */
  open: boolean
  /** Grouped command rows. */
  groups: SlashPaletteGroup[]
  /** Index of the currently-highlighted row across all groups. */
  activeIndex: number
  onActiveIndexChange: (index: number) => void
  /** Full query string the user typed after `/`. */
  query: string
  /** Called when a row is selected (click or keyboard). */
  onSelect: (row: SlashPaletteRow) => void
  /** Empty-state copy when no rows match. Defaults to "No commands found". */
  emptyLabel?: string
  /** Match-count copy shown in the footer, e.g. "14 commands matched". */
  matchCountLabel?: string
  /** Additional classes (e.g. positioning overrides). */
  className?: string
}

export const SlashPalette: React.FC<SlashPaletteProps> = ({
  open,
  groups,
  activeIndex,
  onActiveIndexChange,
  query,
  onSelect,
  emptyLabel = "No commands found",
  matchCountLabel,
  className,
}) => {
  if (!open) return null

  const flatRows = groups.flatMap((group) => group.rows)
  const totalRows = flatRows.length

  return (
    <div
      className={`absolute left-6 right-6 bottom-[calc(100%+10px)] z-10 bg-elevated border border-border-strong rounded-lg shadow-md overflow-hidden${
        className ? ` ${className}` : ""
      }`}
      role="listbox"
      aria-label="Composer slash commands"
    >
      <div className="px-3.5 py-3 border-b border-border font-mono text-xs text-text">
        <span className="text-primary mr-1">/</span>
        <span className="text-text">{query}</span>
        <span
          className="inline-block w-[7px] h-[12px] ml-0.5 bg-primary align-middle animate-pulse"
          aria-hidden="true"
        />
      </div>

      {totalRows === 0 ? (
        <div className="px-3.5 py-4 text-xs text-text-subtle">
          {emptyLabel}
        </div>
      ) : (
        <div>
          {(() => {
            let runningIndex = 0
            return groups.map((group) => (
              <div key={group.id}>
                <div className="px-2.5 pt-2 pb-1.5 font-mono text-[10px] text-text-subtle uppercase tracking-wider">
                  {group.label}
                </div>
                {group.rows.map((row) => {
                  const rowIndex = runningIndex++
                  const selected = rowIndex === activeIndex
                  const rowCls =
                    "w-full flex items-center gap-2.5 px-3.5 py-2 cursor-pointer font-sans text-[13px] text-left transition-colors" +
                    (selected
                      ? " bg-primary/10 [box-shadow:inset_2px_0_0_rgb(var(--color-primary))]"
                      : " hover:bg-primary/5")
                  return (
                    <button
                      type="button"
                      key={row.id}
                      role="option"
                      aria-selected={selected}
                      data-selected={selected}
                      className={rowCls}
                      onClick={() => onSelect(row)}
                      onMouseEnter={() => onActiveIndexChange(rowIndex)}
                    >
                      {row.icon && (
                        <span
                          className="inline-flex items-center justify-center w-[22px] h-[22px] font-mono text-[11px] text-primary border border-primary/30 rounded-sm flex-shrink-0"
                          aria-hidden="true"
                        >
                          {row.icon}
                        </span>
                      )}
                      <span className="font-mono text-xs text-text min-w-[130px]">
                        {row.command}
                      </span>
                      {row.hint && (
                        <span className="text-text-muted flex-1 text-xs truncate">
                          {row.hint}
                        </span>
                      )}
                      {row.kbd && (
                        <span className="ml-auto font-mono text-[10px] text-text-subtle border border-border bg-surface px-1.5 py-0.5 rounded-sm">
                          {row.kbd}
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>
            ))
          })()}
        </div>
      )}

      <div className="flex gap-4 px-3 py-2 border-t border-border font-mono text-[10px] text-text-subtle uppercase tracking-wider">
        <span>↑↓ navigate</span>
        <span>↩ run</span>
        <span>⌘↩ run + send</span>
        <span>esc close</span>
        {matchCountLabel && (
          <span className="ml-auto">{matchCountLabel}</span>
        )}
      </div>
    </div>
  )
}
