import React from "react"
import { FacetRow, type FacetSpec } from "../shared/FacetRow"
import { IconButton } from "../shared/IconButton"
import { SendButton } from "../shared/SendButton"
import { TokenMeter } from "../shared/TokenMeter"
import {
  SlashPalette,
  type SlashPaletteGroup,
  type SlashPaletteRow,
} from "../shared/SlashPalette"

/**
 * V5 · Radial Command — palette-first, single-line pill composer. Every
 * capability (model, persona, temperature, sources, attachments, prompts,
 * etc.) is reachable via the slash palette that opens above the composer
 * the moment the user types `/`. The `⌘K` button is a visual affordance
 * for the same palette.
 *
 * Structure (designer source: composer-redesign.html lines 566–672):
 *
 *   ┌ facets (above box) ─────────────────────────────────────────┐
 *   │ src irb-archive · 14  mdl haiku-4-5  tmp 0.7  per Hoffman …│
 *   └──────────────────────────────────────────────────────────────┘
 *   ╔═ v5-box  (rounded pill, focus: border-primary + glow) ═╗
 *   ║  >_  <textarea …>  [⌘K]  [⎙]  [◉]  (↩) ║
 *   ╚════════════════════════════════════════════════════════╝
 *
 * When `paletteOpen` is true, the inline `SlashPalette` renders
 * absolutely-positioned above the composer — NOT the centered ⌘K modal.
 *
 * Compact density (extension sidepanel) tightens everything but keeps
 * the pill-and-palette pattern recognisable at ~360px.
 */

export interface RadialCommandV5IconButton {
  id: string
  label: string
  icon: React.ReactNode
  active?: boolean
  pressed?: boolean
  onClick?: () => void
}

export interface RadialCommandV5Props {
  // --- Text ---
  message: string
  onMessageChange: (value: string) => void
  placeholder?: string
  textareaRef?: React.RefObject<HTMLTextAreaElement>
  /**
   * Parent-provided keydown handler. Runs **before** V5's built-in
   * Cmd/Ctrl+Enter → onSend. Call `e.preventDefault()` to suppress the
   * default. Plain Enter inserts a newline.
   */
  onKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void

  // --- Send ---
  onSend: () => void
  sending?: boolean
  stopStreaming?: () => void
  canSend?: boolean

  // --- Slash palette (inline, above the composer) ---
  paletteOpen?: boolean
  paletteGroups?: SlashPaletteGroup[]
  paletteActiveIndex?: number
  onPaletteActiveIndexChange?: (index: number) => void
  paletteQuery?: string
  onPaletteSelect?: (row: SlashPaletteRow) => void
  paletteMatchCountLabel?: string
  /** Called when the ⌘K affordance is clicked. Surface typically opens the palette. */
  onPaletteTrigger?: () => void

  // --- Facets (meta row above the composer pill, simple API) ---
  /** Facet chips above the pill. Ignored when `facetsSlot` is provided. */
  facets?: FacetSpec[]
  /** Token meter (renders trailing in the facet row). Ignored when `facetsSlot` is provided. */
  tokens?: { used: number; max: number }

  // --- Icon buttons (inline, right of textarea; simple API) ---
  /** Ignored when `inlineSlot` is provided. */
  iconButtons?: RadialCommandV5IconButton[]

  // --- Slot overrides (power-user API for surface wire-up) ---
  /**
   * Full replacement for the facet row above the pill. When provided,
   * `facets` and `tokens` are ignored. Use for interactive metadata
   * widgets (model pickers, persona switchers) the simple Facet chips
   * can't represent.
   */
  facetsSlot?: React.ReactNode
  /**
   * Replaces the inline content between the textarea and the send
   * button (normally: ⌘K trigger + iconButtons). When provided,
   * `iconButtons` and the built-in ⌘K button are ignored.
   */
  inlineSlot?: React.ReactNode
  /** Replaces just the round SendButton. */
  sendSlot?: React.ReactNode
  /** Rendered above the facets row — warnings, notices. */
  noticesSlot?: React.ReactNode
  /**
   * Replaces the built-in textarea (and the `>_` caret column). When
   * provided, `message`, `onMessageChange`, `placeholder`, `textareaRef`,
   * and `onKeyDown` are ignored — the caller owns text input entirely.
   * Use to drop in a richer textarea component (slash menu, mentions,
   * paste handling) like Playground's `ComposerTextarea`.
   */
  textareaSlot?: React.ReactNode

  // --- Layout ---
  density?: "desktop" | "compact"
  forceFocused?: boolean
}

export const RadialCommandV5: React.FC<RadialCommandV5Props> = ({
  message,
  onMessageChange,
  placeholder = "Ask anything · / for commands",
  textareaRef,
  onKeyDown,
  onSend,
  sending = false,
  stopStreaming,
  canSend = true,
  paletteOpen = false,
  paletteGroups = [],
  paletteActiveIndex = 0,
  onPaletteActiveIndexChange,
  paletteQuery = "",
  onPaletteSelect,
  paletteMatchCountLabel,
  onPaletteTrigger,
  facets = [],
  tokens,
  iconButtons = [],
  facetsSlot,
  inlineSlot,
  sendSlot,
  noticesSlot,
  textareaSlot,
  density = "desktop",
  forceFocused = false,
}) => {
  const compact = density === "compact"
  const wrapperPad = compact ? "p-2.5" : "px-6 pt-3.5 pb-5"
  const boxFocusCls = forceFocused
    ? "border-primary [box-shadow:var(--glow-primary)]"
    : "focus-within:border-primary focus-within:[box-shadow:var(--glow-primary)]"

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    onKeyDown?.(e)
    if (e.defaultPrevented) return
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      if (canSend && !sending) onSend()
    }
  }

  const handleSend = () => {
    if (sending) {
      stopStreaming?.()
      return
    }
    if (canSend) onSend()
  }

  const hasDefaultFacets = facets.length > 0 || tokens

  return (
    <div
      className={`${wrapperPad} border-t border-border bg-bg relative`}
      data-variant="v5"
      data-density={density}
    >
      {noticesSlot && <div className="mb-2">{noticesSlot}</div>}
      {paletteOpen && onPaletteSelect && onPaletteActiveIndexChange && (
        <SlashPalette
          open={paletteOpen}
          groups={paletteGroups}
          activeIndex={paletteActiveIndex}
          onActiveIndexChange={onPaletteActiveIndexChange}
          query={paletteQuery}
          onSelect={onPaletteSelect}
          matchCountLabel={paletteMatchCountLabel}
        />
      )}

      {facetsSlot ??
        (hasDefaultFacets ? (
          <FacetRow
            facets={facets}
            trailing={
              tokens && (
                <TokenMeter
                  used={tokens.used}
                  max={tokens.max}
                  showUnit={!compact}
                />
              )
            }
            aria-label="Composer facets"
          />
        ) : null)}

      <div
        className={`bg-surface border border-border rounded-full flex items-center gap-2.5 transition ${boxFocusCls} ${
          compact ? "px-3 py-1 mt-2" : "px-4 py-1 mt-2.5"
        }`}
      >
        {textareaSlot ? (
          <div className="flex-1 min-w-0">{textareaSlot}</div>
        ) : (
          <>
            <span
              className="font-mono text-primary text-sm select-none flex-shrink-0"
              aria-hidden="true"
            >
              &gt;_
            </span>
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => onMessageChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              className={`flex-1 bg-transparent text-text border-0 outline-none resize-none font-sans leading-relaxed py-1.5 ${
                compact ? "text-[13px]" : "text-sm"
              } min-h-[22px]`}
              aria-label="Message"
            />
          </>
        )}
        {inlineSlot ?? (
          <>
            {onPaletteTrigger && (
              <button
                type="button"
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary border border-primary/40 font-mono text-[11px] hover:bg-primary/15 flex-shrink-0"
                onClick={onPaletteTrigger}
                aria-label="Open command palette"
              >
                <span aria-hidden="true">⌘K</span>
              </button>
            )}
            {iconButtons.map((btn) => (
              <IconButton
                key={btn.id}
                label={btn.label}
                active={btn.active}
                pressed={btn.pressed}
                onClick={btn.onClick}
                density={density}
              >
                {btn.icon}
              </IconButton>
            ))}
          </>
        )}
        {sendSlot ?? (
          <SendButton
            onClick={handleSend}
            disabled={!canSend && !sending}
            stopping={sending}
            shape="round"
            density={density}
          />
        )}
      </div>
    </div>
  )
}
