import React from "react"
import { Pill } from "../shared/Pill"
import { SourceChip } from "../shared/SourceChip"
import { IconButton } from "../shared/IconButton"
import { TokenMeter } from "../shared/TokenMeter"
import { SendButton } from "../shared/SendButton"

/**
 * V1 · Terminal Stack — one of three user-selectable composer variants
 * from the Primer design. Presentational-only: all state lives in the
 * hook layer (`useComposerText`, `useComposerSubmit`, etc.) and flows
 * through props.
 *
 * Structure (designer source: composer-redesign.html lines 234–320):
 *
 *   ╔═ v1-box (focus: border-primary + glow-primary) ═══════════╗
 *   ║  v1-rail    [14 ▣ sources] [+source] [Web] [MCP] [RAG]    ║
 *   ║  v1-prompt  >_  <textarea …>                              ║
 *   ║  v1-bar     [haiku-4-5] [0.7] [OCR] ⎙ ◉ ✿ /   127▓8K Send ║
 *   ╚═══════════════════════════════════════════════════════════╝
 *
 * Density:
 *   - desktop  — full bar with kbd hints, Send label + ⌘↩
 *   - compact  — hides kbd hints, send button shrinks to ↩ glyph; for
 *                extension sidepanel (~360px container).
 */

export interface ChipSpec {
  id: string
  label: string
  active?: boolean
  variant?: "default" | "on" | "accent"
  onClick?: () => void
}

export interface IconButtonSpec {
  id: string
  label: string
  icon: React.ReactNode
  active?: boolean
  pressed?: boolean
  onClick?: () => void
}

export interface SourceChipSpec {
  count: number
  label: string
  glyph?: string
  onClick?: () => void
}

export interface TerminalStackV1Props {
  // --- Text ---
  message: string
  onMessageChange: (value: string) => void
  placeholder?: string
  textareaRef?: React.RefObject<HTMLTextAreaElement>
  /**
   * Parent-provided keydown handler. Runs **before** V1's built-in
   * Cmd/Ctrl+Enter → onSend handler. Call `e.preventDefault()` to suppress
   * V1's default; otherwise V1 adds send-on-modifier-enter on top. Plain
   * Enter is never intercepted — it passes through to insert a newline.
   */
  onKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void

  // --- Send ---
  onSend: () => void
  sending?: boolean
  stopStreaming?: () => void
  canSend?: boolean

  // --- Top rail (simple API) ---
  /** Primary source chip. Ignored when `topSlot` is provided. */
  sourceChip?: SourceChipSpec
  /** Additional top-rail pills. Ignored when `topSlot` is provided. */
  topChips?: ChipSpec[]

  // --- Bottom bar (simple API) ---
  /** Model / parameter pills on the left of the bottom bar. Ignored when `bottomBarSlot` is provided. */
  bottomChips?: ChipSpec[]
  /** Icon buttons. Ignored when `bottomBarSlot` is provided. */
  iconButtons?: IconButtonSpec[]
  /** Token meter. Omit to hide. Ignored when `bottomBarSlot` is provided. */
  tokens?: { used: number; max: number }

  // --- Slot overrides (power-user API for surface wire-up) ---
  /**
   * Full replacement for the top-rail content. When provided, `sourceChip`
   * and `topChips` are ignored and this node is rendered in their place.
   * Use when a surface needs interactive widgets the simple-API chips
   * can't represent (model dropdowns, MCP popovers, compare multi-select).
   */
  topSlot?: React.ReactNode
  /**
   * Full replacement for the bottom-bar content. When provided,
   * `bottomChips`, `iconButtons`, `tokens`, and the default SendButton are
   * all ignored — the entire bottom strip (including your own send
   * button) is the caller's responsibility. Use to drop in a fully-
   * featured toolbar like Playground's ComposerToolbar.
   */
  bottomBarSlot?: React.ReactNode
  /**
   * Replaces just the built-in SendButton. Ignored when `bottomBarSlot`
   * is set (caller owns the whole bar in that case).
   */
  sendSlot?: React.ReactNode
  /**
   * Rendered above the composer box. Use for warnings, notices,
   * validation errors, compare-mode banners, context-budget callouts.
   */
  noticesSlot?: React.ReactNode
  /**
   * Rendered inside the focus box, absolutely positioned to fill it.
   * Use for popover overlays that anchor to the composer: SlashCommandMenu,
   * MentionsMenu, draft-saved toast.
   */
  overlaysSlot?: React.ReactNode
  /**
   * Replaces the built-in textarea (and the `>_` caret column). When
   * provided, `message`, `onMessageChange`, `placeholder`, `textareaRef`,
   * and `onKeyDown` are ignored — the caller owns text input entirely.
   * Use this to drop in a richer textarea component (paste-collapse,
   * mention/slash trigger detection, perf tracking) like Playground's
   * `ComposerTextarea`.
   */
  textareaSlot?: React.ReactNode

  // --- Layout ---
  density?: "desktop" | "compact"
  /** Controlled focus-visible state for Storybook/tests. */
  forceFocused?: boolean
}

export const TerminalStackV1: React.FC<TerminalStackV1Props> = ({
  message,
  onMessageChange,
  placeholder = "Ask the Primer…",
  textareaRef,
  onKeyDown,
  onSend,
  sending = false,
  stopStreaming,
  canSend = true,
  sourceChip,
  topChips = [],
  bottomChips = [],
  iconButtons = [],
  tokens,
  topSlot,
  bottomBarSlot,
  sendSlot,
  noticesSlot,
  overlaysSlot,
  textareaSlot,
  density = "desktop",
  forceFocused = false,
}) => {
  const compact = density === "compact"
  const wrapperPad = compact ? "p-2.5" : "px-6 pt-4 pb-5"
  const boxFocusCls = forceFocused
    ? "border-primary [box-shadow:var(--glow-primary)]"
    : "focus-within:border-primary focus-within:[box-shadow:var(--glow-primary)]"

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    onKeyDown?.(e)
    if (e.defaultPrevented) return
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    if (sending) {
      stopStreaming?.()
      return
    }
    if (canSend) onSend()
  }

  // Top rail: slot takes precedence over simple-API chips.
  const hasSimpleTop = Boolean(sourceChip) || topChips.length > 0
  const topContent: React.ReactNode = topSlot ?? (hasSimpleTop ? (
    <>
      {sourceChip && (
        <SourceChip
          count={sourceChip.count}
          label={sourceChip.label}
          glyph={sourceChip.glyph}
          onClick={sourceChip.onClick}
        />
      )}
      {topChips.map((chip) => (
        <Pill
          key={chip.id}
          variant={chip.variant ?? (chip.active ? "on" : "default")}
          onClick={chip.onClick}
        >
          {chip.label}
        </Pill>
      ))}
    </>
  ) : null)

  return (
    <div
      className={`${wrapperPad} border-t border-border bg-bg`}
      data-variant="v1"
      data-density={density}
    >
      {noticesSlot && <div className="mb-2">{noticesSlot}</div>}
      <div
        className={`relative bg-surface border border-border rounded-lg transition ${boxFocusCls}`}
      >
        {overlaysSlot}
        {topContent && (
          <div
            className={`flex items-center gap-2.5 flex-wrap ${
              compact ? "px-2 pt-1.5" : "px-3 pt-2"
            }`}
            data-testid="v1-top-rail"
          >
            {topContent}
          </div>
        )}

        {textareaSlot ? (
          <div
            className={compact ? "px-2.5 pt-2 pb-1" : "px-3.5 pt-2.5 pb-1"}
            data-testid="v1-textarea-region"
          >
            {textareaSlot}
          </div>
        ) : (
          <div
            className={`flex gap-2.5 items-start ${
              compact ? "px-2.5 pt-2 pb-1" : "px-3.5 pt-2.5 pb-1"
            }`}
            data-testid="v1-textarea-region"
          >
            <span
              className="font-mono text-primary text-sm leading-6 pt-0.5 tracking-tighter select-none"
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
              className={`flex-1 bg-transparent text-text border-0 outline-none resize-none font-sans leading-relaxed py-0.5 ${
                compact ? "text-[13px] min-h-[44px]" : "text-sm min-h-[46px]"
              }`}
              aria-label="Message"
            />
          </div>
        )}

        <div
          className={`flex items-center gap-1.5 flex-wrap border-t border-border ${
            compact ? "px-1.5 py-1.5" : "px-2.5 py-2"
          }`}
          data-testid="v1-bottom-bar"
        >
          {bottomBarSlot ?? (
            <>
              {bottomChips.map((chip) => (
                <Pill
                  key={chip.id}
                  variant={chip.variant ?? (chip.active ? "on" : "default")}
                  onClick={chip.onClick}
                >
                  {chip.label}
                </Pill>
              ))}
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
              <span className="flex-1" />
              {tokens && (
                <TokenMeter
                  used={tokens.used}
                  max={tokens.max}
                  showUnit={!compact}
                />
              )}
              {sendSlot ?? (
                <SendButton
                  onClick={handleSend}
                  disabled={!canSend && !sending}
                  stopping={sending}
                  density={density}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
