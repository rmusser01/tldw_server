import React from "react"
import { BriefField } from "../shared/BriefField"
import { IconButton } from "../shared/IconButton"
import { TokenMeter } from "../shared/TokenMeter"
import { SendButton } from "../shared/SendButton"

/**
 * V3 · Split Brief — researcher-oriented composer variant. Left pane is
 * the "brief" (persona, sources, model, temperature, tool config) as
 * labelled key/value chips; right pane is the question.
 *
 * Structure (designer source: composer-redesign.html lines 403–482):
 *
 *   ╔═ v3-box ════════════════════════════════════════════════════════╗
 *   ║  Brief          │                                               ║
 *   ║  [src] irb · 14 │   ┌──────────────────────────────────────┐    ║
 *   ║  [mdl] haiku    │   │ <textarea…>                          │    ║
 *   ║  [tmp] 0.7      │   │                                      │    ║
 *   ║  [per] Hoffman  │   └──────────────────────────────────────┘    ║
 *   ║  ...            │   [⎙ ◉ / @]    ← ◎ 284▓8K · $0.003  Send ⌘↩ ║
 *   ╚═════════════════╧═══════════════════════════════════════════════╝
 *
 * Compact density (for extension sidepanel ≤360px) collapses the brief
 * into a horizontal chip strip above the textarea; field keys are
 * hidden and only the values remain.
 */

export interface BriefFieldSpec {
  id: string
  /** Short key label shown on desktop: "src", "mdl", "tmp", etc. */
  fieldKey?: string
  /** Right-hand value — typically a label string with an optional glyph. */
  value: React.ReactNode
  active?: boolean
  onClick?: () => void
  /** Screen-reader-only label for icon-only values. */
  "aria-label"?: string
}

export interface BriefSectionSpec {
  id: string
  /** Section header — e.g. "Brief" or "Prompts". Only shown at desktop. */
  label?: string
  fields: BriefFieldSpec[]
}

export interface IconButtonSpec {
  id: string
  label: string
  icon: React.ReactNode
  active?: boolean
  pressed?: boolean
  onClick?: () => void
}

export interface SplitBriefV3Props {
  // --- Text ---
  message: string
  onMessageChange: (value: string) => void
  placeholder?: string
  textareaRef?: React.RefObject<HTMLTextAreaElement>
  /**
   * Parent-provided keydown handler. Runs before V3's built-in
   * Cmd/Ctrl+Enter → onSend handler. Call `e.preventDefault()` to
   * suppress V3's default. Plain Enter is never intercepted.
   */
  onKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void

  // --- Send ---
  onSend: () => void
  sending?: boolean
  stopStreaming?: () => void
  canSend?: boolean

  // --- Brief panel (simple API) ---
  /**
   * Labelled field chips for the brief. Ignored when `briefSlot` is
   * provided. Defaults to `[]` so V3 renders an empty brief panel
   * gracefully when a caller hasn't yet adapted its data — important
   * for the dispatcher pattern where a user may pick V3 in settings
   * before the surface has provided brief data.
   */
  briefSections?: BriefSectionSpec[]

  // --- Bottom bar (simple API) ---
  /** Icon buttons. Ignored when `bottomBarSlot` is provided. */
  iconButtons?: IconButtonSpec[]
  /** Token meter data. Ignored when `bottomBarSlot` is provided. */
  tokens?: { used: number; max: number }
  /**
   * Optional cost estimate rendered adjacent to the token meter,
   * e.g. "≈ $0.003". Ignored when `bottomBarSlot` is provided.
   */
  costLabel?: string

  // --- Slot overrides (power-user API for surface wire-up) ---
  /**
   * Full replacement for the left brief panel. When provided,
   * `briefSections` is ignored. Use when the surface wants to host
   * interactive widgets (persona dropdown, model picker) that the
   * field-chip API can't represent.
   */
  briefSlot?: React.ReactNode
  /**
   * Full replacement for the bottom bar. When provided, `iconButtons`,
   * `tokens`, `costLabel`, and the default SendButton are ignored —
   * the entire bottom strip (including the send button) is the
   * caller's responsibility.
   */
  bottomBarSlot?: React.ReactNode
  /**
   * Replaces just the built-in SendButton. Ignored when
   * `bottomBarSlot` is set.
   */
  sendSlot?: React.ReactNode
  /** Rendered above the composer box — warnings, notices, errors. */
  noticesSlot?: React.ReactNode
  /**
   * Rendered inside the focus box, absolutely positioned. Use for
   * popover overlays (SlashCommandMenu, MentionsMenu).
   */
  overlaysSlot?: React.ReactNode
  /**
   * Replaces the right-pane textarea entirely. When provided, `message`,
   * `onMessageChange`, `placeholder`, `textareaRef`, and `onKeyDown` are
   * ignored — the caller owns text input. Drop in a richer textarea
   * (paste-collapse, mention/slash detection, perf tracking) like
   * Playground's `ComposerTextarea`.
   */
  textareaSlot?: React.ReactNode

  // --- Layout ---
  density?: "desktop" | "compact"
  forceFocused?: boolean
}

export const SplitBriefV3: React.FC<SplitBriefV3Props> = ({
  message,
  onMessageChange,
  placeholder = "Your question…",
  textareaRef,
  onKeyDown,
  onSend,
  sending = false,
  stopStreaming,
  canSend = true,
  briefSections = [],
  iconButtons = [],
  tokens,
  costLabel,
  briefSlot,
  bottomBarSlot,
  sendSlot,
  noticesSlot,
  overlaysSlot,
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

  // Desktop: left brief column, right question column
  // Compact:  horizontal brief strip (single row of value-only chips),
  //           question area below
  const boxLayoutCls = compact
    ? "flex flex-col"
    : "grid grid-cols-[240px_1fr]"

  // Brief panel — slot takes precedence over briefSections.
  const defaultBrief = compact ? (
    <div
      className="flex items-center gap-1.5 px-2 py-2 border-b border-border bg-surface2/40 overflow-x-auto"
      role="group"
      aria-label="Brief"
    >
      {briefSections.flatMap((section) =>
        section.fields.map((field) => (
          <BriefField
            key={`${section.id}:${field.id}`}
            fieldKey={field.fieldKey}
            value={field.value}
            active={field.active}
            onClick={field.onClick}
            hideKey
            aria-label={field["aria-label"]}
          />
        ))
      )}
    </div>
  ) : (
    <div
      role="group"
      className="border-r border-border bg-surface2/40 p-3.5 flex flex-col gap-2.5"
      aria-label="Brief"
    >
      {briefSections.map((section, idx) => (
        <React.Fragment key={section.id}>
          {section.label && (
            <div
              className={`font-mono text-[10px] text-text-subtle uppercase tracking-wider${
                idx > 0 ? " mt-1" : ""
              }`}
            >
              {section.label}
            </div>
          )}
          {section.fields.map((field) => (
            <BriefField
              key={field.id}
              fieldKey={field.fieldKey}
              value={field.value}
              active={field.active}
              onClick={field.onClick}
              aria-label={field["aria-label"]}
            />
          ))}
        </React.Fragment>
      ))}
    </div>
  )

  return (
    <div
      className={`${wrapperPad} border-t border-border bg-bg`}
      data-variant="v3"
      data-density={density}
    >
      {noticesSlot && <div className="mb-2">{noticesSlot}</div>}
      <div
        className={`${boxLayoutCls} relative bg-surface border border-border rounded-lg overflow-hidden transition ${boxFocusCls}`}
      >
        {overlaysSlot}
        {briefSlot ?? defaultBrief}

        {/* Question panel */}
        <div className="flex flex-col" data-testid="v3-question-pane">
          {textareaSlot ?? (
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => onMessageChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              className={`w-full bg-transparent text-text border-0 outline-none resize-none font-sans leading-relaxed flex-1 ${
                compact
                  ? "px-3 py-2.5 text-[13px] min-h-[70px]"
                  : "px-4 py-3.5 text-sm min-h-[100px]"
              }`}
              aria-label="Question"
            />
          )}
          <div
            className={`flex items-center gap-2 border-t border-border ${
              compact ? "px-1.5 py-1.5" : "px-2.5 py-2"
            }`}
            data-testid="v3-bottom-bar"
          >
            {bottomBarSlot ?? (
              <>
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
                {costLabel && !compact && (
                  <span className="font-mono text-[11px] text-text-subtle">
                    · {costLabel}
                  </span>
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
    </div>
  )
}
