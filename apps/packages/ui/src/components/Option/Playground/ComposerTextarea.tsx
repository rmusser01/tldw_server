import React from "react"
import { useTranslation } from "react-i18next"
import {
  SlashCommandMenu,
  type SlashCommandItem
} from "@/components/Sidepanel/Chat/SlashCommandMenu"
import { MentionsDropdown } from "./MentionsDropdown"

export type ComposerTextareaProps = {
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
  attachTextarea?: React.RefCallback<HTMLTextAreaElement>
  value: string
  displayValue: string
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  onPaste: (e: React.ClipboardEvent<HTMLTextAreaElement>) => void
  onFocus: () => void
  onSelect: () => void
  onCompositionStart: () => void
  onCompositionEnd: () => void
  onMouseDown: () => void
  onMouseUp: () => void
  placeholder: string
  disabled?: boolean
  isProMode: boolean
  isMobile: boolean
  isConnectionReady: boolean
  isCollapsed: boolean
  compactWhenInactive?: boolean
  rows?: number
  // Form bindings
  formInputProps: Record<string, any>
  // Slash commands
  showSlashMenu: boolean
  slashCommands: SlashCommandItem[]
  slashActiveIndex: number
  onSlashSelect: (cmd: SlashCommandItem) => void
  onSlashActiveIndexChange: (idx: number) => void
  slashEmptyLabel: string
  // Mentions
  showMentions: boolean
  filteredTabs: any[]
  mentionPosition: any
  onMentionSelect: (tab: any) => void
  onMentionsClose: () => void
  onMentionRefetch: () => Promise<void>
  onMentionsOpen: () => Promise<void>
  // Draft
  draftSaved: boolean
}

export const ComposerTextarea = React.memo(function ComposerTextarea({
  textareaRef,
  attachTextarea,
  value,
  displayValue,
  onChange,
  onKeyDown,
  onPaste,
  onFocus,
  onSelect,
  onCompositionStart,
  onCompositionEnd,
  onMouseDown,
  onMouseUp,
  placeholder,
  isProMode,
  isMobile,
  isCollapsed,
  compactWhenInactive = false,
  rows = 1,
  formInputProps,
  showSlashMenu,
  slashCommands,
  slashActiveIndex,
  onSlashSelect,
  onSlashActiveIndexChange,
  slashEmptyLabel,
  showMentions,
  filteredTabs,
  mentionPosition,
  onMentionSelect,
  onMentionsClose,
  onMentionRefetch,
  onMentionsOpen,
  draftSaved
}: ComposerTextareaProps) {
  const { t } = useTranslation(["sidepanel"])
  const minHeight = isMobile
    ? "40px"
    : compactWhenInactive
      ? isProMode
        ? "44px"
        : "40px"
      : isProMode
        ? "60px"
        : "44px"

  return (
    <div className="relative min-w-0 px-1 pb-2 pt-1">
      <SlashCommandMenu
        open={showSlashMenu}
        commands={slashCommands}
        activeIndex={slashActiveIndex}
        onActiveIndexChange={onSlashActiveIndexChange}
        onSelect={onSlashSelect}
        emptyLabel={slashEmptyLabel}
        className="relative mx-3 mb-2"
      />
      <textarea
        id="textarea-message"
        data-testid="chat-input"
        onCompositionStart={onCompositionStart}
        onCompositionEnd={onCompositionEnd}
        onMouseDown={onMouseDown}
        onMouseUp={onMouseUp}
        onKeyDown={(e) => {
          try {
            onKeyDown(e)
          } catch (err) {
            console.error("[ComposerTextarea] onKeyDown error:", err)
          }
        }}
        onFocus={onFocus}
        ref={attachTextarea ?? textareaRef}
        className={`w-full resize-none bg-transparent text-base leading-6 text-text placeholder:text-text-muted/80 focus-within:outline-none focus:ring-0 focus-visible:ring-0 ring-0 border-0 ${isProMode ? "px-3 py-2.5" : "px-3 py-2"}`}
        onPaste={(e) => {
          try {
            onPaste(e)
          } catch (err) {
            console.error("[ComposerTextarea] onPaste error:", err)
          }
        }}
        rows={rows}
        style={{
          minHeight
        }}
        tabIndex={0}
        placeholder={placeholder}
        aria-label={t("sidepanel:composer.messageAriaLabel", "Message input")}
        {...formInputProps}
        value={displayValue}
        onChange={onChange}
        onSelect={onSelect}
      />

      <MentionsDropdown
        show={showMentions}
        tabs={filteredTabs}
        mentionPosition={mentionPosition}
        onSelectTab={onMentionSelect}
        onClose={onMentionsClose}
        textareaRef={textareaRef}
        refetchTabs={onMentionRefetch}
        onMentionsOpen={onMentionsOpen}
      />
      {/* Draft saved indicator */}
      {draftSaved && (
        <span
          className="absolute bottom-1 right-2 text-label text-text-muted transition-opacity pointer-events-none"
          role="status"
          aria-live="polite"
        >
          {t("sidepanel:composer.draftSaved", "Draft saved")}
        </span>
      )}
    </div>
  )
})
