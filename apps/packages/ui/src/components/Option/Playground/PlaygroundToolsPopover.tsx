import React from "react"
import {
  Switch,
  Radio,
  Tooltip,
  Popover
} from "antd"
import {
  ChevronDown,
  ChevronRight,
  EraserIcon,
  GitBranch,
  Globe,
  Headphones,
  ImageIcon,
  SlidersHorizontal,
  WandSparkles,
  FileText,
  Settings2,
  ArrowRight
} from "lucide-react"
import { Link } from "react-router-dom"
import { Button as TldwButton } from "@/components/Common/Button"
import type { KnowledgeTab } from "@/components/Knowledge"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlaygroundToolsPopoverProps {
  // Tools popover state
  toolsPopoverOpen: boolean
  onToolsPopoverChange: (open: boolean) => void
  isProMode: boolean

  // Attachments
  onOpenImageGenerate: () => void
  onOpenKnowledgePanel: (tab: KnowledgeTab) => void

  // OCR
  useOCR: boolean
  onUseOCRChange: (value: boolean) => void

  // Web search
  hasWebSearch: boolean
  webSearch: boolean
  onWebSearchChange: (checked: boolean) => void
  simpleInternetSearch: boolean
  onSimpleInternetSearchChange: (checked: boolean) => void
  defaultInternetSearchOn: boolean
  onDefaultInternetSearchOnChange: (checked: boolean) => void
  onNavigateWebSearchSettings: () => void

  // Advanced section
  advancedToolsExpanded: boolean
  onAdvancedToolsExpandedChange: (expanded: boolean) => void
  allowExternalImages: boolean
  onAllowExternalImagesChange: (checked: boolean) => void
  showMoodBadge: boolean
  onShowMoodBadgeChange: (checked: boolean) => void
  showMoodConfidence: boolean
  onShowMoodConfidenceChange: (checked: boolean) => void
  onOpenRawRequest: () => void

  // Voice
  voiceChatAvailable: boolean
  voiceChatUnavailableReason?: string | null
  voiceChatEnabled: boolean
  voiceChatState: string
  voiceChatStatusLabel: string
  onVoiceChatToggle: () => void
  isSending: boolean
  voiceChatSettingsFields: React.ReactNode

  // Image provider
  imageProviderControl: React.ReactNode

  // Footer actions
  historyLength: number
  onClearContext: () => void

  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundToolsPopover: React.FC<PlaygroundToolsPopoverProps> =
  React.memo(function PlaygroundToolsPopover(props) {
    const {
      toolsPopoverOpen,
      onToolsPopoverChange,
      isProMode,
      onOpenImageGenerate,
      onOpenKnowledgePanel,
      useOCR,
      onUseOCRChange,
      hasWebSearch,
      webSearch,
      onWebSearchChange,
      simpleInternetSearch,
      onSimpleInternetSearchChange,
      defaultInternetSearchOn,
      onDefaultInternetSearchOnChange,
      onNavigateWebSearchSettings,
      advancedToolsExpanded,
      onAdvancedToolsExpandedChange,
      allowExternalImages,
      onAllowExternalImagesChange,
      showMoodBadge,
      onShowMoodBadgeChange,
      showMoodConfidence,
      onShowMoodConfidenceChange,
      onOpenRawRequest,
      voiceChatAvailable,
      voiceChatUnavailableReason,
      voiceChatEnabled,
      voiceChatState,
      voiceChatStatusLabel,
      onVoiceChatToggle,
      isSending,
      voiceChatSettingsFields,
      imageProviderControl,
      historyLength,
      onClearContext,
      t
    } = props

    const content = (
      <div className="flex w-72 flex-col gap-2 p-1">
        {/* ATTACHMENTS Section */}
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider px-2">
            {t("playground:tools.attachments", "Attachments")}
          </span>
          <button
            type="button"
            onClick={() => {
              onToolsPopoverChange(false)
              onOpenImageGenerate()
            }}
            className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
          >
            <span>
              {t("playground:imageGeneration.modalTitle", "Generate image")}
            </span>
            <WandSparkles className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onOpenKnowledgePanel("context")}
            className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
          >
            <span>
              {t(
                "playground:attachments.manageContext",
                "Manage in Knowledge Panel"
              )}
            </span>
            <Settings2 className="h-4 w-4" />
          </button>
        </div>

        <div className="border-t border-border my-1" />

        <div className="flex items-center justify-between px-2">
          <span className="text-sm text-text">{t("playground:tools.useOCR", "Use OCR")}</span>
          <Switch
            size="small"
            checked={useOCR}
            onChange={(value) => onUseOCRChange(value)}
          />
        </div>

        <div className="border-t border-border my-1" />

        {/* WEB SEARCH Section */}
        <div className="flex flex-col gap-2">
          <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
            {t("playground:tools.webSearch", "Web Search")}
          </span>
          {hasWebSearch ? (
            <>
              <div className="flex items-center justify-between gap-2 px-2">
                <span className="flex items-center gap-2 text-sm text-text">
                  <Globe className="h-4 w-4 text-text-subtle" />
                  {t(
                    "playground:tools.webSearchEnabled",
                    "Enable web search"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={webSearch}
                  onChange={(checked) => onWebSearchChange(checked)}
                />
              </div>
              <div className="flex items-center justify-between gap-2 px-2">
                <span className="text-sm text-text">
                  {t(
                    "playground:tools.webSearchSimpleMode",
                    "Simple search mode"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={simpleInternetSearch}
                  onChange={(checked) =>
                    onSimpleInternetSearchChange(checked)
                  }
                />
              </div>
              <div className="flex items-center justify-between gap-2 px-2">
                <span className="text-sm text-text">
                  {t(
                    "playground:tools.webSearchDefaultOn",
                    "Default on for new chats"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={defaultInternetSearchOn}
                  onChange={(checked) =>
                    onDefaultInternetSearchOnChange(checked)
                  }
                />
              </div>
              <button
                type="button"
                onClick={() => {
                  onToolsPopoverChange(false)
                  onNavigateWebSearchSettings()
                }}
                className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
              >
                <span>
                  {t(
                    "playground:tools.webSearchOpenSettings",
                    "Open web search settings"
                  )}
                </span>
                <ArrowRight className="h-4 w-4" />
              </button>
            </>
          ) : (
            <p className="px-2 text-xs text-text-muted">
              {t(
                "playground:tools.webSearchUnavailable",
                "Web search is unavailable on this server."
              )}
            </p>
          )}
        </div>

        <div className="border-t border-border my-1" />

        {/* ADVANCED Section (collapsible) */}
        <button
          type="button"
          onClick={() =>
            onAdvancedToolsExpandedChange(!advancedToolsExpanded)
          }
          className="flex items-center justify-between px-2 py-1 text-[10px] font-semibold uppercase text-text-muted tracking-wider hover:text-text transition"
        >
          <span>{t("playground:tools.advanced", "Advanced")}</span>
          <ChevronRight
            className={`h-3 w-3 transition-transform ${advancedToolsExpanded ? "rotate-90" : ""}`}
          />
        </button>

        {advancedToolsExpanded && (
          <div className="flex flex-col gap-2">
            <div className="flex flex-col gap-1.5 px-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-text">
                  {t(
                    "playground:tools.allowExternalImages",
                    "Load external images in chat"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={allowExternalImages}
                  onChange={(checked) =>
                    onAllowExternalImagesChange(checked)
                  }
                />
              </div>
              <p className="text-[11px] text-text-muted">
                {t(
                  "playground:tools.allowExternalImagesHelp",
                  "When off, external image URLs are blocked and shown as links."
                )}
              </p>
            </div>

            <div className="border-t border-border my-1" />

            <div className="flex flex-col gap-1.5 px-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-text">
                  {t(
                    "playground:tools.showMoodBadge",
                    "Show mood badge in chat"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={showMoodBadge}
                  onChange={(checked) => onShowMoodBadgeChange(checked)}
                />
              </div>
              <p className="text-[11px] text-text-muted">
                {t(
                  "playground:tools.showMoodBadgeHelp",
                  'Shows labels like "Mood: neutral" on assistant messages.'
                )}
              </p>

              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-text">
                  {t(
                    "playground:tools.showMoodConfidence",
                    "Show mood confidence (%)"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={showMoodConfidence}
                  disabled={!showMoodBadge}
                  onChange={(checked) =>
                    onShowMoodConfidenceChange(checked)
                  }
                />
              </div>
              <p className="text-[11px] text-text-muted">
                {t(
                  "playground:tools.showMoodConfidenceHelp",
                  "Adds confidence percentage when available."
                )}
              </p>
            </div>

            <div className="border-t border-border my-1" />

            <div className="flex flex-col gap-1.5 px-2">
              <button
                type="button"
                onClick={onOpenRawRequest}
                className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
              >
                <span>
                  {t(
                    "playground:tools.rawChatRequest",
                    "View raw chat JSON"
                  )}
                </span>
                <FileText className="h-4 w-4" />
              </button>
              <p className="text-[11px] text-text-muted">
                {t(
                  "playground:tools.rawChatRequestHelp",
                  "Shows the chat request payload preview generated from your current composer input."
                )}
              </p>
            </div>

            <div className="border-t border-border my-1" />

            {/* Voice Settings */}
            <div className="flex flex-col gap-2 px-2">
              <Tooltip
                title={
                  voiceChatAvailable
                    ? voiceChatStatusLabel
                    : voiceChatUnavailableReason ??
                      t(
                        "playground:voiceChat.unavailableTitle",
                        "Voice chat unavailable"
                      )
                }
              >
                <span className="inline-flex w-full">
                  <button
                    type="button"
                    onClick={onVoiceChatToggle}
                    disabled={!voiceChatAvailable || isSending}
                    className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm transition disabled:cursor-not-allowed disabled:opacity-50 ${
                      voiceChatState === "error"
                        ? "text-danger"
                        : voiceChatEnabled && voiceChatState !== "idle"
                          ? "bg-surface2 text-primaryStrong"
                          : "text-text hover:bg-surface2"
                    }`}
                  >
                    <span>
                      {t(
                        "playground:tools.voiceSettings",
                        "Voice settings"
                      )}
                    </span>
                    <Headphones className="h-4 w-4" />
                  </button>
                </span>
              </Tooltip>
              <div
                className={`flex flex-col gap-2 text-xs ${
                  !voiceChatAvailable
                    ? "pointer-events-none opacity-50"
                    : ""
                }`}
              >
                {voiceChatSettingsFields}
              </div>
            </div>

            {imageProviderControl}
          </div>
        )}

        <div className="border-t border-border my-1" />

        {/* Footer Actions */}
        <Link
          to="/model-playground"
          title={
            t(
              "playground:actions.workspacePlayground",
              "Compare models"
            ) as string
          }
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
        >
          <span>
            {t("playground:actions.compareModels", "Compare models")}
          </span>
          <GitBranch className="h-4 w-4" />
        </Link>
        <button
          type="button"
          onClick={onClearContext}
          disabled={historyLength === 0}
          title={t("tooltip.clearContext") as string}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-danger transition hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-40 disabled:text-text-muted disabled:hover:bg-transparent"
        >
          <span>
            {t(
              "playground:actions.clearConversation",
              "Clear conversation"
            )}
          </span>
          <EraserIcon className="h-4 w-4" />
        </button>
      </div>
    )

    return (
      <Popover
        trigger="click"
        placement="topRight"
        content={content}
        overlayClassName="playground-more-tools"
        open={toolsPopoverOpen}
        onOpenChange={onToolsPopoverChange}
      >
        <TldwButton
          variant="outline"
          size="sm"
          shape={isProMode ? "rounded" : "pill"}
          iconOnly={!isProMode}
          ariaLabel={
            t("playground:composer.moreTools", "More tools") as string
          }
          title={
            t("playground:composer.moreTools", "More tools") as string
          }
          data-testid="tools-button"
        >
          {isProMode ? (
            <span>
              {t("playground:composer.toolsButton", "+Tools")}
            </span>
          ) : (
            <>
              <SlidersHorizontal
                className="h-4 w-4"
                aria-hidden="true"
              />
              <span className="sr-only">
                {t("playground:composer.moreTools", "More tools")}
              </span>
            </>
          )}
        </TldwButton>
      </Popover>
    )
  })
