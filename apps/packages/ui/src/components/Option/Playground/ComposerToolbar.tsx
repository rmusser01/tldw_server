import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { Button as TldwButton } from "@/components/Common/Button"
import { PromptSelect } from "@/components/Common/PromptSelect"
import { ConnectionStatus } from "@/components/Layouts/ConnectionStatus"
import { PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY } from "@/utils/output-formatting-guide"
import { Tooltip } from "antd"
import {
  ChevronDown,
  FileIcon,
  FileText,
  Gauge,
  Globe,
  MicIcon,
  Search
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { useStorage } from "@plasmohq/storage/hook"

import { ComposerToolbarOverflow } from "./ComposerToolbarOverflow"
import {
  ParameterPresets,
  type PromptTemplate,
  SessionCostEstimation,
  SystemPromptTemplatesButton
} from "./playground-features"

export type ComposerToolbarProps = {
  isProMode: boolean
  isMobile: boolean
  isConnectionReady: boolean
  isSending: boolean
  optionsExpanded?: boolean
  sendControlPlacement?: "toolbar" | "external"
  modeLauncherButton?: React.ReactNode
  compareControl?: React.ReactNode
  // Row 1: primary selectors (pre-rendered by parent)
  modelSelectButton: React.ReactNode
  mcpControl: React.ReactNode
  // Row action controls (pre-rendered by parent)
  sendControl: React.ReactNode
  researchLaunchButton?: React.ReactNode
  attachmentButton: React.ReactNode
  toolsButton: React.ReactNode
  voiceChatButton: React.ReactNode
  modelUsageBadge: React.ReactNode
  // Prompt select
  selectedSystemPrompt: string | undefined
  systemPrompt: string | undefined
  setSystemPrompt: (prompt: string) => void
  setSelectedSystemPrompt: (id: string | undefined) => void
  setSelectedQuickPrompt: (prompt: string | undefined) => void
  // Ephemeral toggle
  temporaryChat: boolean
  onToggleTemporaryChat: (next: boolean) => void
  privateChatLocked: boolean
  isFireFoxPrivateMode: boolean
  persistenceTooltip: React.ReactNode
  // Knowledge / context
  contextToolsOpen: boolean
  onToggleKnowledgePanel: (tab: string) => void
  // Web search
  webSearch: boolean
  onToggleWebSearch: () => void
  hasWebSearch: boolean
  // Model settings
  onOpenModelSettings: () => void
  modelSummaryLabel: string
  promptSummaryLabel: string
  // Dictation
  hasDictation: boolean
  speechAvailable: boolean
  speechUsesServer: boolean
  isListening: boolean
  isServerDictating: boolean
  voiceChatEnabled: boolean
  speechTooltip: string
  onDictationToggle: () => void
  // Pro-only: parameter presets & templates
  onTemplateSelect: (template: PromptTemplate) => void
  // Pro-only: cost estimation
  selectedModel: string | null
  resolvedProviderKey: string
  messages: any[]
  // Pro-only: context counts
  selectedDocumentsCount: number
  uploadedFilesCount: number
  // Persistence hints
  serverChatId: string | null
  showServerPersistenceHint: boolean
  onDismissServerPersistenceHint: () => void
  onFocusConnectionCard: () => void
  contextItems?: ComposerContextItem[]
}

export type ComposerContextItem = {
  id: string
  label: string
  value?: string
  tone?: "neutral" | "active" | "warning"
  onClick?: () => void
}

/**
 * Unified toolbar for the chat composer.
 * Renders all controls once and uses `isProMode` for layout differences.
 * On mobile, secondary controls collapse into an overflow popover.
 */
export const ComposerToolbar = React.memo(function ComposerToolbar(
  props: ComposerToolbarProps
) {
  const { t } = useTranslation([
    "playground",
    "common",
    "option",
    "sidepanel",
    "settings"
  ])
  const {
    isProMode,
    isMobile,
    isConnectionReady,
    modeLauncherButton,
    compareControl,
    modelSelectButton,
    mcpControl,
    sendControl,
    optionsExpanded = true,
    sendControlPlacement = "toolbar",
    researchLaunchButton,
    attachmentButton,
    toolsButton,
    voiceChatButton,
    modelUsageBadge,
    selectedSystemPrompt,
    systemPrompt,
    setSystemPrompt,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    temporaryChat,
    onToggleTemporaryChat,
    privateChatLocked,
    isFireFoxPrivateMode,
    persistenceTooltip,
    contextToolsOpen,
    onToggleKnowledgePanel,
    webSearch,
    onToggleWebSearch,
    hasWebSearch,
    onOpenModelSettings,
    modelSummaryLabel,
    promptSummaryLabel,
    hasDictation,
    speechAvailable,
    speechUsesServer,
    isListening,
    isServerDictating,
    voiceChatEnabled,
    speechTooltip,
    onDictationToggle,
    onTemplateSelect,
    selectedModel,
    resolvedProviderKey,
    messages,
    selectedDocumentsCount,
    uploadedFilesCount,
    serverChatId,
    showServerPersistenceHint,
    onDismissServerPersistenceHint,
    onFocusConnectionCard,
    contextItems = []
  } = props
  const toolbarSendControl =
    sendControlPlacement === "toolbar" ? sendControl : null

  const ephemeralDisabled = privateChatLocked || isFireFoxPrivateMode
  const [advancedControlsOpen, setAdvancedControlsOpen] = useStorage(
    "playgroundComposerAdvancedControlsOpen",
    false
  )
  const [casualAdvancedControlsOpen, setCasualAdvancedControlsOpen] =
    useStorage("playgroundComposerCasualAdvancedControlsOpen", false)
  const [appendFormattingGuidePrompt, setAppendFormattingGuidePrompt] =
    useStorage(PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY, false)

  // --- Shared sub-elements ---
  const ephemeralToggle = React.useMemo(
    () => (
      <Tooltip title={persistenceTooltip}>
        <span>
          <button
            type="button"
            onClick={() => onToggleTemporaryChat(!temporaryChat)}
            disabled={ephemeralDisabled}
            aria-pressed={temporaryChat}
            aria-label={
              temporaryChat
                ? (t(
                    "playground:composer.ephemeralLabel",
                    "Temporary"
                  ) as string)
                : (t("playground:composer.savedLabel", "Saved") as string)
            }
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text ${
              temporaryChat
                ? "bg-primary/10 text-primaryStrong"
                : "text-text-muted"
            } ${ephemeralDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            {temporaryChat
              ? t("playground:composer.ephemeralLabel", "Temporary")
              : t("playground:composer.savedLabel", "Saved")}
          </button>
        </span>
      </Tooltip>
    ),
    [
      ephemeralDisabled,
      onToggleTemporaryChat,
      persistenceTooltip,
      t,
      temporaryChat
    ]
  )

  const searchContextButton = React.useMemo(
    () => (
      <button
        type="button"
        onClick={() => onToggleKnowledgePanel("search")}
        title={
          contextToolsOpen
            ? (t(
                "playground:composer.contextKnowledgeClose",
                "Close Search & Context"
              ) as string)
            : (t(
                "playground:composer.contextKnowledge",
                "Search & Context"
              ) as string)
        }
        aria-pressed={contextToolsOpen}
        aria-expanded={contextToolsOpen}
        aria-label={
          contextToolsOpen
            ? (t(
                "playground:composer.contextKnowledgeClose",
                "Close Search & Context"
              ) as string)
            : (t(
                "playground:composer.contextKnowledge",
                "Search & Context"
              ) as string)
        }
        data-testid="knowledge-search-toggle"
        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text ${
          contextToolsOpen
            ? "bg-primary/10 text-primaryStrong"
            : "text-text-muted"
        }`}>
        <Search className="h-3 w-3" />
        <span className={isProMode ? "" : "truncate"}>
          {contextToolsOpen
            ? t(
                "playground:composer.contextKnowledgeClose",
                "Close Search & Context"
              )
            : t("playground:composer.contextKnowledge", "Search & Context")}
        </span>
      </button>
    ),
    [contextToolsOpen, isProMode, onToggleKnowledgePanel, t]
  )

  const webSearchButton = React.useMemo(
    () =>
      hasWebSearch ? (
        <button
          type="button"
          onClick={onToggleWebSearch}
          aria-pressed={webSearch}
          aria-label={t("playground:tools.webSearch", "Web search") as string}
          title={t("playground:tools.webSearch", "Web search") as string}
          data-testid="web-search-toggle"
          className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text ${
            webSearch ? "bg-primary/10 text-primaryStrong" : "text-text-muted"
          }`}>
          <Globe className="h-3 w-3" />
          <span>{t("playground:actions.webSearchShort", "Web")}</span>
        </button>
      ) : null,
    [hasWebSearch, onToggleWebSearch, t, webSearch]
  )

  const dictationButton = React.useMemo(
    () =>
      hasDictation ? (
        <Tooltip title={speechTooltip}>
          <button
            type="button"
            onClick={onDictationToggle}
            disabled={!speechAvailable || voiceChatEnabled}
            data-testid="dictation-button"
            className={`inline-flex items-center justify-center rounded-md text-xs transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50 ${
              speechAvailable &&
              ((speechUsesServer && isServerDictating) ||
                (!speechUsesServer && isListening))
                ? "text-primaryStrong"
                : "text-text-muted"
            } ${isProMode ? "px-2 py-1" : "h-9 w-9 p-0"}`}
            aria-label={
              !speechAvailable
                ? (t(
                    "playground:actions.speechUnavailableTitle",
                    "Dictation unavailable"
                  ) as string)
                : speechUsesServer
                  ? isServerDictating
                    ? (t(
                        "playground:actions.speechStop",
                        "Stop dictation"
                      ) as string)
                    : (t(
                        "playground:actions.speechStart",
                        "Start dictation"
                      ) as string)
                  : isListening
                    ? (t(
                        "playground:actions.speechStop",
                        "Stop dictation"
                      ) as string)
                    : (t(
                        "playground:actions.speechStart",
                        "Start dictation"
                      ) as string)
            }>
            <MicIcon className="h-4 w-4" />
          </button>
        </Tooltip>
      ) : null,
    [
      hasDictation,
      isListening,
      isProMode,
      isServerDictating,
      onDictationToggle,
      speechAvailable,
      speechTooltip,
      speechUsesServer,
      t,
      voiceChatEnabled
    ]
  )

  const chatSettingsButton = React.useMemo(
    () =>
      isProMode ? (
        <Tooltip title={t("common:currentChatModelSettings") as string}>
          <button
            type="button"
            onClick={onOpenModelSettings}
            aria-label={t("common:currentChatModelSettings") as string}
            className="inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-xs text-text transition hover:bg-surface2">
            <Gauge className="h-4 w-4" aria-hidden="true" />
            <span className="flex flex-col items-start text-left">
              <span className="font-medium">
                {t("playground:composer.chatSettings", "Chat Settings")}
              </span>
              <span className="text-xs text-text-muted">
                {modelSummaryLabel} • {promptSummaryLabel}
              </span>
            </span>
          </button>
        </Tooltip>
      ) : (
        <Tooltip title={t("common:currentChatModelSettings") as string}>
          <TldwButton
            variant="outline"
            shape="pill"
            iconOnly
            onClick={onOpenModelSettings}
            ariaLabel={t("common:currentChatModelSettings") as string}
            className="text-text-muted">
            <Gauge className="h-4 w-4" aria-hidden="true" />
            <span className="sr-only">
              {t("playground:composer.chatSettings", "Chat Settings")}
            </span>
          </TldwButton>
        </Tooltip>
      ),
    [isProMode, modelSummaryLabel, onOpenModelSettings, promptSummaryLabel, t]
  )

  const tabsCountBadge = React.useMemo(
    () =>
      isProMode && selectedDocumentsCount > 0 ? (
        <button
          type="button"
          onClick={() => {
            const chips = document.querySelector<HTMLElement>(
              "[data-playground-tabs='true']"
            )
            if (chips) {
              chips.focus()
              chips.scrollIntoView({ block: "nearest" })
            }
          }}
          title={
            t(
              "playground:composer.contextTabsHint",
              "Review or remove referenced tabs, or add more from your open browser tabs."
            ) as string
          }
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-text">
          <FileText className="h-3 w-3 text-text-subtle" />
          <span>
            {
              t("playground:composer.contextTabs", {
                defaultValue: "{{count}} tabs",
                count: selectedDocumentsCount
              } as any) as string
            }
          </span>
        </button>
      ) : null,
    [isProMode, selectedDocumentsCount, t]
  )

  const filesCountBadge = React.useMemo(
    () =>
      isProMode && uploadedFilesCount > 0 ? (
        <button
          type="button"
          onClick={() => {
            const files = document.querySelector<HTMLElement>(
              "[data-playground-uploads='true']"
            )
            if (files) {
              files.focus()
              files.scrollIntoView({ block: "nearest" })
            }
          }}
          title={
            t(
              "playground:composer.contextFilesHint",
              "Review attached files, remove them, or add more."
            ) as string
          }
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-text">
          <FileIcon className="h-3 w-3 text-text-subtle" />
          <span>
            {
              t("playground:composer.contextFiles", {
                defaultValue: "{{count}} files",
                count: uploadedFilesCount
              } as any) as string
            }
          </span>
        </button>
      ) : null,
    [isProMode, t, uploadedFilesCount]
  )

  const promptSelectControl = React.useMemo(
    () => (
      <PromptSelect
        selectedSystemPrompt={selectedSystemPrompt}
        systemPrompt={systemPrompt}
        setSystemPrompt={setSystemPrompt}
        setSelectedSystemPrompt={setSelectedSystemPrompt}
        setSelectedQuickPrompt={setSelectedQuickPrompt}
        iconClassName="h-4 w-4"
        className="text-text-muted hover:text-text"
      />
    ),
    [
      selectedSystemPrompt,
      setSelectedQuickPrompt,
      setSelectedSystemPrompt,
      setSystemPrompt,
      systemPrompt
    ]
  )

  const characterSelectControl = React.useMemo(
    () => (
      <AssistantSelect
        variant="dropdown"
        showLabel={isProMode ? undefined : false}
        iconClassName="h-4 w-4"
        className="text-text-muted hover:text-text"
      />
    ),
    [isProMode]
  )

  const hiddenContextItemIds = React.useMemo(
    () =>
      new Set([
        "providerStatus",
        "routingPolicy",
        "modelCapabilities",
        "summaryCheckpoint",
        "conversationState",
        "imageEventSync"
      ]),
    []
  )
  const getContextItemTestId = React.useCallback(
    (item: ComposerContextItem) => {
      if (item.id === "sessionStatus") {
        return "composer-session-status-chip"
      }
      return undefined
    },
    []
  )

  const renderContextItem = (
    item: ComposerContextItem,
    dataTestId?: string
  ) => {
    const baseClasses =
      item.tone === "warning"
        ? "border-warn/40 bg-warn/10 text-warn"
        : item.tone === "active"
          ? "border-primary/40 bg-primary/10 text-primaryStrong"
          : "border-border bg-surface2 text-text-muted"
    const content = (
      <>
        <span className="font-medium">{item.label}</span>
        {item.value ? (
          <span className="max-w-[150px] truncate text-text">{item.value}</span>
        ) : null}
      </>
    )

    if (item.onClick) {
      return (
        <button
          key={item.id}
          type="button"
          onClick={item.onClick}
          data-testid={dataTestId}
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] transition hover:bg-surface2 ${baseClasses}`}
          title={item.value ? `${item.label}: ${item.value}` : item.label}>
          {content}
        </button>
      )
    }

    return (
      <span
        key={item.id}
        data-testid={dataTestId}
        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${baseClasses}`}
        title={item.value ? `${item.label}: ${item.value}` : item.label}>
        {content}
      </span>
    )
  }

  const modelContextItem = contextItems.find((item) => item.id === "model")
  const casualContextItems = contextItems.filter(
    (item) =>
      item.id !== "model" &&
      item.id !== "temporary" &&
      !hiddenContextItemIds.has(item.id)
  )
  const visibleContextItems = contextItems.filter(
    (item) => !hiddenContextItemIds.has(item.id)
  )
  const casualModelItem: ComposerContextItem = modelContextItem
    ? {
        ...modelContextItem,
        onClick: modelContextItem.onClick ?? onOpenModelSettings
      }
    : {
        id: "model",
        label: t("playground:composer.context.model", "Model"),
        value: selectedModel ? modelSummaryLabel : t("common:none", "None"),
        tone: selectedModel ? "active" : "warning",
        onClick: onOpenModelSettings
      }
  const persistenceChipClass = temporaryChat
    ? "border-warn/40 bg-warn/10 text-warn"
    : "border-border bg-surface2 text-text-muted"
  const persistenceChipLabel = temporaryChat
    ? (t("playground:composer.ephemeralLabel", "Temporary") as string)
    : (t("playground:composer.savedLabel", "Saved") as string)
  const advancedChipClass = casualAdvancedControlsOpen
    ? "border-primary/40 bg-primary/10 text-primaryStrong"
    : "border-border bg-surface2 text-text-muted"
  const formattingGuideToggleClass = appendFormattingGuidePrompt
    ? "bg-primary/10 text-primaryStrong"
    : "text-text-muted"
  const casualModeContextGroupLabel = t(
    "playground:composer.groups.modeAndContext",
    "Mode and context controls"
  ) as string
  const runInputGroupLabel = t(
    "playground:composer.groups.runInput",
    "Run input controls"
  ) as string
  const mobileModeModelGroupLabel = t(
    "playground:composer.groups.mobileModeAndModel",
    "Mobile mode and model controls"
  ) as string
  const mobileContextGroupLabel = t(
    "playground:composer.groups.mobileContext",
    "Mobile context controls"
  ) as string
  const mobileRunInputGroupLabel = t(
    "playground:composer.groups.mobileRunInput",
    "Mobile run input controls"
  ) as string
  const proContextPanelLabel = t(
    "playground:composer.groups.contextSetupControls",
    "Context setup controls"
  ) as string
  const proContextPanelHeading = t(
    "playground:composer.groups.contextSetup",
    "Context setup"
  ) as string
  const proGenerationPanelLabel = t(
    "playground:composer.groups.modelToolsRunControls",
    "Model, tools, and run controls"
  ) as string
  const proGenerationPanelHeading = t(
    "playground:composer.groups.modelToolsRun",
    "Model, tools, and run"
  ) as string
  const advancedControlsGroupLabel = t(
    "playground:composer.groups.advancedComposerControls",
    "Advanced composer controls"
  ) as string
  const formattingGuideControl = (
    <button
      type="button"
      onClick={() =>
        setAppendFormattingGuidePrompt(!appendFormattingGuidePrompt)
      }
      aria-pressed={appendFormattingGuidePrompt}
      data-testid="composer-formatting-guide-toggle"
      className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text ${formattingGuideToggleClass}`}
      title={
        t(
          "playground:composer.outputFormattingGuideHint",
          "Append a formatting style guide to the system prompt."
        ) as string
      }>
      <FileText className="h-3.5 w-3.5" aria-hidden="true" />
      <span>
        {t("playground:composer.outputFormattingGuide", "Formatting guide")}
      </span>
    </button>
  )
  const casualAdvancedControlsToggle = (
    <button
      type="button"
      onClick={() =>
        setCasualAdvancedControlsOpen(!casualAdvancedControlsOpen)
      }
      aria-expanded={casualAdvancedControlsOpen}
      aria-pressed={casualAdvancedControlsOpen}
      aria-controls="composer-casual-advanced-controls-row"
      data-testid="composer-casual-advanced-chip"
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition hover:bg-surface2 ${advancedChipClass}`}>
      <span>{t("playground:composer.advancedControls", "Advanced controls")}</span>
      <ChevronDown
        className={`h-3 w-3 transition-transform ${
          casualAdvancedControlsOpen ? "rotate-180" : ""
        }`}
        aria-hidden="true"
      />
    </button>
  )

  const casualContextStrip = (
    <div
      data-testid="composer-context-strip"
      aria-label={
        t("playground:composer.activeContext", "Active context") as string
      }
      className="flex flex-wrap items-center gap-1.5 border-t border-border/50 pt-2">
      {modelSelectButton ? (
        <span
          data-testid="composer-casual-model-selector-chip"
          className="inline-flex items-center">
          {modelSelectButton}
        </span>
      ) : (
        renderContextItem(casualModelItem)
      )}
      {casualContextItems.map((item) =>
        renderContextItem(item, getContextItemTestId(item))
      )}
      {modelUsageBadge ? (
        <span
          data-testid="composer-casual-token-chip"
          className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text-muted">
          {modelUsageBadge}
        </span>
      ) : null}
      <Tooltip title={persistenceTooltip}>
        <span>
          <button
            type="button"
            onClick={() => onToggleTemporaryChat(!temporaryChat)}
            disabled={ephemeralDisabled}
            data-testid="composer-casual-persistence-chip"
            aria-pressed={temporaryChat}
            aria-label={
              temporaryChat
                ? (t(
                    "playground:composer.ephemeralLabel",
                    "Temporary"
                  ) as string)
                : (t("playground:composer.savedLabel", "Saved") as string)
            }
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition hover:bg-surface2 ${persistenceChipClass} ${ephemeralDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <span>{persistenceChipLabel}</span>
          </button>
        </span>
      </Tooltip>
      {casualAdvancedControlsToggle}
    </div>
  )

  const contextStrip =
    visibleContextItems.length > 0 ? (
      <div
        data-testid="composer-context-strip"
        aria-label={
          t("playground:composer.activeContext", "Active context") as string
        }
        className="flex flex-wrap items-center gap-1.5 border-t border-border/50 pt-2">
        {visibleContextItems.map((item) =>
          renderContextItem(item, getContextItemTestId(item))
        )}
      </div>
    ) : null

  const proAdvancedControls =
    isProMode && !isMobile ? (
      <div className="border-t border-border/50 pt-2">
        <button
          type="button"
          onClick={() => setAdvancedControlsOpen(!advancedControlsOpen)}
          aria-expanded={advancedControlsOpen}
          aria-controls="composer-pro-advanced-controls-row"
          data-testid="composer-advanced-toggle"
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs font-medium text-text-muted transition hover:bg-surface2 hover:text-text">
          <span>
            {t("playground:composer.advancedControls", "Advanced controls")}
          </span>
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${
              advancedControlsOpen ? "rotate-180" : ""
            }`}
            aria-hidden="true"
          />
        </button>
        {advancedControlsOpen && (
          <div
            id="composer-pro-advanced-controls-row"
            role="group"
            aria-label={advancedControlsGroupLabel}
            data-playground-toolbar-row="advanced"
            className="mt-2 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <ParameterPresets compact />
              {formattingGuideControl}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <SystemPromptTemplatesButton onSelect={onTemplateSelect} />
              {messages.length > 0 && (
                <SessionCostEstimation
                  modelId={selectedModel}
                  provider={resolvedProviderKey}
                  messages={messages}
                />
              )}
            </div>
          </div>
        )}
      </div>
    ) : null

  const mobileToolbarLayout = (
    <div className="flex flex-col gap-2">
      <div
        role="group"
        aria-label={mobileModeModelGroupLabel}
        data-playground-toolbar-row="primary"
        className="flex flex-wrap items-center gap-2">
        {modeLauncherButton}
        <ConnectionStatus showLabel={false} className="px-1 py-0.5" />
        {modelSelectButton}
        {promptSelectControl}
        {characterSelectControl}
        {compareControl}
      </div>
      <div
        data-playground-toolbar-row="actions"
        className="flex items-center justify-between gap-2">
        <div
          role="group"
          aria-label={mobileContextGroupLabel}
          className="flex items-center gap-2">
          {ephemeralToggle}
          {searchContextButton}
          {webSearchButton}
        </div>
        <div
          role="group"
          aria-label={mobileRunInputGroupLabel}
          className="flex items-center gap-2">
          {modelUsageBadge}
          <ComposerToolbarOverflow
            isProMode={isProMode}
            isConnectionReady={isConnectionReady}
            contextToolsOpen={contextToolsOpen}
            onToggleKnowledgePanel={onToggleKnowledgePanel}
            webSearch={webSearch}
            onToggleWebSearch={onToggleWebSearch}
            hasWebSearch={hasWebSearch}
            onOpenModelSettings={onOpenModelSettings}
            hasDictation={hasDictation}
            speechAvailable={speechAvailable}
            speechUsesServer={speechUsesServer}
            isListening={isListening}
            isServerDictating={isServerDictating}
            voiceChatEnabled={voiceChatEnabled}
            onDictationToggle={onDictationToggle}
            temporaryChat={temporaryChat}
            onFocusConnectionCard={onFocusConnectionCard}
          />
          {researchLaunchButton}
          {voiceChatButton}
          {attachmentButton}
          {toolsButton}
          {toolbarSendControl}
        </div>
      </div>
    </div>
  )

  const casualToolbarLayout = (
    <div
      data-playground-toolbar-layout="casual"
      className="flex flex-col gap-2">
      <div
        data-playground-toolbar-row="actions"
        className="flex flex-nowrap items-center gap-2 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div
          role="group"
          aria-label={casualModeContextGroupLabel}
          className="flex flex-nowrap items-center gap-2 text-text-muted">
          {modeLauncherButton}
          {mcpControl}
          {searchContextButton}
          {promptSelectControl}
          {characterSelectControl}
        </div>
        <div
          role="group"
          aria-label={runInputGroupLabel}
          className="ml-auto flex flex-nowrap items-center gap-2">
          {compareControl}
          {dictationButton}
          {researchLaunchButton}
          {voiceChatButton}
          {attachmentButton}
          {toolsButton}
          {chatSettingsButton}
          {toolbarSendControl}
        </div>
      </div>
    </div>
  )
  const casualAdvancedControlsPanel =
    casualAdvancedControlsOpen && !isMobile && !isProMode ? (
      <div
        role="group"
        aria-label={advancedControlsGroupLabel}
        data-playground-toolbar-row="advanced"
        className="rounded-md border border-border/60 bg-surface2/60 p-2">
        <div
          id="composer-casual-advanced-controls-row"
          data-testid="composer-casual-advanced-controls-row"
          className="flex flex-nowrap items-center gap-2 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <ConnectionStatus showLabel={false} className="px-1 py-0.5" />
          {tabsCountBadge}
          {filesCountBadge}
          <ParameterPresets compact />
          <SystemPromptTemplatesButton onSelect={onTemplateSelect} />
          {formattingGuideControl}
        </div>
      </div>
    ) : null

  const proSplitToolbarLayout = (
    <div
      data-playground-toolbar-layout="pro-split"
      className="flex flex-col gap-2">
      <div className="grid gap-2 lg:grid-cols-2">
        <section
          role="group"
          aria-label={proContextPanelLabel}
          data-testid="composer-pro-context-panel"
          className="rounded-md border border-border/60 bg-surface2/60 p-2">
          <div className="mb-2 text-[11px] font-semibold text-text-muted">
            {proContextPanelHeading}
          </div>
          <div
            data-playground-toolbar-row="primary"
            className="flex flex-wrap items-center gap-2">
            {modeLauncherButton}
            {compareControl}
            {ephemeralToggle}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-text-muted">
            {searchContextButton}
            {webSearchButton}
            {tabsCountBadge}
            {filesCountBadge}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {promptSelectControl}
            {characterSelectControl}
          </div>
        </section>
        <section
          role="group"
          aria-label={proGenerationPanelLabel}
          data-testid="composer-pro-generation-panel"
          className="rounded-md border border-border/60 bg-surface2/60 p-2">
          <div className="mb-2 text-[11px] font-semibold text-text-muted">
            {proGenerationPanelHeading}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ConnectionStatus showLabel={false} className="px-1 py-0.5" />
            {mcpControl}
            {modelSelectButton}
          </div>
          <div
            data-playground-toolbar-row="actions"
            className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2 text-text-muted">
              {dictationButton}
              {modelUsageBadge}
              {chatSettingsButton}
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              {researchLaunchButton}
              {voiceChatButton}
              {attachmentButton}
              {toolsButton}
              {toolbarSendControl}
            </div>
          </div>
        </section>
      </div>
      {proAdvancedControls}
    </div>
  )

  const collapsedMobileToolbarLayout = isMobile ? (
    <div
      role="group"
      aria-label={
        t("playground:composer.quickActions", "Composer quick actions") as string
      }
      data-playground-toolbar-row="collapsed-actions"
      className="flex items-center justify-end gap-2">
      {attachmentButton}
    </div>
  ) : null

  return (
    <div
      data-testid="composer-options-panel"
      className="mt-2 flex flex-col gap-1">
      {optionsExpanded ? (
        <>
          {isMobile
            ? mobileToolbarLayout
            : isProMode
              ? proSplitToolbarLayout
              : casualToolbarLayout}

          {/* Pro-only: persistence hints (desktop only) */}
          {!isMobile && isProMode && !temporaryChat && !isConnectionReady && (
            <button
              type="button"
              onClick={onFocusConnectionCard}
              title={
                t(
                  "playground:composer.persistence.connectToSave",
                  "Connect your server to sync chats."
                ) as string
              }
              className="inline-flex w-fit items-center gap-1 text-xs font-medium text-primary hover:text-primaryStrong">
              {t(
                "playground:composer.persistence.connectToSave",
                "Connect your server to sync chats."
              )}
            </button>
          )}
          {isProMode &&
            !temporaryChat &&
            serverChatId &&
            showServerPersistenceHint && (
              <p className="max-w-md text-xs text-text-muted">
                <span className="font-semibold">
                  {t(
                    "playground:composer.persistence.serverInlineTitle",
                    "Saved locally + on your server"
                  )}
                  {": "}
                </span>
                {t(
                  "playground:composer.persistence.serverInlineBody",
                  "This chat is stored both in this browser and on your tldw server, so you can reopen it from server history, keep a long-term record, and analyze it alongside other conversations."
                )}
                <button
                  type="button"
                  onClick={onDismissServerPersistenceHint}
                  title={t("common:dismiss", "Dismiss") as string}
                  className="ml-1 text-xs font-medium text-primary hover:text-primaryStrong">
                  {t("common:dismiss", "Dismiss")}
                </button>
              </p>
            )}
          {!isMobile && !isProMode ? casualContextStrip : contextStrip}
          {!isMobile && !isProMode ? casualAdvancedControlsPanel : null}
        </>
      ) : (
        collapsedMobileToolbarLayout
      )}
    </div>
  )
})
