import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { Button as TldwButton } from "@/components/Common/Button"
import { BuddyManagementButton } from "@/components/Common/PersonaBuddy/BuddyManagementButton"
import { PromptSelect } from "@/components/Common/PromptSelect"
import { ConnectionStatus } from "@/components/Layouts/ConnectionStatus"
import {
  getPersistenceModeControlLabel,
  getPersistenceModePillLabel
} from "@/hooks/playground/usePersistenceMode"
import type { AssistantSelection } from "@/types/assistant-selection"
import { PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY } from "@/utils/output-formatting-guide"
import { Modal, Tooltip } from "antd"
import {
  ChevronDown,
  FileIcon,
  FileText,
  Gauge,
  Globe,
  MicIcon,
  Search,
  Users
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { useStorage } from "@plasmohq/storage/hook"

import {
  ComposerToolbarOverflow,
  type ComposerToolbarRolePlayActions
} from "./ComposerToolbarOverflow"
import { useCasualComposerDisclosure } from "./hooks/useCasualComposerDisclosure"
import { openModelSelector } from "./playground-cockpit-actions"
import {
  ParameterPresets,
  ParameterPresetsDropdown,
  type PromptTemplate,
  SessionCostEstimation,
  SystemPromptTemplatesButton,
  SystemPromptTemplatesModal
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
  openUIRequestButton?: React.ReactNode
  openUIRequestMode?: boolean
  toolChoice?: "auto" | "none" | "required"
  nestedDisclosureOpen?: boolean
  focusFallbackRef?: React.RefObject<HTMLElement | null>
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
  setSystemPrompt: (prompt: string | undefined) => void
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
  currentProvider?: string | null
  resolvedProviderKey: string
  messages: any[]
  // Pro-only: context counts
  selectedDocumentsCount: number
  uploadedFilesCount: number
  // Persistence hints
  serverChatId: string | null
  promptAssistContextKey?: string
  promptAssistBackendKey?: string | null
  promptAssistAuthorizationRevision?: string | null
  showServerPersistenceHint: boolean
  onDismissServerPersistenceHint: () => void
  onFocusConnectionCard: () => void
  contextItems?: ComposerContextItem[]
  rolePlayActions?: ComposerToolbarRolePlayActions
  onTrackedAssistantSelection?: (
    selection: AssistantSelection,
    isCurrent?: () => boolean
  ) => boolean | Promise<boolean>
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
 * Casual secondary controls share a task-based disclosure on every viewport.
 * Pro keeps its direct controls and mobile overflow.
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
    openUIRequestButton,
    openUIRequestMode = false,
    toolChoice = "none",
    nestedDisclosureOpen = false,
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
    currentProvider,
    resolvedProviderKey,
    messages,
    selectedDocumentsCount,
    uploadedFilesCount,
    serverChatId,
    promptAssistContextKey,
    promptAssistBackendKey,
    promptAssistAuthorizationRevision,
    showServerPersistenceHint,
    onDismissServerPersistenceHint,
    onFocusConnectionCard,
    contextItems = [],
    rolePlayActions: providedRolePlayActions
  } = props
  const toolbarSendControl =
    sendControlPlacement === "toolbar" ? sendControl : null
  const ephemeralDisabled = privateChatLocked || isFireFoxPrivateMode
  const [advancedControlsOpen, setAdvancedControlsOpen] = useStorage(
    "playgroundComposerAdvancedControlsOpen",
    false
  )
  const {
    open: casualAdvancedControlsOpen,
    setExplicitOpen: setCasualAdvancedControlsOpen,
    loading: casualDisclosureLoading,
    unavailable: casualDisclosureUnavailable
  } = useCasualComposerDisclosure()
  const [assistantPickerOpen, setAssistantPickerOpen] = React.useState(false)
  const [promptPickerOpen, setPromptPickerOpen] = React.useState(false)
  const isProModeRef = React.useRef(isProMode)
  isProModeRef.current = isProMode
  const pendingCustomizationRevealRef = React.useRef(false)
  const handleAssistantPickerOpen = React.useCallback((open: boolean) => {
    setAssistantPickerOpen(open)
    if (open && !isProModeRef.current)
      pendingCustomizationRevealRef.current = true
  }, [])
  const handlePromptPickerOpen = React.useCallback((open: boolean) => {
    setPromptPickerOpen(open)
    if (open && !isProModeRef.current)
      pendingCustomizationRevealRef.current = true
  }, [])
  React.useEffect(() => {
    if (!assistantPickerOpen && !promptPickerOpen)
      pendingCustomizationRevealRef.current = false
    if (
      !pendingCustomizationRevealRef.current ||
      casualDisclosureLoading ||
      isProMode
    )
      return
    pendingCustomizationRevealRef.current = false
    if (!casualAdvancedControlsOpen) void setCasualAdvancedControlsOpen(true)
  }, [
    assistantPickerOpen,
    promptPickerOpen,
    casualDisclosureLoading,
    isProMode,
    casualAdvancedControlsOpen,
    setCasualAdvancedControlsOpen
  ])
  const toolbarRef = React.useRef<HTMLDivElement>(null)
  const customizeTriggerRef = React.useRef<HTMLButtonElement>(null)
  const lastFocusedControlRef = React.useRef<HTMLElement | null>(null)
  const previousPickerOpenRef = React.useRef(false)
  const [appendFormattingGuidePrompt, setAppendFormattingGuidePrompt] =
    useStorage(PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY, false)
  const [systemPromptsOpen, setSystemPromptsOpen] = React.useState(false)
  const [generationStyleOpen, setGenerationStyleOpen] = React.useState(false)

  const mobileRolePlayActions = React.useMemo<ComposerToolbarRolePlayActions>(
    () => ({
      onOpenSystemPrompts:
        providedRolePlayActions?.onOpenSystemPrompts ??
        (() => setSystemPromptsOpen(true)),
      onOpenGenerationStyle:
        providedRolePlayActions?.onOpenGenerationStyle ??
        (() => setGenerationStyleOpen(true)),
      ...(providedRolePlayActions?.onOpenRolePlaySetup
        ? { onOpenRolePlaySetup: providedRolePlayActions.onOpenRolePlaySetup }
        : {})
    }),
    [
      providedRolePlayActions?.onOpenGenerationStyle,
      providedRolePlayActions?.onOpenRolePlaySetup,
      providedRolePlayActions?.onOpenSystemPrompts
    ]
  )

  const handleMobileTemplateSelect = React.useCallback(
    (template: PromptTemplate) => {
      onTemplateSelect(template)
      setSystemPromptsOpen(false)
    },
    [onTemplateSelect]
  )

  // Retained hidden controls and detached portals are both invalid return targets.
  // Repair only focus still owned by that disappearing control, never another task.
  React.useLayoutEffect(() => {
    const isVisible = (element: HTMLElement | null) => {
      if (!element?.isConnected || ("disabled" in element && element.disabled))
        return false
      let ancestor: HTMLElement | null = element
      while (ancestor) {
        const style = window.getComputedStyle(ancestor)
        if (
          ancestor.hidden ||
          ancestor.getAttribute("aria-hidden") === "true" ||
          style.display === "none" ||
          style.visibility === "hidden" ||
          style.visibility === "collapse"
        )
          return false
        ancestor = ancestor.parentElement
      }
      return true
    }
    const repairOwnedFocus = () => {
      const previous = lastFocusedControlRef.current
      if (!previous || isVisible(previous)) return
      if (
        document.activeElement !== document.body &&
        document.activeElement !== previous
      )
        return
      if (!isProMode) {
        if (!casualDisclosureLoading && isVisible(customizeTriggerRef.current))
          customizeTriggerRef.current?.focus()
        return
      }
      const candidates =
        toolbarRef.current?.querySelectorAll<HTMLButtonElement>(
          "button:not(:disabled)"
        )
      const target = Array.from(candidates ?? []).find(isVisible)
      if (target) target.focus()
      else if (isVisible(props.focusFallbackRef?.current ?? null))
        props.focusFallbackRef?.current?.focus()
    }
    const pickerOpen = assistantPickerOpen || promptPickerOpen
    const pickerClosing = previousPickerOpenRef.current && !pickerOpen
    previousPickerOpenRef.current = pickerOpen
    // Let an existing picker return to its visible trigger before qualifying it.
    if (!pickerClosing) repairOwnedFocus()
    // Existing picker focus return is scheduled; qualify its eventual target too.
    const frame = window.requestAnimationFrame(repairOwnedFocus)
    return () => window.cancelAnimationFrame(frame)
  }, [
    isProMode,
    isMobile,
    optionsExpanded,
    casualAdvancedControlsOpen,
    casualDisclosureLoading,
    assistantPickerOpen,
    promptPickerOpen,
    nestedDisclosureOpen,
    props.focusFallbackRef
  ])

  const closeCasualCustomization = () => {
    void setCasualAdvancedControlsOpen(false)
    customizeTriggerRef.current?.focus()
  }
  const handleCustomizationKeyDown = (
    event: React.KeyboardEvent<HTMLDivElement>
  ) => {
    if (
      isProMode ||
      !casualAdvancedControlsOpen ||
      event.key !== "Escape" ||
      event.defaultPrevented ||
      event.nativeEvent.isComposing ||
      nestedDisclosureOpen ||
      assistantPickerOpen ||
      promptPickerOpen ||
      systemPromptsOpen ||
      generationStyleOpen
    )
      return
    if (
      (event.target as HTMLElement).closest(
        '[role="dialog"], [role="listbox"], [role="menu"]'
      )
    )
      return
    event.preventDefault()
    event.stopPropagation()
    closeCasualCustomization()
    customizeTriggerRef.current?.focus()
  }

  // --- Shared sub-elements ---
  const persistenceChipLabel = getPersistenceModePillLabel(t, temporaryChat)
  const persistenceControlLabel = getPersistenceModeControlLabel(
    t,
    temporaryChat
  )
  const ephemeralToggle = React.useMemo(
    () => (
      <Tooltip title={persistenceTooltip}>
        <span>
          <button
            type="button"
            onClick={() => onToggleTemporaryChat(!temporaryChat)}
            disabled={ephemeralDisabled}
            aria-pressed={!temporaryChat}
            aria-label={persistenceControlLabel}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text ${
              temporaryChat
                ? "bg-primary/10 text-primaryStrong"
                : "text-text-muted"
            } ${ephemeralDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            {persistenceChipLabel}
          </button>
        </span>
      </Tooltip>
    ),
    [
      ephemeralDisabled,
      onToggleTemporaryChat,
      persistenceTooltip,
      persistenceChipLabel,
      persistenceControlLabel,
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
            // NOTE: A dedicated "start in flight" flag is not available via
            // props here, so the disabled state cannot reflect the brief
            // getUserMedia window. Double-click re-entry during a start is
            // already guarded synchronously at the hook level (useServerDictation
            // /useMicStream startingRef), which makes a second click a no-op.
            // Surfacing an in-flight state on the button would require new prop
            // plumbing from the parent and is intentionally out of scope here.
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
            size="sm"
            onClick={onOpenModelSettings}
            ariaLabel={t("common:currentChatModelSettings") as string}
            className="text-text-muted">
            <Gauge className="h-4 w-4" aria-hidden="true" />
            <span>
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
        selectedModel={selectedModel}
        currentProvider={currentProvider}
        promptAssistContextKey={
          promptAssistContextKey ?? serverChatId ?? "playground-draft"
        }
        promptAssistBackendKey={promptAssistBackendKey}
        promptAssistAuthorizationRevision={promptAssistAuthorizationRevision}
        onOpenChange={handlePromptPickerOpen}
        onSelectModel={() =>
          openModelSelector({
            returnFocusSelector: "[data-testid='chat-prompt-select']"
          })
        }
        iconClassName="h-4 w-4"
        className="text-text-muted hover:text-text"
      />
    ),
    [
      selectedSystemPrompt,
      selectedModel,
      currentProvider,
      promptAssistContextKey,
      promptAssistBackendKey,
      promptAssistAuthorizationRevision,
      serverChatId,
      setSelectedQuickPrompt,
      setSelectedSystemPrompt,
      setSystemPrompt,
      systemPrompt,
      handlePromptPickerOpen
    ]
  )

  const characterSelectControl = React.useMemo(
    () => (
      <AssistantSelect
        variant="dropdown"
        onTrackedSelection={props.onTrackedAssistantSelection}
        showLabel
        onOpenChange={handleAssistantPickerOpen}
        iconClassName="h-4 w-4"
        className="text-text-muted hover:text-text"
      />
    ),
    [props.onTrackedAssistantSelection, handleAssistantPickerOpen]
  )

  const rolePlaySetupButton = React.useMemo(() => {
    if (!providedRolePlayActions?.onOpenRolePlaySetup) return null

    const label = t(
      "playground:composer.rolePlaySetup",
      "Role-play setup"
    ) as string

    return (
      <Tooltip title={label}>
        <button
          type="button"
          onClick={providedRolePlayActions.onOpenRolePlaySetup}
          aria-label={label}
          data-testid="composer-role-play-setup"
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-text-muted transition hover:bg-surface2 hover:text-text">
          <Users className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{label}</span>
        </button>
      </Tooltip>
    )
  }, [providedRolePlayActions?.onOpenRolePlaySetup, t])

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
      if (item.id === "messageCount") {
        return "composer-message-count-chip"
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

    const onClick =
      !isProMode && (item.id === "rolePlayIdentity" || item.id === "character")
        ? (providedRolePlayActions?.onOpenRolePlaySetup ?? item.onClick)
        : item.onClick
    if (onClick) {
      return (
        <button
          key={item.id}
          type="button"
          onClick={onClick}
          aria-label={item.value ? `${item.label}: ${item.value}` : item.label}
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
  const advancedChipClass = casualAdvancedControlsOpen
    ? "border-primary/40 bg-primary/10 text-primaryStrong"
    : "border-border bg-surface2 text-text-muted"
  const formattingGuideToggleClass = appendFormattingGuidePrompt
    ? "bg-primary/10 text-primaryStrong"
    : "text-text-muted"
  const customizeLabel = t(
    "playground:composer.customizeChat",
    "Customize chat"
  ) as string
  const researchGroupLabel = t(
    "playground:composer.groups.researchContext",
    "Research & context"
  ) as string
  const characterGroupLabel = t(
    "playground:composer.groups.characterRole",
    "Character & your role"
  ) as string
  const responseGroupLabel = t(
    "playground:composer.groups.responseSettings",
    "Response settings"
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
  const proContextPanelHeading = t(
    "playground:composer.groups.contextSetup",
    "Context setup"
  ) as string
  const proGenerationPanelHeading = t(
    "playground:composer.groups.modelToolsRun",
    "Model, tools, and run"
  ) as string
  const advancedControlsGroupLabel = t(
    "playground:composer.groups.advancedComposerControls",
    "Advanced composer controls"
  ) as string
  const casualAdvancedControlsId = "composer-casual-advanced-controls-row"
  const proAdvancedControlsId = "composer-pro-advanced-controls-row"
  const proContextHeadingId = "composer-pro-context-heading"
  const proGenerationHeadingId = "composer-pro-generation-heading"
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
      ref={customizeTriggerRef}
      type="button"
      onClick={() => {
        if (casualDisclosureLoading) return
        if (casualAdvancedControlsOpen) closeCasualCustomization()
        else void setCasualAdvancedControlsOpen(true)
      }}
      disabled={casualDisclosureLoading}
      aria-busy={casualDisclosureLoading}
      aria-expanded={casualAdvancedControlsOpen}
      aria-controls={
        casualAdvancedControlsOpen ? casualAdvancedControlsId : undefined
      }
      data-testid="composer-casual-advanced-chip"
      className={`inline-flex min-h-[36px] items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium transition hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-50 ${advancedChipClass}`}>
      <span>{customizeLabel}</span>
      <ChevronDown
        className={`h-3 w-3 transition-transform ${casualAdvancedControlsOpen ? "rotate-180" : ""}`}
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
            aria-pressed={!temporaryChat}
            aria-label={persistenceControlLabel}
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition hover:bg-surface2 ${persistenceChipClass} ${ephemeralDisabled ? "cursor-not-allowed opacity-50" : ""}`}>
            <span>{persistenceChipLabel}</span>
          </button>
        </span>
      </Tooltip>
      {toolChoice !== "none" ? (
        <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primaryStrong">
          {t("playground:composer.mcpSelectedSummary", {
            defaultValue: "MCP tools selected: {{choice}}",
            choice:
              toolChoice === "required"
                ? t("playground:composer.toolChoiceRequired", "Required")
                : t("playground:composer.toolChoiceAuto", "Auto")
          })}
        </span>
      ) : null}
      {appendFormattingGuidePrompt ? (
        <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primaryStrong">
          {t(
            "playground:composer.formattingGuideActive",
            "Formatting guide on"
          )}
        </span>
      ) : null}
      {openUIRequestMode ? (
        <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] text-primaryStrong">
          {t(
            "playground:composer.interactiveResponseActive",
            "Interactive response on"
          )}
        </span>
      ) : null}
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
  const persistentContextStrip = !isProMode ? casualContextStrip : contextStrip

  const proAdvancedControls =
    isProMode && !isMobile ? (
      <div className="border-t border-border/50 pt-2">
        <button
          type="button"
          onClick={() => setAdvancedControlsOpen(!advancedControlsOpen)}
          aria-expanded={advancedControlsOpen}
          aria-controls={
            advancedControlsOpen ? proAdvancedControlsId : undefined
          }
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
            id={proAdvancedControlsId}
            role="group"
            aria-label={advancedControlsGroupLabel}
            data-playground-toolbar-row="advanced"
            className="mt-2 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <ParameterPresets compact />
              {formattingGuideControl}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <SystemPromptTemplatesButton
                onSelect={onTemplateSelect}
                systemPrompt={systemPrompt}
                onSystemPromptChange={setSystemPrompt}
              />
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
        {openUIRequestButton}
      </div>
      <div
        data-playground-toolbar-row="actions"
        className="flex flex-wrap items-center justify-between gap-2">
        <div
          role="group"
          aria-label={mobileContextGroupLabel}
          className="flex min-w-0 flex-wrap items-center gap-2">
          {ephemeralToggle}
          {searchContextButton}
          {webSearchButton}
        </div>
        <div
          role="group"
          aria-label={mobileRunInputGroupLabel}
          className="flex min-w-0 flex-wrap items-center justify-end gap-2">
          {modelUsageBadge}
          <ComposerToolbarOverflow
            serverChatId={serverChatId}
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
            rolePlayActions={mobileRolePlayActions}
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

  const dictationActive =
    (speechUsesServer && isServerDictating) ||
    (!speechUsesServer && isListening)
  const casualToolbarLayout = (
    <div
      data-playground-toolbar-layout="casual"
      className="flex min-w-0 flex-col gap-2">
      <div
        role="group"
        aria-label={
          t(
            "playground:composer.quickActions",
            "Composer quick actions"
          ) as string
        }
        data-playground-toolbar-row="primary"
        className="flex min-w-0 flex-wrap items-center gap-2">
        {modelSelectButton ? (
          <span
            data-testid="composer-casual-model-selector-chip"
            className="inline-flex min-w-0 max-w-full items-center">
            {modelSelectButton}
          </span>
        ) : (
          renderContextItem(casualModelItem)
        )}
        {attachmentButton}
        {casualAdvancedControlsToggle}
        {dictationActive ? dictationButton : null}
        {voiceChatEnabled ? voiceChatButton : null}
        {toolbarSendControl}
      </div>
    </div>
  )
  // Keep existing selector event listeners and editor drafts alive while hidden.
  const casualAdvancedControlsPanel = !isProMode ? (
    <section
      id={casualAdvancedControlsId}
      hidden={!casualAdvancedControlsOpen}
      role="region"
      aria-label={customizeLabel}
      data-testid="composer-casual-advanced-controls-row"
      data-playground-toolbar-row="advanced"
      className={`min-w-0 gap-3 rounded-md border border-border/60 bg-surface2/60 p-2 lg:grid-cols-3 ${casualAdvancedControlsOpen ? "grid" : "hidden"}`}>
      <div role="group" aria-label={researchGroupLabel} className="min-w-0">
        <h3 className="mb-2 text-xs font-semibold text-text">
          {researchGroupLabel}
        </h3>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {modeLauncherButton}
          {searchContextButton}
          {webSearchButton}
          {researchLaunchButton}
          {mcpControl}
          {toolsButton}
          <ConnectionStatus showLabel className="px-1 py-0.5" />
        </div>
      </div>
      <div role="group" aria-label={characterGroupLabel} className="min-w-0">
        <h3 className="mb-2 text-xs font-semibold text-text">
          {characterGroupLabel}
        </h3>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {characterSelectControl}
          {rolePlaySetupButton}
          <BuddyManagementButton
            target={
              serverChatId
                ? { scope_type: "conversation", scope_id: serverChatId }
                : null
            }
            onConversationSettings={
              providedRolePlayActions?.onOpenRolePlaySetup
            }
          />
        </div>
      </div>
      <div role="group" aria-label={responseGroupLabel} className="min-w-0">
        <h3 className="mb-2 text-xs font-semibold text-text">
          {responseGroupLabel}
        </h3>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {promptSelectControl}
          {chatSettingsButton}
          <button
            type="button"
            onClick={mobileRolePlayActions.onOpenGenerationStyle}
            className="rounded-md px-2 py-1 text-xs text-text-muted hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus">
            {t("playground:presets.generationStyle", "Generation style")}
          </button>
          <button
            type="button"
            onClick={mobileRolePlayActions.onOpenSystemPrompts}
            className="rounded-md px-2 py-1 text-xs text-text-muted hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus">
            {t("playground:templates.button", "System prompts")}
          </button>
          {compareControl}
          {openUIRequestButton}
          {formattingGuideControl}
          {!dictationActive ? dictationButton : null}
          {!voiceChatEnabled ? voiceChatButton : null}
        </div>
      </div>
    </section>
  ) : null

  const proSplitToolbarLayout = (
    <div
      data-playground-toolbar-layout="pro-split"
      className="flex flex-col gap-2">
      <div className="grid gap-2 lg:grid-cols-2">
        <section
          role="group"
          aria-labelledby={proContextHeadingId}
          data-testid="composer-pro-context-panel"
          className="rounded-md border border-border/60 bg-surface2/60 p-2">
          <h3
            id={proContextHeadingId}
            className="mb-2 text-[11px] font-semibold text-text-muted">
            {proContextPanelHeading}
          </h3>
          <div
            data-playground-toolbar-row="primary"
            className="flex flex-wrap items-center gap-2">
            {modeLauncherButton}
            {compareControl}
            {openUIRequestButton}
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
            {rolePlaySetupButton}
            <BuddyManagementButton
              target={
                serverChatId
                  ? { scope_type: "conversation", scope_id: serverChatId }
                  : null
              }
              onConversationSettings={
                providedRolePlayActions?.onOpenRolePlaySetup
              }
            />
          </div>
        </section>
        <section
          role="group"
          aria-labelledby={proGenerationHeadingId}
          data-testid="composer-pro-generation-panel"
          className="rounded-md border border-border/60 bg-surface2/60 p-2">
          <h3
            id={proGenerationHeadingId}
            className="mb-2 text-[11px] font-semibold text-text-muted">
            {proGenerationPanelHeading}
          </h3>
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
        t(
          "playground:composer.quickActions",
          "Composer quick actions"
        ) as string
      }
      data-playground-toolbar-row="collapsed-actions"
      className="flex items-center justify-end gap-2">
      {attachmentButton}
    </div>
  ) : null

  return (
    <div
      ref={toolbarRef}
      onFocusCapture={(event) => {
        lastFocusedControlRef.current = event.target as HTMLElement
      }}
      onKeyDown={handleCustomizationKeyDown}
      data-testid="composer-options-panel"
      className="mt-2 flex min-w-0 flex-col gap-1">
      <SystemPromptTemplatesModal
        open={systemPromptsOpen}
        onClose={() => setSystemPromptsOpen(false)}
        onSelect={handleMobileTemplateSelect}
        systemPrompt={systemPrompt}
        onSystemPromptChange={setSystemPrompt}
      />
      <Modal
        title={t("playground:presets.title", "Generation style") as string}
        open={generationStyleOpen}
        onCancel={() => setGenerationStyleOpen(false)}
        footer={null}
        width={420}
        className="generation-style-modal">
        <ParameterPresetsDropdown
          onChange={() => setGenerationStyleOpen(false)}
        />
      </Modal>
      {!isProMode ? (
        <>
          {casualToolbarLayout}
          {persistentContextStrip}
          {casualAdvancedControlsPanel}
          {casualDisclosureUnavailable ? (
            <p role="status" className="text-xs text-text-muted">
              {t(
                "playground:composer.customizePreferenceUnavailable",
                "Display preference unavailable. Customize chat still works for this visit."
              )}
            </p>
          ) : null}
        </>
      ) : optionsExpanded ? (
        <>
          {isMobile ? mobileToolbarLayout : proSplitToolbarLayout}

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
          {persistentContextStrip}
        </>
      ) : (
        <>
          {collapsedMobileToolbarLayout}
          {!isMobile ? persistentContextStrip : null}
        </>
      )}
    </div>
  )
})
