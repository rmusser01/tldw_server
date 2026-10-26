import React from "react";
import { Modal, Tooltip } from "antd";
import {
  ChevronDown,
  FileText,
  Gauge,
  Globe,
  MicIcon,
  Search,
  SlidersHorizontal,
  Users,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  getPersistenceModeControlLabel,
  getPersistenceModePillLabel,
} from "@/hooks/playground/usePersistenceMode";
import type { AssistantSelection } from "@/types/assistant-selection";
import { useStorage } from "@plasmohq/storage/hook";
import { AssistantSelect } from "@/components/Common/AssistantSelect";
import { BuddyManagementButton } from "@/components/Common/PersonaBuddy/BuddyManagementButton";
import { PromptSelect } from "@/components/Common/PromptSelect";
import { PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY } from "@/utils/output-formatting-guide";
import type { ComposerToolbarRolePlayActions } from "./ComposerToolbarOverflow";
import { composerShortcutHint } from "./composer-shortcut-hint";
import {
  ParameterPresets,
  ParameterPresetsDropdown,
  SessionCostEstimation,
  SystemPromptTemplatesModal,
  type PromptTemplate,
} from "./playground-features";
import {
  openModelSelector,
  OPEN_ASSISTANT_SELECT_EVENT,
  OPEN_PROMPT_SELECT_EVENT,
} from "./playground-cockpit-actions";

export type ComposerToolbarProps = {
  isProMode: boolean;
  isMobile: boolean;
  isConnectionReady: boolean;
  isSending: boolean;
  /** '@' tab mentions work here (CC-11); the footer hint names them only then. */
  mentionsAvailable?: boolean;
  optionsExpanded?: boolean;
  optionsLoading?: boolean;
  optionsUnavailable?: boolean;
  nestedDisclosureOpen?: boolean;
  focusFallbackRef?: React.RefObject<HTMLElement | null>;
  openUIRequestMode?: boolean;
  toolChoice?: "auto" | "none" | "required";
  trackedAssistantSelectionLoading?: boolean;
  onTrackedAssistantSelection?: (
    selection: AssistantSelection,
    isCurrent?: () => boolean,
  ) => boolean | Promise<boolean>;
  onToggleOptions: () => void;
  onRolePlayTriggerCapture?: (element: HTMLButtonElement) => void;
  promptAssistAction?: React.ReactNode;
  sendControlPlacement?: "toolbar" | "external";
  modeLauncherButton?: React.ReactNode;
  compareControl?: React.ReactNode;
  openUIRequestButton?: React.ReactNode;
  // Row 1: primary selectors (pre-rendered by parent)
  modelSelectButton: React.ReactNode;
  mcpControl: React.ReactNode;
  // Row action controls (pre-rendered by parent)
  sendControl: React.ReactNode;
  researchLaunchButton?: React.ReactNode;
  attachmentButton: React.ReactNode;
  toolsButton: React.ReactNode;
  voiceChatButton: React.ReactNode;
  modelUsageBadge: React.ReactNode;
  // Prompt select
  selectedSystemPrompt: string | undefined;
  systemPrompt: string | undefined;
  setSystemPrompt: (prompt: string | undefined) => void;
  setSelectedSystemPrompt: (id: string | undefined) => void;
  setSelectedQuickPrompt: (prompt: string | undefined) => void;
  // Ephemeral toggle
  temporaryChat: boolean;
  onToggleTemporaryChat: (next: boolean) => void;
  privateChatLocked: boolean;
  isFireFoxPrivateMode: boolean;
  persistenceTooltip: React.ReactNode;
  // Knowledge / context
  contextToolsOpen: boolean;
  onToggleKnowledgePanel: (tab: string) => void;
  onOpenKnowledgePanel: (tab: string) => void;
  // Web search
  webSearch: boolean;
  onToggleWebSearch: () => void;
  hasWebSearch: boolean;
  // Model settings
  onOpenModelSettings: () => void;
  modelSummaryLabel: string;
  promptSummaryLabel: string;
  // Dictation
  hasDictation: boolean;
  speechAvailable: boolean;
  speechUsesServer: boolean;
  isListening: boolean;
  isServerDictating: boolean;
  voiceChatEnabled: boolean;
  speechTooltip: string;
  onDictationToggle: () => void;
  // Pro-only: parameter presets & templates
  onTemplateSelect: (template: PromptTemplate) => void;
  // Pro-only: cost estimation
  selectedModel: string | null;
  currentProvider?: string | null;
  resolvedProviderKey: string;
  messages: any[];
  // Pro-only: context counts
  selectedDocumentsCount: number;
  uploadedFilesCount: number;
  // Persistence hints
  serverChatId: string | null;
  promptAssistContextKey?: string;
  promptAssistBackendKey?: string | null;
  promptAssistAuthorizationRevision?: string | null;
  showServerPersistenceHint: boolean;
  onDismissServerPersistenceHint: () => void;
  onFocusConnectionCard: () => void;
  contextItems?: ComposerContextItem[];
  rolePlayActions?: ComposerToolbarRolePlayActions;
};

export type ComposerContextItem = {
  id: string;
  label: string;
  value?: string;
  tone?: "neutral" | "active" | "warning";
  onClick?: () => void;
};

/** Shared composer footer; tools stay mounted so starter events can open their selectors. */
export const ComposerToolbar = React.memo(function ComposerToolbar(
  props: ComposerToolbarProps,
) {
  const { t } = useTranslation([
    "playground",
    "common",
    "option",
    "sidepanel",
    "settings",
  ]);
  const {
    isMobile,
    isProMode,
    isConnectionReady,
    mentionsAvailable = false,
    optionsExpanded = false,
    optionsLoading = false,
    optionsUnavailable = false,
    nestedDisclosureOpen = false,
    openUIRequestMode = false,
    toolChoice = "none",
    onToggleOptions,
    onRolePlayTriggerCapture,
    modelSelectButton,
    attachmentButton,
    sendControl,
    sendControlPlacement = "toolbar",
    modeLauncherButton,
    mcpControl,
    compareControl,
    openUIRequestButton,
    researchLaunchButton,
    toolsButton,
    voiceChatButton,
    promptAssistAction,
    modelUsageBadge,
    selectedSystemPrompt,
    systemPrompt,
    setSystemPrompt,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    selectedModel,
    currentProvider,
    resolvedProviderKey,
    messages,
    temporaryChat,
    onToggleTemporaryChat,
    privateChatLocked,
    isFireFoxPrivateMode,
    persistenceTooltip,
    contextToolsOpen,
    onToggleKnowledgePanel,
    onOpenKnowledgePanel,
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
    rolePlayActions,
  } = props;
  const toggleRef = React.useRef<HTMLButtonElement>(null);
  const toolbarRef = React.useRef<HTMLDivElement>(null);
  const lastFocusedControlRef = React.useRef<HTMLElement | null>(null);
  const previousPickerOpenRef = React.useRef(false);
  const [assistantPickerOpen, setAssistantPickerOpen] = React.useState(false);
  const [promptPickerOpen, setPromptPickerOpen] = React.useState(false);
  const pendingCustomizationRevealRef = React.useRef(false);
  const handleAssistantPickerOpen = React.useCallback((open: boolean) => {
    setAssistantPickerOpen(open);
    if (open) pendingCustomizationRevealRef.current = true;
  }, []);
  const handlePromptPickerOpen = React.useCallback((open: boolean) => {
    setPromptPickerOpen(open);
    if (open) pendingCustomizationRevealRef.current = true;
  }, []);
  React.useEffect(() => {
    if (!assistantPickerOpen && !promptPickerOpen)
      pendingCustomizationRevealRef.current = false;
    if (!pendingCustomizationRevealRef.current || optionsLoading) return;
    pendingCustomizationRevealRef.current = false;
    if (!optionsExpanded) onToggleOptions();
  }, [
    assistantPickerOpen,
    promptPickerOpen,
    optionsLoading,
    optionsExpanded,
    onToggleOptions,
  ]);
  React.useEffect(() => {
    const revealTools = () => {
      if (optionsLoading) pendingCustomizationRevealRef.current = true;
      else if (!optionsExpanded) onToggleOptions();
    };
    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, revealTools);
    window.addEventListener(OPEN_PROMPT_SELECT_EVENT, revealTools);
    return () => {
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, revealTools);
      window.removeEventListener(OPEN_PROMPT_SELECT_EVENT, revealTools);
    };
  }, [optionsExpanded, optionsLoading, onToggleOptions]);
  const [appendFormattingGuidePrompt, setAppendFormattingGuidePrompt] =
    useStorage(PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY, false);
  const [systemPromptsOpen, setSystemPromptsOpen] = React.useState(false);
  const [generationStyleOpen, setGenerationStyleOpen] = React.useState(false);
  // Retained hidden controls and detached portals are both invalid return targets.
  // Repair only focus still owned by that disappearing control, never another task.
  React.useLayoutEffect(() => {
    const isVisible = (element: HTMLElement | null) => {
      if (!element?.isConnected || ("disabled" in element && element.disabled))
        return false;
      let ancestor: HTMLElement | null = element;
      while (ancestor) {
        const style = window.getComputedStyle(ancestor);
        if (
          ancestor.hidden ||
          ancestor.getAttribute("aria-hidden") === "true" ||
          style.display === "none" ||
          style.visibility === "hidden" ||
          style.visibility === "collapse" ||
          (ancestor.tagName === "DETAILS" &&
            !ancestor.hasAttribute("open") &&
            !ancestor.querySelector("summary")?.contains(element))
        )
          return false;
        ancestor = ancestor.parentElement;
      }
      return true;
    };
    const repairOwnedFocus = () => {
      const previous = lastFocusedControlRef.current;
      if (!previous || isVisible(previous)) return;
      if (
        document.activeElement !== document.body &&
        document.activeElement !== previous
      )
        return;
      const summary =
        previous
          .closest("details:not([open])")
          ?.querySelector<HTMLElement>("summary") ?? null;
      if (isVisible(summary)) {
        summary?.focus();
        return;
      }
      if (!optionsLoading && isVisible(toggleRef.current)) {
        toggleRef.current?.focus();
        return;
      }
      const candidates =
        toolbarRef.current?.querySelectorAll<HTMLButtonElement>(
          "button:not(:disabled)",
        );
      const target = Array.from(candidates ?? []).find(isVisible);
      if (target) target.focus();
      else if (isVisible(props.focusFallbackRef?.current ?? null))
        props.focusFallbackRef?.current?.focus();
    };
    const pickerOpen = assistantPickerOpen || promptPickerOpen;
    const pickerClosing = previousPickerOpenRef.current && !pickerOpen;
    previousPickerOpenRef.current = pickerOpen;
    // Let an existing picker return to its visible trigger before qualifying it.
    if (!pickerClosing) repairOwnedFocus();
    // Existing picker focus return is scheduled; qualify its eventual target too.
    const frame = window.requestAnimationFrame(repairOwnedFocus);
    return () => window.cancelAnimationFrame(frame);
  }, [
    isProMode,
    isMobile,
    optionsExpanded,
    optionsLoading,
    assistantPickerOpen,
    promptPickerOpen,
    nestedDisclosureOpen,
    systemPromptsOpen,
    generationStyleOpen,
    props.focusFallbackRef,
  ]);

  const dictationActive = speechUsesServer ? isServerDictating : isListening;
  const persistenceLabel = getPersistenceModePillLabel(t, temporaryChat);
  const persistenceControlLabel = getPersistenceModeControlLabel(
    t,
    temporaryChat,
  );
  const controlClass =
    "inline-flex min-h-11 items-center gap-2 rounded-lg px-3 py-2 text-sm text-text-muted transition-colors hover:bg-surface2 hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus";
  const activeContextItems = contextItems.filter(
    (item) =>
      item.id !== "model" &&
      item.id !== "temporary" &&
      (item.tone === "active" ||
        item.tone === "warning" ||
        [
          "character",
          "rolePlayIdentity",
          "rolePlayBehavior",
          "rolePlayScene",
        ].includes(item.id)),
  );
  const neutralContextItems = contextItems.filter(
    (item) => item.tone === "neutral" && !activeContextItems.includes(item),
  );
  const dictationControl = hasDictation ? (
    <Tooltip title={speechTooltip}>
      <button
        type="button"
        onClick={onDictationToggle}
        disabled={!speechAvailable || voiceChatEnabled}
        aria-label={
          !speechAvailable
            ? (t(
                "playground:actions.speechUnavailableTitle",
                "Dictation unavailable",
              ) as string)
            : dictationActive
              ? (t("playground:actions.speechStop", "Stop dictation") as string)
              : (t(
                  "playground:actions.speechStart",
                  "Start dictation",
                ) as string)
        }
        aria-pressed={dictationActive}
        data-testid="dictation-button"
        className={`${controlClass} disabled:cursor-not-allowed disabled:opacity-50 ${dictationActive ? "bg-primary/10 text-primaryStrong" : ""}`}
      >
        <MicIcon className="h-4 w-4" aria-hidden="true" />
        <span className="hidden sm:inline">
          {dictationActive
            ? t("playground:actions.speechStop", "Stop dictation")
            : t("playground:actions.speechStart", "Start dictation")}
        </span>
      </button>
    </Tooltip>
  ) : null;

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
            returnFocusSelector: "[data-testid='chat-prompt-select']",
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
      handlePromptPickerOpen,
    ],
  );

  const characterSelectControl = React.useMemo(
    () => (
      <AssistantSelect
        variant="dropdown"
        onTrackedSelection={props.onTrackedAssistantSelection}
        trackedSelectionLoading={props.trackedAssistantSelectionLoading}
        showLabel
        onOpenChange={handleAssistantPickerOpen}
        iconClassName="h-4 w-4"
        className="text-text-muted hover:text-text"
      />
    ),
    [
      props.onTrackedAssistantSelection,
      props.trackedAssistantSelectionLoading,
      handleAssistantPickerOpen,
    ],
  );

  return (
    <div
      ref={toolbarRef}
      data-testid="composer-toolbar"
      onFocusCapture={(event) => {
        lastFocusedControlRef.current = event.target as HTMLElement;
      }}
      onKeyDown={(event) => {
        if (
          event.key === "Escape" &&
          !event.defaultPrevented &&
          !event.nativeEvent.isComposing &&
          !nestedDisclosureOpen &&
          !assistantPickerOpen &&
          !promptPickerOpen &&
          !systemPromptsOpen &&
          !generationStyleOpen &&
          !(event.target as HTMLElement).closest(
            '[role="dialog"], [role="listbox"], [role="menu"]',
          ) &&
          optionsExpanded
        ) {
          event.preventDefault();
          event.stopPropagation();
          onToggleOptions();
          toggleRef.current?.focus();
        }
      }}
      className="w-full"
    >
      <SystemPromptTemplatesModal
        open={systemPromptsOpen}
        onClose={() => setSystemPromptsOpen(false)}
        onSelect={(template: PromptTemplate) => {
          onTemplateSelect(template);
          setSystemPromptsOpen(false);
        }}
        systemPrompt={systemPrompt}
        onSystemPromptChange={setSystemPrompt}
      />
      <Modal
        title={t("playground:presets.title", "Generation style") as string}
        open={generationStyleOpen}
        onCancel={() => setGenerationStyleOpen(false)}
        footer={null}
        width={420}
      >
        <ParameterPresetsDropdown
          onChange={() => setGenerationStyleOpen(false)}
        />
      </Modal>

      {(activeContextItems.length > 0 ||
        selectedDocumentsCount > 0 ||
        uploadedFilesCount > 0) && (
        <div
          data-testid="composer-context-strip"
          aria-label={
            t("playground:composer.activeContext", "Active context") as string
          }
          className="flex flex-wrap items-center gap-2 px-2 pb-3"
        >
          {activeContextItems.map((item) => {
            const className = `inline-flex min-h-9 items-center gap-1.5 rounded-md px-2 py-1 text-xs ${item.tone === "warning" ? "bg-warn/10 text-text" : "bg-surface2 text-text"}`;
            const content = (
              <>
                <span className="font-medium">{item.label}</span>
                {item.value && (
                  <span className="max-w-[16rem] truncate">{item.value}</span>
                )}
              </>
            );
            const testId =
              item.id === "sessionStatus"
                ? "composer-session-status-chip"
                : undefined;
            return item.onClick ? (
              <button
                key={item.id}
                type="button"
                onClick={
                  !isProMode &&
                  (item.id === "rolePlayIdentity" || item.id === "character") &&
                  rolePlayActions?.onOpenRolePlaySetup
                    ? rolePlayActions.onOpenRolePlaySetup
                    : item.onClick
                }
                data-testid={testId}
                title={item.value ? `${item.label}: ${item.value}` : item.label}
                className={`${className} transition-colors hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus`}
              >
                {content}
              </button>
            ) : (
              <span
                key={item.id}
                data-testid={testId}
                title={item.value ? `${item.label}: ${item.value}` : item.label}
                className={className}
              >
                {content}
              </span>
            );
          })}
          {selectedDocumentsCount > 0 && (
            <button
              type="button"
              onClick={() => onOpenKnowledgePanel("context")}
              className={controlClass}
            >
              <FileText className="h-4 w-4" aria-hidden="true" />
              {t("playground:composer.contextTabs", {
                defaultValue: "{{count}} tabs",
                count: selectedDocumentsCount,
              })}
            </button>
          )}
          {uploadedFilesCount > 0 && (
            <button
              type="button"
              onClick={() => onOpenKnowledgePanel("context")}
              className={controlClass}
            >
              <FileText className="h-4 w-4" aria-hidden="true" />
              {t("playground:composer.contextFiles", {
                defaultValue: "{{count}} files",
                count: uploadedFilesCount,
              })}
            </button>
          )}
        </div>
      )}

      {optionsUnavailable && (
        <p role="status" className="px-2 text-xs text-text-muted">
          {t(
            "playground:composer.toolsPreferenceUnavailable",
            "Display preference unavailable. Tools still work for this visit.",
          )}
        </p>
      )}
      {(toolChoice !== "none" ||
        appendFormattingGuidePrompt ||
        openUIRequestMode) && (
        <div className="flex flex-wrap gap-2 px-2 pb-2 text-xs text-text-muted">
          {toolChoice !== "none" && (
            <span>
              {t("playground:composer.mcpSelectedSummary", {
                defaultValue: "MCP tools selected: {{choice}}",
                choice:
                  toolChoice === "required"
                    ? t("playground:composer.toolChoiceRequired", "Required")
                    : t("playground:composer.toolChoiceAuto", "Auto"),
              })}
            </span>
          )}
          {appendFormattingGuidePrompt && (
            <span>
              {t(
                "playground:composer.formattingGuideActive",
                "Formatting guide on",
              )}
            </span>
          )}
          {openUIRequestMode && (
            <span>
              {t(
                "playground:composer.interactiveResponseActive",
                "Interactive response on",
              )}
            </span>
          )}
        </div>
      )}
      <section
        id="composer-options-panel"
        data-testid="composer-options-panel"
        role="region"
        aria-label={
          t("playground:composer.toolsPanel", "Composer tools") as string
        }
        hidden={!optionsExpanded}
        className="mb-3 max-h-[35dvh] overflow-y-auto overscroll-contain border-y border-border py-4 [scrollbar-color:rgb(var(--color-border))_transparent]"
      >
        <div className="grid gap-5 px-2 md:grid-cols-3 md:gap-0 md:divide-x md:divide-border">
          <section
            aria-labelledby="composer-context-heading"
            className="min-w-0 md:pr-4"
          >
            <h2
              id="composer-context-heading"
              className="mb-2 text-sm font-semibold text-text"
            >
              {t("playground:composer.groups.contextSetup", "Context setup")}
            </h2>
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-1">
                <button
                  type="button"
                  onClick={() => onToggleKnowledgePanel("search")}
                  aria-expanded={contextToolsOpen}
                  aria-pressed={contextToolsOpen}
                  data-testid="knowledge-search-toggle"
                  className={controlClass}
                >
                  <Search className="h-4 w-4" aria-hidden="true" />
                  {contextToolsOpen
                    ? t(
                        "playground:composer.contextKnowledgeClose",
                        "Close Search & Context",
                      )
                    : t(
                        "playground:composer.contextKnowledge",
                        "Search & Context",
                      )}
                </button>
                {hasWebSearch && (
                  <button
                    type="button"
                    onClick={onToggleWebSearch}
                    aria-pressed={webSearch}
                    aria-label={
                      t("playground:tools.webSearch", "Web search") as string
                    }
                    data-testid="web-search-toggle"
                    className={`${controlClass} ${webSearch ? "bg-primary/10 text-primaryStrong" : ""}`}
                  >
                    <Globe className="h-4 w-4" aria-hidden="true" />
                    {t("playground:tools.webSearch", "Web search")}
                  </button>
                )}
                {researchLaunchButton}
              </div>
              <div className="flex flex-wrap items-center gap-1">
                {modeLauncherButton}
                {mcpControl}
              </div>
            </div>
          </section>
          <section
            aria-labelledby="composer-instructions-heading"
            className="min-w-0 border-t border-border pt-4 md:border-0 md:px-4 md:pt-0"
          >
            <h2
              id="composer-instructions-heading"
              className="mb-2 text-sm font-semibold text-text"
            >
              {t(
                "playground:composer.instructionsAndPeople",
                "Prompts & people",
              )}
            </h2>
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-1">
                {promptSelectControl}
                <span aria-hidden="true" className="h-4 w-px bg-border" />
                <button
                  type="button"
                  onClick={
                    rolePlayActions?.onOpenSystemPrompts ??
                    (() => setSystemPromptsOpen(true))
                  }
                  className={controlClass}
                >
                  {t("playground:templates.button", "System prompts")}
                </button>
                {promptAssistAction}
              </div>
              <div className="flex flex-wrap items-center gap-1">
                {characterSelectControl}
              </div>
              <div className="flex flex-wrap items-center gap-1">
                <div
                  onClickCapture={(event) => {
                    const button = event.currentTarget.querySelector("button");
                    if (button) onRolePlayTriggerCapture?.(button);
                  }}
                >
                  <BuddyManagementButton
                    target={
                      serverChatId
                        ? { scope_type: "conversation", scope_id: serverChatId }
                        : null
                    }
                    onConversationSettings={
                      rolePlayActions?.onOpenRolePlaySetup
                    }
                  />
                </div>
                {rolePlayActions?.onOpenRolePlaySetup && (
                  <button
                    type="button"
                    onClick={rolePlayActions.onOpenRolePlaySetup}
                    data-testid="composer-role-play-setup"
                    className={controlClass}
                  >
                    <Users className="h-4 w-4" aria-hidden="true" />
                    {t("playground:composer.rolePlaySetup", "Role-play setup")}
                  </button>
                )}
              </div>
            </div>
          </section>
          <section
            aria-labelledby="composer-response-heading"
            className="min-w-0 border-t border-border pt-4 md:border-0 md:pl-4 md:pt-0"
          >
            <h2
              id="composer-response-heading"
              className="mb-2 text-sm font-semibold text-text"
            >
              {t("playground:composer.responseAndTools", "Response & tools")}
            </h2>
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-1">
                <button
                  type="button"
                  onClick={onOpenModelSettings}
                  className={controlClass}
                  title={`${modelSummaryLabel} · ${promptSummaryLabel}`}
                >
                  <Gauge className="h-4 w-4" aria-hidden="true" />
                  {t("playground:composer.chatSettings", "Chat Settings")}
                </button>
                <button
                  type="button"
                  onClick={
                    rolePlayActions?.onOpenGenerationStyle ??
                    (() => setGenerationStyleOpen(true))
                  }
                  className={controlClass}
                >
                  {t("playground:presets.generationStyle", "Generation style")}
                </button>
                {isProMode && <ParameterPresets />}
                {compareControl}
              </div>
              <div className="flex flex-wrap items-center gap-1">
                <button
                  type="button"
                  onClick={() =>
                    setAppendFormattingGuidePrompt(!appendFormattingGuidePrompt)
                  }
                  aria-pressed={appendFormattingGuidePrompt}
                  data-testid="composer-formatting-guide-toggle"
                  className={`${controlClass} ${appendFormattingGuidePrompt ? "bg-primary/10 text-primaryStrong" : ""}`}
                >
                  <FileText className="h-4 w-4" aria-hidden="true" />
                  {t(
                    "playground:composer.outputFormattingGuide",
                    "Formatting guide",
                  )}
                </button>
                {openUIRequestButton}
                {toolsButton}
                {messages.length > 0 && modelUsageBadge}
                {isProMode && messages.length > 0 && (
                  <SessionCostEstimation
                    modelId={selectedModel}
                    provider={resolvedProviderKey}
                    messages={messages}
                  />
                )}
                {neutralContextItems.map((item) =>
                  item.onClick ? (
                    <button
                      key={item.id}
                      type="button"
                      onClick={item.onClick}
                      className={controlClass}
                    >
                      {item.label}: {item.value}
                    </button>
                  ) : (
                    <span key={item.id} className="text-xs text-text-muted">
                      {item.label}: {item.value}
                    </span>
                  ),
                )}
              </div>
            </div>
          </section>
        </div>
        {isProMode && !temporaryChat && !isConnectionReady && (
          <button
            type="button"
            onClick={onFocusConnectionCard}
            className={controlClass}
          >
            {t(
              "playground:composer.persistence.connectToSave",
              "Connect your server to sync chats.",
            )}
          </button>
        )}
        {isProMode &&
          !temporaryChat &&
          serverChatId &&
          showServerPersistenceHint && (
            <p className="mt-3 px-2 text-xs text-text-muted">
              {t(
                "playground:composer.persistence.serverInlineTitle",
                "Saved locally + on your server",
              )}
              {": "}
              {t(
                "playground:composer.persistence.serverInlineBody",
                "This chat is stored both in this browser and on your tldw server, so you can reopen it from server history, keep a long-term record, and analyze it alongside other conversations.",
              )}
              <button
                type="button"
                onClick={onDismissServerPersistenceHint}
                className="ml-2 text-primaryStrong"
              >
                {t("common:dismiss", "Dismiss")}
              </button>
            </p>
          )}
      </section>

      <div
        data-testid="composer-bottom-bar"
        className="flex flex-wrap items-center justify-between gap-x-2 gap-y-2 px-1"
      >
        <div className="flex max-w-full shrink-0 flex-wrap items-center gap-1">
          {attachmentButton}
          <button
            ref={toggleRef}
            type="button"
            onClick={onToggleOptions}
            disabled={optionsLoading}
            aria-busy={optionsLoading}
            data-testid="composer-options-toggle"
            aria-controls="composer-options-panel"
            aria-expanded={optionsExpanded}
            className={`${controlClass} ${optionsExpanded ? "bg-surface2 text-text" : ""}`}
          >
            <SlidersHorizontal
              className="hidden h-4 w-4 sm:block"
              aria-hidden="true"
            />
            <span>{t("playground:composer.toolsLabel", "Tools")}</span>
            <ChevronDown
              className={`h-3 w-3 transition-transform motion-reduce:transition-none ${optionsExpanded ? "rotate-180" : ""}`}
              aria-hidden="true"
            />
          </button>
          <span className="text-xs text-text-muted sm:px-2">
            {composerShortcutHint(t, mentionsAvailable)}
          </span>
        </div>
        <div
          className={`flex min-w-0 items-center gap-2 ${isMobile ? "w-full justify-between" : "flex-1 justify-end"}`}
        >
          <Tooltip title={persistenceTooltip}>
            <span className="text-xs">
              <button
                type="button"
                onClick={() => onToggleTemporaryChat(!temporaryChat)}
                disabled={privateChatLocked || isFireFoxPrivateMode}
                aria-pressed={!temporaryChat}
                aria-label={persistenceControlLabel}
                data-testid="composer-persistence"
                className={`min-h-9 rounded-md px-2 transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed ${temporaryChat ? "bg-warn/10 text-text" : "text-text-muted"}`}
              >
                {persistenceLabel}
              </button>
            </span>
          </Tooltip>
          <div
            hidden={!optionsExpanded && !dictationActive && !voiceChatEnabled}
            onClickCapture={() => {
              if (!optionsExpanded) toggleRef.current?.focus();
            }}
            className={`${optionsExpanded || dictationActive || voiceChatEnabled ? "flex" : ""} shrink-0 items-center gap-1`}
          >
            {dictationControl}
            {voiceChatButton}
          </div>
          <div
            className={`min-w-0 ${isMobile ? "max-w-[10rem]" : "max-w-[18rem]"} [&>button]:max-w-full`}
          >
            {modelSelectButton}
          </div>
          <div
            data-testid="composer-inline-send-control"
            className="flex shrink-0 items-center gap-2"
          >
            {sendControlPlacement === "toolbar" && sendControl}
          </div>
        </div>
      </div>
    </div>
  );
});
