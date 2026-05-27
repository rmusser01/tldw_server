import React from "react";
import { useTranslation } from "react-i18next";
import {
  DEGRADED_STATE_LABEL,
  ERROR_STATE_LABEL,
  READY_STATE_LABEL,
} from "@/design-system";
import {
  RotateCcw,
  Settings2,
  SlidersHorizontal,
  Square,
  UserRound,
  Wrench,
} from "lucide-react";
import type { ChatModelUsabilityStatus } from "@/utils/chat-model-availability";
import {
  formatCockpitMessageCount,
  useCockpitMessageCount,
} from "./playground-cockpit-state";
import {
  cockpitRailDisabledActionClass,
  cockpitRailStyles,
  cockpitRailToneClass,
} from "./playground-cockpit-rail-styles";
import { PlaygroundRailSection } from "./PlaygroundRailSection";

export type RuntimeSettingSummary = {
  label: string;
  value: string;
  source?: "default" | "override";
};

export type RuntimeToolStateCount = {
  label: string;
  value: number | string;
};

export type RuntimeToolSummary = {
  state: "available" | "unavailable" | "disabled" | "degraded";
  label: string;
  detail?: string | null;
  stateCounts?: RuntimeToolStateCount[];
  onOpen?: () => void;
};

export type RuntimeToolChoice = "auto" | "required" | "none";

export type RuntimeAssistantSummary = {
  mode: "none" | "character" | "persona";
  name?: string | null;
  detail?: string | null;
};

export type PlaygroundRuntimeInspectorProps = {
  streaming: boolean;
  selectedProvider?: string | null | undefined;
  selectedModel: string | null | undefined;
  providerRouteLabel?: string | null;
  modelUsabilityStatus?: ChatModelUsabilityStatus | null;
  modelUsabilityCanSend?: boolean | null;
  modelUsabilityDetail?: string | null;
  runtimeStatus?: "ready" | "streaming" | "loading" | "error" | "degraded";
  runtimeStatusDetail?: string | null;
  messageCount: number;
  threadSearchOpen: boolean;
  selectedCharacterName?: string | null | undefined;
  assistantSummary?: RuntimeAssistantSummary;
  onOpenModelSettings: () => void;
  onOpenCharacterSettings?: () => void;
  onOpenAssistantSelect?: () => void;
  onClearAssistant?: () => void;
  onInspectAssistant?: () => void;
  onOpenSceneDirector?: () => void;
  toolChoice?: RuntimeToolChoice;
  onToolChoiceChange?: (choice: RuntimeToolChoice) => void;
  onOpenMcpSettings?: () => void;
  canStopStreaming?: boolean;
  onStopStreaming?: () => void;
  canRegenerate?: boolean;
  onRegenerate?: () => void;
  emptyAssistantResponse?: boolean;
  settingSummaries?: RuntimeSettingSummary[];
  toolSummary?: RuntimeToolSummary | null;
};

const toolStateClass = (state: RuntimeToolSummary["state"]) => {
  if (state === "available") return cockpitRailToneClass("success");
  if (state === "degraded") return cockpitRailToneClass("warning");
  return cockpitRailToneClass("muted");
};

export const PlaygroundRuntimeInspector = ({
  streaming,
  selectedProvider,
  selectedModel,
  providerRouteLabel,
  modelUsabilityStatus = null,
  modelUsabilityCanSend = null,
  modelUsabilityDetail = null,
  runtimeStatus,
  runtimeStatusDetail,
  messageCount,
  threadSearchOpen,
  selectedCharacterName,
  assistantSummary,
  onOpenModelSettings,
  onOpenCharacterSettings,
  onOpenAssistantSelect,
  onClearAssistant,
  onInspectAssistant,
  onOpenSceneDirector,
  toolChoice,
  onToolChoiceChange,
  onOpenMcpSettings,
  canStopStreaming = false,
  onStopStreaming,
  canRegenerate = false,
  onRegenerate,
  emptyAssistantResponse = false,
  settingSummaries = [],
  toolSummary = null,
}: PlaygroundRuntimeInspectorProps) => {
  const { t } = useTranslation("playground");
  const runControlsId = React.useId();
  const effectiveMessageCount = useCockpitMessageCount(messageCount);
  const selectedModelHasProvider = Boolean(selectedModel?.includes(":"));
  const selectedModelSeparator = selectedModel?.indexOf(":") ?? -1;
  let displayProvider = selectedProvider || null;
  let displayModel = selectedModel || null;

  if (!selectedProvider && selectedModelSeparator > 0) {
    displayProvider = selectedModel?.slice(0, selectedModelSeparator) || null;
    displayModel = selectedModel?.slice(selectedModelSeparator + 1) || null;
  } else if (
    selectedProvider &&
    selectedModel?.startsWith(`${selectedProvider}:`)
  ) {
    displayModel = selectedModel.slice(selectedProvider.length + 1);
  }

  const routeLabel =
    providerRouteLabel ||
    (selectedModelHasProvider
      ? selectedModel
      : displayProvider && displayModel
        ? `${displayProvider}:${displayModel}`
        : selectedModel || null);
  const modelUsabilityBlocks =
    Boolean(modelUsabilityStatus) &&
    modelUsabilityStatus !== "ready" &&
    !(
      modelUsabilityStatus === "degraded" &&
      modelUsabilityCanSend !== false
    );
  const status =
    runtimeStatus ||
    (streaming
      ? "streaming"
      : modelUsabilityStatus === "loading"
        ? "loading"
        : modelUsabilityBlocks
          ? "error"
          : "ready");
  const statusLabel =
    status === "streaming"
      ? t("cockpit.streaming", "Streaming")
      : status === "loading"
        ? t("cockpit.modelChecking", "Checking model")
      : status === "degraded"
        ? t("cockpit.degraded", DEGRADED_STATE_LABEL)
        : status === "error"
          ? t("cockpit.error", ERROR_STATE_LABEL)
          : t("cockpit.ready", READY_STATE_LABEL);
  const messageLabel = formatCockpitMessageCount(
    t("cockpit.messageCount", {
      count: effectiveMessageCount,
      defaultValue: "{{count}} messages",
      defaultValue_one: "{{count}} message",
    }),
    effectiveMessageCount,
  );
  const effectiveAssistantSummary: RuntimeAssistantSummary =
    assistantSummary ||
    (selectedCharacterName
      ? {
          mode: "character",
          name: selectedCharacterName,
          detail: t("cockpit.characterSelected", "Character selected"),
        }
      : {
          mode: "none",
          name: null,
          detail: t(
            "cockpit.noAssistantDetail",
            "No persona or character will shape replies.",
          ),
        });
  const assistantLabel =
    effectiveAssistantSummary.name ||
    t("cockpit.noAssistantSelected", "No assistant selected");
  const noAssistantDetail = t(
    "cockpit.noAssistantDetail",
    "No persona or character will shape replies.",
  );
  const assistantModeLabel =
    effectiveAssistantSummary.mode === "persona"
      ? t("cockpit.personaMode", "Persona")
      : effectiveAssistantSummary.mode === "character"
        ? t("cockpit.characterMode", "Character")
        : t("cockpit.noAssistantMode", "None");
  const assistantDetail =
    effectiveAssistantSummary.mode === "none" &&
    effectiveAssistantSummary.detail === assistantLabel
      ? noAssistantDetail
      : effectiveAssistantSummary.detail ||
        (effectiveAssistantSummary.mode === "none"
          ? noAssistantDetail
          : assistantModeLabel);
  const openAssistant = onOpenAssistantSelect || onOpenCharacterSettings;
  const openSceneDirector = onOpenSceneDirector || onOpenCharacterSettings;
  const toolChoices: Array<{ value: RuntimeToolChoice; label: string }> = [
    { value: "auto", label: t("cockpit.toolChoiceAuto", "Auto") },
    { value: "required", label: t("cockpit.toolChoiceRequired", "Required") },
    { value: "none", label: t("cockpit.toolChoiceNone", "None") },
  ];
  const mcpToolsLabel = t("cockpit.mcpTools", "MCP tools");
  const toolCardLabel =
    toolSummary?.label === mcpToolsLabel
      ? t("cockpit.chatToolAccess", "Chat tool access")
      : toolSummary?.label;
  const canChooseMcpTools = toolSummary?.state === "available";
  const stopControlEnabled =
    streaming && canStopStreaming && Boolean(onStopStreaming);
  const regenerateControlEnabled =
    !streaming && canRegenerate && Boolean(onRegenerate);
  const stopDisabledReason = stopControlEnabled
    ? null
    : t("cockpit.stopUnavailableIdle", "No turn is running.");
  const regenerateDisabledReason = regenerateControlEnabled
    ? null
    : streaming
      ? t(
          "cockpit.regenerateUnavailableStreaming",
          "Wait for the current turn to finish before regenerating.",
        )
      : t(
          "cockpit.regenerateUnavailableNoAssistant",
          "Regenerate becomes available after an assistant response.",
        );
  const stopReasonId = `${runControlsId}-stop-disabled-reason`;
  const regenerateReasonId = `${runControlsId}-regenerate-disabled-reason`;

  return (
    <div
      data-testid="playground-runtime-inspector"
      data-message-count={effectiveMessageCount}
      className={cockpitRailStyles.stack}
    >
      <PlaygroundRailSection
        label={t("cockpit.runtimeState", "Runtime state")}
        title={t("cockpit.runtime", "Runtime")}
      >
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={cockpitRailStyles.value}>{statusLabel}</p>
            {runtimeStatusDetail ? (
              <p className={cockpitRailStyles.muted}>{runtimeStatusDetail}</p>
            ) : null}
          </div>
          <span className="shrink-0 rounded-full border border-border bg-surface2 px-2 py-0.5 text-[10px] font-semibold text-text-muted">
            {streaming
              ? t("cockpit.turnRunning", "Turn running")
              : t("cockpit.turnIdle", "Turn idle")}
          </span>
        </div>
        <dl className="mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-1 text-xs text-text-muted">
          <dt>{t("cockpit.provider", "Provider")}</dt>
          <dd className="truncate text-text">
            {displayProvider ||
              t("cockpit.noProviderSelected", "No provider selected")}
          </dd>
          <dt>{t("cockpit.model", "Model")}</dt>
          <dd className="truncate text-text">
            {displayModel || t("cockpit.noModelSelected", "No model selected")}
          </dd>
        </dl>
        {routeLabel ? (
          <div className={`mt-2 ${cockpitRailStyles.compactInset}`}>
            <p className="text-[10px] font-semibold uppercase text-text-muted">
              {t("cockpit.providerRoute", "Provider route")}
            </p>
            <p className="mt-0.5 truncate text-xs font-medium text-text">
              {routeLabel}
            </p>
            <p className="sr-only">
              {t("cockpit.route", `Route ${routeLabel}`, {
                route: routeLabel,
              })}
            </p>
            {modelUsabilityDetail ? (
              <p className={cockpitRailStyles.muted}>{modelUsabilityDetail}</p>
            ) : null}
          </div>
        ) : null}
      </PlaygroundRailSection>

      <PlaygroundRailSection
        label={t("cockpit.modelRoute", "Model route")}
        title={t("cockpit.modelRoute", "Model route")}
      >
        <div className="mt-2 flex flex-col gap-2">
          <button
            type="button"
            onClick={onOpenModelSettings}
            data-cockpit-model-settings-trigger
            className={`${cockpitRailStyles.action} justify-start gap-1.5`}
            aria-label={t("cockpit.openModelSettings", "Open model settings")}
          >
            <Settings2 className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.modelSettings", "Model settings")}
          </button>
        </div>
        <div className="mt-3">
          <p className="text-[10px] font-semibold uppercase text-text-muted">
            {t("cockpit.providerModelSettings", "Provider:model settings")}
          </p>
          {settingSummaries.length > 0 ? (
            <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1 text-xs">
              {settingSummaries.map((setting) => (
                <React.Fragment key={setting.label}>
                  <dt className="truncate text-text-muted">{setting.label}</dt>
                  <dd className="flex items-center justify-end gap-1.5 font-medium text-text">
                    <span>{setting.value}</span>
                    {setting.source ? (
                      <span
                        className={
                          setting.source === "override"
                            ? "rounded border border-focus/40 bg-focus/10 px-1.5 py-0.5 text-[10px] font-semibold text-focus"
                            : "rounded border border-border bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted"
                        }
                      >
                        {setting.source === "override"
                          ? t("cockpit.settingOverride", "Override")
                          : t("cockpit.settingInherited", "Inherited")}
                      </span>
                    ) : null}
                  </dd>
                </React.Fragment>
              ))}
            </dl>
          ) : (
            <p className={cockpitRailStyles.muted}>
              {t(
                "cockpit.settingsDefault",
                "Default settings for this provider:model.",
              )}
            </p>
          )}
        </div>
      </PlaygroundRailSection>

      <PlaygroundRailSection
        label={t("cockpit.assistant", "Assistant")}
        title={t("cockpit.assistant", "Assistant")}
      >
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={cockpitRailStyles.value}>{assistantLabel}</p>
            <p className={cockpitRailStyles.muted}>{assistantDetail}</p>
          </div>
          <span className="shrink-0 rounded-full border border-border bg-surface2 px-2 py-0.5 text-[10px] font-semibold text-text-muted">
            {assistantModeLabel}
          </span>
        </div>
        <div className="mt-2 flex flex-col gap-2">
          {openAssistant ? (
            <button
              type="button"
              onClick={openAssistant}
              data-cockpit-assistant-select-trigger
              className={`${cockpitRailStyles.action} justify-start gap-1.5`}
              aria-label={t(
                "cockpit.selectCharacterPersona",
                "Select character or persona",
              )}
            >
              <UserRound className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.selectAssistant", "Select assistant")}
            </button>
          ) : null}
          {effectiveAssistantSummary.mode !== "none" && onInspectAssistant ? (
            <button
              type="button"
              onClick={onInspectAssistant}
              className={`${cockpitRailStyles.action} justify-start gap-1.5`}
              aria-label={t("cockpit.manageAssistant", "Manage assistant")}
            >
              <Settings2 className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.manageAssistant", "Manage assistant")}
            </button>
          ) : null}
          {effectiveAssistantSummary.mode !== "none" && onClearAssistant ? (
            <button
              type="button"
              onClick={onClearAssistant}
              className={`${cockpitRailStyles.action} justify-start gap-1.5`}
              aria-label={t("cockpit.clearAssistant", "Clear assistant")}
            >
              <UserRound className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.clearAssistant", "Clear assistant")}
            </button>
          ) : null}
          {effectiveAssistantSummary.mode === "character" && openSceneDirector ? (
            <button
              type="button"
              onClick={openSceneDirector}
              className={`${cockpitRailStyles.action} justify-start gap-1.5`}
              aria-label={t(
                "cockpit.openSceneDirector",
                "Open Scene Director",
              )}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.sceneDirector", "Scene Director")}
            </button>
          ) : null}
        </div>
        {effectiveAssistantSummary.mode === "persona" ? (
          <p className={cockpitRailStyles.muted}>
            {t(
              "cockpit.personaActorUnavailable",
              "Scene Director is available for character-backed chats.",
            )}
          </p>
        ) : null}
      </PlaygroundRailSection>

      <PlaygroundRailSection
        label={mcpToolsLabel}
        title={mcpToolsLabel}
      >
        {toolSummary ? (
          <div className={`mt-2 ${cockpitRailStyles.inset}`}>
            <div className="flex items-start gap-2">
              <span
                className={`rounded border p-1 ${toolStateClass(
                  toolSummary.state,
                )}`}
              >
                <Wrench className="h-3.5 w-3.5" aria-hidden="true" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text">
                  {toolCardLabel}
                </p>
                {toolSummary.detail ? (
                  <p className="mt-0.5 text-xs text-text-muted">
                    {toolSummary.detail}
                  </p>
                ) : null}
              </div>
            </div>
            {toolSummary.stateCounts?.length ? (
              <dl
                aria-label={t(
                  "cockpit.mcpToolStateCounts",
                  "MCP tool state counts",
                )}
                className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1 text-[11px] text-text-muted"
              >
                {toolSummary.stateCounts.map((count) => (
                  <React.Fragment key={count.label}>
                    <dt className="truncate">{count.label}</dt>
                    <dd className="font-medium text-text">{count.value}</dd>
                  </React.Fragment>
                ))}
              </dl>
            ) : null}
            {toolSummary.onOpen ? (
              <button
                type="button"
                className={`${cockpitRailStyles.action} mt-2 gap-1.5`}
                onClick={toolSummary.onOpen}
                aria-label={t("cockpit.openMcpTools", "Open MCP tools")}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                {t("cockpit.openMcpTools", "Open MCP tools")}
              </button>
            ) : null}
            {canChooseMcpTools && toolChoice && onToolChoiceChange ? (
              <div
                className="mt-3 grid grid-cols-3 gap-1"
                aria-label={t("cockpit.mcpToolChoice", "MCP tool choice")}
              >
                {toolChoices.map((choice) => (
                  <button
                    key={choice.value}
                    type="button"
                    className={cockpitRailStyles.action}
                    aria-label={t(
                      `cockpit.mcpToolChoice.${choice.value}`,
                      `MCP tool choice ${choice.label}`,
                    )}
                    aria-pressed={toolChoice === choice.value}
                    onClick={() => onToolChoiceChange(choice.value)}
                  >
                    {choice.label}
                  </button>
                ))}
              </div>
            ) : null}
            {onOpenMcpSettings ? (
              <button
                type="button"
                className={`${cockpitRailStyles.action} mt-2 gap-1.5`}
                onClick={onOpenMcpSettings}
                data-cockpit-mcp-settings-trigger
                aria-label={t("cockpit.configureMcpTools", "Configure MCP tools")}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                {t("cockpit.configureMcpTools", "Configure MCP tools")}
              </button>
            ) : null}
          </div>
        ) : (
          <p className={cockpitRailStyles.muted}>
            {t(
              "cockpit.toolsComposerManaged",
              "Turn tools are managed from the composer.",
            )}
          </p>
        )}
      </PlaygroundRailSection>

      <PlaygroundRailSection
        label={t("cockpit.runControls", "Run controls")}
        title={t("cockpit.runControls", "Run controls")}
      >
        {emptyAssistantResponse && !streaming ? (
          <div
            role="status"
            aria-live="polite"
            aria-label={t(
              "cockpit.emptyAssistantResponse",
              "Empty assistant response",
            )}
            className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-2.5 py-2 text-xs text-warn"
          >
            <p className="font-medium">
              {t(
                "cockpit.emptyAssistantResponseSummary",
                "No response text returned.",
              )}
            </p>
            <p className="mt-1 opacity-90">
              {t(
                "cockpit.emptyAssistantResponseDetail",
                "Regenerate this turn or switch model settings before trying again.",
              )}
            </p>
          </div>
        ) : null}
        <div className="mt-2 grid gap-2">
          <div>
            <button
              type="button"
              disabled={!stopControlEnabled}
              onClick={stopControlEnabled ? onStopStreaming : undefined}
              className={`${cockpitRailDisabledActionClass} w-full justify-start gap-1.5`}
              aria-label={t("cockpit.stopGeneration", "Stop generation")}
              aria-describedby={!stopControlEnabled ? stopReasonId : undefined}
            >
              <Square className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.stopGeneration", "Stop generation")}
            </button>
            {stopDisabledReason ? (
              <p id={stopReasonId} className={cockpitRailStyles.muted}>
                {stopDisabledReason}
              </p>
            ) : null}
          </div>
          <div>
            <button
              type="button"
              disabled={!regenerateControlEnabled}
              onClick={regenerateControlEnabled ? onRegenerate : undefined}
              className={`${cockpitRailDisabledActionClass} w-full justify-start gap-1.5`}
              aria-label={t(
                "cockpit.regenerateLastResponse",
                "Regenerate last response",
              )}
              aria-describedby={
                !regenerateControlEnabled ? regenerateReasonId : undefined
              }
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.regenerateLastResponse", "Regenerate last response")}
            </button>
            {regenerateDisabledReason ? (
              <p id={regenerateReasonId} className={cockpitRailStyles.muted}>
                {regenerateDisabledReason}
              </p>
            ) : null}
          </div>
        </div>
        <div className="mt-3 border-t border-border/70 pt-2">
          <p className="text-[10px] font-semibold uppercase text-text-muted">
            {t("cockpit.timeline", "Timeline")}
          </p>
          <p className={cockpitRailStyles.value}>{messageLabel}</p>
          <p className={cockpitRailStyles.muted}>
            {threadSearchOpen
              ? t("cockpit.searchOpen", "Search open")
              : t("cockpit.searchClosed", "Search closed")}
          </p>
        </div>
      </PlaygroundRailSection>
    </div>
  );
};
