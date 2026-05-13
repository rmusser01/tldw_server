import React from "react";
import { useTranslation } from "react-i18next";
import { getDesignSystemState } from "@/design-system";
import { Settings2, SlidersHorizontal, UserRound, Wrench } from "lucide-react";
import {
  formatCockpitMessageCount,
  useCockpitMessageCount,
} from "./playground-cockpit-state";

const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export type RuntimeSettingSummary = {
  label: string;
  value: string;
};

export type RuntimeToolSummary = {
  state: "available" | "unavailable" | "disabled" | "degraded";
  label: string;
  detail?: string | null;
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
  runtimeStatus?: "ready" | "streaming" | "error" | "degraded";
  runtimeStatusDetail?: string | null;
  messageCount: number;
  threadSearchOpen: boolean;
  selectedCharacterName?: string | null | undefined;
  assistantSummary?: RuntimeAssistantSummary;
  onOpenModelSettings: () => void;
  onOpenCharacterSettings?: () => void;
  onOpenAssistantSelect?: () => void;
  onOpenSceneDirector?: () => void;
  toolChoice?: RuntimeToolChoice;
  onToolChoiceChange?: (choice: RuntimeToolChoice) => void;
  onOpenMcpSettings?: () => void;
  canStopStreaming?: boolean;
  onStopStreaming?: () => void;
  canRegenerate?: boolean;
  onRegenerate?: () => void;
  settingSummaries?: RuntimeSettingSummary[];
  toolSummary?: RuntimeToolSummary | null;
};

const DEGRADED_STATE_LABEL = getDesignSystemState("degraded").label;
const ERROR_STATE_LABEL = getDesignSystemState("error").label;
const READY_STATE_LABEL = getDesignSystemState("ready").label;

const toolStateClass = (state: RuntimeToolSummary["state"]) => {
  if (state === "available") return "border-success/40 bg-success/10 text-success";
  if (state === "degraded") return "border-warning/40 bg-warning/10 text-warning";
  return "border-border bg-surface2 text-text-muted";
};

export const PlaygroundRuntimeInspector = ({
  streaming,
  selectedProvider,
  selectedModel,
  providerRouteLabel,
  runtimeStatus,
  runtimeStatusDetail,
  messageCount,
  threadSearchOpen,
  selectedCharacterName,
  assistantSummary,
  onOpenModelSettings,
  onOpenCharacterSettings,
  onOpenAssistantSelect,
  onOpenSceneDirector,
  toolChoice,
  onToolChoiceChange,
  onOpenMcpSettings,
  canStopStreaming = false,
  onStopStreaming,
  canRegenerate = false,
  onRegenerate,
  settingSummaries = [],
  toolSummary = null,
}: PlaygroundRuntimeInspectorProps) => {
  const { t } = useTranslation("playground");
  const effectiveMessageCount = useCockpitMessageCount(messageCount);
  const selectedModelParts =
    !selectedProvider && selectedModel && selectedModel.includes(":")
      ? selectedModel.split(":")
      : null;
  const displayProvider =
    selectedProvider || selectedModelParts?.[0] || null;
  const displayModel =
    selectedModelParts && selectedModelParts.length > 1
      ? selectedModelParts.slice(1).join(":")
      : selectedProvider && selectedModel?.startsWith(`${selectedProvider}:`)
        ? selectedModel.slice(selectedProvider.length + 1)
        : selectedModel || null;
  const routeLabel =
    providerRouteLabel ||
    (selectedModel?.includes(":")
      ? selectedModel
      : displayProvider && displayModel
        ? `${displayProvider}:${displayModel}`
        : selectedModel || null);
  const status = runtimeStatus || (streaming ? "streaming" : "ready");
  const statusLabel =
    status === "streaming"
      ? t("cockpit.streaming", "Streaming")
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
          detail: t("cockpit.noAssistantSelected", "No assistant selected"),
        });
  const assistantLabel =
    effectiveAssistantSummary.name ||
    t("cockpit.noAssistantSelected", "No assistant selected");
  const assistantModeLabel =
    effectiveAssistantSummary.mode === "persona"
      ? t("cockpit.personaMode", "Persona")
      : effectiveAssistantSummary.mode === "character"
        ? t("cockpit.characterMode", "Character")
        : t("cockpit.noAssistantMode", "None");
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

  return (
    <div
      data-testid="playground-runtime-inspector"
      data-message-count={effectiveMessageCount}
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section
        className={railSectionClass}
        aria-label={t("cockpit.runtimeState", "Runtime state")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.runtime", "Runtime")}</h2>
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={railValueClass}>{statusLabel}</p>
            {runtimeStatusDetail ? (
              <p className={railMutedClass}>{runtimeStatusDetail}</p>
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
          <div className="mt-2 rounded-md border border-border bg-bg px-2 py-1.5">
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
          </div>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-2">
          {canStopStreaming && onStopStreaming ? (
            <button
              type="button"
              onClick={onStopStreaming}
              className={railActionClass}
              aria-label={t("cockpit.stopGeneration", "Stop generation")}
            >
              {t("cockpit.stopGeneration", "Stop generation")}
            </button>
          ) : null}
          {!streaming && canRegenerate && onRegenerate ? (
            <button
              type="button"
              onClick={onRegenerate}
              className={railActionClass}
              aria-label={t(
                "cockpit.regenerateLastResponse",
                "Regenerate last response",
              )}
            >
              {t("cockpit.regenerateLastResponse", "Regenerate last response")}
            </button>
          ) : null}
        </div>
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.modelAndChatSettings", "Model & Chat settings")}
      >
        <h2 className={railHeadingClass}>
          {t("cockpit.modelChat", "Model & Chat")}
        </h2>
        <div className="mt-2 flex flex-col gap-2">
          <button
            type="button"
            onClick={onOpenModelSettings}
            className={`${railActionClass} justify-start gap-1.5`}
            aria-label={t(
              "cockpit.openModelChatSettings",
              "Open Model & Chat settings",
            )}
          >
            <Settings2 className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.modelChatSettings", "Model & Chat settings")}
          </button>
        </div>
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.characterPersona", "Character / Persona")}
      >
        <h2 className={railHeadingClass}>
          {t("cockpit.characterPersona", "Character / Persona")}
        </h2>
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={railValueClass}>{assistantLabel}</p>
            <p className={railMutedClass}>
              {effectiveAssistantSummary.detail || assistantModeLabel}
            </p>
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
              className={`${railActionClass} justify-start gap-1.5`}
              aria-label={t(
                "cockpit.selectCharacterPersona",
                "Select character or persona",
              )}
            >
              <UserRound className="h-3.5 w-3.5" aria-hidden="true" />
              {t("cockpit.selectAssistant", "Select assistant")}
            </button>
          ) : null}
          {effectiveAssistantSummary.mode === "character" && openSceneDirector ? (
            <button
              type="button"
              onClick={openSceneDirector}
              className={`${railActionClass} justify-start gap-1.5`}
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
          <p className={railMutedClass}>
            {t(
              "cockpit.personaActorUnavailable",
              "Scene Director is available for character-backed chats.",
            )}
          </p>
        ) : null}
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.scopedSettings", "Scoped settings")}
      >
        <h2 className={railHeadingClass}>
          {t("cockpit.scopedSettings", "Scoped settings")}
        </h2>
        {settingSummaries.length > 0 ? (
          <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1 text-xs">
            {settingSummaries.map((setting) => (
              <React.Fragment key={setting.label}>
                <dt className="truncate text-text-muted">{setting.label}</dt>
                <dd className="font-medium text-text">{setting.value}</dd>
              </React.Fragment>
            ))}
          </dl>
        ) : (
          <p className={railMutedClass}>
            {t(
              "cockpit.settingsDefault",
              "Using default settings for this provider:model.",
            )}
          </p>
        )}
      </section>

      <section
        className={railSectionClass}
        aria-label={mcpToolsLabel}
      >
        <h2 className={railHeadingClass}>{mcpToolsLabel}</h2>
        {toolSummary ? (
          <div className="mt-2 rounded-md border border-border bg-bg px-2.5 py-2">
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
            {toolSummary.onOpen ? (
              <button
                type="button"
                className={`${railActionClass} mt-2 gap-1.5`}
                onClick={toolSummary.onOpen}
                aria-label={t("cockpit.openMcpTools", "Open MCP tools")}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                {t("cockpit.openTools", "Open tools")}
              </button>
            ) : null}
            {toolChoice && onToolChoiceChange ? (
              <div
                className="mt-3 grid grid-cols-3 gap-1"
                aria-label={t("cockpit.mcpToolChoice", "MCP tool choice")}
              >
                {toolChoices.map((choice) => (
                  <button
                    key={choice.value}
                    type="button"
                    className={railActionClass}
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
                className={`${railActionClass} mt-2 gap-1.5`}
                onClick={onOpenMcpSettings}
                aria-label={t("cockpit.configureMcpTools", "Configure MCP tools")}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
                {t("cockpit.configureMcp", "Configure MCP")}
              </button>
            ) : null}
          </div>
        ) : (
          <p className={railMutedClass}>
            {t(
              "cockpit.toolsComposerManaged",
              "Turn tools are managed from the composer.",
            )}
          </p>
        )}
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.conversationVolume", "Conversation volume")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.timeline", "Timeline")}</h2>
        <p className={railValueClass}>{messageLabel}</p>
        <p className={railMutedClass}>
          {threadSearchOpen
            ? t("cockpit.searchOpen", "Search open")
            : t("cockpit.searchClosed", "Search closed")}
        </p>
      </section>
    </div>
  );
};
