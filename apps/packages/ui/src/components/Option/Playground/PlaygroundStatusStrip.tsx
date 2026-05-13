import { useTranslation } from "react-i18next";
import { getDesignSystemState } from "@/design-system";
import type { PlaygroundCockpitMode } from "./PlaygroundCockpitShell";
import {
  formatCockpitMessageCount,
  useCockpitMessageCount,
} from "./playground-cockpit-state";
import { AlertTriangle, CircleCheck, Loader2, Search, Settings2, Square } from "lucide-react";

export type PlaygroundStatusStripProps = {
  mode: PlaygroundCockpitMode;
  streaming: boolean;
  selectedProvider?: string | null;
  selectedModel: string | null | undefined;
  messageCount: number;
  sessionLabel: string;
  hasContext: boolean;
  contextSummary?: string[];
  temporaryChat?: boolean;
  degraded?: boolean;
  degradedChecks?: string[];
  errorMessage?: string | null;
  onStopStreaming?: () => void;
  onOpenSearchContext?: () => void;
  onOpenModelSettings?: () => void;
};

const pillClass =
  "inline-flex min-h-[24px] max-w-full items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text";
const actionClass =
  "inline-flex min-h-[26px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

const DEGRADED_STATE_LABEL = getDesignSystemState("degraded").label;
const ERROR_STATE_LABEL = getDesignSystemState("error").label;
const READY_STATE_LABEL = getDesignSystemState("ready").label;

export const PlaygroundStatusStrip = ({
  mode,
  streaming,
  selectedProvider,
  selectedModel,
  messageCount,
  sessionLabel,
  hasContext,
  contextSummary = [],
  temporaryChat,
  degraded = false,
  degradedChecks = [],
  errorMessage,
  onStopStreaming,
  onOpenSearchContext,
  onOpenModelSettings,
}: PlaygroundStatusStripProps) => {
  const { t } = useTranslation("playground");
  const effectiveMessageCount = useCockpitMessageCount(messageCount);
  const isDegraded = degraded || degradedChecks.length > 0;
  const routeLabel =
    selectedProvider && selectedModel
      ? `${selectedProvider}:${selectedModel}`
      : selectedModel || t("cockpit.noModelSelected", "No model selected");
  const runtimeLabel = errorMessage
    ? t("cockpit.error", ERROR_STATE_LABEL)
    : isDegraded
      ? t("cockpit.degraded", DEGRADED_STATE_LABEL)
      : streaming
        ? t("cockpit.streaming", "Streaming")
        : t("cockpit.ready", READY_STATE_LABEL);
  const messageLabel = formatCockpitMessageCount(
    t("cockpit.messageCount", {
      count: effectiveMessageCount,
      defaultValue: "{{count}} messages",
      defaultValue_one: "{{count}} message",
    }),
    effectiveMessageCount,
  );

  return (
    <footer
      role="status"
      data-message-count={effectiveMessageCount}
      aria-label={t("cockpit.chatStatus", "Chat status")}
      aria-live="polite"
      aria-atomic="false"
      className="flex min-w-0 flex-wrap items-center justify-between gap-3 border-t border-border bg-surface px-3 py-2 text-xs text-text-muted"
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span
          className={`${pillClass} gap-1.5 ${
            errorMessage
              ? "border-error/40 bg-error/10 text-error"
              : isDegraded
                ? "border-warning/40 bg-warning/10 text-warning"
                : streaming
                  ? "border-info/40 bg-info/10 text-info"
                  : "border-success/40 bg-success/10 text-success"
          }`}
        >
          {errorMessage ? (
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          ) : streaming ? (
            <Loader2 className="h-3.5 w-3.5" aria-hidden="true" />
          ) : isDegraded ? (
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <CircleCheck className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          {runtimeLabel}
        </span>
        <span className={pillClass}>
          {mode === "focus"
            ? t("cockpit.focus", "Focus")
            : t("cockpit.cockpit", "Cockpit")}
        </span>
        <span className={pillClass}>{sessionLabel}</span>
        {typeof temporaryChat === "boolean" ? (
          <span className={pillClass}>
            {temporaryChat
              ? t("cockpit.temporary", "Temporary")
              : t("cockpit.saved", "Saved")}
          </span>
        ) : null}
        {hasContext ? (
          <span className={pillClass}>
            {t("cockpit.contextActive", "Context active")}
          </span>
        ) : null}
        {contextSummary.map((item, index) => (
          <span className={pillClass} key={`context-${index}-${item}`}>
            {item}
          </span>
        ))}
        {errorMessage ? (
          <span className={pillClass}>{errorMessage}</span>
        ) : (
          degradedChecks.map((check, index) => (
            <span className={pillClass} key={`degraded-${index}-${check}`}>
              {check}
            </span>
          ))
        )}
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className="max-w-[18rem] truncate font-medium text-text">
          {routeLabel}
        </span>
        <span>{messageLabel}</span>
        {streaming && onStopStreaming ? (
          <button
            type="button"
            className={actionClass}
            onClick={onStopStreaming}
            aria-label={t("cockpit.stopGeneration", "Stop generation")}
          >
            <Square className="h-3 w-3" aria-hidden="true" />
            {t("cockpit.stop", "Stop")}
          </button>
        ) : null}
        {hasContext && onOpenSearchContext ? (
          <button
            type="button"
            className={actionClass}
            onClick={onOpenSearchContext}
            aria-label={t("cockpit.openSearchContext", "Open Search & Context")}
          >
            <Search className="h-3 w-3" aria-hidden="true" />
            {t("cockpit.context", "Context")}
          </button>
        ) : null}
        {!selectedModel && onOpenModelSettings ? (
          <button
            type="button"
            className={actionClass}
            onClick={onOpenModelSettings}
            aria-label={t("cockpit.openModelSettings", "Open model settings")}
          >
            <Settings2 className="h-3 w-3" aria-hidden="true" />
            {t("cockpit.model", "Model")}
          </button>
        ) : null}
      </div>
    </footer>
  );
};
