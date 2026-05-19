import { useTranslation } from "react-i18next";
import {
  DEGRADED_STATE_LABEL,
  ERROR_STATE_LABEL,
  LOADING_STATE_LABEL,
  READY_STATE_LABEL,
} from "@/design-system";
import type { PlaygroundCockpitMode } from "./PlaygroundCockpitShell";
import {
  formatCockpitMessageCount,
  useCockpitMessageCount,
} from "./playground-cockpit-state";
import {
  AlertTriangle,
  CircleCheck,
  Loader2,
  Search,
  Settings2,
  Square,
} from "lucide-react";

type PlaygroundStatusStripCompositionStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error";

type PlaygroundStatusStripRuntimeState =
  | "error"
  | "server-blocked"
  | "streaming"
  | "loading"
  | "missing-model"
  | "degraded"
  | "ready";

type PlaygroundStatusStripSessionStatus =
  | "idle"
  | "loading"
  | "loaded"
  | "failed";

export type PlaygroundStatusStripProps = {
  mode: PlaygroundCockpitMode;
  streaming: boolean;
  selectedProvider?: string | null;
  selectedModel: string | null | undefined;
  messageCount: number;
  sessionLabel: string;
  sessionTitle?: string | null;
  sessionStatus?: PlaygroundStatusStripSessionStatus | null;
  sessionStatusLabel?: string | null;
  sessionDetail?: string | null;
  sessionError?: string | null;
  hasContext: boolean;
  contextSummary?: string[];
  temporaryChat?: boolean;
  characterChatActive?: boolean;
  degraded?: boolean;
  degradedChecks?: string[];
  errorMessage?: string | null;
  serverBlocked?: boolean;
  compositionStatus?: PlaygroundStatusStripCompositionStatus;
  onStopStreaming?: () => void;
  onOpenSearchContext?: () => void;
  onOpenModelSettings?: () => void;
};

const pillClass =
  "inline-flex min-h-[24px] max-w-full items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text";
const actionClass =
  "inline-flex min-h-[26px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export const PlaygroundStatusStrip = ({
  streaming,
  selectedProvider,
  selectedModel,
  messageCount,
  sessionLabel,
  sessionTitle,
  sessionStatus,
  sessionStatusLabel,
  sessionDetail,
  sessionError,
  hasContext,
  temporaryChat,
  characterChatActive = false,
  degraded = false,
  degradedChecks = [],
  errorMessage,
  serverBlocked = false,
  compositionStatus = "idle",
  onStopStreaming,
  onOpenSearchContext,
  onOpenModelSettings,
}: PlaygroundStatusStripProps) => {
  const { t } = useTranslation("playground");
  const effectiveMessageCount = useCockpitMessageCount(messageCount);
  const isDegraded = degraded || degradedChecks.length > 0;
  const hasSelectedModel = Boolean(selectedModel?.trim());
  const isContextLoading = compositionStatus === "loading";
  const runtimeState: PlaygroundStatusStripRuntimeState = errorMessage
    ? "error"
    : serverBlocked
      ? "server-blocked"
      : streaming
        ? "streaming"
        : isContextLoading
          ? "loading"
          : !hasSelectedModel
            ? "missing-model"
            : isDegraded
              ? "degraded"
              : "ready";
  const routeLabel =
    selectedProvider && hasSelectedModel
      ? `${selectedProvider}:${selectedModel}`
      : hasSelectedModel
        ? selectedModel
        : t("cockpit.noModelSelected", "No model selected");
  const runtimeLabel =
    runtimeState === "error"
      ? t("cockpit.error", ERROR_STATE_LABEL)
      : runtimeState === "server-blocked"
        ? t("cockpit.serverUnavailable", "Server unavailable")
        : runtimeState === "streaming"
          ? t("cockpit.streaming", "Streaming")
          : runtimeState === "loading"
            ? t("cockpit.loadingContext", `${LOADING_STATE_LABEL} context`)
            : runtimeState === "missing-model"
              ? t("cockpit.noModelSelected", "No model selected")
              : runtimeState === "degraded"
                ? t("cockpit.degraded", DEGRADED_STATE_LABEL)
                : t("cockpit.ready", READY_STATE_LABEL);
  const messageLabel = formatCockpitMessageCount(
    t("cockpit.messageCount", {
      count: effectiveMessageCount,
      defaultValue: "{{count}} messages",
      defaultValue_one: "{{count}} message",
    }),
    effectiveMessageCount,
  );
  const degradedChatAvailableLabel =
    isDegraded && !errorMessage && !serverBlocked && hasSelectedModel
      ? t("cockpit.degradedChatAvailable", "Chat remains available.")
      : null;
  const serverBlockedReason =
    runtimeState === "server-blocked"
      ? t(
          "cockpit.serverUnavailableRecovery",
          "Reconnect to the server or review server settings before sending.",
        )
      : null;
  const missingModelReason =
    runtimeState === "missing-model"
      ? t(
          "cockpit.chooseModelBeforeSending",
          "Choose a model before sending.",
        )
      : null;
  const contextLoadingReason =
    runtimeState === "loading"
      ? t(
          "cockpit.contextPreviewLoading",
          "Context preview is loading.",
        )
      : null;
  const normalizedSessionTitle = sessionTitle?.trim() || null;
  const normalizedSessionStatusLabel = sessionStatusLabel?.trim() || null;
  const normalizedSessionDetail = sessionDetail?.trim() || null;
  const normalizedSessionError = sessionError?.trim() || null;
  const hasSessionDetail =
    Boolean(normalizedSessionDetail) &&
    normalizedSessionDetail !== normalizedSessionStatusLabel;
  const isCriticalSessionStatus =
    sessionStatus === "loading" || sessionStatus === "failed";
  const sessionDetailLabel =
    normalizedSessionError ||
    (isCriticalSessionStatus && hasSessionDetail ? normalizedSessionDetail : null);
  const showSessionStatusLabel = Boolean(
    isCriticalSessionStatus &&
      normalizedSessionStatusLabel &&
      normalizedSessionStatusLabel !== sessionLabel &&
      normalizedSessionStatusLabel !== READY_STATE_LABEL &&
      normalizedSessionStatusLabel !== t("cockpit.idle", "Idle"),
  );
  const hasCriticalSessionState = showSessionStatusLabel || Boolean(sessionDetailLabel);
  const characterPersistenceLabel = characterChatActive
    ? serverBlocked
      ? t("cockpit.characterLocalDraft", "Local character chat draft")
      : temporaryChat
        ? t("cockpit.characterTemporary", "Temporary character chat")
        : sessionStatus === "loaded"
          ? t("cockpit.characterSaved", "Saved character chat")
          : t("cockpit.characterLocalDraft", "Local character chat draft")
    : null;

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
            runtimeState === "error" || runtimeState === "server-blocked"
              ? "border-error/40 bg-error/10 text-error"
              : runtimeState === "degraded" ||
                  runtimeState === "missing-model"
                ? "border-warning/40 bg-warning/10 text-warning"
                : runtimeState === "streaming" || runtimeState === "loading"
                  ? "border-info/40 bg-info/10 text-info"
                  : "border-success/40 bg-success/10 text-success"
          }`}
        >
          {runtimeState === "error" ||
          runtimeState === "server-blocked" ||
          runtimeState === "missing-model" ? (
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          ) : runtimeState === "streaming" || runtimeState === "loading" ? (
            <Loader2 className="h-3.5 w-3.5" aria-hidden="true" />
          ) : runtimeState === "degraded" ? (
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <CircleCheck className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          {runtimeLabel}
        </span>
        {characterPersistenceLabel ? (
          <span className={pillClass}>{characterPersistenceLabel}</span>
        ) : null}
        {hasCriticalSessionState ? (
          <>
            <span className={pillClass}>{sessionLabel}</span>
            {normalizedSessionTitle ? (
              <span className={pillClass}>{normalizedSessionTitle}</span>
            ) : null}
            {showSessionStatusLabel ? (
              <span className={pillClass}>{normalizedSessionStatusLabel}</span>
            ) : null}
            {sessionDetailLabel ? (
              <span
                className={`${pillClass} ${
                  normalizedSessionError
                    ? "border-error/40 bg-error/10 text-error"
                    : ""
                }`}
              >
                {sessionDetailLabel}
              </span>
            ) : null}
          </>
        ) : null}
        {errorMessage ? (
          <span className={pillClass}>{errorMessage}</span>
        ) : (
          <>
            {serverBlockedReason ? (
              <span className={pillClass}>{serverBlockedReason}</span>
            ) : null}
            {missingModelReason ? (
              <span className={pillClass}>{missingModelReason}</span>
            ) : null}
            {contextLoadingReason ? (
              <span className={pillClass}>{contextLoadingReason}</span>
            ) : null}
            {!serverBlocked
              ? degradedChecks.map((check, index) => (
                  <span
                    className={pillClass}
                    key={`degraded-${index}-${check}`}
                  >
                    {check}
                  </span>
                ))
              : null}
            {degradedChatAvailableLabel ? (
              <span className={pillClass}>{degradedChatAvailableLabel}</span>
            ) : null}
          </>
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
        {errorMessage && onOpenModelSettings ? (
          <button
            type="button"
            className={actionClass}
            onClick={onOpenModelSettings}
            aria-label={t(
              "cockpit.reviewModelSettings",
              "Review model settings",
            )}
          >
            <Settings2 className="h-3 w-3" aria-hidden="true" />
            {t("cockpit.reviewSettings", "Review settings")}
          </button>
        ) : !hasSelectedModel && onOpenModelSettings ? (
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
