import { useTranslation } from "react-i18next";
import { getDesignSystemState } from "@/design-system";
import type { PlaygroundCockpitMode } from "./PlaygroundCockpitShell";

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
};

const pillClass =
  "inline-flex min-h-[24px] max-w-full items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text";

const interpolateCountFallback = (value: string, count: number) =>
  value.replace(/\{\{\s*count\s*\}\}/g, String(count));

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
}: PlaygroundStatusStripProps) => {
  const { t } = useTranslation("playground");
  const isDegraded = degraded || degradedChecks.length > 0;
  const runtimeLabel = errorMessage
    ? t("cockpit.error", ERROR_STATE_LABEL)
    : isDegraded
      ? t("cockpit.degraded", DEGRADED_STATE_LABEL)
      : streaming
        ? t("cockpit.streaming", "Streaming")
        : t("cockpit.ready", READY_STATE_LABEL);
  const messageLabel = interpolateCountFallback(
    t("cockpit.messageCount", "{{count}} messages", {
      count: messageCount,
      defaultValue_one: "{{count}} message",
    }),
    messageCount,
  );

  return (
    <footer
      role="status"
      aria-label={t("cockpit.chatStatus", "Chat status")}
      aria-live="polite"
      aria-atomic="false"
      className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-t border-border bg-surface px-3 py-2 text-xs text-text-muted"
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className={pillClass}>{runtimeLabel}</span>
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
        {contextSummary.map((item) => (
          <span className={pillClass} key={item}>
            {item}
          </span>
        ))}
        {errorMessage ? (
          <span className={pillClass}>{errorMessage}</span>
        ) : (
          degradedChecks.map((check) => (
            <span className={pillClass} key={check}>
              {check}
            </span>
          ))
        )}
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        {selectedProvider ? (
          <span className="max-w-[10rem] truncate">{selectedProvider}</span>
        ) : null}
        <span className="max-w-[18rem] truncate">
          {selectedModel || t("cockpit.noModelSelected", "No model selected")}
        </span>
        <span>{messageLabel}</span>
      </div>
    </footer>
  );
};
