import { useTranslation } from "react-i18next";
import { getDesignSystemState } from "@/design-system";

const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export type PlaygroundRuntimeInspectorProps = {
  streaming: boolean;
  selectedProvider?: string | null | undefined;
  selectedModel: string | null | undefined;
  providerRouteLabel?: string | null;
  runtimeStatus?: "ready" | "streaming" | "error" | "degraded";
  runtimeStatusDetail?: string | null;
  messageCount: number;
  threadSearchOpen: boolean;
  selectedCharacterName: string | null | undefined;
  onOpenModelSettings: () => void;
  onOpenCharacterSettings: () => void;
  canStopStreaming?: boolean;
  onStopStreaming?: () => void;
  canRegenerate?: boolean;
  onRegenerate?: () => void;
};

const interpolateCountFallback = (value: string, count: number) =>
  value.replace(/\{\{\s*count\s*\}\}/g, String(count));

const DEGRADED_STATE_LABEL = getDesignSystemState("degraded").label;
const ERROR_STATE_LABEL = getDesignSystemState("error").label;
const READY_STATE_LABEL = getDesignSystemState("ready").label;

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
  onOpenModelSettings,
  onOpenCharacterSettings,
  canStopStreaming = false,
  onStopStreaming,
  canRegenerate = false,
  onRegenerate,
}: PlaygroundRuntimeInspectorProps) => {
  const { t } = useTranslation("playground");
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
  const messageLabel = interpolateCountFallback(
    t("cockpit.messageCount", "{{count}} messages", {
      count: messageCount,
      defaultValue_one: "{{count}} message",
    }),
    messageCount,
  );

  return (
    <div
      data-testid="playground-runtime-inspector"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section
        className={railSectionClass}
        aria-label={t("cockpit.runtimeState", "Runtime state")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.runtime", "Runtime")}</h2>
        <p className={railValueClass}>{statusLabel}</p>
        {runtimeStatusDetail ? (
          <p className={railMutedClass}>{runtimeStatusDetail}</p>
        ) : null}
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
          <p className={railMutedClass}>
            {t("cockpit.route", `Route ${routeLabel}`, {
              route: routeLabel,
            })}
          </p>
        ) : null}
        {canStopStreaming && onStopStreaming ? (
          <button
            type="button"
            onClick={onStopStreaming}
            className={`${railActionClass} mt-3`}
            aria-label={t("cockpit.stopGeneration", "Stop generation")}
          >
            {t("cockpit.stopGeneration", "Stop generation")}
          </button>
        ) : null}
        {!streaming && canRegenerate && onRegenerate ? (
          <button
            type="button"
            onClick={onRegenerate}
            className={`${railActionClass} mt-3`}
            aria-label={t(
              "cockpit.regenerateLastResponse",
              "Regenerate last response",
            )}
          >
            {t("cockpit.regenerateLastResponse", "Regenerate last response")}
          </button>
        ) : null}
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.modelAndCharacter", "Model and character")}
      >
        <h2 className={railHeadingClass}>
          {t("cockpit.modelCharacter", "Model & character")}
        </h2>
        <div className="mt-2 flex flex-col gap-2">
          <button
            type="button"
            onClick={onOpenModelSettings}
            className={railActionClass}
            aria-label={t("cockpit.openModelSettings", "Open model settings")}
          >
            {t("cockpit.modelSettings", "Model settings")}
          </button>
          <button
            type="button"
            onClick={onOpenCharacterSettings}
            className={railActionClass}
            aria-label={t(
              "cockpit.openCharacterSettings",
              "Open character settings",
            )}
          >
            {t("cockpit.character", "Character")}
          </button>
        </div>
        <p className={railMutedClass}>
          {selectedCharacterName ||
            t("cockpit.noCharacterSelected", "No character selected")}
        </p>
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
