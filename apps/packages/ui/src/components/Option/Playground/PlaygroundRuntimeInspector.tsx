import { useTranslation } from "react-i18next";

const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export type PlaygroundRuntimeInspectorProps = {
  streaming: boolean;
  selectedModel: string | null | undefined;
  messageCount: number;
  threadSearchOpen: boolean;
  selectedCharacterName: string | null | undefined;
};

const dispatchEvent = (eventName: string) => {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(eventName));
};

const interpolateCountFallback = (value: string, count: number) =>
  value.replace(/\{\{\s*count\s*\}\}/g, String(count));

export const PlaygroundRuntimeInspector = ({
  streaming,
  selectedModel,
  messageCount,
  threadSearchOpen,
  selectedCharacterName,
}: PlaygroundRuntimeInspectorProps) => {
  const { t } = useTranslation("playground");
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
        <p className={railValueClass}>
          {streaming
            ? t("cockpit.streaming", "Streaming")
            : t("cockpit.ready", "Ready")}
        </p>
        <p className={railMutedClass}>
          {selectedModel || t("cockpit.noModelSelected", "No model selected")}
        </p>
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
            onClick={() => dispatchEvent("tldw:open-model-settings")}
            className={railActionClass}
            aria-label={t("cockpit.openModelSettings", "Open model settings")}
          >
            {t("cockpit.modelSettings", "Model settings")}
          </button>
          <button
            type="button"
            onClick={() => dispatchEvent("tldw:open-actor-settings")}
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
