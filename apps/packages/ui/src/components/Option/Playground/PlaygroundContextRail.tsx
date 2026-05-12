import { useTranslation } from "react-i18next";

const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "mt-3 inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export type PlaygroundContextRailProps = {
  hasContext: boolean;
  contextSummary: string[];
  sessionLabel: string;
  historyLinked: boolean;
  webSearch: boolean;
  onToggleWebSearch: () => void;
  temporaryChat: boolean;
  onToggleTemporaryChat: (next: boolean) => void;
  contextCounts: {
    files: number;
    knowledge: number;
    media: number;
    research: number;
  };
  onOpenSearchContext: () => void;
};

export const PlaygroundContextRail = ({
  hasContext,
  contextSummary,
  sessionLabel,
  historyLinked,
  webSearch,
  onToggleWebSearch,
  temporaryChat,
  onToggleTemporaryChat,
  contextCounts,
  onOpenSearchContext,
}: PlaygroundContextRailProps) => {
  const { t } = useTranslation("playground");
  const countLabels = [
    contextCounts.research > 0
      ? contextCounts.research === 1
        ? t("cockpit.contextResearchCountOne", "1 research attachment")
        : t(
            "cockpit.contextResearchCountMany",
            `${contextCounts.research} research attachments`,
          )
      : null,
    contextCounts.files > 0
      ? contextCounts.files === 1
        ? t("cockpit.contextFilesCountOne", "1 file")
        : t("cockpit.contextFilesCountMany", `${contextCounts.files} files`)
      : null,
    contextCounts.knowledge > 0
      ? contextCounts.knowledge === 1
        ? t("cockpit.contextKnowledgeCountOne", "1 knowledge item")
        : t(
            "cockpit.contextKnowledgeCountMany",
            `${contextCounts.knowledge} knowledge items`,
          )
      : null,
    contextCounts.media > 0
      ? contextCounts.media === 1
        ? t("cockpit.contextMediaCountOne", "1 media scope")
        : t(
            "cockpit.contextMediaCountMany",
            `${contextCounts.media} media scopes`,
          )
      : null,
  ].filter((item): item is string => Boolean(item));

  return (
    <div
      data-testid="playground-context-rail"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section
        className={railSectionClass}
        aria-label={t("cockpit.conversationContext", "Conversation context")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.context", "Context")}</h2>
        <p className={railValueClass}>
          {hasContext
            ? t("cockpit.contextActive", "Context active")
            : t("cockpit.noExtraContext", "No extra context")}
        </p>
        {contextSummary.length > 0 || countLabels.length > 0 ? (
          <ul className="mt-2 space-y-1 text-xs text-text-muted">
            {contextSummary.map((item) => (
              <li key={item}>{item}</li>
            ))}
            {countLabels.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        <button
          type="button"
          onClick={onToggleWebSearch}
          className={railActionClass}
          aria-label={t("cockpit.webSearch", "Web search")}
          aria-pressed={webSearch}
        >
          {webSearch
            ? t("cockpit.webSearchOn", "Web search on")
            : t("cockpit.webSearchOff", "Web search off")}
        </button>
        <button
          type="button"
          onClick={onOpenSearchContext}
          className={railActionClass}
          aria-label={t("cockpit.openSearchContext", "Open Search & Context")}
        >
          {t("cockpit.searchContext", "Search & Context")}
        </button>
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.conversationSession", "Conversation session")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.session", "Session")}</h2>
        <p className={railValueClass}>{sessionLabel}</p>
        <p className={railMutedClass}>
          {historyLinked
            ? t("cockpit.historyLinked", "History linked")
            : t("cockpit.noSavedHistory", "No saved history yet")}
        </p>
        <button
          type="button"
          onClick={() => onToggleTemporaryChat(!temporaryChat)}
          className={railActionClass}
          aria-pressed={temporaryChat}
          aria-label={
            temporaryChat
              ? t("cockpit.saveConversation", "Save conversation")
              : t("cockpit.useTemporaryChat", "Use temporary chat")
          }
        >
          {temporaryChat
            ? t("cockpit.saveConversation", "Save conversation")
            : t("cockpit.useTemporaryChat", "Use temporary chat")}
        </button>
      </section>
    </div>
  );
};
