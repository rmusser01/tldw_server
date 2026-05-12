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
};

const openSearchAndContext = () => {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent("tldw:open-knowledge-panel", {
      detail: { tab: "search" },
    }),
  );
};

export const PlaygroundContextRail = ({
  hasContext,
  contextSummary,
  sessionLabel,
  historyLinked,
}: PlaygroundContextRailProps) => {
  const { t } = useTranslation("playground");

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
        {contextSummary.length > 0 ? (
          <ul className="mt-2 space-y-1 text-xs text-text-muted">
            {contextSummary.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        <button
          type="button"
          onClick={openSearchAndContext}
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
      </section>
    </div>
  );
};
