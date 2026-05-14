import React from "react";
import { useTranslation } from "react-i18next";
import {
  BookOpen,
  Database,
  FileText,
  Globe2,
  Layers3,
  Search,
  Trash2,
  UserRound,
} from "lucide-react";
import { getDesignSystemState } from "@/design-system";

const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "mt-3 inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";
const clearActionClass =
  "inline-flex shrink-0 items-center rounded border border-border bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";
const DEGRADED_STATE_LABEL = getDesignSystemState("degraded").label;
const PROMPT_SELECT_TRIGGER_SELECTOR = "[data-cockpit-prompt-select-trigger]";

type ContextCountItem = {
  label: string;
  clearLabel: string;
  onClear: (() => void) | undefined;
};

export type PlaygroundPromptContextState = "none" | "system" | "quick" | "custom";

export type PlaygroundPromptSummary = {
  state: PlaygroundPromptContextState;
  label: string;
  detail?: string | null;
};

export type PlaygroundContextSourceState =
  | "active"
  | "degraded"
  | "disabled"
  | "available";

export type PlaygroundContextSource = {
  id: string;
  kind:
    | "research"
    | "file"
    | "knowledge"
    | "media"
    | "web"
    | "prompt"
    | "assistant";
  label: string;
  title: string;
  detail?: string | null;
  state?: PlaygroundContextSourceState;
  onOpen?: () => void;
  onRemove?: () => void;
  openLabel?: string;
  removeLabel?: string;
};

export type PlaygroundContextRailProps = {
  hasContext: boolean;
  contextSummary: string[];
  contextSources?: PlaygroundContextSource[];
  sessionLabel: string;
  sessionTitle?: string | null;
  sessionStatus?: "idle" | "loading" | "loaded" | "failed";
  sessionStatusLabel?: string | null;
  sessionDetail?: string | null;
  sessionError?: string | null;
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
  promptSummary?: PlaygroundPromptSummary;
  promptSelectControl?: React.ReactNode;
  onClearPrompt?: () => void;
  onOpenSearchContext: () => void;
  onClearFiles?: () => void;
  onClearKnowledge?: () => void;
  onClearMedia?: () => void;
  onClearResearch?: () => void;
};

const sourceIcon = (kind: PlaygroundContextSource["kind"]) => {
  const className = "h-3.5 w-3.5";
  if (kind === "web") return <Globe2 className={className} aria-hidden="true" />;
  if (kind === "file") return <FileText className={className} aria-hidden="true" />;
  if (kind === "knowledge") return <BookOpen className={className} aria-hidden="true" />;
  if (kind === "media") return <Database className={className} aria-hidden="true" />;
  if (kind === "assistant") return <UserRound className={className} aria-hidden="true" />;
  return <Layers3 className={className} aria-hidden="true" />;
};

const sourceStateClass = (state: PlaygroundContextSourceState = "active") => {
  if (state === "degraded") return "border-warning/40 bg-warning/10 text-warning";
  if (state === "disabled") return "border-border bg-surface2 text-text-muted";
  if (state === "available") return "border-info/40 bg-info/10 text-info";
  return "border-success/40 bg-success/10 text-success";
};

const sessionStatusClass = (
  status: NonNullable<PlaygroundContextRailProps["sessionStatus"]> = "idle",
) => {
  if (status === "failed") return "border-danger/40 bg-danger/10 text-danger";
  if (status === "loading") return "border-info/40 bg-info/10 text-info";
  if (status === "loaded") return "border-success/40 bg-success/10 text-success";
  return "border-border bg-surface2 text-text-muted";
};

export const PlaygroundContextRail = ({
  hasContext,
  contextSummary,
  contextSources = [],
  sessionLabel,
  sessionTitle,
  sessionStatus = "idle",
  sessionStatusLabel,
  sessionDetail,
  sessionError,
  historyLinked,
  webSearch,
  onToggleWebSearch,
  temporaryChat,
  onToggleTemporaryChat,
  contextCounts,
  promptSummary,
  promptSelectControl,
  onClearPrompt,
  onOpenSearchContext,
  onClearFiles,
  onClearKnowledge,
  onClearMedia,
  onClearResearch,
}: PlaygroundContextRailProps) => {
  const { t } = useTranslation("playground");
  const railRef = React.useRef<HTMLDivElement | null>(null);
  const clearPromptFromRail = React.useCallback(() => {
    onClearPrompt?.();
    railRef.current
      ?.querySelector<HTMLElement>(PROMPT_SELECT_TRIGGER_SELECTOR)
      ?.focus();
  }, [onClearPrompt]);
  const activeSourceCount = contextSources.filter(
    (source) => source.state !== "disabled",
  ).length;
  const effectivePromptSummary =
    promptSummary || {
      state: "none" as const,
      label: t("cockpit.noPromptSelected", "No prompt selected"),
      detail: t(
        "cockpit.noPromptContext",
        "No prompt context will be added.",
      ),
    };
  const promptActive = effectivePromptSummary.state !== "none";
  const effectiveSessionStatusLabel =
    sessionStatusLabel ||
    (sessionStatus === "failed"
      ? t("cockpit.sessionLoadFailed", "Load failed")
      : sessionStatus === "loading"
        ? t("cockpit.sessionLoading", "Loading conversation")
        : sessionStatus === "loaded"
          ? t("cockpit.sessionReady", "Ready")
          : t("cockpit.sessionIdle", "Idle"));
  const sourceCountLabel =
    activeSourceCount === 1
      ? t("cockpit.activeSourceCountOne", "1 active source")
      : t("cockpit.activeSourceCountMany", "{{count}} active sources", {
          count: activeSourceCount,
        });
  const countLabels = ([
    contextCounts.research > 0
      ? contextCounts.research === 1
        ? {
            label: t(
              "cockpit.contextResearchCountOne",
              "1 research attachment",
            ),
            clearLabel: t(
              "cockpit.clearResearchContext",
              "Clear research context",
            ),
            onClear: onClearResearch,
          }
        : {
            label: t(
              "cockpit.contextResearchCountMany",
              `${contextCounts.research} research attachments`,
              { count: contextCounts.research },
            ),
            clearLabel: t(
              "cockpit.clearResearchContext",
              "Clear research context",
            ),
            onClear: onClearResearch,
          }
      : null,
    contextCounts.files > 0
      ? contextCounts.files === 1
        ? {
            label: t("cockpit.contextFilesCountOne", "1 file"),
            clearLabel: t("cockpit.clearFiles", "Clear files"),
            onClear: onClearFiles,
          }
        : {
            label: t(
              "cockpit.contextFilesCountMany",
              `${contextCounts.files} files`,
              { count: contextCounts.files },
            ),
            clearLabel: t("cockpit.clearFiles", "Clear files"),
            onClear: onClearFiles,
          }
      : null,
    contextCounts.knowledge > 0
      ? contextCounts.knowledge === 1
        ? {
            label: t("cockpit.contextKnowledgeCountOne", "1 knowledge item"),
            clearLabel: t("cockpit.clearKnowledge", "Clear knowledge"),
            onClear: onClearKnowledge,
          }
        : {
            label: t(
              "cockpit.contextKnowledgeCountMany",
              `${contextCounts.knowledge} knowledge items`,
              { count: contextCounts.knowledge },
            ),
            clearLabel: t("cockpit.clearKnowledge", "Clear knowledge"),
            onClear: onClearKnowledge,
          }
      : null,
    contextCounts.media > 0
      ? contextCounts.media === 1
        ? {
            label: t("cockpit.contextMediaCountOne", "1 media scope"),
            clearLabel: t("cockpit.clearMediaScopes", "Clear media scopes"),
            onClear: onClearMedia,
          }
        : {
            label: t(
              "cockpit.contextMediaCountMany",
              `${contextCounts.media} media scopes`,
              { count: contextCounts.media },
            ),
            clearLabel: t("cockpit.clearMediaScopes", "Clear media scopes"),
            onClear: onClearMedia,
          }
      : null,
  ] satisfies Array<ContextCountItem | null>).filter(
    (item): item is ContextCountItem => Boolean(item),
  );

  return (
    <div
      ref={railRef}
      data-testid="playground-context-rail"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section
        className={railSectionClass}
        aria-label={t("cockpit.conversationContext", "Conversation context")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.context", "Context")}</h2>
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={railValueClass}>
              {hasContext
                ? t("cockpit.contextActive", "Context active")
                : t("cockpit.noExtraContext", "No extra context")}
            </p>
            <p className={railMutedClass}>
              {hasContext
                ? sourceCountLabel
                : t(
                    "cockpit.noContextDetail",
                    "Nothing extra will be added to the next reply.",
                  )}
            </p>
          </div>
          <span
            className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${sourceStateClass(
              hasContext ? "active" : "disabled",
            )}`}
          >
            {hasContext
              ? t("cockpit.armed", "Armed")
              : t("cockpit.idle", "Idle")}
          </span>
        </div>
        {contextSummary.length > 0 || countLabels.length > 0 ? (
          <ul
            className="mt-2 flex flex-wrap gap-1.5 text-xs text-text-muted"
            aria-label={t("cockpit.contextSummary", "Context summary")}
          >
            {contextSummary.map((item, index) => (
              <li
                key={`summary-${index}-${item}`}
                className="rounded border border-border bg-surface2 px-2 py-0.5"
              >
                {item}
              </li>
            ))}
            {countLabels.map((item, index) => (
              <li
                key={`count-${index}-${item.label}`}
                className="flex items-center gap-1 rounded border border-border bg-surface2 px-2 py-0.5"
              >
                <span className="min-w-0 truncate">{item.label}</span>
                {item.onClear ? (
                  <button
                    type="button"
                    className={clearActionClass}
                    aria-label={item.clearLabel}
                    title={item.clearLabel}
                    onClick={item.onClear}
                  >
                    {t("cockpit.clear", "Clear")}
                  </button>
                ) : null}
              </li>
            ))}
          </ul>
        ) : null}
        {contextSources.length > 0 ? (
          <ul
            aria-label={t("cockpit.contextSources", "Context sources")}
            className="mt-3 space-y-2"
          >
            {contextSources.map((source) => {
              const removeLabel =
                source.removeLabel ||
                t("cockpit.removeSource", "Remove {{title}}", {
                  title: source.title,
                });
              const openLabel =
                source.openLabel ||
                t("cockpit.openSource", "Open {{title}}", {
                  title: source.title,
                });
              return (
                <li
                  key={source.id}
                  className="rounded-md border border-border bg-bg px-2.5 py-2"
                >
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 rounded border border-border bg-surface2 p-1 text-text-muted">
                      {sourceIcon(source.kind)}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-semibold uppercase text-text-muted">
                          {source.label}
                        </span>
                        <span
                          className={`rounded-full border px-1.5 py-0.5 text-[10px] ${sourceStateClass(
                            source.state,
                          )}`}
                        >
                          {source.state === "disabled"
                            ? t("cockpit.disabled", "Disabled")
                            : source.state === "degraded"
                              ? t("cockpit.degraded", DEGRADED_STATE_LABEL)
                              : source.state === "available"
                                ? t("cockpit.available", "Available")
                                : t("cockpit.active", "Active")}
                        </span>
                      </div>
                      <p className="mt-1 truncate text-sm font-medium text-text">
                        {source.title}
                      </p>
                      {source.detail ? (
                        <p className="mt-0.5 line-clamp-2 text-xs text-text-muted">
                          {source.detail}
                        </p>
                      ) : null}
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {source.onOpen ? (
                        <button
                          type="button"
                          className={clearActionClass}
                          aria-label={openLabel}
                          title={openLabel}
                          onClick={source.onOpen}
                        >
                          <Search className="h-3 w-3" aria-hidden="true" />
                        </button>
                      ) : null}
                      {source.onRemove ? (
                        <button
                          type="button"
                          className={clearActionClass}
                          aria-label={removeLabel}
                          title={removeLabel}
                          onClick={source.onRemove}
                        >
                          <Trash2 className="h-3 w-3" aria-hidden="true" />
                        </button>
                      ) : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="mt-3 rounded-md border border-dashed border-border bg-bg px-2.5 py-2 text-xs text-text-muted">
            {t(
              "cockpit.contextEmptyWorkbench",
              "Add web search, files, knowledge, media, or research context before sending.",
            )}
          </div>
        )}
        <div className="mt-3 grid grid-cols-1 gap-2">
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
        </div>
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.promptContext", "Prompt context")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.prompts", "Prompts")}</h2>
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={railValueClass}>{effectivePromptSummary.label}</p>
            {effectivePromptSummary.detail ? (
              <p className={railMutedClass}>{effectivePromptSummary.detail}</p>
            ) : null}
          </div>
          <span
            className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${sourceStateClass(
              promptActive ? "active" : "disabled",
            )}`}
          >
            {promptActive
              ? t("cockpit.active", "Active")
              : t("cockpit.idle", "Idle")}
          </span>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {promptSelectControl}
          {promptActive && onClearPrompt ? (
            <button
              type="button"
              className={railActionClass}
              aria-label={t("cockpit.clearPrompt", "Clear prompt")}
              onClick={clearPromptFromRail}
            >
              {t("cockpit.clearPrompt", "Clear prompt")}
            </button>
          ) : null}
        </div>
      </section>

      <section
        className={railSectionClass}
        aria-label={t("cockpit.conversationSession", "Conversation session")}
      >
        <h2 className={railHeadingClass}>{t("cockpit.session", "Session")}</h2>
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={railValueClass}>{sessionLabel}</p>
            {sessionTitle ? (
              <p className="mt-0.5 truncate text-xs font-medium text-text">
                {sessionTitle}
              </p>
            ) : null}
          </div>
          <span
            className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${sessionStatusClass(
              sessionStatus,
            )}`}
          >
            {effectiveSessionStatusLabel}
          </span>
        </div>
        {sessionDetail ? (
          <p className={railMutedClass}>{sessionDetail}</p>
        ) : null}
        {sessionError && sessionError !== sessionDetail ? (
          <p className="mt-1 rounded border border-danger/30 bg-danger/10 px-2 py-1 text-xs text-danger">
            {sessionError}
          </p>
        ) : null}
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
