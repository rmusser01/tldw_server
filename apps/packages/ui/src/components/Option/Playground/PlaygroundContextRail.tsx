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
import { DEGRADED_STATE_LABEL, READY_STATE_LABEL } from "@/design-system";
import { PlaygroundCompositionPreview } from "./PlaygroundCompositionPreview";
import type { PlaygroundCompositionPreviewSummary } from "./playground-composition-preview";
import {
  cockpitRailStyles,
  cockpitRailToneClass,
} from "./playground-cockpit-rail-styles";
import { PlaygroundRailSection } from "./PlaygroundRailSection";

const railActionClass = cockpitRailStyles.action;
const PROMPT_SELECT_TRIGGER_SELECTOR = "[data-cockpit-prompt-select-trigger]";

type ContextCountItem = {
  label: string;
  clearLabel: string;
  onClear: (() => void) | undefined;
};

export type PlaygroundPromptContextState =
  | "none"
  | "system"
  | "quick"
  | "custom";

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
  assistantControl?: React.ReactNode;
  characterSessionsPanel?: React.ReactNode;
  onClearPrompt?: () => void;
  onOpenSearchContext: () => void;
  onClearFiles?: () => void;
  onClearKnowledge?: () => void;
  onClearMedia?: () => void;
  onClearResearch?: () => void;
  compositionPreviewSummary?: PlaygroundCompositionPreviewSummary;
  setupRecoveryMode?: boolean;
};

const sourceIcon = (kind: PlaygroundContextSource["kind"]) => {
  const className = "h-3.5 w-3.5";
  if (kind === "web")
    return <Globe2 className={className} aria-hidden="true" />;
  if (kind === "file")
    return <FileText className={className} aria-hidden="true" />;
  if (kind === "knowledge")
    return <BookOpen className={className} aria-hidden="true" />;
  if (kind === "media")
    return <Database className={className} aria-hidden="true" />;
  if (kind === "assistant")
    return <UserRound className={className} aria-hidden="true" />;
  return <Layers3 className={className} aria-hidden="true" />;
};

const sourceStateClass = (state: PlaygroundContextSourceState = "active") => {
  if (state === "degraded") return cockpitRailToneClass("warning");
  if (state === "disabled") return cockpitRailToneClass("muted");
  if (state === "available") return cockpitRailToneClass("info");
  return cockpitRailToneClass("muted");
};

const sessionStatusClass = (
  status: NonNullable<PlaygroundContextRailProps["sessionStatus"]> = "idle",
) => {
  if (status === "failed") return cockpitRailToneClass("danger");
  if (status === "loading") return cockpitRailToneClass("info");
  if (status === "loaded") return cockpitRailToneClass("success");
  return cockpitRailToneClass("muted");
};

export const PlaygroundContextRail = ({
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
  assistantControl,
  characterSessionsPanel,
  onClearPrompt,
  onOpenSearchContext,
  onClearFiles,
  onClearKnowledge,
  onClearMedia,
  onClearResearch,
  compositionPreviewSummary,
  setupRecoveryMode = false,
}: PlaygroundContextRailProps) => {
  const { t } = useTranslation("playground");
  const railRef = React.useRef<HTMLDivElement | null>(null);
  const clearPromptFromRail = React.useCallback(() => {
    onClearPrompt?.();
    railRef.current
      ?.querySelector<HTMLElement>(PROMPT_SELECT_TRIGGER_SELECTOR)
      ?.focus();
  }, [onClearPrompt]);
  const sources = contextSources.filter(
    (source) =>
      !(source.kind === "prompt" && promptSelectControl) &&
      !(source.kind === "assistant" && assistantControl),
  );
  const selectedSourceCount = sources.filter(
    (source) => source.state !== "disabled",
  ).length;
  const effectivePromptSummary = promptSummary || {
    state: "none" as const,
    label: t("cockpit.noPromptSelected", "No prompt selected"),
    detail: t(
      "cockpit.noAdditionalSystemPrompt",
      "No additional system prompt selected.",
    ),
  };
  const promptSelected = effectivePromptSummary.state !== "none";
  const showActionablePromptEmptyState =
    !promptSelected && Boolean(compositionPreviewSummary);
  const promptManagementLabel = showActionablePromptEmptyState
    ? t("cockpit.promptReady", "Ready to add prompt")
    : effectivePromptSummary.label;
  const promptManagementDetail = showActionablePromptEmptyState
    ? t(
        "cockpit.promptReadyDetail",
        "Select a prompt to add system instructions.",
      )
    : effectivePromptSummary.detail;
  const effectiveSessionStatusLabel =
    sessionStatusLabel ||
    (sessionStatus === "failed"
      ? t("cockpit.sessionLoadFailed", "Load failed")
      : sessionStatus === "loading"
        ? t("cockpit.sessionLoading", "Loading conversation")
        : sessionStatus === "loaded"
          ? t("cockpit.sessionReady", READY_STATE_LABEL)
          : t("cockpit.sessionIdle", "Idle"));
  const sourceCountLabel =
    selectedSourceCount === 1
      ? t("cockpit.selectedSourceCountOne", "1 selected source")
      : t("cockpit.selectedSourceCountMany", "{{count}} selected sources", {
          count: selectedSourceCount,
        });
  const countLabels = (
    [
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
    ] satisfies Array<ContextCountItem | null>
  ).filter((item): item is ContextCountItem => Boolean(item));
  const recoveryModeKey = setupRecoveryMode ? "setup" : "standard";

  return (
    <div
      ref={railRef}
      data-testid="playground-context-rail"
      className={cockpitRailStyles.stack}
    >
      <PlaygroundRailSection
        key={`instructions-${recoveryModeKey}`}
        label={t("cockpit.instructions", "Instructions")}
        title={t("cockpit.instructions", "Instructions")}
        defaultOpen={!setupRecoveryMode}
      >
        <p className="text-xs text-text-muted">
          {t("cockpit.prompt", "Prompt")}
        </p>
        <p className={cockpitRailStyles.value}>{promptManagementLabel}</p>
        {promptSelected && promptManagementDetail && (
          <p className={cockpitRailStyles.muted}>{promptManagementDetail}</p>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-1">
          {promptSelectControl}
          {promptSelected && onClearPrompt && (
            <button
              type="button"
              className={railActionClass}
              aria-label={t("cockpit.clearPrompt", "Clear prompt")}
              onClick={clearPromptFromRail}
            >
              {t("cockpit.clearPrompt", "Clear prompt")}
            </button>
          )}
        </div>
        {assistantControl}
      </PlaygroundRailSection>

      <PlaygroundRailSection
        label={t("cockpit.contextStack", "Context stack")}
        title={t("cockpit.sources", "Sources")}
      >
        {selectedSourceCount > 0 && (
          <p className={cockpitRailStyles.muted}>{sourceCountLabel}</p>
        )}
        {sources.length > 0 ? (
          <ul aria-label={t("cockpit.contextSources", "Context sources")}>
            {sources.map((source) => {
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
                  className="border-b border-border/60 py-3 last:border-b-0"
                >
                  <div className="flex items-start gap-2">
                    <span className="mt-1 shrink-0 text-text-muted">
                      {sourceIcon(source.kind)}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-text-muted">
                          {source.label}
                        </span>
                        {source.state && source.state !== "active" && (
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
                                  : t("cockpit.selected", "Selected")}
                          </span>
                        )}
                      </div>
                      <p
                        className="mt-1 break-words text-sm font-medium text-text"
                        title={source.title}
                      >
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
                          className={cockpitRailStyles.clearAction}
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
                          className={cockpitRailStyles.clearAction}
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
          <p className={cockpitRailStyles.muted}>
            {t("cockpit.noSourcesAttached", "No sources attached.")}
          </p>
        )}
        {!sources.length && contextSummary.length > 0 && (
          <ul
            className="mt-2 space-y-1 text-xs text-text-muted"
            aria-label={t("cockpit.contextSummary", "Context summary")}
          >
            {contextSummary.map((item, index) => (
              <li key={`${index}-${item}`}>{item}</li>
            ))}
          </ul>
        )}
        <div className="mt-3 flex flex-wrap gap-1">
          <button
            type="button"
            onClick={onOpenSearchContext}
            className={railActionClass}
            aria-label={t("cockpit.openSearchContext", "Open Search & Context")}
          >
            <Search className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.addSources", "Add sources")}
          </button>
          <button
            type="button"
            onClick={onToggleWebSearch}
            className={`${railActionClass} ${webSearch ? "bg-primary/10 text-primaryStrong" : ""}`}
            aria-label={t("cockpit.webSearch", "Web search")}
            aria-pressed={webSearch}
          >
            <Globe2 className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
            {webSearch
              ? t("cockpit.webSearchOn", "Web search on")
              : t("cockpit.webSearchOff", "Web search off")}
          </button>
        </div>
        {countLabels.length > 0 && (
          <ul
            className="mt-3 space-y-1 text-xs text-text-muted"
            aria-label={t("cockpit.sourceSummary", "Source summary")}
          >
            {countLabels.map((item, index) => (
              <li
                key={`${index}-${item.label}`}
                className="flex items-center justify-between gap-2"
              >
                <span>{item.label}</span>
                {item.onClear && (
                  <button
                    type="button"
                    className={cockpitRailStyles.clearAction}
                    aria-label={item.clearLabel}
                    title={item.clearLabel}
                    onClick={item.onClear}
                  >
                    {t("cockpit.clear", "Clear")}
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </PlaygroundRailSection>

      {characterSessionsPanel}
      <PlaygroundRailSection
        key={`session-${recoveryModeKey}`}
        label={t("cockpit.session", "Session")}
        title={t("cockpit.conversation", "Conversation")}
        defaultOpen={!setupRecoveryMode || sessionStatus === "failed"}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={cockpitRailStyles.value}>{sessionLabel}</p>
            {sessionTitle && (
              <p
                className="mt-1 break-words text-xs text-text"
                title={sessionTitle}
              >
                {sessionTitle}
              </p>
            )}
          </div>
          {sessionStatus !== "idle" && (
            <span
              className={`rounded px-2 py-1 text-xs ${sessionStatusClass(sessionStatus)}`}
            >
              {effectiveSessionStatusLabel}
            </span>
          )}
        </div>
        {sessionDetail && (
          <p className={cockpitRailStyles.muted}>{sessionDetail}</p>
        )}
        {sessionError && sessionError !== sessionDetail && (
          <p className="mt-2 rounded bg-danger/10 p-2 text-xs text-text">
            {sessionError}
          </p>
        )}
        {historyLinked && (
          <p className={cockpitRailStyles.muted}>
            {t("cockpit.historyLinked", "History linked")}
          </p>
        )}
        <button
          type="button"
          onClick={() => onToggleTemporaryChat(!temporaryChat)}
          className={`${railActionClass} mt-2`}
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
      </PlaygroundRailSection>
      {compositionPreviewSummary && (
        <PlaygroundCompositionPreview
          summary={compositionPreviewSummary}
          defaultOpen={false}
        />
      )}
    </div>
  );
};
