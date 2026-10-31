import React from "react";
import { useTranslation } from "react-i18next";
import { Clock, MessageSquareText } from "lucide-react";

import {
  useServerChatHistory,
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
  type ServerChatHistoryItem,
} from "@/hooks/useServerChatHistory";
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner";
import { useConnectionState } from "@/hooks/useConnectionState";
import { useConnectionStore } from "@/store/connection";
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat";
import { formatRelativeTime } from "@/utils/dateFormatters";
import { cn } from "@/libs/utils";
import { cockpitRailStyles } from "./playground-cockpit-rail-styles";
import { PlaygroundRailSection } from "./PlaygroundRailSection";

type CharacterChatSessionsPanelProps = {
  activeCharacterId?: string | number | null;
  activeCharacterName?: string | null;
  activeServerChatId?: string | null;
  enabled?: boolean;
  limit?: number;
};

const DEFAULT_SESSION_LIMIT = 5;
const GLOBAL_CHAT_SCOPE = { type: "global" as const };
const isGlobalSession = (session: ServerChatHistoryItem) =>
  (!session.scope_type || session.scope_type === "global") &&
  !session.workspace_id;

const normalizeId = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return null;
};

const getSessionUpdatedAt = (chat: ServerChatHistoryItem): string | null => {
  if (
    typeof chat.updated_at === "string" &&
    chat.updated_at.trim().length > 0
  ) {
    return chat.updated_at.trim();
  }
  if (
    typeof chat.last_active === "string" &&
    chat.last_active.trim().length > 0
  ) {
    return chat.last_active.trim();
  }
  if (
    typeof chat.created_at === "string" &&
    chat.created_at.trim().length > 0
  ) {
    return chat.created_at.trim();
  }
  return null;
};

const getSessionCharacterIdentity = (
  session: ServerChatHistoryItem,
): string | null => {
  const characterId = normalizeId(session.character_id);
  if (characterId) return characterId;
  if (session.assistant_kind !== "character") return null;
  return normalizeId(session.assistant_id);
};

const getSessionCharacterLabel = (
  session: ServerChatHistoryItem,
  activeCharacterId: string | number | null | undefined,
  activeCharacterName: string | null | undefined,
  t: ReturnType<typeof useTranslation>["t"],
): string | null => {
  const explicitName =
    typeof session.character_name === "string" &&
    session.character_name.trim().length > 0
      ? session.character_name.trim()
      : typeof session.assistant_name === "string" &&
          session.assistant_name.trim().length > 0
        ? session.assistant_name.trim()
        : null;
  if (explicitName) return explicitName;

  const sessionCharacterId = getSessionCharacterIdentity(session);
  if (!sessionCharacterId) return null;
  if (
    activeCharacterName &&
    normalizeId(activeCharacterId) === sessionCharacterId
  ) {
    return activeCharacterName;
  }
  return t("characterChatSessions.characterFallback", "Character {{id}}", {
    id: sessionCharacterId,
  });
};

const nativeCharacterId = (value: unknown): number | undefined => {
  const normalized = normalizeId(value);
  if (!normalized || !/^\d+$/.test(normalized)) return undefined;
  const id = Number(normalized);
  return Number.isSafeInteger(id) && id > 0 ? id : undefined;
};

type SessionListProps = {
  label: string;
  sessions: ServerChatHistoryItem[];
  activeCharacterId?: string | number | null;
  activeCharacterName?: string | null;
  activeServerChatId?: string | null;
  onSelectSession: (chat: ServerChatHistoryItem) => void;
  canResume: boolean;
};

const CharacterSessionList = ({
  label,
  sessions,
  activeCharacterId,
  activeCharacterName,
  activeServerChatId,
  onSelectSession,
  canResume,
}: SessionListProps) => {
  const { t } = useTranslation(["playground", "common"]);

  if (sessions.length === 0) return null;
  const titles = sessions.map(
    (session) =>
      session.title?.trim() ||
      t("characterChatSessions.untitled", "Untitled character chat"),
  );
  const duplicateTitles = new Set(
    titles.filter((title, index) => titles.indexOf(title) !== index),
  );

  return (
    <ul aria-label={label} className="mt-2 space-y-2">
      {sessions.map((session) => {
        const isActive = activeServerChatId === session.id;
        const updatedAt = getSessionUpdatedAt(session);
        const updatedLabel = updatedAt
          ? formatRelativeTime(updatedAt, t, { compact: true })
          : null;
        const title =
          typeof session.title === "string" && session.title.trim().length > 0
            ? session.title.trim()
            : t("characterChatSessions.untitled", "Untitled character chat");
        const duplicateTitle = duplicateTitles.has(title);
        const actionTitle = duplicateTitle
          ? t(
              "characterChatSessions.titleReference",
              "{{title}}, session {{id}}",
              { title, id: session.id },
            )
          : title;
        const topic =
          typeof session.topic_label === "string" &&
          session.topic_label.trim().length > 0
            ? session.topic_label.trim()
            : null;
        const messageCount =
          typeof session.message_count === "number" &&
          Number.isFinite(session.message_count)
            ? Math.max(0, Math.trunc(session.message_count))
            : null;
        const characterLabel = getSessionCharacterLabel(
          session,
          activeCharacterId,
          activeCharacterName,
          t,
        );
        const persistenceLabel = t("characterChatSessions.saved", "Saved");

        return (
          <li
            key={session.id}
            className={cn(
              cockpitRailStyles.inset,
              isActive && "border-primary/40 bg-primary/5",
            )}
          >
            <div className="flex items-start gap-2">
              <span className="mt-0.5 rounded border border-border bg-surface2 p-1 text-text-muted">
                <MessageSquareText className="h-3.5 w-3.5" aria-hidden="true" />
              </span>
              <div className="min-w-0 flex-1">
                <p
                  className="truncate text-sm font-medium text-text"
                  title={title}
                >
                  {title}
                </p>
                {duplicateTitle ? (
                  <p className="mt-1 break-all text-[11px] text-text-muted">
                    {t(
                      "characterChatSessions.sessionReference",
                      "Session {{id}}",
                      { id: session.id },
                    )}
                  </p>
                ) : null}
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted">
                  {characterLabel ? (
                    <span className="truncate">{characterLabel}</span>
                  ) : null}
                  {updatedLabel ? (
                    <time
                      dateTime={updatedAt ?? undefined}
                      title={updatedAt ?? undefined}
                      className="inline-flex items-center gap-1"
                    >
                      <Clock className="h-3 w-3" aria-hidden="true" />
                      {updatedLabel}
                    </time>
                  ) : null}
                  {topic ? <span className="truncate">{topic}</span> : null}
                  {messageCount != null ? (
                    <span>
                      {t(
                        "characterChatSessions.messageCount",
                        "{{count}} messages",
                        { count: messageCount },
                      )}
                    </span>
                  ) : null}
                  <span>{persistenceLabel}</span>
                </div>
              </div>
              <button
                type="button"
                disabled={isActive || !canResume}
                onClick={() => {
                  onSelectSession(session);
                }}
                aria-label={
                  isActive
                    ? t(
                        "characterChatSessions.currentSessionAction",
                        "Current {{title}}",
                        {
                          title: actionTitle,
                        },
                      )
                    : t(
                        "characterChatSessions.resumeSessionAction",
                        "Resume {{title}}",
                        {
                          title: actionTitle,
                        },
                      )
                }
                className={cn(
                  cockpitRailStyles.action,
                  "shrink-0 px-2 py-1 text-[11px]",
                  isActive && "cursor-default opacity-70",
                )}
              >
                {isActive
                  ? t("characterChatSessions.current", "Current")
                  : t("characterChatSessions.resume", "Resume")}
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
};

export const CharacterChatSessionsPanel = ({
  activeCharacterId,
  activeCharacterName,
  activeServerChatId,
  enabled = true,
  limit = DEFAULT_SESSION_LIMIT,
}: CharacterChatSessionsPanelProps) => {
  const { t } = useTranslation(["playground", "common"]);
  const selectServerChat = useSelectServerChat();
  const verified = useChatDraftOwner(() => {});
  const { isConnected } = useConnectionState();
  const characterId = nativeCharacterId(activeCharacterId);
  const hasCharacter = normalizeId(activeCharacterId) != null;
  const supportedCharacter = !hasCharacter || characterId != null;
  const owner = React.useMemo(
    () =>
      verified.ownerKey && verified.requestScope
        ? {
            key: verified.ownerKey,
            requestScope: verified.requestScope,
            isCurrent: verified.isCurrent,
          }
        : undefined,
    [verified.ownerKey, verified.requestScope, verified.isCurrent],
  );
  // The full-page selector and this transient view belong to the global chat area.
  const identity = React.useMemo(
    () => ({ owner, activeCharacterId }),
    [owner, activeCharacterId],
  );
  const [view, setView] = React.useState({
    identity,
    all: false,
    page: 1,
    other: false,
  });
  const currentView =
    view.identity === identity
      ? view
      : { identity, all: false, page: 1, other: false };
  const currentIdentity = React.useRef<object | null>(identity);
  currentIdentity.current = identity;
  React.useEffect(
    () => () => {
      currentIdentity.current = null;
    },
    [],
  );
  const ownerCurrent = !!owner?.isCurrent();
  const canResume = enabled && isConnected && ownerCurrent;
  const latestEnabled = React.useRef(enabled);
  latestEnabled.current = enabled;
  const canDispatch = () =>
    latestEnabled.current &&
    useConnectionStore.getState().state.isConnected &&
    owner?.isCurrent() &&
    currentIdentity.current === identity;
  const previewLimit = Math.max(
    1,
    Math.min(
      200,
      Number.isFinite(limit) ? Math.trunc(limit) : DEFAULT_SESSION_LIMIT,
    ),
  );
  const pageLimit = currentView.all
    ? SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE
    : previewLimit;
  const primary = useServerChatHistory("", {
    enabled: enabled && ownerCurrent && supportedCharacter,
    owner,
    scope: GLOBAL_CHAT_SCOPE,
    characterId,
    mode: "overview",
    page: currentView.page,
    limit: pageLimit,
    filterMode: "character",
  });
  const other = useServerChatHistory("", {
    enabled: enabled && ownerCurrent && currentView.other,
    owner,
    scope: GLOBAL_CHAT_SCOPE,
    mode: "overview",
    page: 1,
    limit: previewLimit,
    filterMode: "character",
  });
  const sessions =
    ownerCurrent && supportedCharacter
      ? primary.data.filter((session) => {
          const id = nativeCharacterId(getSessionCharacterIdentity(session));
          return (
            isGlobalSession(session) &&
            id != null &&
            (!hasCharacter || id === characterId)
          );
        })
      : [];
  const otherSessions =
    ownerCurrent && currentView.other
      ? other.data.filter((session) => {
          const id = nativeCharacterId(getSessionCharacterIdentity(session));
          return isGlobalSession(session) && id != null && id !== characterId;
        })
      : [];
  const unverifiedSessions = sessions.length !== primary.data.length;
  const pageOffset = (currentView.page - 1) * pageLimit;
  // The endpoint may report a page-length count when its count lookup fails.
  const hasNext =
    primary.data.length >= pageLimit ||
    primary.total > pageOffset + primary.data.length;
  const lastSession =
    !activeServerChatId && currentView.page === 1 ? sessions[0] : null;
  const lastSessionTitle =
    lastSession?.title?.trim() ||
    t("characterChatSessions.untitled", "Untitled character chat");
  const select = (session: ServerChatHistoryItem) => {
    if (!canDispatch()) return;
    selectServerChat(session);
  };
  const changeView = (next: Partial<typeof currentView>) => {
    if (!owner?.isCurrent() || currentIdentity.current !== identity) return;
    setView({ ...currentView, ...next });
  };
  const primaryError =
    primary.sidebarRefreshState === "recoverable-error" ||
    primary.sidebarRefreshState === "hard-error";
  const listLabel = activeCharacterName
    ? t(
        currentView.all
          ? "characterChatSessions.allCharacterListLabel"
          : "characterChatSessions.currentCharacterListLabel",
        currentView.all
          ? "All sessions for {{name}}"
          : "Recent sessions for {{name}}",
        { name: activeCharacterName },
      )
    : t(
        "characterChatSessions.currentCharacterFallbackLabel",
        "Recent character sessions",
      );

  const historyFeedback = (
    query: typeof primary,
    count: number,
    isPrimary: boolean,
  ) => {
    const failed =
      query.sidebarRefreshState === "recoverable-error" ||
      query.sidebarRefreshState === "hard-error";
    if (!isConnected && count > 0)
      return (
        <p role="status" className={cockpitRailStyles.muted}>
          {t(
            "characterChatSessions.offlineCached",
            "Reconnect to load or resume character sessions. Showing the last loaded page.",
          )}
        </p>
      );
    if (!isConnected)
      return (
        <p role="status" className={cockpitRailStyles.emptyInset}>
          {t(
            "characterChatSessions.unavailable",
            "Connect to your server to view character sessions.",
          )}
        </p>
      );
    if (!ownerCurrent)
      return verified.verificationState === "pending" ? (
        <p role="status" className={cockpitRailStyles.emptyInset}>
          {t(
            "characterChatSessions.verifyingAccount",
            "Verifying your server account...",
          )}
        </p>
      ) : (
        <div className="space-y-2">
          <p role="alert" className={cockpitRailStyles.emptyInset}>
            {t(
              "characterChatSessions.verificationFailed",
              "Could not verify your server account. Retry or check your connection settings.",
            )}
          </p>
          <button
            type="button"
            disabled={!enabled}
            onClick={() => {
              if (
                latestEnabled.current &&
                useConnectionStore.getState().state.isConnected &&
                currentIdentity.current === identity
              )
                verified.retryVerification();
            }}
            className={cockpitRailStyles.action}
          >
            {t(
              "characterChatSessions.retryVerification",
              "Retry account verification",
            )}
          </button>
        </div>
      );
    if (
      query.isLoading ||
      (query.sidebarRefreshState === "idle" && count === 0)
    )
      return (
        <p role="status" className={cockpitRailStyles.emptyInset}>
          {t("characterChatSessions.loading", "Loading character sessions...")}
        </p>
      );
    if (failed || query.isShowingStaleData)
      return (
        <div className="mt-2 space-y-2">
          <p
            role={count > 0 ? "status" : "alert"}
            className={cockpitRailStyles.emptyInset}
          >
            {count > 0
              ? t(
                  "characterChatSessions.stale",
                  "Showing character sessions from the last successful refresh.",
                )
              : query.sidebarRefreshState === "recoverable-error"
                ? t(
                    "characterChatSessions.refreshFailed",
                    "Unable to refresh character sessions right now.",
                  )
                : t(
                    "characterChatSessions.loadFailed",
                    "Character sessions could not be loaded.",
                  )}
          </p>
          <button
            type="button"
            disabled={!canResume || query.isFetching}
            className={cockpitRailStyles.action}
            onClick={() => {
              if (canDispatch()) void query.refetch();
            }}
          >
            {t("characterChatSessions.retry", "Retry character sessions")}
          </button>
        </div>
      );
    if (count > 0 || (isPrimary && unverifiedSessions)) return null;
    return (
      <p className={cockpitRailStyles.emptyInset}>
        {!isPrimary
          ? t(
              "characterChatSessions.otherEmpty",
              "No other characters in this recent sample.",
            )
          : currentView.page > 1
            ? t(
                "characterChatSessions.pageEmpty",
                "No sessions on this page. Return to the previous page to see earlier results.",
              )
            : hasCharacter
              ? t(
                  "characterChatSessions.characterEmpty",
                  "No conversations with this character yet.",
                )
              : t(
                  "characterChatSessions.empty",
                  "No character conversations yet.",
                )}
      </p>
    );
  };

  return (
    <section
      role="region"
      aria-label={t(
        "characterChatSessions.regionLabel",
        "Character chat sessions",
      )}
    >
      <PlaygroundRailSection
        label={t("characterChatSessions.label", "Character sessions")}
        title={t("characterChatSessions.title", "Character conversations")}
      >
        <p className={cockpitRailStyles.value}>
          {t("characterChatSessions.resumeTitle", "Resume role-play")}
        </p>
        <p className={cockpitRailStyles.muted}>
          {t(
            "characterChatSessions.resumeDetail",
            "Recent character conversations stay separate from saved role-play setups.",
          )}
        </p>
        {lastSession ? (
          <button
            type="button"
            disabled={!canResume}
            onClick={() => select(lastSession)}
            aria-label={t(
              "characterChatSessions.resumeLastAction",
              "Resume last character chat: {{title}}",
              { title: lastSessionTitle },
            )}
            className={cn(
              cockpitRailStyles.action,
              "mt-3 w-full justify-center",
            )}
          >
            {t(
              "characterChatSessions.resumeLast",
              "Resume last character chat",
            )}
          </button>
        ) : null}
        <div
          className="mt-3 space-y-2"
          aria-busy={primary.isFetching || undefined}
        >
          {!supportedCharacter ? (
            <p role="status" className={cockpitRailStyles.emptyInset}>
              {t(
                "characterChatSessions.localCard",
                "Server sessions are available for characters saved on this server.",
              )}
            </p>
          ) : (
            historyFeedback(primary, sessions.length, true)
          )}
          {unverifiedSessions && supportedCharacter && ownerCurrent ? (
            <p role="alert" className={cockpitRailStyles.emptyInset}>
              {t(
                "characterChatSessions.unverified",
                "Some sessions in this page could not be verified for this character. Retry to refresh.",
              )}
              <button
                type="button"
                disabled={!canResume || primary.isFetching}
                onClick={() => {
                  if (canDispatch()) void primary.refetch();
                }}
                className={cn(cockpitRailStyles.action, "mt-2")}
              >
                {t("characterChatSessions.retry", "Retry character sessions")}
              </button>
            </p>
          ) : null}
          {supportedCharacter ? (
            <CharacterSessionList
              label={listLabel}
              sessions={sessions}
              activeCharacterId={characterId}
              activeCharacterName={activeCharacterName}
              activeServerChatId={activeServerChatId}
              canResume={canResume}
              onSelectSession={select}
            />
          ) : null}
          {supportedCharacter && ownerCurrent && enabled ? (
            <button
              type="button"
              aria-expanded={currentView.all}
              onClick={() => changeView({ all: !currentView.all, page: 1 })}
              className={cockpitRailStyles.action}
            >
              {currentView.all
                ? t(
                    "characterChatSessions.backToRecent",
                    "Back to recent sessions",
                  )
                : hasCharacter
                  ? t(
                      "characterChatSessions.allSessions",
                      "All sessions for this character",
                    )
                  : t(
                      "characterChatSessions.allCharacters",
                      "All character sessions",
                    )}
            </button>
          ) : null}
          {currentView.all && supportedCharacter && ownerCurrent ? (
            <nav
              aria-label={t(
                "characterChatSessions.pagination",
                "Character sessions pages",
              )}
              className="flex flex-wrap items-center gap-2"
            >
              <button
                type="button"
                className={cockpitRailStyles.action}
                disabled={currentView.page === 1}
                aria-label={t(
                  "characterChatSessions.previousPage",
                  "Previous sessions page",
                )}
                onClick={() => changeView({ page: currentView.page - 1 })}
              >
                {t("characterChatSessions.previous", "Previous")}
              </button>
              <span role="status" className={cockpitRailStyles.muted}>
                {t("characterChatSessions.page", "Page {{page}}", {
                  page: currentView.page,
                })}
              </span>
              <button
                type="button"
                className={cockpitRailStyles.action}
                disabled={
                  !canResume || primary.isFetching || !hasNext || primaryError
                }
                aria-label={t(
                  "characterChatSessions.nextPage",
                  "Next sessions page",
                )}
                onClick={() => changeView({ page: currentView.page + 1 })}
              >
                {t("characterChatSessions.next", "Next")}
              </button>
            </nav>
          ) : null}
        </div>
        {hasCharacter && ownerCurrent && enabled ? (
          <div className="mt-3">
            <button
              type="button"
              aria-expanded={currentView.other}
              onClick={() => changeView({ other: !currentView.other })}
              className={cockpitRailStyles.action}
            >
              {t(
                "characterChatSessions.otherRecents",
                "Recent chats with other characters",
              )}
            </button>
            {currentView.other ? (
              <div className="mt-2 space-y-2">
                <p className={cockpitRailStyles.muted}>
                  {t(
                    "characterChatSessions.otherSample",
                    "A sample from the {{count}} most recent character chats; older chats may not appear here.",
                    { count: previewLimit },
                  )}
                </p>
                {historyFeedback(other, otherSessions.length, false)}
                <CharacterSessionList
                  label={t(
                    "characterChatSessions.otherCharactersListLabel",
                    "Other character sessions",
                  )}
                  sessions={otherSessions}
                  activeServerChatId={activeServerChatId}
                  canResume={canResume}
                  onSelectSession={select}
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </PlaygroundRailSection>
    </section>
  );
};

export default CharacterChatSessionsPanel;
