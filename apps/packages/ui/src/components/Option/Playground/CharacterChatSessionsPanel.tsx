import React from "react";
import { useTranslation } from "react-i18next";
import { Clock, MessageSquareText } from "lucide-react";

import {
  useServerChatHistory,
  type ServerChatHistoryItem,
} from "@/hooks/useServerChatHistory";
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat";
import { formatRelativeTime } from "@/utils/dateFormatters";
import { cn } from "@/libs/utils";
import {
  cockpitRailStyles,
  cockpitRailToneClass,
} from "./playground-cockpit-rail-styles";
import { PlaygroundRailSection } from "./PlaygroundRailSection";

type CharacterChatSessionsPanelProps = {
  activeCharacterId?: string | number | null;
  activeCharacterName?: string | null;
  activeServerChatId?: string | null;
  enabled?: boolean;
  limit?: number;
};

const DEFAULT_SESSION_LIMIT = 5;

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

const partitionCharacterSessions = (
  sessions: ServerChatHistoryItem[],
  activeCharacterId: string | number | null | undefined,
): {
  currentCharacterSessions: ServerChatHistoryItem[];
  otherCharacterSessions: ServerChatHistoryItem[];
} => {
  const normalizedActiveId = normalizeId(activeCharacterId);
  if (!normalizedActiveId) {
    return {
      currentCharacterSessions: sessions,
      otherCharacterSessions: [],
    };
  }

  const currentCharacterSessions: ServerChatHistoryItem[] = [];
  const otherCharacterSessions: ServerChatHistoryItem[] = [];

  for (const session of sessions) {
    const sessionCharacterId = getSessionCharacterIdentity(session);
    if (sessionCharacterId === normalizedActiveId) {
      currentCharacterSessions.push(session);
    } else {
      otherCharacterSessions.push(session);
    }
  }

  return {
    currentCharacterSessions,
    otherCharacterSessions,
  };
};

type SessionListProps = {
  label: string;
  sessions: ServerChatHistoryItem[];
  activeCharacterId?: string | number | null;
  activeCharacterName?: string | null;
  activeServerChatId?: string | null;
  onSelectSession: (chat: ServerChatHistoryItem) => void;
};

const CharacterSessionList = ({
  label,
  sessions,
  activeCharacterId,
  activeCharacterName,
  activeServerChatId,
  onSelectSession,
}: SessionListProps) => {
  const { t } = useTranslation(["playground", "common"]);

  if (sessions.length === 0) return null;

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
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted">
                  {characterLabel ? (
                    <span className="truncate">{characterLabel}</span>
                  ) : null}
                  {updatedLabel ? (
                    <span className="inline-flex items-center gap-1">
                      <Clock className="h-3 w-3" aria-hidden="true" />
                      {updatedLabel}
                    </span>
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
                disabled={isActive}
                onClick={() => {
                  onSelectSession(session);
                }}
                aria-label={
                  isActive
                    ? t(
                        "characterChatSessions.currentSessionAction",
                        "Current {{title}}",
                        {
                          title,
                        },
                      )
                    : t(
                        "characterChatSessions.resumeSessionAction",
                        "Resume {{title}}",
                        {
                          title,
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
  const {
    data = [],
    isLoading,
    sidebarRefreshState,
    hasUsableData,
    isShowingStaleData,
  } = useServerChatHistory("", {
    enabled,
    mode: "overview",
    page: 1,
    limit,
    filterMode: "character",
  });
  const sessions = React.useMemo(
    () => (Array.isArray(data) ? data.slice(0, limit) : []),
    [data, limit],
  );
  const { currentCharacterSessions, otherCharacterSessions } = React.useMemo(
    () => partitionCharacterSessions(sessions, activeCharacterId),
    [activeCharacterId, sessions],
  );
  const currentCharacterLabel = activeCharacterName
    ? t(
        "characterChatSessions.currentCharacterListLabel",
        "Recent sessions for {{name}}",
        { name: activeCharacterName },
      )
    : t(
        "characterChatSessions.currentCharacterFallbackLabel",
        "Recent character sessions",
      );
  const hasSessions = sessions.length > 0;
  const lastSession = !activeServerChatId && hasSessions ? sessions[0] : null;
  const lastSessionTitle =
    typeof lastSession?.title === "string" &&
    lastSession.title.trim().length > 0
      ? lastSession.title.trim()
      : lastSession
        ? t("characterChatSessions.untitled", "Untitled character chat")
        : null;
  const hardErrorWithoutData =
    sidebarRefreshState === "hard-error" && !hasUsableData;
  const sessionBadgeTone = hasSessions
    ? "success"
    : hardErrorWithoutData
      ? "danger"
      : "muted";

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
        title={t("characterChatSessions.title", "Recent character chats")}
      >
        <div className="mt-1 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className={cockpitRailStyles.value}>
              {t("characterChatSessions.resumeTitle", "Resume role-play")}
            </p>
            <p className={cockpitRailStyles.muted}>
              {t(
                "characterChatSessions.resumeDetail",
                "Recent character conversations stay separate from saved role-play setups.",
              )}
            </p>
          </div>
          <span
            className={cn(
              "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold",
              cockpitRailToneClass(sessionBadgeTone),
            )}
          >
            {hasSessions
              ? t("characterChatSessions.available", "Available")
              : hardErrorWithoutData
                ? t("characterChatSessions.errorBadge", "Error")
                : t("characterChatSessions.emptyBadge", "None")}
          </span>
        </div>

        {lastSession && lastSessionTitle ? (
          <button
            type="button"
            onClick={() => {
              selectServerChat(lastSession);
            }}
            aria-label={t(
              "characterChatSessions.resumeLastAction",
              "Resume last character chat: {{title}}",
              { title: lastSessionTitle },
            )}
            className={cn(
              cockpitRailStyles.action,
              "mt-3 w-full justify-center border-primary/40 bg-primary/10 text-primary hover:bg-primary/15",
            )}
          >
            {t("characterChatSessions.resumeLast", "Resume last character chat")}
          </button>
        ) : null}

        {isLoading && !hasUsableData ? (
          <div
            role="status"
            aria-live="polite"
            className={cn("mt-3", cockpitRailStyles.emptyInset)}
          >
            {t(
              "characterChatSessions.loading",
              "Loading character sessions...",
            )}
          </div>
        ) : sidebarRefreshState === "recoverable-error" && !hasUsableData ? (
          <div role="alert" className={cn("mt-3", cockpitRailStyles.emptyInset)}>
            {t(
              "characterChatSessions.refreshFailed",
              "Unable to refresh character sessions right now.",
            )}
          </div>
        ) : hardErrorWithoutData ? (
          <div
            role="alert"
            className={cn(
              "mt-3",
              cockpitRailStyles.emptyInset,
              "border-danger/30 bg-danger/10 text-danger",
            )}
          >
            {t(
              "characterChatSessions.loadFailed",
              "Character sessions could not be loaded.",
            )}
          </div>
        ) : hasSessions ? (
          <>
            {isShowingStaleData ? (
              <p
                role="status"
                aria-live="polite"
                className="mt-2 rounded border border-warning/30 bg-warning/10 px-2 py-1 text-xs text-text-subtle"
              >
                {t(
                  "characterChatSessions.stale",
                  "Showing character sessions from the last successful refresh.",
                )}
              </p>
            ) : null}
            <CharacterSessionList
              label={currentCharacterLabel}
              sessions={currentCharacterSessions}
              activeCharacterId={activeCharacterId}
              activeCharacterName={activeCharacterName}
              activeServerChatId={activeServerChatId}
              onSelectSession={selectServerChat}
            />
            <CharacterSessionList
              label={t(
                "characterChatSessions.otherCharactersListLabel",
                "Other character sessions",
              )}
              sessions={otherCharacterSessions}
              activeCharacterId={activeCharacterId}
              activeCharacterName={activeCharacterName}
              activeServerChatId={activeServerChatId}
              onSelectSession={selectServerChat}
            />
          </>
        ) : (
          <div className={cn("mt-3", cockpitRailStyles.emptyInset)}>
            {t(
              "characterChatSessions.empty",
              "No character conversations yet.",
            )}
          </div>
        )}
      </PlaygroundRailSection>
    </section>
  );
};

export default CharacterChatSessionsPanel;
