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
    return chat.updated_at;
  }
  if (
    typeof chat.last_active === "string" &&
    chat.last_active.trim().length > 0
  ) {
    return chat.last_active;
  }
  if (
    typeof chat.created_at === "string" &&
    chat.created_at.trim().length > 0
  ) {
    return chat.created_at;
  }
  return null;
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
    const sessionCharacterId = normalizeId(session.character_id);
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
  activeServerChatId?: string | null;
  onSelectSession: (chat: ServerChatHistoryItem) => void;
};

const CharacterSessionList = ({
  label,
  sessions,
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
                </div>
              </div>
              <button
                type="button"
                disabled={isActive}
                onClick={() => {
                  if (!isActive) onSelectSession(session);
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
  const sessions = Array.isArray(data) ? data.slice(0, limit) : [];
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
            className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${cockpitRailToneClass(
              hasSessions ? "success" : "muted",
            )}`}
          >
            {hasSessions
              ? t("characterChatSessions.available", "Available")
              : t("characterChatSessions.emptyBadge", "None")}
          </span>
        </div>

        {isLoading && !hasUsableData ? (
          <div className={`mt-3 ${cockpitRailStyles.emptyInset}`}>
            {t(
              "characterChatSessions.loading",
              "Loading character sessions...",
            )}
          </div>
        ) : sidebarRefreshState === "recoverable-error" && !hasUsableData ? (
          <div className={`mt-3 ${cockpitRailStyles.emptyInset}`}>
            {t(
              "characterChatSessions.refreshFailed",
              "Unable to refresh character sessions right now.",
            )}
          </div>
        ) : hasSessions ? (
          <>
            {isShowingStaleData ? (
              <p className="mt-2 rounded border border-warning/30 bg-warning/10 px-2 py-1 text-xs text-text-subtle">
                {t(
                  "characterChatSessions.stale",
                  "Showing character sessions from the last successful refresh.",
                )}
              </p>
            ) : null}
            <CharacterSessionList
              label={currentCharacterLabel}
              sessions={currentCharacterSessions}
              activeServerChatId={activeServerChatId}
              onSelectSession={selectServerChat}
            />
            <CharacterSessionList
              label={t(
                "characterChatSessions.otherCharactersListLabel",
                "Other character sessions",
              )}
              sessions={otherCharacterSessions}
              activeServerChatId={activeServerChatId}
              onSelectSession={selectServerChat}
            />
          </>
        ) : (
          <div className={`mt-3 ${cockpitRailStyles.emptyInset}`}>
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
