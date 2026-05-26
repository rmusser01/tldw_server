import React from "react"
import { useTranslation } from "react-i18next"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import {
  type EffectiveAssistantState,
  resolveEffectiveAssistantState
} from "@/hooks/chat/effective-assistant-state"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import {
  type ServerChatHistoryItem,
  useServerChatHistory
} from "@/hooks/useServerChatHistory"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useStoreMessageOption } from "@/store/option"
import { getAssistantSelectionMode } from "@/types/assistant-selection"
import { dispatchOpenAssistantSelect } from "@/utils/assistant-select-events"
import { isTrackedCharacterChatSource } from "@/utils/character-chat-session"

const isTrackedSession = (item: ServerChatHistoryItem): boolean =>
  item.assistant_kind === "character" ||
  item.assistant_kind === "persona" ||
  (item.character_id != null && isTrackedCharacterChatSource(item.source))

const resolveModeLabel = (
  mode: ReturnType<typeof resolveEffectiveAssistantState>["mode"]
): string => {
  if (mode === "overlay") return "Overlay personality"
  if (mode === "tracked_character") return "Tracked character chat"
  if (mode === "tracked_persona") return "Tracked persona chat"
  return "Plain chat"
}

export const CharacterControlRail = () => {
  const { t } = useTranslation(["playground", "common"])
  const historyId = useStoreMessageOption((state) => state.historyId)
  const serverChatId = useStoreMessageOption((state) => state.serverChatId)
  const serverChatAssistantKind = useStoreMessageOption(
    (state) => state.serverChatAssistantKind
  )
  const serverChatAssistantId = useStoreMessageOption(
    (state) => state.serverChatAssistantId
  )
  const serverChatCharacterId = useStoreMessageOption(
    (state) => state.serverChatCharacterId
  )
  const serverChatLoadState = useStoreMessageOption(
    (state) => state.serverChatLoadState
  )
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const clearChat = useClearChat()
  const selectServerChat = useSelectServerChat()
  const { data: serverChatHistory = [] } = useServerChatHistory("", {
    filterMode: "all"
  })

  const effectiveAssistantState = React.useMemo<EffectiveAssistantState>(
    () => {
      const resolved = resolveEffectiveAssistantState({
        tracked: {
          assistantKind: serverChatAssistantKind,
          assistantId: serverChatAssistantId,
          characterId: serverChatCharacterId
        },
        settings: settings ?? null,
        draftSelection: selectedAssistant
      })

      if (resolved.mode !== "plain") {
        return resolved
      }

      if (
        serverChatId &&
        (serverChatLoadState === "loading" ||
          serverChatLoadState === "failed") &&
        selectedAssistant &&
        getAssistantSelectionMode(selectedAssistant) === "tracked"
      ) {
        return {
          mode:
            selectedAssistant.kind === "persona"
              ? "tracked_persona"
              : "tracked_character",
          kind: selectedAssistant.kind,
          id: selectedAssistant.id,
          displayName: selectedAssistant.name ?? null,
          avatarUrl: selectedAssistant.avatar_url ?? null,
          systemPromptSnapshot: selectedAssistant.system_prompt ?? null,
          source: "tracked" as const
        }
      }

      return resolved
    },
    [
      selectedAssistant,
      serverChatId,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatCharacterId,
      serverChatLoadState,
      settings
    ]
  )

  const trackedSessions = React.useMemo(
    () => serverChatHistory.filter(isTrackedSession).slice(0, 6),
    [serverChatHistory]
  )
  const isTrackedMode =
    effectiveAssistantState.mode === "tracked_character" ||
    effectiveAssistantState.mode === "tracked_persona"

  const overlayTab =
    effectiveAssistantState.kind === "persona" ? "persona" : "character"
  const overlayActionLabel =
    effectiveAssistantState.mode === "overlay"
      ? t("playground:characterRail.changeOverlay", "Change overlay")
      : t("playground:characterRail.applyOverlay", "Apply overlay")

  const handleOverlaySelect = React.useCallback(() => {
    dispatchOpenAssistantSelect({
      tab: overlayTab,
      applyAs: "overlay",
      source: "character-control-rail"
    })
  }, [overlayTab])

  const handleClearOverlay = React.useCallback(async () => {
    await updateSettings({
      assistantOverlay: null
    })
    await setSelectedAssistant(null)
  }, [setSelectedAssistant, updateSettings])

  const handleStartTracked = React.useCallback(
    async (tab: "character" | "persona") => {
      await updateSettings({
        assistantOverlay: null
      })
      await setSelectedAssistant(null)
      clearChat()
      dispatchOpenAssistantSelect({
        tab,
        applyAs: "tracked",
        source: "character-control-rail"
      })
    },
    [clearChat, setSelectedAssistant, updateSettings]
  )

  return (
    <aside
      data-testid="character-control-rail"
      aria-label={t("playground:characterRail.title", "Character controls")}
      className="flex h-full min-h-0 w-full flex-col border-l border-border bg-surface"
    >
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-text">
              {t("playground:characterRail.title", "Character controls")}
            </h2>
            <p className="mt-1 text-xs text-text-muted">
              {t(
                "playground:characterRail.subtitle",
                "Overlay and tracked character/persona controls for this chat."
              )}
            </p>
          </div>
          <span className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text">
            {resolveModeLabel(effectiveAssistantState.mode)}
          </span>
        </div>
        {effectiveAssistantState.displayName && (
          <p className="mt-2 text-sm text-text">
            {effectiveAssistantState.displayName}
          </p>
        )}
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 py-4">
        <section className="space-y-2">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              {t("playground:characterRail.overlayHeading", "Overlay")}
            </h3>
            <p className="mt-1 text-xs text-text-muted">
              {isTrackedMode
                ? t(
                    "playground:characterRail.overlayUnavailableTracked",
                    "Overlay is unavailable while this chat is already tracked to a character or persona."
                  )
                : t(
                    "playground:characterRail.overlayBody",
                    "Use a character or persona as the assistant personality without changing conversation ownership."
                  )}
            </p>
          </div>
          {!isTrackedMode ? (
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleOverlaySelect}
                className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-1.5 text-xs font-medium text-text hover:bg-surface"
              >
                {overlayActionLabel}
              </button>
              {effectiveAssistantState.mode === "overlay" && (
                <button
                  type="button"
                  onClick={() => void handleClearOverlay()}
                  className="inline-flex items-center rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2"
                >
                  {t("playground:characterRail.clearOverlay", "Clear overlay")}
                </button>
              )}
            </div>
          ) : null}
        </section>

        <section className="space-y-2">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              {t("playground:characterRail.trackedHeading", "Tracked chats")}
            </h3>
            <p className="mt-1 text-xs text-text-muted">
              {t(
                "playground:characterRail.trackedBody",
                "Start or open chats that stay attached to a specific character or persona."
              )}
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={() => void handleStartTracked("character")}
              className="inline-flex items-center justify-center rounded-md border border-border bg-surface2 px-3 py-1.5 text-xs font-medium text-text hover:bg-surface"
            >
              {t(
                "playground:characterRail.startTrackedCharacter",
                "Start tracked character chat"
              )}
            </button>
            <button
              type="button"
              onClick={() => void handleStartTracked("persona")}
              className="inline-flex items-center justify-center rounded-md border border-border bg-surface2 px-3 py-1.5 text-xs font-medium text-text hover:bg-surface"
            >
              {t(
                "playground:characterRail.startTrackedPersona",
                "Start tracked persona chat"
              )}
            </button>
          </div>
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            {t("playground:characterRail.trackedSessions", "Tracked sessions")}
          </h3>
          {trackedSessions.length > 0 ? (
            <div className="flex flex-col gap-2">
              {trackedSessions.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => selectServerChat(item)}
                  aria-label={item.title || t("common:untitled", "Untitled")}
                  className="flex items-start justify-between rounded-md border border-border bg-surface px-3 py-2 text-left hover:bg-surface2"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-text">
                      {item.title || t("common:untitled", "Untitled")}
                    </span>
                    <span className="mt-1 block text-[11px] text-text-muted">
                      {item.assistant_kind === "persona"
                        ? t("playground:characterRail.personaSession", "Persona")
                        : t("playground:characterRail.characterSession", "Character")}
                    </span>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-text-muted">
              {t(
                "playground:characterRail.noTrackedSessions",
                "No tracked character or persona sessions yet."
              )}
            </p>
          )}
        </section>
      </div>
    </aside>
  )
}
