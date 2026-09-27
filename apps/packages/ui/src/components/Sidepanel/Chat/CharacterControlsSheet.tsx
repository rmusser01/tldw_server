import React from "react"
import { useTranslation } from "react-i18next"
import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { Button } from "@/components/Common/Button"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { useServerChatHistory, type ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useStoreMessageOption } from "@/store/option"
import type { AssistantSelection } from "@/types/assistant-selection"
import { openCharactersWorkspace } from "@/utils/characters-route"

type CharacterControlsSheetProps = {
  beforeTrackedStart?: (() => Promise<void>) | (() => void)
  onRequestClose?: () => void
  resetChat?: () => void
}

const isTrackedSession = (item: ServerChatHistoryItem): boolean =>
  item.assistant_kind === "character" ||
  item.assistant_kind === "persona" ||
  item.character_id != null

const resolveAssistantModeLabel = (
  mode: ReturnType<typeof resolveEffectiveAssistantState>["mode"]
) => {
  if (mode === "overlay") return "Overlay personality"
  if (mode === "tracked_character") return "Tracked character chat"
  if (mode === "tracked_persona") return "Tracked persona chat"
  return "Plain chat"
}

export const CharacterControlsSheet = ({
  beforeTrackedStart,
  onRequestClose,
  resetChat
}: CharacterControlsSheetProps) => {
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
  const setServerChatAssistantKind = useStoreMessageOption(
    (state) => state.setServerChatAssistantKind
  )
  const setServerChatAssistantId = useStoreMessageOption(
    (state) => state.setServerChatAssistantId
  )
  const setServerChatCharacterId = useStoreMessageOption(
    (state) => state.setServerChatCharacterId
  )
  const setServerChatPersonaMemoryMode = useStoreMessageOption(
    (state) => state.setServerChatPersonaMemoryMode
  )
  const setServerChatMetaLoaded = useStoreMessageOption(
    (state) => state.setServerChatMetaLoaded
  )
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)
  const [trackedPickerTab, setTrackedPickerTab] = React.useState<
    "character" | "persona" | null
  >(null)
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const fallbackClearChat = useClearChat()
  const clearChat = resetChat ?? fallbackClearChat
  const selectServerChat = useSelectServerChat()
  const { data: serverChatHistory = [] } = useServerChatHistory("", {
    filterMode: "all"
  })

  const effectiveAssistantState = React.useMemo(
    () =>
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: serverChatAssistantKind,
          assistantId: serverChatAssistantId,
          characterId: serverChatCharacterId
        },
        settings: settings ?? null,
        draftSelection: selectedAssistant
      }),
    [
      selectedAssistant,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatCharacterId,
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
  const expressionCharacterId =
    effectiveAssistantState.kind === "character"
      ? effectiveAssistantState.id
      : null

  const handleClearAssistantOverlay = React.useCallback(async () => {
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
      await beforeTrackedStart?.()
      setTrackedPickerTab(tab)
    },
    [beforeTrackedStart, setSelectedAssistant, updateSettings]
  )

  const handleTrackedSelectionComplete = React.useCallback((selection: AssistantSelection) => {
    setTrackedPickerTab(null)
    clearChat()
    const personaMemoryMode = selection.metadata?.personaMemoryMode
    setServerChatAssistantKind(selection.kind)
    setServerChatAssistantId(selection.id)
    setServerChatCharacterId(
      selection.kind === "character" ? selection.id : null
    )
    setServerChatPersonaMemoryMode(
      selection.kind === "persona" &&
        (personaMemoryMode === "read_only" || personaMemoryMode === "read_write")
        ? personaMemoryMode
        : null
    )
    // clearChat resets server metadata. Restore the draft tracked identity
    // afterwards so the first send creates the matching server session.
    setServerChatMetaLoaded(false)
    onRequestClose?.()
  }, [
    clearChat,
    onRequestClose,
    setServerChatAssistantId,
    setServerChatAssistantKind,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    setServerChatPersonaMemoryMode
  ])

  const handleOpenExpressionEditor = React.useCallback(async () => {
    await openCharactersWorkspace({
      from: "sidepanel-character-controls",
      focus: "expressions",
      characterId: expressionCharacterId
    })
  }, [expressionCharacterId])

  const handleOpenTrackedSession = React.useCallback(
    (item: ServerChatHistoryItem) => {
      onRequestClose?.()
      selectServerChat(item)
    },
    [onRequestClose, selectServerChat]
  )

  return (
    <div
      data-testid="chat-character-controls-sheet"
      className="flex flex-col gap-4"
    >
      <div className="space-y-2">
        <div className="space-y-1">
          <p className="text-sm text-text">
            {resolveAssistantModeLabel(effectiveAssistantState.mode)}
          </p>
          {effectiveAssistantState.displayName ? (
            <p className="text-xs text-text-subtle">
              {effectiveAssistantState.displayName}
            </p>
          ) : null}
        </div>
        {expressionCharacterId != null ? (
          <Button
            variant="outline"
            onClick={() => void handleOpenExpressionEditor()}
          >
            {t(
              "playground:characterRail.editExpressions",
              "Edit character expressions"
            )}
          </Button>
        ) : null}
      </div>

      <section className="space-y-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-text-subtle">
            {t("playground:characterRail.overlayHeading", "Overlay")}
          </p>
          <p className="text-sm text-text-subtle">
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
          <div className="flex flex-wrap items-center gap-2">
            <AssistantSelect
              variant="dropdown"
              showLabel
              selectionModePreference="overlay"
              labelOverride={overlayActionLabel}
              iconClassName="h-4 w-4"
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
            />
            {effectiveAssistantState.mode === "overlay" ? (
              <Button
                variant="outline"
                onClick={() => void handleClearAssistantOverlay()}
              >
                {t("playground:characterRail.clearOverlay", "Clear overlay")}
              </Button>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="space-y-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-text-subtle">
            {t("playground:characterRail.trackedHeading", "Tracked chats")}
          </p>
          <p className="text-sm text-text-subtle">
            {t(
              "playground:characterRail.trackedBody",
              "Start or open chats that stay attached to a specific character or persona."
            )}
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <Button
            variant="outline"
            onClick={() => void handleStartTracked("character")}
          >
            {t(
              "playground:characterRail.startTrackedCharacter",
              "Start tracked character chat"
            )}
          </Button>
          <Button
            variant="outline"
            onClick={() => void handleStartTracked("persona")}
          >
            {t(
              "playground:characterRail.startTrackedPersona",
              "Start tracked persona chat"
            )}
          </Button>
        </div>
        {trackedPickerTab ? (
          <div data-testid="tracked-assistant-picker" className="pt-2">
            <AssistantSelect
              variant="inline"
              selectionModePreference="tracked"
              initialTab={trackedPickerTab}
              onSelectionComplete={handleTrackedSelectionComplete}
            />
          </div>
        ) : null}
      </section>

      <section className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-text-subtle">
          {t("playground:characterRail.trackedSessions", "Tracked sessions")}
        </p>
        {trackedSessions.length > 0 ? (
          <div className="flex flex-col gap-2">
            {trackedSessions.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => handleOpenTrackedSession(item)}
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
  )
}
