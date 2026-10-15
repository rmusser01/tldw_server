import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { Button } from "@/components/Common/Button"
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import {
  type ServerChatHistoryItem,
  useServerChatHistory
} from "@/hooks/useServerChatHistory"
import { useStoreMessageOption } from "@/store/option"
import { isProtectedTrackedPersonaUnavailable, type AssistantSelection } from "@/types/assistant-selection"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { openCharactersWorkspace } from "@/utils/characters-route"
import React from "react"
import { useTranslation } from "react-i18next"

type CharacterControlsSheetProps = {
  trackedSelectionLoading?: boolean
  onTrackedSelection?: (
    selection: AssistantSelection,
    isCurrent?: () => boolean
  ) => boolean | Promise<boolean>
  onRequestClose?: () => void
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
  trackedSelectionLoading = false,
  onTrackedSelection,
  onRequestClose
}: CharacterControlsSheetProps) => {
  const { t } = useTranslation(["playground", "common", "option"])
  const historySelection = useHistorySelectionContext()
  const personaGuidanceId = React.useId()
  const trackedPersonaUnavailable = isProtectedTrackedPersonaUnavailable(Boolean(historySelection), "persona", "tracked")
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
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)
  const [trackedPicker, setTrackedPicker] = React.useState<{
    tab: "character" | "persona"
    intent: number
  } | null>(null)
  const trackedPickerTab = trackedPicker?.tab ?? null
  const trackedPickerIntentRef = React.useRef(0)
  React.useLayoutEffect(
    () => () => {
      trackedPickerIntentRef.current++
    },
    []
  )
  const retireTrackedPicker = React.useCallback(() => {
    trackedPickerIntentRef.current++
    setTrackedPicker(null)
  }, [])
  const handleOverlayOpenChange = React.useCallback(
    (open: boolean) => {
      if (open) retireTrackedPicker()
    },
    [retireTrackedPicker]
  )
  const handleTrackedSelection = React.useCallback(
    (selection: AssistantSelection, pickerIsCurrent?: () => boolean) => {
      const intent = trackedPickerIntentRef.current
      return (
        onTrackedSelection?.(
          selection,
          () =>
            trackedPickerIntentRef.current === intent &&
            pickerIsCurrent?.() !== false
        ) ?? false
      )
    },
    [onTrackedSelection]
  )
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
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
    retireTrackedPicker()
    const intent = trackedPickerIntentRef.current
    const source = useStoreMessageOption.getState()
    const isCurrent = () => {
      const chat = useStoreMessageOption.getState()
      return (
        trackedPickerIntentRef.current === intent &&
        chat.chatDraftAddress === source.chatDraftAddress &&
        chat.chatDraftRevision === source.chatDraftRevision &&
        chat.historyId === source.historyId &&
        chat.serverChatId === source.serverChatId
      )
    }
    await updateSettings({
      assistantOverlay: null
    })
    if (!isCurrent()) return
    await setSelectedAssistant(null, { isCurrent })
  }, [retireTrackedPicker, setSelectedAssistant, updateSettings])

  const handleStartTracked = (tab: "character" | "persona") => {
    if (isProtectedTrackedPersonaUnavailable(Boolean(historySelection), tab, "tracked")) return
    trackedPickerIntentRef.current++
    setTrackedPicker({ tab, intent: trackedPickerIntentRef.current })
  }

  const handleTrackedSelectionComplete = () => {
    setTrackedPicker(null)
    onRequestClose?.()
  }

  const handleOpenExpressionEditor = React.useCallback(async () => {
    await openCharactersWorkspace({
      from: "sidepanel-character-controls",
      focus: "expressions",
      characterId: expressionCharacterId
    })
  }, [expressionCharacterId])

  const handleOpenTrackedSession = React.useCallback(
    (item: ServerChatHistoryItem) => {
      if (selectServerChat(item) === false) return
      retireTrackedPicker()
      onRequestClose?.()
    },
    [onRequestClose, retireTrackedPicker, selectServerChat]
  )

  return (
    <div
      data-testid="chat-character-controls-sheet"
      className="flex flex-col gap-4">
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
            onClick={() => void handleOpenExpressionEditor()}>
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
              onOpenChange={handleOverlayOpenChange}
              labelOverride={overlayActionLabel}
              iconClassName="h-4 w-4"
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
            />
            {effectiveAssistantState.mode === "overlay" ? (
              <Button
                variant="outline"
                onClick={() => void handleClearAssistantOverlay()}>
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
            onClick={() => void handleStartTracked("character")}>
            {t(
              "playground:characterRail.startTrackedCharacter",
              "Start tracked character chat"
            )}
          </Button>
          <Button
            variant="outline"
            disabled={trackedPersonaUnavailable}
            aria-describedby={trackedPersonaUnavailable ? personaGuidanceId : undefined}
            onClick={() => void handleStartTracked("persona")}>
            {t(
              "playground:characterRail.startTrackedPersona",
              "Start tracked persona chat"
            )}
          </Button>
        </div>
        {trackedPersonaUnavailable && (
          <p id={personaGuidanceId} className="text-sm text-text-muted">
            {t("option:assistant.trackedPersonaUnavailable", "Tracked persona chat is unavailable here. Use a persona prompt overlay in a regular chat, or choose a character for tracked chat.")}
          </p>
        )}
        {trackedPickerTab ? (
          <div data-testid="tracked-assistant-picker" className="pt-2">
            <AssistantSelect
              key={trackedPicker?.intent}
              variant="inline"
              selectionModePreference="tracked"
              initialTab={trackedPickerTab}
              onTrackedSelection={handleTrackedSelection}
              trackedSelectionLoading={trackedSelectionLoading}
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
                className="flex items-start justify-between rounded-md border border-border bg-surface px-3 py-2 text-left hover:bg-surface2">
                <span className="min-w-0">
                  <span className="block truncate text-sm text-text">
                    {item.title || t("common:untitled", "Untitled")}
                  </span>
                  <span className="mt-1 block text-[11px] text-text-muted">
                    {item.assistant_kind === "persona"
                      ? t("playground:characterRail.personaSession", "Persona")
                      : t(
                          "playground:characterRail.characterSession",
                          "Character"
                        )}
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
