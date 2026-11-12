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
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { useConnectionState } from "@/hooks/useConnectionState"
import { formatRelativeTime } from "@/utils/dateFormatters"
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
  (!item.scope_type || item.scope_type === "global") &&
  !item.workspace_id &&
  item.character_id != null

const SESSION_PAGE_SIZE = 6

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
  const { isConnected } = useConnectionState()
  const verified = useChatDraftOwner(() => {})
  const owner = React.useMemo(() => verified.ownerKey && verified.requestScope ? {
    key: verified.ownerKey,
    requestScope: verified.requestScope,
    isCurrent: verified.isCurrent
  } : undefined, [verified.ownerKey, verified.requestScope, verified.isCurrent])
  const [historyPage, setHistoryPage] = React.useState(1)
  React.useEffect(() => setHistoryPage(1), [verified.ownerKey])
  const history = useServerChatHistory("", {
    enabled: isConnected && !!owner?.isCurrent(),
    owner,
    scope: { type: "global" },
    mode: "overview",
    filterMode: "character",
    page: historyPage,
    limit: SESSION_PAGE_SIZE
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
    () => owner?.isCurrent() ? history.data.filter(isTrackedSession) : [],
    [history.data, owner]
  )
  const historyFailed = history.sidebarRefreshState === "recoverable-error" || history.sidebarRefreshState === "hard-error"
  const unverifiedSessions = history.data.length > trackedSessions.length
  const hasNextHistoryPage = history.total > historyPage * SESSION_PAGE_SIZE

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
      if (!isConnected || !owner?.isCurrent()) return
      if (selectServerChat(item) === false) return
      retireTrackedPicker()
      onRequestClose?.()
    },
    [isConnected, onRequestClose, owner, retireTrackedPicker, selectServerChat]
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
          {t("playground:characterRail.characterSessions", "Character sessions")}
        </p>
        {!isConnected ? (
          <p role="status" className="text-xs text-text-muted">{t("playground:characterRail.sessionsOffline", "Connect to your server to view character sessions.")}</p>
        ) : !owner?.isCurrent() ? (
          <div role="status" className="space-y-2 text-xs text-text-muted">
            <p>{verified.verificationState === "pending"
              ? t("playground:characterRail.sessionsVerifying", "Verifying your server account...")
              : t("playground:characterRail.sessionsAccountError", "Your server account could not be verified.")}</p>
            {verified.verificationState === "error" ? <Button variant="outline" onClick={verified.retryVerification}>{t("playground:characterRail.retryVerification", "Retry account verification")}</Button> : null}
          </div>
        ) : history.isLoading || (history.sidebarRefreshState === "idle" && trackedSessions.length === 0) ? (
          <p role="status" className="text-xs text-text-muted">{t("playground:characterRail.sessionsLoading", "Loading character sessions...")}</p>
        ) : historyFailed || history.isShowingStaleData ? (
          <div className="space-y-2">
            <p role={trackedSessions.length ? "status" : "alert"} className="text-xs text-text-muted">
              {trackedSessions.length
                ? t("playground:characterRail.sessionsStale", "Showing character sessions from the last successful refresh.")
                : t("playground:characterRail.sessionsLoadFailed", "Character sessions could not be loaded.")}
            </p>
            <Button variant="outline" disabled={history.isFetching} onClick={() => void history.refetch()}>{t("playground:characterRail.retrySessions", "Retry character sessions")}</Button>
          </div>
        ) : null}
        {isConnected && owner?.isCurrent() && unverifiedSessions ? <div role="alert" className="space-y-2 text-xs text-text-muted">
          <p>{t("playground:characterRail.sessionsUnverified", "Some sessions on this page could not be verified for character chat. Retry to refresh.")}</p>
          <Button variant="outline" disabled={history.isFetching} onClick={() => void history.refetch()}>{t("playground:characterRail.retrySessions", "Retry character sessions")}</Button>
        </div> : null}
        {trackedSessions.length > 0 ? (
          <div className="flex flex-col gap-2">
            {trackedSessions.map((item) => {
              const title = item.title?.trim() || t("common:untitled", "Untitled")
              const duplicate = trackedSessions.some((other) => other.id !== item.id && (other.title?.trim() || t("common:untitled", "Untitled")) === title)
              const current = serverChatId === item.id
              const updatedAt = item.updated_at || item.last_active || item.created_at
              const characterName = item.character_name?.trim() || item.assistant_name?.trim()
              return (
              <button
                key={item.id}
                type="button"
                disabled={current || !isConnected || !owner?.isCurrent()}
                onClick={() => handleOpenTrackedSession(item)}
                aria-label={current
                  ? duplicate
                    ? t("playground:characterRail.currentSessionReferenceAction", "Current {{title}}, session {{id}}", { title, id: item.id })
                    : t("playground:characterRail.currentSessionAction", "Current {{title}}", { title })
                  : duplicate ? t("playground:characterRail.resumeSessionAction", "Resume {{title}}, session {{id}}", { title, id: item.id }) : title}
                className="flex items-start justify-between rounded-md border border-border bg-surface px-3 py-2 text-left hover:bg-surface2">
                <span className="min-w-0">
                  <span className="block truncate text-sm text-text">
                    {title}
                  </span>
                  {duplicate ? <span className="block break-all text-[11px] text-text-muted">{t("playground:characterRail.sessionReference", "Session {{id}}", { id: item.id })}</span> : null}
                  <span className="mt-1 block text-[11px] text-text-muted">
                    {characterName || (item.assistant_kind === "persona"
                      ? t("playground:characterRail.personaSession", "Persona")
                      : t(
                          "playground:characterRail.characterSession",
                          "Character"
                        ))}
                  </span>
                  {updatedAt ? <time dateTime={updatedAt} title={updatedAt} className="block text-[11px] text-text-muted">{formatRelativeTime(updatedAt, t, { compact: true })}</time> : null}
                </span>
                {current ? <span className="text-xs text-text-muted">{t("playground:characterRail.currentSession", "Current")}</span> : null}
              </button>
            )})}
          </div>
        ) : isConnected && owner?.isCurrent() && !history.isLoading && !historyFailed && history.sidebarRefreshState === "ready" && history.data.length === 0 ? (
          <p className="text-xs text-text-muted">
            {historyPage > 1 ? t("playground:characterRail.pageEmpty", "No sessions on this page. Return to the previous page.") : t(
              "playground:characterRail.noCharacterSessions",
              "No character sessions yet."
            )}
          </p>
        ) : null}
        {isConnected && owner?.isCurrent() && (historyPage > 1 || hasNextHistoryPage) ? <nav aria-label={t("playground:characterRail.pagesLabel", "Character sessions pages")} className="flex items-center gap-2">
          <Button variant="outline" disabled={historyPage === 1} aria-label={t("playground:characterRail.previousPage", "Previous sessions page")} onClick={() => setHistoryPage(historyPage - 1)}>{t("common:previous", "Previous")}</Button>
          <span role="status" className="text-xs text-text-muted">{t("playground:characterRail.pageLabel", "Page")} {historyPage}</span>
          <Button variant="outline" disabled={history.isFetching || historyFailed || !hasNextHistoryPage} aria-label={t("playground:characterRail.nextPage", "Next sessions page")} onClick={() => setHistoryPage(historyPage + 1)}>{t("common:next", "Next")}</Button>
        </nav> : null}
      </section>
    </div>
  )
}
