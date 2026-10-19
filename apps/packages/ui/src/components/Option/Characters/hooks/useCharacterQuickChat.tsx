import React from "react"
import type { TFunction } from "i18next"
import { useNavigate } from "react-router-dom"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { useDraftPersistence, type DraftMetadataObject } from "@/hooks/useDraftPersistence"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { captureActorSceneSendForTarget, type ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { sendNativeHistoryCharacter } from "@/hooks/chat/native-history-character-send"
import { NativeHistoryRoleplaySendError, nativeCharacterPreflightMessage, resolveNativeCharacterConfiguration } from "@/hooks/chat/native-character-preflight"
import { classifyCharacterChatFailureRecovery } from "@/hooks/chat/useCharacterChatMode"
import { useStoreChatModelSettings } from "@/store/model"
import { loadHistoryTurnRecoveries } from "@/db/dexie/history-selection"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { listChatCreationRecovery, readChatCreationRecovery } from "@/services/pending-chat-promotion"
import { conversationDraftKey, freshChatDraftAddress, type ChatDraftHandle } from "@/utils/chat-draft-address"
import { requestSettingsNavigation } from "@/utils/settings-return"
import { buildCharacterChatReadiness, buildCharacterChatSendReadiness, getCharacterChatReadinessCopy, type ModelDescriptor } from "@/utils/chat-model-availability"
import { buildCharacterChatPath } from "@/routes/route-paths"
import { buildCharacterSelectionPayload } from "../utils"
import type { Message } from "@/store/option"

type QuickMessage = { id: string; role: "user" | "assistant"; content: string; preview?: boolean }
type QuickCard = Record<string, unknown>
type PopupRequest = { epoch: number; abort: AbortController }

const numericId = (record: QuickCard): number | null => {
  const id = Number(record.id ?? record.character_id ?? record.characterId)
  return Number.isSafeInteger(id) && id > 0 ? id : null
}
const displayMessages = (messages: Message[]): QuickMessage[] => messages.map((message, index) => ({
  id: message.id || `preview-${index}`, role: message.isBot ? "assistant" : "user", content: message.message
}))

export interface UseCharacterQuickChatDeps {
  t: TFunction
  activeQuickChatModel: string | null
  isServerConnected?: boolean
  availableModels?: ModelDescriptor[] | null
}

export function useCharacterQuickChat(deps: UseCharacterQuickChatDeps) {
  const { t, activeQuickChatModel, isServerConnected = true, availableModels } = deps
  const navigate = useNavigate()
  const [quickChatCharacter, setQuickChatCharacter] = React.useState<QuickCard | null>(null)
  const [quickChatMessages, setQuickChatMessages] = React.useState<QuickMessage[]>([])
  const [quickChatSessionId, setQuickChatSessionId] = React.useState<string | null>(null)
  const [quickChatLoading, setQuickChatLoading] = React.useState(false)
  const [quickChatCopied, setQuickChatCopied] = React.useState(false)
  const [quickChatSending, setQuickChatSending] = React.useState(false)
  const [quickChatError, setQuickChatError] = React.useState<string | null>(null)
  const [quickChatNeedsReview, setQuickChatNeedsReview] = React.useState(false)
  const [address, setAddress] = React.useState<ChatDraftHandle["address"] | null>(null)
  const form = useSimpleForm({ initialValues: { message: "" } })
  const openingIntent = React.useRef<object | null>(null)
  const epoch = React.useRef(0)
  const mounted = React.useRef(true)
  const cardRequest = React.useRef<AbortController | null>(null)
  const request = React.useRef<PopupRequest | null>(null)
  const verified = React.useRef(false)
  const needsReview = React.useRef(false)
  const sessionId = React.useRef<string | null>(null)
  const live = React.useRef({ activeQuickChatModel, isServerConnected })
  live.current = { activeQuickChatModel, isServerConnected }
  const connectionGeneration = React.useRef(0)
  const previousConnection = React.useRef(isServerConnected)
  React.useLayoutEffect(() => {
    if (previousConnection.current && !isServerConnected) connectionGeneration.current++
    previousConnection.current = isServerConnected
  }, [isServerConnected])
  const controller = useHistorySelection({ onCapture: capture => setQuickChatMessages(displayMessages(formatSelectedHistory(capture).messages)) })
  const resetHistory = controller.reset
  const resetForm = form.reset
  const clearPopup = React.useCallback(() => {
    epoch.current++
    openingIntent.current = null
    cardRequest.current?.abort()
    cardRequest.current = null
    request.current?.abort.abort()
    request.current = null
    verified.current = false
    sessionId.current = null
    needsReview.current = false
    resetHistory()
    setQuickChatCharacter(null)
    setQuickChatMessages([])
    setQuickChatSessionId(null)
    setQuickChatSending(false)
    setQuickChatLoading(false)
    setQuickChatCopied(false)
    setQuickChatError(null)
    setQuickChatNeedsReview(false)
    setAddress(null)
    resetForm()
  }, [resetHistory, resetForm])
  const owner = useChatDraftOwner(clearPopup)
  React.useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; request.current?.abort.abort(); cardRequest.current?.abort() }
  }, [])
  const popupEpoch = epoch.current
  const ownerIsCurrent = owner.isCurrent
  const current = React.useCallback(() => mounted.current && epoch.current === popupEpoch && ownerIsCurrent(), [popupEpoch, ownerIsCurrent])
  const handle: ChatDraftHandle | null = address && owner.ownerKey ? { address, ownerHash: owner.ownerKey } : null
  const storageKey = address ? `${conversationDraftKey("tldw:playgroundChatDraft", address)}:owner:${owner.ownerKey ?? "unresolved"}` : "tldw:quickChat:unresolved"
  const metadata = () => ({ chatDraftHandle: handle as unknown as DraftMetadataObject })
  const persistence = useDraftPersistence({ storageKey, isCurrent: current,
    getValue: () => form.getValues().message, setValue: value => form.setValues({ message: value }),
    getMetadata: metadata, enabled: Boolean(quickChatCharacter && handle && owner.isCurrent()),
    hasUnpersistedInput: Boolean(form.values.message) })
  const retain = async (isCurrent = current) => {
    const revision = form.getRevision()
    const raw = form.getValues().message
    if (!handle || !persistence.draftReady || !isCurrent()) return false
    const saved = await persistence.retainDraft(storageKey, { content: raw, metadata: metadata() })
    return saved && isCurrent() && form.getRevision() === revision
  }
  const markReview = () => { needsReview.current = true; setQuickChatNeedsReview(true) }
  const settingsState = useStoreChatModelSettings()
  const modelSelection = resolveModelSelectorSelection(availableModels, activeQuickChatModel, settingsState.apiProvider,
    { scope: settingsState.activeSettingsScope, providerOverride: settingsState.apiProvider })
  const settings = settingsState.getEffectiveSettings(modelSelection.settingsScope)
  const readBoundPrivateSource = (id: string | null) => {
    const selected = controller.getCurrent()
    return id && selected.status === "ready" && selected.owner?.kind === "native" &&
      selected.owner.scope?.type !== "workspace" && selected.owner.conversation_id === id &&
      selected.view?.conversation_id === id && selected.capture?.status === "captured" && selected.bookmarkScope &&
      selected.owner.validate_lease() && owner.isCurrent()
      ? { owner: selected.owner, view: selected.view } : null
  }
  const popupSendBlocked = !verified.current || !owner.isCurrent() || !persistence.draftReady || quickChatSending || quickChatNeedsReview
  const quickChatReadiness = quickChatSessionId
    ? buildCharacterChatSendReadiness({ isServerConnected, isSendBlocked: popupSendBlocked || !readBoundPrivateSource(quickChatSessionId) })
    : buildCharacterChatReadiness({ isServerConnected, selectedCharacter: quickChatCharacter,
      selectedModel: activeQuickChatModel, explicitProvider: settings.apiProvider, requireExplicitProvider: true, availableModels,
      isSendBlocked: popupSendBlocked })
  const reviewOutcome = async (isCurrent: () => boolean) => {
    if (!handle || !isCurrent()) return
    try {
      const attempt = await readChatCreationRecovery(handle)
      const selected = controller.getCurrent()
      const turns = selected.bookmarkScope && selected.view ? await loadHistoryTurnRecoveries(selected.bookmarkScope, selected.view) : []
      if (!isCurrent()) return
      const knownOwner = selected.status === "ready" && selected.capture?.status === "captured" &&
        selected.view?.conversation_id === attempt?.chatId && selected.owner?.kind === "native" &&
        selected.owner.conversation_id === attempt?.chatId && selected.owner.validate_lease()
      if (turns.some(entry => entry.turn.state !== "refused") || (attempt && attempt.state !== "prepared" && attempt.state !== "settled" && !knownOwner)) markReview()
    } catch { if (isCurrent()) markReview() }
  }

  const closeQuickChat = async (_options?: { preserveSession?: boolean }) => {
    if (form.getValues().message && !(await retain())) {
      if (current()) setQuickChatError(t("settings:manageCharacters.quickChat.retainFailed", { defaultValue: "Your draft could not be kept. Copy it before closing this popup." }))
      return
    }
    if (mounted.current && epoch.current === popupEpoch) clearPopup()
  }

  const loadQuickCard = async (id: number, isCurrent: () => boolean) => {
    if (!owner.requestScope || !isCurrent()) return
    cardRequest.current?.abort()
    const abort = new AbortController()
    cardRequest.current = abort
    const loadingCurrent = () => isCurrent() && cardRequest.current === abort && !abort.signal.aborted
    setQuickChatLoading(true)
    setQuickChatError(null)
    try {
      const card = await tldwClient.getCharacter(String(id), { requestScope: owner.requestScope, signal: abort.signal })
      if (!loadingCurrent()) return
      if (String(card?.id) !== String(id)) throw new Error("owner_conversation_mismatch")
      verified.current = true
      setQuickChatCharacter(card)
      const greeting = buildCharacterSelectionPayload(card).greeting
      setQuickChatMessages(greeting?.trim() ? [{ id: "character-preview", role: "assistant", content: greeting, preview: true }] : [])
    } catch {
      if (loadingCurrent()) setQuickChatError(t("settings:manageCharacters.quickChat.cardLoadFailed", { defaultValue: "Could not load this character for your account. Your draft is kept. Retry when the server is available." }))
    } finally {
      if (loadingCurrent()) { cardRequest.current = null; setQuickChatLoading(false) }
    }
  }
  const retryQuickChat = async () => {
    if (!owner.isCurrent()) { owner.retryVerification(); return }
    const id = quickChatCharacter && numericId(quickChatCharacter)
    if (id && !verified.current && !request.current) await loadQuickCard(id, current)
  }
  const retryCard = React.useRef(retryQuickChat)
  retryCard.current = retryQuickChat
  React.useEffect(() => { if (owner.ownerKey) void retryCard.current() }, [owner.ownerKey])

  const openQuickChat = async (record: QuickCard) => {
    const intent = {}
    openingIntent.current = intent
    const replacing = () => mounted.current && epoch.current === popupEpoch && openingIntent.current === intent
    if (quickChatCharacter && form.getValues().message && !(await retain(replacing))) {
      if (replacing()) setQuickChatError(t("settings:manageCharacters.quickChat.retainFailed", { defaultValue: "Your draft could not be kept. Copy it before closing this popup." }))
      return
    }
    if (!replacing()) return
    clearPopup()
    openingIntent.current = intent
    const openingEpoch = epoch.current
    const openingCurrent = () => mounted.current && epoch.current === openingEpoch && openingIntent.current === intent && owner.isCurrent()
    const id = numericId(record)
    setAddress(freshChatDraftAddress(false))
    // The catalog may be stale after an account change; display only the scoped card.
    setQuickChatCharacter({ id })
    if (!id) {
      setQuickChatError(t("settings:manageCharacters.quickChat.unsupportedCharacter", { defaultValue: "Quick chat is only available for server-synced characters." }))
      return
    }
    if (!owner.isCurrent()) return
    await loadQuickCard(id, openingCurrent)
  }

  const copyQuickChatDraft = async () => {
    const raw = form.getValues().message
    if (!raw || !current() || request.current) return
    const operation = { epoch: epoch.current, abort: new AbortController() }
    const revision = form.getRevision()
    request.current = operation
    setQuickChatSending(true)
    setQuickChatCopied(false)
    try {
      await navigator.clipboard.writeText(raw)
      if (current() && request.current === operation && form.getRevision() === revision) setQuickChatCopied(true)
    } catch {
      if (current() && request.current === operation) setQuickChatError(t("settings:manageCharacters.quickChat.copyFailed", { defaultValue: "Could not copy your draft. Select the text and copy it manually; your popup stays open." }))
    } finally {
      if (current() && request.current === operation) { request.current = null; setQuickChatSending(false) }
    }
  }

  const sendQuickChatMessage = async () => {
    const raw = form.getValues().message
    if (!raw.trim() || !quickChatCharacter || !handle || !verified.current || request.current || needsReview.current) return
    if (!quickChatReadiness.canStart) {
      setQuickChatError(getCharacterChatReadinessCopy(quickChatReadiness, t, { characterName: buildCharacterSelectionPayload(quickChatCharacter).name }).title)
      return
    }
    const operation = { epoch: epoch.current, abort: new AbortController() }
    request.current = operation
    const generation = connectionGeneration.current
    const revision = form.getRevision()
    const popupCurrent = current
    const boundSource = readBoundPrivateSource(sessionId.current)
    const preparing = () => popupCurrent() && request.current === operation && !operation.abort.signal.aborted &&
      form.getRevision() === revision && live.current.isServerConnected && connectionGeneration.current === generation &&
      (boundSource ? controller.getCurrent().owner === boundSource.owner && controller.getCurrent().view === boundSource.view && boundSource.owner.validate_lease()
        : live.current.activeQuickChatModel === activeQuickChatModel && useStoreChatModelSettings.getState() === settingsState)
    setQuickChatSending(true)
    setQuickChatError(null)
    try {
      let actorSceneSend: ActorSceneSend | undefined
      if (!boundSource) {
        if (sessionId.current) throw new Error("roleplay_resume_unavailable")
        const selected = controller.getCurrent()
        const sceneAddress = selected.status === "idle" && !selected.owner && !selected.view ? handle.address : null
        const historyIsCurrent = controller.fence()
        const target = captureActorSettingsTarget({ address: sceneAddress,
          requestScope: owner.requestScope, accountIsCurrent: owner.isCurrent, isCurrent: () => preparing() && historyIsCurrent() })
        const capture = captureActorSceneSendForTarget(target, target?.account, () => preparing() && historyIsCurrent())
        if (!capture) throw new Error("actor_settings_unavailable")
        actorSceneSend = await capture.read()
        if (!preparing()) return
        await resolveNativeCharacterConfiguration({ temporary: false, settings, model: activeQuickChatModel!, toolChoice: "none", clientToolCount: 0,
          actorEnabled: actorSceneSend.scene.settings.isEnabled }, preparing)
      }
      if (!(await retain(preparing))) throw new Error("native_history_draft_not_retained")
      const outstanding = await listChatCreationRecovery(handle.ownerHash, { type: "global" })
      if (!preparing()) return
      if (outstanding.unavailable) throw new Error("chat_creation_recovery_unavailable")
      if (!sessionId.current && outstanding.attempts.some(attempt => attempt.state !== "prepared" && String(attempt.characterId) === String(numericId(quickChatCharacter)) && attempt.handle.address.id !== handle.address.id)) {
        markReview()
        throw new Error(t("settings:manageCharacters.quickChat.previousAttempt", { defaultValue: "An earlier attempt may have saved this character chat. Review retained drafts in full chat before starting another conversation." }))
      }
      if (!boundSource && !actorSceneSend?.isCurrent()) return
      await sendNativeHistoryCharacter({ controller, actorSceneSend, privateDraft: { handle, requestScope: owner.requestScope!, isCurrent: popupCurrent }, surface: "playground",
        originIsCurrent: preparing, signal: operation.abort.signal, temporary: false,
        serverChatId: sessionId.current, scope: { type: "global" }, characterId: numericId(quickChatCharacter),
        model: activeQuickChatModel!, settings, toolChoice: "none", message: raw, image: "", t,
        setServerChatId: id => { if (popupCurrent() && request.current === operation) { sessionId.current = id; setQuickChatSessionId(id) } },
        setMessages: next => { if (popupCurrent() && request.current === operation) setQuickChatMessages(previous =>
          displayMessages(typeof next === "function" ? next(previous.map(row => ({ id: row.id, isBot: row.role === "assistant", message: row.content, name: row.role, sources: [], createdAt: 0 }))) : next)) },
        releaseActivity: () => {} })
      if (popupCurrent() && request.current === operation && form.getRevision() === revision) {
        form.setValues({ message: "" })
        persistence.clearDraft()
      }
    } catch (error) {
      const failureIsCurrent = () => popupCurrent() && request.current === operation &&
        !(error instanceof NativeHistoryRoleplaySendError && !error.isCurrent())
      if (!failureIsCurrent()) return
      await reviewOutcome(failureIsCurrent)
      if (!failureIsCurrent()) return
      const preflightMessage = nativeCharacterPreflightMessage(error, t)
      const recovery = !preflightMessage && error instanceof NativeHistoryRoleplaySendError ? classifyCharacterChatFailureRecovery(error, t) : null
      setQuickChatError(preflightMessage || recovery?.message || (error instanceof Error ? error.message : String(error)))
    } finally {
      if (popupCurrent() && request.current === operation) { request.current = null; setQuickChatSending(false) }
    }
  }

  const handlePromoteQuickChat = async () => {
    if (!quickChatCharacter || request.current || !handle || !current()) return
    const revision = form.getRevision()
    const raw = form.getValues().message
    const operation = { epoch: epoch.current, abort: new AbortController() }
    request.current = operation
    const id = sessionId.current
    const selection = controller.getCurrent()
    const fence = controller.fence()
    const leavingCurrent = () => current() && request.current === operation && !operation.abort.signal.aborted && form.getRevision() === revision
    setQuickChatSending(true)
    setQuickChatError(null)
    try {
      if (!(await retain(leavingCurrent))) throw new Error("native_history_draft_not_retained")
      if (raw) { await navigator.clipboard.writeText(raw); if (!leavingCurrent()) return }
      let path: string
      if (id) {
        if (selection.owner?.kind !== "native" || selection.owner.conversation_id !== id || !selection.owner.validate_lease())
          throw new Error("history_selection_not_ready")
        const chat = await tldwClient.getChat(id, { requestScope: selection.owner.request_scope, scope: selection.owner.scope })
        if (!leavingCurrent() || !fence()) return
        if (chat.id !== id || String(chat.character_id) !== String(numericId(quickChatCharacter))) throw new Error("owner_conversation_mismatch")
        const savedPath = await controller.prepareExpansionPath()
        if (!leavingCurrent() || !fence()) return
        if (!savedPath || savedPath === "/chat") throw new Error("history_selection_not_ready")
        path = savedPath
      } else path = needsReview.current ? "/chat" : buildCharacterChatPath({ characterId: numericId(quickChatCharacter) })
      if (!leavingCurrent() || !requestSettingsNavigation(path) || !leavingCurrent()) return
      navigate(path)
      clearPopup()
    } catch {
      if (leavingCurrent()) setQuickChatError(t("settings:manageCharacters.quickChat.continueFailed", { defaultValue: "Could not continue. Your popup and draft are still here. Check draft storage, clipboard access and the saved conversation, then try again." }))
    } finally {
      if (current() && request.current === operation) { request.current = null; setQuickChatSending(false) }
    }
  }

  return { quickChatCharacter, quickChatMessages, quickChatDraft: form.values.message,
    setQuickChatDraft: (message: string) => form.setValues({ message }), quickChatSessionId,
    quickChatSending, quickChatError, quickChatNeedsReview, quickChatReadiness, quickChatLoading, quickChatCopied,
    quickChatOwnerReady: owner.isCurrent(), quickChatCardReady: verified.current, quickChatCanRetry: Boolean(quickChatCharacter && !verified.current && numericId(quickChatCharacter)),
    openQuickChat, closeQuickChat, retryQuickChat, copyQuickChatDraft, sendQuickChatMessage, handlePromoteQuickChat }
}
