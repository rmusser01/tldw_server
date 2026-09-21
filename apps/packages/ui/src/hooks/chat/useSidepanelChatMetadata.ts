import React from "react"
import { useStoreMessageOption } from "@/store/option"
import { useSidepanelChatTabsStore } from "@/store/sidepanel-chat-tabs"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { watchServerChatLoadAuthority } from "@/services/server-chat-load-authority"
import { getSelectedAssistantOperationRevision, type SelectedAssistantCommitOptions } from "@/hooks/useSelectedAssistant"
import { resolveServerChatAssistantIdentity } from "@/hooks/chat/useServerChatLoader"
import { characterToAssistantSelection, personaToAssistantSelection, type AssistantSelection } from "@/types/assistant-selection"
import { normalizeConversationState } from "@/utils/conversation-state"

const METADATA_FAILURE = "Conversation details could not be loaded. Retry before sending queued requests."
const METADATA_WAITING = "Waiting for conversation details before sending queued requests."

export type QueuedDispatchGuard = (() => void) & {
  publishServerChatId: (id: string | null, publish: (id: string | null) => void) => void
  publishHistoryId: (id: string | null, publish: (id: string | null) => void) => void
}

/** Restore the saved actor before a sidepanel queue can choose its transport. */
export const useSidepanelChatMetadata = (
  setSelectedAssistant: (selection: AssistantSelection | null, options?: SelectedAssistantCommitOptions) => Promise<unknown> | void
) => {
  const { serverChatId, serverChatMetaLoaded, serverChatLoadState } = useStoreMessageOption()
  const selectionSetter = React.useRef(setSelectedAssistant)
  selectionSetter.current = setSelectedAssistant
  const [retryRevision, retry] = React.useReducer((value: number) => value + 1, 0)
  const lifetime = React.useRef({ active: true, revision: 0, completionRevision: 0 })
  const ownPublication = React.useRef<{ field: "serverChatId" | "historyId"; id: string | null } | null>(null)

  React.useLayoutEffect(() => {
    const ownerLifetime = lifetime.current
    ownerLifetime.active = true
    const invalidate = () => { ownerLifetime.revision++; ownerLifetime.completionRevision++ }
    const stopAccount = watchChatAccountChanges((changed) => { if (changed) invalidate() })
    const stopConversation = useStoreMessageOption.subscribe((state, previous) => {
      if (state.serverChatId !== previous.serverChatId || state.historyId !== previous.historyId ||
        (previous.serverChatMetaLoaded && !state.serverChatMetaLoaded)) ownerLifetime.revision++
      const publication = ownPublication.current
      const serverChanged = state.serverChatId !== previous.serverChatId
      const historyChanged = state.historyId !== previous.historyId
      // Only a guarded synchronous publication belongs to this dispatch.
      // Another view can also replace null IDs while its request is pending.
      const ownServer = publication?.field === "serverChatId" && publication.id === state.serverChatId && !historyChanged
      const ownHistory = publication?.field === "historyId" && publication.id === state.historyId && !serverChanged
      if ((serverChanged || historyChanged) && !ownServer && !ownHistory) ownerLifetime.completionRevision++
    })
    const stopTab = useSidepanelChatTabsStore.subscribe((state, previous) => {
      if (state.activeTabId !== previous.activeTabId) invalidate()
    })
    return () => {
      ownerLifetime.active = false
      invalidate()
      stopAccount()
      stopConversation()
      stopTab()
    }
  }, [])

  React.useEffect(() => {
    if (!serverChatId || serverChatMetaLoaded) return
    const controller = new AbortController()
    const revision = lifetime.current.revision
    const current = () => lifetime.current.active && lifetime.current.revision === revision &&
      !controller.signal.aborted && useStoreMessageOption.getState().serverChatId === serverChatId
    const load = async () => {
      let snapshot: ServicePromptSnapshot | undefined
      let stopAuthority: (() => void) | undefined
      try {
        const state = useStoreMessageOption.getState()
        state.setServerChatLoadState("loading")
        state.setServerChatLoadError(null)
        snapshot = await loadServicePromptSnapshot([], { signal: controller.signal })
        const abort = () => controller.abort()
        snapshot.scopeInvalidatedSignal.addEventListener("abort", abort, { once: true })
        if (snapshot.scopeSignal.aborted || snapshot.scopeInvalidatedSignal.aborted || !current()) return
        stopAuthority = watchServerChatLoadAuthority(snapshot, controller)
        const options = { signal: snapshot.scopeSignal, requestScope: snapshot.requestScope }
        const chat = await tldwClient.getChat(serverChatId, options)
        if (!current()) return
        const identity = resolveServerChatAssistantIdentity(chat as unknown as Record<string, unknown>)
        const { assistantKind, assistantId, characterId, personaMemoryMode } = identity
        const selectionRevision = getSelectedAssistantOperationRevision()
        let selection: AssistantSelection | undefined
        if (assistantKind === "persona" && assistantId) {
          const profile = await tldwClient.getPersonaProfile(assistantId, options).catch(() => null)
          selection = personaToAssistantSelection(profile ?? { id: assistantId, name: chat.assistant_name ?? chat.title ?? "Persona" })
        } else if (assistantKind === "character" && assistantId) {
          const character = await tldwClient.getCharacter(assistantId, options).catch(() => null)
          selection = characterToAssistantSelection(character ?? { id: assistantId, name: chat.assistant_name ?? chat.title ?? "Assistant" })
        }
        if (!current()) return
        if (selection && selectionRevision === getSelectedAssistantOperationRevision()) {
          await selectionSetter.current(selection, { isCurrent: current })
          if (!current()) return
        }
        state.setServerChatTitle(String(chat.title || ""))
        state.setServerChatCharacterId(characterId)
        state.setServerChatAssistantKind(assistantKind)
        state.setServerChatAssistantId(assistantId)
        state.setServerChatPersonaMemoryMode(personaMemoryMode)
        state.setServerChatState(normalizeConversationState(
          chat.state ?? (chat as typeof chat & { conversation_state?: string }).conversation_state
        ))
        state.setServerChatVersion(chat.version ?? null)
        state.setServerChatTopic(chat.topic_label ?? null)
        state.setServerChatClusterId(chat.cluster_id ?? null)
        state.setServerChatSource(chat.source ?? null)
        state.setServerChatExternalRef(chat.external_ref ?? null)
        state.setServerChatLoadError(null)
        state.setServerChatLoadState("loaded")
        state.setServerChatMetaLoaded(true)
      } catch {
        if (!current() || useStoreMessageOption.getState().serverChatMetaLoaded) return
        const state = useStoreMessageOption.getState()
        state.setServerChatLoadError(METADATA_FAILURE)
        state.setServerChatLoadState("failed")
      } finally {
        stopAuthority?.()
        snapshot?.release()
      }
    }
    void load()
    return () => { controller.abort() }
  }, [serverChatId, serverChatMetaLoaded, retryRevision])

  const isReady = !serverChatId || serverChatMetaLoaded
  const completionRevision = lifetime.current.completionRevision
  const isQueuedCompletionCurrent = React.useCallback(
    () => lifetime.current.active && lifetime.current.completionRevision === completionRevision,
    [completionRevision]
  )
  const failed = !isReady && serverChatLoadState === "failed"
  const retryMetadata = React.useCallback(() => {
    if (lifetime.current.active && useStoreMessageOption.getState().serverChatId === serverChatId) retry()
  }, [serverChatId])
  const captureQueuedDispatchGuard = React.useCallback((conversationId?: string | null) => {
    let revision = lifetime.current.revision
    let expectedServerChatId = serverChatId
    let expectedHistoryId = useStoreMessageOption.getState().historyId
    const ownedCompletionRevision = lifetime.current.completionRevision
    const assertOwner = () => {
      const state = useStoreMessageOption.getState()
      if (!lifetime.current.active || lifetime.current.revision !== revision || state.serverChatId !== expectedServerChatId ||
        state.historyId !== expectedHistoryId) {
        throw new Error("Conversation changed. The queued request was not sent.")
      }
    }
    const assertCurrent = () => {
      assertOwner()
      const state = useStoreMessageOption.getState()
      if (conversationId && conversationId !== state.historyId && conversationId !== state.serverChatId) {
        throw new Error("Conversation changed. The queued request was not sent.")
      }
      if (state.serverChatId && !state.serverChatMetaLoaded) {
        throw new Error(state.serverChatLoadState === "failed" ? METADATA_FAILURE : METADATA_WAITING)
      }
    }
    assertCurrent()
    const publishIdentity = (field: "serverChatId" | "historyId", id: string | null, publish: (id: string | null) => void) => {
      assertOwner()
      const expected = field === "serverChatId" ? expectedServerChatId : expectedHistoryId
      if (expected && !(field === "historyId" && expected === "temp") && id !== expected) {
        throw new Error("Conversation changed. The queued request was not sent.")
      }
      ownPublication.current = { field, id }
      try {
        publish(id)
      } finally {
        ownPublication.current = null
      }
      if (!lifetime.current.active || lifetime.current.completionRevision !== ownedCompletionRevision ||
        useStoreMessageOption.getState()[field] !== id) {
        throw new Error("Conversation changed. The queued request was not sent.")
      }
      if (field === "serverChatId") expectedServerChatId = id
      else expectedHistoryId = id
      revision = lifetime.current.revision
    }
    return Object.assign(assertCurrent, {
      publishServerChatId: (id: string | null, publish: (id: string | null) => void) => {
        publishIdentity("serverChatId", id, publish)
      },
      publishHistoryId: (id: string | null, publish: (id: string | null) => void) => {
        publishIdentity("historyId", id, publish)
      },
    }) satisfies QueuedDispatchGuard
  }, [serverChatId])
  return { isReady, failed, retryMetadata, captureQueuedDispatchGuard, isQueuedCompletionCurrent }
}
