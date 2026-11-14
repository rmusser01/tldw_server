import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { captureChatDraftLoad, captureChatDraftSource, confirmTemporaryChatDiscard } from "@/hooks/chat/chat-draft-transition"
import React from "react"
import { message } from "antd"

import { PageAssistDatabase, type OnDeviceHistoryScope } from "@/db/dexie/chat"
import {
  formatToChatHistory,
  formatToMessage,
  getPromptById,
  getSessionFiles
} from "@/db/dexie/helpers"
import { lastUsedChatModelEnabled } from "@/services/model-settings"
import { updatePageTitle } from "@/utils/update-page-title"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

interface LoadLocalConversationDeps {
  setServerChatId: (id: string | null) => void
  setHistoryId: (id: string) => void
  setHistory: (history: any) => void
  setMessages: (messages: any[]) => void
  setSelectedModel: (id: string) => void
  setSelectedSystemPrompt: (id: string | null) => void
  setSystemPrompt: (prompt: string) => void
  setContextFiles: (files: any[]) => void
}

interface UseLoadLocalConversationOptions {
  t: (key: string, options?: any) => string
  errorLogPrefix: string
  errorDefaultMessage: string
}

export function useLoadLocalConversation(
  deps: LoadLocalConversationDeps,
  options: UseLoadLocalConversationOptions
) {
  const selection = useHistorySelectionContext()
  // Inline setters and controller state changes must not restart an effect-owned load.
  const inputs = React.useRef({ deps, options, selection })
  inputs.current = { deps, options, selection }
  const dbRef = React.useRef<PageAssistDatabase | null>(null)
  const mountedRef = React.useRef(false)
  const loadGenerationRef = React.useRef(0)

  React.useEffect(() => {
    mountedRef.current = true
    const invalidate = () => { loadGenerationRef.current += 1 }
    const stop = watchChatAccountChanges(invalidated => { if (invalidated) invalidate() })
    return () => {
      mountedRef.current = false
      invalidate()
      stop()
    }
  }, [])

  if (!dbRef.current) {
    dbRef.current = new PageAssistDatabase()
  }

  return React.useCallback(
    async function loadLocalConversation(conversationId: string, readerScope?: OnDeviceHistoryScope): Promise<boolean> {
      if (!mountedRef.current || (readerScope && !readerScope.isCurrent())) return false
      const source = captureChatDraftSource()
      if (!confirmTemporaryChatDiscard() || !source.isCurrent()) return false
      const acceptDraft = captureChatDraftLoad("switch")
      const {
        deps: {
          setServerChatId,
          setHistoryId,
          setHistory,
          setMessages,
          setSelectedModel,
          setSelectedSystemPrompt,
          setSystemPrompt,
          setContextFiles
        },
        options: { t, errorLogPrefix, errorDefaultMessage },
        selection
      } = inputs.current
      const { beginLoad, fence, loadConversation, getCurrent } = selection ?? {}
      const generation = ++loadGenerationRef.current
      const restoreRevision = usePlaygroundSessionStore.getState().restoreRevision
      if (!mountedRef.current) return false
      beginLoad?.()
      let selectionCurrent = fence?.() || (() => true)
      let snapshot: ServicePromptSnapshot | undefined
      let settled = false
      // The controller advances its own fence while opening. This caller lease
      // must track navigation/account changes independently of that fence.
      const isCurrentLoad = () => acceptDraft.isCurrent() && (settled || (
        mountedRef.current && (!readerScope || readerScope.isCurrent()) &&
        !snapshot?.scopeSignal.aborted && !snapshot?.scopeInvalidatedSignal.aborted &&
        generation === loadGenerationRef.current &&
        restoreRevision === usePlaygroundSessionStore.getState().restoreRevision
      ))
      const isCurrent = () => isCurrentLoad() && selectionCurrent()
      try {
        const db = dbRef.current!
        if (readerScope && !loadConversation) return false
        const preview = readerScope ? await db.getOnDeviceChatPreview(conversationId, readerScope) : null
        const historyDetails = readerScope ? preview?.historyInfo : await db.getHistoryInfo(conversationId)
        if (!isCurrent() || !historyDetails) return false
        // Profile-owned forks browse independently of the inference account.
        // The controller still validates the local profile before publishing.
        const profileOwned = loadConversation && historyDetails.local_owner_key &&
          !historyDetails.server_scope_key && !historyDetails.server_chat_id &&
          historyDetails.message_source !== "server"
        const unboundLocal = loadConversation && !historyDetails.local_owner_key && !historyDetails.server_scope_key && !historyDetails.server_chat_id && historyDetails.message_source !== "server"
        if (!profileOwned && !unboundLocal) {
          snapshot = await loadServicePromptSnapshot([])
          if (!isCurrent() || historyDetails.server_scope_key !== serverChatMirrorOwnerKey(snapshot)) return false
        }
        if (unboundLocal && !readerScope) {
          const offered = await loadConversation!({ historyId: conversationId, isCurrent: isCurrentLoad, resumeAfterBind: () => isCurrentLoad() ? loadLocalConversation(conversationId, readerScope) : Promise.resolve(false) })
          if (offered && isCurrentLoad() && getCurrent?.().owner?.kind === "local")
            return loadLocalConversation(conversationId, readerScope)
          return false
        }
        const history = preview?.messages ?? await db.getChatHistory(conversationId)
        if (!isCurrent()) return false
        // Prepare every fallible read before changing the owned selection or IDs.
        const isLastUsedChatModel = await lastUsedChatModelEnabled()
        if (!isCurrent()) return false
        const lastUsedPrompt = historyDetails.last_used_prompt
        let promptContent = lastUsedPrompt?.prompt_content ?? ""
        const prompt = lastUsedPrompt?.prompt_id ? await getPromptById(lastUsedPrompt.prompt_id) : undefined
        if (!isCurrent()) return false
        if (prompt && !promptContent.trim()) promptContent = prompt.content
        const session = await getSessionFiles(conversationId)
        if (!isCurrent()) return false
        if (loadConversation && fence) {
          let accepted = false
          if (!await loadConversation({ historyId: conversationId, isCurrent: isCurrentLoad, onDeviceScope: readerScope, resumeAfterBind: () => isCurrentLoad() ? loadLocalConversation(conversationId, readerScope) : Promise.resolve(false) }, null, receipt => { if (!readerScope || receipt.owner.kind === "local") accepted = acceptDraft(receipt) })) return false
          if (generation !== loadGenerationRef.current) return false
          selectionCurrent = fence()
          const selected = getCurrent?.()
          if (!accepted && selected?.owner && selected.owner.kind !== "unavailable" &&
            (!readerScope || selected.owner.kind === "local") && selected.view &&
            selected.owner.conversation_id === (historyDetails.server_chat_id || conversationId)) {
            // A review state may locate a private draft; it does not authorize sending.
            accepted = acceptDraft({ owner: selected.owner, view: selected.view })
          }
          if (!accepted) return false
        }
        if (!isCurrent()) return false
        const selected = getCurrent?.()
        if (selected?.owner?.kind === "unavailable") return false
        setServerChatId(selected?.owner?.kind === "native" ? selected.owner.conversation_id : null)
        setHistoryId(conversationId)
        if (!selected || selected.capture?.status !== "captured") {
          setHistory(formatToChatHistory(history))
          setMessages(formatToMessage(history))
        }

        if (isLastUsedChatModel && historyDetails.model_id) setSelectedModel(historyDetails.model_id)
        if (prompt) setSelectedSystemPrompt(prompt.id)
        if (lastUsedPrompt) setSystemPrompt(promptContent)
        setContextFiles(session)

        updatePageTitle(
          historyDetails?.title || t("common:untitled", { defaultValue: "Untitled" })
        )
        if (!isCurrent() || (selected && (getCurrent?.().owner !== selected.owner || getCurrent?.().view !== selected.view))) return false
        // H1 retains this callback in an account-bound owner's lease. Once the
        // complete destination is accepted, widget lifetime must not revoke it.
        // Core still fences New/address ABA; H1 owns account/provider lifetime.
        settled = !!loadConversation
        return true
      } catch (error) {
        if (!isCurrent()) return false
        // eslint-disable-next-line no-console
        console.error(`${errorLogPrefix}:`, error)
        message.error(
          t("common:error.friendlyLocalHistorySummary", {
            defaultValue: errorDefaultMessage
          })
        )
        return false
      } finally {
        snapshot?.release()
      }
    },
    []
  )
}

/** Read-only comparison presentation; this never upgrades an unsupported send capture. */
export async function restoreReadableLocalComparison(
  controller: import('@/hooks/chat/useHistorySelection').HistorySelectionController,
  display: (value: { history: ReturnType<typeof formatToChatHistory>; messages: ReturnType<typeof formatToMessage> }) => void,
  supplied?: Awaited<ReturnType<typeof import('@/db/dexie/helpers').getFullChatData>>
): Promise<boolean> {
  const initial = controller.getCurrent()
  const owner = initial.owner
  const view = initial.view
  if (owner?.kind !== 'local' || !view || initial.error !== 'unsupported_comparison_history' || initial.capture?.snapshot.owner_key !== owner.owner_key) return false
  const current = controller.fence()
  const { getFullChatData } = await import('@/db/dexie/helpers')
  const data = supplied === undefined ? await getFullChatData(owner.conversation_id) : supplied
  if (!current() || !data || data.historyInfo.id !== owner.conversation_id) return false
  const { getLocalHistoryOwner } = await import('@/db/dexie/history-selection')
  let verified
  try { verified = await getLocalHistoryOwner(owner.conversation_id) } catch (error) {
    console.warn("Failed to verify readable comparison owner", error)
    return false
  }
  if (!current() || controller.getCurrent().owner !== owner || controller.getCurrent().view !== view || verified.owner_key !== owner.owner_key || verified.profile_id !== owner.profile_id) return false
  display({ history: formatToChatHistory(data.messages), messages: formatToMessage(data.messages) })
  return true
}
