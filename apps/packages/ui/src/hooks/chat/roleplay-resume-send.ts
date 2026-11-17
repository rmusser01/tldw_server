import { useStoreMessageOption } from "@/store/option"
import type { Dispatch, SetStateAction } from "react"
import type { Message } from "@/store/option"
import type {
  HistoryTurnRecovery,
  HistoryBookmarkScope
} from "@/db/dexie/types"
import {
  loadHistoryTurnRecoveries,
  saveHistoryTurnRecovery,
  retainHistoryTurnRecoveryInMemory,
  dismissHistoryTurnRecovery,
  historyDigest
} from "@/db/dexie/history-selection"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import {
  captureHistorySnapshot,
  assertHistoryConversationTarget
} from "@/services/chat-history-selection"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"
import {
  loadServicePromptSnapshot,
  type ServicePromptSnapshot
} from "@/services/service-prompts"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import {
  parseRoleplayResumeIntent,
  parseRoleplayResumeRecovery,
  parseRoleplayResumeAppendReceipt,
  buildRoleplayResumeCompletion,
  parseRoleplayResumeCompletionReceipt,
  requireRoleplayResumeDetail,
  isDefinitiveRoleplayResumeRefusal,
  RoleplayResumeResponseError,
  readSavedRoleplayPresentation,
  type RoleplayResumeRecovery
} from "@/services/tldw/roleplay-resume"
import type { HistorySelectionController } from "./useHistorySelection"
import {
  NativeHistoryContextRefusal,
  NativeHistorySettingsReadError
} from "./native-character-preflight"

export type SavedRoleplaySendParams = {
  controller: HistorySelectionController
  signal: AbortSignal
  originIsCurrent: () => boolean
  sourceIsCurrent: () => boolean
  message: string
  image?: string
  unsupportedContext?: boolean
  serverChatId?: string | null
  historyId?: string | null
  setMessages: Dispatch<SetStateAction<Message[]>>
  acknowledgeInput?: () => boolean
  retry?: { scope: HistoryBookmarkScope; turn: HistoryTurnRecovery }
}

/** The existing journal owns uncertainty; only original owner receipts may promote it. */
export const sendSavedRoleplay = async (
  params: SavedRoleplaySendParams
): Promise<boolean> => {
  const current = params.controller.getCurrent()
  if (
    current.owner?.kind !== "native" ||
    current.status !== "ready" ||
    !current.view ||
    !current.bookmarkScope ||
    current.capture?.status !== "captured" ||
    current.owner.scope?.type === "workspace"
  )
    throw new NativeHistorySettingsReadError(
      new Error("roleplay_resume_unavailable"),
      params.sourceIsCurrent
    )
  const owner = current.owner,
    originView = current.view
  const isSourceCurrent = () =>
    owner.validate_lease() && params.sourceIsCurrent()
  const preparingIsCurrent = () =>
    isSourceCurrent() &&
    !params.signal.aborted &&
    params.originIsCurrent() &&
    params.controller.getCurrent().owner === owner &&
    params.controller.getCurrent().view === originView
  let snapshot: ServicePromptSnapshot | undefined
  let entry:
    | { scope: HistoryBookmarkScope; turn: HistoryTurnRecovery }
    | undefined
  let protocol: RoleplayResumeRecovery | undefined
  let retainedResultTexts: string[] = []
  let resultText = "",
    appendDispatched = false,
    completionDispatched = false,
    completionWasUnconfirmed = false,
    completionConfirmed = false
  let journalStarted = false,
    settled = false,
    loadedRecovery = false
  const authCurrent = () =>
    owner.validate_lease() && !snapshot?.scopeInvalidatedSignal.aborted
  const nextTransportIsCurrent = () =>
    authCurrent() && preparingIsCurrent() && !snapshot?.scopeSignal.aborted
  const samePresentation = () =>
    authCurrent() &&
    params.sourceIsCurrent() &&
    params.controller.getCurrent().owner === owner &&
    params.controller.getCurrent().view === originView
  const projection = (
    state: HistoryTurnRecovery["state"]
  ): HistoryTurnRecovery => ({
    ...entry!.turn,
    state,
    result_text: resultText,
    roleplay_resume: protocol!,
    ...(retainedResultTexts.length
      ? { retained_result_texts: retainedResultTexts }
      : {}),
    ...(protocol?.append ? { input_id: protocol.append.message_id } : {}),
    ...(protocol?.completion?.saved
      ? { assistant_id: protocol.completion.assistant_message_id! }
      : {})
  })
  const retain = async (
    state: HistoryTurnRecovery["state"],
    extra: Partial<HistoryTurnRecovery> = {}
  ) => {
    await saveHistoryTurnRecovery(entry!.scope, entry!.turn.origin_view, {
      ...projection(state),
      ...extra,
      persistence: "server"
    })
  }
  const showRecovery = async () => {
    if (samePresentation()) await params.controller.refreshRecovery()
  }
  const finishSaved = async () => {
    const response = protocol!.completion!
    settled = true
    // Own following advances H1; acknowledge only the still-selected original input.
    if (preparingIsCurrent()) params.acknowledgeInput?.()
    await retain("generated_unsaved", {
      outcome: "complete",
      settled_message_id: response.assistant_message_id!,
      model_name: response.model ?? response.provider,
      model_id: response.model ?? undefined
    })
    const retained = (
      await loadHistoryTurnRecoveries(entry!.scope, originView)
    ).find((value) => value.turn.operation_id === entry!.turn.operation_id)
    if (samePresentation()) {
      const followed = await params.controller.followResult(
        originView,
        response.assistant_message_id!,
        () => authCurrent() && params.sourceIsCurrent()
      )
      const installed = params.controller.getCurrent()
      if (
        followed !== false &&
        authCurrent() &&
        installed.owner === owner &&
        installed.status === "ready" &&
        installed.view?.cursor.kind === "after_message" &&
        installed.view.cursor.message_id === response.assistant_message_id
      ) {
        if (retained?.turn.retained_result_texts?.length) {
          await params.controller.refreshRecovery()
          return
        }
        await dismissHistoryTurnRecovery(
          entry!.scope,
          entry!.turn.origin_view,
          entry!.turn.operation_id,
          "completed",
          { validate_lease: authCurrent }
        )
        return
      }
    }
    await showRecovery()
  }
  try {
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    if (!params.message.trim() || params.image || params.unsupportedContext)
      throw new Error("roleplay_resume_explicit_context_unsupported")
    if (!params.retry && params.controller.takeBranchIntent?.(originView))
      throw new Error("roleplay_resume_tail_required")
    await assertHistoryConversationTarget(owner, {
      historyId: params.historyId,
      serverChatId: params.serverChatId
    })
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    const retained = await loadHistoryTurnRecoveries(
      current.bookmarkScope,
      originView
    )
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    if (params.retry) {
      entry = retained.find(
        (value) =>
          value.scope.profile_id === params.retry!.scope.profile_id &&
          value.scope.client_session_id ===
            params.retry!.scope.client_session_id &&
          value.turn.operation_id === params.retry!.turn.operation_id
      )
      if (!entry?.turn.roleplay_resume)
        throw new Error("roleplay_resume_recovery_changed")
    } else {
      const unfinished = retained.filter(
        (value) =>
          value.turn.roleplay_resume &&
          value.turn.state !== "refused" &&
          value.turn.outcome !== "complete"
      )
      if (unfinished.length)
        throw new Error("roleplay_resume_review_required")
    }
    if (entry) {
      loadedRecovery = true
      protocol = parseRoleplayResumeRecovery(entry.turn.roleplay_resume)
      completionWasUnconfirmed = entry.turn.state === "unknown" && Boolean(protocol.append)
      if (
        entry.turn.owner_key !== originView.owner_key ||
        entry.turn.conversation_id !== owner.conversation_id ||
        protocol.intent.input.content !== params.message
      )
        throw new Error("roleplay_resume_recovery_changed")
      resultText = entry.turn.result_text
      retainedResultTexts = [...(entry.turn.retained_result_texts ?? [])]
      journalStarted = true
    }
    snapshot = await loadServicePromptSnapshot([], {
      signal: params.signal,
      requestScope: owner.request_scope
    })
    if (
      chatDraftOwnerKey(snapshot.requestScope) !==
        chatDraftOwnerKey(owner.request_scope) ||
      !nextTransportIsCurrent()
    )
      throw new Error("request_config_scope_changed")
    if (protocol?.completion?.saved) {
      await finishSaved()
      return true
    }
    const capability = await getServerCapabilities({
      requestScope: snapshot.requestScope,
      signal: snapshot.scopeSignal
    })
    if (!nextTransportIsCurrent()) throw new Error("stale_selection")
    if (capability.hasRoleplayResumeV1 !== true)
      throw new Error("roleplay_resume_server_unsupported")
    if (!entry) {
      const detail = await tldwClient.getChat(owner.conversation_id, {
        requestScope: snapshot.requestScope,
        signal: snapshot.scopeSignal,
        requireRoleplayResume: true,
        scope: { type: "global" }
      })
      if (!nextTransportIsCurrent()) throw new Error("stale_selection")
      const expected = requireRoleplayResumeDetail(
        owner.conversation_id,
        detail
      )
      if (
        samePresentation() &&
        useStoreMessageOption.getState().serverChatId === owner.conversation_id
      )
        useStoreMessageOption
          .getState()
          .setServerChatRoleplaySetup(readSavedRoleplayPresentation(detail))
      const capture = await captureHistorySnapshot(
        owner,
        originView,
        "send",
        snapshot.scopeSignal
      )
      if (!nextTransportIsCurrent()) throw new Error("stale_selection")
      if (
        capture.status !== "captured" ||
        originView.interpretation.kind !== "parent_graph_v1" ||
        (expected.tail.message_id === null
          ? originView.cursor.kind !== "empty" || capture.rows.length !== 0
          : originView.cursor.kind !== "after_message" ||
            originView.cursor.message_id !== expected.tail.message_id ||
            capture.rows.at(-1)?.id !== expected.tail.message_id)
      )
        throw new Error("roleplay_resume_tail_required")
      protocol = {
        intent: parseRoleplayResumeIntent({
          version: 1,
          conversation_id: owner.conversation_id,
          input: {
            message_id: crypto.randomUUID(),
            content: params.message,
            expected
          },
          assistant_message_id: crypto.randomUUID()
        })
      }
      entry = {
        scope: current.bookmarkScope,
        turn: {
          operation_id: crypto.randomUUID(),
          origin_view: originView,
          owner_key: originView.owner_key,
          conversation_id: owner.conversation_id,
          persistence: "server",
          created_at: Date.now(),
          input_text: params.message,
          input_images: [],
          result_text: "",
          state: "dispatching",
          roleplay_resume: protocol,
          selection_digest: await historyDigest(
            capture.rows.map((row) => ({ id: row.id, revision: row.revision }))
          ),
          request_context_digest: await historyDigest(protocol.intent)
        }
      }
      if (!nextTransportIsCurrent()) throw new Error("stale_selection")
      await retain("dispatching")
      journalStarted = true
      if (!nextTransportIsCurrent()) throw new Error("stale_selection")
    }
    if (!protocol!.append) {
      const receipt = await tldwClient.appendRoleplayResumeInput(
        owner.conversation_id,
        protocol!.intent.input,
        {
          requestScope: snapshot.requestScope,
          signal: snapshot.scopeSignal,
          onDispatch: () => {
            appendDispatched = true
          }
        }
      )
      protocol = {
        ...protocol!,
        append: parseRoleplayResumeAppendReceipt(
          owner.conversation_id,
          protocol!.intent.input,
          receipt
        )
      }
      await retain("accepted_unsent")
    }
    if (!nextTransportIsCurrent()) {
      await showRecovery()
      return false
    }
    const request = buildRoleplayResumeCompletion(
      protocol!.intent,
      protocol!.append!
    )
    // Persist uncertainty before transport so cold recovery never implies an unsent reply.
    await retain("unknown")
    if (!nextTransportIsCurrent()) {
      await retain(completionWasUnconfirmed ? "unknown" : "accepted_unsent")
      await showRecovery()
      return false
    }
    const raw = await tldwClient.completeCharacterChatTurn(
      owner.conversation_id,
      request,
      {
        requestScope: snapshot.requestScope,
        signal: snapshot.scopeSignal,
        onDispatch: () => {
          completionDispatched = true
        }
      }
    )
    if (typeof raw.assistant_content === "string")
      resultText = raw.assistant_content
    const receipt = parseRoleplayResumeCompletionReceipt(
      owner.conversation_id,
      request,
      raw
    )
    completionConfirmed = true
    resultText = receipt.assistant_content
    protocol = { ...protocol!, completion: receipt }
    if (receipt.saved) await finishSaved()
    else {
      await retain("generated_unsaved")
      await showRecovery()
    }
    return receipt.saved
  } catch (error) {
    if (error instanceof RoleplayResumeResponseError && error.generatedText) {
      if (
        protocol?.completion &&
        protocol.completion.assistant_content !== error.generatedText
      ) {
        retainedResultTexts.push(error.generatedText)
        resultText = protocol.completion.assistant_content
      } else resultText = error.generatedText
    }
    if (journalStarted && entry && protocol) {
      const freshDefinitiveRefusal =
        !loadedRecovery &&
        !protocol.append &&
        !completionDispatched &&
        !resultText &&
        entry.turn.state !== "unknown" &&
        isDefinitiveRoleplayResumeRefusal(error)
      const failureState: HistoryTurnRecovery["state"] = freshDefinitiveRefusal
        ? "refused"
        : !completionConfirmed && (completionDispatched || completionWasUnconfirmed)
          ? "unknown"
          : settled || resultText
            ? "generated_unsaved"
            : protocol.append
              ? "accepted_unsent"
              : appendDispatched || loadedRecovery ? "unknown" : "dispatching"
      try {
        await retain(failureState)
        await showRecovery()
      } catch {
        if (resultText) {
          await retainHistoryTurnRecoveryInMemory(
            entry.scope,
            entry.turn.origin_view,
            {
              ...projection(failureState),
              ...(protocol.completion?.saved
                ? {
                    outcome: "complete",
                    settled_message_id:
                      protocol.completion.assistant_message_id!
                  }
                : {})
            }
          )
          await showRecovery().catch(() => undefined)
        }
        // This fallback never paints a replacement view or grants message ancestry.
        if (samePresentation() && resultText)
          params.setMessages([
            ...formatSelectedHistory(current.capture).messages,
            {
              id: `recovered-${entry.turn.operation_id}`,
              isBot: true,
              name: "Recovered reply — save status unverified",
              message: resultText,
              sources: [],
              createdAt: Date.now()
            }
          ])
      }
      if (settled) return true
      if (freshDefinitiveRefusal)
        throw new NativeHistoryContextRefusal(error, samePresentation)
      throw error
    }
    throw new NativeHistorySettingsReadError(error, preparingIsCurrent)
  } finally {
    snapshot?.release()
  }
}
