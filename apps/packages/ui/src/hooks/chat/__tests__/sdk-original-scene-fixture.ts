import React from "react"
import { vi } from "vitest"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useChatActions as useRawChatActions } from "../useChatActions"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftSource } from "../chat-draft-transition"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { captureActorSceneSendForTarget } from "../actor-scene-source"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

// These SDK cases retain a real private Scene source without granting H1 authority.
export const useOriginalSdkSceneFixture = (
  options: Parameters<typeof useRawChatActions>[0],
  getAccount: () => Pick<
    ServicePromptSnapshot,
    "requestScope" | "scopeInvalidatedSignal"
  >
) => {
  const mounted = React.useRef(true)
  React.useEffect(
    () => () => {
      mounted.current = false
    },
    []
  )
  const initialized = React.useRef(false)
  if (!initialized.current) {
    const store = useStoreMessageOption
    store.getState().startNewChatDraft(options.temporaryChat)
    const address = store.getState().chatDraftAddress
    if (address?.kind === "fresh" && options.scope)
      store.getState().setChatDraftAddress({
        ...address,
        scope: structuredClone(options.scope)
      })
    store.setState({
      historyId: options.historyId,
      serverChatId: options.serverChatId,
      temporaryChat: options.temporaryChat,
      messages: options.messages,
      history: options.history,
      selectedModel: options.selectedModel ?? store.getState().selectedModel,
      toolChoice: options.toolChoice,
      serverChatAssistantKind: options.serverChatAssistantKind,
      serverChatMetaLoaded: options.serverChatMetaLoaded,
      serverChatCharacterId: options.serverChatCharacterId
    })
    for (const key of ["setHistoryId", "setServerChatId"] as const) {
      const spy = options[key]
      if (!vi.isMockFunction(spy)) {
        if (spy !== store.getState()[key])
          throw new Error(
            "SDK fixture requires the real publisher or a publishing spy"
          )
        continue
      }
      const original = spy.getMockImplementation()
      spy.mockImplementation((...args: unknown[]) => {
        original?.(...args)
        const publish = store.getState()[key] as (...values: unknown[]) => void
        publish(...args)
      })
    }
    initialized.current = true
  }
  const actions = useRawChatActions(options)
  return {
    ...actions,
    onSubmit: async (input: Parameters<typeof actions.onSubmit>[0]) => {
      if (input.actorSceneSend) return actions.onSubmit(input)
      const entry = getAccount()
      const scopeKey = chatDraftOwnerKey(entry.requestScope)
      const account = {
        requestScope: entry.requestScope,
        isCurrent: () =>
          !entry.scopeInvalidatedSignal.aborted &&
          getAccount().scopeInvalidatedSignal ===
            entry.scopeInvalidatedSignal &&
          chatDraftOwnerKey(getAccount().requestScope) === scopeKey
      }
      const original = useStoreMessageOption.getState()
      const session = usePlaygroundSessionStore.getState()
      const commandIsCurrent = () => {
        const current = useStoreMessageOption.getState()
        const currentSession = usePlaygroundSessionStore.getState()
        return (
          mounted.current &&
          account.isCurrent() &&
          current.chatDraftAddress === original.chatDraftAddress &&
          current.chatDraftRevision === original.chatDraftRevision &&
          current.temporaryChat === original.temporaryChat &&
          currentSession.restoreRevision === session.restoreRevision &&
          currentSession.sourceSelectionRevision ===
            session.sourceSelectionRevision
        )
      }
      const source = captureChatDraftSource(commandIsCurrent)
      const target = captureActorSettingsTarget({
        address: useStoreMessageOption.getState().chatDraftAddress,
        requestScope: account.requestScope,
        accountIsCurrent: account.isCurrent,
        isCurrent: source.isCurrent
      })
      const capture = captureActorSceneSendForTarget(
        target,
        account,
        source.isCurrent,
        commandIsCurrent
      )
      if (!capture) throw new Error("SDK fixture original Scene is unavailable")
      return actions.onSubmit({
        ...input,
        actorSceneSend: await capture.read()
      })
    }
  }
}
