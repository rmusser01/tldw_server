import React from "react"
import { captureActorSceneSend, type ActorSceneSend } from "./chat/actor-scene-source"
import { useTranslation } from "react-i18next"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { nativeCharacterPreflightMessage } from "./chat/native-character-preflight"
import type { UseComposerTextResult } from "@/components/Chat/composer/hooks/useComposerText"
import type { GreetingOption, GreetingSeed } from "@/utils/character-greetings"
import { chatDraftAddressKey } from "@/utils/chat-draft-address"
import type { HistorySelectionController } from "./chat/useHistorySelection"
import type { ChatSubmitResult } from "./chat/chat-action-utils"

type Options = {
  form: UseComposerTextResult["form"]
  retainCurrentDraft: () => Promise<boolean>
  submit: (input: { message: string; image: string; greetingSeed: GreetingSeed; greetingIsCurrent: () => boolean; actorSceneSend: ActorSceneSend }) => Promise<ChatSubmitResult>
  ready: boolean
  isConnectionReady: boolean
  busy: boolean
  readContext: () => unknown[]
  controller: HistorySelectionController | null
  focusComposer: () => void
  preflight: () => void
}

/** Both composers retain their exact unsent input before a deliberate assistant-only start. */
export const useCharacterGreetingStart = (options: Options) => {
  const { t } = useTranslation(["sidepanel", "playground"])
  const active = React.useRef(options)
  active.current = options
  const mounted = React.useRef(true)
  const connectionReady = React.useRef(options.isConnectionReady)
  const connectionGeneration = React.useRef(0)
  React.useLayoutEffect(() => {
    if (connectionReady.current && !options.isConnectionReady) connectionGeneration.current++
    connectionReady.current = options.isConnectionReady
  }, [options.isConnectionReady])
  const inFlight = React.useRef<{ current: () => boolean } | null>(null)
  const [activity, setActivity] = React.useState<{ current: () => boolean } | null>(null)
  // A newer source retires old UI activity immediately; its pending storage/request result stays fenced.
  useStoreMessageOption(state => state.chatDraftRevision)
  const isPreparing = React.useCallback(() => inFlight.current?.current() === true, [])
  const starting = activity?.current() === true
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const start = async (option: GreetingOption, choiceIsCurrent: () => boolean): Promise<void> => {
    if (isPreparing()) return
    const readinessGeneration = connectionGeneration.current
    const source = useStoreMessageOption.getState()
    const modelSettings = useStoreChatModelSettings.getState()
    const tools = useMcpToolsStore.getState()
    const session = usePlaygroundSessionStore.getState()
    const controller = options.controller
    const sourceFence = controller?.fence()
    const revision = options.form.getRevision()
    const values = { ...options.form.getValues() }
    const seed = option.seed ? structuredClone(option.seed) : undefined
    const inputKeys = ["selectedModel", "selectedModelRevision", "toolChoice", "temporaryChat", "webSearch", "selectedSystemPrompt", "uploadedFiles",
      "contextFiles", "documentContext", "selectedKnowledge", "ragMediaIds", "ragPinnedResults", "replyTarget",
      "messageSteeringMode", "messageSteeringForceNarrate", "messageSteeringRevision"] as const
    const inputs = options.readContext()
    const navigationCurrent = () => {
      const now = usePlaygroundSessionStore.getState()
      return now.restoreRevision === session.restoreRevision && now.sourceSelectionRevision === session.sourceSelectionRevision &&
        now.serverChatSelectionIntent === session.serverChatSelectionIntent
    }
    const current = () => {
      const now = useStoreMessageOption.getState()
      const context = active.current.readContext()
      return mounted.current && connectionReady.current && connectionGeneration.current === readinessGeneration && choiceIsCurrent() && options.form.getRevision() === revision && navigationCurrent() &&
        now.chatDraftAddress === source.chatDraftAddress && now.chatDraftRevision === source.chatDraftRevision &&
        useStoreChatModelSettings.getState() === modelSettings && useMcpToolsStore.getState() === tools &&
        inputKeys.every(key => Object.is(now[key], source[key])) &&
        inputs.length === context.length && inputs.every((value, index) => Object.is(value, context[index]))
    }
    const ownsBind = () => {
      const now = useStoreMessageOption.getState()
      const selection = controller?.getCurrent()
      const address = now.chatDraftAddress
      const transition = now.chatDraftTransition
      return mounted.current && navigationCurrent() && address?.kind === "native" && transition?.kind === "bind" &&
        transition.sourceAddress === source.chatDraftAddress && transition.destinationKey === chatDraftAddressKey(address) &&
        selection?.status === "ready" && selection.owner?.kind === "native" && selection.owner.validate_lease() &&
        selection.owner.conversation_id === address.id && selection.view?.conversation_id === address.id &&
        selection.view.owner_key === address.ownerKey
    }
    const operation = { current: () => current() || ownsBind() }
    inFlight.current = operation
    const initiator = typeof document !== "undefined" && document.activeElement instanceof HTMLElement && document.activeElement.tagName === "BUTTON" ? document.activeElement : null
    let focusOwned = Boolean(initiator)
    const leaveFocus = (event: Event) => { if (event.target !== initiator) focusOwned = false }
    document.addEventListener("focusin", leaveFocus)
    document.addEventListener("pointerdown", leaveFocus)
    try {
      const selection = controller?.getCurrent()
      if (!connectionReady.current) throw new Error("native_history_server_unavailable")
      if (!options.ready || options.busy || source.streaming || source.isLoading || source.isProcessing ||
          !seed || source.chatDraftAddress?.kind !== "fresh" || source.historyId || source.serverChatId ||
          source.messages.length || source.history.length || session.serverChatSelectionIntent ||
          !controller || selection?.status !== "idle" || selection.owner || !sourceFence?.() || !current())
        throw new Error("native_history_greeting_requires_fresh")
      if (values.image) throw new Error("native_history_assets_unsupported")
      const sceneCapture = captureActorSceneSend(controller, () => mounted.current && navigationCurrent())
      if (!sceneCapture) throw new Error("actor_settings_unavailable")
      options.preflight()
      setActivity(operation)
      const actorSceneSend = await sceneCapture.read()
      if (!sourceFence() || !current()) throw new Error("stale_selection")
      if (!(await options.retainCurrentDraft())) throw new Error("native_history_draft_not_retained")
      if (!sourceFence() || !current() || controller.getCurrent().status !== "idle" || controller.getCurrent().owner)
        throw new Error("stale_selection")
      const result = await options.submit({ message: "", image: "", greetingSeed: seed, greetingIsCurrent: current, actorSceneSend })
      if (result.status !== "submitted") throw new Error(result.status === "failed" ? result.errorMessage : result.reason)
      if (ownsBind() && focusOwned && initiator && (document.activeElement === initiator ||
          (!initiator.isConnected && document.activeElement === document.body))) options.focusComposer()
    } catch (error) {
      throw new Error(nativeCharacterPreflightMessage(error, t) || t("sidepanel:greetingPicker.startFailed", "Could not save the opening. Your draft is still here. Review the connection and retained drafts before starting again."))
    } finally {
      document.removeEventListener("focusin", leaveFocus)
      document.removeEventListener("pointerdown", leaveFocus)
      if (inFlight.current === operation) {
        inFlight.current = null
        if (mounted.current) setActivity(null)
      }
    }
  }
  return { starting, isPreparing, start }
}
