import {
  applyChatSettingsPatch,
  chatSettingsStorageForKey,
  getChatSettingsForKey,
  getChatSettingsRecovery,
  type ChatSettingsRecovery,
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey,
  type ChatSettingsTarget
} from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"
import React from "react"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { captureChatDraftSource } from "./chat-draft-transition"

import { useStorage } from "@plasmohq/storage/hook"

import { useHistorySelectionContext } from "./useHistorySelection"

type SettingsPresentation = { storageKey: string; settings: ChatSettingsRecord | null }

type UseChatSettingsRecordParams = {
  historyId: string | null
  serverChatId: string | null
}

export const useChatSettingsRecord = ({
  historyId,
  serverChatId
}: UseChatSettingsRecordParams) => {
  const controller = useHistorySelectionContext()
  const mode = controller?.settingsMode?.(serverChatId) ?? "ordinary"
  const guarded = mode !== "ordinary"
  const stableHistoryId = historyId && historyId !== "temp" ? historyId : null
  const draftAddress = useStoreMessageOption(state => state.chatDraftAddress)
  const draftRevision = useStoreMessageOption(state => state.chatDraftRevision)
  const restoreRevision = usePlaygroundSessionStore(state => state.restoreRevision)
  const selectionRevision = usePlaygroundSessionStore(state => state.sourceSelectionRevision)
  const renderedOwner = controller?.getCurrent?.().owner
  const nativeOwner = !guarded && renderedOwner?.kind === "native" && renderedOwner.conversation_id === serverChatId
    ? renderedOwner : null
  const draftAccount = controller?.draftAccount
  const ownerPending = Boolean(serverChatId && !nativeOwner) || Boolean(!stableHistoryId && !serverChatId &&
    (!draftAccount?.requestScope || !draftAccount.isCurrent() || draftAddress?.kind !== "fresh"))
  const selectionSignal = controller?.getSignal?.()
  const { historyId: renderedHistoryId, serverChatId: renderedServerId } = useStoreMessageOption.getState()
  const target = React.useMemo<ChatSettingsTarget>(() => ({
    historyId: stableHistoryId,
    serverChatId,
    requestScope: nativeOwner?.request_scope ?? (!serverChatId ? draftAccount?.requestScope ?? undefined : undefined),
    scope: nativeOwner?.scope ?? (draftAddress?.kind === "fresh" ? draftAddress.scope : undefined),
    draftAddress,
    signal: nativeOwner ? selectionSignal : undefined
  }), [stableHistoryId, serverChatId, nativeOwner, draftAccount?.requestScope, draftAddress, selectionSignal])
  const chatKey = guarded || ownerPending ? "h1-settings-presentation" : resolveChatSettingsKey(target)
  const renderedSource = React.useRef<object | null>(null)
  const draftRequestScope = serverChatId ? null : draftAccount?.requestScope
  const draftIsCurrent = serverChatId ? null : draftAccount?.isCurrent
  // The tuple marks the rendered source lifetime, including same-ID restoration.
  const sourceToken = React.useMemo(() => [
    chatKey, guarded, mode, ownerPending, nativeOwner, serverChatId, stableHistoryId,
    draftRequestScope, draftIsCurrent, selectionSignal, draftAddress, draftRevision,
    restoreRevision, selectionRevision
  ], [chatKey, guarded, mode, ownerPending, nativeOwner, serverChatId, stableHistoryId,
    draftRequestScope, draftIsCurrent, selectionSignal, draftAddress, draftRevision,
    restoreRevision, selectionRevision])
  const renderedH1 = React.useMemo(() => {
    const fence = controller?.fence?.() ?? (() => true)
    return () => renderedSource.current === sourceToken && fence()
  }, [controller, sourceToken])
  React.useLayoutEffect(() => {
    renderedSource.current = sourceToken
    return () => { if (renderedSource.current === sourceToken) renderedSource.current = null }
  }, [sourceToken])
  const storageKey = React.useMemo(
    () => getChatSettingsStorageKey(chatKey),
    [chatKey]
  )

  const presentationStorage = React.useMemo(() => {
    const base = chatSettingsStorageForKey(chatKey)
    const stamp = (key: string, raw: unknown): SettingsPresentation => ({ storageKey: key, settings: normalizeChatSettingsRecord(
      chatKey.startsWith("server:owned:") || chatKey.startsWith("draft:owned:")
        ? (raw as { value?: ChatSettingsRecord } | null)?.value : raw
    ) })
    const subscriptions = new WeakMap<Parameters<typeof base.watch>[0], Parameters<typeof base.watch>[0]>()
    return {
      get: async (key: string) => stamp(key, await base.get(key)),
      watch: (callbacks: Parameters<typeof base.watch>[0]) => {
        const mapped = Object.fromEntries(Object.entries(callbacks).map(([key, callback]) => [key, (...args: Parameters<typeof callback>) => {
          const [change, ...context] = args
          return callback({ ...change, newValue: stamp(key, change.newValue) }, ...context)
        }]))
        subscriptions.set(callbacks, mapped)
        return base.watch(mapped)
      },
      unwatch: (callbacks: Parameters<typeof base.watch>[0]) => base.unwatch(subscriptions.get(callbacks) ?? callbacks),
      set: () => Promise.reject(new Error("chat_settings_owner_unavailable")),
      remove: () => Promise.reject(new Error("chat_settings_owner_unavailable"))
    } as unknown as typeof base
  }, [chatKey])
  const [rawSettings] = useStorage<SettingsPresentation | undefined>({ key: storageKey, instance: presentationStorage })
  React.useEffect(() => {
    if (!guarded && !ownerPending) void getChatSettingsForKey(chatKey)
  }, [chatKey, guarded, ownerPending])
  const settings = React.useMemo(
    () =>
      guarded
        ? mode === "fork"
          ? (controller?.forkSettings ?? null)
          : null
        : ownerPending || rawSettings?.storageKey !== storageKey ? null : rawSettings.settings,
    [rawSettings, guarded, ownerPending, storageKey, mode, controller?.forkSettings]
  )

  const renderedIsCurrent = React.useCallback(() => {
    const live = useStoreMessageOption.getState()
    const session = usePlaygroundSessionStore.getState()
    return renderedH1() &&
      live.historyId === renderedHistoryId && live.serverChatId === renderedServerId &&
      live.chatDraftAddress === draftAddress && live.chatDraftRevision === draftRevision &&
      session.restoreRevision === restoreRevision && session.sourceSelectionRevision === selectionRevision
  }, [renderedH1, renderedHistoryId, renderedServerId, draftAddress, draftRevision, restoreRevision, selectionRevision])
  const captureTarget = React.useCallback((): ChatSettingsTarget => {
    if (!renderedIsCurrent() || mode !== "ordinary" || ownerPending)
      throw new Error("chat_settings_owner_unavailable")
    const source = captureChatDraftSource(() => renderedIsCurrent() &&
      (nativeOwner ? nativeOwner.validate_lease() && controller?.getCurrent().owner === nativeOwner :
        stableHistoryId ? true : draftAccount?.isCurrent() === true))
    return { ...target, validate_lease: source.isCurrent }
  }, [renderedIsCurrent, mode, ownerPending, nativeOwner, controller, stableHistoryId, draftAccount, target])
  const [recoveryPresentation, setRecoveryPresentation] = React.useState<{
    source: typeof sourceToken; value: ChatSettingsRecovery | null; error: string | null
  } | null>(null)
  const recoveryReadAttempt = React.useRef(0)
  const refreshRecovery = React.useCallback(async (capturedTarget?: ChatSettingsTarget) => {
    if (!renderedIsCurrent() || guarded || !serverChatId || ownerPending) return
    const attempt = ++recoveryReadAttempt.current
    try {
      const value = await getChatSettingsRecovery(capturedTarget ?? captureTarget())
      if (renderedIsCurrent() && recoveryReadAttempt.current === attempt) setRecoveryPresentation({ source: sourceToken, value, error: null })
    } catch (error) {
      if (renderedIsCurrent() && recoveryReadAttempt.current === attempt) setRecoveryPresentation({ source: sourceToken, value: null,
        error: error instanceof Error ? error.message : "The previous settings change could not be read." })
    }
  }, [renderedIsCurrent, guarded, serverChatId, ownerPending, captureTarget, sourceToken])
  React.useEffect(() => { void refreshRecovery() }, [refreshRecovery])
  const ownsRecovery = recoveryPresentation?.source === sourceToken
  const recoveryReady = !serverChatId || guarded || (ownsRecovery && !recoveryPresentation.error)
  const settingsRecovery = ownsRecovery ? recoveryPresentation.value : null
  const settingsRecoveryError = ownsRecovery ? recoveryPresentation.error : null
  const updateSettings = React.useCallback(async (patch: Partial<ChatSettingsRecord>, recoveryOperationId?: string, reconcileOnly = false) => {
    if (!renderedIsCurrent()) throw new Error("chat_settings_owner_unavailable")
    if (mode === "pending") throw new Error("fork_settings_owner_unavailable")
    if (mode === "fork") return controller!.updateForkSettings(patch)
    const capturedTarget = captureTarget()
    try { return await applyChatSettingsPatch({ ...capturedTarget, patch, recoveryOperationId, reconcileOnly }) }
    finally { await refreshRecovery(capturedTarget) }
  }, [renderedIsCurrent, mode, controller, captureTarget, refreshRecovery])

  const isReady = guarded ? mode === "fork" : !ownerPending && recoveryReady && rawSettings?.storageKey === storageKey
  return { settings, updateSettings, chatKey, target, captureTarget, isReady, isCurrent: renderedIsCurrent, mode,
    settingsRecovery, settingsRecoveryError, refreshRecovery }
}
