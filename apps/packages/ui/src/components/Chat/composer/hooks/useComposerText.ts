import { readChatCreationRecovery, listChatCreationRecovery, copyChatCreationRecovery, watchChatCreationRecovery, type ChatCreationRecovery } from "@/services/pending-chat-promotion"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { reviewNativeChatCreation } from "@/hooks/chat/native-history-character-send"
import { normalizeConversationState } from "@/utils/conversation-state"
import { requestSettingsNavigation, SETTINGS_NAVIGATION_REQUEST_EVENT, type SettingsNavigationRequestDetail, type InitialChatSetupAttempt } from "@/utils/settings-return"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { connectionAuthoritiesMatch, deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import {
  type DraftMetadataObject,
  useDraftPersistence
} from "@/hooks/useDraftPersistence"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import React from "react"
import { useStoreMessageOption } from "@/store/option"
import { chatDraftAddressKey, conversationDraftKey, freshChatDraftAddress, isChatDraftHandle, type ChatDraftAddress, type ChatDraftHandle } from "@/utils/chat-draft-address"

/**
 * Shared composer text primitive consumed by both Playground and Sidepanel.
 *
 * Bundles three pieces every composer needs:
 *   - message/image form state (`useSimpleForm`)
 *   - draft persistence keyed per surface (`useDraftPersistence`)
 *   - a focus helper that no-ops on mobile to avoid unwanted keyboard pop-ups
 *
 * Auto-resize (`useDynamicTextareaSize`) is intentionally NOT bundled here —
 * Playground sizes against a derived display value (paste-collapse overlay),
 * while Sidepanel sizes against the raw form value. Callers wire it in one
 * line using `textareaMaxHeight` from this hook. Passing ownership to the
 * caller is simpler than gating auto-resize behind an option flag.
 *
 * This is the foundation of the Phase 2 hook extraction. Playground's
 * `useComposerInput` wraps this and adds collapse/perf overlays; Sidepanel
 * adopts it directly, replacing its inline equivalents in `form.tsx`.
 */

export type RecoveredChatDraft = {
  key: string
  surfaceKey: string
  handle: ChatDraftHandle & { address: Extract<ChatDraftAddress, { kind: "fresh" }> }
  content: string
  metadata?: DraftMetadataObject
  updatedAt: number
  creationAttempt?: ChatCreationRecovery
}
export type ChatDraftRecoveryListing = { drafts: RecoveredChatDraft[]; unavailable: boolean }

export interface UseComposerTextOptions {
  draftAddress?: ChatDraftAddress
  recoverySurface?: string
  recoverySurfaceTabs?: boolean
  /** Persistence key for draft messages. Each surface uses its own key. */
  draftKey: string
  /** Explicit first-ID binding; ignored unless this source is already mounted. */
  bindFromDraftKey?: string
  /** Keep active browser tabs from replacing each other's unsent text. */
  tabScopedDraft?: boolean
  /** Textarea ref owned by the caller (usually the composer component). */
  textareaRef: React.RefObject<HTMLTextAreaElement>
  /** Pro mode gets a taller textarea (160px) vs casual (120px). */
  isProMode?: boolean
  /** Explicit max height override; wins over `isProMode`. */
  maxHeight?: number
  /**
   * Optional hook into draft-restore to carry surface-specific metadata
   * (e.g., Playground's `collapsedRange` + `wasExpanded`).
   */
  getDraftMetadata?: () => DraftMetadataObject | undefined
  /**
   * Called by `useDraftPersistence` when a stored draft is restored with
   * metadata. Callers override when they need to restore collapse state etc.
   */
  restoreWithMetadata?: (value: string, metadata?: DraftMetadataObject) => void
  /** Disable draft persistence entirely. Default true. */
  draftEnabled?: boolean
}

export type ComposerPromptAssistMutation =
  | { revision: number; source: "owner" }
  | {
      revision: number
      source: "optimistic_reset"
      attemptId: number
    }

export interface UseComposerTextResult {
  creationRecovery: { active: ChatCreationRecovery | null; unavailable: boolean; copyInputRequired: boolean }
  retainCurrentDraft: () => Promise<boolean>
  reviewCreatedChat: () => Promise<boolean>
  copyCreationText: () => Promise<boolean>
  listRetainedDrafts: () => Promise<ChatDraftRecoveryListing>
  copyRetainedDraft: (draft: RecoveredChatDraft, surfaceKey: string, destination?: Extract<ChatDraftAddress, { kind: "fresh" }>) => Promise<boolean>
  /** The underlying `useSimpleForm` instance. Callers can use `getInputProps("message")`. */
  form: ReturnType<typeof useSimpleForm<{ message: string; image: string }>>
  /** Monotonic signal that advances once for each committed message change. */
  messageRevision: number
  /** The committed message change and whether the owner explicitly reset it for an attempt. */
  promptAssistMutation: ComposerPromptAssistMutation
  /** Resets the owner and returns the exact optimistic attempt token. */
  beginPromptAssistReset: () => number
  /** Restore an unsuccessful attempt only if its original composer is still untouched. */
  restorePromptAssistAttempt: (attemptId: number) => void
  /** Marks a specific owner attempt as definitively sent or enqueued. */
  markPromptAssistAttemptSaved: (attemptId: number, acknowledgedDelivery?: boolean) => void
  /** Most recently completed owner attempt, or null before the first completion. */
  promptAssistSavedAttemptId: number | null
  /** Convenience setter for the message field. */
  setMessageValue: (value: string) => void
  /** Focuses the textarea (no-op on mobile to avoid unwanted keyboard pop-up). */
  textAreaFocus: () => void
  /** Retain the rendered source through delivery; focus also belongs to its initiator. */
  captureCompletionGuard: (forFocus?: boolean) => () => boolean
  /** Whether the draft was recently persisted — for the "Draft saved" indicator. */
  draftSaved: boolean
  /** Whether this owner's saved draft has finished restoring. */
  draftReady: boolean
  /** Existing account verification status, independent of draft hydration. */
  draftOwnerVerificationState: "pending" | "ready" | "error"
  /** Retry the existing account check; ownership boundaries remain enforced. */
  retryDraftOwnerVerification: () => void
  draftOwnerKey: string | null
  isDraftOwnerCurrent: () => boolean
  /** Clear current persistence; pass the acknowledged attempt for delayed sends. */
  clearDraft: (attemptId?: number) => void
  /** Computed max height actually applied to the textarea. Surfaced for tests. */
  textareaMaxHeight: number
}

const DEFAULT_MAX_HEIGHT_CASUAL = 120
const DEFAULT_MAX_HEIGHT_PRO = 160

export function useComposerText(
  options: UseComposerTextOptions
): UseComposerTextResult {
  const {
    draftKey,
    draftAddress,
    recoverySurface,
    recoverySurfaceTabs = false,
    bindFromDraftKey,
    tabScopedDraft = false,
    textareaRef,
    isProMode = false,
    maxHeight: explicitMaxHeight,
    getDraftMetadata,
    restoreWithMetadata,
    draftEnabled = true
  } = options

  const form = useSimpleForm<{ message: string; image: string }>({
    initialValues: { message: "", image: "" }
  })
  const { getValues, getRevision, reset, setValues } = form
  const [promptAssistMutation, setPromptAssistMutation] =
    React.useState<ComposerPromptAssistMutation>({
      revision: 0,
      source: "owner"
    })
  const [savedAttempt, setSavedAttempt] =
    React.useState<{ id: number; acknowledgedDelivery: boolean } | null>(null)
  const promptAssistSavedAttemptId = savedAttempt?.id ?? null
  const attemptSourceRef = React.useRef<{
    id: number
    storageKey: string
    draftKey: string
    address: ChatDraftAddress
    draftRevision: number
    ownerIsCurrent: () => boolean
    values: { message: string; image: string }
    metadata?: DraftMetadataObject
    messageRevision: number
    imageRevision: number
  } | null>(null)
  const [restoreAttemptId, setRestoreAttemptId] = React.useState<number | null>(
    null
  )
  const imageRevisionRef = React.useRef(0)
  const previousImageRef = React.useRef(form.values.image)
  React.useLayoutEffect(() => {
    if (previousImageRef.current !== form.values.image) {
      previousImageRef.current = form.values.image
      imageRevisionRef.current++
    }
  }, [form.values.image])
  const previousMessageRef = React.useRef(form.values.message)
  const nextPromptAssistAttemptIdRef = React.useRef(0)
  const pendingPromptAssistResetRef = React.useRef<{
    attemptId: number
    fromValue: string
  } | null>(null)

  React.useLayoutEffect(() => {
    if (Object.is(previousMessageRef.current, form.values.message)) return
    const previousMessage = previousMessageRef.current
    previousMessageRef.current = form.values.message
    const pendingReset = pendingPromptAssistResetRef.current
    pendingPromptAssistResetRef.current = null
    setPromptAssistMutation((mutation) =>
      pendingReset &&
      Object.is(pendingReset.fromValue, previousMessage) &&
      form.values.message === ""
        ? {
            revision: mutation.revision + 1,
            source: "optimistic_reset",
            attemptId: pendingReset.attemptId
          }
        : { revision: mutation.revision + 1, source: "owner" }
    )
  }, [form.values.message])

  const setMessageValue = React.useCallback(
    (value: string) => {
      form.setFieldValue("message", value)
    },
    [form]
  )

  const restoreMessage = React.useCallback(
    (value: string, metadata?: DraftMetadataObject) => {
      form.setFieldValue("message", value)
      restoreWithMetadata?.(value, metadata)
    },
    [form, restoreWithMetadata]
  )

  const initialSetup = React.useRef<{
    attempt: InitialChatSetupAttempt
    phase: "captured" | "admitted" | "cleared" | "waiting"
    address: ChatDraftAddress
    draftRevision: number
    inputRevision: number
    values: { message: string; image: string } | null
    metadata?: DraftMetadataObject
    config: TldwConfig | null
    settle: (retained: boolean) => void
  } | null>(null)
  const mounted = React.useRef(true)
  const initialBoundarySeen = React.useRef(false)
  const initialOwnershipCancelled = React.useRef(false)
  const [initialSetupPending, setInitialSetupPending] = React.useState(false)
  const [initialSetupComplete, setInitialSetupComplete] = React.useState(false)
  React.useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; initialSetup.current?.attempt.cancel() }
  }, [])
  const ownSetupEvent = (source?: Event) => initialSetup.current &&
    (source as CustomEvent<{ initialSetupToken?: object }> | undefined)?.detail?.initialSetupToken === initialSetup.current.attempt.token
  const {
    ownerKey,
    requestScope,
    isCurrent,
    hasEverOwned,
    verificationState,
    retryVerification
  } = useChatDraftOwner((source) => {
    const checkpoint = initialSetup.current
    if (checkpoint && ownSetupEvent(source) && checkpoint.phase === "admitted" &&
      getRevision() === checkpoint.inputRevision) checkpoint.phase = "cleared"
    else checkpoint?.attempt.cancel()
    pendingPromptAssistResetRef.current = null
    setSavedAttempt(null)
    attemptSourceRef.current = null
    setRestoreAttemptId(null)
    reset()
    restoreWithMetadata?.("", undefined)
    if (checkpoint && initialSetup.current === checkpoint && checkpoint.phase === "cleared") {
      checkpoint.inputRevision = getRevision()
    }
  }, (_invalidated, source) => {
    initialBoundarySeen.current = true
    if (!ownSetupEvent(source)) initialSetup.current?.attempt.cancel()
  }, () => {
    const checkpoint = initialSetup.current
    if (checkpoint?.phase === "captured" || checkpoint?.phase === "admitted") checkpoint.attempt.cancel()
    if (initialOwnershipCancelled.current) {
      initialOwnershipCancelled.current = false
      reset()
      restoreWithMetadata?.("", undefined)
    }
  })

  const markPromptAssistAttemptSaved = React.useCallback(
    (attemptId: number, acknowledgedDelivery = false) => setSavedAttempt({ id: attemptId, acknowledgedDelivery }), []
  )

  const metadata = React.useCallback(() => ({
    ...getDraftMetadata?.(),
    ...(draftAddress?.kind === "fresh" && ownerKey ? { chatDraftHandle: { address: draftAddress, ownerHash: ownerKey } as unknown as DraftMetadataObject } : {})
  }), [getDraftMetadata, draftAddress, ownerKey])
  const attemptSource = attemptSourceRef.current
  const transition = useStoreMessageOption.getState().chatDraftTransition
  const currentAttemptSource = attemptSource && attemptSource.id === nextPromptAssistAttemptIdRef.current && draftAddress &&
    (draftAddress === attemptSource.address || (transition?.kind === "bind" &&
      transition.sourceAddress === attemptSource.address && transition.destinationKey === chatDraftAddressKey(draftAddress)))
  const preserveEmptyDraft = Boolean(currentAttemptSource &&
    (pendingPromptAssistResetRef.current || promptAssistMutation.source === "optimistic_reset") &&
    !(savedAttempt?.id === attemptSource.id && savedAttempt.acknowledgedDelivery))
  const { draftSaved, draftReady, clearDraft: clearPersistedDraft, listDrafts, retainDraft, acknowledgeBoundDraft, flushDraft } = useDraftPersistence({
    storageKey: `${draftKey}:owner:${ownerKey ?? "unresolved"}`,
    bindFromStorageKey: bindFromDraftKey && ownerKey ? `${bindFromDraftKey}:owner:${ownerKey}` : undefined,
    tabScoped: tabScopedDraft,
    preserveEmptyDraft,
    hasUnpersistedInput: form.values.image.length > 0,
    legacyStorageKey: draftKey,
    isCurrent,
    getValue: () => form.values.message,
    getMetadata: draftAddress ? metadata : getDraftMetadata,
    setValue: (value) => setMessageValue(value),
    setValueWithMetadata: restoreMessage,
    enabled: draftEnabled && ownerKey !== null && !initialSetupPending
  })
  const retainCurrentDraft = React.useCallback(async () => {
    if (!draftReady || !ownerKey || !mounted.current || !isCurrent()) return false
    const source = useStoreMessageOption.getState()
    const revision = getRevision()
    const values = { ...getValues() }
    if (values.image) return false
    const current = () => mounted.current && isCurrent() && getRevision() === revision &&
      (!draftAddress || (useStoreMessageOption.getState().chatDraftAddress === draftAddress &&
        useStoreMessageOption.getState().chatDraftRevision === source.chatDraftRevision))
    if (!current()) return false
    if (!values.message && !values.image) return true
    let savedMetadata: DraftMetadataObject
    try { savedMetadata = structuredClone(metadata()) }
    catch { return false }
    const acknowledged = await retainDraft(`${draftKey}:owner:${ownerKey}`, { content: values.message, metadata: savedMetadata })
    return acknowledged && current()
  }, [draftReady, ownerKey, isCurrent, getRevision, getValues, draftAddress, metadata, retainDraft, draftKey])
  const historySelection = useHistorySelectionContext()
  const setupComposer = React.useRef({ draftAddress, getDraftMetadata, restoreWithMetadata, historySelection })
  React.useLayoutEffect(() => { setupComposer.current = { draftAddress, getDraftMetadata, restoreWithMetadata, historySelection } },
    [draftAddress, getDraftMetadata, restoreWithMetadata, historySelection])
  React.useEffect(() => {
    if (recoverySurface !== "tldw:playgroundChatDraft") return
    const request = (event: Event) => {
      const detail = (event as CustomEvent<SettingsNavigationRequestDetail>).detail
      if (detail?.destination !== "/settings/tldw" || detail.beginInitialSetup) return
      const opening = useStoreMessageOption.getState()
      detail.beginInitialSetup = () => {
        const source = useStoreMessageOption.getState()
        const view = setupComposer.current
        const address = source.chatDraftAddress
        const selection = view.historySelection?.getCurrent()
        const values = getValues()
        if (!mounted.current || initialSetup.current || hasEverOwned() || initialBoundarySeen.current ||
          address !== opening.chatDraftAddress || source.chatDraftRevision !== opening.chatDraftRevision ||
          address !== view.draftAddress || address?.kind !== "fresh" || address.scope.type !== "global" ||
          !selection || selection.owner || selection.view || selection.status !== "idle" ||
          source.historyId || source.serverChatId || source.messages.length || source.history.length || source.queuedMessages.length ||
          source.isLoading || source.isProcessing || source.streaming || (!values.message && !values.image)) return null
        let metadata: DraftMetadataObject | undefined
        try {
          const current = view.getDraftMetadata?.()
          metadata = current === undefined ? undefined : JSON.parse(JSON.stringify(current))
        } catch { return null }
        let historyCurrent = view.historySelection!.fence()
        let intentRevision = usePlaygroundSessionStore.getState().sourceSelectionRevision
        const metadataSignature = JSON.stringify(metadata)
        let resolve!: (retained: boolean) => void
        const completed = new Promise<boolean>(done => { resolve = done })
        const checkpoint = {
          attempt: null as unknown as InitialChatSetupAttempt,
          phase: "captured" as "captured" | "admitted" | "cleared" | "waiting",
          address: address as ChatDraftAddress, draftRevision: source.chatDraftRevision, inputRevision: getRevision(),
          values: values as { message: string; image: string } | null, metadata, config: null as TldwConfig | null,
          settle: (retained: boolean) => {
            if (initialSetup.current !== checkpoint) return
            initialSetup.current = null
            if (!retained && (checkpoint.phase === "captured" || checkpoint.phase === "admitted") && !hasEverOwned()) {
              initialOwnershipCancelled.current = true
            }
            checkpoint.values = null
            checkpoint.metadata = undefined
            checkpoint.config = null
            if (mounted.current) { setInitialSetupPending(false); setInitialSetupComplete(false) }
            resolve(retained)
          }
        }
        const assertCurrent = () => {
          const current = useStoreMessageOption.getState()
          if (!mounted.current || initialSetup.current !== checkpoint || !historyCurrent() ||
            current.chatDraftAddress !== checkpoint.address || current.chatDraftRevision !== checkpoint.draftRevision ||
            getRevision() !== checkpoint.inputRevision ||
            usePlaygroundSessionStore.getState().sourceSelectionRevision !== intentRevision ||
            current.historyId || current.serverChatId || current.messages.length || current.history.length || current.queuedMessages.length ||
            ((checkpoint.phase === "captured" || checkpoint.phase === "admitted") &&
              (hasEverOwned() || JSON.stringify(setupComposer.current.getDraftMetadata?.()) !== metadataSignature))) {
            checkpoint.settle(false)
            throw createServicePromptScopeChangedError()
          }
        }
        checkpoint.attempt = {
          token: {}, assertCurrent,
          admit: () => {
            assertCurrent()
            if (checkpoint.phase !== "captured") throw createServicePromptScopeChangedError()
            usePlaygroundSessionStore.getState().cancelPendingRestore()
            usePlaygroundSessionStore.getState().markSourceSelectionIntent()
            intentRevision = usePlaygroundSessionStore.getState().sourceSelectionRevision
            checkpoint.phase = "admitted"
          },
          afterBoundary: config => {
            const current = useStoreMessageOption.getState()
            const selection = setupComposer.current.historySelection!.getCurrent()
            if (initialSetup.current !== checkpoint || checkpoint.phase !== "cleared" ||
              current.chatDraftAddress !== null || current.chatDraftRevision !== source.chatDraftRevision + 1 ||
              getRevision() !== checkpoint.inputRevision || selection.owner || selection.view ||
              !useStoreMessageOption.getState().setChatDraftAddress(freshChatDraftAddress(address.temporary), {
                expectedRevision: current.chatDraftRevision, sourceAddress: null
              })) {
              checkpoint.settle(false)
              throw createServicePromptScopeChangedError()
            }
            const accepted = useStoreMessageOption.getState()
            checkpoint.address = accepted.chatDraftAddress!
            checkpoint.draftRevision = accepted.chatDraftRevision
            checkpoint.config = config
            historyCurrent = setupComposer.current.historySelection!.fence()
          },
          complete: config => {
            assertCurrent()
            if (checkpoint.phase !== "cleared" || !connectionAuthoritiesMatch(config, checkpoint.config)) {
              checkpoint.settle(false)
              return completed
            }
            checkpoint.phase = "waiting"
            setInitialSetupComplete(true)
            return completed
          },
          cancel: () => checkpoint.settle(false)
        }
        initialSetup.current = checkpoint
        setInitialSetupPending(true)
        return checkpoint.attempt
      }
    }
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, request)
    return () => window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, request)
  }, [recoverySurface, getValues, getRevision, hasEverOwned])
  const draftRevision = useStoreMessageOption(state => state.chatDraftRevision)
  const sourceSelectionRevision = usePlaygroundSessionStore(state => state.sourceSelectionRevision)
  const restoreRevision = usePlaygroundSessionStore(state => state.restoreRevision)
  const renderedAddress = draftAddress ?? useStoreMessageOption.getState().chatDraftAddress
  const captureCompletionGuard = React.useCallback((forFocus = false) => {
    const focused = typeof document !== "undefined" ? document.activeElement : null
    const inputRevision = getRevision()
    const sourceIsCurrent = () => mounted.current && isCurrent() &&
      usePlaygroundSessionStore.getState().sourceSelectionRevision === sourceSelectionRevision &&
      usePlaygroundSessionStore.getState().restoreRevision === restoreRevision
    const source = useStoreMessageOption.getState()
    const beganCurrent = draftReady && renderedAddress !== null && sourceIsCurrent() && source.chatDraftAddress === renderedAddress && source.chatDraftRevision === draftRevision
    return () => {
      const current = useStoreMessageOption.getState(), transition = current.chatDraftTransition
      const sameSource = current.chatDraftAddress === renderedAddress && current.chatDraftRevision === draftRevision
      const ownBind = current.chatDraftAddress !== null && current.chatDraftRevision === draftRevision + 1 && transition?.kind === "bind" &&
        transition.sourceAddress === renderedAddress && transition.destinationKey === chatDraftAddressKey(current.chatDraftAddress)
      return beganCurrent && sourceIsCurrent() && (sameSource || ownBind) &&
        (!forFocus || (focused !== null && typeof document !== "undefined" && focused !== document.body && document.activeElement === focused && getRevision() === inputRevision))
    }
  }, [draftReady, isCurrent, renderedAddress, draftRevision, sourceSelectionRevision, restoreRevision, getRevision])
  React.useLayoutEffect(() => {
    const checkpoint = initialSetup.current
    if (!checkpoint) return
    try { checkpoint.attempt.assertCurrent() } catch { return }
    if (!initialSetupComplete || checkpoint.phase !== "waiting") return
    if (verificationState === "error") { checkpoint.settle(false); return }
    if (!ownerKey || !requestScope || !isCurrent()) return
    const expected = checkpoint.config
    if (!expected || requestScope.config.serverUrl.replace(/\/+$/, "") !== expected.serverUrl.replace(/\/+$/, "") ||
      requestScope.config.authMode !== expected.authMode || (requestScope.config.authSource || "manual") !== "manual" ||
      requestScope.config.expectedSingleUserApiKeyScope !== deriveSingleUserApiKeyCredentialScope(expected.authMode, expected.apiKey)) {
      checkpoint.settle(false)
      return
    }
    const values = checkpoint.values!
    setValues(values)
    setupComposer.current.restoreWithMetadata?.(values.message, checkpoint.metadata)
    checkpoint.settle(true)
  }, [initialSetupComplete, ownerKey, requestScope, isCurrent, verificationState, setValues,
    form.values.message, form.values.image, draftRevision, sourceSelectionRevision, historySelection])
  const [creationStatus, setCreationStatus] = React.useState<{ owner: string | null; address: ChatDraftAddress | undefined; revision: number | null; active: ChatCreationRecovery | null; unavailable: boolean }>({ owner: null, address: undefined, revision: null, active: null, unavailable: false })
  const creationRecovery = creationStatus.owner === ownerKey && creationStatus.address === draftAddress && creationStatus.revision === draftRevision
    ? { active: creationStatus.active, unavailable: creationStatus.unavailable }
    : { active: null, unavailable: false }
  const canonicalCreationHandle = creationRecovery.active ? JSON.stringify(creationRecovery.active.handle) : null
  React.useEffect(() => {
    if (!ownerKey || !draftAddress || draftAddress.kind !== "fresh" || draftAddress.temporary || !isCurrent()) return
    const source = useStoreMessageOption.getState()
    if (source.chatDraftAddress !== draftAddress) return
    let mounted = true
    let generation = 0
    const current = () => mounted && isCurrent() && useStoreMessageOption.getState().chatDraftAddress === source.chatDraftAddress &&
      useStoreMessageOption.getState().chatDraftRevision === source.chatDraftRevision
    const handle = { address: draftAddress, ownerHash: ownerKey }
    const refresh = async () => {
      const token = ++generation
      try {
        const active = await readChatCreationRecovery(handle)
        if (current() && token === generation) setCreationStatus({ owner: ownerKey, address: draftAddress, revision: source.chatDraftRevision, active, unavailable: false })
      } catch {
        if (current() && token === generation) setCreationStatus({ owner: ownerKey, address: draftAddress, revision: source.chatDraftRevision, active: null, unavailable: true })
      }
    }
    const stop = watchChatCreationRecovery([handle, ...(canonicalCreationHandle ? [JSON.parse(canonicalCreationHandle) as ChatDraftHandle] : [])], () => { void refresh() })
    void refresh()
    return () => { mounted = false; generation++; stop() }
  }, [ownerKey, draftAddress, draftRevision, isCurrent, canonicalCreationHandle])
  const creationOperationId = creationRecovery.active?.operationId
  const creationClientChatId = creationRecovery.active?.clientChatId
  const creationRevision = creationStatus.revision
  const copyCreationText = React.useCallback(async () => {
    if (!ownerKey || !draftAddress || draftAddress.kind !== "fresh" || !creationOperationId || !isCurrent()) return false
    const current = () => isCurrent() && useStoreMessageOption.getState().chatDraftAddress === draftAddress &&
      useStoreMessageOption.getState().chatDraftRevision === creationRevision
    if (!current()) return false
    try {
      const attempt = await readChatCreationRecovery({ address: draftAddress, ownerHash: ownerKey })
      if (!current() || attempt?.operationId !== creationOperationId || !attempt.inputText) return false
      await navigator.clipboard.writeText(attempt.inputText)
      return current()
    } catch (error) { if (current()) throw error; return false }
  }, [ownerKey, draftAddress, creationOperationId, creationRevision, isCurrent])
  const creationRenderedHistoryFence = historySelection?.fence()
  const creationRenderedInputRevision = getRevision()
  const reviewCreatedChat = React.useCallback(async () => {
    if (
      !creationRenderedHistoryFence?.() ||
      getRevision() !== creationRenderedInputRevision ||
      !creationOperationId ||
      getValues().image ||
      !ownerKey ||
      !draftAddress ||
      draftAddress.kind !== "fresh" ||
      !historySelection ||
      !isCurrent() ||
      !requestSettingsNavigation("/chat")
    )
      return false
    const inputRevision = getRevision()
    const raw = getValues().message
    const current = () =>
      isCurrent() &&
      useStoreMessageOption.getState().chatDraftAddress === draftAddress &&
      useStoreMessageOption.getState().chatDraftRevision === creationRevision &&
      getRevision() === inputRevision
    if (!current()) return false
    return reviewNativeChatCreation({
      controller: historySelection,
      handle: { address: draftAddress, ownerHash: ownerKey },
      expectedOperationId: creationOperationId,
      expectedClientChatId: creationClientChatId,
      isCurrent: current,
      inputText: raw,
      retainInput: async () =>
        !raw ||
        retainDraft(`${draftKey}:owner:${ownerKey}`, {
          content: raw,
          metadata: metadata()
        }),
      copyInput: async () => {
        if (!current()) return false
        if (raw) await navigator.clipboard.writeText(raw)
        return current()
      },
      onAccepted: (chat) => {
        const state = useStoreMessageOption.getState()
        state.setServerChatId(chat.id)
        state.setServerChatTitle(chat.title ?? "")
        state.setServerChatCharacterId(chat.character_id ?? null)
        state.setServerChatAssistantKind(
          chat.character_id != null ? "character" : null
        )
        state.setServerChatMetaLoaded(true)
        state.setServerChatState(normalizeConversationState(chat.state))
        state.setServerChatSource(chat.source ?? null)
        state.setServerChatVersion(chat.version ?? null)
      }
    })
  }, [
    ownerKey,
    draftAddress,
    creationOperationId,
    creationClientChatId,
    creationRevision,
    creationRenderedHistoryFence,
    creationRenderedInputRevision,
    historySelection,
    isCurrent,
    getValues,
    getRevision,
    retainDraft,
    draftKey,
    metadata
  ])

  const beginPromptAssistReset = React.useCallback(() => {
    flushDraft()
    const attemptId = ++nextPromptAssistAttemptIdRef.current
    const current = useStoreMessageOption.getState()
    let originalMetadata: DraftMetadataObject | undefined
    try {
      const value = getDraftMetadata?.()
      if (value !== undefined)
        originalMetadata = JSON.parse(
          JSON.stringify(value)
        ) as DraftMetadataObject
    } catch {
      // Unavailable metadata must not prevent recovering the original input.
    }
    attemptSourceRef.current =
      ownerKey &&
      draftAddress &&
      current.chatDraftAddress === draftAddress &&
      isCurrent()
        ? {
            id: attemptId,
            storageKey: `${draftKey}:owner:${ownerKey}`,
            draftKey,
            address: draftAddress,
            draftRevision: current.chatDraftRevision,
            ownerIsCurrent: isCurrent,
            values: { ...form.values },
            metadata: originalMetadata,
            messageRevision:
              promptAssistMutation.revision + Number(form.values.message !== ""),
            imageRevision:
              imageRevisionRef.current + Number(form.values.image !== "")
          }
        : null
    setRestoreAttemptId(null)
    pendingPromptAssistResetRef.current = {
      attemptId,
      fromValue: form.values.message
    }
    reset()
    return attemptId
  }, [
    form,
    reset,
    ownerKey,
    draftAddress,
    draftKey,
    flushDraft,
    getDraftMetadata,
    isCurrent,
    promptAssistMutation.revision
  ])

  const restorePromptAssistAttempt = React.useCallback((attemptId: number) => {
    const attempt = attemptSourceRef.current
    if (
      attempt?.id === attemptId &&
      attemptId === nextPromptAssistAttemptIdRef.current &&
      attempt.ownerIsCurrent()
    )
      setRestoreAttemptId(attemptId)
  }, [])

  React.useLayoutEffect(() => {
    const attempt = attemptSourceRef.current
    if (!attempt) return
    const current = useStoreMessageOption.getState()
    const transition = current.chatDraftTransition
    const sameSource =
      current.chatDraftAddress === attempt.address &&
      current.chatDraftRevision === attempt.draftRevision &&
      draftKey === attempt.draftKey
    const acceptedBind =
      current.chatDraftRevision === attempt.draftRevision + 1 &&
      transition?.kind === "bind" &&
      transition.sourceAddress === attempt.address &&
      transition.destinationKey ===
        chatDraftAddressKey(current.chatDraftAddress) &&
      bindFromDraftKey === attempt.draftKey
    if (!sameSource && !acceptedBind) {
      attemptSourceRef.current = null
      return
    }
    if (
      restoreAttemptId !== attempt.id ||
      attempt.id !== nextPromptAssistAttemptIdRef.current ||
      !attempt.ownerIsCurrent() ||
      !draftReady ||
      draftAddress !== current.chatDraftAddress ||
      form.values.message !== "" ||
      form.values.image !== "" ||
      imageRevisionRef.current !== attempt.imageRevision ||
      promptAssistMutation.revision !== attempt.messageRevision ||
      (attempt.values.message !== "" &&
        (promptAssistMutation.source !== "optimistic_reset" ||
          promptAssistMutation.attemptId !== attempt.id))
    )
      return
    attemptSourceRef.current = null
    pendingPromptAssistResetRef.current = null
    setRestoreAttemptId(null)
    form.setValues(attempt.values)
    restoreWithMetadata?.(attempt.values.message, attempt.metadata)
  }, [
    restoreAttemptId,
    draftAddress,
    draftKey,
    bindFromDraftKey,
    draftReady,
    form,
    promptAssistMutation,
    restoreWithMetadata
  ])
  React.useEffect(() => {
    const source = attemptSourceRef.current
    if (!savedAttempt?.acknowledgedDelivery || !source || source.address.kind !== "fresh" || savedAttempt.id !== source.id ||
      savedAttempt.id !== nextPromptAssistAttemptIdRef.current || previousMessageRef.current !== "" || !isCurrent()) return
    const transition = useStoreMessageOption.getState().chatDraftTransition
    if (transition?.kind !== "bind" || !transition.sourceAddress || !draftAddress ||
      transition.destinationKey !== chatDraftAddressKey(draftAddress) ||
      chatDraftAddressKey(transition.sourceAddress) !== chatDraftAddressKey(source.address)) return
    void acknowledgeBoundDraft(source.storageKey)
  }, [savedAttempt, draftAddress, form.values.message, isCurrent, acknowledgeBoundDraft])

  const clearDraft = React.useCallback((attemptId?: number) => {
    if (attemptId !== undefined && (attemptId !== nextPromptAssistAttemptIdRef.current || previousMessageRef.current !== "")) return
    clearPersistedDraft()
  }, [clearPersistedDraft])

  const validRecoverySurface = React.useCallback((surface: string) => Boolean(recoverySurface &&
    (surface === recoverySurface || (recoverySurfaceTabs && surface.startsWith(`${recoverySurface}:`) &&
      surface.slice(recoverySurface.length + 1).length > 0 && !surface.slice(recoverySurface.length + 1).includes(":")))), [recoverySurface, recoverySurfaceTabs])
  const listRetainedDrafts = React.useCallback(async (): Promise<ChatDraftRecoveryListing> => {
    const source = useStoreMessageOption.getState()
    const current = () => draftReady && isCurrent() && source.chatDraftAddress === draftAddress &&
      useStoreMessageOption.getState().chatDraftAddress === source.chatDraftAddress &&
      useStoreMessageOption.getState().chatDraftRevision === source.chatDraftRevision
    if (!ownerKey || !draftAddress || !recoverySurface || !current()) return { drafts: [], unavailable: false }
    const currentScope = draftAddress.kind === "local" ? { type: "global" as const } : draftAddress.scope
    const [result, creations] = await Promise.all([listDrafts(), listChatCreationRecovery(ownerKey, currentScope)])
    if (!current()) return { drafts: [], unavailable: false }
    const suffix = `:owner:${ownerKey}`
    const drafts: RecoveredChatDraft[] = []
    for (const record of result.drafts) {
      const handle = record.metadata?.chatDraftHandle
      if (!isChatDraftHandle(handle) || handle.ownerHash !== ownerKey || handle.address.kind !== "fresh" ||
        (handle.address.scope.type !== currentScope.type || (handle.address.scope.type === "workspace" && currentScope.type === "workspace" && handle.address.scope.workspaceId !== currentScope.workspaceId)) || !record.storageKey.endsWith(suffix)) continue
      const marker = record.storageKey.lastIndexOf(":conversation:")
      if (marker < 0) continue
      const surfaceKey = record.storageKey.slice(0, marker)
      if (!validRecoverySurface(surfaceKey) || record.storageKey !== `${conversationDraftKey(surfaceKey, handle.address)}${suffix}` ||
        (surfaceKey === draftKey.slice(0, draftKey.lastIndexOf(":conversation:")) &&
          chatDraftAddressKey(handle.address) === chatDraftAddressKey(draftAddress))) continue
      drafts.push({ key: record.storageKey, surfaceKey, handle: { ...handle, address: handle.address },
        content: record.content, metadata: record.metadata, updatedAt: record.updatedAt })
    }
    const associations = await Promise.allSettled(drafts.map(draft => draft.handle.address.temporary ? Promise.resolve(null) : readChatCreationRecovery(draft.handle)))
    if (!current()) return { drafts: [], unavailable: false }
    associations.forEach((result, index) => {
      if (result.status === "fulfilled" && result.value) drafts[index].creationAttempt = result.value
    })
    for (const attempt of creations.attempts) {
      if (attempt.handle.address.kind !== "fresh" || (attempt.surface === "playground") !== (recoverySurface === "tldw:playgroundChatDraft") ||
        chatDraftAddressKey(attempt.handle.address) === chatDraftAddressKey(draftAddress)) continue
      const retained = drafts.find(draft => chatDraftAddressKey(draft.handle.address) === chatDraftAddressKey(attempt.handle.address))
      if (retained) retained.creationAttempt = attempt
      else drafts.push({ key: `${conversationDraftKey(recoverySurface, attempt.handle.address)}${suffix}`, surfaceKey: recoverySurface,
        handle: { ...attempt.handle, address: attempt.handle.address }, content: attempt.inputText, updatedAt: attempt.createdAt, creationAttempt: attempt })
    }
    return { drafts: drafts.sort((a, b) => b.updatedAt - a.updatedAt), unavailable: result.unavailable || creations.unavailable || associations.some(result => result.status === "rejected") }
  }, [draftReady, isCurrent, ownerKey, draftAddress, recoverySurface, draftKey, validRecoverySurface, listDrafts])
  const copyRetainedDraft = React.useCallback(async (draft: RecoveredChatDraft, surfaceKey: string, destination?: Extract<ChatDraftAddress, { kind: "fresh" }>) => {
    if (!ownerKey || !validRecoverySurface(surfaceKey)) return false
    const current = (await listRetainedDrafts()).drafts.find(candidate => candidate.key === draft.key &&
      candidate.handle.address.temporary === draft.handle.address.temporary)
    if (!current) return false
    const address = destination ?? current.handle.address
    if (!isChatDraftHandle({ address, ownerHash: ownerKey }) || address.kind !== "fresh" ||
      address.temporary !== current.handle.address.temporary || address.scope.type !== current.handle.address.scope.type ||
      (address.scope.type === "workspace" && current.handle.address.scope.type === "workspace" && address.scope.workspaceId !== current.handle.address.scope.workspaceId)) return false
    if (!(await copyChatCreationRecovery(current.handle, { address, ownerHash: ownerKey }))) return false
    return retainDraft(`${conversationDraftKey(surfaceKey, address)}:owner:${ownerKey}`,
      { content: current.content, metadata: { ...current.metadata, chatDraftHandle: { address, ownerHash: ownerKey } as unknown as DraftMetadataObject } })
  }, [listRetainedDrafts, ownerKey, retainDraft, validRecoverySurface])

  const textareaMaxHeight =
    explicitMaxHeight ??
    (isProMode ? DEFAULT_MAX_HEIGHT_PRO : DEFAULT_MAX_HEIGHT_CASUAL)

  const textAreaFocus = React.useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    if (el.selectionStart === el.selectionEnd) {
      const ua = typeof navigator !== "undefined" ? navigator.userAgent : ""
      const isMobile =
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
          ua
        )
      if (!isMobile) {
        el.focus()
      } else {
        el.blur()
      }
    }
  }, [textareaRef])

  return {
    retainCurrentDraft,
    creationRecovery: { ...creationRecovery, copyInputRequired: Boolean(creationRecovery.active &&
      (creationRecovery.active.state === "settled" || form.values.message !== creationRecovery.active.inputText)) },
    reviewCreatedChat,
    copyCreationText,
    listRetainedDrafts,
    copyRetainedDraft,
    form,
    messageRevision: promptAssistMutation.revision,
    promptAssistMutation,
    beginPromptAssistReset,
    restorePromptAssistAttempt,
    markPromptAssistAttemptSaved,
    promptAssistSavedAttemptId,
    setMessageValue,
    textAreaFocus,
    captureCompletionGuard,
    draftSaved,
    draftReady,
    draftOwnerVerificationState: verificationState,
    retryDraftOwnerVerification: () => {
      retryVerification()
      historySelection?.draftAccount?.retryVerification()
    },
    draftOwnerKey: ownerKey,
    isDraftOwnerCurrent: isCurrent,
    clearDraft,
    textareaMaxHeight
  }
}
