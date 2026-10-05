import {
  type DraftMetadataObject,
  useDraftPersistence
} from "@/hooks/useDraftPersistence"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import React from "react"
import { useStoreMessageOption } from "@/store/option"
import { chatDraftAddressKey, conversationDraftKey, isChatDraftHandle, type ChatDraftAddress, type ChatDraftHandle } from "@/utils/chat-draft-address"

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
  /** Marks a specific owner attempt as definitively sent or enqueued. */
  markPromptAssistAttemptSaved: (attemptId: number, acknowledgedDelivery?: boolean) => void
  /** Most recently completed owner attempt, or null before the first completion. */
  promptAssistSavedAttemptId: number | null
  /** Convenience setter for the message field. */
  setMessageValue: (value: string) => void
  /** Focuses the textarea (no-op on mobile to avoid unwanted keyboard pop-up). */
  textAreaFocus: () => void
  /** Whether the draft was recently persisted — for the "Draft saved" indicator. */
  draftSaved: boolean
  /** Whether this owner's saved draft has finished restoring. */
  draftReady: boolean
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
  const [promptAssistMutation, setPromptAssistMutation] =
    React.useState<ComposerPromptAssistMutation>({
      revision: 0,
      source: "owner"
    })
  const [savedAttempt, setSavedAttempt] =
    React.useState<{ id: number; acknowledgedDelivery: boolean } | null>(null)
  const promptAssistSavedAttemptId = savedAttempt?.id ?? null
  const attemptSourceRef = React.useRef<{ id: number; storageKey: string; address: ChatDraftAddress } | null>(null)
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

  const { ownerKey, isCurrent } = useChatDraftOwner(() => {
    pendingPromptAssistResetRef.current = null
    setSavedAttempt(null)
    attemptSourceRef.current = null
    form.reset()
    restoreWithMetadata?.("", undefined)
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
    legacyStorageKey: draftKey,
    isCurrent,
    getValue: () => form.values.message,
    getMetadata: draftAddress ? metadata : getDraftMetadata,
    setValue: (value) => setMessageValue(value),
    setValueWithMetadata: restoreMessage,
    enabled: draftEnabled && ownerKey !== null
  })
  const beginPromptAssistReset = React.useCallback(() => {
    flushDraft()
    const attemptId = ++nextPromptAssistAttemptIdRef.current
    attemptSourceRef.current = ownerKey && draftAddress?.kind === "fresh"
      ? { id: attemptId, storageKey: `${draftKey}:owner:${ownerKey}`, address: draftAddress } : null
    pendingPromptAssistResetRef.current = { attemptId, fromValue: form.values.message }
    form.reset()
    return attemptId
  }, [form, ownerKey, draftAddress, draftKey, flushDraft])
  React.useEffect(() => {
    const source = attemptSourceRef.current
    if (!savedAttempt?.acknowledgedDelivery || !source || savedAttempt.id !== source.id ||
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
    const result = await listDrafts()
    if (!current()) return { drafts: [], unavailable: false }
    const currentScope = draftAddress.kind === "local" ? { type: "global" as const } : draftAddress.scope
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
    return { drafts: drafts.sort((a, b) => b.updatedAt - a.updatedAt), unavailable: result.unavailable }
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
    listRetainedDrafts,
    copyRetainedDraft,
    form,
    messageRevision: promptAssistMutation.revision,
    promptAssistMutation,
    beginPromptAssistReset,
    markPromptAssistAttemptSaved,
    promptAssistSavedAttemptId,
    setMessageValue,
    textAreaFocus,
    draftSaved,
    draftReady,
    clearDraft,
    textareaMaxHeight
  }
}
