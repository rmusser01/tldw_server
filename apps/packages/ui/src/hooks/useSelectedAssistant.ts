import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import type { AssistantSelection } from "@/types/assistant-selection"
import {
  normalizeAssistantSelection,
  preserveAssistantSelectionMode
} from "@/types/assistant-selection"
import {
  SELECTED_ASSISTANT_STORAGE_KEY,
  parseSelectedAssistantValue,
  selectedAssistantStorage,
  selectedAssistantSyncStorage
} from "@/utils/selected-assistant-storage"
import {
  SELECTED_CHARACTER_STORAGE_KEY,
  selectedCharacterStorage,
  selectedCharacterSyncStorage
} from "@/utils/selected-character-storage"
import React from "react"

import { useStorage } from "@plasmohq/storage/hook"

type OwnedSelection = { ownerKey: string; selection: AssistantSelection | null }
type Subscriber = (value: OwnedSelection) => void

// Plasmo does not reset isLoading when a key changes. Stamp even an empty read
// so readiness proves the current owner's key has actually finished loading.
const ownedAssistantStorage = {
  get: async (key: string) => {
    const saved = await selectedAssistantStorage.get<OwnedSelection>(key)
    const ownerKey = key.slice(
      `${SELECTED_ASSISTANT_STORAGE_KEY}:owner:`.length
    )
    return saved?.ownerKey === ownerKey ? saved : { ownerKey, selection: null }
  },
  set: (key: string, value: OwnedSelection) =>
    selectedAssistantStorage.set(key, value),
  remove: (key: string) => selectedAssistantStorage.remove(key),
  watch: (callbacks: Parameters<typeof selectedAssistantStorage.watch>[0]) =>
    selectedAssistantStorage.watch(callbacks),
  unwatch: (
    callbacks: Parameters<typeof selectedAssistantStorage.unwatch>[0]
  ) => selectedAssistantStorage.unwatch(callbacks)
} as typeof selectedAssistantStorage

const selectedAssistantSubscribers = new Set<Subscriber>()
let selectedAssistantOperationRevision = 0
let selectedAssistantCommitChain: Promise<void> = Promise.resolve()
// Storage watches may precede ACK; keep the accepted baseline for this queue.
const pendingSelectedAssistantWrites = new Map<
  string,
  {
    selection: AssistantSelection | null
    count: number
    revision: number
    dirty: boolean
  }
>()

export const getSelectedAssistantOperationRevision = (): number =>
  selectedAssistantOperationRevision

export type SelectedAssistantCommitOptions = {
  isCurrent?: () => boolean
}

const notifySelectedAssistantSubscribers = (value: OwnedSelection) => {
  selectedAssistantSubscribers.forEach((subscriber) => {
    subscriber(value)
  })
}

export const useSelectedAssistant = (
  initialValue: AssistantSelection | null = null
) => {
  const { ownerKey, isCurrent, verificationState, retryVerification } =
    useChatDraftOwner(() => {
      // Invalidate queued writers before another principal can use this hook.
      selectedAssistantOperationRevision++
    })
  const assistantKey = `${SELECTED_ASSISTANT_STORAGE_KEY}:owner:${ownerKey ?? "unresolved"}`
  const normalizedInitialValue = React.useMemo(
    () => normalizeAssistantSelection(initialValue),
    [initialValue]
  )
  const storageResult = useStorage<OwnedSelection | null>(
    { key: assistantKey, instance: ownedAssistantStorage },
    null
  ) as readonly [
    OwnedSelection | null,
    (value: OwnedSelection | null) => Promise<void> | void,
    (
      | {
          isLoading?: boolean
          setRenderValue?: (value: OwnedSelection | null) => void
        }
      | undefined
    )
  ]
  const [record, , meta] = storageResult
  // A keyed storage hook may briefly retain its previous key's render value.
  const selectedAssistant =
    isCurrent() && record?.ownerKey === ownerKey
      ? ownerKey && pendingSelectedAssistantWrites.has(ownerKey)
        ? pendingSelectedAssistantWrites.get(ownerKey)!.selection
        : record.selection
      : null
  const latestSelectedAssistantRef = React.useRef<AssistantSelection | null>(
    normalizedInitialValue
  )
  const setRenderValueRef = React.useRef(
    meta?.setRenderValue ?? (() => undefined)
  )

  React.useEffect(() => {
    // Earlier releases saved private payloads without ownership. Never adopt them.
    void Promise.all([
      selectedAssistantStorage.remove(SELECTED_ASSISTANT_STORAGE_KEY),
      selectedAssistantSyncStorage.remove(SELECTED_ASSISTANT_STORAGE_KEY),
      selectedCharacterStorage.remove(SELECTED_CHARACTER_STORAGE_KEY),
      selectedCharacterSyncStorage.remove(SELECTED_CHARACTER_STORAGE_KEY)
    ]).catch(() => {})
  }, [])

  React.useEffect(() => {
    setRenderValueRef.current = meta?.setRenderValue ?? (() => undefined)
  }, [meta?.setRenderValue])

  React.useEffect(() => {
    latestSelectedAssistantRef.current = normalizeAssistantSelection(
      parseSelectedAssistantValue(selectedAssistant)
    )
  }, [selectedAssistant])

  React.useEffect(() => {
    const subscriber: Subscriber = (value) => {
      if (!isCurrent() || value.ownerKey !== ownerKey) return
      latestSelectedAssistantRef.current = value.selection
      setRenderValueRef.current(value)
    }
    selectedAssistantSubscribers.add(subscriber)
    return () => {
      selectedAssistantSubscribers.delete(subscriber)
    }
  }, [ownerKey, isCurrent])

  const setSelectedAssistantWithBroadcast = React.useCallback(
    async (
      next: AssistantSelection | null,
      options: SelectedAssistantCommitOptions = {}
    ): Promise<boolean> => {
      if (options.isCurrent && !options.isCurrent()) return false
      if (!ownerKey || !isCurrent()) {
        if (next)
          throw new Error(
            "Wait for account verification before selecting an assistant."
          )
        return false
      }
      const normalizedCurrent = normalizeAssistantSelection(
        parseSelectedAssistantValue(
          pendingSelectedAssistantWrites.has(ownerKey)
            ? pendingSelectedAssistantWrites.get(ownerKey)!.selection
            : latestSelectedAssistantRef.current
        )
      )
      const normalizedNext = preserveAssistantSelectionMode(
        normalizeAssistantSelection(next),
        normalizedCurrent
      )
      const writeState = pendingSelectedAssistantWrites.get(ownerKey) ?? {
        selection: normalizedCurrent,
        count: 0,
        revision: 0,
        dirty: false
      }
      writeState.count++
      pendingSelectedAssistantWrites.set(ownerKey, writeState)
      const operationRevision = ++selectedAssistantOperationRevision
      writeState.revision = operationRevision
      const isOperationCurrent = () =>
        operationRevision === selectedAssistantOperationRevision && isCurrent()
      const isCallerCurrent = () => options.isCurrent?.() ?? true

      const persistSelection = async (
        selection: AssistantSelection | null,
        shouldContinue: () => boolean
      ): Promise<boolean> => {
        // The hook setter publishes optimistically before storage acknowledges it.
        writeState.dirty = true
        await ownedAssistantStorage.set(assistantKey, { ownerKey, selection })
        if (!shouldContinue()) return false
        writeState.dirty = false
        writeState.selection = selection
        latestSelectedAssistantRef.current = selection
        notifySelectedAssistantSubscribers({ ownerKey, selection })
        return true
      }

      // Compensate only this owned key's latest writer. Retired account work may
      // restore its private source record; it cannot claim a current chat owner.
      const canRestore = () =>
        pendingSelectedAssistantWrites.get(ownerKey) === writeState &&
        writeState.revision === operationRevision
      const restoreAccepted = async () => {
        if (!writeState.dirty || !canRestore()) return
        try {
          await persistSelection(normalizedCurrent, canRestore)
        } catch (error) {
          if (canRestore()) {
            latestSelectedAssistantRef.current = normalizedCurrent
            notifySelectedAssistantSubscribers({
              ownerKey,
              selection: normalizedCurrent
            })
          }
          throw new Error("assistant_selection_restore_failed", {
            cause: error
          })
        }
      }
      const commit = async (): Promise<boolean> => {
        if (!isOperationCurrent() || !isCallerCurrent()) {
          await restoreAccepted()
          return false
        }
        let committed: boolean
        try {
          committed = await persistSelection(
            normalizedNext,
            () => isOperationCurrent() && isCallerCurrent()
          )
        } catch (error) {
          await restoreAccepted()
          throw error
        }
        if (!committed) await restoreAccepted()
        return committed
      }
      const operation = selectedAssistantCommitChain.then(commit, commit)
      selectedAssistantCommitChain = operation.then(
        () => undefined,
        () => undefined
      )
      try {
        return (await operation) && isOperationCurrent() && isCallerCurrent()
      } finally {
        // ponytail: failed rollback stays in RAM until retry; durable recovery needs a transactional store.
        if (--writeState.count === 0 && !writeState.dirty)
          pendingSelectedAssistantWrites.delete(ownerKey)
      }
    },
    [assistantKey, ownerKey, isCurrent]
  )

  const normalizedSelectedAssistant = React.useMemo(
    () =>
      normalizeAssistantSelection(
        parseSelectedAssistantValue(selectedAssistant)
      ),
    [selectedAssistant]
  )
  const readSelection = React.useCallback(async () => {
    if (!ownerKey || !isCurrent()) return null
    const pending = pendingSelectedAssistantWrites.get(ownerKey)
    if (pending)
      return normalizeAssistantSelection(
        parseSelectedAssistantValue(pending.selection)
      )
    const revision = selectedAssistantOperationRevision
    const saved =
      await selectedAssistantStorage.get<OwnedSelection>(assistantKey)
    if (
      !isCurrent() ||
      revision !== selectedAssistantOperationRevision ||
      saved?.ownerKey !== ownerKey
    )
      return null
    return normalizeAssistantSelection(
      parseSelectedAssistantValue(saved.selection)
    )
  }, [ownerKey, isCurrent, assistantKey])

  return [
    normalizedSelectedAssistant,
    setSelectedAssistantWithBroadcast,
    {
      isLoading:
        !ownerKey ||
        record?.ownerKey !== ownerKey ||
        (meta?.isLoading ?? false),
      verificationState,
      retryVerification,
      ownerKey: isCurrent() ? ownerKey : null,
      assistantKey: ownerKey ? assistantKey : null,
      isCurrent,
      readSelection,
      setRenderValue: (selection: AssistantSelection | null) => {
        if (ownerKey && isCurrent())
          meta?.setRenderValue?.({ ownerKey, selection })
      }
    }
  ] as const
}
