import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
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

type OwnedSelection = { ownerKey: string; selection: AssistantSelection | null }
type Subscriber = (value: OwnedSelection) => void

// Plasmo does not reset isLoading when a key changes. Stamp even an empty read
// so readiness proves the current owner's key has actually finished loading.
const ownedAssistantStorage = {
  get: async (key: string) => {
    const saved = await selectedAssistantStorage.get<OwnedSelection>(key)
    const ownerKey = key.slice(`${SELECTED_ASSISTANT_STORAGE_KEY}:owner:`.length)
    return saved?.ownerKey === ownerKey ? saved : { ownerKey, selection: null }
  },
  set: (key: string, value: OwnedSelection) => selectedAssistantStorage.set(key, value),
  remove: (key: string) => selectedAssistantStorage.remove(key),
  watch: (callbacks: Parameters<typeof selectedAssistantStorage.watch>[0]) => selectedAssistantStorage.watch(callbacks),
  unwatch: (callbacks: Parameters<typeof selectedAssistantStorage.unwatch>[0]) => selectedAssistantStorage.unwatch(callbacks)
} as typeof selectedAssistantStorage

const selectedAssistantSubscribers = new Set<Subscriber>()
let selectedAssistantOperationRevision = 0
let selectedAssistantCommitChain: Promise<void> = Promise.resolve()

export const getSelectedAssistantOperationRevision = (): number => selectedAssistantOperationRevision

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
  const { ownerKey, isCurrent } = useChatDraftOwner(() => {
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
    | {
        isLoading?: boolean
        setRenderValue?: (value: OwnedSelection | null) => void
      }
    | undefined
  ]
  const [record, setSelectedAssistant, meta] = storageResult
  // A keyed storage hook may briefly retain its previous key's render value.
  const selectedAssistant = isCurrent() && record?.ownerKey === ownerKey
    ? record.selection
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
    ) => {
      if (options.isCurrent && !options.isCurrent()) return
      if (!ownerKey || !isCurrent()) {
        if (next) throw new Error("Wait for account verification before selecting an assistant.")
        return
      }
      const normalizedCurrent = normalizeAssistantSelection(
        parseSelectedAssistantValue(latestSelectedAssistantRef.current)
      )
      const normalizedNext = preserveAssistantSelectionMode(
        normalizeAssistantSelection(next),
        normalizedCurrent
      )
      const operationRevision = ++selectedAssistantOperationRevision
      const isOperationCurrent = () =>
        operationRevision === selectedAssistantOperationRevision && isCurrent()
      const isCallerCurrent = () => options.isCurrent?.() ?? true

      const persistSelection = async (
        selection: AssistantSelection | null,
        shouldContinue: () => boolean
      ): Promise<boolean> => {
        latestSelectedAssistantRef.current = selection
        await setSelectedAssistant({ ownerKey, selection })
        if (!shouldContinue()) return false
        notifySelectedAssistantSubscribers({ ownerKey, selection })
        return true
      }

      const commit = async () => {
        if (!isOperationCurrent() || !isCallerCurrent()) return
        const committed = await persistSelection(
          normalizedNext,
          () => isOperationCurrent() && isCallerCurrent()
        )
        if (committed || !isOperationCurrent() || isCallerCurrent()) return

        await persistSelection(normalizedCurrent, isOperationCurrent)
      }
      const operation = selectedAssistantCommitChain.then(commit, commit)
      selectedAssistantCommitChain = operation.catch(() => undefined)
      await operation
    },
    [setSelectedAssistant, ownerKey, isCurrent]
  )

  const normalizedSelectedAssistant = React.useMemo(
    () => normalizeAssistantSelection(parseSelectedAssistantValue(selectedAssistant)),
    [selectedAssistant]
  )
  const readSelection = React.useCallback(async () => {
    if (!ownerKey || !isCurrent()) return null
    const saved = await selectedAssistantStorage.get<OwnedSelection>(assistantKey)
    if (!isCurrent() || saved?.ownerKey !== ownerKey) return null
    return normalizeAssistantSelection(parseSelectedAssistantValue(saved.selection))
  }, [ownerKey, isCurrent, assistantKey])

  return [
    normalizedSelectedAssistant,
    setSelectedAssistantWithBroadcast,
    {
      isLoading: !ownerKey || record?.ownerKey !== ownerKey || (meta?.isLoading ?? false),
      assistantKey: ownerKey ? assistantKey : null,
      isCurrent,
      readSelection,
      setRenderValue: (selection: AssistantSelection | null) => {
        if (ownerKey && isCurrent()) meta?.setRenderValue?.({ ownerKey, selection })
      }
    }
  ] as const
}
