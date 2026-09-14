import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import type { AssistantSelection } from "@/types/assistant-selection"
import {
  assistantSelectionToCharacter,
  characterToAssistantSelection,
  getAssistantSelectionMode,
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
  parseSelectedCharacterValue,
  selectedCharacterStorage,
  selectedCharacterSyncStorage
} from "@/utils/selected-character-storage"

type Subscriber = (value: AssistantSelection | null) => void

const selectedAssistantSubscribers = new Set<Subscriber>()
let selectedAssistantOperationRevision = 0
let selectedAssistantCommitChain: Promise<void> = Promise.resolve()

export type SelectedAssistantCommitOptions = {
  isCurrent?: () => boolean
}

const notifySelectedAssistantSubscribers = (value: AssistantSelection | null) => {
  selectedAssistantSubscribers.forEach((subscriber) => {
    subscriber(value)
  })
}

const syncLegacyCharacterSelectionMirror = async (
  selection: AssistantSelection | null
) => {
  if (getAssistantSelectionMode(selection) === "overlay") {
    await selectedCharacterStorage
      .remove(SELECTED_CHARACTER_STORAGE_KEY)
      .catch(() => {})
    await selectedCharacterSyncStorage
      .remove(SELECTED_CHARACTER_STORAGE_KEY)
      .catch(() => {})
    return
  }

  const legacyCharacter =
    assistantSelectionToCharacter<Record<string, unknown>>(selection)

  if (legacyCharacter) {
    await selectedCharacterStorage
      .set(SELECTED_CHARACTER_STORAGE_KEY, legacyCharacter)
      .catch(() => {})
  } else {
    await selectedCharacterStorage
      .remove(SELECTED_CHARACTER_STORAGE_KEY)
      .catch(() => {})
  }

  await selectedCharacterSyncStorage
    .remove(SELECTED_CHARACTER_STORAGE_KEY)
    .catch(() => {})
}

const clearAssistantSyncSelection = async () => {
  await selectedAssistantSyncStorage
    .remove(SELECTED_ASSISTANT_STORAGE_KEY)
    .catch(() => {})
}

export const useSelectedAssistant = (
  initialValue: AssistantSelection | null = null
) => {
  const normalizedInitialValue = React.useMemo(
    () => normalizeAssistantSelection(initialValue),
    [initialValue]
  )
  const storageResult = useStorage<AssistantSelection | null>(
    { key: SELECTED_ASSISTANT_STORAGE_KEY, instance: selectedAssistantStorage },
    normalizedInitialValue
  ) as readonly [
    AssistantSelection | null,
    (value: AssistantSelection | null) => Promise<void> | void,
    | {
        isLoading?: boolean
        setRenderValue?: (value: AssistantSelection | null) => void
      }
    | undefined
  ]
  const [selectedAssistant, setSelectedAssistant, meta] = storageResult
  const migratedRef = React.useRef(false)
  const latestSelectedAssistantRef = React.useRef<AssistantSelection | null>(
    normalizedInitialValue
  )
  const setRenderValueRef = React.useRef(
    meta?.setRenderValue ?? (() => undefined)
  )

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
      latestSelectedAssistantRef.current = value
      setRenderValueRef.current(value)
    }
    selectedAssistantSubscribers.add(subscriber)
    return () => {
      selectedAssistantSubscribers.delete(subscriber)
    }
  }, [])

  const setSelectedAssistantWithBroadcast = React.useCallback(
    async (
      next: AssistantSelection | null,
      options: SelectedAssistantCommitOptions = {}
    ) => {
      const normalizedCurrent = normalizeAssistantSelection(
        parseSelectedAssistantValue(latestSelectedAssistantRef.current)
      )
      const normalizedNext = preserveAssistantSelectionMode(
        normalizeAssistantSelection(next),
        normalizedCurrent
      )
      const operationRevision = ++selectedAssistantOperationRevision
      const isOperationCurrent = () =>
        operationRevision === selectedAssistantOperationRevision
      const isCallerCurrent = () => options.isCurrent?.() ?? true

      const persistSelection = async (
        selection: AssistantSelection | null,
        shouldContinue: () => boolean
      ): Promise<boolean> => {
        latestSelectedAssistantRef.current = selection
        if (!selection) {
          await syncLegacyCharacterSelectionMirror(null)
          if (!shouldContinue()) return false
          await clearAssistantSyncSelection()
          if (!shouldContinue()) return false
          await setSelectedAssistant(null)
          if (!shouldContinue()) return false
          notifySelectedAssistantSubscribers(null)
          return true
        }

        await setSelectedAssistant(selection)
        if (!shouldContinue()) return false
        await clearAssistantSyncSelection()
        if (!shouldContinue()) return false
        await syncLegacyCharacterSelectionMirror(selection)
        if (!shouldContinue()) return false
        notifySelectedAssistantSubscribers(selection)
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
    [setSelectedAssistant]
  )

  React.useEffect(() => {
    if (meta?.isLoading || migratedRef.current) return

    const normalizedLocalSelection = normalizeAssistantSelection(
      parseSelectedAssistantValue(selectedAssistant)
    )
    if (normalizedLocalSelection) {
      migratedRef.current = true
      void clearAssistantSyncSelection()
      void syncLegacyCharacterSelectionMirror(normalizedLocalSelection)
      return
    }

    migratedRef.current = true
    let cancelled = false

    const migrate = async () => {
      try {
        const assistantSyncRaw =
          await selectedAssistantSyncStorage.get<AssistantSelection | null>(
            SELECTED_ASSISTANT_STORAGE_KEY
          )
        const assistantSyncSelection = normalizeAssistantSelection(
          parseSelectedAssistantValue(assistantSyncRaw)
        )
        if (assistantSyncSelection && !cancelled) {
          await setSelectedAssistantWithBroadcast(assistantSyncSelection)
          return
        }

        const legacyLocalRaw =
          await selectedCharacterStorage.get<Record<string, unknown> | null>(
            SELECTED_CHARACTER_STORAGE_KEY
          )
        const legacyLocalSelection = characterToAssistantSelection(
          parseSelectedCharacterValue<Record<string, unknown>>(legacyLocalRaw)
        )
        if (legacyLocalSelection && !cancelled) {
          await setSelectedAssistantWithBroadcast(legacyLocalSelection)
          return
        }

        const legacySyncRaw =
          await selectedCharacterSyncStorage.get<Record<string, unknown> | null>(
            SELECTED_CHARACTER_STORAGE_KEY
          )
        const legacySyncSelection = characterToAssistantSelection(
          parseSelectedCharacterValue<Record<string, unknown>>(legacySyncRaw)
        )
        if (legacySyncSelection && !cancelled) {
          await setSelectedAssistantWithBroadcast(legacySyncSelection)
        }
      } catch {
        // ignore migration failures
      }
    }

    void migrate()
    return () => {
      cancelled = true
    }
  }, [meta?.isLoading, selectedAssistant, setSelectedAssistantWithBroadcast])

  const normalizedSelectedAssistant = React.useMemo(
    () => normalizeAssistantSelection(parseSelectedAssistantValue(selectedAssistant)),
    [selectedAssistant]
  )

  return [
    normalizedSelectedAssistant,
    setSelectedAssistantWithBroadcast,
    {
      isLoading: meta?.isLoading ?? false,
      setRenderValue: meta?.setRenderValue ?? (() => undefined)
    }
  ] as const
}
