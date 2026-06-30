import React from "react"
import type { Character } from "@/types/character"
import {
  assistantSelectionToCharacter,
  characterToAssistantSelection,
  getAssistantSelectionMode
} from "@/types/assistant-selection"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"

type StoredCharacter = Character

export const useSelectedCharacter = <T = StoredCharacter>(
  initialValue: T | null = null
) => {
  const initialAssistantSelection = React.useMemo(
    () =>
      characterToAssistantSelection(
        initialValue as (StoredCharacter & Record<string, unknown>) | null
      ),
    [initialValue]
  )
  const [selectedAssistant, setSelectedAssistant, meta] = useSelectedAssistant(
    initialAssistantSelection
  )

  const selectedCharacter = React.useMemo(
    () => assistantSelectionToCharacter<T>(selectedAssistant),
    [selectedAssistant]
  )
  const selectedCharacterMode = React.useMemo(
    () => getAssistantSelectionMode(selectedAssistant),
    [selectedAssistant]
  )
  const setSelectedCharacterWithBroadcast = React.useCallback(
    async (next: T | null) => {
      const nextSelection = characterToAssistantSelection(
        next as (StoredCharacter & Record<string, unknown>) | null
      )
      const nextSelectionMode = getAssistantSelectionMode(nextSelection)
      if (nextSelection && selectedCharacterMode && !nextSelectionMode) {
        nextSelection.metadata = {
          ...(nextSelection.metadata ?? {}),
          selectionMode: selectedCharacterMode
        }
      }
      await setSelectedAssistant(nextSelection)
    },
    [selectedCharacterMode, setSelectedAssistant]
  )

  return [selectedCharacter, setSelectedCharacterWithBroadcast, meta] as const
}
