import React from "react"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { startSavedChat } from "@/hooks/chat/start-saved-chat"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useStoreMessageOption } from "@/store/option"
import type { Character } from "@/types/character"

/** `startSavedChat` for a surface that has no `clearChat` of its own (the phone focus bar). */
export const useStartSavedChat = () => {
  const clearChat = useClearChat()
  const setTemporaryChat = useStoreMessageOption(
    (state) => state.setTemporaryChat
  )
  const [, setSelectedCharacter] = useSelectedCharacter<Character | null>(null)
  return React.useCallback(
    () => startSavedChat({ clearChat, setTemporaryChat, setSelectedCharacter }),
    [clearChat, setSelectedCharacter, setTemporaryChat]
  )
}
