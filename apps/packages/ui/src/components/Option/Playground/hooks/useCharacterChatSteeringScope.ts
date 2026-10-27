import React from "react"

import { useStoreMessageOption } from "@/store/option"
import {
  hasActiveMessageSteering,
  resolveMessageSteering
} from "@/utils/message-steering"

/**
 * Keep role-play steering ("Continue as user", "Impersonate user", "Force
 * narrate") inside character chats.
 *
 * The reply's More menu offers steering, and its Clear action, only while a
 * character chat is active (CM-09). The steering intent itself is page-wide
 * option-store state that every send applies, and a chat switch or turning on
 * compare does not reset it. Without this hook, steering armed in a character
 * chat would shape the next reply in a plain or compare chat, where the user
 * can neither see nor clear it.
 *
 * @param characterChatActive - true while the Playground shows a character
 *   chat whose steering controls are on screen.
 */
export const useCharacterChatSteeringScope = (
  characterChatActive: boolean
): void => {
  // The same test the send path uses to decide whether to apply steering.
  const steeringArmed = useStoreMessageOption((state) =>
    hasActiveMessageSteering(
      resolveMessageSteering({
        mode: state.messageSteeringMode,
        forceNarrate: state.messageSteeringForceNarrate
      })
    )
  )
  React.useEffect(() => {
    if (characterChatActive || !steeringArmed) return
    useStoreMessageOption.getState().clearMessageSteering()
  }, [characterChatActive, steeringArmed])
}
