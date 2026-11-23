type StartSavedChatDeps = {
  clearChat: () => boolean | void
  setTemporaryChat: (temporary: boolean) => void
  setSelectedCharacter: (character: null) => unknown
}

/**
 * The header's "New chat": clear to a fresh saved chat with no character.
 * Returns false when the clear was refused (an unsaved temporary chat the
 * user chose to keep, or a settings page that blocked leaving).
 *
 * Shared by the header, the phone focus bar and the composer's `/new`
 * (CO-01, CC-11). Kept free of hook imports so a caller that already holds
 * `clearChat` does not pull in the clear-chat module graph.
 */
export const startSavedChat = ({
  clearChat,
  setTemporaryChat,
  setSelectedCharacter
}: StartSavedChatDeps): boolean => {
  if (clearChat() === false) return false
  setTemporaryChat(false)
  void setSelectedCharacter(null)
  return true
}
