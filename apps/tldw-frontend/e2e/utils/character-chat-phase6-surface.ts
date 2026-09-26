export type CharacterChatSessionSurface = {
  isSessionVisible: () => Promise<boolean>
  isFocusMode: () => Promise<boolean>
  exitFocusMode: () => Promise<void>
  restoreDesktopContextRail: () => Promise<void>
  selectCompactContextTab: () => Promise<void>
  getViewportWidth: () => Promise<number>
}

/**
 * Reveals Character Chat sessions through the surface's visible layout controls.
 */
export const revealCharacterChatSessions = async (
  surface: CharacterChatSessionSurface,
): Promise<void> => {
  if (await surface.isFocusMode()) {
    await surface.exitFocusMode()
  }
  if (await surface.isSessionVisible()) return

  if ((await surface.getViewportWidth()) >= 1024) {
    await surface.restoreDesktopContextRail()
  } else {
    await surface.selectCompactContextTab()
  }
}
