import { isExtensionRuntime } from "@/utils/browser-runtime"

type Translate = (key: string, defaultValue: string) => string

/**
 * In the extension '@' inserts open browser tabs, only when the user turned
 * tab mentions on (CC-11). On the web '@' searches notes (decision [4],
 * note-mentions.ts), so the hint names it there only when the server has
 * Notes.
 */
export const composerMentionsAvailable = (
  tabMentionsEnabled: unknown,
  isExtension: boolean = isExtensionRuntime(),
  notesAvailable = false
): boolean => (isExtension ? tabMentionsEnabled === true : notesAvailable)

/** The composer footer's hint names only the shortcuts that work here. */
export const composerShortcutHint = (
  t: Translate,
  mentionsAvailable: boolean
): string =>
  mentionsAvailable
    ? t("playground:composer.shortcutHint", "/ commands · @ mentions")
    : t("playground:composer.shortcutHintCommands", "/ commands")
