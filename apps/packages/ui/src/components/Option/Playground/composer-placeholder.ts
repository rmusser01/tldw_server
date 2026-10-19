import { isExtensionRuntime } from "@/utils/browser-runtime"

type Translate = (key: string, defaultValue: string) => string

/**
 * '@' mentions insert open browser tabs, which only the extension can read
 * and only when the user turned tab mentions on (CC-11). Anywhere else the
 * composer must not promise them.
 */
export const composerMentionsAvailable = (
  tabMentionsEnabled: unknown,
  isExtension: boolean = isExtensionRuntime()
): boolean => isExtension && tabMentionsEnabled === true

export const composerPlaceholder = (
  t: Translate,
  mentionsAvailable: boolean
): string =>
  mentionsAvailable
    ? t(
        "playground:composer.placeholderWithMentions",
        "Type a message... (/ commands, @ mentions)"
      )
    : t(
        "playground:composer.placeholderCommands",
        "Type a message... (/ for commands)"
      )
