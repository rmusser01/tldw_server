import type { ServerChatSaveStatus } from "@/store/server-chat-save-status"

/**
 * Where a chat is saved, derived only from acknowledged state (CS-03 / XS-05,
 * #3104):
 * - "temporary": a temporary chat, which is not saved;
 * - "local": saved on this device only (no server chat id);
 * - "serverSaving": a server chat whose latest write is still in flight;
 * - "serverFailed": a server chat whose latest write was not acknowledged;
 * - "server": a server chat whose latest write was acknowledged, or that was
 *   opened from the server and has not been written to since.
 *
 * Server connectivity is deliberately not an input: being connected does not
 * mean a chat has been saved there.
 */
export const CHAT_PERSISTENCE_KINDS = [
  "temporary",
  "local",
  "serverSaving",
  "serverFailed",
  "server"
] as const

export type ChatPersistenceKind = (typeof CHAT_PERSISTENCE_KINDS)[number]

export type ChatPersistenceCopy = {
  /** Short status, e.g. for a badge or the first line of a tooltip. */
  pill: string
  /** One sentence explaining where the chat is saved. */
  description: string
}

// i18next's overloaded TFunction is only assignable to a loosely typed
// signature, so the default-value parameter stays `any`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Translate = (key: string, defaultValue?: any) => unknown

export const resolveChatPersistenceKind = ({
  temporaryChat,
  serverChatId,
  serverSaveStatus
}: {
  temporaryChat: boolean
  serverChatId: string | null | undefined
  serverSaveStatus: ServerChatSaveStatus
}): ChatPersistenceKind => {
  if (temporaryChat) return "temporary"
  if (typeof serverChatId !== "string" || serverChatId.trim().length === 0) {
    return "local"
  }
  if (serverSaveStatus === "saving") return "serverSaving"
  if (serverSaveStatus === "failed") return "serverFailed"
  return "server"
}

const translate = (t: Translate, key: string, defaultValue: string): string => {
  const value = t(`playground:composer.persistence.${key}`, defaultValue)
  return typeof value === "string" ? value : defaultValue
}

export const getChatPersistenceCopy = (
  t: Translate,
  kind: ChatPersistenceKind
): ChatPersistenceCopy => {
  switch (kind) {
    case "temporary":
      return {
        pill: translate(t, "ephemeralPill", "Temporary"),
        description: translate(
          t,
          "ephemeral",
          "Temporary chat: not saved in history and cleared when you close this window."
        )
      }
    case "serverSaving":
      return {
        pill: translate(t, "serverSavingPill", "Saving to server…"),
        description: translate(
          t,
          "serverSaving",
          "Saved on this device. Waiting for your tldw server to confirm the latest changes."
        )
      }
    case "serverFailed":
      return {
        pill: translate(t, "serverFailedPill", "Couldn't save to server"),
        description: translate(
          t,
          "serverFailed",
          "Saved on this device. Your tldw server didn't confirm the latest changes."
        )
      }
    case "server":
      return {
        pill: translate(t, "serverPill", "Saved on server"),
        description: translate(
          t,
          "server",
          "Saved on your tldw server and on this device."
        )
      }
    case "local":
    default:
      return {
        pill: translate(t, "localPill", "Saved on this device"),
        description: translate(
          t,
          "local",
          "Saved on this device only. This chat is not on your tldw server."
        )
      }
  }
}
