import Dexie from "dexie"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { PageAssitDatabase as ChromeDB } from "../index"
import { historyDigest } from "./history-selection"
import type { HistoryInfo, LegacyChatSource } from "./types"

export async function withLegacyChatStorageLock<T>(
  operation: (storage?: ChromeDB) => Promise<T>
): Promise<T> {
  if (!isExtensionRuntime()) return operation()
  if (typeof chrome === "undefined" || !chrome.storage?.local ||
      typeof navigator === "undefined" || !navigator.locks?.request) {
    throw new Error("Legacy chat storage is unavailable")
  }
  return navigator.locks.request("tldw:legacy-chat-storage", () => operation(new ChromeDB()))
}

export function withLegacyChatErasure(
  operation: (storage: ChromeDB | undefined, waitForErase: (erasure: Promise<void>) => Promise<void>) => Promise<void>
): Promise<void> {
  return withLegacyChatStorageLock(async (storage) => {
    let pending: Promise<void> | undefined
    try {
      await operation(storage, (erasure) => {
        pending = erasure
        return Dexie.waitFor(erasure)
      })
    } finally {
      // A Dexie timeout does not cancel the Chrome operation.
      await pending?.catch(() => {})
    }
  })
}

export async function eraseLegacyChatSource(storage: ChromeDB, history: HistoryInfo) {
  const receipt: LegacyChatSource | undefined = history.tldw_history_legacy_source
  if (!receipt) return
  const histories = await storage.getChatHistories()
  const candidates = histories.filter((candidate) => candidate.id === history.id)
  if (!candidates.length || candidates.some((candidate) => historyDigest(candidate) !== receipt.history)) return

  const messages = await storage.getStoredChatMessages(history.id)
  if (messages !== undefined && historyDigest(messages) !== receipt.messages) return
  const files = await storage.getSessionFilesInfo(history.id)
  if (files !== null && historyDigest(files) !== receipt.files) return

  // An empty array cannot distinguish a message key from another storage record.
  const hasMessages = !!messages?.length && messages.every((message) => typeof message.role === "string")
  await storage.deleteChatHistory(history.id, histories, hasMessages)
}
