import { message } from "antd"
import { ChatExportTooLargeError } from "@/services/chat-transcript-export"

const exporting = new Set<string>()

/**
 * Run one chat's export with visible progress (CS-09): "Exporting chat…"
 * while it runs, then whether it worked. A second click on the same chat
 * while it is exporting is ignored. A chat over the reading limit is too
 * large to export here, which no retry fixes, so it isn't offered.
 *
 * `run` resolves `false` when the export didn't happen (the device export
 * reports that rather than throwing).
 */
export const runChatExport = async ({
  chatId,
  t,
  run,
  isCurrent
}: {
  chatId: string
  t: (key: string, fallback: string) => string
  run: () => Promise<unknown>
  /** Whether the place that asked still wants the result shown. */
  isCurrent?: () => boolean
}) => {
  const key = `chat-export:${chatId}`
  if (exporting.has(key)) return
  exporting.add(key)
  message.loading({ key, content: t("common:chatSidebar.exporting", "Exporting chat…"), duration: 0 })
  try {
    const done = await run()
    if (done === false) throw new Error("chat_export_failed")
    if (isCurrent?.() === false) message.destroy(key)
    else message.success({ key, content: t("common:chatSidebar.exportSuccess", "Chat exported.") })
  } catch (error) {
    console.warn("[chat-export] Could not export the chat", error)
    if (isCurrent?.() === false) {
      message.destroy(key)
      return
    }
    message.error({
      key,
      content: error instanceof ChatExportTooLargeError
        ? t("common:chatSidebar.exportTooLarge", "This chat is too large to export here. Use Chatbooks to export it.")
        : t("common:chatSidebar.exportFailed", "Couldn't export this chat. Try again.")
    })
  } finally {
    exporting.delete(key)
  }
}
