import type { ACPUpdate } from "./types"

export const ACP_MESSAGE_UPDATE_TYPES = new Set([
  "text", "assistant_text", "user_text", "agent_message_chunk", "user_message_chunk"
])

/** Project the event ledger without changing raw updates or joining separate turns. */
export function projectACPUpdates(updates: ACPUpdate[]): ACPUpdate[] {
  const projected: ACPUpdate[] = []
  const tools = new Map<string, number>()
  let previousChunk: number | null = null
  let cancellationShown = false
  for (const update of updates) {
    if (update.type === "prompt_start" || update.type === "history_turn_boundary") {
      if (update.type === "prompt_start") cancellationShown = false
      tools.clear()
      previousChunk = null
      continue
    }
    if (update.type === "prompt_complete") {
      const result = update.data.raw_result as Record<string, unknown> | undefined
      if (result?.stopReason === "cancelled" && !cancellationShown) {
        projected.push({ ...update, type: "cancelled" })
        cancellationShown = true
      }
      tools.clear()
      previousChunk = null
      continue
    }
    if (update.type === "user_text" || update.type === "user_message_chunk") cancellationShown = false
    if (update.type === "cancelled") {
      if (cancellationShown) continue
      cancellationShown = true
    }
    const native = update.data.update as Record<string, unknown> | undefined
    const content = native?.content as Record<string, unknown> | undefined
    const textChunk = ["agent_message_chunk", "user_message_chunk", "agent_thought_chunk"].includes(update.type)
      && content?.type === "text" && typeof content.text === "string"
    if (textChunk && previousChunk !== null) {
      const previous = projected[previousChunk]
      const previousNative = previous.data.update as Record<string, unknown>
      const previousContent = previousNative.content as Record<string, unknown>
      if (previous.type === update.type && previousNative.messageId === native?.messageId) {
        projected[previousChunk] = {
          ...update,
          data: { ...update.data, update: { ...native, content: {
            ...content, text: String(previousContent.text) + content.text
          } } }
        }
        continue
      }
    }
    previousChunk = null
    if ((update.type === "tool_call" || update.type === "tool_call_update") && native) {
      const toolId = native.toolCallId
      if (typeof toolId === "string" && toolId) {
        const index = tools.get(toolId)
        if (index !== undefined) {
          const previous = projected[index]
          projected[index] = { ...update, type: "tool_call", data: {
            ...update.data, update: { ...(previous.data.update as Record<string, unknown>), ...native }
          } }
          continue
        }
        tools.set(toolId, projected.length)
      }
      projected.push({ ...update, type: "tool_call" })
    } else {
      projected.push(update)
    }
    if (textChunk) previousChunk = projected.length - 1
    if (update.type === "error" || update.type === "cancelled" || update.type === "user_text") tools.clear()
  }
  return projected
}

export function countACPUpdateMessages(updates: ACPUpdate[]): number {
  return projectACPUpdates(updates).filter((update) => ACP_MESSAGE_UPDATE_TYPES.has(update.type)).length
}
