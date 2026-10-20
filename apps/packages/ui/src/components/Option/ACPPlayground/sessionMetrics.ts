import type { ACPSession } from "@/services/acp/types"

import { countACPUpdateMessages } from "@/services/acp/update-projection"

export const getSessionMessageCount = (
  session: Pick<ACPSession, "updates" | "messageCount">
): number => {
  const updatesMessageCount = countACPUpdateMessages(session.updates)
  const serverMessageCount = typeof session.messageCount === "number" ? session.messageCount : 0
  return Math.max(updatesMessageCount, serverMessageCount)
}

export const getSessionTokenUsage = (
  session: Pick<ACPSession, "updates" | "usage">
): number | null => {
  let total = 0
  let hasAnyUsage = false

  for (const update of session.updates) {
    const data = update.data as Record<string, unknown>
    const usage = (data.usage || null) as Record<string, unknown> | null

    const totalFromUsage = typeof usage?.total_tokens === "number" ? usage.total_tokens : null
    const totalFromFlat = typeof data.total_tokens === "number" ? data.total_tokens : null

    if (typeof totalFromUsage === "number") {
      total += totalFromUsage
      hasAnyUsage = true
      continue
    }

    if (typeof totalFromFlat === "number") {
      total += totalFromFlat
      hasAnyUsage = true
      continue
    }

    const promptTokens = typeof data.prompt_tokens === "number" ? data.prompt_tokens : 0
    const completionTokens = typeof data.completion_tokens === "number" ? data.completion_tokens : 0
    if (promptTokens > 0 || completionTokens > 0) {
      total += promptTokens + completionTokens
      hasAnyUsage = true
    }
  }

  const serverTotal = typeof session.usage?.total_tokens === "number"
    ? session.usage.total_tokens
    : null

  if (hasAnyUsage && serverTotal !== null) {
    return Math.max(total, serverTotal)
  }
  if (hasAnyUsage) {
    return total
  }
  return serverTotal
}
