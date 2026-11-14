/**
 * Recency groups for the chat history list (CS-08): Today, This week (the
 * previous seven days) and Older. The list arrives newest first, so grouping
 * keeps each chat's order.
 */
export type ChatRecencyGroup = "today" | "thisWeek" | "older"

const DAY_MS = 24 * 60 * 60 * 1000

const startOfDay = (date: Date) =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()

export const getChatRecencyGroup = (
  timestampMs: number | null | undefined,
  now: Date = new Date()
): ChatRecencyGroup => {
  if (typeof timestampMs !== "number" || !Number.isFinite(timestampMs)) return "older"
  const today = startOfDay(now)
  if (timestampMs >= today) return "today"
  if (timestampMs >= today - 6 * DAY_MS) return "thisWeek"
  return "older"
}

export const CHAT_RECENCY_GROUP_LABELS: Record<
  ChatRecencyGroup,
  { key: string; defaultLabel: string }
> = {
  today: { key: "common:chatSidebar.groups.today", defaultLabel: "Today" },
  thisWeek: { key: "common:chatSidebar.groups.thisWeek", defaultLabel: "This week" },
  older: { key: "common:chatSidebar.groups.older", defaultLabel: "Older" }
}

export const groupChatsByRecency = <
  T extends { updatedAtMs?: number | null; createdAtMs?: number | null }
>(
  chats: readonly T[],
  now: Date = new Date()
): Array<{ group: ChatRecencyGroup; chats: T[] }> => {
  const order: ChatRecencyGroup[] = ["today", "thisWeek", "older"]
  const buckets = new Map<ChatRecencyGroup, T[]>()
  for (const chat of chats) {
    const group = getChatRecencyGroup(chat.updatedAtMs ?? chat.createdAtMs, now)
    const bucket = buckets.get(group) ?? []
    bucket.push(chat)
    buckets.set(group, bucket)
  }
  return order
    .filter((group) => buckets.has(group))
    .map((group) => ({ group, chats: buckets.get(group) as T[] }))
}
