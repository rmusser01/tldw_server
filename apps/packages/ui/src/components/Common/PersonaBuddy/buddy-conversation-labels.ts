import type { BuddyConversationSummary } from "@/services/buddies"

/** Disambiguate loaded conversations without changing their saved titles or IDs. */
export const buddyConversationLabels = (
  conversations: BuddyConversationSummary[],
  locale?: string
): Map<string, string> => {
  const groups = new Map<string, BuddyConversationSummary[]>()
  for (const conversation of conversations) {
    const key = conversation.title.trim()
    groups.set(key, [...(groups.get(key) ?? []), conversation])
  }
  const labels = new Map<string, string>()
  const formatter = new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  })
  for (const group of groups.values()) {
    const dates = group.map((conversation) => {
      if (
        typeof conversation.created_at !== "string" ||
        !conversation.created_at.trim()
      )
        return ""
      const date = new Date(conversation.created_at)
      return Number.isNaN(date.getTime()) ? "" : formatter.format(date)
    })
    group.forEach((conversation, index) => {
      if (group.length === 1) {
        labels.set(conversation.id, conversation.title)
        return
      }
      let context = dates[index]
      if (!context || dates.filter((date) => date === context).length > 1) {
        const shortId = conversation.id.slice(0, 8)
        const id = group.some(
          (other) =>
            other.id !== conversation.id && other.id.startsWith(shortId)
        )
          ? conversation.id
          : shortId
        context = [context, id].filter(Boolean).join(" · ")
      }
      // Put context first so a long generated title cannot hide the distinction.
      labels.set(conversation.id, `${context} — ${conversation.title}`)
    })
  }
  return labels
}
