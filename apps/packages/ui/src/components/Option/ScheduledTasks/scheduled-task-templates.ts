export type ScheduledTaskTemplateId =
  | "reminder"
  | "watch"
  | "ingest"
  | "recurring_question"
  | "agent_task"
  | "advanced"

export type ScheduledTaskTemplateState =
  | "available"
  | "handoff_only"
  | "needs_setup"
  | "managed_in_watchlists"
  | "planned"
  | "unavailable"

export type ScheduledTaskTemplateCategory =
  | "reminder"
  | "watch"
  | "ingest"
  | "research"
  | "agent"
  | "advanced"

export interface ScheduledTaskTemplate {
  id: ScheduledTaskTemplateId
  category: ScheduledTaskTemplateCategory
  title: string
  intent: string
  description: string
  state: ScheduledTaskTemplateState
  primaryActionLabel: string
  secondaryActionLabel?: string
  examples?: string[]
  keywords: string[]
}

export type ScheduledTaskTemplateFilterId =
  | "all"
  | "available_now"
  | ScheduledTaskTemplateCategory

export interface ScheduledTaskTemplateFilter {
  id: ScheduledTaskTemplateFilterId
  label: string
}

const SENSITIVE_URL_PARAM_PATTERN =
  /(^|[?&#])(token|api[_-]?key|key|secret|session|sid|auth|code|invite)=/i

export const SCHEDULED_TASK_TEMPLATES: readonly ScheduledTaskTemplate[] = [
  {
    id: "reminder",
    category: "reminder",
    title: "Reminder",
    intent: "Remind me later or repeatedly",
    description: "Schedule a one-time or recurring reminder.",
    state: "available",
    primaryActionLabel: "Create reminder",
    secondaryActionLabel: "Create another",
    examples: ["Remind me tomorrow", "Repeat every week", "Monthly check-in"],
    keywords: ["remind", "reminder", "later", "daily", "weekly", "monthly"]
  },
  {
    id: "watch",
    category: "watch",
    title: "Watch for new items",
    intent: "Tell me when something new appears",
    description: "Get notified when a source has new matching items.",
    state: "handoff_only",
    primaryActionLabel: "Continue in Watchlists",
    secondaryActionLabel: "Copy setup summary",
    examples: [
      "Repository issues",
      "RSS feeds",
      "Forums",
      "Vendor advisories"
    ],
    keywords: ["watch", "monitor", "new", "changes", "alert", "notify"]
  },
  {
    id: "ingest",
    category: "ingest",
    title: "Ingest new content",
    intent: "Add new content to my library/search",
    description: "Add new source content to your library and search surfaces.",
    state: "handoff_only",
    primaryActionLabel: "Continue in Watchlists",
    secondaryActionLabel: "Copy setup summary",
    examples: ["Make a channel searchable", "Index a feed", "Download source content"],
    keywords: [
      "ingest",
      "scrape",
      "download",
      "index",
      "searchable",
      "library",
      "channel"
    ]
  },
  {
    id: "recurring_question",
    category: "research",
    title: "Recurring question",
    intent: "Keep asking this question as new data arrives",
    description: "Keep checking for an answer as new data arrives.",
    state: "planned",
    primaryActionLabel: "View planned capability",
    examples: ["Ask again when new data arrives", "Keep looking for an answer"],
    keywords: ["question", "answer", "rag", "search again", "keep looking"]
  },
  {
    id: "agent_task",
    category: "agent",
    title: "Agent task",
    intent: "Send a prompt/message to an agent later",
    description: "Send a message to an agent at a scheduled time.",
    state: "planned",
    primaryActionLabel: "View planned capability",
    examples: ["Prompt an assistant tomorrow", "Message an agent later"],
    keywords: ["agent", "prompt", "message", "assistant", "acp"]
  },
  {
    id: "advanced",
    category: "advanced",
    title: "Advanced task",
    intent: "I know the domain I need",
    description: "Choose the workspace that owns deeper automation setup.",
    state: "handoff_only",
    primaryActionLabel: "Choose destination",
    secondaryActionLabel: "Copy setup summary",
    examples: ["Custom workflow", "Advanced automation"],
    keywords: ["advanced", "workflow", "custom"]
  }
] as const

export const SCHEDULED_TASK_TEMPLATE_FILTERS: readonly ScheduledTaskTemplateFilter[] = [
  { id: "all", label: "All" },
  { id: "available_now", label: "Available now" },
  { id: "watch", label: "Watch" },
  { id: "ingest", label: "Ingest" },
  { id: "research", label: "Research" },
  { id: "agent", label: "Agent" },
  { id: "advanced", label: "Advanced" }
] as const

export const getScheduledTaskTemplate = (
  id: ScheduledTaskTemplateId | string | null | undefined
): ScheduledTaskTemplate | null =>
  SCHEDULED_TASK_TEMPLATES.find((template) => template.id === id) ?? null

export const filterScheduledTaskTemplates = (
  filterId: ScheduledTaskTemplateFilterId
): readonly ScheduledTaskTemplate[] => {
  if (filterId === "all") {
    return SCHEDULED_TASK_TEMPLATES
  }

  if (filterId === "available_now") {
    return SCHEDULED_TASK_TEMPLATES.filter((template) => template.state === "available")
  }

  return SCHEDULED_TASK_TEMPLATES.filter((template) => template.category === filterId)
}

export const findScheduledTaskTemplates = (
  query: string | null | undefined
): ScheduledTaskTemplate[] => {
  const normalizedQuery = query?.trim().toLowerCase()
  if (!normalizedQuery) {
    return []
  }

  return SCHEDULED_TASK_TEMPLATES.map((template, index) => ({
    template,
    index,
    score: template.keywords.reduce(
      (score, keyword) =>
        normalizedQuery.includes(keyword.toLowerCase()) ? score + 1 : score,
      0
    )
  }))
    .filter((match) => match.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((match) => match.template)
}

export const getScheduledTaskTemplateStateLabel = (
  state: ScheduledTaskTemplateState
): string => {
  switch (state) {
    case "available":
      return "Available"
    case "handoff_only":
      return "Handoff only"
    case "needs_setup":
      return "Needs setup"
    case "managed_in_watchlists":
      return "Managed in Watchlists"
    case "planned":
      return "Planned capability"
    case "unavailable":
      return "Unavailable"
  }
}

const appearsToBeUrl = (value: string): boolean =>
  /^[a-z][a-z0-9+.-]*:\/\//i.test(value) || /^www\./i.test(value)

export const toSafeHandoffSourceText = (value: unknown): string | null => {
  if (typeof value !== "string") {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  if (SENSITIVE_URL_PARAM_PATTERN.test(trimmed)) {
    return null
  }

  if (appearsToBeUrl(trimmed) && trimmed.includes("#")) {
    return null
  }

  return trimmed
}
