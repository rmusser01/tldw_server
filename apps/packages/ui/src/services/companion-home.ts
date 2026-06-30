import type { ReadingItemSummary } from "@/types/collections"

import {
  fetchCompanionWorkspaceSnapshot,
  type CompanionActivityItem,
  type CompanionGoal,
  type CompanionNotification
} from "@/services/companion"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type CompanionHomeSurface = "options" | "sidepanel"

export type CompanionHomeSource =
  | "canonical_inbox"
  | "goal"
  | "reading"
  | "note"
  | "scheduled_task"

export type CompanionHomeEntityType =
  | "notification"
  | "goal"
  | "reading_item"
  | "note"
  | "scheduled_task_result"

export type CompanionHomeDegradedSource = "workspace" | "reading" | "notes"

export type CompanionHomeItem = {
  id: string
  entityId: string
  entityType: CompanionHomeEntityType
  source: CompanionHomeSource
  title: string
  summary: string
  updatedAt: string | null
  href?: string
}

export type CompanionHomeSnapshot = {
  surface: CompanionHomeSurface
  inbox: CompanionHomeItem[]
  needsAttention: CompanionHomeItem[]
  resumeWork: CompanionHomeItem[]
  goalsFocus: CompanionHomeItem[]
  recentActivity: CompanionHomeItem[]
  readingQueue: CompanionHomeItem[]
  degradedSources: CompanionHomeDegradedSource[]
  summary: {
    activityCount: number
    inboxCount: number
    needsAttentionCount: number
    resumeWorkCount: number
  }
}

type NormalizedNoteEntry = {
  id: string
  title: string
  summary: string
  updatedAt: string | null
  completed: boolean
}

const STALE_AGE_MS = 7 * 24 * 60 * 60 * 1000

const toNonEmptyString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const toTimestamp = (value: string | null | undefined): number | null => {
  if (!value) return null
  const next = Date.parse(value)
  return Number.isFinite(next) ? next : null
}

const isStale = (value: string | null | undefined): boolean => {
  const timestamp = toTimestamp(value)
  if (timestamp == null) return true
  return Date.now() - timestamp >= STALE_AGE_MS
}

const summarizeGoalProgress = (goal: CompanionGoal): string => {
  const completed = Number(goal.progress?.completed_count)
  const target =
    Number(goal.config?.target_count) ||
    Number(goal.progress?.target_count) ||
    Number(goal.progress?.target)

  if (Number.isFinite(completed) && Number.isFinite(target) && target > 0) {
    return `${completed} / ${target} complete`
  }
  if (Number.isFinite(completed)) {
    return `${completed} completed`
  }
  return goal.description?.trim() || "Progress needs an explicit update."
}

const hasMeaningfulGoalProgress = (goal: CompanionGoal): boolean => {
  const progress = goal.progress || {}
  return Object.values(progress).some((value) => {
    if (typeof value === "number") {
      return Number.isFinite(value) && value > 0
    }
    if (typeof value === "string") {
      return value.trim().length > 0
    }
    if (typeof value === "boolean") {
      return value
    }
    return false
  })
}

const isOpenReadingItem = (item: ReadingItemSummary): boolean =>
  item.status === "saved" || item.status === "reading" || typeof item.status === "undefined"

const normalizeInboxEntityType = (
  notification: CompanionNotification
): CompanionHomeEntityType => {
  const linkType = String(notification.link_type || "").toLowerCase()
  if (linkType.includes("goal")) return "goal"
  if (linkType.includes("reading")) return "reading_item"
  if (linkType.includes("note") || linkType.includes("document")) return "note"
  return "notification"
}

const routeForEntityType = (
  surface: CompanionHomeSurface,
  entityType: CompanionHomeEntityType
): string | undefined => {
  if (surface === "sidepanel" && (entityType === "reading_item" || entityType === "note")) {
    return undefined
  }
  if (entityType === "reading_item") return "/collections"
  if (entityType === "note") return "/notes"
  return "/companion"
}

const buildInboxItem = (
  surface: CompanionHomeSurface,
  notification: CompanionNotification
): CompanionHomeItem => {
  const entityType = normalizeInboxEntityType(notification)
  const entityId = notification.link_id || String(notification.id)
  return {
    id: `inbox:${notification.id}`,
    entityId,
    entityType,
    source: "canonical_inbox",
    title: notification.title,
    summary: notification.message,
    updatedAt: notification.created_at || null,
    href: routeForEntityType(surface, entityType)
  }
}

const buildGoalItem = (
  surface: CompanionHomeSurface,
  goal: CompanionGoal
): CompanionHomeItem => ({
  id: `goal:${goal.id}`,
  entityId: goal.id,
  entityType: "goal",
  source: "goal",
  title: goal.title,
  summary: summarizeGoalProgress(goal),
  updatedAt: goal.updated_at || goal.created_at || null,
  href: routeForEntityType(surface, "goal")
})

const buildReadingItem = (
  surface: CompanionHomeSurface,
  item: ReadingItemSummary
): CompanionHomeItem => ({
  id: `reading:${item.id}`,
  entityId: item.id,
  entityType: "reading_item",
  source: "reading",
  title: item.title,
  summary: item.summary || item.domain || item.url || "Saved for later reading.",
  updatedAt: item.updated_at || item.created_at || null,
  href: routeForEntityType(surface, "reading_item")
})

const buildNoteItem = (
  surface: CompanionHomeSurface,
  note: NormalizedNoteEntry
): CompanionHomeItem => ({
  id: `note:${note.id}`,
  entityId: note.id,
  entityType: "note",
  source: "note",
  title: note.title,
  summary: note.summary,
  updatedAt: note.updatedAt,
  href: routeForEntityType(surface, "note")
})

const normalizeActivityEntityType = (
  activity: CompanionActivityItem
): CompanionHomeEntityType => {
  const sourceType = String(activity.source_type || "").toLowerCase()
  if (sourceType.includes("goal")) return "goal"
  if (sourceType.includes("reading")) return "reading_item"
  if (sourceType.includes("note") || sourceType.includes("document")) return "note"
  return "notification"
}

const sourceForEntityType = (
  entityType: CompanionHomeEntityType
): CompanionHomeSource => {
  if (entityType === "goal") return "goal"
  if (entityType === "reading_item") return "reading"
  if (entityType === "note") return "note"
  return "canonical_inbox"
}

const titleCaseToken = (value: string): string =>
  value.length > 0 ? value[0].toUpperCase() + value.slice(1) : value

const humanizeActivityEvent = (value: string): string => {
  const normalized = toNonEmptyString(value)
  if (!normalized) return "Recent activity"
  return normalized
    .split(/[._-]/)
    .filter(Boolean)
    .map(titleCaseToken)
    .join(" ")
}

const buildActivityItem = (
  surface: CompanionHomeSurface,
  activity: CompanionActivityItem
): CompanionHomeItem => {
  const entityType = normalizeActivityEntityType(activity)
  const metadata =
    activity.metadata && typeof activity.metadata === "object" ? activity.metadata : {}

  return {
    id: `activity:${activity.id}`,
    entityId: activity.source_id || activity.id,
    entityType,
    source: sourceForEntityType(entityType),
    title:
      toNonEmptyString((metadata as Record<string, unknown>).title) ||
      toNonEmptyString((metadata as Record<string, unknown>).page_title) ||
      humanizeActivityEvent(activity.event_type),
    summary:
      toNonEmptyString((metadata as Record<string, unknown>).summary) ||
      toNonEmptyString((metadata as Record<string, unknown>).selection) ||
      toNonEmptyString((metadata as Record<string, unknown>).page_url) ||
      "Recent companion activity.",
    updatedAt: activity.created_at || null,
    href: routeForEntityType(surface, entityType)
  }
}

const collectAmbiguousInboxEntityIds = (items: CompanionHomeItem[]): Set<string> => {
  const entityTypesById = new Map<string, Set<CompanionHomeEntityType>>()

  items.forEach((item) => {
    const observedTypes =
      entityTypesById.get(item.entityId) ?? new Set<CompanionHomeEntityType>()
    observedTypes.add(item.entityType)
    entityTypesById.set(item.entityId, observedTypes)
  })

  return new Set(
    [...entityTypesById.entries()]
      .filter(([, observedTypes]) => observedTypes.size > 1)
      .map(([entityId]) => entityId)
  )
}

const buildCanonicalInboxEntityKeys = (items: CompanionHomeItem[]): Set<string> => {
  const keys = new Set<string>()
  const ambiguousEntityIds = collectAmbiguousInboxEntityIds(items)

  items.forEach((item) => {
    keys.add(`${item.entityType}:${item.entityId}`)
    if (ambiguousEntityIds.has(item.entityId)) {
      keys.add(`any:${item.entityId}`)
    }
  })
  return keys
}

const shouldSuppressDerivedItem = (
  item: Pick<CompanionHomeItem, "entityId" | "entityType">,
  inboxEntityKeys: Set<string>
): boolean => {
  return (
    inboxEntityKeys.has(`${item.entityType}:${item.entityId}`) ||
    inboxEntityKeys.has(`any:${item.entityId}`)
  )
}

const normalizeNoteEntries = (payload: unknown): NormalizedNoteEntry[] => {
  const rawItems = Array.isArray((payload as { items?: unknown[] } | null)?.items)
    ? ((payload as { items: unknown[] }).items)
    : Array.isArray(payload)
      ? payload
      : []

  return rawItems
    .map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return null
      }

      const record = item as Record<string, unknown>
      const id =
        toNonEmptyString(record.id) ||
        toNonEmptyString(record.note_id) ||
        toNonEmptyString(record.uuid)
      if (!id) return null

      const title =
        toNonEmptyString(record.title) ||
        toNonEmptyString(record.name) ||
        "Untitled note"
      const content =
        toNonEmptyString(record.summary) ||
        toNonEmptyString(record.content) ||
        toNonEmptyString(record.text) ||
        "Unfinished note."
      const status = String(record.status || "").toLowerCase()
      const completed =
        record.completed === true ||
        record.is_finished === true ||
        status === "complete" ||
        status === "completed" ||
        status === "done" ||
        status === "archived"
      const updatedAt =
        toNonEmptyString(record.updated_at) ||
        toNonEmptyString(record.modified_at) ||
        toNonEmptyString(record.last_modified_at) ||
        toNonEmptyString(record.created_at)

      return {
        id,
        title,
        summary: content,
        updatedAt,
        completed
      }
    })
    .filter((item): item is NormalizedNoteEntry => item !== null)
}

export const fetchCompanionHomeSnapshot = async (
  surface: CompanionHomeSurface
): Promise<CompanionHomeSnapshot> => {
  const degradedSources: CompanionHomeDegradedSource[] = []

  const [workspaceResult, readingResult, notesResult] = await Promise.allSettled([
    fetchCompanionWorkspaceSnapshot(),
    tldwClient.getReadingList({
      page: 1,
      size: 25,
      status: ["saved", "reading"]
    }),
    tldwClient.listNotes({
      page: 1,
      results_per_page: 25,
      include_keywords: false
    })
  ])

  const workspace =
    workspaceResult.status === "fulfilled"
      ? workspaceResult.value
      : (degradedSources.push("workspace"), null)
  const readingItems =
    readingResult.status === "fulfilled"
      ? readingResult.value.items
      : (degradedSources.push("reading"), [])
  const noteEntries =
    notesResult.status === "fulfilled"
      ? normalizeNoteEntries(notesResult.value)
      : (degradedSources.push("notes"), [])

  const inbox = Array.isArray(workspace?.inbox)
    ? workspace.inbox.map((notification) => buildInboxItem(surface, notification))
    : []
  const inboxEntityKeys = buildCanonicalInboxEntityKeys(inbox)

  const activeGoals = Array.isArray(workspace?.goals)
    ? workspace.goals.filter((goal) => goal.status === "active")
    : []
  const resumeGoals = activeGoals
    .map((goal) => buildGoalItem(surface, goal))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))
  const attentionGoals = activeGoals
    .filter((goal) => !hasMeaningfulGoalProgress(goal) || isStale(goal.updated_at))
    .map((goal) => buildGoalItem(surface, goal))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))

  const openReadingItems = readingItems.filter(isOpenReadingItem)
  const resumeReading = openReadingItems
    .map((item) => buildReadingItem(surface, item))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))
  const attentionReading = openReadingItems
    .filter((item) => item.status === "saved" && isStale(item.updated_at || item.created_at))
    .map((item) => buildReadingItem(surface, item))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))

  const resumeNotes = noteEntries
    .filter((note) => !note.completed)
    .map((note) => buildNoteItem(surface, note))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))
  const attentionNotes = noteEntries
    .filter((note) => !note.completed && isStale(note.updatedAt))
    .map((note) => buildNoteItem(surface, note))
    .filter((item) => !shouldSuppressDerivedItem(item, inboxEntityKeys))

  const needsAttention = [...attentionGoals, ...attentionReading, ...attentionNotes]
  const resumeWork = [...resumeGoals, ...resumeReading, ...resumeNotes]
  const goalsFocus = activeGoals.map((goal) => buildGoalItem(surface, goal))
  const recentActivity = Array.isArray(workspace?.activity)
    ? workspace.activity.map((activity) => buildActivityItem(surface, activity))
    : []
  const readingQueue = openReadingItems.map((item) => buildReadingItem(surface, item))

  return {
    surface,
    inbox,
    needsAttention,
    resumeWork,
    goalsFocus,
    recentActivity,
    readingQueue,
    degradedSources,
    summary: {
      activityCount: workspace?.activityTotal ?? 0,
      inboxCount: inbox.length,
      needsAttentionCount: needsAttention.length,
      resumeWorkCount: resumeWork.length
    }
  }
}
