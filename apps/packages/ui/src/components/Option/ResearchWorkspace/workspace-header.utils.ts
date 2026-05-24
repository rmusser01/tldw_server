import type {
  SavedWorkspace,
  WorkspaceCollection,
  WorkspaceSource
} from "@/types/workspace"

export interface WorkspaceCollectionGroup {
  id: string
  name: string
  collection: WorkspaceCollection | null
  workspaces: SavedWorkspace[]
}

export interface WorkspaceTemplatePreset {
  id: string
  label: string
  workspaceName: string
  noteTitle: string
  noteContent: string
  keywords: string[]
}

export const WORKSPACE_TEMPLATE_PRESETS: WorkspaceTemplatePreset[] = [
  {
    id: "literature_review",
    label: "Literature Review",
    workspaceName: "Literature Review Workspace",
    noteTitle: "Literature Review Plan",
    noteContent:
      "Research goal:\n\nKey questions:\n- \n\nEvidence matrix:\n- Claim:\n- Supporting sources:\n- Contradictions:\n\nNext actions:\n- ",
    keywords: ["literature", "evidence", "synthesis"]
  },
  {
    id: "interview_analysis",
    label: "Interview Analysis",
    workspaceName: "Interview Analysis Workspace",
    noteTitle: "Interview Findings",
    noteContent:
      "Participants:\n- \n\nThemes:\n1. \n2. \n3. \n\nQuotations to verify:\n- \n\nOpen follow-ups:\n- ",
    keywords: ["interviews", "qualitative", "themes"]
  },
  {
    id: "product_brief",
    label: "Product Brief",
    workspaceName: "Product Brief Workspace",
    noteTitle: "Product Brief Draft",
    noteContent:
      "Problem statement:\n\nTarget user:\n\nCore requirements:\n- \n\nRisks and unknowns:\n- \n\nLaunch checklist:\n- ",
    keywords: ["product", "brief", "launch"]
  }
]

const getTimeDelta = (date: Date, now: Date): number => {
  return Math.max(0, now.getTime() - date.getTime())
}

export const formatWorkspaceLastAccessed = (
  lastAccessedAt: Date,
  now: Date = new Date()
): string => {
  const deltaMs = getTimeDelta(lastAccessedAt, now)
  const minute = 60 * 1000
  const hour = 60 * minute
  const day = 24 * hour
  const week = 7 * day

  if (deltaMs < minute) return "just now"
  if (deltaMs < hour) return `${Math.floor(deltaMs / minute)}m ago`
  if (deltaMs < day) return `${Math.floor(deltaMs / hour)}h ago`
  if (deltaMs < week) return `${Math.floor(deltaMs / day)}d ago`

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(lastAccessedAt)
}

export const filterSavedWorkspaces = (
  workspaces: SavedWorkspace[],
  query: string
): SavedWorkspace[] => {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) return workspaces

  return workspaces.filter((workspace) => {
    const haystack = `${workspace.name} ${workspace.tag}`.toLowerCase()
    return haystack.includes(normalizedQuery)
  })
}

export const groupWorkspacesByCollection = (
  collections: WorkspaceCollection[],
  workspaces: SavedWorkspace[]
): WorkspaceCollectionGroup[] => {
  const collectionGroups = collections.map<WorkspaceCollectionGroup>((collection) => ({
    id: collection.id,
    name: collection.name,
    collection,
    workspaces: []
  }))
  const groupsById = new Map(
    collectionGroups.map((group) => [group.id, group] as const)
  )
  const unassignedGroup: WorkspaceCollectionGroup = {
    id: "unassigned",
    name: "Unassigned",
    collection: null,
    workspaces: []
  }

  for (const workspace of workspaces) {
    if (!workspace.collectionId) {
      unassignedGroup.workspaces.push(workspace)
      continue
    }

    const group = groupsById.get(workspace.collectionId)
    if (!group) {
      unassignedGroup.workspaces.push(workspace)
      continue
    }

    group.workspaces.push(workspace)
  }

  return [...collectionGroups, unassignedGroup]
}

const toDateStamp = (date: Date): string => {
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, "0")
  const day = String(date.getUTCDate()).padStart(2, "0")
  return `${year}${month}${day}`
}

const toIsoDate = (date: Date): string => {
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, "0")
  const day = String(date.getUTCDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

const toBibtexValue = (value: string): string => {
  return value.replace(/[{}]/g, "").replace(/\s+/g, " ").trim()
}

const toSlug = (value: string): string => {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return normalized || "workspace"
}

const toBibtexKeyChunk = (value: string): string => {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 24) || "source"
}

export const createWorkspaceBibtexFilename = (
  workspaceName: string,
  now: Date = new Date()
): string => {
  return `${toSlug(workspaceName)}-citations-${toDateStamp(now)}.bib`
}

export const buildWorkspaceBibtex = (
  sources: WorkspaceSource[],
  options?: {
    workspaceTag?: string
    now?: Date
  }
): string => {
  const now = options?.now || new Date()
  const workspaceTagChunk = toBibtexKeyChunk(options?.workspaceTag || "workspace")

  const entries = sources.map((source, index) => {
    const entryDate = source.addedAt instanceof Date ? source.addedAt : now
    const year = entryDate.getUTCFullYear()
    const key = `${workspaceTagChunk}${year}${String(index + 1).padStart(2, "0")}`
    const fields: string[] = [
      `  title = {${toBibtexValue(source.title)}}`,
      `  year = {${year}}`,
      `  note = {media_id=${source.mediaId}; type=${source.type}}`
    ]

    if (source.url && source.url.trim().length > 0) {
      const safeUrl = toBibtexValue(source.url)
      fields.push(`  url = {${safeUrl}}`)
      fields.push(`  urldate = {${toIsoDate(entryDate)}}`)
    }

    return `@misc{${key},\n${fields.join(",\n")}\n}`
  })

  return entries.join("\n\n")
}
