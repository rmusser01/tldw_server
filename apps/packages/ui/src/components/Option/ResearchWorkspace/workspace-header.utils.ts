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
  sourceChecklist: string[]
  suggestedPrompts: string[]
  studioRecommendations: string[]
  nextSteps: string[]
}

export const WORKSPACE_TEMPLATE_PRESETS: WorkspaceTemplatePreset[] = [
  {
    id: "literature_review",
    label: "Literature Review",
    workspaceName: "Literature Review Workspace",
    noteTitle: "Literature Review Plan",
    noteContent:
      "Research goal:\n\nKey questions:\n- \n\nEvidence matrix:\n- Claim:\n- Supporting sources:\n- Contradictions:\n\nNext actions:\n- ",
    keywords: ["literature", "evidence", "synthesis"],
    sourceChecklist: [
      "Add core papers, prior reviews, and recent contradictory studies.",
      "Tag sources by method, population, and evidence strength.",
      "Flag gaps where the current source set cannot support a claim."
    ],
    suggestedPrompts: [
      "Compare the strongest and weakest evidence across selected sources.",
      "Identify claims that need another source before synthesis.",
      "Draft a short literature gap summary with cited support."
    ],
    studioRecommendations: [
      "Literature matrix",
      "Evidence synthesis",
      "Research gap brief"
    ],
    nextSteps: [
      "Import at least three seed sources.",
      "Group sources by theme before drafting the synthesis.",
      "Review contradictions before exporting the final brief."
    ]
  },
  {
    id: "interview_analysis",
    label: "Interview Analysis",
    workspaceName: "Interview Analysis Workspace",
    noteTitle: "Interview Findings",
    noteContent:
      "Participants:\n- \n\nThemes:\n1. \n2. \n3. \n\nQuotations to verify:\n- \n\nOpen follow-ups:\n- ",
    keywords: ["interviews", "qualitative", "themes"],
    sourceChecklist: [
      "Add interview transcripts, notes, or recordings for each participant.",
      "Tag sources by participant segment, session date, and confidence.",
      "Separate direct quotes from moderator notes before synthesis."
    ],
    suggestedPrompts: [
      "Summarize recurring themes and unresolved follow-ups.",
      "Extract representative quotes for each major theme.",
      "List contradictions between participant segments."
    ],
    studioRecommendations: [
      "Theme synthesis",
      "Quote table",
      "Follow-up question list"
    ],
    nextSteps: [
      "Import transcripts or interview notes.",
      "Create tags for participant groups and research questions.",
      "Verify sensitive quotes before sharing or exporting."
    ]
  },
  {
    id: "product_brief",
    label: "Product Brief",
    workspaceName: "Product Brief Workspace",
    noteTitle: "Product Brief Draft",
    noteContent:
      "Problem statement:\n\nTarget user:\n\nCore requirements:\n- \n\nRisks and unknowns:\n- \n\nLaunch checklist:\n- ",
    keywords: ["product", "brief", "launch"],
    sourceChecklist: [
      "Add customer evidence, competitive references, and internal decisions.",
      "Tag sources by user segment, business priority, and release risk.",
      "Mark assumptions that still need validation."
    ],
    suggestedPrompts: [
      "Draft a decision-ready product brief from the selected sources.",
      "Summarize launch risks and the evidence behind each one.",
      "Turn source notes into measurable product requirements."
    ],
    studioRecommendations: [
      "Executive brief",
      "Requirements summary",
      "Risk register"
    ],
    nextSteps: [
      "Import customer and market evidence.",
      "Link decisions to supporting sources before review.",
      "Export the brief after risks and assumptions are resolved."
    ]
  }
]

const formatWorkspaceTemplateBulletList = (items: string[]): string =>
  items.map((item) => `- ${item}`).join("\n")

const formatWorkspaceTemplateChecklist = (items: string[]): string =>
  items.map((item) => `- [ ] ${item}`).join("\n")

export const buildWorkspaceTemplateNoteContent = (
  template: WorkspaceTemplatePreset
): string =>
  [
    template.noteContent.trim(),
    "## Source checklist",
    formatWorkspaceTemplateChecklist(template.sourceChecklist),
    "## Suggested prompts",
    formatWorkspaceTemplateBulletList(template.suggestedPrompts),
    "## Studio recommendations",
    formatWorkspaceTemplateBulletList(template.studioRecommendations),
    "## Next steps",
    formatWorkspaceTemplateChecklist(template.nextSteps)
  ].join("\n\n")

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
