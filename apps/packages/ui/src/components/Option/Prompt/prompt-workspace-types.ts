import type { PromptSourceSystem, PromptSyncStatus } from "@/db/dexie/types"

export type PromptSavedView = "all" | "favorites" | "recent" | "most_used" | "untagged"

export type PromptListSortKey = "title" | "modifiedAt" | null
export type PromptListSortOrder = "ascend" | "descend" | null

export type PromptListQueryState = {
  searchText: string
  typeFilter:
    | "all"
    | "system"
    | "quick"
    | "mixed"
    | "recipe"
    | "recipe_system"
    | "recipe_user"
  syncFilter: "all" | PromptSyncStatus
  usageFilter: "all" | "used" | "unused"
  tagFilter: string[]
  tagMatchMode: "any" | "all"
  sort: {
    key: PromptListSortKey
    order: PromptListSortOrder
  }
  page: number
  pageSize: number
  savedView: PromptSavedView
}

export type PromptSelectionState = {
  selectedIds: string[]
}

export type PromptPanelState = {
  open: boolean
  promptId: string | null
}

export type PromptWorkspaceState = {
  query: PromptListQueryState
  selection: PromptSelectionState
  panel: PromptPanelState
  isCompactViewport: boolean
}

export type PromptRowVM = {
  id: string
  title: string
  author?: string
  details?: string
  previewSystem?: string
  previewUser?: string
  keywords: string[]
  favorite: boolean
  syncStatus: PromptSyncStatus
  sourceSystem: PromptSourceSystem
  serverId?: number | null
  updatedAt?: number
  createdAt: number
  usageCount: number
  lastUsedAt?: number | null
  kind?: "prompt" | "recipe" | "quarantined_recipe"
  recipeTarget?: "system" | "user"
}

export const PROMPTS_WORKSPACE_MOBILE_BREAKPOINT_PX = 768

export const DEFAULT_PROMPT_QUERY_STATE: PromptListQueryState = {
  searchText: "",
  typeFilter: "all",
  syncFilter: "all",
  usageFilter: "all",
  tagFilter: [],
  tagMatchMode: "any",
  sort: {
    key: null,
    order: null
  },
  page: 1,
  pageSize: 20,
  savedView: "all"
}
