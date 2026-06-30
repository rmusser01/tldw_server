import { createWithEqualityFn } from "zustand/traditional"
import type { McpToolDefinition } from "@/services/tldw/mcp"
import type {
  ChatToolFilterCounts,
  ResolvedMcpTool
} from "@/utils/chat-tools"
import type { McpDisabledToolPreferences } from "@/services/settings/ui-settings"

export type McpHealthState =
  | "unknown"
  | "healthy"
  | "unhealthy"
  | "unavailable"

const EMPTY_TOOL_COUNTS: ChatToolFilterCounts = {
  discovered: 0,
  executable: 0,
  disabled: 0,
  colliding: 0,
  chatEnabled: 0
}

const EMPTY_DISABLED_TOOL_PREFERENCES: McpDisabledToolPreferences = {
  version: 1,
  scopes: {}
}

type McpToolFilterStoreState = {
  discoveredTools: ResolvedMcpTool[]
  availableTools: ResolvedMcpTool[]
  chatTools: ResolvedMcpTool[]
  disabledToolPreferences: McpDisabledToolPreferences
  activeToolPreferenceScope: string
  disabledToolNames: string[]
  collisionToolNames: string[]
  toolCounts: ChatToolFilterCounts
}

export type McpToolsState = McpToolFilterStoreState & {
  tools: McpToolDefinition[]
  healthState: McpHealthState
  toolsLoading: boolean
  toolCatalog: string
  toolCatalogId: number | null
  toolModules: string[]
  toolCatalogStrict: boolean
  setTools: (tools: McpToolDefinition[]) => void
  setToolFilterState: (state: McpToolFilterStoreState) => void
  setHealthState: (state: McpHealthState) => void
  setToolsLoading: (loading: boolean) => void
  setToolCatalog: (catalog: string) => void
  setToolCatalogId: (catalogId: number | null) => void
  setToolModules: (moduleIds: string[]) => void
  setToolCatalogStrict: (strict: boolean) => void
}

export const useMcpToolsStore = createWithEqualityFn<McpToolsState>((set) => ({
  tools: [],
  discoveredTools: [],
  availableTools: [],
  chatTools: [],
  disabledToolPreferences: EMPTY_DISABLED_TOOL_PREFERENCES,
  activeToolPreferenceScope: "default",
  disabledToolNames: [],
  collisionToolNames: [],
  toolCounts: EMPTY_TOOL_COUNTS,
  healthState: "unknown",
  toolsLoading: false,
  toolCatalog: "",
  toolCatalogId: null,
  toolModules: [],
  toolCatalogStrict: false,
  setTools: (tools) => set({ tools }),
  setToolFilterState: (state) =>
    set({
      ...state,
      tools: state.availableTools.map((tool) => tool.tool as McpToolDefinition)
    }),
  setHealthState: (healthState) => set({ healthState }),
  setToolsLoading: (toolsLoading) => set({ toolsLoading }),
  setToolCatalog: (toolCatalog) => set({ toolCatalog }),
  setToolCatalogId: (toolCatalogId) => set({ toolCatalogId }),
  setToolModules: (toolModules) => set({ toolModules }),
  setToolCatalogStrict: (toolCatalogStrict) => set({ toolCatalogStrict })
}))
