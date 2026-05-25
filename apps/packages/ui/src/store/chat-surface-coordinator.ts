import { createStore } from "zustand/vanilla"
import { createWithEqualityFn } from "zustand/traditional"

export type ChatSurfaceId = "webui" | "extension"

export type OptionalPanelId =
  | "server-history"
  | "mcp-tools"
  | "audio-health"
  | "model-catalog"
  | "character-control"
export type OptionalPanelVisibility = Partial<Record<OptionalPanelId, boolean>>

export type ChatSurfaceCoordinatorState = {
  routeId: string | null
  surface: ChatSurfaceId | null
  visiblePanels: OptionalPanelVisibility
  engagedPanels: OptionalPanelVisibility
  setRouteContext: (value: {
    routeId: string
    surface: ChatSurfaceId
  }) => void
  setPanelVisible: (panel: OptionalPanelId, visible: boolean) => void
  markPanelEngaged: (panel: OptionalPanelId) => void
}

const DEFAULT_OPTIONAL_PANELS: Record<OptionalPanelId, boolean> = {
  "server-history": false,
  "mcp-tools": false,
  "audio-health": false,
  "model-catalog": false,
  "character-control": false
}

export const createChatSurfaceCoordinatorState = (
  set: (
    partial:
      | Partial<ChatSurfaceCoordinatorState>
      | ((state: ChatSurfaceCoordinatorState) => Partial<ChatSurfaceCoordinatorState>)
  ) => void
): ChatSurfaceCoordinatorState => ({
  routeId: null,
  surface: null,
  visiblePanels: { ...DEFAULT_OPTIONAL_PANELS },
  engagedPanels: { ...DEFAULT_OPTIONAL_PANELS },
  setRouteContext: (value) => set(value),
  setPanelVisible: (panel, visible) =>
    set((state) => ({
      visiblePanels: {
        ...state.visiblePanels,
        [panel]: visible
      }
    })),
  markPanelEngaged: (panel) =>
    set((state) => ({
      engagedPanels: {
        ...state.engagedPanels,
        [panel]: true
      }
    }))
})

export const createChatSurfaceCoordinatorStore = () =>
  createStore<ChatSurfaceCoordinatorState>((set) =>
    createChatSurfaceCoordinatorState(set)
  )

export const useChatSurfaceCoordinatorStore =
  createWithEqualityFn<ChatSurfaceCoordinatorState>()((set) =>
    createChatSurfaceCoordinatorState(set)
  )

export const shouldEnableOptionalResource = (
  state: Pick<ChatSurfaceCoordinatorState, "visiblePanels" | "engagedPanels">,
  panel: OptionalPanelId
): boolean => Boolean(state.visiblePanels[panel] && state.engagedPanels[panel])
