import { createWithEqualityFn } from "zustand/traditional"

type NodeType = "part" | "chapter" | "scene" | null

type WritingPlaygroundState = {
  activeSessionId: string | null
  activeSessionName: string | null
  setActiveSessionId: (activeSessionId: string | null) => void
  setActiveSessionName: (activeSessionName: string | null) => void
  activeProjectId: string | null
  setActiveProjectId: (id: string | null) => void
  activeNodeId: string | null
  setActiveNodeId: (id: string | null) => void
  activeNodeType: NodeType
  setActiveNodeType: (type: NodeType) => void
  editorMode: "plain" | "tiptap"
  setEditorMode: (mode: "plain" | "tiptap") => void
  focusMode: boolean
  setFocusMode: (enabled: boolean) => void
  analysisModalOpen: "pulse" | "plot" | "timeline" | "web" | null
  setAnalysisModalOpen: (modal: "pulse" | "plot" | "timeline" | "web" | null) => void
}

export const useWritingPlaygroundStore = createWithEqualityFn<WritingPlaygroundState>((set) => ({
  activeSessionId: null,
  activeSessionName: null,
  setActiveSessionId: (activeSessionId) => set({ activeSessionId }),
  setActiveSessionName: (activeSessionName) => set({ activeSessionName }),
  activeProjectId: null,
  setActiveProjectId: (id) => set({ activeProjectId: id }),
  activeNodeId: null,
  setActiveNodeId: (id) => set({ activeNodeId: id, ...(id === null && { activeNodeType: null }) }),
  activeNodeType: null,
  setActiveNodeType: (type) => set({ activeNodeType: type }),
  editorMode: "plain" as const,
  setEditorMode: (mode) => set({ editorMode: mode }),
  focusMode: false,
  setFocusMode: (enabled) => set({ focusMode: enabled }),
  analysisModalOpen: null,
  setAnalysisModalOpen: (modal) => set({ analysisModalOpen: modal }),
}))
