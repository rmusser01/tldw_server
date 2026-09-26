import type { WorkspaceSlice } from './types'
import type { WorkspaceState } from '../workspace'
import type { GeneratedArtifact, WorkspaceBanner } from '@/types/workspace'
import { DEFAULT_WORKSPACE_BANNER, DEFAULT_WORKSPACE_NOTE } from '@/types/workspace'

// TODO: These helpers need to be exported from workspace.ts
import {
  generateWorkspaceId,
  reviveDateOrNull,
  reviveDateOrUndefined,
  sanitizeWorkspaceBanner
} from '../workspace'

type CaptureNoteMode = "append" | "replace"

// Extract StudioActions from workspace.ts (action interface only)
type StudioActions = Pick<
  WorkspaceState,
  | 'addArtifact'
  | 'updateArtifactStatus'
  | 'removeArtifact'
  | 'restoreArtifact'
  | 'clearArtifacts'
  | 'setNotes'
  | 'setWorkspaceBanner'
  | 'clearWorkspaceBannerImage'
  | 'resetWorkspaceBanner'
  | 'setIsGeneratingOutput'
  | 'setCurrentNote'
  | 'updateNoteContent'
  | 'updateNoteTitle'
  | 'updateNoteKeywords'
  | 'clearCurrentNote'
  | 'captureToCurrentNote'
  | 'loadNote'
>

export const createStudioSlice: WorkspaceSlice<StudioActions> = (set, get) => ({
  addArtifact: (artifactData) => {
    const artifact: GeneratedArtifact = {
      ...artifactData,
      id: generateWorkspaceId(),
      createdAt: new Date()
    }
    set((state) => ({
      generatedArtifacts: [artifact, ...state.generatedArtifacts]
    }))
    return artifact
  },

  updateArtifactStatus: (id, status, updates = {}) =>
    set((state) => ({
      generatedArtifacts: state.generatedArtifacts.map((a) =>
        a.id === id
          ? {
              ...a,
              status,
              ...updates,
              ...(status === "completed" ? { completedAt: new Date() } : {})
            }
          : a
      )
    })),

  removeArtifact: (id) =>
    set((state) => ({
      generatedArtifacts: state.generatedArtifacts.filter((a) => a.id !== id)
    })),

  restoreArtifact: (artifact, options) =>
    set((state) => {
      if (state.generatedArtifacts.some((entry) => entry.id === artifact.id)) {
        return state
      }

      const nextArtifacts = [...state.generatedArtifacts]
      const insertionIndex = Math.min(
        Math.max(options?.index ?? nextArtifacts.length, 0),
        nextArtifacts.length
      )
      nextArtifacts.splice(insertionIndex, 0, {
        ...artifact,
        createdAt: reviveDateOrNull(artifact.createdAt) || new Date(),
        completedAt: reviveDateOrUndefined(artifact.completedAt)
      })

      return { generatedArtifacts: nextArtifacts }
    }),

  clearArtifacts: () => set({ generatedArtifacts: [] }),

  setNotes: (notes) => set({ notes }),

  setWorkspaceBanner: (bannerUpdate) =>
    set((state) => {
      const nextCandidate: WorkspaceBanner = {
        title:
          bannerUpdate.title !== undefined
            ? bannerUpdate.title
            : state.workspaceBanner.title,
        subtitle:
          bannerUpdate.subtitle !== undefined
            ? bannerUpdate.subtitle
            : state.workspaceBanner.subtitle,
        image:
          bannerUpdate.image !== undefined
            ? bannerUpdate.image
            : state.workspaceBanner.image
      }

      return {
        workspaceBanner: sanitizeWorkspaceBanner(nextCandidate)
      }
    }),

  clearWorkspaceBannerImage: () =>
    set((state) => ({
      workspaceBanner: {
        ...state.workspaceBanner,
        image: null
      }
    })),

  resetWorkspaceBanner: () =>
    set({
      workspaceBanner: { ...DEFAULT_WORKSPACE_BANNER }
    }),

  setIsGeneratingOutput: (isGenerating, outputType = null) =>
    set({
      isGeneratingOutput: isGenerating,
      generatingOutputType: isGenerating ? outputType : null
    }),

  // Note management actions
  setCurrentNote: (note) =>
    set((state) => ({
      currentNote: note || { ...DEFAULT_WORKSPACE_NOTE },
      ownedNoteEditorSession: state.ownedNoteEditorSession + 1
    })),

  updateNoteContent: (content) =>
    set((state) => ({
      currentNote: { ...state.currentNote, content, isDirty: true }
    })),

  updateNoteTitle: (title) =>
    set((state) => ({
      currentNote: { ...state.currentNote, title, isDirty: true }
    })),

  updateNoteKeywords: (keywords) =>
    set((state) => ({
      currentNote: { ...state.currentNote, keywords, isDirty: true }
    })),

  clearCurrentNote: () =>
    set((state) => ({
      currentNote: { ...DEFAULT_WORKSPACE_NOTE },
      ownedNoteEditorSession: state.ownedNoteEditorSession + 1
    })),

  captureToCurrentNote: ({ title, content, mode = "append" }) =>
    set((state) => {
      const trimmedContent = content.trim()
      if (!trimmedContent) return state

      const cleanedTitle = (title || "").trim().slice(0, 120)
      const heading = cleanedTitle ? `## ${cleanedTitle}\n\n` : ""
      const captureBlock = `${heading}${trimmedContent}`
      const existingContent = state.currentNote.content.trim()
      const resolvedMode: CaptureNoteMode =
        mode === "replace" ? "replace" : "append"

      const nextContent =
        resolvedMode === "replace" || existingContent.length === 0
          ? captureBlock
          : `${existingContent}\n\n---\n\n${captureBlock}`
      const nextTitle =
        state.currentNote.title.trim() || cleanedTitle || state.currentNote.title

      return {
        currentNote: {
          ...state.currentNote,
          title: nextTitle,
          content: nextContent,
          isDirty: true
        }
      }
    }),

  loadNote: (note) =>
    set((state) => ({
      ownedNoteEditorSession: state.ownedNoteEditorSession + 1,
      currentNote: {
        id: note.id,
        title: note.title,
        content: note.content,
        keywords: note.keywords || [],
        version: note.version,
        isDirty: false
      }
    }))
})
