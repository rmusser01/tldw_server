import { createWithEqualityFn } from "zustand/traditional"
import type {
  AudioStudioAdmissionIntent,
  AudioStudioClip,
  AudioStudioClipType,
  AudioStudioProject as ServiceAudioStudioProject,
  AudioStudioSection,
  AudioStudioTrack,
  AudioStudioWorkflow
} from "@/services/audio-studio"

export type {
  AudioStudioWorkflow,
  AudioStudioSection,
  AudioStudioTrack,
  AudioStudioClip,
  AudioStudioClipType
}

export type AudioStudioWorkflowDefinition = {
  id: AudioStudioWorkflow
  label: string
  description: string
  sectionLabel: string
  generationLabel: string
}

export const AUDIO_STUDIO_WORKFLOWS: AudioStudioWorkflowDefinition[] = [
  {
    id: "narration",
    label: "Narration",
    description: "Long-form narration with chapters, voices, and subtitles.",
    sectionLabel: "Chapters",
    generationLabel: "Speech"
  },
  {
    id: "podcast",
    label: "Podcast",
    description: "Multi-speaker episodes with segments, intros, and beds.",
    sectionLabel: "Segments",
    generationLabel: "Speech"
  },
  {
    id: "briefing",
    label: "Briefing",
    description: "Source-driven audio summaries with sections and provenance.",
    sectionLabel: "Brief sections",
    generationLabel: "Speech"
  },
  {
    id: "music",
    label: "Music",
    description: "Prompt-based music generation for cues and beds.",
    sectionLabel: "Cues",
    generationLabel: "Music"
  }
]

export type AudioStudioProject = ServiceAudioStudioProject & {
  project_id: string
  title: string
  workflow: AudioStudioWorkflow
  status: string
  sections: AudioStudioSection[]
  tracks: AudioStudioTrack[]
  clips: AudioStudioClip[]
  settings: Record<string, unknown>
}

export type AudioStudioRevisionConflict = {
  local: AudioStudioProject
  incoming: AudioStudioProject
}

type AudioStudioStore = {
  jobIntents: Record<string, AudioStudioAdmissionIntent>
  activeWorkflow: AudioStudioWorkflow
  projects: AudioStudioProject[]
  activeProjectId: string | null
  activeProject: AudioStudioProject | null
  dirtyProjectIds: Record<string, true>
  revisionConflicts: Record<string, AudioStudioRevisionConflict>
  setActiveWorkflow: (workflow: AudioStudioWorkflow) => void
  setProjects: (projects: ServiceAudioStudioProject[]) => void
  setActiveProjectId: (projectId: string | null) => void
  upsertProjectFromServer: (project: ServiceAudioStudioProject) => void
  markProjectDirty: (projectId: string) => void
  markProjectClean: (projectId: string) => void
  updateProjectLocal: (
    projectId: string,
    updates: Partial<AudioStudioProject>
  ) => void
  resetAudioStudio: () => void
}

const normalizeProject = (
  project: ServiceAudioStudioProject
): AudioStudioProject => ({
  ...project,
  project_id: project.project_id,
  title: project.title,
  workflow: project.workflow,
  status: project.status,
  sections: (project.sections ?? []).map((section) => ({ ...section, order: section.order ?? section.order_index ?? 0 })),
  tracks: (project.tracks ?? []).map((track) => ({ ...track, order: track.order ?? track.order_index ?? 0 })),
  clips: project.clips ?? [],
  settings: project.settings ?? {}
})

const getRevision = (project: ServiceAudioStudioProject): string | undefined =>
  project.revision_id ?? project.current_revision_id

const resolveActiveProject = (
  projects: AudioStudioProject[],
  activeProjectId: string | null
) => projects.find((project) => project.project_id === activeProjectId) ?? null

const initialState = {
  jobIntents: {} as Record<string, AudioStudioAdmissionIntent>,
  activeWorkflow: "narration" as AudioStudioWorkflow,
  projects: [] as AudioStudioProject[],
  activeProjectId: null as string | null,
  activeProject: null as AudioStudioProject | null,
  dirtyProjectIds: {} as Record<string, true>,
  revisionConflicts: {} as Record<string, AudioStudioRevisionConflict>
}

export const useAudioStudioStore = createWithEqualityFn<AudioStudioStore>(
  (set, get) => ({
    ...initialState,

    setActiveWorkflow: (workflow) =>
      set((state) => {
        const activeProject =
          state.activeProject?.workflow === workflow
            ? state.activeProject
            : state.projects.find((project) => project.workflow === workflow) ?? null
        return {
          activeWorkflow: workflow,
          activeProjectId: activeProject?.project_id ?? null,
          activeProject
        }
      }),

    setProjects: (projects) => {
      for (const project of projects) get().upsertProjectFromServer(project)
    },

    setActiveProjectId: (projectId) =>
      set((state) => ({
        activeProjectId: projectId,
        activeProject: resolveActiveProject(state.projects, projectId)
      })),

    upsertProjectFromServer: (project) => {
      const state = get()
      const existing = state.projects.find(
        (candidate) => candidate.project_id === project.project_id
      )
      const incoming = normalizeProject({ ...existing, ...project })
      const isDirty = Boolean(state.dirtyProjectIds[incoming.project_id])
      const hasRevisionConflict =
        isDirty &&
        existing &&
        getRevision(existing) !== undefined &&
        getRevision(incoming) !== undefined &&
        getRevision(existing) !== getRevision(incoming)

      if (hasRevisionConflict) {
        set({
          revisionConflicts: {
            ...state.revisionConflicts,
            [incoming.project_id]: {
              local: existing,
              incoming
            }
          }
        })
        return
      }

      if (isDirty) return

      const projects = existing
        ? state.projects.map((candidate) =>
            candidate.project_id === incoming.project_id ? incoming : candidate
          )
        : [...state.projects, incoming]

      set({
        projects,
        activeProject: resolveActiveProject(projects, state.activeProjectId),
        revisionConflicts: Object.fromEntries(
          Object.entries(state.revisionConflicts).filter(
            ([projectId]) => projectId !== incoming.project_id
          )
        )
      })
    },

    markProjectDirty: (projectId) =>
      set((state) => ({
        dirtyProjectIds: {
          ...state.dirtyProjectIds,
          [projectId]: true
        }
      })),

    markProjectClean: (projectId) =>
      set((state) => {
        const { [projectId]: _dirty, ...dirtyProjectIds } = state.dirtyProjectIds
        return { dirtyProjectIds }
      }),

    updateProjectLocal: (projectId, updates) =>
      set((state) => {
        const projects = state.projects.map((project) =>
          project.project_id === projectId ? { ...project, ...updates } : project
        )
        return {
          projects,
          activeProject: resolveActiveProject(projects, state.activeProjectId),
          dirtyProjectIds: {
            ...state.dirtyProjectIds,
            [projectId]: true
          }
        }
      }),

    resetAudioStudio: () => set({ ...initialState })
  })
)

type AudioStudioDevWindow = Window & {
  __tldw_useAudioStudioStore?: typeof useAudioStudioStore
}

if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  ;(window as AudioStudioDevWindow).__tldw_useAudioStudioStore =
    useAudioStudioStore
}
