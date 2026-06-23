import { beforeEach, describe, expect, it } from "vitest"
import {
  AUDIO_STUDIO_WORKFLOWS,
  useAudioStudioStore,
  type AudioStudioProject
} from "@/store/audio-studio"

const project = (overrides: Partial<AudioStudioProject>): AudioStudioProject => ({
  project_id: "project-1",
  title: "Untitled",
  workflow: "narration",
  status: "draft",
  revision_id: "rev-1",
  updated_at: "2026-06-23T12:00:00Z",
  sections: [],
  tracks: [],
  clips: [],
  settings: {},
  ...overrides
})

describe("audio studio store", () => {
  beforeEach(() => {
    useAudioStudioStore.getState().resetAudioStudio()
  })

  it("defines all four first-class workflows in priority order", () => {
    expect(AUDIO_STUDIO_WORKFLOWS.map((workflow) => workflow.id)).toEqual([
      "narration",
      "podcast",
      "briefing",
      "music"
    ])
    expect(AUDIO_STUDIO_WORKFLOWS.map((workflow) => workflow.label)).toEqual([
      "Narration",
      "Podcast",
      "Briefing",
      "Music"
    ])
  })

  it("switches active workflow and tracks active project", () => {
    const store = useAudioStudioStore.getState()

    store.setActiveWorkflow("podcast")
    store.setProjects([project({ project_id: "pod-1", workflow: "podcast" })])
    store.setActiveProjectId("pod-1")

    expect(useAudioStudioStore.getState().activeWorkflow).toBe("podcast")
    expect(useAudioStudioStore.getState().activeProject?.project_id).toBe("pod-1")
  })

  it("does not keep a hidden project active after switching workflows", () => {
    const store = useAudioStudioStore.getState()
    store.setProjects([
      project({ project_id: "nar-1", workflow: "narration" }),
      project({ project_id: "music-1", workflow: "music" })
    ])
    store.setActiveProjectId("nar-1")

    store.setActiveWorkflow("music")

    expect(useAudioStudioStore.getState().activeProjectId).toBe("music-1")
    expect(useAudioStudioStore.getState().activeProject?.workflow).toBe("music")

    store.setActiveWorkflow("briefing")

    expect(useAudioStudioStore.getState().activeProjectId).toBeNull()
    expect(useAudioStudioStore.getState().activeProject).toBeNull()
  })

  it("does not overwrite local dirty project changes with a newer server revision", () => {
    const store = useAudioStudioStore.getState()
    store.setProjects([project({ title: "Local title", revision_id: "rev-1" })])
    store.markProjectDirty("project-1")

    store.upsertProjectFromServer(
      project({
        title: "Server title",
        revision_id: "rev-2",
        updated_at: "2026-06-23T12:05:00Z"
      })
    )

    const state = useAudioStudioStore.getState()
    expect(state.projects[0].title).toBe("Local title")
    expect(state.revisionConflicts["project-1"]?.incoming.title).toBe("Server title")
  })
})
