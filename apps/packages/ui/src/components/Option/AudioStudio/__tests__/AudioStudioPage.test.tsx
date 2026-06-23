import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const generationMocks = vi.hoisted(() => ({
  mutateAsync: vi.fn()
}))

const projectHookMocks = vi.hoisted(() => ({
  useProjects: vi.fn(),
  createProject: vi.fn(),
  updateProject: vi.fn(),
  upsertSection: vi.fn(),
  createPending: false,
  updatePending: false,
  upsertSectionPending: false
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: "/audio-studio", search: "", hash: "", state: null }),
  useNavigate: () => vi.fn()
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <main>{children}</main>
  )
}))

vi.mock("@/components/Option/AudiobookStudio/ContentInput/TextEditor", () => ({
  TextEditor: () => <div>Paste or type your content</div>
}))

vi.mock("@/components/Option/AudiobookStudio/ChapterEditor/ChapterList", () => ({
  ChapterList: () => <div>Chapter List</div>
}))

vi.mock("@/components/Option/AudiobookStudio/Generation/GenerationPanel", () => ({
  GenerationPanel: () => <div>Voice Settings</div>
}))

vi.mock("@/components/Option/AudiobookStudio/Output/OutputPanel", () => ({
  OutputPanel: () => <div>Audiobook Player</div>
}))

vi.mock("@/hooks/useAudioStudioGeneration", () => ({
  useCreateAudioStudioGeneration: () => ({
    mutateAsync: generationMocks.mutateAsync,
    isPending: false
  })
}))

vi.mock("@/hooks/useAudioStudioProjects", () => ({
  useAudioStudioProjects: (...args: unknown[]) =>
    projectHookMocks.useProjects(...args),
  useCreateAudioStudioProject: () => ({
    mutateAsync: projectHookMocks.createProject,
    isPending: projectHookMocks.createPending
  }),
  useUpdateAudioStudioProject: () => ({
    mutateAsync: projectHookMocks.updateProject,
    isPending: projectHookMocks.updatePending
  }),
  useUpsertAudioStudioSection: () => ({
    mutateAsync: projectHookMocks.upsertSection,
    isPending: projectHookMocks.upsertSectionPending
  })
}))

import { AudioStudioPage } from "../AudioStudioPage"
import {
  useAudioStudioStore,
  type AudioStudioProject
} from "@/store/audio-studio"

const setActiveProject = (overrides: Partial<AudioStudioProject> = {}) => {
  useAudioStudioStore.getState().setProjects([
    {
      project_id: "project-1",
      title: "Working project",
      workflow: "music",
      status: "draft",
      current_revision_id: "revision-current",
      sections: [
        {
          section_id: "section-1",
          workflow: "narration",
          title: "Intro",
          order: 0
        }
      ],
      tracks: [
        {
          track_id: "music-track-1",
          name: "Music",
          kind: "music",
          order: 0
        }
      ],
      clips: [],
      settings: {},
      ...overrides
    } satisfies AudioStudioProject
  ])
  useAudioStudioStore.getState().setActiveProjectId("project-1")
}

describe("AudioStudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    projectHookMocks.createPending = false
    projectHookMocks.updatePending = false
    projectHookMocks.upsertSectionPending = false
    projectHookMocks.upsertSection.mockResolvedValue({
      section_id: "section_draft",
      workflow: "podcast",
      title: "Podcast script",
      body_text: "Saved script",
      order_index: 0,
      current_revision_id: "revision-section-saved"
    })
    projectHookMocks.useProjects.mockReturnValue({
      isLoading: false,
      isError: false,
      error: null
    })
    useAudioStudioStore.getState().resetAudioStudio()
  })

  it("renders all workflow labels as first-class choices", () => {
    render(<AudioStudioPage />)

    expect(projectHookMocks.useProjects).toHaveBeenCalledWith({
      workflow: "narration",
      includeArchived: false
    })
    expect(screen.getByRole("heading", { name: "Audio Studio" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Narration/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Podcast/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Briefing/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Music/ })).toBeInTheDocument()
  })

  it("gives workflow tabs keyboard navigation and panel relationships", () => {
    render(<AudioStudioPage />)

    const narrationTab = screen.getByRole("tab", { name: /Narration/ })
    const podcastTab = screen.getByRole("tab", { name: /Podcast/ })
    const panel = screen.getByRole("tabpanel", { name: /Narration/ })

    expect(narrationTab).toHaveAttribute("aria-controls", "audio-studio-workflow-panel")
    expect(narrationTab).toHaveAttribute("tabindex", "0")
    expect(podcastTab).toHaveAttribute("tabindex", "-1")
    expect(panel).toHaveAttribute("id", "audio-studio-workflow-panel")

    fireEvent.keyDown(narrationTab, { key: "ArrowRight" })

    expect(useAudioStudioStore.getState().activeWorkflow).toBe("podcast")
  })

  it("shows imported audiobook controls in Narration without the old top heading", () => {
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    expect(screen.getByText("Paste or type your content")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: "Chapters" }))
    expect(screen.getByText("Chapter List")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: "Voice" }))
    expect(screen.getByText("Voice Settings")).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Audiobook Studio" })
    ).not.toBeInTheDocument()
  })

  it("surfaces Podcast and Briefing as production workflows", () => {
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const { rerender } = render(<AudioStudioPage />)

    expect(screen.getByText("Podcast script")).toBeInTheDocument()
    expect(screen.getByText("Speakers")).toBeInTheDocument()

    useAudioStudioStore.getState().setActiveWorkflow("briefing")
    rerender(<AudioStudioPage />)

    expect(screen.getByText("Briefing outline")).toBeInTheDocument()
    expect(screen.getByText("Source notes")).toBeInTheDocument()
  })

  it("shows Music prompt, lyrics, style, and provider controls", () => {
    useAudioStudioStore.getState().setActiveWorkflow("music")

    render(<AudioStudioPage />)

    expect(screen.getByLabelText("Prompt")).toBeInTheDocument()
    expect(screen.getByLabelText("Lyrics")).toBeInTheDocument()
    expect(screen.getByLabelText("Style")).toBeInTheDocument()
    expect(screen.getByLabelText("Provider")).toBeInTheDocument()
    expect(
      screen.queryByRole("option", { name: "Server default" })
    ).not.toBeInTheDocument()
  })

  it("shows non-disruptive project loading and error states", () => {
    projectHookMocks.useProjects.mockReturnValueOnce({
      isLoading: true,
      isError: false,
      error: null
    })
    const { rerender } = render(<AudioStudioPage />)

    expect(
      screen.getByText("Loading Audio Studio projects...")
    ).toBeInTheDocument()

    projectHookMocks.useProjects.mockReturnValueOnce({
      isLoading: false,
      isError: true,
      error: new Error("Nope")
    })
    rerender(<AudioStudioPage />)

    expect(
      screen.getByText("Audio Studio projects could not load.")
    ).toBeInTheDocument()
  })

  it("creates new projects through the server-backed create hook", async () => {
    useAudioStudioStore.getState().setActiveWorkflow("podcast")

    render(<AudioStudioPage />)

    fireEvent.click(screen.getByRole("button", { name: "New Audio Studio project" }))

    await waitFor(() => expect(projectHookMocks.createProject).toHaveBeenCalled())
    expect(projectHookMocks.createProject).toHaveBeenCalledWith({
      title: "Untitled Podcast Project",
      workflow: "podcast"
    })
    expect(
      useAudioStudioStore
        .getState()
        .projects.some((project) => project.revision_id === "local-draft")
    ).toBe(false)
  })

  it("selects an existing project from the project sidebar", () => {
    useAudioStudioStore.getState().setProjects([
      {
        project_id: "project-1",
        title: "First project",
        workflow: "narration",
        status: "draft",
        current_revision_id: "revision-1",
        sections: [],
        tracks: [],
        clips: [],
        settings: {}
      },
      {
        project_id: "project-2",
        title: "Second project",
        workflow: "narration",
        status: "draft",
        current_revision_id: "revision-2",
        sections: [],
        tracks: [],
        clips: [],
        settings: {}
      }
    ] satisfies AudioStudioProject[])
    useAudioStudioStore.getState().setActiveProjectId("project-1")

    render(<AudioStudioPage />)

    fireEvent.click(screen.getByRole("button", { name: /Second project/ }))

    expect(useAudioStudioStore.getState().activeProjectId).toBe("project-2")
  })

  it("saves active project edits through the update hook", async () => {
    setActiveProject({
      workflow: "briefing",
      description: "Existing description",
      settings: { voice: "Ava" }
    })
    useAudioStudioStore.getState().setActiveWorkflow("briefing")

    render(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Project title"), {
      target: { value: "Renamed project" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(projectHookMocks.updateProject).toHaveBeenCalled())
    expect(projectHookMocks.updateProject).toHaveBeenCalledWith({
      title: "Renamed project",
      description: "Existing description",
      settings: { voice: "Ava" },
      base_revision_id: "revision-current"
    })
  })

  it("disables save and generation for projects without a real revision", () => {
    setActiveProject({
      workflow: "music",
      current_revision_id: undefined,
      revision_id: "local-draft"
    })
    useAudioStudioStore.getState().setActiveWorkflow("music")

    render(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" }
    })

    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Generate music" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Queue music generation" })
    ).toBeDisabled()
  })

  it("does not keep a hidden project active after changing workflows", () => {
    setActiveProject({ workflow: "narration" })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    fireEvent.click(screen.getByRole("tab", { name: /Music/ }))
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" }
    })

    expect(useAudioStudioStore.getState().activeProject).toBeNull()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Generate music" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Generate music" }))

    expect(generationMocks.mutateAsync).not.toHaveBeenCalled()
  })

  it("keeps render and export controls disabled while TASK-2351 owns implementation", () => {
    setActiveProject({ workflow: "narration" })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    expect(
      screen.getByRole("button", { name: "Create preview render" })
    ).toBeDisabled()
    expect(screen.getByRole("button", { name: "Create export" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Create preview render" })
    ).toHaveAttribute("title", "Render/export controls land in TASK-2351.")
    expect(
      screen.getByRole("button", { name: "Create export" })
    ).toHaveAttribute("title", "Render/export controls land in TASK-2351.")
  })

  it("queues music generation with controlled Music workflow inputs", async () => {
    setActiveProject({ workflow: "music" })
    useAudioStudioStore.getState().setActiveWorkflow("music")

    render(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" }
    })
    fireEvent.change(screen.getByLabelText("Lyrics"), {
      target: { value: "Hold the first phrase" }
    })
    fireEvent.change(screen.getByLabelText("Style"), {
      target: { value: "cinematic, ambient" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate music" }))

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "music",
        provider: "ace_step",
        target_resource_kind: "track",
        target_resource_id: "music-track-1",
        target_revision_id: "revision-current",
        options: {
          prompt: "Warm documentary intro",
          lyrics: "Hold the first phrase",
          style: "cinematic, ambient",
          duration: 45
        }
      })
    )
    expect(
      generationMocks.mutateAsync.mock.calls[0][0].idempotency_key.length
    ).toBeGreaterThanOrEqual(16)
  })

  it("queues shared music generation from the side panel", async () => {
    setActiveProject({ workflow: "music" })
    useAudioStudioStore.getState().setActiveWorkflow("music")

    render(<AudioStudioPage />)

    fireEvent.click(screen.getByRole("button", { name: "Queue music generation" }))

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "music",
        provider: "ace_step",
        target_resource_kind: "track",
        target_resource_id: "music-track-1",
        target_revision_id: "revision-current"
      })
    )
  })

  it("queues shared speech generation for the first available section", async () => {
    setActiveProject({ workflow: "narration" })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    fireEvent.click(screen.getByRole("button", { name: "Queue speech generation" }))

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        provider: "tts",
        target_resource_kind: "section",
        target_resource_id: "section-1",
        target_revision_id: "revision-current"
      })
    )
  })

  it("queues Podcast and Briefing inline speech actions from their saved draft sections", async () => {
    setActiveProject({ workflow: "podcast" })
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const { rerender } = render(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Podcast script"), {
      target: { value: "Host: Welcome." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate segment speech" }))

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        provider: "tts",
        target_resource_kind: "section",
        target_resource_id: "section_podcast_script",
        target_revision_id: "revision-section-saved"
      })
    )

    generationMocks.mutateAsync.mockClear()
    projectHookMocks.upsertSection.mockResolvedValueOnce({
      section_id: "section_briefing_outline",
      workflow: "briefing",
      title: "Briefing outline",
      body_text: "Briefing text",
      order_index: 0,
      current_revision_id: "revision-briefing-saved"
    })
    setActiveProject({ workflow: "briefing" })
    useAudioStudioStore.getState().setActiveWorkflow("briefing")
    rerender(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Briefing outline"), {
      target: { value: "Briefing text" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate briefing sections" }))

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        target_resource_id: "section_briefing_outline",
        target_revision_id: "revision-briefing-saved"
      })
    )
  })

  it("saves Podcast and Briefing draft text before queuing inline speech generation", async () => {
    setActiveProject({ workflow: "podcast", sections: [] })
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const { rerender } = render(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Podcast script"), {
      target: { value: "Host: Welcome.\nGuest: Good to be here." }
    })
    fireEvent.change(screen.getByLabelText("Host speaker"), {
      target: { value: "Ava" }
    })
    fireEvent.change(screen.getByLabelText("Guest speaker"), {
      target: { value: "Noah" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate segment speech" }))

    await waitFor(() => expect(projectHookMocks.upsertSection).toHaveBeenCalled())
    expect(projectHookMocks.upsertSection).toHaveBeenCalledWith({
      sectionId: "section_podcast_script",
      payload: {
        base_revision_id: "revision-current",
        title: "Podcast script",
        body_text: "Host: Welcome.\nGuest: Good to be here.",
        order_index: 0,
        settings: {
          hostSpeaker: "Ava",
          guestSpeaker: "Noah"
        }
      }
    })
    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        target_resource_id: "section_podcast_script",
        target_revision_id: "revision-section-saved"
      })
    )

    generationMocks.mutateAsync.mockClear()
    projectHookMocks.upsertSection.mockClear()
    projectHookMocks.upsertSection.mockResolvedValueOnce({
      section_id: "section_briefing_outline",
      workflow: "briefing",
      title: "Briefing outline",
      body_text: "Top story and implications.",
      order_index: 0,
      current_revision_id: "revision-briefing-saved"
    })
    setActiveProject({ workflow: "briefing", sections: [] })
    useAudioStudioStore.getState().setActiveWorkflow("briefing")
    rerender(<AudioStudioPage />)

    fireEvent.change(screen.getByLabelText("Briefing outline"), {
      target: { value: "Top story and implications." }
    })
    fireEvent.change(screen.getByLabelText("Source notes"), {
      target: { value: "Source note A" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate briefing sections" }))

    await waitFor(() => expect(projectHookMocks.upsertSection).toHaveBeenCalled())
    expect(projectHookMocks.upsertSection).toHaveBeenCalledWith({
      sectionId: "section_briefing_outline",
      payload: {
        base_revision_id: "revision-current",
        title: "Briefing outline",
        body_text: "Top story and implications.",
        order_index: 0,
        settings: {
          sourceNotes: "Source note A"
        }
      }
    })
    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled())
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        target_resource_id: "section_briefing_outline",
        target_revision_id: "revision-briefing-saved"
      })
    )
  })

  it("disables shared speech generation without a usable section target", () => {
    setActiveProject({ workflow: "narration", sections: [] })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    expect(
      screen.getByRole("button", { name: "Queue speech generation" })
    ).toBeDisabled()
  })
})
