import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const generationMocks = vi.hoisted(() => ({
  mutateAsync: vi.fn()
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

import { AudioStudioPage } from "../AudioStudioPage"
import { useAudioStudioStore } from "@/store/audio-studio"

const setActiveProject = (overrides: Record<string, unknown> = {}) => {
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
    } as any
  ])
  useAudioStudioStore.getState().setActiveProjectId("project-1")
}

describe("AudioStudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAudioStudioStore.getState().resetAudioStudio()
  })

  it("renders all workflow labels as first-class choices", () => {
    render(<AudioStudioPage />)

    expect(screen.getByRole("heading", { name: "Audio Studio" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Narration/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Podcast/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Briefing/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Music/ })).toBeInTheDocument()
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

  it("disables shared speech generation without a usable section target", () => {
    setActiveProject({ workflow: "narration", sections: [] })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    expect(
      screen.getByRole("button", { name: "Queue speech generation" })
    ).toBeDisabled()
  })
})
