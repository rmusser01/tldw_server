import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { AudiobookStudioPage } from "../AudiobookStudioPage"
import { OutputPanel } from "../Output/OutputPanel"
import {
  useAudiobookStudioStore,
  type AudioChapter
} from "@/store/audiobook-studio"

const mocks = vi.hoisted(() => ({
  saveProject: vi.fn(async () => undefined),
  createNewProject: vi.fn(async () => "new-project"),
  downloadChapter: vi.fn(),
  downloadAllChapters: vi.fn(),
  generateAllChapters: vi.fn(async () => undefined),
  cancelGeneration: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string, values?: Record<string, unknown>) =>
      fallback.replace(/\{\{(\w+)\}\}/g, (_match, key: string) =>
        String(values?.[key] ?? "")
      )
  })
}))

vi.mock("@/components/Common/DismissibleBetaAlert", () => ({
  DismissibleBetaAlert: ({
    message,
    description,
    className
  }: {
    message: React.ReactNode
    description?: React.ReactNode
    className?: string
  }) => (
    <aside className={className}>
      <strong>{message}</strong>
      {description ? <p>{description}</p> : null}
    </aside>
  )
}))

vi.mock("@/hooks/useAudiobookProjects", () => ({
  useCurrentProject: () => ({
    saveProject: mocks.saveProject,
    createNewProject: mocks.createNewProject
  })
}))

vi.mock("@/hooks/useAudiobookGeneration", () => ({
  useAudiobookGeneration: () => ({
    downloadChapter: mocks.downloadChapter,
    downloadAllChapters: mocks.downloadAllChapters,
    generateAllChapters: mocks.generateAllChapters,
    cancelGeneration: mocks.cancelGeneration
  })
}))

vi.mock("../ContentInput/TextEditor", () => ({
  TextEditor: () => <section data-testid="content-workflow">Content workflow</section>
}))

vi.mock("../ChapterEditor/ChapterList", () => ({
  ChapterList: () => <section data-testid="chapter-workflow">Chapter workflow</section>
}))

vi.mock("../Generation/GenerationPanel", () => ({
  GenerationPanel: () => (
    <section data-testid="generation-workflow">Generation workflow</section>
  )
}))

vi.mock("../ProjectManagement/ProjectListView", () => ({
  ProjectListView: () => (
    <section data-testid="project-list-workflow">Project list workflow</section>
  )
}))

vi.mock("../ProjectManagement/ProjectMetadataForm", () => ({
  ProjectMetadataForm: ({ open }: { open: boolean }) =>
    open ? <section data-testid="metadata-form">Metadata form</section> : null
}))

const baseState = {
  currentProjectId: null,
  rawContent: "",
  chapters: [] as AudioChapter[],
  isGenerating: false,
  generationQueue: [] as string[],
  currentGeneratingId: null,
  defaultVoiceConfig: {},
  projectTitle: "Untitled Audiobook",
  projectAuthor: "",
  projectDescription: "",
  projectCoverImageUrl: null
}

const resetStore = (overrides: Partial<typeof baseState> = {}) => {
  useAudiobookStudioStore.setState({
    ...baseState,
    ...overrides
  })
}

const makeChapter = (
  id: string,
  status: AudioChapter["status"],
  extra: Partial<AudioChapter> = {}
): AudioChapter => ({
  id,
  title: `Chapter ${id}`,
  content: `Content ${id}`,
  order: Number(id.replace(/\D/g, "")) || 0,
  voiceConfig: {},
  status,
  ...extra
})

describe("AudiobookStudioPage", () => {
  beforeEach(() => {
    mocks.saveProject.mockClear()
    mocks.createNewProject.mockClear()
    resetStore()
  })

  afterEach(() => {
    resetStore()
  })

  it("shows a recoverable draft state instead of claiming an unsaved new project is saved", () => {
    render(<AudiobookStudioPage />)

    expect(screen.getByRole("heading", { name: "Audiobook Studio" })).toBeVisible()
    expect(screen.getByText("Beta Feature")).toBeVisible()
    expect(screen.getByRole("button", { name: "My Projects" })).toBeVisible()
    expect(screen.getByRole("button", { name: "New" })).toBeVisible()
    expect(screen.getByDisplayValue("Untitled Audiobook")).toBeVisible()

    expect(
      screen.getByRole("status", { name: "Project save status" })
    ).toHaveTextContent("Draft not saved")
    expect(
      screen.getByRole("button", { name: "Save project" })
    ).toBeVisible()
  })

  it("keeps dense project, chapter, generation, and output status visible for returning users", async () => {
    const user = userEvent.setup()
    resetStore({
      currentProjectId: "project-1",
      rawContent: "# Intro\n\nDraft audiobook content.",
      projectTitle: "Conference Reader",
      isGenerating: true,
      currentGeneratingId: "chapter-2",
      chapters: [
        makeChapter("chapter-1", "completed", {
          audioBlob: new Blob(["audio"], { type: "audio/mpeg" }),
          audioUrl: "blob:chapter-1",
          audioDuration: 90
        }),
        makeChapter("chapter-2", "pending"),
        makeChapter("chapter-3", "error")
      ]
    })

    render(<AudiobookStudioPage />)

    expect(screen.getByDisplayValue("Conference Reader")).toBeVisible()
    expect(screen.getByText("3 chapters")).toBeVisible()
    expect(screen.getByText("1 completed")).toBeVisible()
    expect(screen.getByText("2 pending")).toBeVisible()
    expect(screen.getByText("Generating...")).toBeVisible()

    const tabs = within(screen.getByRole("tablist")).getAllByRole("tab")
    expect(tabs.map((tab) => tab.textContent?.replace(/\s+/g, "").trim())).toEqual([
      "Content",
      "Chapters3",
      "GenerateGenerating...",
      "Output1/3"
    ])

    await user.click(screen.getByRole("tab", { name: /chapters/i }))
    expect(screen.getByTestId("chapter-workflow")).toBeVisible()

    await user.click(screen.getByRole("tab", { name: /generate/i }))
    expect(screen.getByTestId("generation-workflow")).toBeVisible()

    await user.click(screen.getByRole("tab", { name: /output/i }))
    expect(screen.getByText("1 chapters ready")).toBeVisible()
  })

  it("marks edited projects as unsaved and confirms manual save recovery", async () => {
    const user = userEvent.setup()
    resetStore({
      currentProjectId: "project-1",
      rawContent: "A changed manuscript",
      projectTitle: "Changed Project"
    })

    render(<AudiobookStudioPage />)

    const saveStatus = screen.getByRole("status", {
      name: "Project save status"
    })

    await waitFor(() => expect(saveStatus).toHaveTextContent("Unsaved changes"))

    await user.click(screen.getByRole("button", { name: "Save project" }))

    expect(mocks.saveProject).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(saveStatus).toHaveTextContent("Saved just now"))
  })

  it("clears stale saved status when switching away from a saved project", async () => {
    const user = userEvent.setup()
    resetStore({
      currentProjectId: "project-1",
      rawContent: "Saved manuscript",
      projectTitle: "Saved Project"
    })

    render(<AudiobookStudioPage />)

    const saveStatus = screen.getByRole("status", {
      name: "Project save status"
    })

    await waitFor(() => expect(saveStatus).toHaveTextContent("Unsaved changes"))
    await user.click(screen.getByRole("button", { name: "Save project" }))
    await waitFor(() => expect(saveStatus).toHaveTextContent("Saved just now"))

    act(() => {
      useAudiobookStudioStore.setState({
        ...baseState,
        currentProjectId: null
      })
    })

    await waitFor(() => expect(saveStatus).toHaveTextContent("Draft not saved"))
  })
})

describe("Audiobook OutputPanel", () => {
  beforeEach(() => {
    resetStore()
  })

  it("sets the expectation that generated chapters appear after generation", () => {
    render(<OutputPanel />)

    expect(
      screen.getByText("Generated chapters appear here after generation.")
    ).toBeVisible()
  })
})
