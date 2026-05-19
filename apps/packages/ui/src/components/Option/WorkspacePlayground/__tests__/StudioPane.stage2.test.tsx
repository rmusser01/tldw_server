import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import type { WorkspaceSource } from "@/types/workspace"
import { StudioPane } from "../StudioPane"
const { mockScheduleWorkspaceUndoAction, mockUndoWorkspaceAction } = vi.hoisted(
  () => ({
    mockScheduleWorkspaceUndoAction: vi.fn(),
    mockUndoWorkspaceAction: vi.fn()
  })
)

const {
  mockGenerateQuiz,
  mockCreateQuiz,
  mockCreateQuestion,
  mockGenerateFlashcardsService,
  mockListDecks,
  mockCreateDeck,
  mockCreateFlashcard,
  mockCreateFlashcardsBulk,
  mockRagSearch,
  mockSynthesizeSpeech,
  mockGenerateSlidesFromMedia,
  mockCreateChatCompletion,
  mockGetMediaDetails,
  mockUpsertWorkspace,
  mockAddArtifact,
  mockUpdateArtifactStatus,
  mockRemoveArtifact,
  mockSetIsGeneratingOutput,
  mockSetAudioSettings,
  mockCaptureToCurrentNote,
  mockGetChatModels,
  messageOptionStoreState,
  chatModelSettingsStoreState,
  mockMessageSuccess,
  mockMessageError,
  mockMessageInfo,
  workspaceStoreState
} = vi.hoisted(() => {
  const generateQuiz = vi.fn()
  const createQuiz = vi.fn()
  const createQuestion = vi.fn()
  const generateFlashcardsService = vi.fn()
  const listDecks = vi.fn()
  const createDeck = vi.fn()
  const createFlashcard = vi.fn()
  const createFlashcardsBulk = vi.fn()
  const ragSearch = vi.fn()
  const synthesizeSpeech = vi.fn()
  const generateSlidesFromMedia = vi.fn()
  const createChatCompletion = vi.fn()
  const getMediaDetails = vi.fn()
  const upsertWorkspace = vi.fn()

  const addArtifact = vi.fn()
  const updateArtifactStatus = vi.fn()
  const removeArtifact = vi.fn()
  const restoreArtifact = vi.fn()
  const setIsGeneratingOutput = vi.fn()
  const setAudioSettings = vi.fn()
  const captureToCurrentNote = vi.fn()
  const getChatModels = vi.fn()

  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageInfo = vi.fn()
  const defaultSources: WorkspaceSource[] = [
    {
      id: "source-1",
      mediaId: 101,
      title: "DSPy Prompting Talk",
      type: "video",
      status: "ready",
      addedAt: new Date("2026-02-18T00:00:00.000Z")
    }
  ]

  const state = {
    selectedSourceIds: ["source-1"],
    selectedSourceFolderIds: [] as string[],
    sources: defaultSources,
    workspaceId: "workspace-a",
    workspaceName: "Workspace A",
    getSelectedMediaIds: () => [101],
    getEffectiveSelectedSources: () =>
      state.sources.filter((source: { id: string }) =>
        state.selectedSourceIds.includes(source.id)
      ),
    getEffectiveSelectedMediaIds: () =>
      state
        .getEffectiveSelectedSources()
        .map((source: { mediaId: number }) => source.mediaId),
    generatedArtifacts: [] as Array<any>,
    isGeneratingOutput: false,
    generatingOutputType: null as any,
    workspaceTag: "workspace:test",
    studyMaterialsPolicy: "workspace",
    audioSettings: {
      provider: "tldw" as const,
      model: "kokoro",
      voice: "af_heart",
      speed: 1,
      format: "mp3" as const
    },
    addArtifact,
    updateArtifactStatus,
    removeArtifact,
    restoreArtifact,
    setIsGeneratingOutput,
    setAudioSettings,
    captureToCurrentNote,
    noteFocusTarget: null as { field: "title" | "content"; token: number } | null
  }
  const messageOptionState = {
    selectedModel: "gpt-4o-mini",
    setSelectedModel: vi.fn(),
    ragSearchMode: "hybrid" as "hybrid" | "vector" | "fts",
    setRagSearchMode: vi.fn(),
    ragTopK: 8,
    setRagTopK: vi.fn(),
    ragEnableGeneration: true,
    setRagEnableGeneration: vi.fn(),
    ragEnableCitations: true,
    setRagEnableCitations: vi.fn(),
    ragAdvancedOptions: { min_score: 0.2, enable_reranking: true } as Record<
      string,
      unknown
    >,
    setRagAdvancedOptions: vi.fn()
  }
  const chatModelSettingsState = {
    apiProvider: undefined as string | undefined,
    temperature: 0.7,
    topP: 1,
    numPredict: 800,
    setApiProvider: vi.fn(),
    setTemperature: vi.fn(),
    setTopP: vi.fn(),
    setNumPredict: vi.fn(),
    updateSetting: vi.fn()
  }

  return {
    mockGenerateQuiz: generateQuiz,
    mockCreateQuiz: createQuiz,
    mockCreateQuestion: createQuestion,
    mockGenerateFlashcardsService: generateFlashcardsService,
    mockListDecks: listDecks,
    mockCreateDeck: createDeck,
    mockCreateFlashcard: createFlashcard,
    mockCreateFlashcardsBulk: createFlashcardsBulk,
    mockRagSearch: ragSearch,
    mockSynthesizeSpeech: synthesizeSpeech,
    mockGenerateSlidesFromMedia: generateSlidesFromMedia,
    mockCreateChatCompletion: createChatCompletion,
    mockGetMediaDetails: getMediaDetails,
    mockUpsertWorkspace: upsertWorkspace,
    mockAddArtifact: addArtifact,
    mockUpdateArtifactStatus: updateArtifactStatus,
    mockRemoveArtifact: removeArtifact,
    mockSetIsGeneratingOutput: setIsGeneratingOutput,
    mockSetAudioSettings: setAudioSettings,
    mockCaptureToCurrentNote: captureToCurrentNote,
    mockGetChatModels: getChatModels,
    messageOptionStoreState: messageOptionState,
    chatModelSettingsStoreState: chatModelSettingsState,
    mockMessageSuccess: messageSuccess,
    mockMessageError: messageError,
    mockMessageInfo: messageInfo,
    workspaceStoreState: state
  }
})
let isMobile = false

const interpolate = (
  template: string,
  values: Record<string, unknown> | undefined
) =>
  template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_match, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => isMobile
}))

vi.mock("../StudioPane/QuickNotesSection", () => ({
  QuickNotesSection: () => <div data-testid="quick-notes" />
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceStudioNoSourcesHint: () => "Select sources first"
}))

vi.mock("@/types/workspace", () => ({
  OUTPUT_TYPES: []
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn().mockResolvedValue([])
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn().mockReturnValue("kokoro")
}))

vi.mock("@/services/quizzes", () => ({
  generateQuiz: mockGenerateQuiz,
  createQuiz: mockCreateQuiz,
  createQuestion: mockCreateQuestion
}))

vi.mock("@/services/flashcards", () => ({
  generateFlashcards: mockGenerateFlashcardsService,
  listDecks: mockListDecks,
  createDeck: mockCreateDeck,
  createFlashcard: mockCreateFlashcard,
  createFlashcardsBulk: mockCreateFlashcardsBulk
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    ragSearch: mockRagSearch,
    synthesizeSpeech: mockSynthesizeSpeech,
    generateSlidesFromMedia: mockGenerateSlidesFromMedia,
    listVisualStyles: vi.fn().mockResolvedValue([]),
    createChatCompletion: mockCreateChatCompletion,
    getMediaDetails: mockGetMediaDetails,
    upsertWorkspace: mockUpsertWorkspace,
    exportPresentation: vi.fn(),
    downloadOutput: vi.fn()
  }
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getChatModels: mockGetChatModels,
    getProviderDisplayName: (provider: string) => provider
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof messageOptionStoreState) => unknown
  ) => selector(messageOptionStoreState)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (
    selector: (state: typeof chatModelSettingsStoreState) => unknown
  ) => selector(chatModelSettingsStoreState)
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/components/Common/Mermaid", () => ({
  default: ({ code }: { code: string }) => <div data-testid="mermaid">{code}</div>
}))

vi.mock("../undo-manager", () => ({
  WORKSPACE_UNDO_WINDOW_MS: 10000,
  scheduleWorkspaceUndoAction: mockScheduleWorkspaceUndoAction,
  undoWorkspaceAction: mockUndoWorkspaceAction
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      useMessage: () => [
        {
          open: vi.fn(),
          destroy: vi.fn(),
          success: mockMessageSuccess,
          error: mockMessageError,
          info: mockMessageInfo,
          warning: vi.fn()
        },
        <></>
      ]
    }
  }
})

if (!(globalThis as unknown as { ResizeObserver?: unknown }).ResizeObserver) {
  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

Object.defineProperty(HTMLMediaElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined)
})

Object.defineProperty(HTMLMediaElement.prototype, "pause", {
  configurable: true,
  value: vi.fn()
})

const expandOutputTypesSection = () => {
  const toggle = screen.getByRole("button", { name: /Output Types/i })
  if (toggle.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

const expandGeneratedOutputsSection = () => {
  const toggle = screen.getByRole("button", { name: /Generated Outputs/i })
  if (toggle.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

const expandMoreOutputsSection = () => {
  const toggle = screen.queryByRole("button", {
    name: /More outputs/i
  })
  if (toggle?.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

const renderStudioPane = () => {
  const renderResult = render(<StudioPane />)
  expandOutputTypesSection()
  expandGeneratedOutputsSection()
  return renderResult
}

describe("StudioPane Stage 2 workflows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Modal.destroyAll()
    localStorage.removeItem("tldw:workspace-playground:recent-output-types:v1")
    isMobile = false
    mockUndoWorkspaceAction.mockReturnValue(true)
    mockScheduleWorkspaceUndoAction.mockImplementation(
      ({
        apply
      }: {
        apply: () => void
        undo: () => void
      }) => {
        apply()
        return { id: "undo-1", expiresAt: Date.now() + 10000 }
      }
    )

    workspaceStoreState.selectedSourceIds = ["source-1"]
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceName = "Workspace A"
    workspaceStoreState.getSelectedMediaIds = () => [101]
    workspaceStoreState.generatedArtifacts = []
    workspaceStoreState.isGeneratingOutput = false
    workspaceStoreState.generatingOutputType = null
    workspaceStoreState.noteFocusTarget = null
    workspaceStoreState.studyMaterialsPolicy = "workspace"
    mockCaptureToCurrentNote.mockReset()
    messageOptionStoreState.selectedModel = "gpt-4o-mini"
    chatModelSettingsStoreState.apiProvider = undefined
    chatModelSettingsStoreState.temperature = 0.7
    chatModelSettingsStoreState.topP = 1
    chatModelSettingsStoreState.numPredict = 800

    let artifactCounter = 0
    mockAddArtifact.mockImplementation((artifactData: any) => {
      artifactCounter += 1
      const artifact = {
        ...artifactData,
        id: `artifact-${artifactCounter}`,
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
      workspaceStoreState.generatedArtifacts = [artifact, ...workspaceStoreState.generatedArtifacts]
      return artifact
    })

    mockUpdateArtifactStatus.mockImplementation(
      (id: string, status: string, updates: Record<string, unknown> = {}) => {
        workspaceStoreState.generatedArtifacts = workspaceStoreState.generatedArtifacts.map(
          (artifact) =>
            artifact.id === id
              ? {
                  ...artifact,
                  status,
                  ...updates
                }
              : artifact
        )
      }
    )

    mockSetIsGeneratingOutput.mockImplementation((isGenerating: boolean, outputType: string | null = null) => {
      workspaceStoreState.isGeneratingOutput = isGenerating
      workspaceStoreState.generatingOutputType = isGenerating ? outputType : null
    })

    mockListDecks.mockResolvedValue([])
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "Term", back: "Definition" }],
      count: 1
    })
    mockGetChatModels.mockResolvedValue([])
    mockCreateDeck.mockResolvedValue({ id: 1, name: "Workspace Flashcards" })
    mockCreateFlashcard.mockResolvedValue({ uuid: "card-1" })
    mockCreateFlashcardsBulk.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 4 }],
      count: 1,
      total: 1
    })
    mockRagSearch.mockResolvedValue({ generation: "summary" })
    mockSynthesizeSpeech.mockResolvedValue(new ArrayBuffer(8))
    mockGenerateSlidesFromMedia.mockResolvedValue({
      id: "presentation-1",
      title: "Slides",
      theme: "default",
      slides: [],
      version: 1,
      created_at: "2026-02-18T00:00:00.000Z"
    })
    mockCreateQuiz.mockResolvedValue({ id: 11, name: "Quiz", description: "" })
    mockUpsertWorkspace.mockResolvedValue({
      id: "workspace-a",
      name: "Workspace A",
      study_materials_policy: "workspace"
    })
    mockCreateQuestion.mockResolvedValue({
      id: 21,
      quiz_id: 11,
      question_type: "multiple_choice",
      question_text: "Q",
      options: ["A", "B"],
      correct_answer: "A",
      explanation: "Because",
      points: 1,
      order_index: 0,
      deleted: false,
      client_id: "test",
      version: 1
    })
  })

  afterEach(() => {
    Modal.destroyAll()
    cleanup()
  })

  it("dispatches discuss event for completed artifacts", () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-discuss",
        type: "summary",
        title: "Summary",
        status: "completed",
        content: "Discuss this summary",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    const dispatchSpy = vi.spyOn(window, "dispatchEvent")

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Discuss in chat" }))

    expect(dispatchSpy).toHaveBeenCalledTimes(1)
    const dispatchedEvent = dispatchSpy.mock.calls[0]?.[0] as CustomEvent<any>
    expect(dispatchedEvent.type).toBe("workspace-playground:discuss-artifact")
    expect(dispatchedEvent.detail).toEqual(
      expect.objectContaining({
        artifactId: "artifact-discuss",
        artifactType: "summary",
        title: "Summary",
        content: "Discuss this summary"
      })
    )
  })

  it("saves artifact content to note draft with append and replace modes", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-note",
        type: "summary",
        title: "Summary",
        status: "completed",
        content: "Artifact content for notes",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Save to notes" }))
    fireEvent.click(await screen.findByText("Append to notes"))

    expect(mockCaptureToCurrentNote).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Summary",
        content: "Artifact content for notes",
        mode: "append"
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "Save to notes" }))
    fireEvent.click(await screen.findByText("Replace note draft"))

    expect(mockCaptureToCurrentNote).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Summary",
        content: "Artifact content for notes",
        mode: "replace"
      })
    )
  })

  it("generates one quiz from the selected source bundle", async () => {
    workspaceStoreState.selectedSourceIds = ["source-1", "source-2"]
    workspaceStoreState.getSelectedMediaIds = () => [101, 202]
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-2",
        mediaId: 202,
        title: "E2E DB Media",
        type: "document",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ] as WorkspaceSource[]

    renderStudioPane()

    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "Project Falcon improved retention from 64 percent to 82 percent."
      }
    })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  title: "Workspace Quiz",
                  description: "Quiz description",
                  questions: [
                    {
                      question_type: "multiple_choice",
                      question_text: "What improved to 82 percent?",
                      options: ["Retention", "Rollout", "Revenue"],
                      correct_answer: "Retention",
                      explanation: "The source states retention improved to 82 percent."
                    }
                  ]
                })
              }
            }
          ],
          usage: {
            total_tokens: 42
          }
        })
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Quiz" }))

    await waitFor(() => {
      expect(mockCreateQuiz).toHaveBeenCalledTimes(1)
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o-mini",
        response_format: { type: "json_object" }
      }),
      expect.any(Object)
    )

    expect(mockUpsertWorkspace).toHaveBeenCalledWith("workspace-a", {
      name: "Workspace A",
      study_materials_policy: "workspace"
    })

    expect(mockCreateQuiz).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Workspace Quiz",
        description: "Quiz description",
        media_id: 101,
        source_bundle_json: [
          { source_type: "media", source_id: "101" },
          { source_type: "media", source_id: "202" }
        ],
        workspace_id: "workspace-a"
      }),
    )

    expect(mockCreateQuestion).toHaveBeenCalledTimes(1)

    expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
      expect.stringMatching(/^artifact-/),
      "completed",
      expect.objectContaining({
        serverId: 11,
        data: expect.objectContaining({
          quizId: 11,
          sourceMediaIds: [101, 202],
          sourceBundle: [
            { source_type: "media", source_id: "101" },
            { source_type: "media", source_id: "202" }
          ]
        })
      })
    )
  })

  it("keeps quiz ownership general when studyMaterialsPolicy is null", async () => {
    workspaceStoreState.selectedSourceIds = ["source-1"]
    workspaceStoreState.getSelectedMediaIds = () => [101]
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ] as WorkspaceSource[]
    workspaceStoreState.studyMaterialsPolicy = null
    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "Project Falcon improved retention from 64 percent to 82 percent."
      }
    })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  title: "Workspace Quiz",
                  questions: [
                    {
                      question_type: "multiple_choice",
                      question_text: "What improved to 82 percent?",
                      options: ["Retention", "Rollout", "Revenue"],
                      correct_answer: "Retention"
                    }
                  ]
                })
              }
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Quiz" }))

    await waitFor(() => {
      expect(mockCreateQuiz).toHaveBeenCalledTimes(1)
    })

    expect(mockUpsertWorkspace).not.toHaveBeenCalled()
    expect(mockCreateQuiz.mock.calls[0]?.[0]).not.toHaveProperty("workspace_id")
    expect(mockCreateQuiz.mock.calls[0]?.[0]).not.toHaveProperty("workspace_tag")
    expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
      expect.stringMatching(/^artifact-/),
      "completed",
      expect.objectContaining({
        data: expect.objectContaining({
          workspaceId: null
        })
      })
    )
  })

  it("renders mind map diagrams from fenced mermaid content", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-mindmap",
        type: "mindmap",
        title: "Mind Map",
        status: "completed",
        content:
          "```mermaid\nmindmap\n  root((Workspace))\n    Findings\n```",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "View" }))

    expect(await screen.findByTestId("mermaid")).toHaveTextContent(
      /mindmap\s+root\(\(Workspace\)\)\s+Findings/
    )
    expect(
      await screen.findByRole("button", { name: "Export SVG" })
    ).toBeInTheDocument()
  })

  it("falls back to raw content for non-mermaid mind map output", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-mindmap-raw",
        type: "mindmap",
        title: "Mind Map",
        status: "completed",
        content: "This output could not be converted into Mermaid markup.",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "View" }))

    expect(
      await screen.findByText(/Unable to render this mind map as a diagram/)
    ).toBeInTheDocument()
    expect(
      await screen.findByText("This output could not be converted into Mermaid markup.")
    ).toBeInTheDocument()
  })

  it("parses markdown tables and exports CSV", async () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:table")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})
    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {})

    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-table",
        type: "data_table",
        title: "Data Table",
        status: "completed",
        content:
          "| Name | Score |\n|---|---|\n| Alice | 89 |\n| Bob | 95 |",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByRole("button", { name: "View" }))

    fireEvent.change(await screen.findByPlaceholderText("Filter table rows"), {
      target: { value: "Bob" }
    })

    expect(screen.getByText("Bob")).toBeInTheDocument()
    expect(screen.queryByText("Alice")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Export CSV" }))

    await waitFor(() => {
      expect(createObjectUrlSpy).toHaveBeenCalled()
    })
    expect(anchorClickSpy).toHaveBeenCalled()
    expect(revokeObjectUrlSpy).toHaveBeenCalled()

    const csvBlob = createObjectUrlSpy.mock.calls.at(-1)?.[0] as Blob & {
      type?: string
    }
    expect(csvBlob).toBeTruthy()
    expect(csvBlob.type).toContain("text/csv")

    createObjectUrlSpy.mockRestore()
    revokeObjectUrlSpy.mockRestore()
    anchorClickSpy.mockRestore()
  })

  it("saves flashcard edits back into artifact content and structured data", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-flashcards",
        type: "flashcards",
        title: "Flashcards",
        status: "completed",
        content: "Front: Old front\nBack: Old back",
        data: {
          flashcards: [{ front: "Old front", back: "Old back" }]
        },
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByTestId("studio-artifact-edit-artifact-flashcards"))

    fireEvent.change(await screen.findByPlaceholderText("Front (question or term)"), {
      target: { value: "Updated front" }
    })
    fireEvent.change(await screen.findByPlaceholderText("Back (answer or definition)"), {
      target: { value: "Updated back" }
    })
    const flashcardSaveButtons = screen.getAllByRole("button", { name: "Save changes" })
    fireEvent.click(flashcardSaveButtons[flashcardSaveButtons.length - 1]!)

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-flashcards",
        "completed",
        expect.objectContaining({
          content: "Front: Updated front\nBack: Updated back",
          data: expect.objectContaining({
            flashcards: [{ front: "Updated front", back: "Updated back" }]
          })
        })
      )
    })
  }, 15000)

  it("removes a flashcard draft with undo parity in editor", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-flashcards",
        type: "flashcards",
        title: "Flashcards",
        status: "completed",
        content: "Front: First\nBack: First back\n\nFront: Second\nBack: Second back",
        data: {
          flashcards: [
            { front: "First", back: "First back" },
            { front: "Second", back: "Second back" }
          ]
        },
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByTestId("studio-artifact-edit-artifact-flashcards"))

    const firstFrontInput = await screen.findByDisplayValue("First")
    const flashcardsEditor = firstFrontInput.closest(
      ".ant-modal-confirm-content"
    ) as HTMLElement
    fireEvent.click(
      within(
        firstFrontInput.closest(".rounded.border") as HTMLElement
      ).getByRole("button", { name: "Remove" })
    )

    await waitFor(() => {
      expect(
        within(flashcardsEditor).queryByDisplayValue("First")
      ).not.toBeInTheDocument()
    })
    expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalled()

    const scheduledConfig =
      mockScheduleWorkspaceUndoAction.mock.calls.at(-1)?.[0]
    expect(scheduledConfig).toBeDefined()
    ;(scheduledConfig as { undo: () => void }).undo()

    await waitFor(() => {
      expect(within(flashcardsEditor).getByDisplayValue("First")).toBeInTheDocument()
    })
  })

  it("saves quiz edits back into artifact content and structured data", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-quiz",
        type: "quiz",
        title: "Quiz",
        status: "completed",
        content:
          "Quiz: Quiz\nTotal Questions: 1\n\nQ1: Old question\n  A. Old option\nAnswer: Old answer\nExplanation: Old explanation\n",
        data: {
          questions: [
            {
              question: "Old question",
              options: ["Old option"],
              answer: "Old answer",
              explanation: "Old explanation"
            }
          ]
        },
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByTestId("studio-artifact-edit-artifact-quiz"))

    fireEvent.change(await screen.findByPlaceholderText("Question prompt"), {
      target: { value: "Updated question" }
    })
    fireEvent.change(await screen.findByPlaceholderText("Options (one per line)"), {
      target: { value: "Option A\nOption B" }
    })
    fireEvent.change(await screen.findByPlaceholderText("Correct answer"), {
      target: { value: "Option A" }
    })
    fireEvent.change(await screen.findByPlaceholderText("Explanation (optional)"), {
      target: { value: "Updated explanation" }
    })
    const quizSaveButtons = screen.getAllByRole("button", { name: "Save changes" })
    fireEvent.click(quizSaveButtons[quizSaveButtons.length - 1]!)

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-quiz",
        "completed",
        expect.objectContaining({
          content: expect.stringContaining("Q1: Updated question"),
          data: expect.objectContaining({
            questions: [
              {
                question: "Updated question",
                options: ["Option A", "Option B"],
                answer: "Option A",
                explanation: "Updated explanation"
              }
            ]
          })
        })
      )
    })
  }, 15000)

  it("removes a quiz draft question with undo parity in editor", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-quiz",
        type: "quiz",
        title: "Quiz",
        status: "completed",
        content:
          "Quiz: Quiz\nTotal Questions: 2\n\nQ1: First question\nAnswer: First answer\n\nQ2: Second question\nAnswer: Second answer\n",
        data: {
          questions: [
            {
              question: "First question",
              options: ["A"],
              answer: "First answer",
              explanation: ""
            },
            {
              question: "Second question",
              options: ["B"],
              answer: "Second answer",
              explanation: ""
            }
          ]
        },
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByTestId("studio-artifact-edit-artifact-quiz"))

    const firstQuestionInput = await screen.findByDisplayValue("First question")
    const quizEditor = firstQuestionInput.closest(
      ".ant-modal-confirm-content"
    ) as HTMLElement
    fireEvent.click(
      within(
        firstQuestionInput.closest(".rounded.border") as HTMLElement
      ).getByRole("button", { name: "Remove" })
    )

    await waitFor(() => {
      expect(
        within(quizEditor).queryByDisplayValue("First question")
      ).not.toBeInTheDocument()
    })
    expect(mockScheduleWorkspaceUndoAction).toHaveBeenCalled()

    const scheduledConfig =
      mockScheduleWorkspaceUndoAction.mock.calls.at(-1)?.[0]
    expect(scheduledConfig).toBeDefined()
    ;(scheduledConfig as { undo: () => void }).undo()

    await waitFor(() => {
      expect(within(quizEditor).getByDisplayValue("First question")).toBeInTheDocument()
    })
  })

  it("uses structured flashcard generation with one scoped deck and bulk saves", async () => {
    mockListDecks.mockResolvedValue([
      { id: 4, name: "Biology Deck", card_count: 0, created_at: null, updated_at: null }
    ])
    mockGetChatModels.mockResolvedValue([
      {
        id: "gpt-4o-mini",
        name: "GPT-4o mini",
        provider: "openai"
      }
    ])
    mockGetMediaDetails.mockResolvedValue({
      content: "ATP powers cellular respiration in cells."
    })
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "ATP", back: "Cellular energy" }],
      count: 1
    })

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Audio Settings" }))
    const autoDeckLabel = await screen.findByText("Auto (create new deck)")
    fireEvent.mouseDown(autoDeckLabel.closest(".ant-select-selector") || autoDeckLabel)
    fireEvent.click(await screen.findByText("Biology Deck"))

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockCreateFlashcardsBulk).toHaveBeenCalledTimes(1)
    })

    expect(mockGenerateFlashcardsService).toHaveBeenCalledWith(
      expect.objectContaining({
        text: expect.stringContaining("DSPy Prompting Talk"),
        model: "gpt-4o-mini",
        provider: "openai"
      })
    )
    expect(mockGenerateFlashcardsService).not.toHaveBeenCalledWith(
      expect.objectContaining({
        text: ""
      })
    )
    expect(mockRagSearch).not.toHaveBeenCalled()
    expect(mockCreateFlashcardsBulk).toHaveBeenCalledWith([
      expect.objectContaining({
        deck_id: 4,
        front: "ATP",
        back: "Cellular energy",
        source_ref_id: "101"
      })
    ], expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(mockCreateFlashcard).not.toHaveBeenCalled()
    expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
      expect.stringMatching(/^artifact-/),
      "completed",
      expect.objectContaining({
        serverId: 4,
        data: expect.objectContaining({
          deckId: 4,
          sourceMediaIds: [101]
        })
      })
    )
  }, 15000)

  it("creates a fresh general deck for auto flashcard generation", async () => {
    workspaceStoreState.studyMaterialsPolicy = null
    mockListDecks.mockResolvedValue([
      { id: 4, name: "Biology Deck" },
      { id: 9, name: "Chemistry Deck" }
    ])
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "ATP", back: "Cellular energy" }],
      count: 1
    })
    mockCreateDeck.mockResolvedValue({
      id: 12,
      name: "Workspace A Flashcards - DSPy Prompting Talk"
    })

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockCreateDeck).toHaveBeenCalledTimes(1)
    })

    expect(mockUpsertWorkspace).not.toHaveBeenCalled()
    expect(mockCreateDeck).toHaveBeenCalledWith(
      {
        name: "Workspace A Flashcards - DSPy Prompting Talk"
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
    expect(mockCreateFlashcardsBulk).toHaveBeenCalledWith([
      expect.objectContaining({
        deck_id: 12,
        front: "ATP",
        back: "Cellular energy"
      })
    ], expect.objectContaining({ signal: expect.any(AbortSignal) }))
  }, 15000)

  it("falls back to per-card flashcard saves when bulk save rejects", async () => {
    mockListDecks.mockResolvedValue([])
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [
        { front: "ATP", back: "Cellular energy" },
        { front: "ADP", back: "Lower energy" }
      ],
      count: 2
    })
    mockCreateDeck.mockResolvedValue({
      id: 9,
      name: "Workspace A Flashcards"
    })
    mockCreateFlashcardsBulk.mockRejectedValueOnce(new Error("Bulk flashcard save failed"))
    mockCreateFlashcard
      .mockResolvedValueOnce({ uuid: "card-a" })
      .mockRejectedValueOnce(new Error("Second card failed"))

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
    expect(mockCreateFlashcardsBulk).toHaveBeenCalledTimes(1)
    })

    await waitFor(() => {
      expect(mockCreateFlashcard).toHaveBeenCalledTimes(2)
    })

    expect(mockCreateFlashcardsBulk).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )

    expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
      expect.stringMatching(/^artifact-/),
      "completed",
      expect.objectContaining({
        serverId: 9,
        content: expect.stringContaining("Created 1 of 2 flashcards (1 failed)"),
        data: expect.objectContaining({
          deckId: 9,
          sourceMediaIds: [101]
        })
      })
    )
  }, 15000)

  it("does not fall back to per-card saves when bulk flashcard save aborts", async () => {
    mockListDecks.mockResolvedValue([])
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "ATP", back: "Cellular energy" }],
      count: 1
    })
    mockCreateDeck.mockResolvedValue({
      id: 12,
      name: "Workspace A Flashcards"
    })
    const abortError = new Error("Aborted")
    abortError.name = "AbortError"
    mockCreateFlashcardsBulk.mockRejectedValueOnce(abortError)

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockCreateFlashcardsBulk).toHaveBeenCalledTimes(1)
    })

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        expect.stringMatching(/^artifact-/),
        "failed",
        expect.objectContaining({
          errorMessage: "Generation canceled before completion."
        })
      )
    })

    expect(mockCreateFlashcard).not.toHaveBeenCalled()
  }, 15000)

  it("falls back to the first available chat model for flashcards when no model is selected", async () => {
    messageOptionStoreState.selectedModel = null
    mockGetChatModels.mockResolvedValue([
      {
        id: "gpt-4o-mini",
        provider: "openai",
        name: "GPT-4o mini",
        context_window: 128000,
        max_output_tokens: 16000,
        supports_vision: false
      }
    ])
    mockGetMediaDetails.mockResolvedValue({
      content: "ATP powers cellular respiration in cells."
    })

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockGenerateFlashcardsService).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4o-mini",
          provider: "openai"
        })
      )
    })
  }, 15000)

  it("disables compare sources generation when fewer than two sources are selected", () => {
    workspaceStoreState.selectedSourceIds = ["source-1"]
    workspaceStoreState.getSelectedMediaIds = () => [101]

    renderStudioPane()
    expandMoreOutputsSection()

    const compareButton = screen.getByRole("button", { name: "Compare Sources" })
    expect(compareButton).toBeDisabled()
  }, 15000)

  it("generates compare sources output with usage metrics", async () => {
    workspaceStoreState.selectedSourceIds = ["source-1", "source-2"]
    workspaceStoreState.getSelectedMediaIds = () => [101, 202]
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-2",
        mediaId: 202,
        title: "E2E DB Media",
        type: "document",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ] as WorkspaceSource[]
    mockGetMediaDetails
      .mockResolvedValueOnce({
        source: { title: "DSPy Prompting Talk" },
        content: {
          text: "Alpha reports retention improved by 18 percent after the Falcon rollout."
        }
      })
      .mockResolvedValueOnce({
        source: { title: "E2E DB Media" },
        content: {
          text: "Beta reports retention improved by 12 percent and attributes gains to training."
        }
      })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "## Agreements\n- Both sources report retention gains.\n\n## Disagreements\n- Alpha reports 18 percent while Beta reports 12 percent."
              }
            }
          ],
          usage: {
            total_tokens: 321,
            total_cost_usd: 0.12
          }
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Compare Sources" }))

    await waitFor(() => {
      expect(mockCreateChatCompletion).toHaveBeenCalled()
    })

    const compareRequest = mockCreateChatCompletion.mock.calls[0]?.[0]
    expect(compareRequest).toMatchObject({
      model: "gpt-4o-mini",
      messages: [
        expect.objectContaining({
          role: "system",
          content: expect.stringContaining("source-grounded comparison analyst")
        }),
        expect.objectContaining({
          role: "user",
          content: expect.stringContaining("Alpha reports retention improved by 18 percent")
        })
      ]
    })
    expect(mockRagSearch).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        expect.stringMatching(/^artifact-/),
        "completed",
        expect.objectContaining({
          content: expect.stringContaining("## Agreements"),
          totalTokens: 321,
          totalCostUsd: 0.12
        })
      )
    })
  }, 15000)

  it("generates data table output from selected source content via chat completion", async () => {
    workspaceStoreState.selectedSourceIds = ["source-1", "source-2"]
    workspaceStoreState.getSelectedMediaIds = () => [101, 202]
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-2",
        mediaId: 202,
        title: "E2E DB Media",
        type: "document",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ] as WorkspaceSource[]
    mockGetMediaDetails
      .mockResolvedValueOnce({
        source: { title: "DSPy Prompting Talk" },
        content: {
          text: "DSPy helps optimize prompting workflows and compound AI pipelines."
        }
      })
      .mockResolvedValueOnce({
        source: { title: "E2E DB Media" },
        content: {
          text: "Hello world from E2E document processing."
        }
      })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "```mermaid\nmindmap\n  root((Workspace Research))\n    Prompting\n      DSPy\n    Documents\n      E2E DB Media\n```"
              }
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Mind Map" }))

    await waitFor(() => {
      expect(mockGetMediaDetails).toHaveBeenCalledTimes(2)
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o-mini",
        messages: [
          expect.objectContaining({
            role: "system",
            content: expect.stringContaining("Mermaid")
          }),
          expect.objectContaining({
            role: "user",
            content: expect.stringContaining("DSPy Prompting Talk")
          })
        ]
      }),
      expect.objectContaining({
        signal: expect.any(AbortSignal)
      })
    )
    expect(mockRagSearch).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        expect.stringMatching(/^artifact-/),
        "completed",
        expect.objectContaining({
          content: expect.stringContaining("mindmap"),
          data: expect.objectContaining({
            mermaid: expect.stringContaining("mindmap")
          })
        })
      )
    })
  }, 15000)

  it("falls back to the first available chat model for mind maps when no model is selected", async () => {
    messageOptionStoreState.selectedModel = null
    mockGetChatModels.mockResolvedValue([
      {
        id: "gpt-4o-mini",
        name: "GPT-4o mini",
        provider: "openai"
      },
      {
        id: "claude-3-5-sonnet",
        name: "Claude 3.5 Sonnet",
        provider: "anthropic"
      }
    ])
    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "DSPy helps optimize prompting workflows and compound AI pipelines."
      }
    })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: "```mermaid\nmindmap\n  root((Workspace))\n    DSPy\n```"
              }
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Mind Map" }))

    await waitFor(() => {
      expect(mockCreateChatCompletion).toHaveBeenCalled()
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o-mini"
      }),
      expect.objectContaining({
        signal: expect.any(AbortSignal)
      })
    )
  }, 15000)

  it("marks mind map generation failed when completion is not Mermaid syntax", async () => {
    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "DSPy helps optimize prompting workflows and compound AI pipelines."
      }
    })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "Central topic: Workspace Research\n- Prompting workflows\n- Compound AI pipelines"
              }
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Mind Map" }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        expect.stringMatching(/^artifact-/),
        "failed",
        expect.objectContaining({
          errorMessage: expect.stringContaining("usable mind map")
        })
      )
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledTimes(1)
    expect(mockMessageError).toHaveBeenCalled()
  }, 15000)

  it("generates data table output from selected source content via chat completion", async () => {
    workspaceStoreState.selectedSourceIds = ["source-1", "source-2"]
    workspaceStoreState.getSelectedMediaIds = () => [101, 202]
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "DSPy Prompting Talk",
        type: "video",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-2",
        mediaId: 202,
        title: "E2E DB Media",
        type: "document",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ] as WorkspaceSource[]
    mockGetMediaDetails
      .mockResolvedValueOnce({
        source: { title: "DSPy Prompting Talk" },
        content: {
          text: "DSPy helps optimize prompting workflows and compound AI pipelines."
        }
      })
      .mockResolvedValueOnce({
        source: { title: "E2E DB Media" },
        content: {
          text: "Hello world from E2E document processing."
        }
      })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "| Source | Fact |\n|---|---|\n| DSPy Prompting Talk | Prompt optimization |\n| E2E DB Media | Hello world |"
              }
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Data Table" }))

    await waitFor(() => {
      expect(mockGetMediaDetails).toHaveBeenCalledTimes(2)
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o-mini",
        temperature: 0.7,
        top_p: 1,
        max_tokens: 800,
        messages: [
          expect.objectContaining({
            role: "system",
            content: expect.stringContaining("Return ONLY a markdown table")
          }),
          expect.objectContaining({
            role: "user",
            content: expect.stringContaining("DSPy Prompting Talk")
          })
        ]
      })
    )
    expect(mockRagSearch).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        expect.stringMatching(/^artifact-/),
        "completed",
        expect.objectContaining({
          content: expect.stringContaining("| Source | Fact |"),
          data: expect.objectContaining({
            table: expect.objectContaining({
              headers: ["Source", "Fact"]
            })
          })
        })
      )
    })
  }, 15000)

  it("falls back to the first available chat model for data tables when no model is selected", async () => {
    messageOptionStoreState.selectedModel = null
    mockGetChatModels.mockResolvedValue([
      {
        id: "gpt-4o-mini",
        name: "GPT-4o mini",
        provider: "openai"
      },
      {
        id: "claude-3-5-sonnet",
        name: "Claude 3.5 Sonnet",
        provider: "anthropic"
      }
    ])
    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "DSPy helps optimize prompting workflows and compound AI pipelines."
      }
    })
    mockCreateChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "| Source | Fact |\n|---|---|\n| DSPy Prompting Talk | Prompt optimization |"
              }
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      )
    )

    renderStudioPane()
    expandMoreOutputsSection()

    fireEvent.click(screen.getByRole("button", { name: "Data Table" }))

    await waitFor(() => {
      expect(mockCreateChatCompletion).toHaveBeenCalled()
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o-mini"
      })
    )
  }, 15000)

  it("renders cumulative workspace usage and per-artifact usage", async () => {
    Modal.destroyAll()
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-usage-a",
        type: "summary",
        title: "Summary",
        status: "completed",
        content: "A",
        totalTokens: 150,
        totalCostUsd: 0.045,
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      },
      {
        id: "artifact-usage-b",
        type: "report",
        title: "Report",
        status: "completed",
        content: "B",
        estimatedTokens: 250,
        estimatedCostUsd: 0.075,
        createdAt: new Date("2026-02-18T10:01:00.000Z")
      }
    ]

    renderStudioPane()

    await waitFor(() => {
      expect(
        screen.getByText((content) => content.includes("Estimated workspace usage:"))
      ).toBeInTheDocument()
      expect(
        screen.getAllByText(/Tokens:/).length
      ).toBeGreaterThanOrEqual(1)
      expect(
        screen.getAllByText(/Cost:/).length
      ).toBeGreaterThanOrEqual(1)
    })
  })

  it("requests voice preview audio from TTS provider", async () => {
    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Audio Settings" }))
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    await waitFor(() => {
      expect(mockSynthesizeSpeech).toHaveBeenCalledWith(
        "This is a quick voice preview from your current audio settings.",
        expect.objectContaining({
          model: "kokoro",
          voice: "af_heart",
          responseFormat: "mp3",
          speed: 1
        })
      )
    })
  }, 15000)

  it("uses fullscreen modal sizing when viewing outputs on mobile", () => {
    isMobile = true
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-summary-mobile",
        type: "summary",
        title: "Summary",
        status: "completed",
        content: "Mobile view content",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    const modalInfoSpy = vi
      .spyOn(Modal, "info")
      .mockImplementation(
        () =>
          ({
            destroy: vi.fn(),
            update: vi.fn()
          }) as any
      )

    renderStudioPane()
    fireEvent.click(screen.getByRole("button", { name: "View" }))

    expect(modalInfoSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        width: "100%",
        style: expect.objectContaining({ top: 0, paddingBottom: 0 }),
        styles: expect.objectContaining({
          body: expect.objectContaining({
            maxHeight: "calc(100dvh - 96px)",
            overflowY: "auto"
          })
        })
      })
    )

    modalInfoSpy.mockRestore()
  }, 15000)
})
