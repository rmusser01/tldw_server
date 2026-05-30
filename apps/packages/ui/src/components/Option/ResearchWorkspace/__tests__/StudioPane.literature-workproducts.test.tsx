import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { AudioGenerationSettings } from "@/types/workspace"
import { StudioPane } from "../StudioPane"

const {
  mockAddArtifact,
  mockUpdateArtifactStatus,
  mockRemoveArtifact,
  mockSetIsGeneratingOutput,
  mockSetAudioSettings,
  mockMessageSuccess,
  mockMessageError,
  mockMessageInfo,
  mockGenerateFlashcardsService,
  mockCreateFlashcardsBulk,
  mockRagSearch,
  mockSynthesizeSpeech,
  mockGenerateSlidesFromMedia,
  mockDownloadOutput,
  mockCreateChatCompletion,
  mockGetMediaDetails,
  mockUpsertWorkspace,
  mockGetChatModels,
  messageOptionStoreState,
  chatModelSettingsStoreState,
  baseAudioSettings,
  workspaceStoreState
} = vi.hoisted(() => {
  const addArtifact = vi.fn()
  const updateArtifactStatus = vi.fn()
  const removeArtifact = vi.fn()
  const restoreArtifact = vi.fn()
  const setIsGeneratingOutput = vi.fn()
  const setAudioSettings = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageInfo = vi.fn()
  const generateFlashcardsService = vi.fn()
  const createFlashcardsBulk = vi.fn()
  const ragSearch = vi.fn()
  const synthesizeSpeech = vi.fn()
  const generateSlidesFromMedia = vi.fn()
  const downloadOutput = vi.fn()
  const createChatCompletion = vi.fn()
  const getMediaDetails = vi.fn()
  const upsertWorkspace = vi.fn()
  const getChatModels = vi.fn()
  const defaultAudioSettings: AudioGenerationSettings = {
    provider: "browser",
    model: "kokoro",
    voice: "af_heart",
    speed: 1,
    format: "mp3"
  }
  const defaultSources = [
    {
      id: "source-1",
      mediaId: 101,
      title: "Paper A",
      type: "pdf" as const,
      status: "ready" as const,
      addedAt: new Date("2026-02-18T00:00:00.000Z")
    },
    {
      id: "source-2",
      mediaId: 202,
      title: "Paper B",
      type: "pdf" as const,
      status: "ready" as const,
      addedAt: new Date("2026-02-18T00:00:00.000Z")
    }
  ]
  const storeState = {
    selectedSourceIds: ["source-1", "source-2"],
    selectedSourceFolderIds: [] as string[],
    sources: defaultSources,
    getSelectedMediaIds: () =>
      storeState.sources
        .filter((source: { id: string }) =>
          storeState.selectedSourceIds.includes(source.id)
        )
        .map((source: { mediaId: number }) => source.mediaId),
    getEffectiveSelectedSources: () =>
      storeState.sources.filter((source: { id: string }) =>
        storeState.selectedSourceIds.includes(source.id)
      ),
    getEffectiveSelectedMediaIds: () =>
      storeState
        .getEffectiveSelectedSources()
        .map((source: { mediaId: number }) => source.mediaId),
    generatedArtifacts: [] as Array<any>,
    isGeneratingOutput: false,
    generatingOutputType: null as any,
    workspaceTag: "workspace:literature",
    workspaceId: "workspace-literature",
    workspaceName: "Literature Review Workspace",
    studyMaterialsPolicy: "workspace" as const,
    audioSettings: { ...defaultAudioSettings },
    addArtifact,
    updateArtifactStatus,
    removeArtifact,
    restoreArtifact,
    setIsGeneratingOutput,
    setAudioSettings,
    captureToCurrentNote: vi.fn(),
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
    numPredict: 1000,
    setApiProvider: vi.fn(),
    setTemperature: vi.fn(),
    setTopP: vi.fn(),
    setNumPredict: vi.fn(),
    updateSetting: vi.fn()
  }
  return {
    mockAddArtifact: addArtifact,
    mockUpdateArtifactStatus: updateArtifactStatus,
    mockRemoveArtifact: removeArtifact,
    mockSetIsGeneratingOutput: setIsGeneratingOutput,
    mockSetAudioSettings: setAudioSettings,
    mockMessageSuccess: messageSuccess,
    mockMessageError: messageError,
    mockMessageInfo: messageInfo,
    mockGenerateFlashcardsService: generateFlashcardsService,
    mockCreateFlashcardsBulk: createFlashcardsBulk,
    mockRagSearch: ragSearch,
    mockSynthesizeSpeech: synthesizeSpeech,
    mockGenerateSlidesFromMedia: generateSlidesFromMedia,
    mockDownloadOutput: downloadOutput,
    mockCreateChatCompletion: createChatCompletion,
    mockGetMediaDetails: getMediaDetails,
    mockUpsertWorkspace: upsertWorkspace,
    mockGetChatModels: getChatModels,
    messageOptionStoreState: messageOptionState,
    chatModelSettingsStoreState: chatModelSettingsState,
    baseAudioSettings: defaultAudioSettings,
    workspaceStoreState: storeState
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("../StudioPane/QuickNotesSection", () => ({
  QuickNotesSection: ({ onCollapse }: { onCollapse: () => void }) => (
    <button type="button" onClick={onCollapse}>
      Collapse notes
    </button>
  )
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceStudioNoSourcesHint: () => "Select sources to start generating outputs"
}))

vi.mock("@/types/workspace", () => ({
  OUTPUT_TYPES: []
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn().mockResolvedValue([])
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn().mockReturnValue(null)
}))

vi.mock("@/services/quizzes", () => ({
  generateQuiz: vi.fn()
}))

vi.mock("@/services/flashcards", () => ({
  generateFlashcards: mockGenerateFlashcardsService,
  listDecks: vi.fn().mockResolvedValue([]),
  createDeck: vi.fn().mockResolvedValue({ id: 1, name: "Workspace Flashcards" }),
  createFlashcard: vi.fn().mockResolvedValue({ uuid: "card-1" }),
  createFlashcardsBulk: mockCreateFlashcardsBulk.mockResolvedValue({
    cards: [{ uuid: "card-1" }]
  })
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
    downloadOutput: mockDownloadOutput
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

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      useMessage: () => [
        {
          open: vi.fn(),
          warning: vi.fn(),
          destroy: vi.fn(),
          success: mockMessageSuccess,
          error: mockMessageError,
          info: mockMessageInfo
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

const createChatCompletionResponse = (
  content: string,
  usage?: Record<string, unknown>
) =>
  new Response(
    JSON.stringify({
      choices: [
        {
          message: {
            content
          }
        }
      ],
      usage
    }),
    {
      status: 200,
      headers: { "content-type": "application/json" }
    }
  )

const expandOutputTypesSection = () => {
  const toggle = screen.getByRole("button", { name: /Output Types/i })
  if (toggle.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

const renderStudioPane = () => {
  const renderResult = render(<StudioPane />)
  expandOutputTypesSection()
  return renderResult
}

const sourceDetail = (title: string, text: string) => ({
  source: { title },
  content: { text }
})

describe("StudioPane literature work products", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem("tldw:research-workspace:recent-output-types:v1")

    workspaceStoreState.selectedSourceIds = ["source-1", "source-2"]
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.sources = [
      {
        id: "source-1",
        mediaId: 101,
        title: "Paper A",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-2",
        mediaId: 202,
        title: "Paper B",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    workspaceStoreState.generatedArtifacts = []
    workspaceStoreState.isGeneratingOutput = false
    workspaceStoreState.generatingOutputType = null
    workspaceStoreState.audioSettings = { ...baseAudioSettings }
    workspaceStoreState.noteFocusTarget = null

    let artifactCounter = 0

    mockAddArtifact.mockImplementation((artifactData: any) => {
      artifactCounter += 1
      const artifact = {
        ...artifactData,
        id: `artifact-${artifactCounter}`,
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
      workspaceStoreState.generatedArtifacts = [
        artifact,
        ...workspaceStoreState.generatedArtifacts
      ]
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

    mockRemoveArtifact.mockImplementation((id: string) => {
      workspaceStoreState.generatedArtifacts = workspaceStoreState.generatedArtifacts.filter(
        (artifact) => artifact.id !== id
      )
    })

    mockSetIsGeneratingOutput.mockImplementation(
      (isGenerating: boolean, outputType: string | null = null) => {
        workspaceStoreState.isGeneratingOutput = isGenerating
        workspaceStoreState.generatingOutputType = isGenerating ? outputType : null
      }
    )

    messageOptionStoreState.selectedModel = "gpt-4o-mini"
    chatModelSettingsStoreState.apiProvider = undefined
    chatModelSettingsStoreState.numPredict = 1000
    mockRagSearch.mockResolvedValue({ generation: "Generated summary" })
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse("Generated summary")
    )
    mockGetMediaDetails.mockImplementation((mediaId: number) =>
      Promise.resolve(
        mediaId === 101
          ? sourceDetail("Paper A", "Paper A studied 240 participants with a survey.")
          : sourceDetail("Paper B", "Paper B used interviews in a hospital setting.")
      )
    )
    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "Term", back: "Definition" }],
      count: 1
    })
    mockSynthesizeSpeech.mockResolvedValue(new ArrayBuffer(8))
    mockGenerateSlidesFromMedia.mockResolvedValue({
      id: "presentation-1",
      title: "Generated Slides",
      theme: "default",
      slides: [],
      version: 1,
      created_at: "2026-02-18T00:00:00.000Z"
    })
    mockGetChatModels.mockResolvedValue([])
  })

  it("keeps Literature Matrix disabled until at least two sources are selected", () => {
    workspaceStoreState.selectedSourceIds = []

    renderStudioPane()

    expect(
      screen.getByRole("button", { name: /literature matrix/i })
    ).toBeDisabled()
  })

  it("enables Literature Matrix when two sources are selected", () => {
    renderStudioPane()

    expect(
      screen.getByRole("button", { name: /literature matrix/i })
    ).toBeEnabled()
  })

  it("fails Literature Matrix generation before the model call when fewer than two usable source contexts remain", async () => {
    mockGetMediaDetails.mockImplementation((mediaId: number) =>
      Promise.resolve(
        mediaId === 101
          ? sourceDetail("Paper A", "Paper A studied 240 participants with a survey.")
          : sourceDetail("Paper B", "")
      )
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: /literature matrix/i }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "failed",
        expect.objectContaining({
          errorMessage: expect.stringContaining("2 usable source")
        })
      )
    })

    expect(mockCreateChatCompletion).not.toHaveBeenCalled()
    expect(mockMessageSuccess).not.toHaveBeenCalled()
  })

  it("generates a Literature Matrix artifact from strict JSON with lineage and source coverage", async () => {
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          rows: [
            {
              source: "Paper A",
              year_or_date: "2024",
              research_question_or_scope: "Survey scope",
              methodology: "Survey",
              sample_corpus_or_setting: "240 users",
              primary_finding: "Finding A",
              limitations: "Small sample",
              future_work: "Replicate",
              contradictions_or_tension: "Tension with Paper B",
              evidence_references: ["Source 1"],
              confidence: "medium"
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: /literature matrix/i }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "literature_matrix",
          reviewStatus: "draft",
          sourceLineage: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          sourceCoverage: expect.objectContaining({
            selectedSourceIds: ["source-1", "source-2"],
            usableSources: [
              { sourceId: "source-1", mediaId: 101, title: "Paper A" },
              { sourceId: "source-2", mediaId: 202, title: "Paper B" }
            ],
            skippedSources: [],
            minimumUsableSourcesMet: true
          }),
          data: expect.objectContaining({
            table: expect.objectContaining({
              headers: expect.arrayContaining([
                "Source",
                "Methodology",
                "Sample Or Setting",
                "Primary Finding",
                "Limitations",
                "Contradictions Or Tension"
              ])
            })
          }),
          reviewChecklist: [
            expect.objectContaining({
              id: "literature_matrix-review-1",
              checked: false
            }),
            expect.objectContaining({
              id: "literature_matrix-review-2",
              checked: false
            }),
            expect.objectContaining({
              id: "literature_matrix-review-3",
              checked: false
            })
          ]
        })
      )
    })

    expect(mockAddArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "data_table",
        title: "Literature Matrix",
        status: "generating",
        templateId: "literature_matrix"
      })
    )
    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        response_format: { type: "json_object" }
      }),
      expect.any(Object)
    )
    expect(mockMessageSuccess).toHaveBeenCalledWith(
      expect.stringContaining("generated successfully")
    )
  })
})
