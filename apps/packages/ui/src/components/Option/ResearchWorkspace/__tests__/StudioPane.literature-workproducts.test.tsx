import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { AudioGenerationSettings } from "@/types/workspace"
import { StudioPane } from "../StudioPane"
import {
  buildCorpusGapMessages,
  buildEvidenceBoundHypothesesMessages,
  buildResearchProposalMessages,
  normalizeLiteratureMatrixResponse,
  normalizeResearchProposalMarkdown
} from "../StudioPane/literature-workproducts"

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
  it("rejects array rows instead of treating them as literature matrix records", () => {
    expect(() =>
      normalizeLiteratureMatrixResponse(JSON.stringify({ rows: [["Paper A"]] }))
    ).toThrow("Literature Matrix JSON did not include any usable rows.")
  })

  it("handles non-string literature matrix responses as empty output", () => {
    expect(() =>
      normalizeLiteratureMatrixResponse(null as unknown as string)
    ).toThrow("Literature Matrix JSON response was empty.")
  })

  it("handles non-string research proposal responses as empty output", () => {
    expect(() =>
      normalizeResearchProposalMarkdown(undefined as unknown as string)
    ).toThrow("Research Proposal Pack response was empty.")
  })

  it("caps compatible artifact context before adding it to literature prompts", () => {
    const sourceContexts = [
      {
        sourceId: "source-1",
        mediaId: 101,
        title: "Paper A",
        text: "Paper A excerpt"
      }
    ]
    const longArtifactContent = "matrix-content ".repeat(500)
    const sourceCoverage = {
      selectedSourceIds: ["source-1"],
      usableSources: [{ sourceId: "source-1", mediaId: 101, title: "Paper A" }],
      skippedSources: [],
      truncatedSources: [],
      sourceContextCharLimit: { perSource: 6000, total: 18000 },
      minimumUsableSourcesMet: true
    }

    const gapPrompt = buildCorpusGapMessages(
      sourceContexts,
      longArtifactContent
    ).user
    const hypothesesPrompt = buildEvidenceBoundHypothesesMessages(
      sourceContexts,
      [{ label: "Literature Matrix", content: longArtifactContent }]
    ).user
    const proposalPrompt = buildResearchProposalMessages(
      sourceContexts,
      sourceCoverage,
      [{ label: "Evidence-Bound Hypotheses", content: longArtifactContent }]
    ).user

    for (const prompt of [gapPrompt, hypothesesPrompt, proposalPrompt]) {
      expect(prompt).toContain("truncated for context budget")
      expect(prompt).not.toContain(longArtifactContent)
    }
  })

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
    workspaceStoreState.selectedSourceIds = ["source-1"]

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

  it("labels literature-review work products as a discoverable group", () => {
    renderStudioPane()

    expect(screen.getByText("Literature Review")).toBeInTheDocument()
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

  it("records context-limit and unknown skip reasons in literature source coverage", async () => {
    workspaceStoreState.selectedSourceIds = [
      "source-1",
      "source-2",
      "source-3",
      "source-4",
      "source-5"
    ]
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
      },
      {
        id: "source-3",
        mediaId: 303,
        title: "Paper C",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-4",
        mediaId: 404,
        title: "Paper D",
        type: "pdf",
        status: "ready",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "source-5",
        mediaId: 505,
        title: "Paper E",
        type: "pdf",
        addedAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    mockGetMediaDetails.mockImplementation((mediaId: number) => {
      if (mediaId === 404) {
        return Promise.resolve(
          sourceDetail("Paper D", "Paper D has usable text beyond the context limit.")
        )
      }
      if (mediaId === 505) {
        return Promise.resolve(sourceDetail("Paper E", ""))
      }
      return Promise.resolve(
        sourceDetail(`Paper ${mediaId}`, "Long source text. ".repeat(1000))
      )
    })
    mockCreateChatCompletion.mockResolvedValueOnce(
      createChatCompletionResponse(
        JSON.stringify({
          rows: [
            {
              source: "Paper A",
              methodology: "survey",
              primary_finding: "Finding A",
              evidence_references: "Paper A"
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
          sourceCoverage: expect.objectContaining({
            skippedSources: expect.arrayContaining([
              expect.objectContaining({
                sourceId: "source-4",
                reason: "context_limit"
              }),
              expect.objectContaining({
                sourceId: "source-5",
                reason: "unknown"
              })
            ])
          })
        })
      )
    })
  })

  it("fails cleanly when Literature Matrix returns invalid JSON", async () => {
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse("not valid json")
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: /literature matrix/i }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "failed",
        expect.objectContaining({
          errorMessage: expect.stringContaining("not valid JSON")
        })
      )
    })

    expect(mockUpdateArtifactStatus).not.toHaveBeenCalledWith(
      "artifact-1",
      "completed",
      expect.anything()
    )
  })

  it("generates a Corpus Gap Finder artifact from strict JSON with lineage and source coverage", async () => {
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          gaps: [
            {
              gap: "Rural care settings are underrepresented.",
              gap_type: "underrepresented_context",
              evidence_basis:
                "Paper A studies urban clinics while Paper B recommends broader care settings.",
              sources: ["Paper A", "Paper B"],
              missing_area: "Rural clinics",
              why_it_matters: "The effect may not transfer to rural care.",
              confidence: "high",
              suggested_follow_up_question:
                "Do the same outcomes hold in rural clinics?"
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: /corpus gap finder/i }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "corpus_gap_finder",
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
            minimumUsableSourcesMet: true
          }),
          data: expect.objectContaining({
            table: expect.objectContaining({
              headers: expect.arrayContaining([
                "Gap",
                "Gap Type",
                "Evidence Basis",
                "Sources",
                "Missing Area",
                "Why It Matters",
                "Confidence",
                "Suggested Follow-up Question"
              ])
            })
          })
        })
      )
    })

    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        response_format: { type: "json_object" }
      }),
      expect.any(Object)
    )
  })

  it("uses a compatible Literature Matrix artifact as optional context for Corpus Gap Finder", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-matrix",
        type: "data_table",
        title: "Literature Matrix",
        status: "completed",
        templateId: "literature_matrix",
        content: "| Source | Primary Finding |\n| --- | --- |\n| Paper A | Finding A |",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          gaps: [
            {
              gap: "No rural care comparison.",
              gap_type: "missing_comparison",
              evidence_basis: "The matrix has no rural comparison rows.",
              sources: ["Paper A"],
              missing_area: "Rural comparison",
              why_it_matters: "Generalizability is unclear.",
              confidence: "medium",
              suggested_follow_up_question:
                "How do rural and urban care settings compare?"
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(screen.getByRole("button", { name: /corpus gap finder/i }))

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "corpus_gap_finder"
        })
      )
    })

    const request = mockCreateChatCompletion.mock.calls[0]?.[0] as {
      messages?: Array<{ role: string; content: string }>
    }
    const userPrompt = request.messages?.find((message) => message.role === "user")
      ?.content
    expect(userPrompt).toContain("Compatible Literature Matrix")
    expect(userPrompt).toContain("Finding A")
  })

  it("generates Evidence-Bound Hypotheses from strict JSON with source coverage", async () => {
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          hypotheses: [
            {
              hypothesis:
                "Rural care settings will show weaker transfer than urban clinics.",
              supporting_findings: [
                "Paper A studies urban clinics.",
                "Paper B calls for broader care settings."
              ],
              supporting_sources: ["Paper A", "Paper B"],
              prediction:
                "Effect sizes will be lower in rural clinics than urban clinics.",
              suggested_methodology:
                "Matched cohort comparison across rural and urban clinics.",
              threats_to_validity: ["Selection bias", "Clinic staffing differences"],
              what_would_falsify_it:
                "Rural clinics show equal or stronger outcomes.",
              confidence: "medium"
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(
      screen.getByRole("button", { name: /evidence-bound hypotheses/i })
    )

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "evidence_bound_hypotheses",
          reviewStatus: "draft",
          sourceCoverage: expect.objectContaining({
            selectedSourceIds: ["source-1", "source-2"],
            minimumUsableSourcesMet: true
          }),
          sourceLineage: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          content: expect.stringContaining("## Evidence-Bound Hypotheses"),
          data: expect.objectContaining({
            hypotheses: [
              expect.objectContaining({
                hypothesis:
                  "Rural care settings will show weaker transfer than urban clinics.",
                confidence: "medium"
              })
            ]
          })
        })
      )
    })

    expect(mockAddArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "report",
        title: "Evidence-Bound Hypotheses",
        status: "generating",
        templateId: "evidence_bound_hypotheses"
      })
    )
    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        response_format: { type: "json_object" }
      }),
      expect.any(Object)
    )
  })

  it("includes compatible Matrix and Gap artifacts as optional context for Evidence-Bound Hypotheses", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-gap",
        type: "data_table",
        title: "Corpus Gap Finder",
        status: "completed",
        templateId: "corpus_gap_finder",
        content: "| Gap | Evidence Basis |\n| --- | --- |\n| Rural settings missing | Papers A and B |",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "artifact-matrix",
        type: "data_table",
        title: "Literature Matrix",
        status: "completed",
        templateId: "literature_matrix",
        content: "| Source | Primary Finding |\n| --- | --- |\n| Paper A | Finding A |",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(
        JSON.stringify({
          hypotheses: [
            {
              hypothesis: "Rural settings require separate validation.",
              supporting_findings: ["Rural settings are missing."],
              supporting_sources: ["Paper A", "Paper B"],
              prediction: "Rural outcomes will differ.",
              suggested_methodology: "Rural validation cohort.",
              threats_to_validity: ["Small rural sample"],
              what_would_falsify_it: "No rural/urban difference.",
              confidence: "medium"
            }
          ]
        })
      )
    )

    renderStudioPane()

    fireEvent.click(
      screen.getByRole("button", { name: /evidence-bound hypotheses/i })
    )

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "evidence_bound_hypotheses"
        })
      )
    })

    const request = mockCreateChatCompletion.mock.calls[0]?.[0] as {
      messages?: Array<{ role: string; content: string }>
    }
    const userPrompt = request.messages?.find((message) => message.role === "user")
      ?.content
    expect(userPrompt).toContain("Compatible Literature Matrix")
    expect(userPrompt).toContain("Compatible Corpus Gap Finder")
    expect(userPrompt).toContain("Rural settings missing")
  })

  it("generates a Research Proposal Pack with source coverage and a source audit", async () => {
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(`# Title
Rural Transfer Study

## Research Question
Do rural settings show the same transfer as urban clinics?

## Literature Overview
Paper A and Paper B provide the current evidence base.

## Evidence Matrix Summary
The selected studies cover urban and hospital settings.

## Identified Gaps
Rural settings are underrepresented.

## Proposed Hypothesis
Rural transfer effects will differ from urban transfer effects.

## Methodology
Run a matched rural and urban cohort comparison.

## Expected Results Or Decision Value
The study clarifies generalizability.

## Contribution
It expands the evidence base into an underrepresented context.

## Risks And Limitations
Rural sample size may be limited.

## Source Audit
- Paper A
- Paper B`)
    )

    renderStudioPane()

    fireEvent.click(
      screen.getByRole("button", { name: /research proposal pack/i })
    )

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "research_proposal_pack",
          reviewStatus: "draft",
          content: expect.stringContaining("## Source Audit"),
          sourceCoverage: expect.objectContaining({
            selectedSourceIds: ["source-1", "source-2"],
            minimumUsableSourcesMet: true
          }),
          sourceLineage: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ]
        })
      )
    })

    expect(mockAddArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "report",
        title: "Research Proposal Pack",
        status: "generating",
        templateId: "research_proposal_pack"
      })
    )
    expect(mockCreateChatCompletion).toHaveBeenCalledWith(
      expect.not.objectContaining({
        response_format: expect.anything()
      }),
      expect.any(Object)
    )
  })

  it("includes compatible Matrix, Gap, and Hypothesis artifacts as proposal context", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-hypothesis",
        type: "report",
        title: "Evidence-Bound Hypotheses",
        status: "completed",
        templateId: "evidence_bound_hypotheses",
        content: "## Evidence-Bound Hypotheses\nRural settings require validation.",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "artifact-gap",
        type: "data_table",
        title: "Corpus Gap Finder",
        status: "completed",
        templateId: "corpus_gap_finder",
        content: "| Gap | Evidence Basis |\n| --- | --- |\n| Rural settings missing | Papers A and B |",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      },
      {
        id: "artifact-matrix",
        type: "data_table",
        title: "Literature Matrix",
        status: "completed",
        templateId: "literature_matrix",
        content: "| Source | Primary Finding |\n| --- | --- |\n| Paper A | Finding A |",
        sourceCoverage: {
          selectedSourceIds: ["source-1", "source-2"],
          usableSources: [
            { sourceId: "source-1", mediaId: 101, title: "Paper A" },
            { sourceId: "source-2", mediaId: 202, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]
    mockCreateChatCompletion.mockResolvedValue(
      createChatCompletionResponse(`# Title
Proposal

## Source Audit
- Paper A
- Paper B`)
    )

    renderStudioPane()

    fireEvent.click(
      screen.getByRole("button", { name: /research proposal pack/i })
    )

    await waitFor(() => {
      expect(mockUpdateArtifactStatus).toHaveBeenCalledWith(
        "artifact-1",
        "completed",
        expect.objectContaining({
          templateId: "research_proposal_pack"
        })
      )
    })

    const request = mockCreateChatCompletion.mock.calls[0]?.[0] as {
      messages?: Array<{ role: string; content: string }>
    }
    const userPrompt = request.messages?.find((message) => message.role === "user")
      ?.content
    expect(userPrompt).toContain("Compatible Literature Matrix")
    expect(userPrompt).toContain("Compatible Corpus Gap Finder")
    expect(userPrompt).toContain("Compatible Evidence-Bound Hypotheses")
    expect(userPrompt).toContain("Source coverage")
  })

  it("exports structured literature tables as CSV and JSON without advertising XLSX", async () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:literature-table")
    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})
    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {})

    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-literature-table",
        type: "data_table",
        title: "Literature Matrix",
        status: "completed",
        templateId: "literature_matrix",
        content:
          "| Source | Methodology | Primary Finding |\n| --- | --- | --- |\n| Paper A | Survey | Finding A |",
        data: {
          table: {
            headers: ["Source", "Methodology", "Primary Finding"],
            rows: [["Paper A", "Survey", "Finding A"]]
          }
        },
        createdAt: new Date("2026-02-18T00:00:00.000Z")
      }
    ]

    renderStudioPane()
    fireEvent.click(screen.getByRole("button", { name: "View" }))

    expect(await screen.findByRole("button", { name: "Export CSV" }))
      .toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export JSON" }))
      .toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /xlsx/i })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))

    await waitFor(() => {
      expect(createObjectUrlSpy).toHaveBeenCalled()
    })
    expect(anchorClickSpy).toHaveBeenCalled()
    expect(revokeObjectUrlSpy).toHaveBeenCalled()

    const jsonBlob = createObjectUrlSpy.mock.calls.at(-1)?.[0] as Blob & {
      type?: string
    }
    expect(jsonBlob).toBeTruthy()
    expect(jsonBlob.type).toContain("application/json")

    createObjectUrlSpy.mockRestore()
    revokeObjectUrlSpy.mockRestore()
    anchorClickSpy.mockRestore()
  })
})
