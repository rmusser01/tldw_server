import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fetchTldwVoiceCatalog } from "@/services/tldw/audio-voices"
import { inferTldwProviderFromModel } from "@/services/tts-provider"
import { StudioPane, estimateGenerationSeconds } from "../StudioPane"

const {
  mockRagSearch,
  mockSynthesizeSpeech,
  mockGenerateSlidesFromMedia,
  mockGenerateFlashcardsService,
  mockListVisualStyles,
  mockAddArtifact,
  mockUpdateArtifactStatus,
  mockRemoveArtifact,
  mockSetIsGeneratingOutput,
  mockSetAudioSettings,
  mockListDecks,
  mockGetChatModels,
  mockGetMediaDetails,
  mockUpsertWorkspace,
  mockCreateDeck,
  mockCreateFlashcard,
  mockCreateFlashcardsBulk,
  mockSetSelectedModel,
  mockSetRagSearchMode,
  mockSetRagTopK,
  mockSetRagEnableGeneration,
  mockSetRagEnableCitations,
  mockSetRagAdvancedOptions,
  mockSetApiProvider,
  mockSetTemperature,
  mockSetTopP,
  mockSetNumPredict,
  mockUpdateModelSetting,
  messageOptionStoreState,
  chatModelSettingsStoreState,
  mockMessageSuccess,
  mockMessageError,
  mockMessageInfo,
  workspaceStoreState
} = vi.hoisted(() => {
  const ragSearch = vi.fn()
  const synthesizeSpeech = vi.fn()
  const generateSlidesFromMedia = vi.fn()
  const generateFlashcardsService = vi.fn()
  const listVisualStyles = vi.fn()
  const addArtifact = vi.fn()
  const updateArtifactStatus = vi.fn()
  const removeArtifact = vi.fn()
  const restoreArtifact = vi.fn()
  const setIsGeneratingOutput = vi.fn()
  const setAudioSettings = vi.fn()
  const listDecks = vi.fn()
  const getChatModels = vi.fn()
  const getMediaDetails = vi.fn()
  const upsertWorkspace = vi.fn()
  const createDeck = vi.fn()
  const createFlashcard = vi.fn()
  const createFlashcardsBulk = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageInfo = vi.fn()
  const setSelectedModel = vi.fn()
  const setRagSearchMode = vi.fn()
  const setRagTopK = vi.fn()
  const setRagEnableGeneration = vi.fn()
  const setRagEnableCitations = vi.fn()
  const setRagAdvancedOptions = vi.fn()
  const setApiProvider = vi.fn()
  const setTemperature = vi.fn()
  const setTopP = vi.fn()
  const setNumPredict = vi.fn()
  const updateModelSetting = vi.fn()

  const state = {
    selectedSourceIds: ["source-1"],
    sources: [] as Array<any>,
    workspaceId: "workspace-a",
    workspaceName: "Workspace A",
    getSelectedMediaIds: () => [101],
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
    noteFocusTarget: null as { field: "title" | "content"; token: number } | null
  }

  const messageOptionState = {
    selectedModel: "gpt-4o-mini",
    setSelectedModel,
    ragSearchMode: "hybrid" as "hybrid" | "vector" | "fts",
    setRagSearchMode,
    ragTopK: 8,
    setRagTopK,
    ragEnableGeneration: true,
    setRagEnableGeneration,
    ragEnableCitations: true,
    setRagEnableCitations,
    ragAdvancedOptions: {
      min_score: 0.2,
      enable_reranking: true
    } as Record<string, unknown>,
    setRagAdvancedOptions
  }

  const chatModelSettingsState = {
    apiProvider: undefined as string | undefined,
    temperature: 0.7,
    topP: 1,
    numPredict: 800,
    setApiProvider,
    setTemperature,
    setTopP,
    setNumPredict,
    updateSetting: updateModelSetting
  }

  setSelectedModel.mockImplementation((nextModel: string) => {
    messageOptionState.selectedModel = nextModel
  })
  setRagSearchMode.mockImplementation((nextMode: "hybrid" | "vector" | "fts") => {
    messageOptionState.ragSearchMode = nextMode
  })
  setRagTopK.mockImplementation((nextTopK: number | null) => {
    messageOptionState.ragTopK = nextTopK
  })
  setRagEnableGeneration.mockImplementation((enabled: boolean) => {
    messageOptionState.ragEnableGeneration = enabled
  })
  setRagEnableCitations.mockImplementation((enabled: boolean) => {
    messageOptionState.ragEnableCitations = enabled
  })
  setRagAdvancedOptions.mockImplementation((nextOptions: Record<string, unknown>) => {
    messageOptionState.ragAdvancedOptions = nextOptions
  })
  setApiProvider.mockImplementation((provider: string) => {
    chatModelSettingsState.apiProvider = provider
  })
  setTemperature.mockImplementation((nextTemperature: number) => {
    chatModelSettingsState.temperature = nextTemperature
  })
  setTopP.mockImplementation((nextTopP: number) => {
    chatModelSettingsState.topP = nextTopP
  })
  setNumPredict.mockImplementation((nextNumPredict: number | undefined) => {
    chatModelSettingsState.numPredict = nextNumPredict
  })
  updateModelSetting.mockImplementation(
    (key: keyof typeof chatModelSettingsState, value: unknown) => {
      ;(chatModelSettingsState as Record<string, unknown>)[key] = value
    }
  )

  return {
    mockRagSearch: ragSearch,
    mockSynthesizeSpeech: synthesizeSpeech,
    mockGenerateSlidesFromMedia: generateSlidesFromMedia,
    mockGenerateFlashcardsService: generateFlashcardsService,
    mockListVisualStyles: listVisualStyles,
    mockAddArtifact: addArtifact,
    mockUpdateArtifactStatus: updateArtifactStatus,
    mockRemoveArtifact: removeArtifact,
    mockSetIsGeneratingOutput: setIsGeneratingOutput,
    mockSetAudioSettings: setAudioSettings,
    mockListDecks: listDecks,
    mockGetChatModels: getChatModels,
    mockGetMediaDetails: getMediaDetails,
    mockUpsertWorkspace: upsertWorkspace,
    mockCreateDeck: createDeck,
    mockCreateFlashcard: createFlashcard,
    mockCreateFlashcardsBulk: createFlashcardsBulk,
    mockSetSelectedModel: setSelectedModel,
    mockSetRagSearchMode: setRagSearchMode,
    mockSetRagTopK: setRagTopK,
    mockSetRagEnableGeneration: setRagEnableGeneration,
    mockSetRagEnableCitations: setRagEnableCitations,
    mockSetRagAdvancedOptions: setRagAdvancedOptions,
    mockSetApiProvider: setApiProvider,
    mockSetTemperature: setTemperature,
    mockSetTopP: setTopP,
    mockSetNumPredict: setNumPredict,
    mockUpdateModelSetting: updateModelSetting,
    messageOptionStoreState: messageOptionState,
    chatModelSettingsStoreState: chatModelSettingsState,
    mockMessageSuccess: messageSuccess,
    mockMessageError: messageError,
    mockMessageInfo: messageInfo,
    workspaceStoreState: state
  }
})
let isMobile = false

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string },
      interpolationOptions?: Record<string, unknown>
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        if (!interpolationOptions) return defaultValueOrOptions
        return defaultValueOrOptions.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(interpolationOptions[token] ?? "")
        )
      }
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
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

vi.mock("@/types/workspace", async () => {
  const actual = await vi.importActual<typeof import("@/types/workspace")>(
    "@/types/workspace"
  )
  return {
    ...actual,
    OUTPUT_TYPES: []
  }
})

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn().mockResolvedValue([])
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn().mockReturnValue("kokoro")
}))

vi.mock("@/services/quizzes", () => ({
  generateQuiz: vi.fn()
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
    getMediaDetails: mockGetMediaDetails,
    upsertWorkspace: mockUpsertWorkspace,
    listVisualStyles: mockListVisualStyles,
    exportPresentation: vi.fn(),
    downloadOutput: vi.fn()
  }
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getChatModels: mockGetChatModels,
    getProviderDisplayName: (provider: string) => provider.toUpperCase()
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

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      useMessage: () => [
        {
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

const expandStudioOptionsSection = () => {
  const toggle = screen.getByRole("button", { name: /Studio Options/i })
  if (toggle.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

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

const renderExpandedStudioPane = () => {
  const renderResult = render(<StudioPane />)
  expandStudioOptionsSection()
  expandOutputTypesSection()
  expandGeneratedOutputsSection()
  return renderResult
}

describe("StudioPane Stage 3 information architecture and UX polish", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem("tldw:workspace-playground:recent-output-types:v1")
    isMobile = false
    workspaceStoreState.selectedSourceIds = ["source-1"]
    workspaceStoreState.getSelectedMediaIds = () => [101]
    workspaceStoreState.generatedArtifacts = []
    workspaceStoreState.isGeneratingOutput = false
    workspaceStoreState.generatingOutputType = null
    workspaceStoreState.noteFocusTarget = null
    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceName = "Workspace A"
    workspaceStoreState.studyMaterialsPolicy = "workspace"
    workspaceStoreState.audioSettings = {
      provider: "tldw",
      model: "kokoro",
      voice: "af_heart",
      speed: 1,
      format: "mp3"
    }
    messageOptionStoreState.selectedModel = "gpt-4o-mini"
    messageOptionStoreState.ragSearchMode = "hybrid"
    messageOptionStoreState.ragTopK = 8
    messageOptionStoreState.ragEnableGeneration = true
    messageOptionStoreState.ragEnableCitations = true
    messageOptionStoreState.ragAdvancedOptions = {
      min_score: 0.2,
      enable_reranking: true
    }
    chatModelSettingsStoreState.apiProvider = undefined
    chatModelSettingsStoreState.temperature = 0.7
    chatModelSettingsStoreState.topP = 1
    chatModelSettingsStoreState.numPredict = 800
    vi.mocked(fetchTldwVoiceCatalog).mockResolvedValue([])
    vi.mocked(inferTldwProviderFromModel).mockReturnValue("kokoro")

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

    mockSetIsGeneratingOutput.mockImplementation(
      (isGenerating: boolean, outputType: string | null = null) => {
        workspaceStoreState.isGeneratingOutput = isGenerating
        workspaceStoreState.generatingOutputType = isGenerating ? outputType : null
      }
    )

    mockListDecks.mockResolvedValue([])
    mockGetMediaDetails.mockResolvedValue({
      source: { title: "DSPy Prompting Talk" },
      content: {
        text: "ATP powers cellular respiration in cells."
      }
    })
    mockCreateDeck.mockResolvedValue({
      id: 7,
      name: "Workspace A - DSPy Prompting Talk"
    })
    mockUpsertWorkspace.mockResolvedValue({
      id: "workspace-a",
      name: "Workspace A",
      study_materials_policy: "workspace"
    })
    mockCreateFlashcardsBulk.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 7 }],
      count: 1,
      total: 1
    })
    mockGetChatModels.mockResolvedValue([
      { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "openai", type: "chat" },
      { id: "llama-3.1-8b", name: "Llama 3.1 8B", provider: "ollama", type: "chat" }
    ])
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
    mockListVisualStyles.mockResolvedValue([
      {
        id: "minimal-academic",
        name: "Minimal Academic",
        scope: "builtin",
        description: "Structured, restrained, study-first slides.",
        generation_rules: {},
        artifact_preferences: [],
        appearance_defaults: { theme: "white" },
        fallback_policy: {},
        version: 1
      },
      {
        id: "timeline",
        name: "Timeline",
        scope: "builtin",
        description: "Chronology-forward deck structure.",
        generation_rules: {},
        artifact_preferences: ["timeline"],
        appearance_defaults: { theme: "beige" },
        fallback_policy: {},
        version: 1
      }
    ])
  })

  it("groups output buttons by category and surfaces description tooltips", async () => {
    renderExpandedStudioPane()

    expect(screen.getByText("Study Aids")).toBeInTheDocument()
    expect(screen.getByText("Analysis")).toBeInTheDocument()
    expect(screen.getByText("Creative")).toBeInTheDocument()

    fireEvent.mouseEnter(screen.getByRole("button", { name: "Summary" }))
    expect(
      await screen.findByText("Create a concise summary of key points")
    ).toBeInTheDocument()
  })

  it("shows contextual audio settings when audio overview is selected", async () => {
    renderExpandedStudioPane()

    expect(
      screen.getByText("Select Audio Summary to configure TTS voice and speed.")
    ).toBeInTheDocument()
    expect(screen.queryByText("TTS Provider")).not.toBeInTheDocument()

    fireEvent.mouseEnter(screen.getByRole("button", { name: "Audio Summary" }))

    expect(await screen.findByText("TTS Provider")).toBeInTheDocument()
  })

  it("passes the selected visual style into slides generation", async () => {
    renderExpandedStudioPane()

    const styleSelect = await screen.findByLabelText("Slides visual style")
    fireEvent.change(styleSelect, {
      target: { value: "builtin::timeline" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Slides" }))

    await waitFor(() => {
      expect(mockGenerateSlidesFromMedia).toHaveBeenCalledWith(
        101,
        expect.objectContaining({
          visualStyleId: "timeline",
          visualStyleScope: "builtin"
        })
      )
    })
  })

  it("creates one workspace-owned deck and bulk saves flashcards for a run", async () => {
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
    ]
    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceName = "Workspace A"
    workspaceStoreState.studyMaterialsPolicy = "workspace"

    mockGenerateFlashcardsService.mockResolvedValue({
      flashcards: [{ front: "ATP", back: "Cellular energy" }],
      count: 1
    })
    mockCreateDeck.mockResolvedValue({
      id: 7,
      name: "Workspace A - DSPy Prompting Talk"
    })
    mockCreateFlashcardsBulk.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 7 }],
      count: 1,
      total: 1
    })

    renderExpandedStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockGenerateFlashcardsService).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4o-mini",
          provider: "openai",
          text: expect.stringContaining("DSPy Prompting Talk")
        })
      )
    })

    expect(mockUpsertWorkspace).toHaveBeenCalledWith("workspace-a", {
      name: "Workspace A",
      study_materials_policy: "workspace"
    })
    expect(mockCreateDeck).toHaveBeenCalledWith(
      expect.objectContaining({
        name: expect.stringContaining("Workspace A"),
        workspace_id: "workspace-a"
      }),
      expect.any(Object)
    )
    expect(mockCreateFlashcardsBulk).toHaveBeenCalledWith([
      expect.objectContaining({
        deck_id: 7,
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
        serverId: 7,
        data: expect.objectContaining({
          deckId: 7,
          sourceMediaIds: [101]
        })
      })
    )
  }, 15000)

  it("retries flashcard generation once when the first draft response has no usable cards", async () => {
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
    workspaceStoreState.studyMaterialsPolicy = "workspace"

    mockGenerateFlashcardsService
      .mockResolvedValueOnce({
        flashcards: [{ front: "", back: "Missing front" }],
        count: 1
      })
      .mockResolvedValueOnce({
        flashcards: [{ front: "ATP", back: "Cellular energy" }],
        count: 1
      })
    mockCreateDeck.mockResolvedValue({
      id: 7,
      name: "Workspace A - DSPy Prompting Talk"
    })
    mockCreateFlashcardsBulk.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 7 }],
      count: 1,
      total: 1
    })

    renderExpandedStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Flashcards" }))

    await waitFor(() => {
      expect(mockGenerateFlashcardsService).toHaveBeenCalledTimes(2)
    })

    expect(mockGenerateFlashcardsService).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        num_cards: 8,
        difficulty: "easy"
      })
    )
    expect(mockCreateFlashcardsBulk).toHaveBeenCalledWith([
      expect.objectContaining({
        deck_id: 7,
        front: "ATP",
        back: "Cellular energy",
        source_ref_id: "101"
      })
    ], expect.objectContaining({ signal: expect.any(AbortSignal) }))
  }, 15000)

  it("round-trips studyMaterialsPolicy through workspace snapshots", async () => {
    const workspaceModule = await vi.importActual<typeof import("@/store/workspace")>(
      "@/store/workspace"
    )

    const snapshotState = {
      ...workspaceStoreState,
      workspaceId: "workspace-a",
      workspaceName: "Workspace A",
      workspaceTag: "workspace:workspace-a",
      workspaceCreatedAt: new Date("2026-02-18T00:00:00.000Z"),
      workspaceChatReferenceId: "workspace-a",
      sources: [],
      selectedSourceIds: [],
      sourceFolders: [],
      sourceFolderMemberships: [],
      selectedSourceFolderIds: [],
      activeFolderId: null,
      generatedArtifacts: [],
      notes: "",
      workspaceBanner: {
        title: "Workspace A",
        subtitle: "Study materials",
        image: null
      },
      currentNote: {
        id: undefined,
        title: "",
        content: "",
        keywords: [],
        version: 1,
        isDirty: false
      },
      studyMaterialsPolicy: "workspace" as const
    }

    const builtSnapshot = workspaceModule.buildWorkspaceSnapshot(
      snapshotState as Parameters<typeof workspaceModule.buildWorkspaceSnapshot>[0]
    )
    expect(builtSnapshot.studyMaterialsPolicy).toBe("workspace")

    const revivedSnapshot = workspaceModule.reviveWorkspaceSnapshot(
      builtSnapshot.workspaceId,
      builtSnapshot
    )
    expect(revivedSnapshot.studyMaterialsPolicy).toBe("workspace")

    const appliedState = workspaceModule.applyWorkspaceSnapshot(revivedSnapshot)
    expect(appliedState.studyMaterialsPolicy).toBe("workspace")
  })

  it("shows dynamic ETA text while generation is running", async () => {
    mockRagSearch.mockImplementation(
      (_query: string, options?: { signal?: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          const abortError = new Error("Aborted")
          abortError.name = "AbortError"
          const signal = options?.signal
          signal?.addEventListener(
            "abort",
            () => {
              reject(abortError)
            },
            { once: true }
          )
        })
    )

    renderExpandedStudioPane()

    fireEvent.click(screen.getByRole("button", { name: "Summary" }))

    await waitFor(() => {
      expect(
        screen.getByText("~8s for 1 source")
      ).toBeInTheDocument()
    })
  })

  it("uses adaptive output container sizing and stable ETA heuristic", () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-summary",
        type: "summary",
        title: "Summary",
        status: "completed",
        content: "Summary text",
        createdAt: new Date("2026-02-18T10:00:00.000Z")
      }
    ]

    const { container } = renderExpandedStudioPane()
    const paneRoot = container.firstElementChild as HTMLElement | null
    expect(paneRoot).not.toBeNull()
    expect(paneRoot?.className).toContain("overflow-y-auto")
    const outputContainer = container.querySelector(".custom-scrollbar")
    expect(outputContainer).toHaveStyle({ maxHeight: "40vh" })

    expect(estimateGenerationSeconds("summary", 1)).toBe(8)
    expect(estimateGenerationSeconds("audio_overview", 3)).toBe(34)
    expect(estimateGenerationSeconds("audio_overview", 3)).toBeGreaterThan(
      estimateGenerationSeconds("summary", 3)
    )
  })

  it("uses larger touch controls for audio settings on mobile", async () => {
    isMobile = true

    const { container } = renderExpandedStudioPane()
    fireEvent.click(screen.getByRole("button", { name: "Audio Settings" }))

    expect(await screen.findByText("TTS Provider")).toBeInTheDocument()
    const selectElements = container.querySelectorAll(".ant-select")
    expect(selectElements.length).toBeGreaterThan(0)
    expect(
      Array.from(selectElements).every((select) =>
        select.classList.contains("ant-select-lg")
      )
    ).toBe(true)

    const speedLabel = screen.getByText("Speed: 1.0x")
    const speedSlider = speedLabel.parentElement?.querySelector(
      ".ant-slider"
    ) as HTMLElement | null
    expect(speedSlider).toBeTruthy()
    expect(speedSlider?.className).toContain("[&_.ant-slider-rail]:!h-2")
    expect(speedSlider?.className).toContain("[&_.ant-slider-track]:!h-2")
    expect(speedSlider?.className).toContain("[&_.ant-slider-handle]:!h-5")
  })

  it("exposes aria-expanded metadata for collapsible studio sections", () => {
    const { container } = render(<StudioPane />)

    const studioOptionsToggle = screen.getByRole("button", {
      name: /Studio Options/
    })
    expect(studioOptionsToggle).toHaveAttribute("aria-expanded", "false")
    expect(studioOptionsToggle).toHaveAttribute(
      "aria-controls",
      "studio-options-section"
    )

    const outputTypesToggle = screen.getByRole("button", {
      name: /Output Types/
    })
    expect(outputTypesToggle).toHaveAttribute("aria-expanded", "true")
    expect(outputTypesToggle).toHaveAttribute(
      "aria-controls",
      "studio-output-types-section"
    )

    const outputsToggle = screen.getByRole("button", {
      name: /Generated Outputs/
    })
    expect(outputsToggle).toHaveAttribute("aria-expanded", "true")
    expect(outputsToggle).toHaveAttribute(
      "aria-controls",
      "studio-generated-outputs-section"
    )

    fireEvent.click(outputsToggle)
    expect(outputsToggle).toHaveAttribute("aria-expanded", "false")
    expect(
      container.querySelector("#studio-generated-outputs-section")
    ).toHaveAttribute("hidden")

    fireEvent.click(outputsToggle)
    expect(outputsToggle).toHaveAttribute("aria-expanded", "true")
    expect(
      container.querySelector("#studio-generated-outputs-section")
    ).not.toHaveAttribute("hidden")

    fireEvent.click(studioOptionsToggle)
    expect(studioOptionsToggle).toHaveAttribute("aria-expanded", "true")
    expect(
      container.querySelector("#studio-options-section")
    ).not.toHaveAttribute("hidden")

    fireEvent.click(studioOptionsToggle)
    expect(studioOptionsToggle).toHaveAttribute("aria-expanded", "false")
    expect(container.querySelector("#studio-options-section")).toHaveAttribute("hidden")

    fireEvent.click(outputTypesToggle)
    expect(outputTypesToggle).toHaveAttribute("aria-expanded", "false")

    fireEvent.click(outputTypesToggle)
    expect(outputTypesToggle).toHaveAttribute("aria-expanded", "true")

    const audioSettingsToggle = screen.getByRole("button", {
      name: /Audio Settings/
    })
    expect(audioSettingsToggle).toHaveAttribute("aria-expanded", "false")
    expect(audioSettingsToggle).toHaveAttribute(
      "aria-controls",
      "studio-audio-settings-panel"
    )
  })

  it("renders audio settings as a connected accordion container", () => {
    renderExpandedStudioPane()

    const accordion = screen.getByTestId("studio-audio-settings-accordion")
    const audioSettingsToggle = screen.getByRole("button", {
      name: /Audio Settings/
    })

    expect(accordion).toContainElement(audioSettingsToggle)
    expect(screen.queryByText("TTS Provider")).not.toBeInTheDocument()

    fireEvent.click(audioSettingsToggle)

    expect(audioSettingsToggle).toHaveAttribute("aria-expanded", "true")
    expect(audioSettingsToggle.className).toContain("border-b")
    expect(screen.getByText("TTS Provider")).toBeInTheDocument()
    expect(accordion).toContainElement(
      screen.getByText("TTS Provider")
    )
  })

  it("switches to KittenTTS voices when a Kitten model is selected", async () => {
    workspaceStoreState.audioSettings = {
      provider: "tldw",
      model: "KittenML/kitten-tts-nano-0.8",
      voice: "af_heart",
      speed: 1,
      format: "mp3"
    }
    vi.mocked(inferTldwProviderFromModel).mockReturnValue("kitten_tts")
    vi.mocked(fetchTldwVoiceCatalog).mockResolvedValue([])

    renderExpandedStudioPane()

    await waitFor(() => {
      expect(fetchTldwVoiceCatalog).toHaveBeenCalledWith("kitten_tts")
      expect(mockSetAudioSettings).toHaveBeenCalledWith(
        expect.objectContaining({ voice: "Bella" })
      )
    })
  })

  it("does not reset the selected voice while the Kitten voice catalog is still loading", async () => {
    workspaceStoreState.audioSettings = {
      provider: "tldw",
      model: "KittenML/kitten-tts-nano-0.8",
      voice: "custom-kitten-voice",
      speed: 1,
      format: "mp3"
    }
    vi.mocked(inferTldwProviderFromModel).mockReturnValue("kitten_tts")
    vi.mocked(fetchTldwVoiceCatalog).mockReturnValue(new Promise(() => {}))

    renderExpandedStudioPane()
    await Promise.resolve()

    expect(fetchTldwVoiceCatalog).toHaveBeenCalledWith("kitten_tts")
    expect(mockSetAudioSettings).not.toHaveBeenCalled()
  })

  it("wires Studio Options controls to model and RAG stores", async () => {
    const { container } = renderExpandedStudioPane()

    await waitFor(() => {
      expect(mockGetChatModels).toHaveBeenCalled()
    })

    expect(screen.getByText("API Provider")).toBeInTheDocument()
    expect(screen.getByText("Model Runtime")).toBeInTheDocument()
    expect(screen.getByText("RAG Settings")).toBeInTheDocument()

    const maxTokensLabel = screen.getByText("Max Tokens")
    const maxTokensInput = maxTokensLabel.parentElement?.querySelector(
      "input"
    ) as HTMLInputElement | null
    expect(maxTokensInput).toBeTruthy()
    if (maxTokensInput) {
      fireEvent.change(maxTokensInput, { target: { value: "1234" } })
    }
    expect(mockSetNumPredict).toHaveBeenCalledWith(1234)

    const enableCitationsRow = screen.getByText("Enable citations").closest("div")
    const citationsSwitch = enableCitationsRow?.querySelector(
      "button[role='switch']"
    ) as HTMLButtonElement | null
    expect(citationsSwitch).toBeTruthy()
    if (citationsSwitch) {
      fireEvent.click(citationsSwitch)
    }
    expect(mockSetRagEnableCitations).toHaveBeenCalledWith(false)

    const rerankingRow = screen.getByText("Enable reranking").closest("div")
    const rerankingSwitch = rerankingRow?.querySelector(
      "button[role='switch']"
    ) as HTMLButtonElement | null
    expect(rerankingSwitch).toBeTruthy()
    if (rerankingSwitch) {
      fireEvent.click(rerankingSwitch)
    }
    expect(mockSetRagAdvancedOptions).toHaveBeenCalledWith(
      expect.objectContaining({ enable_reranking: false })
    )

    expect(container.querySelector("[data-testid='studio-options-accordion']")).toBeTruthy()
  })

  it("disables retrieval-oriented RAG controls when summary is the active output type", async () => {
    renderExpandedStudioPane()

    await waitFor(() => {
      expect(mockGetChatModels).toHaveBeenCalled()
    })

    fireEvent.mouseEnter(screen.getByRole("button", { name: "Summary" }))

    expect(
      screen.getByText(
        "Summary uses the workspace summary prompt and selected source content directly. Retrieval settings below do not apply."
      )
    ).toBeInTheDocument()

    const searchModeLabel = screen.getByText("Search Mode")
    const searchModeInput = searchModeLabel.parentElement?.querySelector(
      "input[role='combobox']"
    ) as HTMLInputElement | null
    expect(searchModeInput).toBeTruthy()
    if (!searchModeInput) {
      throw new Error("Expected Search Mode combobox to be rendered")
    }
    expect(searchModeInput).toBeDisabled()

    const enableGenerationRow = screen.getByText("Enable generation").closest("div")
    const generationSwitch = enableGenerationRow?.querySelector(
      "button[role='switch']"
    ) as HTMLButtonElement | null
    expect(generationSwitch).toBeTruthy()
    expect(generationSwitch).toBeDisabled()

    const enableCitationsRow = screen.getByText("Enable citations").closest("div")
    const citationsSwitch = enableCitationsRow?.querySelector(
      "button[role='switch']"
    ) as HTMLButtonElement | null
    expect(citationsSwitch).toBeTruthy()
    expect(citationsSwitch).toBeDisabled()

    const rerankingRow = screen.getByText("Enable reranking").closest("div")
    const rerankingSwitch = rerankingRow?.querySelector(
      "button[role='switch']"
    ) as HTMLButtonElement | null
    expect(rerankingSwitch).toBeTruthy()
    expect(rerankingSwitch).toBeDisabled()

    fireEvent.click(searchModeInput)
    expect(mockSetRagSearchMode).not.toHaveBeenCalled()
    expect(mockSetRagEnableGeneration).not.toHaveBeenCalled()
    expect(mockSetRagEnableCitations).not.toHaveBeenCalled()
    expect(mockSetRagAdvancedOptions).not.toHaveBeenCalled()
  })

  it("ensures icon-only action buttons include aria-labels", () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-summary-a11y",
        type: "summary",
        title: "A11y Summary",
        status: "completed",
        content: "A11y text",
        createdAt: new Date("2026-02-18T08:00:00.000Z")
      }
    ]

    const { container } = renderExpandedStudioPane()

    const iconOnlyButtons = Array.from(
      container.querySelectorAll("button")
    ).filter((button) => {
      const hasIcon = Boolean(button.querySelector("svg"))
      const hasText = (button.textContent || "").trim().length > 0
      return hasIcon && !hasText
    })

    expect(iconOnlyButtons.length).toBeGreaterThan(0)
    iconOnlyButtons.forEach((button) => {
      expect(button).toHaveAttribute("aria-label")
      expect(button.getAttribute("aria-label")?.trim().length).toBeGreaterThan(0)
    })
  })

  it("keeps grouped artifact actions discoverable and keyboard-operable", async () => {
    workspaceStoreState.generatedArtifacts = [
      {
        id: "artifact-grouped-actions",
        type: "summary",
        title: "Grouped Actions Summary",
        status: "completed",
        content: "Content for grouped action checks.",
        createdAt: new Date("2026-02-18T08:00:00.000Z")
      }
    ]

    const dispatchSpy = vi.spyOn(window, "dispatchEvent")

    renderExpandedStudioPane()

    expect(
      screen.getByTestId("studio-artifact-primary-actions-artifact-grouped-actions")
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("studio-artifact-secondary-actions-artifact-grouped-actions")
    ).toBeInTheDocument()

    expect(screen.getByRole("button", { name: "View" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Download" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Regenerate options" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save to notes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()

    const discussButton = screen.getByRole("button", { name: "Discuss in chat" })
    discussButton.focus()
    expect(discussButton).toHaveFocus()

    fireEvent.keyDown(discussButton, { key: "Enter" })
    expect(dispatchSpy).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Regenerate options" }))
    expect(await screen.findByText("Replace existing")).toBeInTheDocument()
  })

  it("opens browser audio summaries with browser speech controls", async () => {
    const speak = vi.fn()
    const pause = vi.fn()
    const resume = vi.fn()
    const cancel = vi.fn()
    const speechSynthesisDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "speechSynthesis"
    )
    const utteranceDescriptor = Object.getOwnPropertyDescriptor(
      globalThis,
      "SpeechSynthesisUtterance"
    )

    try {
      Object.defineProperty(window, "speechSynthesis", {
        configurable: true,
        value: {
          speaking: false,
          paused: false,
          speak,
          pause,
          resume,
          cancel
        }
      })
      Object.defineProperty(globalThis, "SpeechSynthesisUtterance", {
        configurable: true,
        value: class {
          text: string
          rate = 1
          onstart: (() => void) | null = null
          onpause: (() => void) | null = null
          onresume: (() => void) | null = null
          onend: (() => void) | null = null
          onerror: (() => void) | null = null

          constructor(text: string) {
            this.text = text
          }
        }
      })

      workspaceStoreState.generatedArtifacts = [
        {
          id: "artifact-browser-audio",
          type: "audio_overview",
          title: "Browser Audio Summary",
          status: "completed",
          content: "Browser spoken summary text.",
          audioFormat: "browser",
          createdAt: new Date("2026-02-18T08:00:00.000Z")
        }
      ]

      renderExpandedStudioPane()

      fireEvent.click(screen.getByRole("button", { name: "View" }))

      expect(
        await screen.findByText("Use your browser to play this audio summary.")
      ).toBeInTheDocument()
      expect(screen.getByText("Browser spoken summary text.")).toBeInTheDocument()

      fireEvent.click(screen.getByRole("button", { name: "Play" }))
      expect(speak).toHaveBeenCalledTimes(1)

      fireEvent.click(screen.getByRole("button", { name: "Stop" }))
      expect(cancel).toHaveBeenCalled()
    } finally {
      if (speechSynthesisDescriptor) {
        Object.defineProperty(window, "speechSynthesis", speechSynthesisDescriptor)
      } else {
        delete (window as { speechSynthesis?: unknown }).speechSynthesis
      }

      if (utteranceDescriptor) {
        Object.defineProperty(globalThis, "SpeechSynthesisUtterance", utteranceDescriptor)
      } else {
        delete (globalThis as { SpeechSynthesisUtterance?: unknown }).SpeechSynthesisUtterance
      }
    }
  })
})
