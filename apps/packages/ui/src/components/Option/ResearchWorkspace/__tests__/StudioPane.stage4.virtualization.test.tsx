import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { StudioPane } from "../StudioPane"

const {
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
  const removeArtifact = vi.fn()
  const restoreArtifact = vi.fn()
  const setIsGeneratingOutput = vi.fn()
  const setAudioSettings = vi.fn()
  const captureToCurrentNote = vi.fn()
  const getChatModels = vi.fn()
  const messageSuccess = vi.fn()
  const messageError = vi.fn()
  const messageInfo = vi.fn()

  const state = {
    selectedSourceIds: ["source-1"],
    getSelectedMediaIds: () => [101],
    generatedArtifacts: [] as Array<any>,
    isGeneratingOutput: false,
    generatingOutputType: null as any,
    workspaceTag: "workspace:test",
    audioSettings: {
      provider: "tldw" as const,
      model: "kokoro",
      voice: "af_heart",
      speed: 1,
      format: "mp3" as const
    },
    addArtifact: vi.fn(),
    updateArtifactStatus: vi.fn(),
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
  useMobile: () => false
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
  generateQuiz: vi.fn()
}))

vi.mock("@/services/flashcards", () => ({
  listDecks: vi.fn().mockResolvedValue([]),
  createDeck: vi.fn(),
  createFlashcard: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    ragSearch: vi.fn(),
    synthesizeSpeech: vi.fn(),
    generateSlidesFromMedia: vi.fn(),
    listVisualStyles: vi.fn().mockResolvedValue([]),
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
          warning: vi.fn(),
          open: vi.fn(),
          destroy: vi.fn()
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

const expandGeneratedOutputsSection = () => {
  const toggle = screen.getByRole("button", { name: /Generated Outputs/i })
  if (toggle.getAttribute("aria-expanded") === "false") {
    fireEvent.click(toggle)
  }
}

const renderStudioPane = () => {
  const renderResult = render(<StudioPane />)
  expandGeneratedOutputsSection()
  return renderResult
}

const buildArtifacts = (count: number) =>
  Array.from({ length: count }, (_, index) => ({
    id: `artifact-${index}`,
    type: "summary",
    title: `Artifact ${index}`,
    status: "completed",
    content: `Content ${index}`,
    createdAt: new Date("2026-02-18T00:00:00.000Z")
  }))

describe("StudioPane Stage 4 outputs virtualization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.generatedArtifacts = buildArtifacts(120)
    workspaceStoreState.isGeneratingOutput = false
    workspaceStoreState.generatingOutputType = null
    workspaceStoreState.noteFocusTarget = null
    mockGetChatModels.mockResolvedValue([])
  })

  it("activates virtualization when artifact history is large", () => {
    renderStudioPane()

    const listContainer = screen.getByTestId("generated-outputs-virtualized")
    expect(listContainer).toHaveAttribute("data-virtualized", "true")
    expect(screen.getByText("Artifact 0")).toBeInTheDocument()
    expect(screen.queryByText("Artifact 90")).not.toBeInTheDocument()
  })

  it("keeps row actions functional after scrolling a virtualized list", async () => {
    const confirmSpy = vi
      .spyOn(Modal, "confirm")
      .mockImplementation((config: any) => {
        if (typeof config?.onOk === "function") {
          config.onOk()
        }
        return { destroy: vi.fn(), update: vi.fn() } as any
      })

    renderStudioPane()

    const listContainer = screen.getByTestId("generated-outputs-virtualized")
    fireEvent.scroll(listContainer, { target: { scrollTop: 4_500 } })

    await waitFor(() => {
      expect(screen.getByText("Artifact 30")).toBeInTheDocument()
    })

    const row = screen.getByText("Artifact 30").closest("div.group") as HTMLElement | null
    expect(row).toBeTruthy()
    if (row) {
      expect(
        within(row).getByLabelText("Regenerate options")
      ).toBeInTheDocument()
      fireEvent.click(within(row).getByLabelText("Delete"))
    }

    expect(confirmSpy).toHaveBeenCalledTimes(1)
    expect(mockRemoveArtifact).toHaveBeenCalledWith("artifact-30")
  })
})
