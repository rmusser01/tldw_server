// @vitest-environment jsdom

import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"

const storageValues: Record<string, unknown> = {
  speechToTextLanguage: "fr",
  sttTask: "translate",
  sttResponseFormat: "verbose_json",
  sttTemperature: 0.4,
  sttPrompt: "Global prompt",
  sttUseSegmentation: true,
  sttSegK: 8,
  sttSegMinSegmentSize: 7,
  sttSegLambdaBalance: 0.2,
  sttSegUtteranceExpansionWidth: 3,
  sttSegEmbeddingsProvider: "openai",
  sttSegEmbeddingsModel: "text-embedding-3-small",
  sttComparisonHistory: []
}

let comparisonPanelProps: Record<string, unknown> | null = null
let recordingStripProps: Record<string, unknown> | null = null
const {
  getTranscriptionModelsMock,
  getTranscriptionModelHealthMock,
  transcribeAudioMock,
  createNoteMock,
  notificationErrorMock,
  notificationSuccessMock,
  isTimeoutLikeErrorMock,
  tMock,
  audioPresetControlPropsRef
} = vi.hoisted(() => ({
  getTranscriptionModelsMock: vi.fn(),
  getTranscriptionModelHealthMock: vi.fn(),
  transcribeAudioMock: vi.fn(),
  createNoteMock: vi.fn(),
  notificationErrorMock: vi.fn(),
  notificationSuccessMock: vi.fn(),
  isTimeoutLikeErrorMock: vi.fn(),
  tMock: vi.fn((_key: string, fallback?: string) => fallback || _key),
  audioPresetControlPropsRef: {
    current: null as null | {
      kind: string
      currentConfig: Record<string, unknown>
      onApply: (config: Record<string, unknown>, preset: any) => void
    }
  }
}))

// Mock all dependencies before importing the component
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultVal: unknown) => [
    key in storageValues ? storageValues[key] : defaultVal,
    vi.fn()
  ]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: tMock }),
  Trans: ({ defaults }: { defaults: string }) => <>{defaults}</>
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: getTranscriptionModelsMock,
    getTranscriptionModelHealth: getTranscriptionModelHealthMock,
    transcribeAudio: transcribeAudioMock,
    createNote: createNoteMock
  }
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
  )
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: notificationErrorMock,
    success: notificationSuccessMock,
    info: vi.fn()
  })
}))

vi.mock("@/utils/request-timeout", () => ({
  isTimeoutLikeError: (error: unknown) => {
    const message =
      error instanceof Error ? `${error.name} ${error.message}` : String(error ?? "")
    return /timeout|timed out/i.test(message)
  }
}))

// Mock the sub-components to keep tests focused
vi.mock("../RecordingStrip", () => ({
  RecordingStrip: (props: Record<string, unknown>) => {
    recordingStripProps = props
    return <div data-testid="recording-strip" />
  }
}))

vi.mock("../InlineSettingsPanel", () => ({
  InlineSettingsPanel: (_props: Record<string, unknown>) => (
    <div data-testid="settings-panel" />
  )
}))

vi.mock("../ComparisonPanel", () => ({
  ComparisonPanel: (props: Record<string, unknown>) => {
    comparisonPanelProps = props
    return <div data-testid="comparison-panel" />
  }
}))

vi.mock("../HistoryPanel", () => ({
  HistoryPanel: (_props: Record<string, unknown>) => (
    <div data-testid="history-panel" />
  )
}))

vi.mock("@/components/Option/Audio/AudioPresetControls", () => ({
  AudioPresetControls: (props: {
    kind: string
    currentConfig: Record<string, unknown>
    onApply: (config: Record<string, unknown>, preset: any) => void
  }) => {
    audioPresetControlPropsRef.current = props
    return (
      <button
        type="button"
        data-testid={`${props.kind}-preset-controls`}
        onClick={() =>
          props.onApply(
            {
              models: ["distil-v3"],
              language: "es",
              task: "transcribe",
              response_format: "verbose_json",
              temperature: 0.1,
              segment: true,
              seg_K: 9
            },
            { id: "preset-1", name: "Spanish verbose" }
          )
        }
      >
        Apply mock {props.kind} preset
      </button>
    )
  }
}))

vi.mock("@/db/dexie/stt-recordings", () => ({
  saveSttRecording: vi.fn().mockResolvedValue("rec-1"),
  getSttRecording: vi.fn(),
  deleteSttRecording: vi.fn()
}))

import { SttPlaygroundPage } from "../SttPlaygroundPage"

describe("SttPlaygroundPage", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    comparisonPanelProps = null
    recordingStripProps = null
    getTranscriptionModelsMock.mockReset()
    getTranscriptionModelsMock.mockResolvedValue({
      categories: {
        "Whisper Models": [
          {
            value: "whisper-1",
            label: "Whisper 1",
            description: "Default server model"
          }
        ],
        "Distil-Whisper Models": [
          {
            value: "distil-v3",
            label: "Distil v3",
            description: "Fast distilled model"
          }
        ]
      },
      all_models: ["whisper-1", "distil-v3"]
    })
    getTranscriptionModelHealthMock.mockReset()
    getTranscriptionModelHealthMock.mockResolvedValue({
      available: true,
      usable: true,
      on_demand: false,
      message: "Ready",
      provider: "whisper"
    })
    transcribeAudioMock.mockReset()
    transcribeAudioMock.mockResolvedValue({ text: "test" })
    createNoteMock.mockReset()
    createNoteMock.mockResolvedValue({})
    notificationErrorMock.mockReset()
    notificationSuccessMock.mockReset()
    isTimeoutLikeErrorMock.mockReset()
    isTimeoutLikeErrorMock.mockReturnValue(false)
    tMock.mockClear()
    audioPresetControlPropsRef.current = null
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })

  afterEach(() => {
    warnSpy.mockRestore()
  })

  it("renders the Speech to Text page heading", () => {
    render(<SttPlaygroundPage />)
    expect(
      screen.getByRole("heading", { level: 1, name: "Speech to Text" })
    ).toBeTruthy()
  })

  it("renders model readiness from the server catalog and health check", async () => {
    render(<SttPlaygroundPage />)

    expect(
      await screen.findByRole("status", { name: "STT readiness" })
    ).toHaveTextContent("STT models: Ready")
    expect(getTranscriptionModelHealthMock).toHaveBeenCalledWith("whisper-1")
    expect(comparisonPanelProps?.availableModelOptions).toEqual([
      expect.objectContaining({
        id: "distil-v3",
        label: "Distil v3",
        availability: "unknown"
      }),
      expect.objectContaining({
        id: "whisper-1",
        label: "Whisper 1",
        availability: "ready"
      })
    ])
  })

  it("keeps the route landmark visible while the model catalog is loading", () => {
    getTranscriptionModelsMock.mockImplementationOnce(
      () => new Promise(() => {})
    )

    render(<SttPlaygroundPage />)

    expect(screen.getByTestId("page-shell")).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { level: 1, name: "Speech to Text" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("status", { name: "STT readiness" })
    ).toHaveTextContent("STT models: Unknown")
    expect(recordingStripProps).toEqual(
      expect.objectContaining({
        disabled: true,
        disabledReason: "Loading transcription model catalog."
      })
    )
  })

  it("renders all 3 zones (recording-strip, comparison-panel, history-panel)", () => {
    render(<SttPlaygroundPage />)
    expect(screen.getByTestId("recording-strip")).toBeTruthy()
    expect(screen.getByTestId("comparison-panel")).toBeTruthy()
    expect(screen.getByTestId("history-panel")).toBeTruthy()
  })

  it("settings panel is hidden by default", () => {
    render(<SttPlaygroundPage />)
    expect(screen.queryByTestId("settings-panel")).toBeNull()
  })

  it("applies global STT defaults even before opening settings", () => {
    render(<SttPlaygroundPage />)

    expect(comparisonPanelProps?.sttOptions).toEqual({
      language: "fr",
      task: "translate",
      response_format: "verbose_json",
      temperature: 0.4,
      prompt: "Global prompt",
      segment: true,
      seg_K: 8,
      seg_min_segment_size: 7,
      seg_lambda_balance: 0.2,
      seg_utterance_expansion_width: 3,
      seg_embeddings_provider: "openai",
      seg_embeddings_model: "text-embedding-3-small"
    })
  })

  it("renders STT preset controls and applies saved preset config to comparison settings", async () => {
    render(<SttPlaygroundPage />)

    expect(screen.getByTestId("stt-preset-controls")).toBeTruthy()
    expect(audioPresetControlPropsRef.current?.currentConfig).toEqual(
      expect.objectContaining({
        language: "fr",
        response_format: "verbose_json"
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "Apply mock stt preset" }))

    await waitFor(() => {
      expect(comparisonPanelProps?.selectedModels).toEqual(["distil-v3"])
      expect(comparisonPanelProps?.sttOptions).toEqual(
        expect.objectContaining({
          language: "es",
          response_format: "verbose_json",
          temperature: 0.1,
          segment: true,
          seg_K: 9
        })
      )
    })
    expect(screen.getByTestId("settings-panel")).toBeTruthy()
  })

  it("applies an explicit empty model list from a saved preset", async () => {
    render(<SttPlaygroundPage />)

    act(() => {
      audioPresetControlPropsRef.current?.onApply(
        {
          models: [],
          language: "en"
        },
        { id: "preset-empty", name: "No selected models" }
      )
    })

    await waitFor(() => {
      expect(comparisonPanelProps?.selectedModels).toEqual([])
      expect(comparisonPanelProps?.sttOptions).toEqual(
        expect.objectContaining({
          language: "en"
        })
      )
    })
  })

  it("shows an inline retry control when model loading times out", async () => {
    getTranscriptionModelsMock
      .mockRejectedValueOnce(new Error("timeout while loading transcription models"))
      .mockResolvedValueOnce({ all_models: ["whisper-1", "parakeet-tdt"] })
    isTimeoutLikeErrorMock.mockReturnValue(true)

    render(<SttPlaygroundPage />)

    const loadingErrorAlert = await screen.findByText("Model load failed")
    expect(loadingErrorAlert.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    const retryButton = await screen.findByRole("button", { name: "Retry" })
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Retry" })).toBeNull()
    })
  })

  it("uses the canonical alert for an empty server model catalog", async () => {
    getTranscriptionModelsMock.mockResolvedValue({
      categories: {},
      all_models: []
    })

    render(<SttPlaygroundPage />)

    const noModelsAlert = await screen.findByText("No transcription models available")
    expect(noModelsAlert.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(
      screen.getByText(
        "Configure STT models in your server settings. Check the Audio Setup Guide for instructions."
      )
    ).toBeInTheDocument()
    expect(recordingStripProps).toEqual(
      expect.objectContaining({
        disabled: true,
        disabledReason: "No transcription models are available. Configure STT models before recording."
      })
    )
  })

  it("preserves transcript text when saving to notes fails", async () => {
    createNoteMock.mockRejectedValueOnce(new Error("Notes database unavailable"))
    render(<SttPlaygroundPage />)

    await waitFor(() => expect(comparisonPanelProps).not.toBeNull())
    await act(async () => {
      await (
        comparisonPanelProps?.onSaveToNotes as (
          text: string,
          model: string
        ) => Promise<void>
      )(
        "Transcript text to preserve",
        "whisper-1"
      )
    })

    expect(createNoteMock).toHaveBeenCalledWith(
      "Transcript text to preserve",
      expect.objectContaining({
        metadata: expect.objectContaining({
          origin: "stt-playground",
          stt_model: "whisper-1"
        })
      })
    )
    expect(notificationErrorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Error",
        description: "Notes database unavailable"
      })
    )
  })
})
