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
let historyPanelProps: Record<string, unknown> | null = null
const {
  getTranscriptionModelsMock,
  getTranscriptionModelHealthMock,
  getTranscriptionCapabilitiesMock,
  transcribeAudioMock,
  createNoteMock,
  getSttRecordingMock,
  notificationErrorMock,
  notificationSuccessMock,
  isTimeoutLikeErrorMock,
  tMock,
  audioPresetControlPropsRef
} = vi.hoisted(() => ({
  getTranscriptionModelsMock: vi.fn(),
  getTranscriptionModelHealthMock: vi.fn(),
  getTranscriptionCapabilitiesMock: vi.fn(),
  transcribeAudioMock: vi.fn(),
  createNoteMock: vi.fn(),
  getSttRecordingMock: vi.fn(),
  notificationErrorMock: vi.fn(),
  notificationSuccessMock: vi.fn(),
  isTimeoutLikeErrorMock: vi.fn(),
  tMock: vi.fn((_key: string, fallback?: string) => fallback || _key),
  audioPresetControlPropsRef: {
    current: null as null | {
      kind: string
      currentConfig: Record<string, unknown>
      onApply: (config: Record<string, unknown>, preset: unknown) => void
    }
  }
}))

vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope: async () => ({config:{serverUrl:"http://unit-server",authMode:"multi-user"},userId:"test-owner"}) }))
vi.mock("@/hooks/useChatDraftOwner", () => ({ useChatDraftOwner: () => ({ ownerKey: "test-owner", isCurrent: () => true }) }))

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
    getTranscriptionCapabilities: getTranscriptionCapabilitiesMock,
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
  HistoryPanel: (props: Record<string, unknown>) => {
    historyPanelProps = props
    return <div data-testid="history-panel" />
  }
}))

vi.mock("@/components/Option/Audio/AudioPresetControls", () => ({
  AudioPresetControls: (props: {
    kind: string
    currentConfig: Record<string, unknown>
    onApply: (config: Record<string, unknown>, preset: unknown) => void
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
  getSttRecording: (...args: unknown[]) => getSttRecordingMock(...args),
  deleteSttRecording: vi.fn()
}))

import { SttPlaygroundPage } from "../SttPlaygroundPage"
import type { SttHistoryEntry } from "../HistoryPanel"

describe("SttPlaygroundPage", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    comparisonPanelProps = null
    recordingStripProps = null
    historyPanelProps = null
    storageValues.sttComparisonHistory = []
    delete storageValues["sttComparisonHistory:test-owner"]
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
    getTranscriptionCapabilitiesMock.mockReset()
    getTranscriptionCapabilitiesMock.mockResolvedValue({ models: [] })
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
    getSttRecordingMock.mockReset()
    getSttRecordingMock.mockResolvedValue(null)
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

  it("B05 history restore hands the owned saved audio to the existing recorder", async () => {
    const blob = new Blob(["owned saved audio"])
    getSttRecordingMock.mockResolvedValueOnce({ blob, durationMs: 12800, ownerKey: "test-owner" })
    render(<SttPlaygroundPage />)
    await act(async () => { await (historyPanelProps?.onRecompare as (entry: unknown) => Promise<void>)({ recordingId: "saved", durationMs: 12800 }) })
    expect(recordingStripProps?.restoredRecording).toEqual(expect.objectContaining({ blob, durationMs: 12800, ownerKey: "test-owner" }))
    expect(getSttRecordingMock).toHaveBeenCalledWith("saved", "test-owner")
    expect(comparisonPanelProps?.blob).toBeNull()
  })

  it("B05 history restore cannot overwrite a new source intent before its effect", async () => {
    let resolve!: (value: unknown) => void
    getSttRecordingMock.mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    render(<SttPlaygroundPage />)
    let pending!: Promise<void>
    act(() => { pending = (historyPanelProps?.onRecompare as (entry: unknown) => Promise<void>)({ recordingId: "old", durationMs: 9000 }) })
    act(() => { (recordingStripProps?.onSourceChange as (() => void) | undefined)?.() })
    await act(async () => { resolve({ blob: new Blob(["old history"]), durationMs: 9000 }); await pending })
    expect(comparisonPanelProps?.blob).toBeNull()
    expect(recordingStripProps?.restoredRecording).toBeFalsy()
  })

  it("B05 source handoff persists the reviewed source label and transcription time in the note title", async () => {
    render(<SttPlaygroundPage />)
    await act(async()=>{
      await (comparisonPanelProps?.onSaveToNotes as (text:string,model:string,metadata:{audioSourceLabel:string;createdAt:string})=>Promise<void>)("Reviewed exact text","parakeet",{audioSourceLabel:"New Recording 6.m4a",createdAt:"2026-10-16T22:59:32.000Z"})
    })
    expect(createNoteMock.mock.calls[0][0]).toBe("Reviewed exact text")
    expect(createNoteMock.mock.calls[0][1].title).toContain("New Recording 6.m4a")
    expect(createNoteMock.mock.calls[0][1].title).toContain("2026-10-16T22:59:32.000Z")
  })

  it("B05 source handoff reserves model and transcription time when the filename is long", async () => {
    render(<SttPlaygroundPage />)
    await act(async () => {
      await (comparisonPanelProps?.onSaveToNotes as (text: string, model: string, metadata: { audioSourceLabel: string; createdAt: string }) => Promise<void>)(
        "Reviewed exact text", "parakeet", {
          audioSourceLabel: "recording".repeat(100) + ".m4a",
          createdAt: "2026-10-16T22:59:32.000Z"
        }
      )
    })
    const title = createNoteMock.mock.calls[0][1].title
    expect(title.length).toBeLessThanOrEqual(255)
    expect(title).toContain("parakeet - 2026-10-16T22:59:32.000Z")
    expect(createNoteMock.mock.calls[0][0]).toBe("Reviewed exact text")
  })

  it.each([true, false])("B05 history export retains raw text and available source identity (metadata %s)", async (withMetadata) => {
    const originalClipboard = Object.getOwnPropertyDescriptor(navigator, "clipboard")
    const writeText = vi.fn(async (_text: string) => {})
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } })
    try {
      render(<SttPlaygroundPage />)
      const entry: SttHistoryEntry = {
        id: "history-source", recordingId: "local-recording-1", createdAt: "2026-10-16T23:00:00.000Z", durationMs: 3000,
        results: [{ model: "parakeet", text: "Original raw transcript", metadata: withMetadata ? {
          audioSourceLabel: "New Recording 6.m4a", createdAt: "2026-10-16T22:59:32.000Z"
        } : undefined }]
      }
      await act(async () => {
        await (historyPanelProps?.onExport as (entry: SttHistoryEntry) => Promise<void>)(entry)
      })
      expect(writeText).toHaveBeenCalledTimes(1)
      const markdown = writeText.mock.calls[0][0]
      expect(markdown).toContain("Source: " + (withMetadata ? "New Recording 6.m4a" : "local-recording-1"))
      if (withMetadata) expect(markdown).toContain("Transcribed: 2026-10-16T22:59:32.000Z")
      expect(markdown).toContain("Original raw transcript")
      expect(markdown).toContain("## parakeet")
      expect(markdown).toContain("Duration: 3.0s")
    } finally {
      if (originalClipboard) Object.defineProperty(navigator, "clipboard", originalClipboard)
      else Reflect.deleteProperty(navigator, "clipboard")
    }
  })

  it("renders the Speech to Text page heading", () => {
    render(<SttPlaygroundPage />)
    expect(
      screen.getByRole("heading", { level: 1, name: "Speech to Text" })
    ).toBeTruthy()
  })

  it("B05 unavailable selected model is not hidden by another ready model", async () => {
    getTranscriptionCapabilitiesMock.mockResolvedValue({ models: [
      { id: "distil-v3", availability: "unavailable", availability_source: "health" },
      { id: "whisper-1", availability: "ready", availability_source: "health" }
    ] })
    render(<SttPlaygroundPage />)
    await waitFor(() => expect(comparisonPanelProps?.availableModelOptions).toEqual([
      expect.objectContaining({ id: "distil-v3", availability: "unavailable" }),
      expect.objectContaining({ id: "whisper-1", availability: "ready" })
    ]))
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent("STT models: Ready")
    act(() => (comparisonPanelProps?.onSelectedModelsChange as (models: string[]) => void)(["distil-v3"]))
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent("STT models: Blocked")
    act(() => (comparisonPanelProps?.onSelectedModelsChange as (models: string[]) => void)([]))
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent("STT models: Ready")
  })

  it("renders model readiness from the server catalog and health check", async () => {
    render(<SttPlaygroundPage />)

    await waitFor(() => {
      expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent("STT models: Ready")
    })
    expect(getTranscriptionModelHealthMock).toHaveBeenCalledWith("distil-v3")
    expect(comparisonPanelProps?.availableModelOptions).toEqual([
      expect.objectContaining({
        id: "distil-v3",
        label: "Distil v3",
        availability: "ready"
      }),
      expect.objectContaining({
        id: "whisper-1",
        label: "Whisper 1",
        availability: "unknown"
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
        disabledReason: "Loading transcription model catalog.",
        disabledStatusType: "secondary"
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
        disabledReason: "No transcription models are available. Configure STT models before recording.",
        disabledStatusType: "danger"
      })
    )
  })

  it("only prevents Space scrolling when the recording strip handles the shortcut", () => {
    render(<SttPlaygroundPage />)

    const unhandledSpace = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      code: "Space"
    })
    window.dispatchEvent(unhandledSpace)

    expect(unhandledSpace.defaultPrevented).toBe(false)

    const handleToggle = (event: Event) => event.preventDefault()
    window.addEventListener("stt-toggle-record", handleToggle, { once: true })

    const handledSpace = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      code: "Space"
    })
    window.dispatchEvent(handledSpace)

    expect(handledSpace.defaultPrevented).toBe(true)
  })

  it("sanitizes save-to-notes failure notifications", async () => {
    createNoteMock.mockRejectedValueOnce(
      new Error(
        "Request failed: 500 (POST /api/v1/notes) token=sk_secret_inline Authorization: Bearer sk-secret-inline /Users/alice/private/stt-note.json"
      )
    )
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
      }),
      expect.objectContaining({ requestScope: expect.objectContaining({userId:"test-owner"}), signal: expect.any(AbortSignal) })
    )
    expect(notificationErrorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Error",
        description: expect.stringContaining("POST [server-endpoint]")
      })
    )
    const payload = notificationErrorMock.mock.calls.at(-1)?.[0] as {
      description?: string
    }
    expect(payload.description).toContain("[redacted-secret]")
    expect(payload.description).toContain("[redacted-path]")
    expect(payload.description).not.toContain("/api/v1/notes")
    expect(payload.description).not.toContain("sk_secret_inline")
    expect(payload.description).not.toContain("sk-secret-inline")
    expect(payload.description).not.toContain("/Users/alice")
  })

  it("sanitizes recording history load failure notifications", async () => {
    storageValues["sttComparisonHistory:test-owner"] = { ownerKey: "test-owner", items: [
      {
        id: "hist-1",
        recordingId: "recording-1",
        createdAt: "2026-06-26T12:00:00.000Z",
        durationMs: 1200,
        results: [
          {
            model: "whisper-1",
            text: "Existing transcript"
          }
        ]
      }
    ] }
    getSttRecordingMock.mockRejectedValueOnce(
      new Error(
        "IndexedDB read failed (GET /api/v1/audio/transcriptions/history) secret=sk-history-secret /home/alice/stt-recording.webm"
      )
    )

    render(<SttPlaygroundPage />)

    await waitFor(() => expect(historyPanelProps).not.toBeNull())
    const [entry] = historyPanelProps?.entries as Array<Record<string, unknown>>

    await act(async () => {
      await (
        historyPanelProps?.onRecompare as (
          entry: Record<string, unknown>
        ) => Promise<void>
      )(entry)
    })

    expect(notificationErrorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Error",
        description: expect.stringContaining("GET [server-endpoint]")
      })
    )
    const payload = notificationErrorMock.mock.calls.at(-1)?.[0] as {
      description?: string
    }
    expect(payload.description).toContain("[redacted-secret]")
    expect(payload.description).toContain("[redacted-path]")
    expect(payload.description).not.toContain("/api/v1/audio/transcriptions/history")
    expect(payload.description).not.toContain("sk-history-secret")
    expect(payload.description).not.toContain("/home/alice")
  })
})
