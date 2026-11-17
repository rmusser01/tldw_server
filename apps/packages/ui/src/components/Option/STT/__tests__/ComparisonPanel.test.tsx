import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const {
  mockTranscribeAll,
  mockRetryModel,
  mockDuplicateResult,
  mockSetResultDisabled,
  mockClearResults,
  hookReturnRef,
} =
  vi.hoisted(() => {
    const mockTranscribeAll = vi.fn()
    const mockRetryModel = vi.fn()
    const mockDuplicateResult = vi.fn()
    const mockSetResultDisabled = vi.fn()
    const mockClearResults = vi.fn()
    const hookReturnRef = {
      current: {
        results: [] as any[],
        isRunning: false,
        transcribeAll: mockTranscribeAll,
        retryModel: mockRetryModel,
        duplicateResult: mockDuplicateResult,
        setResultDisabled: mockSetResultDisabled,
        clearResults: mockClearResults,
      },
    }
    return {
      mockTranscribeAll,
      mockRetryModel,
      mockDuplicateResult,
      mockSetResultDisabled,
      mockClearResults,
      hookReturnRef,
    }
  })

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | Record<string, any>) => {
      if (typeof fallback === "string") return fallback
      if (fallback && typeof fallback === "object" && "defaultValue" in fallback)
        return (fallback as any).defaultValue
      return key
    },
  }),
}))

vi.mock("@/hooks/useComparisonTranscribe", () => ({
  useComparisonTranscribe: () => hookReturnRef.current,
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  }),
}))

import { ComparisonPanel } from "../ComparisonPanel"

describe("ComparisonPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hookReturnRef.current = {
      results: [],
      isRunning: false,
      transcribeAll: mockTranscribeAll,
      retryModel: mockRetryModel,
      duplicateResult: mockDuplicateResult,
      setResultDisabled: mockSetResultDisabled,
      clearResults: mockClearResults,
    }
  })

  it("B05 source handoff carries immutable provenance with reviewed text", () => {
    const metadata={createdAt:"2026-10-16T22:59:32.000Z",audioSourceLabel:"New Recording 6.m4a"}
    hookReturnRef.current={...hookReturnRef.current,results:[{id:"source-result",model:"parakeet",text:"Provider text",status:"done",metadata}]}
    const save=vi.fn()
    render(<ComparisonPanel isCurrentOwner={()=>true} blob={new Blob(["unit audio"])} availableModels={["parakeet"]} sttOptions={{}} onSaveToNotes={save} />)
    fireEvent.change(screen.getByRole("textbox",{name:"Transcript from parakeet"}),{target:{value:"Reviewed exact text"}})
    fireEvent.click(screen.getByRole("button",{name:"Save to Notes",exact:true}))
    expect(save).toHaveBeenCalledWith("Reviewed exact text","parakeet",metadata)
  })

  it("B05 no speech result explains recovery and blocks empty handoffs until edited", () => {
    hookReturnRef.current = { ...hookReturnRef.current, results: [
      { id: "silent-result", model: "parakeet", text: "", status: "done", wordCount: 0 }
    ] }
    const save = vi.fn()
    render(<ComparisonPanel isCurrentOwner={() => true} blob={new Blob(["unit audio"])} availableModels={["parakeet"]} sttOptions={{}} onSaveToNotes={save} />)
    expect(screen.getByRole("status")).toHaveTextContent("No speech detected. Try a recording with clear speech.")
    expect(screen.getByRole("button", { name: "Copy", exact: true })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Save to Notes", exact: true })).toBeDisabled()
    fireEvent.change(screen.getByRole("textbox", { name: "Transcript from parakeet" }), { target: { value: "Reviewed manual text" } })
    expect(screen.getByRole("button", { name: "Save to Notes", exact: true })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Save to Notes", exact: true }))
    expect(save).toHaveBeenCalledWith("Reviewed manual text", "parakeet", undefined)
  })

  it("renders model select and disabled transcribe button when no blob", () => {
    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={null}
        availableModels={["whisper-1", "whisper-large-v3"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    // Model select should be present
    expect(screen.getByRole("combobox")).toBeInTheDocument()

    // Transcribe All button should be disabled
    const btn = screen.getByRole("button", { name: /transcribe all/i })
    expect(btn).toBeDisabled()
  })

  it("enables transcribe button when blob and models selected", () => {
    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={new Blob(["audio"], { type: "audio/webm" })}
        availableModels={["whisper-1", "whisper-large-v3"]}
        selectedModels={["whisper-1"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    const btn = screen.getByRole("button", { name: /transcribe all/i })
    expect(btn).not.toBeDisabled()
  })

  it("shows empty state message when no results", () => {
    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={null}
        availableModels={["whisper-1"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    expect(
      screen.getByText(
        /select models and record audio to compare transcription results/i
      )
    ).toBeInTheDocument()
  })

  it("shows classified recovery copy and settings link for STT result errors", () => {
    hookReturnRef.current = {
      ...hookReturnRef.current,
      results: [
        {
          id: "result-error",
          model: "whisper-large",
          text: "",
          status: "error",
          error: "Credentials need attention",
          errorRecovery: "Open Settings -> Speech, check the selected provider credentials, and retry.",
          errorSettingsHref: "/settings/speech"
        }
      ]
    }

    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={new Blob(["audio"], { type: "audio/webm" })}
        availableModels={["whisper-large"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    expect(screen.getByText("Credentials need attention")).toBeInTheDocument()
    expect(
      screen.getByText("Open Settings -> Speech, check the selected provider credentials, and retry.")
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Settings" })).toHaveAttribute(
      "href",
      "/settings/speech"
    )
  })

  it("keeps successful transcripts visible when another selected model fails", () => {
    hookReturnRef.current = {
      ...hookReturnRef.current,
      results: [
        {
          id: "result-success",
          model: "whisper-1",
          text: "Successful transcript remains available",
          status: "done"
        },
        {
          id: "result-error",
          model: "distil-v3",
          text: "",
          status: "error",
          error: "Model unavailable",
          errorRecovery: "Choose a different model and retry."
        }
      ]
    }

    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={new Blob(["audio"], { type: "audio/webm" })}
        availableModels={["whisper-1", "distil-v3"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    expect(screen.getByLabelText("Transcript from whisper-1"))
      .toHaveValue("Successful transcript remains available")
    expect(screen.getByText("Model unavailable")).toBeInTheDocument()
    expect(screen.getByText("Choose a different model and retry.")).toBeInTheDocument()
  })

  it("shows STT provenance metadata and repeat controls", () => {
    hookReturnRef.current = {
      ...hookReturnRef.current,
      results: [
        {
          id: "result-1",
          model: "whisper-large",
          text: "Hello world",
          status: "done",
          latencyMs: 1234,
          wordCount: 2,
          config: {
            model: "whisper-large",
            language: "en",
            task: "translate",
            responseFormat: "verbose_json",
            timestampGranularities: ["word", "segment"],
            segmentationEnabled: true
          },
          metadata: {
            createdAt: "2026-03-06T14:05:09.000Z",
            audioSourceLabel: "Recorded audio",
            audioSizeBytes: 1536,
            clientLatencyMs: 1234,
            language: "en",
            durationSeconds: 2.5,
            segmentCount: 2,
            wordCount: 2
          }
        }
      ]
    }

    render(
      <ComparisonPanel isCurrentOwner={() => true}
        blob={new Blob(["audio"], { type: "audio/webm" })}
        availableModels={["whisper-large"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )

    expect(screen.getByText("2026-03-06 14:05:09 UTC")).toBeInTheDocument()
    expect(screen.getByText("Recorded audio")).toBeInTheDocument()
    expect(screen.getByText("1.5 KB")).toBeInTheDocument()
    expect(screen.getByText("Client measured 1.2s")).toBeInTheDocument()
    expect(screen.getByText("Language en")).toBeInTheDocument()
    expect(screen.getByText("Task translate")).toBeInTheDocument()
    expect(screen.getByText("Format verbose_json")).toBeInTheDocument()
    expect(screen.getByText("Timestamps word, segment")).toBeInTheDocument()
    expect(screen.getByText("Segmentation on")).toBeInTheDocument()
    expect(screen.getByText("Duration 2.5s")).toBeInTheDocument()
    expect(screen.getByText("2 segments")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Duplicate" }))
    fireEvent.click(screen.getByRole("button", { name: "Disable" }))

    expect(mockDuplicateResult).toHaveBeenCalledWith("result-1")
    expect(mockSetResultDisabled).toHaveBeenCalledWith("result-1", true)
  })
  it("B05 replacing or clearing the audio retires the comparison request", () => {
    const props={ availableModels:["parakeet"], sttOptions:{}, onSaveToNotes:vi.fn() };
    const first=new Blob(["first source"]);
    const view=render(<ComparisonPanel isCurrentOwner={() => true} {...props} blob={first} />);
    mockClearResults.mockClear();
    view.rerender(<ComparisonPanel isCurrentOwner={() => true} {...props} blob={null} />);
    expect(mockClearResults).toHaveBeenCalledTimes(1);
    view.rerender(<ComparisonPanel isCurrentOwner={() => true} {...props} blob={new Blob(["second source"])} />);
    expect(mockClearResults).toHaveBeenCalledTimes(2);
    view.unmount();
  })

})
