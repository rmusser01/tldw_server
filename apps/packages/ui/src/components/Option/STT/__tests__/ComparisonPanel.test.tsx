import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { mockTranscribeAll, mockRetryModel, mockClearResults, hookReturnRef } =
  vi.hoisted(() => {
    const mockTranscribeAll = vi.fn()
    const mockRetryModel = vi.fn()
    const mockClearResults = vi.fn()
    const hookReturnRef = {
      current: {
        results: [] as any[],
        isRunning: false,
        transcribeAll: mockTranscribeAll,
        retryModel: mockRetryModel,
        clearResults: mockClearResults,
      },
    }
    return { mockTranscribeAll, mockRetryModel, mockClearResults, hookReturnRef }
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
      clearResults: mockClearResults,
    }
  })

  it("renders model select and disabled transcribe button when no blob", () => {
    render(
      <ComparisonPanel
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
      <ComparisonPanel
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
      <ComparisonPanel
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
      <ComparisonPanel
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
})
