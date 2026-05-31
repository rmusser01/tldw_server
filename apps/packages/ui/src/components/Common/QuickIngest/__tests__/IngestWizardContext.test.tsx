import React from "react"
import { describe, it, expect, vi } from "vitest"
import { render, screen, act } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  IngestWizardProvider,
  useIngestWizard,
} from "../IngestWizardContext"
import { configMatchesPreset } from "../presets"
import type { WizardQueueItem } from "../types"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultOrOpts?: unknown) =>
      typeof defaultOrOpts === "string"
        ? defaultOrOpts
        : (defaultOrOpts as Record<string, string>)?.defaultValue ?? key,
  }),
}))

// ---------------------------------------------------------------------------
// Test harness component
// ---------------------------------------------------------------------------

/** Renders wizard state and exposes action buttons for testing. */
function TestHarness() {
  const {
    state,
    goNext,
    goBack,
    goToStep,
    setQueueItems,
    setPreset,
    setCustomOptions,
    startProcessing,
    skipToProcessing,
    cancelProcessing,
    cancelItem,
    minimize,
    restore,
    reset,
  } = useIngestWizard()

  return (
    <div>
      <span data-testid="currentStep">{state.currentStep}</span>
      <span data-testid="highestStep">{state.highestStep}</span>
      <span data-testid="preset">{state.selectedPreset}</span>
      <span data-testid="queueLen">{state.queueItems.length}</span>
      <span data-testid="status">{state.processingState.status}</span>
      <span data-testid="progressIds">
        {state.processingState.perItemProgress.map((item) => item.id).join(",")}
      </span>
      <span data-testid="isMinimized">{String(state.isMinimized)}</span>
      <span data-testid="presetAnalysis">
        {String(state.presetConfig.common.perform_analysis)}
      </span>
      <span data-testid="presetChunkingMode">
        {state.presetConfig.common.chunking_mode || ""}
      </span>
      <span data-testid="presetAutoChunkingGoal">
        {state.presetConfig.common.auto_chunking_goal || ""}
      </span>
      <span data-testid="presetAutoChunkingUseLlm">
        {String(Boolean(state.presetConfig.common.auto_chunking_use_llm))}
      </span>
      <span data-testid="presetAudioLanguage">
        {state.presetConfig.typeDefaults.audio?.language || ""}
      </span>

      <button onClick={goNext}>goNext</button>
      <button onClick={goBack}>goBack</button>
      <button onClick={() => goToStep(1)}>goToStep1</button>
      <button onClick={() => goToStep(3)}>goToStep3</button>
      <button onClick={() => goToStep(5 as 5)}>goToStep5</button>
      <button
        onClick={() =>
          setQueueItems([
            {
              id: "a",
              detectedType: "audio",
              fileSize: 100,
              icon: "mic",
              validation: { valid: true },
            },
            {
              id: "b",
              detectedType: "video",
              fileSize: 200,
              icon: "video",
              validation: { valid: true },
            },
          ] as WizardQueueItem[])
        }
      >
        setQueue
      </button>
      <button
        onClick={() =>
          setQueueItems([
            {
              id: "valid-a",
              detectedType: "audio",
              fileSize: 100,
              icon: "mic",
              validation: { valid: true },
            },
            {
              id: "invalid-b",
              detectedType: "unknown",
              fileSize: 0,
              icon: "file",
              validation: { valid: false, errors: ["Invalid URL format"] },
            },
          ] as WizardQueueItem[])
        }
      >
        setMixedQueue
      </button>
      <button
        onClick={() =>
          setQueueItems([
            {
              id: "invalid-only",
              detectedType: "unknown",
              fileSize: 0,
              icon: "file",
              validation: { valid: false, errors: ["Unsupported file type"] },
            },
          ] as WizardQueueItem[])
        }
      >
        setInvalidQueue
      </button>
      <button onClick={() => setPreset("deep")}>setDeep</button>
      <button onClick={() => setPreset("custom")}>setCustomPreset</button>
      <button
        onClick={() =>
          setCustomOptions({
            common: {
              perform_analysis: false,
              perform_chunking: false,
              overwrite_existing: true,
            },
          })
        }
      >
        setCustomOpts
      </button>
      <button
        onClick={() =>
          setCustomOptions({
            typeDefaults: {
              audio: {
                language: "fr",
              },
            },
          })
        }
      >
        setAudioLanguage
      </button>
      <button onClick={skipToProcessing}>skipToProcessing</button>
      <button onClick={startProcessing}>startProcessing</button>
      <button onClick={cancelProcessing}>cancelProcessing</button>
      <button onClick={() => cancelItem("a")}>cancelItemA</button>
      <button onClick={minimize}>minimize</button>
      <button onClick={restore}>restore</button>
      <button onClick={reset}>reset</button>
    </div>
  )
}

function renderWithProvider() {
  return render(
    <IngestWizardProvider>
      <TestHarness />
    </IngestWizardProvider>
  )
}

function renderWithInitialState(initialState: Parameters<typeof IngestWizardProvider>[0]["initialState"]) {
  return render(
    <IngestWizardProvider initialState={initialState}>
      <TestHarness />
    </IngestWizardProvider>
  )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("IngestWizardContext", () => {
  describe("hydration", () => {
    it("hydrates the provider from an explicit initial state", () => {
      renderWithInitialState({
        currentStep: 5,
        highestStep: 5,
        queueItems: [
          {
            id: "persisted-url",
            url: "https://example.com/article",
            detectedType: "web",
            icon: "Globe",
            fileSize: 0,
            validation: { valid: true },
          },
        ],
        selectedPreset: "deep",
        customBasePreset: "deep",
        presetConfig: {
          common: {
            perform_analysis: true,
            perform_chunking: true,
            overwrite_existing: true,
          },
          storeRemote: true,
          reviewBeforeStorage: false,
          typeDefaults: {},
          advancedValues: {},
        },
        customOptions: {},
        processingState: {
          status: "complete",
          perItemProgress: [
            {
              id: "persisted-url",
              status: "complete",
              progressPercent: 100,
              currentStage: "Complete",
              estimatedRemaining: 0,
            },
          ],
          elapsed: 8,
          estimatedRemaining: 0,
        },
        results: [
          {
            id: "persisted-url",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
          },
        ],
        isMinimized: true,
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("5")
      expect(screen.getByTestId("highestStep").textContent).toBe("5")
      expect(screen.getByTestId("preset").textContent).toBe("deep")
      expect(screen.getByTestId("queueLen").textContent).toBe("1")
      expect(screen.getByTestId("status").textContent).toBe("complete")
      expect(screen.getByTestId("isMinimized").textContent).toBe("true")
    })

    it("publishes state transitions through onStateChange", async () => {
      const onStateChange = vi.fn()

      render(
        <IngestWizardProvider onStateChange={onStateChange}>
          <TestHarness />
        </IngestWizardProvider>
      )

      await act(async () => {
        await userEvent.click(screen.getByText("goNext"))
      })

      expect(onStateChange).toHaveBeenCalled()
      expect(onStateChange).toHaveBeenLastCalledWith(
        expect.objectContaining({
          currentStep: 2,
          highestStep: 2,
        })
      )
    })
  })

  // -- Navigation -----------------------------------------------------------

  describe("navigation", () => {
    it("goNext advances step from 1 to 2 and updates highestStep", async () => {
      renderWithProvider()
      expect(screen.getByTestId("currentStep").textContent).toBe("1")

      await act(async () => {
        await userEvent.click(screen.getByText("goNext"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("2")
      expect(screen.getByTestId("highestStep").textContent).toBe("2")
    })

    it("goBack goes from 2 to 1", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("goNext"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("2")

      await act(async () => {
        await userEvent.click(screen.getByText("goBack"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("1")
    })

    it("goBack does not go below step 1", async () => {
      renderWithProvider()
      expect(screen.getByTestId("currentStep").textContent).toBe("1")

      await act(async () => {
        await userEvent.click(screen.getByText("goBack"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("1")
    })

    it("goToStep only allows steps <= highestStep", async () => {
      renderWithProvider()
      // highestStep is 1; trying to go to step 3 should be a no-op
      await act(async () => {
        await userEvent.click(screen.getByText("goToStep3"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("1")

      // Advance to step 3 via goNext twice
      await act(async () => {
        await userEvent.click(screen.getByText("goNext"))
        await userEvent.click(screen.getByText("goNext"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("3")
      expect(screen.getByTestId("highestStep").textContent).toBe("3")

      // Now goToStep(1) should work
      await act(async () => {
        await userEvent.click(screen.getByText("goToStep1"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("1")
    })
  })

  // -- Queue ----------------------------------------------------------------

  describe("queue", () => {
    it("setQueueItems updates the queue", async () => {
      renderWithProvider()
      expect(screen.getByTestId("queueLen").textContent).toBe("0")

      await act(async () => {
        await userEvent.click(screen.getByText("setQueue"))
      })
      expect(screen.getByTestId("queueLen").textContent).toBe("2")
    })
  })

  // -- Presets & options ----------------------------------------------------

  describe("presets", () => {
    it("setPreset changes selectedPreset and resolves presetConfig", async () => {
      renderWithProvider()
      expect(screen.getByTestId("preset").textContent).toBe("standard")

      await act(async () => {
        await userEvent.click(screen.getByText("setDeep"))
      })
      expect(screen.getByTestId("preset").textContent).toBe("deep")
    })

    it("defaults chunking-enabled presets to auto chunking", () => {
      renderWithProvider()

      expect(screen.getByTestId("preset").textContent).toBe("standard")
      expect(screen.getByTestId("presetChunkingMode").textContent).toBe("auto")
      expect(screen.getByTestId("presetAutoChunkingGoal").textContent).toBe("balanced")
      expect(screen.getByTestId("presetAutoChunkingUseLlm").textContent).toBe("false")
    })

    it("ignores inactive manual chunking fields when matching auto presets", () => {
      const matchesStandard = configMatchesPreset(
        {
          common: {
            perform_analysis: true,
            perform_chunking: true,
            overwrite_existing: false,
            chunking_mode: "auto",
            auto_chunking_goal: "balanced",
            auto_chunking_use_llm: false,
          },
          storeRemote: true,
          reviewBeforeStorage: false,
          typeDefaults: {
            audio: { diarize: false },
            document: { ocr: true },
            video: { captions: true },
          },
          advancedValues: {
            chunk_method: "tokens",
            chunk_size: 0,
            chunk_overlap: -5,
            hierarchical_chunking: true,
            hierarchical_template: { boundaries: [{ kind: "heading" }] },
          },
        },
        "standard"
      )

      expect(matchesStandard).toBe(true)
    })

    it("setCustomOptions merges custom options into presetConfig", async () => {
      renderWithProvider()
      // Default standard has perform_analysis = true
      expect(screen.getByTestId("presetAnalysis").textContent).toBe("true")

      // Switch to custom preset first, then set options
      await act(async () => {
        await userEvent.click(screen.getByText("setCustomPreset"))
      })
      await act(async () => {
        await userEvent.click(screen.getByText("setCustomOpts"))
      })
      expect(screen.getByTestId("presetAnalysis").textContent).toBe("false")
    })

    it("setCustomOptions switches to a custom configuration when changing a preset", async () => {
      renderWithProvider()

      expect(screen.getByTestId("preset").textContent).toBe("standard")
      expect(screen.getByTestId("presetAnalysis").textContent).toBe("true")

      await act(async () => {
        await userEvent.click(screen.getByText("setCustomOpts"))
      })

      expect(screen.getByTestId("preset").textContent).toBe("custom")
      expect(screen.getByTestId("presetAnalysis").textContent).toBe("false")
    })

    it("treats audio language changes as a custom deviation from the preset", async () => {
      renderWithProvider()

      expect(screen.getByTestId("preset").textContent).toBe("standard")
      expect(screen.getByTestId("presetAudioLanguage").textContent).toBe("")

      await act(async () => {
        await userEvent.click(screen.getByText("setAudioLanguage"))
      })

      expect(screen.getByTestId("preset").textContent).toBe("custom")
      expect(screen.getByTestId("presetAudioLanguage").textContent).toBe("fr")
    })
  })

  // -- Processing -----------------------------------------------------------

  describe("processing", () => {
    it("skipToProcessing jumps to step 4 with running status", async () => {
      renderWithProvider()

      // Add items first so perItemProgress is populated
      await act(async () => {
        await userEvent.click(screen.getByText("setQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("4")
      expect(screen.getByTestId("status").textContent).toBe("running")
    })

    it("initializes processing progress only for valid queue items", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setMixedQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("status").textContent).toBe("running")
      expect(screen.getByTestId("progressIds").textContent).toBe("valid-a")
    })

    it("initializes quick processing progress only for valid queue items", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setMixedQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("4")
      expect(screen.getByTestId("progressIds").textContent).toBe("valid-a")
    })

    it("does not start processing an invalid-only queue", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setInvalidQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("progressIds").textContent).toBe("")
    })

    it("does not quick-process an invalid-only queue", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setInvalidQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("1")
      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("progressIds").textContent).toBe("")
    })

    it("cancelProcessing sets status to cancelled", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setQueue"))
      })
      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })
      expect(screen.getByTestId("status").textContent).toBe("running")

      await act(async () => {
        await userEvent.click(screen.getByText("cancelProcessing"))
      })
      expect(screen.getByTestId("status").textContent).toBe("cancelled")
    })
  })

  // -- Cancel item ----------------------------------------------------------

  describe("cancelItem", () => {
    it("cancels a specific queued item without affecting others", async () => {
      // We test indirectly: after cancelling item "a", processing state should
      // still exist and status should still be "running" (cancel only affects
      // the individual item, not overall status).
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setQueue"))
      })
      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })
      await act(async () => {
        await userEvent.click(screen.getByText("cancelItemA"))
      })
      // Overall status remains running
      expect(screen.getByTestId("status").textContent).toBe("running")
    })
  })

  // -- Minimize / restore ---------------------------------------------------

  describe("minimize / restore", () => {
    it("toggles isMinimized", async () => {
      renderWithProvider()
      expect(screen.getByTestId("isMinimized").textContent).toBe("false")

      await act(async () => {
        await userEvent.click(screen.getByText("minimize"))
      })
      expect(screen.getByTestId("isMinimized").textContent).toBe("true")

      await act(async () => {
        await userEvent.click(screen.getByText("restore"))
      })
      expect(screen.getByTestId("isMinimized").textContent).toBe("false")
    })
  })

  // -- Reset ----------------------------------------------------------------

  describe("reset", () => {
    it("returns to initial state", async () => {
      renderWithProvider()

      // Modify state
      await act(async () => {
        await userEvent.click(screen.getByText("goNext"))
        await userEvent.click(screen.getByText("setQueue"))
        await userEvent.click(screen.getByText("setDeep"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("2")
      expect(screen.getByTestId("preset").textContent).toBe("deep")
      expect(screen.getByTestId("queueLen").textContent).toBe("2")

      await act(async () => {
        await userEvent.click(screen.getByText("reset"))
      })
      expect(screen.getByTestId("currentStep").textContent).toBe("1")
      expect(screen.getByTestId("highestStep").textContent).toBe("1")
      expect(screen.getByTestId("preset").textContent).toBe("standard")
      expect(screen.getByTestId("queueLen").textContent).toBe("0")
      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("isMinimized").textContent).toBe("false")
    })
  })

  // -- Hook guard -----------------------------------------------------------

  describe("useIngestWizard outside provider", () => {
    it("throws an error when used outside IngestWizardProvider", () => {
      // Suppress React error boundary console output
      const spy = vi.spyOn(console, "error").mockImplementation(() => {})

      function BadConsumer() {
        useIngestWizard()
        return null
      }

      expect(() => render(<BadConsumer />)).toThrow(
        "useIngestWizard must be used within an IngestWizardProvider"
      )

      spy.mockRestore()
    })
  })
})
