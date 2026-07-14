import React from "react"
import { describe, it, expect, vi } from "vitest"
import { render, screen, act } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  buildPlaylistIngestRunRequest,
  IngestWizardProvider,
  useIngestWizard,
} from "../IngestWizardContext"
import {
  configMatchesPreset,
  FIRST_SOURCE_QUICK_PRESET_CONFIG,
  resolvePresetMap,
  type PresetMap,
} from "../presets"
import type { WizardQueueItem } from "../types"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultOrOpts?: unknown) =>
      typeof defaultOrOpts === "string"
        ? defaultOrOpts
        : ((defaultOrOpts as Record<string, string>)?.defaultValue ?? key),
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
    applyPlaylistReviewRequired,
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
      <span data-testid="pendingRunRequest">{JSON.stringify(state.pendingRunRequest ?? null)}</span>
      <span data-testid="processingBlock">{JSON.stringify(state.processingBlock ?? null)}</span>
      <span data-testid="queueItems">{JSON.stringify(state.queueItems)}</span>
      <span data-testid="presetAnalysis">{String(state.presetConfig.common.perform_analysis)}</span>
      <span data-testid="presetChunkingMode">{state.presetConfig.common.chunking_mode || ""}</span>
      <span data-testid="presetAutoChunkingGoal">
        {state.presetConfig.common.auto_chunking_goal || ""}
      </span>
      <span data-testid="presetAutoChunkingUseLlm">
        {String(Boolean(state.presetConfig.common.auto_chunking_use_llm))}
      </span>
      <span data-testid="presetAudioLanguage">
        {state.presetConfig.typeDefaults.audio?.language || ""}
      </span>
      <span data-testid="presetProvider">
        {String(state.presetConfig.advancedValues?.api_name || "")}
      </span>
      <span data-testid="presetChunking">
        {String(state.presetConfig.common.perform_chunking)}
      </span>
      <span data-testid="presetDocumentOcr">
        {String(Boolean(state.presetConfig.typeDefaults.document?.ocr))}
      </span>

      <button onClick={goNext}>goNext</button>
      <button onClick={goBack}>goBack</button>
      <button onClick={() => goToStep(1)}>goToStep1</button>
      <button onClick={() => goToStep(3)}>goToStep3</button>
      <button onClick={() => goToStep(5)}>goToStep5</button>
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
      <button
        onClick={() =>
          setCustomOptions({
            typeDefaults: { document: { ocr: true } },
          })
        }
      >
        setDocumentOcr
      </button>
      <button
        onClick={() =>
          setCustomOptions({ advancedValues: { api_name: "openai" } })
        }
      >
        setProvider
      </button>
      <button
        onClick={() =>
          setCustomOptions({ advancedValues: { api_name: undefined } })
        }
      >
        clearProvider
      </button>
      <button onClick={skipToProcessing}>skipToProcessing</button>
      <button onClick={startProcessing}>startProcessing</button>
      <button onClick={cancelProcessing}>cancelProcessing</button>
      <button onClick={() => cancelItem("a")}>cancelItemA</button>
      <button
        onClick={() =>
          applyPlaylistReviewRequired([
            {
              occurrenceId: "occ-review-required",
              reason: "duplicate_target_changed",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "update_metadata_only"],
            },
          ])
        }
      >
        applyReviewRequired
      </button>
      <button
        onClick={() =>
          applyPlaylistReviewRequired([
            {
              occurrenceId: "occ-review-required",
              reason: "duplicate_no_longer_present",
              evidence: {
                kind: "none",
                existingMediaId: null,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: [],
            },
          ])
        }
      >
        applyNoLongerDuplicate
      </button>
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

function renderWithInitialState(
  initialState: Parameters<typeof IngestWizardProvider>[0]["initialState"]
) {
  return render(
    <IngestWizardProvider initialState={initialState}>
      <TestHarness />
    </IngestWizardProvider>
  )
}

function renderWithPresetMap(presetMap: PresetMap) {
  return render(
    <IngestWizardProvider presetMap={presetMap}>
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
    it("uses the captured preset map for the default preset", () => {
      const presetMap = resolvePresetMap({
        standard: {
          ...resolvePresetMap().standard,
          advancedValues: { api_name: "openai" },
        },
      })

      renderWithPresetMap(presetMap)

      expect(screen.getByTestId("presetProvider").textContent).toBe("openai")
    })

    it("uses the captured preset map when switching named presets", async () => {
      const presetMap = resolvePresetMap({
        deep: {
          ...resolvePresetMap().deep,
          advancedValues: { api_name: "anthropic" },
        },
      })
      renderWithPresetMap(presetMap)

      await userEvent.click(screen.getByText("setDeep"))

      expect(screen.getByTestId("presetProvider").textContent).toBe("anthropic")
    })

    it("preserves first-source chunking when another option changes", async () => {
      renderWithInitialState({
        selectedPreset: "quick",
        customBasePreset: "quick",
        presetConfig: FIRST_SOURCE_QUICK_PRESET_CONFIG,
        customOptions: {},
      })

      await userEvent.click(screen.getByText("setDocumentOcr"))

      expect(screen.getByTestId("presetChunking").textContent).toBe("true")
    })

    it("preserves the full first-source config when selecting Custom", async () => {
      renderWithInitialState({
        selectedPreset: "quick",
        customBasePreset: "quick",
        presetConfig: FIRST_SOURCE_QUICK_PRESET_CONFIG,
        customOptions: {},
      })

      await userEvent.click(screen.getByText("setCustomPreset"))

      expect(screen.getByTestId("presetChunking").textContent).toBe("true")
    })

    it("does not resurrect a cleared provider after another option edit", async () => {
      renderWithInitialState({
        selectedPreset: "custom",
        customBasePreset: "standard",
        presetConfig: {
          ...resolvePresetMap().standard,
          advancedValues: { api_name: "openai", temperature: 0.2 },
        },
        customOptions: {
          advancedValues: { api_name: "openai", temperature: 0.2 },
        },
      })

      await userEvent.click(screen.getByText("clearProvider"))
      await userEvent.click(screen.getByText("setAudioLanguage"))

      expect(screen.getByTestId("presetProvider").textContent).toBe("")
    })

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
    it("builds a playlist run request from occurrence authority and explicit review edits", async () => {
      renderWithInitialState({
        currentStep: 3,
        highestStep: 3,
        queueItems: [
          {
            id: "occ-playlist-1",
            url: "https://cached.example.invalid/watch?v=never-authoritative",
            sourceRef: {
              kind: "materialized_playlist_item",
              materializationId: "opaque-owner-bound-materialization",
              occurrenceId: "occ-playlist-1",
            },
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: {
              title: "Playlist row",
              ordinal: 4,
              duplicateStatus: "duplicate_existing",
              materializationExpiresAt: "2099-07-20T00:00:00Z",
            },
            playlistReview: {
              selected: true,
              duplicatePolicy: "update_metadata_only",
              metadataPatch: {
                title: "Edited title",
                author: "Edited author",
                keywordsAdd: ["Research", "research", "video"],
              },
              editedFields: ["title", "keywordsAdd"],
            },
          },
        ] as WizardQueueItem[],
      })

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("status").textContent).toBe("running")
      expect(JSON.parse(screen.getByTestId("pendingRunRequest").textContent || "null")).toEqual({
        inputs: [
          {
            inputKind: "materialized_playlist_item",
            occurrenceId: "occ-playlist-1",
            materializationId: "opaque-owner-bound-materialization",
          },
        ],
        reviewOverrides: {
          "occ-playlist-1": {
            duplicatePolicy: "update_metadata_only",
            metadataPatch: {
              title: "Edited title",
              keywordsAdd: ["Research", "video"],
            },
          },
        },
      })
      expect(screen.getByTestId("pendingRunRequest")).not.toHaveTextContent("never-authoritative")
    })

    it.each([
      ["blank author", { author: "   " }, ["author"]],
      ["overlong title", { title: "x".repeat(501) }, ["title"]],
      [
        "too many keywords",
        {
          keywordsAdd: Array.from({ length: 101 }, (_, index) => `tag-${index}`),
        },
        ["keywordsAdd"],
      ],
      ["overlong keyword", { keywordsAdd: ["x".repeat(129)] }, ["keywordsAdd"]],
    ] as const)(
      "blocks an invalid explicit metadata patch: %s",
      async (_case, metadataPatch, editedFields) => {
        renderWithInitialState({
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-invalid-patch",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "materialization-invalid-patch",
                occurrenceId: "occ-invalid-patch",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                duplicateStatus: "duplicate_existing",
                materializationExpiresAt: "2099-07-20T00:00:00Z",
              },
              playlistReview: {
                selected: true,
                duplicatePolicy: "update_metadata_only",
                metadataPatch,
                editedFields: [...editedFields],
              },
            },
          ] as WizardQueueItem[],
        })

        await act(async () => {
          await userEvent.click(screen.getByText("startProcessing"))
        })

        expect(screen.getByTestId("status").textContent).toBe("idle")
        expect(screen.getByTestId("pendingRunRequest").textContent).toBe("null")
        expect(JSON.parse(screen.getByTestId("processingBlock").textContent || "null")).toEqual({
          code: "review_required",
          occurrenceIds: ["occ-invalid-patch"],
        })
      }
    )

    it("blocks an expired playlist materialization instead of falling back to its cached URL", async () => {
      renderWithInitialState({
        currentStep: 3,
        highestStep: 3,
        queueItems: [
          {
            id: "occ-expired",
            url: "https://cached.example.invalid/watch?v=expired",
            sourceRef: {
              kind: "materialized_playlist_item",
              materializationId: "expired-materialization",
              occurrenceId: "occ-expired",
            },
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: {
              title: "Expired row",
              ordinal: 1,
              materializationExpiresAt: "2020-01-01T00:00:00Z",
            },
          },
        ] as WizardQueueItem[],
      })

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("pendingRunRequest").textContent).toBe("null")
      expect(JSON.parse(screen.getByTestId("processingBlock").textContent || "null")).toEqual({
        code: "materialization_expired",
        occurrenceIds: ["occ-expired"],
      })
    })

    it.each([
      ["missing source authority", undefined],
      [
        "mismatched occurrence authority",
        {
          kind: "materialized_playlist_item" as const,
          materializationId: "materialization-mismatch",
          occurrenceId: "different-occurrence",
        },
      ],
      [
        "empty materialization authority",
        {
          kind: "materialized_playlist_item" as const,
          materializationId: "   ",
          occurrenceId: "occ-invalid-authority",
        },
      ],
    ])("rejects a cached playlist URL with %s", async (_case, sourceRef) => {
      renderWithInitialState({
        currentStep: 3,
        highestStep: 3,
        queueItems: [
          {
            id: "occ-invalid-authority",
            url: "https://cached.example.invalid/watch?v=must-not-submit",
            sourceRef,
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: {
              title: "Persisted materialized row",
              materializationExpiresAt: "2099-07-20T00:00:00Z",
            },
          },
        ] as WizardQueueItem[],
      })

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("pendingRunRequest").textContent).toBe("null")
      expect(JSON.parse(screen.getByTestId("processingBlock").textContent || "null")).toEqual({
        code: "invalid_run_request",
        occurrenceIds: ["occ-invalid-authority"],
      })
      expect(screen.getByTestId("pendingRunRequest")).not.toHaveTextContent("must-not-submit")
    })

    it.each([
      ["source URL", { sourceUrl: "https://cached.example.invalid/source-cue" }],
      ["playlist ID", { playlistId: "playlist-cue" }],
      ["playlist title", { playlistTitle: "Playlist cue" }],
      ["ordinal", { ordinal: 2 }],
      ["channel or uploader", { channelOrUploader: "Channel cue" }],
      ["duration", { durationSeconds: 0 }],
      ["normalized source ID", { normalizedSourceId: "video-cue" }],
      ["materialization expiry", { materializationExpiresAt: "2099-07-20T00:00:00Z" }],
    ] as Array<[string, NonNullable<WizardQueueItem["playlist"]>]>) (
      "rejects a cached materialized URL when only the %s cue survives",
      (_case, playlist) => {
        const result = buildPlaylistIngestRunRequest([
          {
            id: "orphaned-run-cues",
            kind: "url",
            url: "https://cached.example.invalid/must-not-submit-cues",
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist,
          },
        ])

        expect(result).toEqual({
          request: null,
          block: {
            code: "invalid_run_request",
            occurrenceIds: ["orphaned-run-cues"],
          },
        })
      }
    )

    it("keeps duplicate-review-only direct URL metadata eligible", () => {
      const result = buildPlaylistIngestRunRequest([
        {
          id: "direct-duplicate-review",
          kind: "url",
          url: "https://example.com/direct-duplicate-review",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "direct-duplicate-review",
            url: "https://example.com/direct-duplicate-review",
          },
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            title: "Recovered duplicate title",
            duplicateStatus: "duplicate_existing",
          },
          playlistReview: {
            selected: true,
            duplicatePolicy: "skip",
          },
        },
      ])

      expect(result.block).toBeNull()
      expect(result.request?.inputs[0]).toMatchObject({
        inputKind: "direct_url",
        occurrenceId: "direct-duplicate-review",
        url: "https://example.com/direct-duplicate-review",
      })
    })

    it.each([
      [
        "direct URL",
        {
          kind: "direct_url" as const,
          occurrenceId: "different-direct-id",
          url: "https://example.com/direct",
        },
      ],
      [
        "file stub",
        {
          kind: "file_stub" as const,
          occurrenceId: "different-file-id",
        },
      ],
    ])("rejects mismatched %s occurrence authority", (_case, sourceRef) => {
      const id = sourceRef.kind === "direct_url" ? "direct-id" : "file-id"
      const result = buildPlaylistIngestRunRequest([
        {
          id,
          sourceRef,
          ...(sourceRef.kind === "direct_url"
            ? { url: sourceRef.url }
            : { fileName: "restored.txt" }),
          detectedType: sourceRef.kind === "direct_url" ? "web" : "document",
          icon: "File",
          fileSize: 0,
          validation: { valid: true },
        },
      ])

      expect(result.request).toBeNull()
      expect(result.block).toEqual({
        code: "invalid_run_request",
        occurrenceIds: [id],
      })
    })

    it("rejects duplicate canonical occurrence identifiers before serialization", () => {
      const queueItems = [1, 2].map(() => ({
        id: "duplicate-occurrence",
        sourceRef: {
          kind: "direct_url" as const,
          occurrenceId: "duplicate-occurrence",
          url: "https://example.com/duplicate",
        },
        url: "https://example.com/duplicate",
        detectedType: "web" as const,
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true },
      }))

      expect(buildPlaylistIngestRunRequest(queueItems)).toEqual({
        request: null,
        block: {
          code: "invalid_run_request",
          occurrenceIds: ["duplicate-occurrence"],
        },
      })
    })

    it.each([
      [
        "oversized direct URL",
        {
          id: "oversized-url",
          sourceRef: {
            kind: "direct_url" as const,
            occurrenceId: "oversized-url",
            url: "x".repeat(8193),
          },
          url: "display-only",
          detectedType: "web" as const,
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
      [
        "oversized file name",
        {
          id: "oversized-name",
          sourceRef: { kind: "file_stub" as const, occurrenceId: "oversized-name" },
          fileName: "x".repeat(256),
          detectedType: "document" as const,
          icon: "File",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
      [
        "oversized content type",
        {
          id: "oversized-content-type",
          sourceRef: {
            kind: "file_stub" as const,
            occurrenceId: "oversized-content-type",
          },
          fileName: "restored.txt",
          mimeType: "x".repeat(256),
          detectedType: "document" as const,
          icon: "File",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
      [
        "oversized file",
        {
          id: "oversized-file",
          sourceRef: { kind: "file_stub" as const, occurrenceId: "oversized-file" },
          fileName: "restored.txt",
          detectedType: "document" as const,
          icon: "File",
          fileSize: 10 * 1024 ** 4 + 1,
          validation: { valid: true },
        },
      ],
    ])("rejects backend-invalid %s input bounds", (_case, item) => {
      expect(buildPlaylistIngestRunRequest([item])).toEqual({
        request: null,
        block: { code: "invalid_run_request", occurrenceIds: [item.id] },
      })
    })

    it("accepts direct and file inputs at backend contract boundaries", () => {
      const directUrl = "u".repeat(8192)
      const result = buildPlaylistIngestRunRequest([
        {
          id: "direct-boundary",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "direct-boundary",
            url: directUrl,
          },
          playlist: { title: "t".repeat(2000) },
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
        {
          id: "file-boundary",
          sourceRef: { kind: "file_stub", occurrenceId: "file-boundary" },
          fileName: "n".repeat(255),
          mimeType: "c".repeat(255),
          detectedType: "document",
          icon: "File",
          fileSize: 10 * 1024 ** 4,
          validation: { valid: true },
        },
      ])

      expect(result.block).toBeNull()
      expect(result.request?.inputs).toHaveLength(2)
    })

    it("blocks more than 500 selected run inputs while allowing 500 selected", () => {
      const queueItems: WizardQueueItem[] = Array.from({ length: 501 }, (_, index) => {
        const id = `bounded-run-${index + 1}`
        const url = `https://example.com/${index + 1}`
        return {
          id,
          sourceRef: { kind: "direct_url" as const, occurrenceId: id, url },
          url,
          detectedType: "web" as const,
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        }
      })

      expect(buildPlaylistIngestRunRequest(queueItems)).toEqual({
        request: null,
        block: {
          code: "invalid_run_request",
          occurrenceIds: ["bounded-run-501"],
        },
      })

      queueItems[500].playlistReview = { selected: false }
      const bounded = buildPlaylistIngestRunRequest(queueItems)
      expect(bounded.block).toBeNull()
      expect(bounded.request?.inputs).toHaveLength(500)
    })

    it("validates the exact direct URL display title selected by serialization", () => {
      const result = buildPlaylistIngestRunRequest([
        {
          id: "direct-display-title",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "direct-display-title",
            url: "https://example.com/display-title",
          },
          fileName: "x".repeat(2001),
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ])

      expect(result).toEqual({
        request: null,
        block: {
          code: "invalid_run_request",
          occurrenceIds: ["direct-display-title"],
        },
      })

      const blankTitle = buildPlaylistIngestRunRequest([
        {
          id: "blank-direct-display-title",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "blank-direct-display-title",
            url: "https://example.com/blank-display-title",
          },
          fileName: "   ",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ])
      expect(blankTitle.request?.inputs[0]).toMatchObject({
        displayMetadata: { title: null },
      })
    })

    it("rejects seeded in-run policies outside the initial safe actions unless the server allows them", () => {
      const duplicate: WizardQueueItem = {
        id: "seeded-in-run-policy",
        sourceRef: {
          kind: "direct_url",
          occurrenceId: "seeded-in-run-policy",
          url: "https://example.com/seeded-in-run-policy",
        },
        url: "https://example.com/seeded-in-run-policy",
        detectedType: "web",
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true },
        playlist: { duplicateStatus: "duplicate_in_batch" },
        playlistReview: {
          selected: true,
          duplicatePolicy: "include_existing",
        },
      }

      expect(buildPlaylistIngestRunRequest([duplicate])).toEqual({
        request: null,
        block: {
          code: "review_required",
          occurrenceIds: ["seeded-in-run-policy"],
        },
      })

      duplicate.playlistReview = {
        ...duplicate.playlistReview,
        allowedDuplicatePolicies: ["skip", "include_existing", "overwrite"],
      }
      const serverAllowed = buildPlaylistIngestRunRequest([duplicate])
      expect(serverAllowed.block).toBeNull()
      expect(serverAllowed.request?.reviewOverrides?.["seeded-in-run-policy"]).toEqual({
        duplicatePolicy: "include_existing",
      })
    })

    it("merges structured review-required recovery and returns to Review without marking rows submitted", async () => {
      renderWithInitialState({
        currentStep: 4,
        highestStep: 4,
        queueItems: [
          {
            id: "occ-review-required",
            sourceRef: {
              kind: "materialized_playlist_item",
              materializationId: "materialization-review-required",
              occurrenceId: "occ-review-required",
            },
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: {
              duplicateStatus: "duplicate_in_batch",
              materializationExpiresAt: "2099-07-20T00:00:00Z",
            },
            playlistReview: {
              selected: true,
              duplicatePolicy: "skip",
              metadataPatch: { title: "Keep this edit" },
              editedFields: ["title"],
            },
          },
        ] as WizardQueueItem[],
        pendingRunRequest: {
          inputs: [
            {
              inputKind: "materialized_playlist_item",
              materializationId: "materialization-review-required",
              occurrenceId: "occ-review-required",
            },
          ],
        },
        processingState: {
          status: "running",
          perItemProgress: [
            {
              id: "occ-review-required",
              status: "queued",
              progressPercent: 0,
              currentStage: "",
              estimatedRemaining: 0,
            },
          ],
          elapsed: 0,
          estimatedRemaining: 0,
        },
      })

      await act(async () => {
        await userEvent.click(screen.getByText("applyReviewRequired"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("3")
      expect(screen.getByTestId("highestStep").textContent).toBe("4")
      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("progressIds").textContent).toBe("")
      expect(screen.getByTestId("pendingRunRequest").textContent).toBe("null")
      expect(JSON.parse(screen.getByTestId("queueItems").textContent || "[]")[0]).toMatchObject({
        playlist: { duplicateStatus: "duplicate_existing" },
        playlistReview: {
          selected: true,
          duplicateEvidence: {
            kind: "library",
            existingMediaId: 42,
            duplicateOfOccurrenceId: null,
          },
          allowedDuplicatePolicies: ["skip", "update_metadata_only"],
          reviewReason: "duplicate_target_changed",
          metadataPatch: { title: "Keep this edit" },
          editedFields: ["title"],
        },
      })
      expect(
        JSON.parse(screen.getByTestId("queueItems").textContent || "[]")[0].playlistReview
          .duplicatePolicy
      ).toBeUndefined()
      expect(JSON.parse(screen.getByTestId("processingBlock").textContent || "null")).toEqual({
        code: "review_required",
        occurrenceIds: ["occ-review-required"],
      })
    })

    it("clears the Review block when fresh evidence says the row is no longer duplicate", async () => {
      renderWithInitialState({
        currentStep: 4,
        highestStep: 4,
        queueItems: [
          {
            id: "occ-review-required",
            sourceRef: {
              kind: "materialized_playlist_item",
              materializationId: "materialization-no-longer-duplicate",
              occurrenceId: "occ-review-required",
            },
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: {
              duplicateStatus: "duplicate_existing",
              materializationExpiresAt: "2099-07-20T00:00:00Z",
            },
            playlistReview: {
              selected: true,
              duplicatePolicy: "overwrite",
              duplicateEvidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
            },
          },
        ],
        processingBlock: {
          code: "review_required",
          occurrenceIds: ["occ-review-required"],
        },
      })

      await act(async () => {
        await userEvent.click(screen.getByText("applyNoLongerDuplicate"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("3")
      expect(JSON.parse(screen.getByTestId("queueItems").textContent || "[]")[0]).toMatchObject({
        playlist: { duplicateStatus: "new" },
        playlistReview: {
          selected: true,
          duplicateEvidence: {
            kind: "none",
            existingMediaId: null,
            duplicateOfOccurrenceId: null,
          },
          allowedDuplicatePolicies: [],
          reviewReason: "duplicate_no_longer_present",
        },
      })
      expect(
        JSON.parse(screen.getByTestId("queueItems").textContent || "[]")[0].playlistReview
          .duplicatePolicy
      ).toBeUndefined()
      expect(screen.getByTestId("processingBlock").textContent).toBe("null")

      await act(async () => {
        await userEvent.click(screen.getByText("startProcessing"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("4")
      expect(screen.getByTestId("status").textContent).toBe("running")
      expect(JSON.parse(screen.getByTestId("pendingRunRequest").textContent || "null")).toEqual({
        inputs: [
          {
            inputKind: "materialized_playlist_item",
            materializationId: "materialization-no-longer-duplicate",
            occurrenceId: "occ-review-required",
          },
        ],
      })
    })

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

    it("routes an invalid-only quick process to visible review feedback", async () => {
      renderWithProvider()

      await act(async () => {
        await userEvent.click(screen.getByText("setInvalidQueue"))
      })

      await act(async () => {
        await userEvent.click(screen.getByText("skipToProcessing"))
      })

      expect(screen.getByTestId("currentStep").textContent).toBe("3")
      expect(screen.getByTestId("status").textContent).toBe("idle")
      expect(screen.getByTestId("progressIds").textContent).toBe("")
      expect(JSON.parse(screen.getByTestId("processingBlock").textContent || "null")).toEqual({
        code: "invalid_run_request",
        occurrenceIds: [],
      })
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
