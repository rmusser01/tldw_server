import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ImageOcclusionTransferPanel } from "../ImageOcclusionTransferPanel"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

const {
  showUndoNotificationMock,
  generateImageOcclusionAssetsMock,
  uploadFlashcardAssetMock,
  createBulkMutateAsync,
  invalidateQueriesMock,
  createDeckMutateAsync
} = vi.hoisted(() => ({
  showUndoNotificationMock: vi.fn(),
  generateImageOcclusionAssetsMock: vi.fn(),
  uploadFlashcardAssetMock: vi.fn(),
  createBulkMutateAsync: vi.fn(),
  invalidateQueriesMock: vi.fn().mockResolvedValue(undefined),
  createDeckMutateAsync: vi.fn()
}))

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: invalidateQueriesMock
    })
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: showUndoNotificationMock
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("../../hooks", () => ({
  useCreateDeckMutation: vi.fn(() => ({
    mutateAsync: createDeckMutateAsync,
    isPending: false
  })),
  useCreateFlashcardsBulkMutation: vi.fn(() => ({
    mutateAsync: createBulkMutateAsync,
    isPending: false
  })),
  useDecksQuery: vi.fn(() => ({
    data: [
      {
        id: 1,
        name: "Biology",
        deleted: false,
        client_id: "test",
        version: 1
      }
    ],
    isLoading: false
  }))
}))

vi.mock("../ImageOcclusionPanel", () => ({
  ImageOcclusionPanel: ({
    onChange
  }: {
    onChange?: (state: {
      sourceFile: File | null
      sourceUrl: string | null
      selectedRegionId: string | null
      regions: Array<{
        id: string
        label: string
        x: number
        y: number
        width: number
        height: number
      }>
    }) => void
  }) => (
    <button
      type="button"
      data-testid="mock-occlusion-panel-load"
      onClick={() =>
        onChange?.({
          sourceFile: new File(["binary"], "diagram.png", { type: "image/png" }),
          sourceUrl: "blob:occlusion-source",
          selectedRegionId: "region-1",
          regions: [
            {
              id: "region-1",
              label: "Mitochondria",
              x: 0.1,
              y: 0.2,
              width: 0.3,
              height: 0.4
            }
          ]
        })
      }
    >
      Load occlusion state
    </button>
  )
}))

vi.mock("../../utils/image-occlusion-canvas", () => ({
  generateImageOcclusionAssets: generateImageOcclusionAssetsMock
}))

vi.mock("@/services/flashcard-assets", () => ({
  uploadFlashcardAsset: uploadFlashcardAssetMock
}))

describe("ImageOcclusionTransferPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    createDeckMutateAsync.mockReset()
    generateImageOcclusionAssetsMock.mockResolvedValue({
      source: {
        blob: new Blob(["source"], { type: "image/webp" }),
        width: 1200,
        height: 800,
        mimeType: "image/webp"
      },
      regions: [
        {
          regionId: "region-1",
          promptBlob: new Blob(["prompt"], { type: "image/webp" }),
          answerBlob: new Blob(["answer"], { type: "image/webp" }),
          width: 1200,
          height: 800,
          mimeType: "image/webp"
        }
      ]
    })
    uploadFlashcardAssetMock
      .mockResolvedValueOnce({
        asset_uuid: "source-asset",
        reference: "flashcard-asset://source-asset",
        markdown_snippet: "![Source](flashcard-asset://source-asset)",
        mime_type: "image/webp",
        byte_size: 1234
      })
      .mockResolvedValueOnce({
        asset_uuid: "prompt-asset",
        reference: "flashcard-asset://prompt-asset",
        markdown_snippet: "![Prompt](flashcard-asset://prompt-asset)",
        mime_type: "image/webp",
        byte_size: 111
      })
      .mockResolvedValueOnce({
        asset_uuid: "answer-asset",
        reference: "flashcard-asset://answer-asset",
        markdown_snippet: "![Answer](flashcard-asset://answer-asset)",
        mime_type: "image/webp",
        byte_size: 222
      })
    createBulkMutateAsync.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 1 }],
      count: 1,
      total: 1
    })
  })

  it("uploads source and derived images, creates editable drafts, and saves them via bulk create", async () => {
    render(<ImageOcclusionTransferPanel />)

    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.change(screen.getByTestId("flashcards-occlusion-tags"), {
      target: { value: "histology" }
    })

    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))

    await waitFor(() => {
      expect(uploadFlashcardAssetMock).toHaveBeenCalledTimes(3)
    })

    expect(
      await screen.findByTestId("flashcards-occlusion-draft-front-occlusion-region-1")
    ).toHaveValue("Identify the occluded region.\n\n![Prompt](flashcard-asset://prompt-asset)")
    expect(
      screen.getByTestId("flashcards-occlusion-draft-back-occlusion-region-1")
    ).toHaveValue("Mitochondria\n\n![Answer](flashcard-asset://answer-asset)")

    fireEvent.click(screen.getByTestId("flashcards-occlusion-save-button"))

    await waitFor(() => {
      expect(createBulkMutateAsync).toHaveBeenCalledWith([
        {
          deck_id: 1,
          front: "Identify the occluded region.\n\n![Prompt](flashcard-asset://prompt-asset)",
          back: "Mitochondria\n\n![Answer](flashcard-asset://answer-asset)",
          notes:
            "[image-occlusion]\nsource=flashcard-asset://source-asset\nregion=0.1000,0.2000,0.3000,0.4000\nlabel=Mitochondria",
          extra: undefined,
          tags: ["histology", "image-occlusion"],
          model_type: "basic",
          reverse: false,
          is_cloze: false,
          source_ref_type: "manual",
          source_ref_id: "image-occlusion:source-asset:0"
        }
      ])
    })
  })

  it("creates a new occlusion deck with scheduler settings from the selector flow", async () => {
    const fastAcquisitionSettings = {
      new_steps_minutes: [1, 5, 15],
      relearn_steps_minutes: [10],
      graduating_interval_days: 1,
      easy_interval_days: 3,
      easy_bonus: 1.15,
      interval_modifier: 0.9,
      max_interval_days: 3650,
      leech_threshold: 10,
      enable_fuzz: false
    }
    const fastAcquisitionEnvelope = {
      sm2_plus: fastAcquisitionSettings,
      fsrs: {
        target_retention: 0.9,
        maximum_interval_days: 36500,
        enable_fuzz: false
      }
    }
    createDeckMutateAsync.mockResolvedValue({
      id: 9,
      name: "Occlusion deck",
      description: null,
      deleted: false,
      client_id: "test",
      version: 1,
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(fastAcquisitionEnvelope),
      scheduler_settings: fastAcquisitionEnvelope
    })

    render(<ImageOcclusionTransferPanel />)

    fireEvent.mouseDown(screen.getByTestId("flashcards-occlusion-deck"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.change(screen.getByTestId("flashcards-occlusion-new-deck-name"), {
      target: { value: "Occlusion deck" }
    })
    fireEvent.mouseDown(screen.getByTestId("deck-study-defaults-review-prompt-side"))
    fireEvent.click(await screen.findByText("Back first"))
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))

    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))

    await waitFor(() => {
      expect(uploadFlashcardAssetMock).toHaveBeenCalledTimes(3)
    })

    fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))

    await waitFor(() =>
      expect(createDeckMutateAsync).toHaveBeenCalledWith({
        name: "Occlusion deck",
        review_prompt_side: "back",
        scheduler_type: "sm2_plus",
        scheduler_settings: fastAcquisitionEnvelope
      })
    )
  })
})
