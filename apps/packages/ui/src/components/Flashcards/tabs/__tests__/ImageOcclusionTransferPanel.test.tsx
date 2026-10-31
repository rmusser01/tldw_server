import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { useDecksQuery } from "../../hooks"
import { ImageOcclusionTransferPanel } from "../ImageOcclusionTransferPanel"

const undoRequests = vi.hoisted(() => ({ get: vi.fn(), remove: vi.fn() }))
vi.mock("@/services/flashcards", async (original) => ({
  ...await original<typeof import("@/services/flashcards")>(),
  getFlashcard: undoRequests.get,
  deleteFlashcard: undoRequests.remove
}))

const snapshot = () => {
  const controller = new AbortController()
  const scope: ServicePromptSnapshot = {
    scopeKey: "owner-1", requestScope: { config: { serverUrl: "https://owner.test", authMode: "multi-user" }, userId: 1 },
    scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, capability: "unchecked", definitions: {}, release: () => controller.abort()
  }
  return { scope, controller }
}

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
    isLoading: false,
    isSuccess: true
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
    uploadFlashcardAssetMock.mockReset()
    undoRequests.get.mockResolvedValue({ version: 4 })
    undoRequests.remove.mockResolvedValue(undefined)
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

  it.each(["canvas", "upload", "bulk"])("stops the %s continuation after captured authority invalidation", async phase => {
    const { scope, controller } = snapshot()
    const onTransferAction = vi.fn()
    let finish!: (value: unknown) => void
    const deferred = new Promise<unknown>(resolve => { finish = resolve })
    if (phase === "canvas") generateImageOcclusionAssetsMock.mockReturnValueOnce(deferred)
    if (phase === "upload") { uploadFlashcardAssetMock.mockReset(); uploadFlashcardAssetMock.mockReturnValueOnce(deferred) }
    if (phase === "bulk") createBulkMutateAsync.mockReturnValueOnce(deferred)
    render(<ImageOcclusionTransferPanel generationScope={scope} onTransferAction={onTransferAction} />)
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    if (phase === "upload") await waitFor(() => expect(uploadFlashcardAssetMock).toHaveBeenCalledTimes(1))
    if (phase === "bulk") {
      fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))
      await waitFor(() => expect(createBulkMutateAsync).toHaveBeenCalledTimes(1))
      onTransferAction.mockClear()
      messageSpies.success.mockClear()
    }
    await act(async () => {
      controller.abort()
      finish(phase === "canvas" ? { source: { blob: new Blob(["source"]) }, regions: [] } : phase === "upload" ? { reference: "old", asset_uuid: "old" } : { items: [{ uuid: "old" }] })
    })
    expect(uploadFlashcardAssetMock).toHaveBeenCalledTimes(phase === "canvas" ? 0 : phase === "upload" ? 1 : 3)
    expect(showUndoNotificationMock).not.toHaveBeenCalled()
    expect(onTransferAction).not.toHaveBeenCalled()
    expect(messageSpies.success).not.toHaveBeenCalled()
    expect(messageSpies.error).not.toHaveBeenCalled()
  })

  it("blocks unresolved authority before asset work", () => {
    render(<ImageOcclusionTransferPanel generationScope={null} />)
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    expect(generateImageOcclusionAssetsMock).not.toHaveBeenCalled()
  })

  it.each(["current", "invalidated", "unmounted"])("binds retained Undo to its %s owner", async state => {
    const { scope, controller } = snapshot()
    const view = render(<ImageOcclusionTransferPanel generationScope={scope} />)
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))
    await waitFor(() => expect(showUndoNotificationMock).toHaveBeenCalledTimes(1))
    const options = { requestScope: scope.requestScope, signal: scope.scopeSignal }
    expect(uploadFlashcardAssetMock.mock.calls.every(call => call[1]?.requestScope === scope.requestScope)).toBe(true)
    expect(createBulkMutateAsync).toHaveBeenCalledWith({ cards: expect.any(Array), requestOptions: options })
    const undo = showUndoNotificationMock.mock.calls[0][0].onUndo
    if (state === "invalidated") controller.abort()
    if (state === "unmounted") view.unmount()
    if (state === "current") {
      await undo()
      expect(undoRequests.get).toHaveBeenCalledWith("card-1", options)
      expect(undoRequests.remove).toHaveBeenCalledWith("card-1", 4, options)
    } else {
      await expect(undo()).rejects.toMatchObject({ name: "AbortError" })
      expect(undoRequests.get).not.toHaveBeenCalled()
      expect(undoRequests.remove).not.toHaveBeenCalled()
    }
  })

  it("stops Undo between its scoped read and delete when authority changes", async () => {
    const { scope, controller } = snapshot()
    render(<ImageOcclusionTransferPanel generationScope={scope} />)
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))
    await waitFor(() => expect(showUndoNotificationMock).toHaveBeenCalledTimes(1))
    let finish!: (value: { version: number }) => void
    undoRequests.get.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const undo = showUndoNotificationMock.mock.calls[0][0].onUndo()
    const rejection = expect(undo).rejects.toMatchObject({ name: "AbortError" })
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    controller.abort()
    finish({ version: 4 })
    await rejection
    expect(undoRequests.remove).not.toHaveBeenCalled()
  })

  it("discards a late canvas continuation after unmount", async () => {
    const { scope } = snapshot()
    let finish!: (value: unknown) => void
    generateImageOcclusionAssetsMock.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const view = render(<ImageOcclusionTransferPanel generationScope={scope} />)
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    view.unmount()
    await act(async () => { finish({ source: { blob: new Blob(["source"]) }, regions: [] }) })
    expect(uploadFlashcardAssetMock).not.toHaveBeenCalled()
    expect(messageSpies.error).not.toHaveBeenCalled()
  })

  it.each(["current", "invalidated"])("handles a delayed new-deck acknowledgment for its %s owner", async state => {
    const { scope, controller } = snapshot()
    let finish!: (value: unknown) => void
    createDeckMutateAsync.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    render(<ImageOcclusionTransferPanel generationScope={scope} />)
    fireEvent.mouseDown(screen.getByTestId("flashcards-occlusion-deck"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
    fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    expect(createDeckMutateAsync).toHaveBeenCalledWith(expect.objectContaining({ requestOptions: { requestScope: scope.requestScope, signal: scope.scopeSignal } }))
    if (state === "invalidated") controller.abort()
    await act(async () => { finish({ id: 8, name: "New owned deck", version: 1 }) })
    if (state === "current") {
      await waitFor(() => expect(createBulkMutateAsync).toHaveBeenCalledWith({ cards: [expect.objectContaining({ deck_id: 8 })], requestOptions: { requestScope: scope.requestScope, signal: scope.scopeSignal } }))
      expect(screen.getByTestId("flashcards-occlusion-deck")).toHaveTextContent("New owned deck")
    } else {
      expect(createBulkMutateAsync).not.toHaveBeenCalled()
      expect(showUndoNotificationMock).not.toHaveBeenCalled()
      expect(messageSpies.error).not.toHaveBeenCalled()
    }
  })

  it("expires a new occlusion deck proof after a fresh successful empty catalogue", async () => {
    const { scope } = snapshot()
    const original = vi.mocked(useDecksQuery).getMockImplementation()!
    vi.mocked(useDecksQuery).mockReturnValue({ data: [], isSuccess: true, isLoading: false, dataUpdatedAt: 1 } as ReturnType<typeof useDecksQuery>)
    try {
      createDeckMutateAsync.mockResolvedValueOnce({ id: 8, name: "New owned deck", version: 1 })
      const view = render(<ImageOcclusionTransferPanel generationScope={scope} />)
      fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
      fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))
      fireEvent.click(await screen.findByTestId("flashcards-occlusion-save-button"))
      await waitFor(() => expect(createBulkMutateAsync).toHaveBeenCalled())
      expect(screen.getByTestId("flashcards-occlusion-deck")).toHaveTextContent("New owned deck")
      vi.mocked(useDecksQuery).mockReturnValue({ data: [], isSuccess: true, isLoading: false, dataUpdatedAt: 2 } as ReturnType<typeof useDecksQuery>)
      view.rerender(<ImageOcclusionTransferPanel generationScope={scope} />)
      await waitFor(() => expect(screen.getByTestId("flashcards-occlusion-deck")).not.toHaveTextContent("New owned deck"))
    } finally { vi.mocked(useDecksQuery).mockImplementation(original) }
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

  it("keeps drafts and reports warning when bulk create returns zero saved cards without details", async () => {
    createBulkMutateAsync.mockResolvedValueOnce({
      items: [],
      count: 0,
      total: 0
    })
    const onTransferAction = vi.fn()

    render(<ImageOcclusionTransferPanel onTransferAction={onTransferAction} />)

    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))

    await waitFor(() => {
      expect(uploadFlashcardAssetMock).toHaveBeenCalledTimes(3)
    })

    fireEvent.click(screen.getByTestId("flashcards-occlusion-save-button"))

    await waitFor(() => {
      expect(createBulkMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(
      screen.getByTestId("flashcards-occlusion-draft-front-occlusion-region-1")
    ).toHaveValue("Identify the occluded region.\n\n![Prompt](flashcard-asset://prompt-asset)")
    expect(
      screen.getByTestId("flashcards-occlusion-draft-back-occlusion-region-1")
    ).toHaveValue("Mitochondria\n\n![Answer](flashcard-asset://answer-asset)")
    const warningText = await screen.findByText(
      "No image occlusion cards were saved. Details are unavailable; review the drafts and retry."
    )
    expect(warningText.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(messageSpies.warning).toHaveBeenCalledWith(
      "No image occlusion cards were saved. Details are unavailable; review the drafts and retry."
    )
    expect(messageSpies.success).not.toHaveBeenCalledWith(
      "Saved 0 image occlusion cards."
    )
    expect(showUndoNotificationMock).not.toHaveBeenCalled()
    expect(onTransferAction).toHaveBeenLastCalledWith({
      area: "occlusion",
      status: "warning",
      message:
        "No image occlusion cards were saved. Details are unavailable; review the drafts and retry."
    })
  })

  it("renders generation errors through the design-system alert", async () => {
    generateImageOcclusionAssetsMock.mockRejectedValueOnce(
      new Error("Unable to prepare image occlusion assets.")
    )

    render(<ImageOcclusionTransferPanel />)

    fireEvent.click(screen.getByTestId("mock-occlusion-panel-load"))
    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))

    const errorText = await screen.findByText("Unable to prepare image occlusion assets.")

    expect(errorText.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(errorText).not.toHaveClass("font-medium")
    expect(messageSpies.error).toHaveBeenCalledWith("Unable to prepare image occlusion assets.")
  })

  it("renders validation warnings through a warning design-system alert", async () => {
    render(<ImageOcclusionTransferPanel />)

    fireEvent.click(screen.getByTestId("flashcards-occlusion-generate-button"))

    const warningText = await screen.findByText("Select a source image before generating drafts.")
    const warningAlert = warningText.closest('[data-ds-component="Alert"]')

    expect(warningAlert).not.toBeNull()
    expect(warningAlert).toHaveClass("border-warn/30")
    expect(warningText).not.toHaveClass("font-medium")
    expect(messageSpies.warning).toHaveBeenCalledWith(
      "Select a source image before generating drafts."
    )
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
