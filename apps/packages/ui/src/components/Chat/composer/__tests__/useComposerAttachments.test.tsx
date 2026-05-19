import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useComposerAttachments } from "../hooks/useComposerAttachments"

vi.mock("~/libs/to-base64", () => ({
  toBase64: vi.fn(async (file: File) => `data:${file.type};base64,ZmFrZQ==`),
}))

const makeImageFile = () =>
  new File(["fake"], "photo.png", { type: "image/png" })
const makePdfFile = () =>
  new File(["fake"], "doc.pdf", { type: "application/pdf" })
const makeBlockedFile = () =>
  // application/zip is in otherUnsupportedTypes in the real module
  new File(["fake"], "archive.zip", { type: "application/zip" })

const makeEvent = (file: File) =>
  ({
    target: { files: [file], value: "C:\\fakepath\\photo.png" },
  }) as unknown as React.ChangeEvent<HTMLInputElement>

describe("useComposerAttachments", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("reads an image file to base64 and writes it to the form", async () => {
    const setImageField = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField,
      })
    )

    await act(async () => {
      await result.current.onInputChange(makeImageFile())
    })

    expect(setImageField).toHaveBeenCalledWith(
      expect.stringMatching(/^data:image\/png;base64,/)
    )
  })

  it("fires onImageAccepted when an image is attached", async () => {
    const onImageAccepted = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField: vi.fn(),
        onImageAccepted,
      })
    )

    const file = makeImageFile()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onImageAccepted).toHaveBeenCalledWith(file)
  })

  it("blocks image attachments in RAG mode when ragBlocksImages is true", async () => {
    const setImageField = vi.fn()
    const onImageBlockedInRagMode = vi.fn()

    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "rag",
        setImageField,
        ragBlocksImages: true,
        onImageBlockedInRagMode,
      })
    )

    await act(async () => {
      await result.current.onInputChange(makeImageFile())
    })

    expect(onImageBlockedInRagMode).toHaveBeenCalledOnce()
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("forwards non-image files to onDocumentUpload when provided", async () => {
    const onDocumentUpload = vi.fn(async () => undefined)
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField: vi.fn(),
        onDocumentUpload,
      })
    )

    const file = makePdfFile()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onDocumentUpload).toHaveBeenCalledWith(file)
  })

  it("calls onNonImageRejected for non-images when onDocumentUpload is omitted (image-only surface)", async () => {
    const onNonImageRejected = vi.fn()
    const setImageField = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField,
        onNonImageRejected,
      })
    )

    const file = makePdfFile()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onNonImageRejected).toHaveBeenCalledWith(file)
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("calls onUnsupportedType for blocked file types", async () => {
    const onUnsupportedType = vi.fn()
    const setImageField = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField,
        onUnsupportedType,
      })
    )

    const file = makeBlockedFile()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onUnsupportedType).toHaveBeenCalledWith(file)
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("clears the input value after a ChangeEvent-triggered upload", async () => {
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField: vi.fn(),
      })
    )

    const event = makeEvent(makeImageFile())
    await act(async () => {
      await result.current.onFileInputChange(event)
    })

    expect(event.target.value).toBe("")
  })

  it("handleImageUpload clicks the image input ref", () => {
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField: vi.fn(),
      })
    )

    const click = vi.fn()
    Object.defineProperty(result.current.inputRef, "current", {
      value: { click },
      writable: true,
    })

    result.current.handleImageUpload()
    expect(click).toHaveBeenCalledOnce()
  })

  it("handleDocumentUpload clicks the file input ref", () => {
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField: vi.fn(),
      })
    )

    const click = vi.fn()
    Object.defineProperty(result.current.fileInputRef, "current", {
      value: { click },
      writable: true,
    })

    result.current.handleDocumentUpload()
    expect(click).toHaveBeenCalledOnce()
  })
})
