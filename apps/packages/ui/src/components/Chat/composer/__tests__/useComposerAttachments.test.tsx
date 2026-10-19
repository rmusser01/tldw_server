import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useComposerAttachments } from "../hooks/useComposerAttachments"
import { toBase64 } from "~/libs/to-base64"

vi.mock("~/libs/to-base64", () => ({
  toBase64: vi.fn(async (file: File) => `data:${file.type};base64,ZmFrZQ==`),
}))

const decodeImage = vi.fn<() => Promise<void>>()

const makeImageFile = () =>
  new File(["fake"], "photo.png", { type: "image/png" })
const makeImageFileWithoutMime = () =>
  new File(["fake"], "photo.png", { type: "" })
const makeImageFileWithGenericMime = () =>
  new File(["fake"], "photo.jpg", { type: "application/octet-stream" })
const makeExtensionlessFile = () =>
  new File(["fake"], "png", { type: "" })
const makePdfFile = () =>
  new File(["fake"], "doc.pdf", { type: "application/pdf" })
const makeBlockedFile = () =>
  // application/zip is in otherUnsupportedTypes in the real module
  new File(["fake"], "archive.zip", { type: "application/zip" })
const makeBlockedFileWithImageExtension = () =>
  new File(["fake"], "photo.png", { type: "application/zip" })
const makeBlockedFileWithFormattedMime = () => {
  const file = makeBlockedFile()
  Object.defineProperty(file, "type", {
    value: " Application/ZIP ",
    configurable: true,
  })
  return file
}

const makeEvent = (file: File) =>
  ({
    target: { files: [file], value: "C:\\fakepath\\photo.png" },
  }) as unknown as React.ChangeEvent<HTMLInputElement>

describe("useComposerAttachments", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    decodeImage.mockReset().mockResolvedValue(undefined)
    vi.stubGlobal("Image", class { src = ""; decode = decodeImage })
  })

  afterEach(() => vi.unstubAllGlobals())

  it("rejects an undecodable image without accepting bytes or its filename", async () => {
    decodeImage.mockRejectedValueOnce(new Error("Invalid image"))
    const setImageField = vi.fn(),
      onImageAccepted = vi.fn(),
      onImageReadError = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField,
        onImageAccepted,
        onImageReadError,
        captureImageOwnership: () => () => true
      })
    )
    const event = makeEvent(makeImageFile())
    await act(async () => result.current.onInputChange(event))
    expect(setImageField).not.toHaveBeenCalled()
    expect(onImageAccepted).not.toHaveBeenCalled()
    expect(onImageReadError).toHaveBeenCalledTimes(1)
    expect(event.target.value).toBe("")
  })

  it("rejects an image whose owner changes during decoding", async () => {
    let finishDecode: () => void
    decodeImage.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishDecode = resolve
        })
    )
    let current = true
    const setImageField = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        chatMode: "normal",
        setImageField,
        captureImageOwnership: () => () => current
      })
    )
    let pending: Promise<void>
    await act(async () => {
      pending = result.current.onInputChange(makeImageFile())
      await Promise.resolve()
    })
    current = false
    await act(async () => {
      finishDecode!()
      await pending!
    })
    expect(setImageField).not.toHaveBeenCalled()
  })

  it.each(["older-first", "newer-first"])(
    "publishes only the latest selected image when reads finish %s",
    async (order) => {
      let finishOld: (value: string) => void, finishNew: (value: string) => void
      vi.mocked(toBase64)
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              finishOld = resolve
            })
        )
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              finishNew = resolve
            })
        )
      const setImageField = vi.fn(),
        onImageAccepted = vi.fn()
      const { result } = renderHook(() =>
        useComposerAttachments({
          chatMode: "normal",
          setImageField,
          onImageAccepted,
          captureImageOwnership: () => () => true
        })
      )
      const older = new File(["old"], "old.png", { type: "image/png" })
      const newer = new File(["new"], "new.png", { type: "image/png" })
      let oldRead: Promise<void>, newRead: Promise<void>
      act(() => {
        oldRead = result.current.onInputChange(older)
        newRead = result.current.onInputChange(newer)
      })
      const oldDone = async () => {
        finishOld!("data:image/png;base64,b2xk")
        await oldRead!
      }
      const newDone = async () => {
        finishNew!("data:image/png;base64,bmV3")
        await newRead!
      }
      await act(async () => {
        if (order === "older-first") {
          await oldDone()
          await newDone()
        } else {
          await newDone()
          await oldDone()
        }
      })
      expect(setImageField).toHaveBeenCalledTimes(1)
      expect(setImageField).toHaveBeenCalledWith("data:image/png;base64,bmV3")
      expect(onImageAccepted).toHaveBeenCalledTimes(1)
      expect(onImageAccepted).toHaveBeenCalledWith(newer)
    }
  )

  it("reads an image file to base64 and writes it to the form", async () => {
    const setImageField = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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

  it("accepts image attachments when the browser omits the MIME type", async () => {
    const setImageField = vi.fn()
    const onNonImageRejected = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "normal",
        setImageField,
        onNonImageRejected,
      })
    )

    await act(async () => {
      await result.current.onInputChange(makeImageFileWithoutMime())
    })

    expect(setImageField).toHaveBeenCalledWith(
      expect.stringMatching(/^data:image\/png;base64,/)
    )
    expect(onNonImageRejected).not.toHaveBeenCalled()
  })

  it("does not infer images from filenames without extensions", async () => {
    const setImageField = vi.fn()
    const onNonImageRejected = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "normal",
        setImageField,
        onNonImageRejected,
      })
    )

    const file = makeExtensionlessFile()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onNonImageRejected).toHaveBeenCalledWith(file)
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("accepts image attachments before rejecting generic binary MIME types", async () => {
    const setImageField = vi.fn()
    const onUnsupportedType = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "normal",
        setImageField,
        onUnsupportedType,
      })
    )

    await act(async () => {
      await result.current.onInputChange(makeImageFileWithGenericMime())
    })

    expect(setImageField).toHaveBeenCalledWith(
      expect.stringMatching(/^data:image\/jpeg;base64,/)
    )
    expect(onUnsupportedType).not.toHaveBeenCalled()
  })

  it("rejects blocked MIME types even when the filename has an image extension", async () => {
    const setImageField = vi.fn()
    const onUnsupportedType = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "normal",
        setImageField,
        onUnsupportedType,
      })
    )

    const file = makeBlockedFileWithImageExtension()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onUnsupportedType).toHaveBeenCalledWith(file)
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("normalizes MIME type formatting before checking unsupported types", async () => {
    const setImageField = vi.fn()
    const onUnsupportedType = vi.fn()
    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "normal",
        setImageField,
        onUnsupportedType,
      })
    )

    const file = makeBlockedFileWithFormattedMime()
    await act(async () => {
      await result.current.onInputChange(file)
    })

    expect(onUnsupportedType).toHaveBeenCalledWith(file)
    expect(setImageField).not.toHaveBeenCalled()
  })

  it("blocks image attachments in RAG mode when ragBlocksImages is true", async () => {
    const setImageField = vi.fn()
    const onImageBlockedInRagMode = vi.fn()

    const { result } = renderHook(() =>
      useComposerAttachments({
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
        captureImageOwnership: () => () => true,
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
