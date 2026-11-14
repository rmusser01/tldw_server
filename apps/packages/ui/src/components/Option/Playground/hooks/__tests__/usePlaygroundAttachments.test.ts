import type React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { usePlaygroundAttachments } from "../usePlaygroundAttachments"

describe("usePlaygroundAttachments", () => {
  const handleFileUpload = vi.fn<(...args: [File]) => Promise<void>>()
  const notifyImageAttachmentDisabled = vi.fn()
  const setFieldValue = vi.fn()

  const createEvent = (
    file: File,
    value = `C:\\fakepath\\${file.name}`
  ): React.ChangeEvent<HTMLInputElement> =>
    ({
      target: {
        files: [file],
        value
      }
    }) as unknown as React.ChangeEvent<HTMLInputElement>

  beforeEach(() => {
    vi.stubGlobal("Image", class { src = ""; decode = () => Promise.resolve() })
    handleFileUpload.mockReset()
    handleFileUpload.mockResolvedValue(undefined)
    notifyImageAttachmentDisabled.mockReset()
    setFieldValue.mockReset()
  })

  afterEach(() => vi.unstubAllGlobals())

  it("shows the image error without accepting a corrupt file", async () => {
    vi.stubGlobal(
      "Image",
      class {
        src = ""
        decode = () => Promise.reject(new Error("Invalid image"))
      }
    )
    const notifyImageReadError = vi.fn()
    const { result } = renderHook(() =>
      usePlaygroundAttachments({
        chatMode: "chat",
        setFieldValue,
        handleFileUpload,
        notifyImageAttachmentDisabled,
        notifyImageReadError,
        captureImageOwnership: () => () => true
      })
    )
    const event = createEvent(
      new File(["invalid"], "broken.png", { type: "image/png" })
    )
    await act(async () => result.current.onFileInputChange(event))
    expect(notifyImageReadError).toHaveBeenCalledTimes(1)
    expect(setFieldValue).not.toHaveBeenCalled()
    expect(event.target.value).toBe("")
  })

  it("keeps the selected filename with successfully read image bytes", async () => {
    const file = new File(["png"], "diagram.png", { type: "image/png" })
    const event = createEvent(file)
    const { result } = renderHook(() =>
      usePlaygroundAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "chat",
        setFieldValue,
        handleFileUpload,
        notifyImageAttachmentDisabled
      })
    )
    await act(async () => result.current.onFileInputChange(event))
    expect(setFieldValue).toHaveBeenCalledWith(
      "image",
      expect.stringMatching(/^data:image\/png;base64,/)
    )
    expect(setFieldValue).toHaveBeenCalledWith("imageFilename", "diagram.png")
    expect(event.target.value).toBe("")
  })

  it("clears the native input value after a document upload", async () => {
    const file = new File(["notes"], "notes.txt", { type: "text/plain" })
    const event = createEvent(file)
    const { result } = renderHook(() =>
      usePlaygroundAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "chat",
        setFieldValue,
        handleFileUpload,
        notifyImageAttachmentDisabled
      })
    )

    await act(async () => {
      await result.current.onFileInputChange(event)
    })

    expect(handleFileUpload).toHaveBeenCalledTimes(1)
    expect(handleFileUpload).toHaveBeenCalledWith(file)
    expect(event.target.value).toBe("")
  })

  it("clears the native input value when RAG mode rejects an image attachment", async () => {
    const file = new File(["png"], "diagram.png", { type: "image/png" })
    const event = createEvent(file)
    const { result } = renderHook(() =>
      usePlaygroundAttachments({
        captureImageOwnership: () => () => true,
        chatMode: "rag",
        setFieldValue,
        handleFileUpload,
        notifyImageAttachmentDisabled
      })
    )

    await act(async () => {
      await result.current.onFileInputChange(event)
    })

    expect(notifyImageAttachmentDisabled).toHaveBeenCalledTimes(1)
    expect(handleFileUpload).not.toHaveBeenCalled()
    expect(setFieldValue).not.toHaveBeenCalled()
    expect(event.target.value).toBe("")
  })
})
