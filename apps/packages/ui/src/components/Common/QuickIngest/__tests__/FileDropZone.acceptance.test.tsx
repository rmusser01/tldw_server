import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
      interpolation?: Record<string, unknown>
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolation?.[token] ?? "")
        )
      }
      return String(defaultValueOrOptions?.defaultValue || "")
    }
  })
}))

import { FileDropZone } from "../QueueTab/FileDropZone"

describe("FileDropZone acceptance", () => {
  it.each([
    ["scan.pdf", "application/pdf", 250 * 1024 ** 2, "250 MB"],
    ["sample.epub", "application/epub+zip", 250 * 1024 ** 2, "250 MB"],
    ["notes.txt", "text/plain", 250 * 1024 ** 2, "250 MB"],
    ["sample.ogg", "audio/ogg", 1024 ** 3, "1 GB"],
    ["sample.mkv", "", 1024 ** 3, "1 GB"]
  ] as const)("accepts %s at the limit and rejects one more byte", async (name, type, limit, label) => {
    const onFilesAdded = vi.fn()
    const onFilesRejected = vi.fn()
    render(<FileDropZone onFilesAdded={onFilesAdded} onFilesRejected={onFilesRejected} />)
    const atLimit = new File([], name, { type })
    const overLimit = new File([], name, { type })
    Object.defineProperty(atLimit, "size", { value: limit })
    Object.defineProperty(overLimit, "size", { value: limit + 1 })

    await userEvent.setup().upload(screen.getByTestId("qi-file-input"), [atLimit, overLimit])

    expect(onFilesAdded).toHaveBeenCalledExactlyOnceWith([atLimit])
    expect(onFilesRejected).toHaveBeenCalledExactlyOnceWith([`${name} exceeds ${label} quick-ingest limit`])
  })

  it("only autofocuses once when disabled state clears", () => {
    const focusSpy = vi
      .spyOn(HTMLElement.prototype, "focus")
      .mockImplementation(() => undefined)
    const { rerender } = render(
      <FileDropZone
        autoFocus
        running
        onFilesAdded={vi.fn()}
        onFilesRejected={vi.fn()}
      />
    )

    rerender(
      <FileDropZone
        autoFocus
        running={false}
        onFilesAdded={vi.fn()}
        onFilesRejected={vi.fn()}
      />
    )
    rerender(
      <FileDropZone
        autoFocus
        running
        onFilesAdded={vi.fn()}
        onFilesRejected={vi.fn()}
      />
    )
    rerender(
      <FileDropZone
        autoFocus
        running={false}
        onFilesAdded={vi.fn()}
        onFilesRejected={vi.fn()}
      />
    )

    expect(focusSpy).toHaveBeenCalledTimes(1)
    focusSpy.mockRestore()
  })

  it("accepts mkv uploads even when the browser does not provide a MIME type", async () => {
    const user = userEvent.setup()
    const onFilesAdded = vi.fn()
    const onFilesRejected = vi.fn()

    render(
      <FileDropZone
        onFilesAdded={onFilesAdded}
        onFilesRejected={onFilesRejected}
      />
    )

    const input = screen.getByTestId("qi-file-input")
    const mkvFile = new File(["video-bytes"], "sample-video.mkv")

    await user.upload(input, mkvFile)

    expect(onFilesAdded).toHaveBeenCalledTimes(1)
    expect(onFilesAdded.mock.calls[0]?.[0]).toHaveLength(1)
    expect(onFilesAdded.mock.calls[0]?.[0]?.[0]?.name).toBe("sample-video.mkv")
    expect(onFilesRejected).not.toHaveBeenCalled()
  })
})
