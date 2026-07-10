// @vitest-environment jsdom
import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import type { UploadedFile } from "@/db/dexie/types"
import { DocumentProcessingChoices } from "../DocumentProcessingChoices"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: { count?: number }) =>
      fallback?.replace("{{count}}", String(options?.count ?? "")) || _key
  })
}))

const makeFile = (
  filename: string,
  processingMode: UploadedFile["processingMode"] = "add_to_chat",
  overrides: Partial<UploadedFile> = {}
): UploadedFile => ({
  id: filename,
  filename,
  type: filename.endsWith(".pdf") ? "application/pdf" : "text/markdown",
  content: "",
  size: 1024,
  uploadedAt: 1,
  processed: false,
  processingMode,
  processingStatus: "pending",
  processingCapabilities: {
    add_to_chat: { available: true, status: "available" },
    ocr_pages: { available: true, status: "available" },
    ingest_to_library: { available: true, status: "available" }
  },
  ...overrides
})

describe("DocumentProcessingChoices", () => {
  it("applies a batch mode and keeps per-file overrides visible", async () => {
    const user = userEvent.setup()
    const onChangeFiles = vi.fn()

    render(
      <DocumentProcessingChoices
        files={[makeFile("a.pdf"), makeFile("b.pdf")]}
        onChangeFiles={onChangeFiles}
      />
    )

    expect(screen.getByRole("button", { name: /Add to chat/i })).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByRole("button", { name: /OCR pages/i })).toHaveAttribute(
      "aria-pressed",
      "false"
    )

    await user.click(screen.getByRole("button", { name: /OCR pages/i }))
    expect(onChangeFiles).toHaveBeenCalledWith([
      expect.objectContaining({ filename: "a.pdf", processingMode: "ocr_pages" }),
      expect.objectContaining({ filename: "b.pdf", processingMode: "ocr_pages" })
    ])

    const adjustButton = screen.getByRole("button", { name: /Adjust per file/i })
    expect(adjustButton).toHaveAttribute("aria-expanded", "false")
    await user.click(adjustButton)
    expect(adjustButton).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("a.pdf")).toBeInTheDocument()
    expect(screen.getByText("b.pdf")).toBeInTheDocument()
  })

  it("shows disabled OCR backend reasons", () => {
    render(
      <DocumentProcessingChoices
        files={[
          makeFile("notes.md", "add_to_chat", {
            processingCapabilities: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "OCR unavailable: server cannot render .MD pages"
              },
              ingest_to_library: { available: true, status: "available" }
            }
          })
        ]}
        onChangeFiles={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: /OCR pages/i })).toBeDisabled()
    expect(
      screen.getByText("OCR unavailable: server cannot render .MD pages")
    ).toBeInTheDocument()
  })

  it("aggregates unavailable mode reasons across files", () => {
    render(
      <DocumentProcessingChoices
        files={[
          makeFile("notes.md", "add_to_chat", {
            processingCapabilities: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "OCR unavailable for markdown"
              },
              ingest_to_library: { available: true, status: "available" }
            }
          }),
          makeFile("scan.pdf", "add_to_chat", {
            processingCapabilities: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: {
                available: false,
                status: "blocked",
                reason: "PDF exceeds page limit"
              },
              ingest_to_library: { available: true, status: "available" }
            }
          })
        ]}
        onChangeFiles={vi.fn()}
      />
    )

    expect(
      screen.getByText("OCR unavailable for markdown; PDF exceeds page limit")
    ).toBeInTheDocument()
  })

  it("makes ingest-to-library distinct from add-to-chat", async () => {
    const user = userEvent.setup()
    const onChangeFiles = vi.fn()

    render(
      <DocumentProcessingChoices
        files={[makeFile("archive.pdf")]}
        onChangeFiles={onChangeFiles}
      />
    )

    expect(screen.getByText("Chat only context")).toBeInTheDocument()
    expect(screen.getByText("Durable library source")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Ingest to library/i }))
    expect(onChangeFiles).toHaveBeenCalledWith([
      expect.objectContaining({
        filename: "archive.pdf",
        processingMode: "ingest_to_library"
      })
    ])
  })

  it("summarizes mixed processing states", () => {
    render(
      <DocumentProcessingChoices
        files={[
          makeFile("ready.pdf", "add_to_chat", { processingStatus: "ready" }),
          makeFile("pending.pdf", "ocr_pages"),
          makeFile("blocked.pdf", "ocr_pages", {
            processingStatus: "blocked",
            processingBlockedReason: "OCR unavailable"
          })
        ]}
        onChangeFiles={vi.fn()}
      />
    )

    const summary = screen.getByTestId("document-processing-summary")
    expect(within(summary).getByText("2 ready, 1 blocked")).toBeInTheDocument()
  })
})
