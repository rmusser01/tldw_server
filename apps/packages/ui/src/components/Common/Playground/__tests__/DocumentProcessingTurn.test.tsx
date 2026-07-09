// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { DocumentProcessingTurnMetadata } from "@/db/dexie/types"
import { DocumentProcessingTurn } from "../DocumentProcessingTurn"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      fallback?.replace("{{count}}", String(options?.count ?? "")) || _key
  })
}))

const makeMetadata = (
  overrides: Partial<DocumentProcessingTurnMetadata> = {}
): DocumentProcessingTurnMetadata => ({
  status: "processing",
  files: [
    {
      id: "file-1",
      filename: "scan.pdf",
      mode: "ocr_pages",
      status: "processing",
      summary: "Rendering pages"
    }
  ],
  ...overrides
})

describe("DocumentProcessingTurn", () => {
  it("shows the current document processing status and file count", () => {
    render(<DocumentProcessingTurn metadata={makeMetadata()} />)

    expect(screen.getByText("Processing documents")).toBeInTheDocument()
    expect(screen.getByText("1 file")).toBeInTheDocument()
    expect(screen.getByText("scan.pdf")).toBeInTheDocument()
    expect(screen.getByText("OCR pages")).toBeInTheDocument()
  })

  it("surfaces blocked and failed file details", () => {
    render(
      <DocumentProcessingTurn
        metadata={makeMetadata({
          status: "blocked",
          files: [
            {
              id: "blocked",
              filename: "too-large.pdf",
              mode: "add_to_chat",
              status: "blocked",
              summary: "Blocked by 24k token limit"
            },
            {
              id: "failed",
              filename: "archive.pdf",
              mode: "ingest_to_library",
              status: "failed",
              error: "Ingest failed"
            }
          ],
          recoveryActions: ["switch_to_ingest", "remove"]
        })}
      />
    )

    expect(screen.getByText("Document processing blocked")).toBeInTheDocument()
    expect(screen.getByText("Blocked by 24k token limit")).toBeInTheDocument()
    expect(screen.getByText("Ingest failed")).toBeInTheDocument()
    expect(screen.getByText("Switch to ingest")).toBeInTheDocument()
    expect(screen.getByText("Remove file")).toBeInTheDocument()
  })
})
