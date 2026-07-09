// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { AttachmentsSummary } from "../AttachmentsSummary"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Image: ({ alt }: { alt?: string }) => <img alt={alt || "image"} />
}))

describe("AttachmentsSummary integration", () => {
  it("shows large attachment warning with actionable alternatives", async () => {
    const user = userEvent.setup()
    const onOpenKnowledgePanel = vi.fn()
    const onClearFiles = vi.fn()

    render(
      <AttachmentsSummary
        image=""
        documents={[]}
        files={[
          {
            id: "file-1",
            filename: "long-report.pdf",
            size: 13 * 1024 * 1024
          },
          {
            id: "file-2",
            filename: "archive.zip",
            size: 9 * 1024 * 1024
          }
        ]}
        onRemoveImage={vi.fn()}
        onRemoveDocument={vi.fn()}
        onClearDocuments={vi.fn()}
        onRemoveFile={vi.fn()}
        onClearFiles={onClearFiles}
        onOpenKnowledgePanel={onOpenKnowledgePanel}
      />
    )

    await user.click(screen.getByRole("button", { name: /Attachments/i }))

    expect(screen.getByTestId("attachments-large-warning")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Large file attachments can increase latency and context usage."
      )
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Review in context panel" })
    )
    await user.click(screen.getByRole("button", { name: "Clear files" }))

    expect(onOpenKnowledgePanel).toHaveBeenCalledTimes(1)
    expect(onClearFiles).toHaveBeenCalledTimes(1)
  })

  it("does not show large attachment warning for small file batches", async () => {
    const user = userEvent.setup()

    render(
      <AttachmentsSummary
        image=""
        documents={[]}
        files={[
          {
            id: "file-1",
            filename: "note.txt",
            size: 1 * 1024 * 1024
          },
          {
            id: "file-2",
            filename: "summary.md",
            size: 500 * 1024
          }
        ]}
        onRemoveImage={vi.fn()}
        onRemoveDocument={vi.fn()}
        onClearDocuments={vi.fn()}
        onRemoveFile={vi.fn()}
        onClearFiles={vi.fn()}
        onOpenKnowledgePanel={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: /Attachments/i }))
    expect(screen.queryByTestId("attachments-large-warning")).toBeNull()
  })

  it("shows document processing status without hiding remove", async () => {
    const user = userEvent.setup()
    const onRemoveFile = vi.fn()

    render(
      <AttachmentsSummary
        image=""
        documents={[]}
        files={[
          {
            id: "file-1",
            filename: "blocked.pdf",
            size: 1 * 1024 * 1024,
            processingMode: "ocr_pages",
            processingStatus: "blocked"
          }
        ]}
        onRemoveImage={vi.fn()}
        onRemoveDocument={vi.fn()}
        onClearDocuments={vi.fn()}
        onRemoveFile={onRemoveFile}
        onClearFiles={vi.fn()}
        onOpenKnowledgePanel={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: /Attachments/i }))

    expect(screen.getByText("Blocked")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    expect(onRemoveFile).toHaveBeenCalledWith("file-1")
  })
})
