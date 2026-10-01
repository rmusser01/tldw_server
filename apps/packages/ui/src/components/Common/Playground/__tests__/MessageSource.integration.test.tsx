// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { MessageSource } from "../MessageSource"

vi.mock("@/components/Option/Knowledge/KnowledgeIcon", () => ({
  KnowledgeIcon: ({ className }: { className?: string }) => (
    <span data-testid="knowledge-icon" className={className} />
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

describe("MessageSource citation transparency integration", () => {
  it.each([
    "lumen-field-memo.md",
    "/lumen-field-memo.md",
    "./lumen-field-memo.md",
    "//example.com/document",
    "http:document",
    "mailto:research@example.com",
    "java\tscript:alert(1)",
    "https://"
  ])("keeps provenance without inventing navigation for %s", (url) => {
    const { container } = render(
      <MessageSource source={{ url, content: "Preserved source evidence" }} />
    )

    expect(container.querySelector("summary")?.textContent).toBe(url)
    expect(screen.getByText("Preserved source evidence")).toBeInTheDocument()
    expect(container.querySelector("a")).toBeNull()
  })

  it("retains an uploaded filename without content as non-navigable provenance", () => {
    const { container } = render(
      <MessageSource source={{ url: "lumen-field-memo.md" }} />
    )

    expect(screen.getByText("lumen-field-memo.md")).toBeInTheDocument()
    expect(container.querySelector("a")).toBeNull()
  })

  it.each([undefined, "Preserved source evidence"])(
    "retains sanitized absolute source navigation with content %s",
    async (content) => {
      const user = userEvent.setup()
      const onSourceNavigate = vi.fn()
      const source = {
        name: "Published document",
        url: " \nHTTPS://example.com/document?q=1#section ",
        content
      }
      render(
        <MessageSource source={source} onSourceNavigate={onSourceNavigate} />
      )

      if (content) await user.click(screen.getByText(source.name))
      const link = screen.getByRole("link")
      expect(link).toHaveAttribute(
        "href",
        "HTTPS://example.com/document?q=1#section"
      )
      expect(link).toHaveAttribute("rel", "noopener noreferrer")
      await user.click(link)
      expect(onSourceNavigate).toHaveBeenCalledWith(source)
    }
  )

  it("shows why-this-source diagnostics and opens knowledge panel from citation card", async () => {
    const user = userEvent.setup()
    const onOpenKnowledgePanel = vi.fn()

    render(
      <MessageSource
        source={{
          name: "Doc A",
          content: "Quoted snippet",
          score: 0.91,
          metadata: {
            chunk_id: "chunk_2_of_9",
            retrieval_strategy: "hybrid",
            source_type: "media_db",
            reason: "High lexical overlap"
          }
        }}
        onOpenKnowledgePanel={onOpenKnowledgePanel}
      />
    )

    await user.click(screen.getByText("Doc A"))

    expect(screen.getByText("Why this source")).toBeInTheDocument()
    expect(screen.getByText("Relevance:")).toBeInTheDocument()
    expect(screen.getByText("91%")).toBeInTheDocument()
    expect(screen.getByText("Chunk:")).toBeInTheDocument()
    expect(screen.getByText("chunk_2_of_9")).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Open Search & Context" })
    )
    expect(onOpenKnowledgePanel).toHaveBeenCalledTimes(1)
  })
})
