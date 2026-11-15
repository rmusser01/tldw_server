// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
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
  afterEach(() => vi.unstubAllGlobals());
  it("uses the extension options router for authenticated PDF links", async () => {
    vi.stubGlobal("browser", { runtime: { id: "test", getURL: (path: string) => `chrome-extension://test${path}` } });
    render(<MessageSource source={{ name: "Policy", type: "pdf", content: "Evidence", metadata: { media_id: 42, page_number: 352 } }} />);
    await userEvent.setup().click(screen.getByText("Policy"));
    expect(screen.getByRole("link", { name: "Open PDF page 352" })).toHaveAttribute("href", "chrome-extension://test/options.html#/document-workspace?open=42&page=352");
  });
  it.each([{}, [null]])("does not crash on malformed page span metadata %j", page_spans => {
    render(<MessageSource source={{ name: "Policy", type: "pdf", content: "Evidence", metadata: { media_id: 42, page_number: 352, page_spans } } as any} />);
    expect(screen.getByText("PDF p. 352")).toBeInTheDocument();
  });
  it("labels a saved PDF citation and opens its authenticated physical page", async () => {
    render(<MessageSource source={{ name: "Policy", type: "pdf", mode: "rag", content: "Evidence", metadata: {
      media_id: 42, page_number: 352, citation_index: 2, printed_page_label: "319"
    } }} />);
    expect(screen.getByText("[2]")).toBeInTheDocument();
    expect(screen.getByText("Printed p. 319 · PDF p. 352")).toBeInTheDocument();
    await userEvent.setup().click(screen.getByText("Policy"));
    expect(screen.getByRole("link", { name: "Open PDF page 352" })).toHaveAttribute("href", "/document-workspace?open=42&page=352");
  });
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
