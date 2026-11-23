// @vitest-environment jsdom
/**
 * TASK-15622: an idle /chat re-renders every chat row on each health or
 * research poll. Markdown used to hand react-markdown new tag components on
 * every render, so React remounted the whole rendered message each time.
 * These tests pin DOM node identity across re-renders.
 */
import React from "react"
import { render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

const { initializeMock, renderMock } = vi.hoisted(() => ({
  initializeMock: vi.fn(),
  renderMock: vi.fn()
}))

vi.mock("mermaid", () => ({
  default: {
    initialize: initializeMock,
    render: renderMock
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: () => ({ openArtifact: vi.fn() })
}))

const streamedReply = [
  "First paragraph.",
  "",
  "- item one",
  "- item two",
  "",
  "```python",
  "print('hi')",
  "```",
  "",
  "Last paragraph"
].join("\n")

describe("Markdown re-render stability", () => {
  beforeEach(() => {
    renderMock.mockResolvedValue({
      svg: "<svg><text>Rendered diagram</text></svg>"
    })
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    })
  })

  afterEach(() => {
    initializeMock.mockReset()
    renderMock.mockReset()
  })

  it("keeps the rendered paragraph node when re-rendered with identical props", () => {
    const { container, rerender } = render(
      <Markdown message={streamedReply} codeBlockVariant="github" />
    )
    const paragraph = container.querySelector("p")
    expect(paragraph).not.toBeNull()

    rerender(<Markdown message={streamedReply} codeBlockVariant="github" />)

    expect(container.querySelector("p")).toBe(paragraph)
  })

  it("keeps earlier blocks mounted while the last paragraph streams", () => {
    const { container, rerender } = render(
      <Markdown message={streamedReply} codeBlockVariant="github" />
    )
    const firstParagraph = container.querySelector("p")
    const list = container.querySelector("ul")
    const codeBlock = container.querySelector("[data-code-block]")
    expect(firstParagraph).not.toBeNull()
    expect(list).not.toBeNull()
    expect(codeBlock).not.toBeNull()

    rerender(
      <Markdown
        message={`${streamedReply} keeps streaming`}
        codeBlockVariant="github"
      />
    )

    expect(container.textContent).toContain("Last paragraph keeps streaming")
    expect(container.querySelector("p")).toBe(firstParagraph)
    expect(container.querySelector("ul")).toBe(list)
    expect(container.querySelector("[data-code-block]")).toBe(codeBlock)
  })

  it("still applies per-instance heading anchors and search highlights", () => {
    const { container } = render(
      <Markdown
        message={"## Alpha\n\n## Beta\n\n- find me"}
        headingAnchorIds={["section-alpha", "section-beta"]}
        searchQuery="find"
      />
    )

    expect(
      Array.from(container.querySelectorAll("h2")).map((h) => h.id)
    ).toEqual(["section-alpha", "section-beta"])
    expect(container.querySelector("li mark")?.textContent).toBe("find")
  })

  it("keeps a closed Mermaid diagram mounted when text is appended after it", async () => {
    const message = "Intro\n\n```mermaid\ngraph TD\nA --> B\n```\n\nAfter"
    const { container, rerender } = render(
      <Markdown
        message={message}
        codeBlockVariant="github"
        enableMermaidDiagrams
      />
    )
    const diagramSelector = '[aria-label="Mermaid diagram"]'
    await waitFor(() => {
      expect(container.querySelector(diagramSelector)?.textContent).toBe(
        "Rendered diagram"
      )
    })
    const diagram = container.querySelector(diagramSelector)

    rerender(
      <Markdown
        message={`${message} the diagram`}
        codeBlockVariant="github"
        enableMermaidDiagrams
      />
    )

    expect(container.textContent).toContain("After the diagram")
    expect(container.querySelector(diagramSelector)).toBe(diagram)
    expect(renderMock).toHaveBeenCalledTimes(1)
  })
})
