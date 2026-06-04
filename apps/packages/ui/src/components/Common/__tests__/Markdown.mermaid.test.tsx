import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

const mermaidDiagramBlockMock = vi.hoisted(() => vi.fn())

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("../MermaidDiagramBlock", () => ({
  MermaidDiagramBlock: ({ source }: { source: string }) => {
    mermaidDiagramBlockMock(source)
    return <div data-testid="mermaid-diagram-block">{source}</div>
  },
  default: ({ source }: { source: string }) => {
    mermaidDiagramBlockMock(source)
    return <div data-testid="mermaid-diagram-block">{source}</div>
  }
}))

describe("Markdown Mermaid fences", () => {
  beforeEach(() => {
    mermaidDiagramBlockMock.mockClear()
  })

  it("renders enabled Mermaid fences through the Mermaid diagram block", () => {
    render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph TD\nA --> B"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith("graph TD\nA --> B")
  })

  it("keeps Mermaid fences as normal code blocks when Mermaid rendering is disabled", () => {
    const { container } = render(
      <Markdown message={"```mermaid\ngraph TD\nA --> B\n```"} />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(screen.getByText("A --> B")).toBeInTheDocument()
    expect(container.querySelector("pre")).toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("does not treat mmd fences as Mermaid diagrams", () => {
    const { container } = render(
      <Markdown
        message={"```mmd\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(container.querySelector("pre")).toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("preserves GitHub code blocks when Mermaid rendering is disabled", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        codeBlockVariant="github"
        enableMermaidDiagrams={false}
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(container.querySelectorAll("pre")).toHaveLength(1)
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("bypasses the ST-compatible HTML region for enabled Mermaid fences", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
        richTextModeOverride="st_compat"
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block")).toBeInTheDocument()
    expect(
      container.querySelector('[aria-label="Message content"]')
    ).not.toBeInTheDocument()
  })

  it("keeps non-Mermaid ST-compatible markdown on the HTML path when enabled", () => {
    const { container } = render(
      <Markdown
        message={"**Bold** text"}
        enableMermaidDiagrams
        richTextModeOverride="st_compat"
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
    expect(
      container.querySelector('[aria-label="Message content"]')
    ).toBeInTheDocument()
  })
})
