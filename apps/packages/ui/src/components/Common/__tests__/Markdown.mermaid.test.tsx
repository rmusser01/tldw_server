import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

const mermaidDiagramBlockMock = vi.hoisted(() => vi.fn())

type MermaidDiagramBlockMockProps = {
  source: string
  blockIndex?: number
  artifactContextId?: string
  enableArtifactAction?: boolean
}

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("../MermaidDiagramBlock", () => ({
  MermaidDiagramBlock: (props: MermaidDiagramBlockMockProps) => {
    const { source } = props
    mermaidDiagramBlockMock(props)
    return <div data-testid="mermaid-diagram-block">{source}</div>
  },
  default: (props: MermaidDiagramBlockMockProps) => {
    const { source } = props
    mermaidDiagramBlockMock(props)
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
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 0
      })
    )
  })

  it("keeps Mermaid artifact actions disabled by default", () => {
    render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
      />
    )

    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 0,
        enableArtifactAction: false
      })
    )
  })

  it("forwards artifact action opt-in and context to Mermaid blocks", () => {
    render(
      <Markdown
        artifactContextId="assistant-message-42"
        enableMermaidArtifactActions
        enableMermaidDiagrams
        message={"```mermaid\ngraph TD\nA --> B\n```"}
      />
    )

    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        artifactContextId: "assistant-message-42",
        blockIndex: 0,
        enableArtifactAction: true,
        source: "graph TD\nA --> B"
      })
    )
  })

  it("renders enabled Mermaid tilde fences through the Mermaid diagram block", () => {
    render(
      <Markdown
        message={"~~~mermaid\ngraph TD\nA --> B\n~~~"}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph TD\nA --> B"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 0
      })
    )
  })

  it("renders Mermaid fences case-insensitively", () => {
    render(
      <Markdown
        message={"```Mermaid\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph TD\nA --> B"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 0
      })
    )
  })

  it("renders Mermaid fences with CRLF line endings", () => {
    render(
      <Markdown
        message={"```mermaid\r\ngraph TD\r\nA --> B\r\n```"}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph TD\nA --> B"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 0
      })
    )
  })

  it("ignores literal Mermaid fences inside non-Mermaid code blocks", () => {
    render(
      <Markdown
        message={[
          "````markdown",
          "```mermaid",
          "graph TD",
          "A --> B",
          "```",
          "````",
          "",
          "```mermaid",
          "graph LR",
          "C --> D",
          "```"
        ].join("\n")}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph LR\nC --> D"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledTimes(1)
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph LR\nC --> D",
        blockIndex: 1
      })
    )
  })

  it("renders Mermaid fences after indented code blocks", () => {
    render(
      <Markdown
        message={[
          "    indented code",
          "",
          "```mermaid",
          "graph TD",
          "A --> B",
          "```"
        ].join("\n")}
        enableMermaidDiagrams
      />
    )

    expect(screen.getByTestId("mermaid-diagram-block").textContent).toBe(
      "graph TD\nA --> B"
    )
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 1
      })
    )
  })

  it("does not render unclosed Mermaid fences as diagrams", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B"}
        enableMermaidDiagrams
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(screen.getByText("A --> B")).toBeInTheDocument()
    expect(container.querySelector("pre")).toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("forwards the code block index to enabled closed Mermaid fences", () => {
    render(
      <Markdown
        message={[
          "```text",
          "first",
          "```",
          "",
          "```mermaid",
          "graph TD",
          "A --> B",
          "```"
        ].join("\n")}
        enableMermaidDiagrams
      />
    )

    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "graph TD\nA --> B",
        blockIndex: 1
      })
    )
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

  it("preserves plain code blocks when Mermaid rendering is disabled", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        codeBlockVariant="plain"
        enableMermaidDiagrams={false}
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(container.textContent).toContain("graph TD\nA --> B")
    expect(container.querySelector("pre")).not.toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("preserves compact code blocks when Mermaid rendering is disabled", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid\ngraph TD\nA --> B\n```"}
        codeBlockVariant="compact"
        enableMermaidDiagrams={false}
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(screen.getByText("A --> B")).toBeInTheDocument()
    expect(container.querySelector("pre")).toBeInTheDocument()
    expect(mermaidDiagramBlockMock).not.toHaveBeenCalled()
  })

  it("does not treat mermaid-js fences as Mermaid diagrams", () => {
    const { container } = render(
      <Markdown
        message={"```mermaid-js\ngraph TD\nA --> B\n```"}
        enableMermaidDiagrams
      />
    )

    expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
    expect(screen.getByText("graph TD")).toBeInTheDocument()
    expect(container.querySelector("pre")).toBeInTheDocument()
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
