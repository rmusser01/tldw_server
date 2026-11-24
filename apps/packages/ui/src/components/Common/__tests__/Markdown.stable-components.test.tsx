// @vitest-environment jsdom
/**
 * TASK-15622: an idle /chat re-renders every chat row on each health or
 * research poll. Markdown used to hand react-markdown new tag components on
 * every render, so React remounted the whole rendered message each time.
 * These tests pin DOM node identity across re-renders.
 */
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"
import { CodeBlock } from "../CodeBlock"
import { useUiModeStore } from "@/store/ui-mode"

const { initializeMock, openArtifactMock, renderMock } = vi.hoisted(() => ({
  initializeMock: vi.fn(),
  openArtifactMock: vi.fn(),
  renderMock: vi.fn()
}))

vi.mock("mermaid", () => ({
  default: {
    initialize: initializeMock,
    render: renderMock
  }
}))

// Stable `t`, as react-i18next gives: a new one per render would re-run
// every effect that lists it and hide effects that miss their real deps.
const translate = (_key: string, fallback?: string) => fallback || _key
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: translate })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("antd", () => ({
  ConfigProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: () => ({ openArtifact: openArtifactMock, isPinned: false })
}))

// a11yLightCodeTheme / a11yDarkCodeTheme plain colours (utils/code-theme.ts).
const LIGHT_CODE_COLOR = "rgb(31, 35, 40)"
const DARK_CODE_COLOR = "rgb(248, 248, 242)"

const codeColor = (container: HTMLElement) =>
  (container.querySelector("pre") as HTMLElement | null)?.style.color

// Re-renders Markdown's parent with identical Markdown props.
const Parent = ({
  tick,
  variant
}: {
  tick: number
  variant: "default" | "compact" | "github"
}) => (
  <div data-tick={tick}>
    <Markdown message={"```js\nconst a = 1\n```"} codeBlockVariant={variant} />
  </div>
)

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
    openArtifactMock.mockReset()
    renderMock.mockReset()
    document.documentElement.className = ""
    useUiModeStore.setState({ mode: "casual" })
    delete (window as unknown as Record<string, unknown>)
      .__tableArtifactAutoOpenState
  })

  it.each(["default", "compact", "github"] as const)(
    "re-colours %s code blocks when the page theme changes and the parent re-renders",
    async (variant) => {
      document.documentElement.className = "light"
      const { container, rerender } = render(<Parent tick={0} variant={variant} />)
      expect(codeColor(container)).toBe(LIGHT_CODE_COLOR)

      document.documentElement.className = "dark"
      rerender(<Parent tick={1} variant={variant} />)

      await waitFor(() => expect(codeColor(container)).toBe(DARK_CODE_COLOR))
    }
  )

  it("re-colours code blocks when the page theme changes with no parent re-render", async () => {
    document.documentElement.className = "light"
    const { container } = render(<Parent tick={0} variant="github" />)
    expect(codeColor(container)).toBe(LIGHT_CODE_COLOR)

    act(() => {
      document.documentElement.className = "dark"
    })

    await waitFor(() => expect(codeColor(container)).toBe(DARK_CODE_COLOR))
  })

  it("re-colours code blocks when the OS colour scheme changes and <html> has no theme class", () => {
    let prefersDark = false
    const listeners = new Set<() => void>()
    window.matchMedia = vi.fn().mockImplementation(() => ({
      get matches() {
        return prefersDark
      },
      addEventListener: (_type: string, listener: () => void) =>
        listeners.add(listener),
      removeEventListener: (_type: string, listener: () => void) =>
        listeners.delete(listener)
    }))
    const { container } = render(<Parent tick={0} variant="github" />)
    expect(codeColor(container)).toBe(LIGHT_CODE_COLOR)

    act(() => {
      prefersDark = true
      listeners.forEach((listener) => listener())
    })

    expect(codeColor(container)).toBe(DARK_CODE_COLOR)
  })

  it("re-opens the pro-mode table artifact with the rows streamed after the header", async () => {
    useUiModeStore.setState({ mode: "pro" })
    const header = "| A | B |\n| - | - |"
    const { rerender } = render(<Markdown message={header} />)
    await waitFor(() => expect(openArtifactMock).toHaveBeenCalled())

    rerender(<Markdown message={`${header}\n| 1 | 2 |`} />)
    rerender(<Markdown message={`${header}\n| 1 | 2 |\n| 3 | 4 |`} />)

    await waitFor(() => {
      expect(openArtifactMock).toHaveBeenLastCalledWith(
        expect.objectContaining({
          kind: "table",
          table: {
            headers: ["A", "B"],
            rows: [
              ["1", "2"],
              ["3", "4"]
            ]
          }
        }),
        { auto: true }
      )
    })
    // Header only, then one row, then two rows: each is a new artifact id.
    expect(openArtifactMock).toHaveBeenCalledTimes(3)
  })

  it("keeps heading anchors and code block indexes correct across a streamed re-render", () => {
    const message = [
      "## Alpha",
      "",
      "```python",
      "a = 1",
      "```",
      "",
      "## Beta",
      "",
      "```python",
      "b = 2",
      "```",
      "",
      "Tail"
    ].join("\n")
    const anchors = ["section-alpha", "section-beta"]
    const { container, rerender } = render(
      <Markdown message={message} headingAnchorIds={anchors} />
    )
    const expectedOrigins = [0, 1].map((blockIndex) => {
      const { container: single, unmount } = render(
        <CodeBlock language="python" value="x" blockIndex={blockIndex} />
      )
      const origin = single
        .querySelector("[data-artifact-origin]")
        ?.getAttribute("data-artifact-origin")
      unmount()
      return origin
    })
    const readState = () => ({
      headingIds: Array.from(container.querySelectorAll("h2")).map((h) => h.id),
      origins: Array.from(
        container.querySelectorAll("[data-artifact-origin]")
      ).map((element) => element.getAttribute("data-artifact-origin"))
    })
    expect(readState()).toEqual({ headingIds: anchors, origins: expectedOrigins })

    rerender(
      <Markdown message={`${message} keeps streaming`} headingAnchorIds={anchors} />
    )

    expect(container.textContent).toContain("Tail keeps streaming")
    expect(readState()).toEqual({ headingIds: anchors, origins: expectedOrigins })
  })

  it("passes changed searchQuery, codeBlockVariant and allowExternalImages to the rendered blocks", () => {
    const message = [
      "- find me",
      "",
      "```js",
      "const a = 1",
      "```",
      "",
      "![Chart](https://example.com/chart.png)"
    ].join("\n")
    const { container, rerender } = render(
      <Markdown
        message={message}
        codeBlockVariant="github"
        allowExternalImages={false}
      />
    )
    expect(container.querySelector("li mark")).toBeNull()
    expect(container.querySelector("pre")).not.toBeNull()
    expect(container.querySelector("img")).toBeNull()
    expect(container.textContent).toContain("Image: Chart")

    rerender(
      <Markdown
        message={message}
        codeBlockVariant="plain"
        allowExternalImages
        searchQuery="find"
      />
    )

    expect(container.querySelector("li mark")?.textContent).toBe("find")
    expect(container.querySelector("pre")).toBeNull()
    expect(container.textContent).toContain("const a = 1")
    expect(container.querySelector("img")?.getAttribute("src")).toBe(
      "https://example.com/chart.png"
    )
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
