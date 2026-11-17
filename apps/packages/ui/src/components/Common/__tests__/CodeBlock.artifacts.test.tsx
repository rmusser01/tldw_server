// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { CodeBlock } from "../CodeBlock"
import { useUiModeStore } from "@/store/ui-mode"
import { useArtifactsStore } from "@/store/artifacts"

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

const resetArtifactsStore = () => {
  useArtifactsStore.setState((state) => ({
    ...state,
    active: null,
    isOpen: false,
    isPinned: false,
    history: [],
    unreadCount: 0
  }))
}

describe("CodeBlock artifacts", () => {
  beforeEach(() => {
    resetArtifactsStore()
    useUiModeStore.setState({ mode: "casual" })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    resetArtifactsStore()
  })

  it.each([
    ["graphviz", "digraph G {\n  A -> B\n}"],
    ["dot", "digraph G {\n  A -> B\n}"],
    ["diagram", "box A -> box B"]
  ])(
    "keeps %s fences as code artifacts instead of Mermaid diagrams",
    (language, value) => {
      render(<CodeBlock language={language} value={value} />)

      fireEvent.click(screen.getByRole("button", { name: /view/i }))

      expect(useArtifactsStore.getState().active).toEqual(
        expect.objectContaining({
          content: value,
          kind: "code",
          language
        })
      )
    }
  )

  it("opens Mermaid fences as diagram artifacts", () => {
    render(<CodeBlock language="mermaid" value={"graph TD\n  A-->B"} />)

    fireEvent.click(screen.getByRole("button", { name: /view/i }))

    expect(useArtifactsStore.getState().active).toEqual(
      expect.objectContaining({
        content: "graph TD\n  A-->B",
        kind: "diagram",
        language: "mermaid"
      })
    )
  })
})

describe("CodeBlock preview contracts", () => {
  beforeEach(() => {
    resetArtifactsStore()
    useUiModeStore.setState({ mode: "casual" })
    ;(
      window as unknown as { __artifactAutoOpenState?: Map<string, boolean> }
    ).__artifactAutoOpenState?.clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    resetArtifactsStore()
    useUiModeStore.setState({ mode: "casual" })
  })
  it("keeps unequal code lines in one shared column layout", () => {
    const { container } = render(
      <CodeBlock language="python" value={"x = 1\n    longer_value = 2"} />
    )
    const rows = container.querySelectorAll("pre .table-row")
    expect(rows).toHaveLength(2)
    expect(rows[0].parentElement).toBe(rows[1].parentElement)
    expect(rows[0].parentElement).toHaveClass("table")
    expect(rows[1].lastElementChild?.textContent).toBe("    longer_value = 2")
  })
  it.each(["python", "mermaid"])(
    "disables manual and pro automatic artifacts for hostless %s",
    (language) => {
      useUiModeStore.setState({ mode: "pro" })
      render(
        <CodeBlock
          language={language}
          value={Array.from(
            { length: 12 },
            (_, i) => "unique_hostless_line_" + i
          ).join("\n")}
          enableArtifactAction={false}
        />
      )
      expect(
        screen.queryByRole("button", { name: /view/i })
      ).not.toBeInTheDocument()
      expect(useArtifactsStore.getState().active).toBeNull()
      expect(useArtifactsStore.getState().history).toEqual([])
    }
  )
  it("places copy and download in the existing header alongside view", () => {
    render(<CodeBlock language="python" value="header_controls = 1" />)
    const view = screen.getByRole("button", { name: /view/i })
    expect(view.parentElement).toContainElement(
      screen.getByRole("button", { name: "Copy to clipboard" })
    )
    expect(view.parentElement).toContainElement(
      screen.getByRole("button", { name: "Download code" })
    )
  })
})

describe("CodeBlock action effects", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    resetArtifactsStore()
    useUiModeStore.setState({ mode: "casual" })
  })
  it("copies the exact source including indentation", async () => {
    const source = "x = 1\n    longer_value = 2"
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })
    render(
      <CodeBlock
        language="python"
        value={source}
        enableArtifactAction={false}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Copy to clipboard" }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(source))
  })
  it("downloads the exact source with a safe filename and cleans up", async () => {
    const source = "x = 1\n    longer_value = 2"
    const createObjectURL = vi.fn(() => "blob:unit12-download")
    const revokeObjectURL = vi.fn()
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = createObjectURL
        static revokeObjectURL = revokeObjectURL
      }
    )
    let download = ""
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      function () {
        download = this.download
      }
    )
    render(
      <CodeBlock
        language="python"
        value={source}
        enableArtifactAction={false}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Download code" }))
    expect(download).toMatch(/^code_[0-9TZ-]+\.py$/)
    expect(
      await (createObjectURL.mock.calls[0] as unknown as [Blob])[0].text()
    ).toBe(source)
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:unit12-download")
    expect(document.querySelector("a[download]")).toBeNull()
  })
  it("keeps default pro automatic code artifacts enabled", () => {
    resetArtifactsStore()
    useUiModeStore.setState({ mode: "pro" })
    const source = Array.from(
      { length: 12 },
      (_, i) => "chat_default_auto_" + i
    ).join("\n")
    render(<CodeBlock language="python" value={source} />)
    expect(useArtifactsStore.getState().active).toEqual(
      expect.objectContaining({
        content: source,
        language: "python",
        kind: "code"
      })
    )
  })
  it("keeps hostless HTML preview sandboxed and sanitizes its source", async () => {
    const createObjectURL = vi.fn(() => "blob:unit12-html")
    const revokeObjectURL = vi.fn()
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = createObjectURL
        static revokeObjectURL = revokeObjectURL
      }
    )
    const { unmount } = render(
      <CodeBlock
        language="html"
        value={'<p onclick="steal()">Safe</p><script>steal()</script>'}
        enableArtifactAction={false}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "preview" }))
    expect(screen.getByTitle("Preview")).toHaveAttribute(
      "sandbox",
      "allow-scripts"
    )
    const html = await (
      createObjectURL.mock.calls[0] as unknown as [Blob]
    )[0].text()
    expect(html).toContain("<p>Safe</p>")
    expect(html).not.toContain("steal()")
    expect(
      screen.queryByRole("button", { name: /view code/i })
    ).not.toBeInTheDocument()
    unmount()
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:unit12-html")
  })
})
