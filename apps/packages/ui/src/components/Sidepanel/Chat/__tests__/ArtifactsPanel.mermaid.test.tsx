// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ArtifactsPanel } from "../ArtifactsPanel"
import { useArtifactsStore, type ArtifactItem } from "@/store/artifacts"

const mermaidDiagramBlockMock = vi.hoisted(() => vi.fn())

type MermaidDiagramBlockMockProps = {
  source: string
  blockIndex?: number
  artifactContextId?: string
  enableArtifactAction?: boolean
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  message: {
    error: vi.fn()
  }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: vi.fn((path: string) => path)
    },
    tabs: {
      create: vi.fn()
    }
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: any) => unknown) =>
    selector({
      serverChatId: "chat-1",
      serverChatTitle: "Chat title"
    })
}))

vi.mock("@/components/Common/MermaidDiagramBlock", () => ({
  MermaidDiagramBlock: (props: MermaidDiagramBlockMockProps) => {
    mermaidDiagramBlockMock(props)
    return (
      <section data-testid="shared-mermaid-diagram-block">
        <div aria-label="Mock shared Mermaid diagram" role="img">
          {props.source}
        </div>
        {props.enableArtifactAction && (
          <button type="button">View Mermaid diagram</button>
        )}
        <button type="button">Open Mermaid preview</button>
        <button type="button">Copy Mermaid source</button>
        <button type="button">Download Mermaid SVG</button>
      </section>
    )
  }
}))

const diagramArtifact: ArtifactItem = {
  id: "mermaid-assistant-message-1-segment-0-block-0-abc123",
  title: "Mermaid diagram 1",
  content: "graph TD\n  A-->B",
  kind: "diagram",
  language: "mermaid",
  lineCount: 2
}

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

describe("ArtifactsPanel Mermaid artifacts", () => {
  beforeEach(() => {
    mermaidDiagramBlockMock.mockClear()
    resetArtifactsStore()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    document.body.innerHTML = ""
    resetArtifactsStore()
  })

  it("renders diagram artifacts with the shared Mermaid block actions", () => {
    useArtifactsStore.getState().openArtifact(diagramArtifact)

    render(<ArtifactsPanel />)

    expect(screen.getByTestId("artifacts-panel")).toBeInTheDocument()
    expect(screen.getByText("Mermaid diagram 1")).toBeInTheDocument()
    expect(
      screen.getByTestId("shared-mermaid-diagram-block")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("img", { name: "Mock shared Mermaid diagram" })
    ).toHaveTextContent("graph TD A-->B")
    expect(
      screen.getByRole("button", { name: "Open Mermaid preview" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Copy Mermaid source" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Download Mermaid SVG" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "View Mermaid diagram" })
    ).not.toBeInTheDocument()
    expect(mermaidDiagramBlockMock).toHaveBeenCalledWith(
      expect.objectContaining({
        enableArtifactAction: false,
        source: diagramArtifact.content
      })
    )
  })

  it("jumps to the matching Mermaid artifact origin", () => {
    vi.useFakeTimers()
    const origin = document.createElement("div")
    origin.id = `artifact-origin-${diagramArtifact.id}`
    origin.scrollIntoView = vi.fn()
    document.body.appendChild(origin)
    const focusArtifactsTrigger = vi.fn()
    const scrollLatest = vi.fn()
    window.addEventListener("tldw:focus-artifacts-trigger", focusArtifactsTrigger)
    window.addEventListener("tldw:scroll-to-latest", scrollLatest)
    try {
      useArtifactsStore.getState().openArtifact(diagramArtifact)

      render(<ArtifactsPanel />)

      fireEvent.click(screen.getByTestId("artifacts-jump-source"))

      expect(origin.scrollIntoView).toHaveBeenCalledWith({
        behavior: "smooth",
        block: "center"
      })
      expect(useArtifactsStore.getState().isOpen).toBe(false)
      expect(scrollLatest).not.toHaveBeenCalled()

      act(() => {
        vi.runOnlyPendingTimers()
      })
      expect(focusArtifactsTrigger).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener(
        "tldw:focus-artifacts-trigger",
        focusArtifactsTrigger
      )
      window.removeEventListener("tldw:scroll-to-latest", scrollLatest)
    }
  })
})
