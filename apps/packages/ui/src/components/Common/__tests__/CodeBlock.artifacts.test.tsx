// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { CodeBlock } from "../CodeBlock"
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
  })

  afterEach(() => {
    vi.restoreAllMocks()
    resetArtifactsStore()
  })

  it.each([
    ["graphviz", "digraph G {\n  A -> B\n}"],
    ["dot", "digraph G {\n  A -> B\n}"],
    ["diagram", "box A -> box B"]
  ])("keeps %s fences as code artifacts instead of Mermaid diagrams", (language, value) => {
    render(<CodeBlock language={language} value={value} />)

    fireEvent.click(screen.getByRole("button", { name: /view/i }))

    expect(useArtifactsStore.getState().active).toEqual(
      expect.objectContaining({
        content: value,
        kind: "code",
        language
      })
    )
  })

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
