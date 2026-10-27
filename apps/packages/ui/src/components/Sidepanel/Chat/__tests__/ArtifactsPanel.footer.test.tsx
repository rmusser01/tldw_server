// @vitest-environment jsdom
import React from "react"
import { render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ArtifactsPanel } from "../ArtifactsPanel"
import { useArtifactsStore, type ArtifactItem } from "@/store/artifacts"

// XS-18: in a 420px side panel the footer actions must wrap instead of
// running off-screen, and it must not offer a "Run" that can never run.

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
  message: { error: vi.fn(), success: vi.fn() }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: vi.fn((path: string) => path) },
    tabs: { create: vi.fn() }
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: any) => unknown) =>
    selector({ serverChatId: "chat-1", serverChatTitle: "Chat title" })
}))

const tableArtifact: ArtifactItem = {
  id: "table-abc",
  title: "Option",
  content: "Option,Pros,Cons\nA,Fast,Less flexible",
  language: "csv",
  kind: "table",
  table: { headers: ["Option", "Pros", "Cons"], rows: [["A", "Fast", "Less flexible"]] }
}

const codeArtifact: ArtifactItem = {
  id: "code-abc",
  title: "Script",
  content: "print('hi')",
  language: "python",
  kind: "code",
  lineCount: 1
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

describe("ArtifactsPanel footer at side-panel widths (XS-18)", () => {
  beforeEach(resetArtifactsStore)
  afterEach(() => {
    document.body.innerHTML = ""
    resetArtifactsStore()
  })

  it.each([
    ["table", tableArtifact],
    ["code", codeArtifact]
  ] as const)("does not offer a Run action that can never run (%s)", (_kind, artifact) => {
    useArtifactsStore.getState().openArtifact(artifact)
    render(<ArtifactsPanel />)
    expect(screen.queryByRole("button", { name: /run/i })).not.toBeInTheDocument()
    expect(screen.queryByText("Run (N/A)")).not.toBeInTheDocument()
  })

  it("lets the table actions wrap onto more lines instead of overflowing", () => {
    useArtifactsStore.getState().openArtifact(tableArtifact)
    render(<ArtifactsPanel />)
    const actions = screen.getByRole("group", { name: "Artifact actions" })
    expect(actions).toHaveClass("flex-wrap")
    for (const name of ["Jump to source", "Copy", "Download", "Save to Data Tables"]) {
      expect(within(actions).getByText(name)).toBeInTheDocument()
    }
  })
})
