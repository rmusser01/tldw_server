// @vitest-environment jsdom
/**
 * TASK-15622: mermaid 11 renders into a temporary `#d<id>` element appended
 * to <body>. On a parse error it draws an error diagram there and throws
 * without removing it, so every mount of an invalid diagram left one more
 * node in <body>. This uses the real mermaid module on purpose.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MermaidDiagramBlock } from "../MermaidDiagramBlock"

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: () => ({ openArtifact: vi.fn() })
}))

describe("Mermaid render errors", () => {
  it("leaves no mermaid temp element in <body> and still shows the source fallback", async () => {
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    })
    const source = "graph TD\nA --> "

    const { container, unmount } = render(<MermaidDiagramBlock source={source} />)

    expect(
      await screen.findByText(
        "Unable to render Mermaid diagram.",
        undefined,
        { timeout: 10_000 }
      )
    ).toBeInTheDocument()
    expect(
      Array.from(container.querySelectorAll("pre")).map((pre) => pre.textContent)
    ).toContain(source)
    expect(document.body.querySelectorAll('[id^="dmermaid-"]')).toHaveLength(0)
    unmount()
  })
})
