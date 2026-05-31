import React from "react"
import { existsSync } from "node:fs"
import { join } from "node:path"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("next/dynamic", () => ({
  default: (_loader: unknown, options?: { ssr?: boolean }) =>
    function DynamicResearchWorkspacePage() {
      return (
        <div
          data-testid="research-workspace-page"
          data-ssr={String(Boolean(options?.ssr))}
        />
      )
    }
}))

import ResearchWorkspacePage from "../../pages/research-workspace"

const pagesDir = join(__dirname, "../../pages")

describe("Research Workspace Next route files", () => {
  it("provides /research-workspace as the canonical page for the workspace surface", () => {
    render(<ResearchWorkspacePage />)

    expect(screen.getByTestId("research-workspace-page")).toHaveAttribute(
      "data-ssr",
      "false"
    )
  })

  it("does not expose legacy workspace route files", () => {
    expect(existsSync(join(pagesDir, "research-studio.tsx"))).toBe(false)
    expect(existsSync(join(pagesDir, "workspace-playground.tsx"))).toBe(false)
    expect(existsSync(join(pagesDir, "workspace-studio.tsx"))).toBe(false)
  })
})
