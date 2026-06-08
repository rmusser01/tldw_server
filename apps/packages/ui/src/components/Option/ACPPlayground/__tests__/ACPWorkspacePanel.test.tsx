import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { useACPSessionsStore } from "@/store/acp-sessions"
import { ACPWorkspacePanel } from "../ACPWorkspacePanel"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: null
  })
}))

describe("ACPWorkspacePanel canonical Workspace handoff", () => {
  beforeEach(() => {
    useACPSessionsStore.getState().reset()
  })

  it("links no-session users to the canonical Workspaces manager", () => {
    render(<ACPWorkspacePanel />)

    expect(
      screen.getByText(/Select a session to open the workspace terminal/i)
    ).toBeTruthy()
    expect(
      screen.getByText(/Use Workspaces to create or attach the project root/i)
    ).toBeTruthy()
    expect(
      screen.getByRole("link", { name: /manage canonical Workspaces/i })
    ).toHaveAttribute("href", "#/workspaces")
  })
})
