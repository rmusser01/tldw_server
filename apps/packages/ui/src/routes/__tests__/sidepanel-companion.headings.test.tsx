// @vitest-environment jsdom
/**
 * AX-14: one h1 per side-panel page. Companion Home renders its own visible
 * "Companion" h1 on every branch, so the side-panel header must not add a
 * second, hidden one.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: vi.fn((path: string) => path) },
    tabs: { create: vi.fn(() => Promise.resolve({ id: 1 })) }
  }
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue || key
  })
}))
vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
vi.mock("@/hooks/useMessage", () => ({ useMessage: () => ({ temporaryChat: false }) }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: { hasPersona: true, hasPersonalization: true, hasCompanion: true }
  })
}))
vi.mock("~/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => null
}))
vi.mock("@/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => null
}))
vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
// Companion Home's page renders this h1 on its loading and loaded branches
// (CompanionHomePage.tsx).
vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: () => <h1>Companion</h1>
}))
vi.mock("@/services/companion", () => ({
  isCompanionConsentRequiredError: () => false,
  recordExplicitCompanionCapture: vi.fn(async () => null)
}))

import SidepanelCompanion from "../sidepanel-companion"

describe("side-panel Companion headings", () => {
  it("has exactly one h1", () => {
    render(
      <MemoryRouter>
        <SidepanelCompanion />
      </MemoryRouter>
    )

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
  })
})
