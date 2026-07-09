import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/SettingsOptionLayout", () => ({
  SettingsLayout: ({ children }: { children: React.ReactNode }) => (
    <section data-testid="settings-layout">{children}</section>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )
}))

import { SettingsRoute } from "@web/extension/routes/settings-route"

describe("extension settings route", () => {
  it("provides one main landmark around settings content", () => {
    render(
      <SettingsRoute>
        <div>Extension settings</div>
      </SettingsRoute>
    )

    const main = screen.getByRole("main")

    expect(main).toContainElement(screen.getByTestId("settings-layout"))
    expect(main).toHaveTextContent("Extension settings")
  })
})
