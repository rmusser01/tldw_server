// @vitest-environment jsdom
import React from "react"
import fs from "node:fs/promises"
import path from "node:path"
import { MemoryRouter } from "react-router-dom"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { getDesignSystemState } from "@/design-system"

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => <main>{children}</main>
}))

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: ({ onFinish }: { onFinish?: () => void }) => (
    <button type="button" onClick={onFinish}>
      Mock wizard
    </button>
  )
}))

describe("setup onboarding design-system state wiring", () => {
  it("frames setup with the canonical setup-required label and primary action", async () => {
    const { default: OptionSetup } = await import("@/routes/option-setup")

    render(
      <MemoryRouter>
        <OptionSetup />
      </MemoryRouter>
    )

    expect(
      screen.getByText(getDesignSystemState("setup_required").label)
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /start setup|connect/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Mock wizard" })).toBeInTheDocument()
  })

  it("documents canonical onboarding connection states in the form source", async () => {
    const source = await fs.readFile(
      path.resolve(
        process.cwd(),
        "../packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx"
      ),
      "utf8"
    )

    expect(source).toContain('getDesignSystemState("setup_required")')
    expect(source).toContain('getDesignSystemState("auth_required")')
    expect(source).toContain('getDesignSystemState("unavailable")')
    expect(source).toContain('getDesignSystemState("retrying")')
    expect(source).toContain('getDesignSystemState("ready")')
  })
})
