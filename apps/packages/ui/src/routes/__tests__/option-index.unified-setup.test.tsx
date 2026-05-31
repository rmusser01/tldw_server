// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children, hideHeader, hideSidebar }: any) => (
    <main
      data-hide-header={String(Boolean(hideHeader))}
      data-hide-sidebar={String(Boolean(hideSidebar))}
    >
      {children}
    </main>
  )
}))

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({
    mode: "light",
    toggleDarkMode: vi.fn()
  })
}))

vi.mock("@/hooks/useComposerFocus", () => ({
  useFocusComposerOnConnect: vi.fn()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    checkOnce: vi.fn().mockResolvedValue(undefined),
    beginOnboarding: vi.fn(),
    markFirstRunComplete: vi.fn()
  }),
  useConnectionState: () => ({
    phase: "connected"
  }),
  useConnectionUxState: () => ({
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    loading: false,
    error: null
  })
}))

describe("OptionIndex unified setup resolver", () => {
  it("renders setup in focused shell when backend state is not complete", async () => {
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(screen.getByRole("main")).toHaveAttribute("data-hide-header", "true")
    expect(screen.getByRole("main")).toHaveAttribute("data-hide-sidebar", "true")
    expect(
      screen.getByRole("heading", { name: /first-time setup/i })
    ).toBeInTheDocument()
  })
})
