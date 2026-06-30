import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { DictionariesWorkspace } from "../DictionariesWorkspace"

const { useLayoutUiStoreMock } = vi.hoisted(() => ({
  useLayoutUiStoreMock: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  location: {
    pathname: "/dictionaries",
    search: "",
    hash: "",
    state: null,
    key: "dictionaries-layout-test"
  }
}))

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

const demoModeMocks = vi.hoisted(() => ({
  useDemoMode: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => routerMocks.location,
  useNavigate: () => routerMocks.navigate
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => demoModeMocks.useDemoMode()
}))

vi.mock("@/store/layout-ui", () => ({
  useLayoutUiStore: useLayoutUiStoreMock
}))

vi.mock("../Manager", () => ({
  DictionariesManager: () => <div>Dictionaries Manager</div>
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({
    children,
    maxWidthClassName,
    className
  }: {
    children: React.ReactNode
    maxWidthClassName?: string
    className?: string
  }) => (
    <div
      data-testid="dictionaries-page-shell"
      className={[maxWidthClassName, className].filter(Boolean).join(" ")}
    >
      {children}
    </div>
  )
}))

describe("DictionariesWorkspace layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    demoModeMocks.useDemoMode.mockReturnValue({
      demoEnabled: false
    })
  })

  it("uses full-width shell when the sidebar is collapsed", () => {
    useLayoutUiStoreMock.mockImplementation(
      (selector: (state: { chatSidebarCollapsed: boolean }) => boolean) =>
        selector({ chatSidebarCollapsed: true })
    )

    render(<DictionariesWorkspace />)

    expect(screen.getByTestId("dictionaries-page-shell").className).toContain(
      "max-w-none"
    )
  })

  it("uses bounded shell when the sidebar is expanded", () => {
    useLayoutUiStoreMock.mockImplementation(
      (selector: (state: { chatSidebarCollapsed: boolean }) => boolean) =>
        selector({ chatSidebarCollapsed: false })
    )

    render(<DictionariesWorkspace />)

    const shell = screen.getByTestId("dictionaries-page-shell")
    expect(shell.className).toContain("max-w-5xl")
    expect(shell.className).not.toContain("max-w-none")
  })

  it("preserves demo copy when connection state is in demo mode", () => {
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "demo_mode",
      hasCompletedFirstRun: true
    })
    demoModeMocks.useDemoMode.mockReturnValue({
      demoEnabled: true
    })

    render(<DictionariesWorkspace />)

    expect(
      screen.getByText("Explore Chat dictionaries in demo mode")
    ).toBeInTheDocument()
  })

  it("shows auth guidance instead of generic not-connected copy", () => {
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "error_auth",
      hasCompletedFirstRun: true
    })

    render(<DictionariesWorkspace />)

    expect(
      screen.getByText(
        "Add your credentials before Chat dictionaries can load data."
      )
    ).toBeInTheDocument()
  })
})
