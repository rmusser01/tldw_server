import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { WorldBooksWorkspace } from "../WorldBooksWorkspace"

const { useLayoutUiStoreMock, navigateMock } = vi.hoisted(() => ({
  useLayoutUiStoreMock: vi.fn(),
  navigateMock: vi.fn()
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
  useNavigate: () => navigateMock
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasWorldBooks: true },
    loading: false
  })
}))

vi.mock("@/store/layout-ui", () => ({
  useLayoutUiStore: useLayoutUiStoreMock
}))

vi.mock("../Manager", () => ({
  WorldBooksManager: () => <div>World Books Manager</div>
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
      data-testid="world-books-page-shell"
      className={[maxWidthClassName, className].filter(Boolean).join(" ")}
    >
      {children}
    </div>
  )
}))

describe("WorldBooksWorkspace layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses full-width shell when the sidebar is collapsed", () => {
    useLayoutUiStoreMock.mockImplementation(
      (selector: (state: { chatSidebarCollapsed: boolean }) => boolean) =>
        selector({ chatSidebarCollapsed: true })
    )

    render(<WorldBooksWorkspace />)

    expect(screen.getByTestId("world-books-page-shell").className).toContain(
      "max-w-none"
    )
  })

  it("uses full-width shell when the sidebar is expanded", () => {
    useLayoutUiStoreMock.mockImplementation(
      (selector: (state: { chatSidebarCollapsed: boolean }) => boolean) =>
        selector({ chatSidebarCollapsed: false })
    )

    render(<WorldBooksWorkspace />)

    const shell = screen.getByTestId("world-books-page-shell")
    expect(shell.className).toContain("max-w-none")
  })
})
