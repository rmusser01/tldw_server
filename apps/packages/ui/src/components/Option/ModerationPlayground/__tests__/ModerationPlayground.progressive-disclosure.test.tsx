// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: null, isFetching: false, error: null, refetch: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/moderation", () => ({
  getModerationSettings: vi.fn(),
  getEffectivePolicy: vi.fn(),
  reloadModeration: vi.fn(),
  listUserOverrides: vi.fn(),
  getManagedBlocklist: vi.fn().mockResolvedValue({
    data: { version: "v1", items: [] },
    etag: null
  }),
  testModeration: vi.fn(),
  getUserOverride: vi.fn(),
  setUserOverride: vi.fn(),
  deleteUserOverride: vi.fn()
}))

import { ModerationPlaygroundShell } from "../ModerationPlaygroundShell"

const renderShell = () =>
  render(
    <MemoryRouter>
      <ModerationPlaygroundShell />
    </MemoryRouter>
  )

describe("ModerationPlayground disclosure UX", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem("moderation-playground-onboarded")
  })

  it("allows dismissing onboarding and persists dismissal", async () => {
    renderShell()

    expect(
      screen.getByText("Welcome to Content Rules")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByText(/explore on my own/i))

    await waitFor(() => {
      expect(
        screen.queryByText("Welcome to Content Rules")
      ).not.toBeInTheDocument()
    })

    expect(localStorage.getItem("moderation-playground-onboarded")).toBe("true")
  })

  it("shows Policy & Settings tab content by default", async () => {
    localStorage.setItem("moderation-playground-onboarded", "true")
    renderShell()

    // The Policy tab should be selected by default
    const policyTab = screen.getByRole("tab", { name: /policy/i })
    expect(policyTab).toHaveAttribute("aria-selected", "true")

    // Policy panel content should be visible
    expect(await screen.findByText(/personal data protection/i)).toBeInTheDocument()
  })

  it("switches tab content when clicking a different tab", async () => {
    localStorage.setItem("moderation-playground-onboarded", "true")
    renderShell()

    // Click the Blocklist Studio tab
    fireEvent.click(screen.getByRole("tab", { name: /blocklist/i }))

    // Blocklist panel content should appear
    expect(await screen.findByText(/syntax reference/i)).toBeInTheDocument()

    // Policy tab should no longer be selected
    expect(screen.getByRole("tab", { name: /policy/i })).toHaveAttribute(
      "aria-selected",
      "false"
    )
  })

  it("defaults to server scope in the context bar", () => {
    localStorage.setItem("moderation-playground-onboarded", "true")
    renderShell()

    const serverOption = screen.getByRole("option", {
      name: /server/i
    }) as HTMLOptionElement
    expect(serverOption.selected).toBe(true)
  })

  it("renders all 5 tabs", () => {
    localStorage.setItem("moderation-playground-onboarded", "true")
    renderShell()

    expect(screen.getByRole("tab", { name: /policy/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /blocklist/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /overrides/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /test/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /advanced/i })).toBeInTheDocument()
  })

  it("navigates to Advanced tab and shows its content", async () => {
    localStorage.setItem("moderation-playground-onboarded", "true")
    renderShell()

    fireEvent.click(screen.getByRole("tab", { name: /advanced/i }))

    expect(screen.getByRole("tab", { name: /advanced/i })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    // Advanced panel should render (lazy loaded)
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument()
    })
  })
})
