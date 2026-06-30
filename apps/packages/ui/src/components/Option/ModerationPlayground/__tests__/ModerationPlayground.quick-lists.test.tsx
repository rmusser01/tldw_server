// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import * as moderationService from "@/services/moderation"

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
  testModeration: vi.fn(),
  getUserOverride: vi.fn(),
  setUserOverride: vi.fn(),
  deleteUserOverride: vi.fn()
}))

import { ModerationPlaygroundShell } from "../ModerationPlaygroundShell"

describe("ModerationPlayground quick phrase lists", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem("moderation-playground-onboarded", "true")

    vi.spyOn(moderationService, "getUserOverride").mockResolvedValue({
      exists: true,
      override: {}
    } as any)
    vi.spyOn(moderationService, "setUserOverride").mockResolvedValue({ persisted: true } as any)
    vi.spyOn(moderationService, "deleteUserOverride").mockResolvedValue({ status: "deleted" })
  })

  /**
   * Navigate to User Overrides tab, switch scope to user, enter ID, and load.
   * The context bar scope select + user ID input are always visible.
   * After loading, we switch to the User Overrides tab to see phrase lists.
   */
  const switchToUserScopeAndLoadUser = async () => {
    // Change scope to user via the context bar select
    const scopeSelect = screen.getByRole("combobox") as HTMLSelectElement
    fireEvent.change(scopeSelect, { target: { value: "user" } })

    // Enter user ID in context bar input
    const userInput = screen.getByPlaceholderText("Enter User ID")
    fireEvent.change(userInput, { target: { value: "alice" } })

    // Click Load button in context bar
    fireEvent.click(screen.getByRole("button", { name: /^load$/i }))

    // Navigate to User Overrides tab
    fireEvent.click(screen.getByRole("tab", { name: /overrides/i }))

    // Wait for the override to be loaded via the hook's useEffect
    await waitFor(() => {
      expect(moderationService.getUserOverride).toHaveBeenCalledWith("alice")
    })

    // Wait for the lazy-loaded panel to render
    await screen.findByText("User Override Editor")
  }

  const renderShell = () =>
    render(
      <MemoryRouter initialEntries={["/moderation/rules"]}>
        <ModerationPlaygroundShell />
      </MemoryRouter>
    )

  it("renders phrase composer in user overrides tab after loading a user", async () => {
    renderShell()
    await switchToUserScopeAndLoadUser()

    expect(screen.getByText("Add Phrase Rule")).toBeInTheDocument()
    expect(screen.getByTestId("phrase-pattern-input")).toBeInTheDocument()
  })

  it("adds ban and notify items to separate lists", async () => {
    renderShell()
    await switchToUserScopeAndLoadUser()

    const phraseInput = screen.getByTestId("phrase-pattern-input")

    // Add a banned phrase (default action is "block" = Ban)
    fireEvent.change(phraseInput, { target: { value: "danger phrase" } })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))

    // Switch to Notify action and add a notify phrase
    fireEvent.click(screen.getByRole("radio", { name: /notify/i }))
    fireEvent.change(phraseInput, { target: { value: "watch phrase" } })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))

    expect(screen.getByText("Banned Phrases")).toBeInTheDocument()
    expect(screen.getByText("Notify Phrases")).toBeInTheDocument()
    expect(screen.getByText("danger phrase")).toBeInTheDocument()
    expect(screen.getByText("watch phrase")).toBeInTheDocument()
  })

  it("prevents duplicate quick rules", async () => {
    renderShell()
    await switchToUserScopeAndLoadUser()

    const phraseInput = screen.getByTestId("phrase-pattern-input")

    fireEvent.change(phraseInput, { target: { value: "duplicate phrase" } })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))

    fireEvent.change(phraseInput, { target: { value: "duplicate phrase" } })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))

    expect(screen.getAllByText("duplicate phrase")).toHaveLength(1)
  })

  it("allows regex quick rule without browser syntax validation", async () => {
    renderShell()
    await switchToUserScopeAndLoadUser()

    // Check the Regex checkbox
    const regexCheckbox = screen.getByRole("checkbox", { name: /regex/i })
    fireEvent.click(regexCheckbox)

    fireEvent.change(screen.getByTestId("phrase-pattern-input"), {
      target: { value: "(" }
    })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))

    expect(screen.getByText("(")).toBeInTheDocument()
  })

  it("renders phase labels from loaded override rules", async () => {
    vi.spyOn(moderationService, "getUserOverride").mockResolvedValue({
      exists: true,
      override: {
        rules: [
          { id: "r1", pattern: "alpha", is_regex: false, action: "block", phase: "input" },
          { id: "r2", pattern: "beta", is_regex: false, action: "warn", phase: "output" }
        ]
      }
    } as any)

    renderShell()
    await switchToUserScopeAndLoadUser()

    // Phase labels are displayed as the raw phase value in the new panel
    expect(screen.getByText("input")).toBeInTheDocument()
    expect(screen.getByText("output")).toBeInTheDocument()
  })

  it("drops loaded rules with non-boolean is_regex values", async () => {
    vi.spyOn(moderationService, "getUserOverride").mockResolvedValue({
      exists: true,
      override: {
        rules: [{ id: "r1", pattern: "alpha", is_regex: "false", action: "block", phase: "both" }]
      }
    } as any)

    renderShell()
    await switchToUserScopeAndLoadUser()

    expect(screen.queryByText("alpha")).not.toBeInTheDocument()
  })

  it("includes rules in setUserOverride payload on save", async () => {
    renderShell()
    await switchToUserScopeAndLoadUser()

    fireEvent.change(screen.getByTestId("phrase-pattern-input"), {
      target: { value: "save me" }
    })
    fireEvent.click(screen.getByRole("button", { name: /add rule/i }))
    fireEvent.click(screen.getByRole("button", { name: /save override/i }))

    await waitFor(() => {
      expect(moderationService.setUserOverride).toHaveBeenCalled()
    })

    const call = vi.mocked(moderationService.setUserOverride).mock.calls.at(-1)
    expect(call?.[0]).toBe("alice")
    expect(call?.[1]).toMatchObject({
      rules: [
        expect.objectContaining({
          pattern: "save me",
          is_regex: false,
          action: "block",
          phase: "both"
        })
      ]
    })
  })
})
