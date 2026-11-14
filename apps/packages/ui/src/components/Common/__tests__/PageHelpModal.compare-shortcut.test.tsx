/**
 * CM-05 (#3101), owner decision D4: Compare stays hidden in /chat until it is
 * verified, so the shortcuts reference must not list Alt+Shift+C while the
 * compare feature is off.
 */
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { PageHelpModalHost } from "../PageHelpModalHost"
import { useTutorialStore } from "@/store/tutorials"

const flags = vi.hoisted(() => ({ compareEnabled: false }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("@/hooks/useFeatureFlags", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/hooks/useFeatureFlags")>()
  return {
    ...actual,
    useFeatureFlag: (flag: string) =>
      flag === actual.FEATURE_FLAGS.COMPARE_MODE
        ? ([flags.compareEnabled, vi.fn(), true] as const)
        : actual.useFeatureFlag(flag as never)
  }
})

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/hooks/keyboard/useKeyboardShortcuts")
  >()),
  isMac: false
}))

vi.mock("@/tutorials", () => ({
  getTutorialsForRoute: () => []
}))

const openShortcuts = async () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <PageHelpModalHost />
    </MemoryRouter>
  )
  act(() => window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal")))
  return screen.findByRole("dialog", { name: "Page Help" })
}

describe("PageHelpModal compare shortcut", () => {
  beforeEach(() => {
    useTutorialStore.getState().closeHelpModal()
  })

  it("does not list Alt+Shift+C while compare is unavailable", async () => {
    flags.compareEnabled = false
    const dialog = await openShortcuts()

    expect(within(dialog).getByText("Open mode launcher")).toBeVisible()
    expect(within(dialog).queryByText("Alt + Shift + C")).not.toBeInTheDocument()
    expect(
      within(dialog).queryByText(/Toggle compare mode/i)
    ).not.toBeInTheDocument()
  })

  it("lists Alt+Shift+C once compare is enabled", async () => {
    flags.compareEnabled = true
    const dialog = await openShortcuts()

    expect(within(dialog).getByText("Alt + Shift + C")).toBeVisible()
  })
})
