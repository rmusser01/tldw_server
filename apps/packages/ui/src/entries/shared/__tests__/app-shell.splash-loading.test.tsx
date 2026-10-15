// @vitest-environment jsdom
import React, { useState } from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/hooks/useTheme", () => ({ useTheme: () => ({ antdTheme: {} }) }))
vi.mock("@/components/Common/PageAssistProvider", () => ({
  PageAssistProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )
}))
vi.mock("@/context/FontSizeProvider", () => ({
  FontSizeProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )
}))
vi.mock("@/components/Common/LocaleJsonDiagnostics", () => ({
  LocaleJsonDiagnostics: () => null
}))
vi.mock("@/hooks/useSplashScreen", () => ({
  useSplashScreen: () => ({
    visible: true,
    card: { name: "default", effect: null },
    message: "Requested splash",
    show: vi.fn(),
    dismiss: vi.fn()
  })
}))

const Router = ({ children }: { children: React.ReactNode }) => <>{children}</>
const WorkingChat = () => {
  const [count, setCount] = useState(0)
  return (
    <button onClick={() => setCount(count + 1)}>Continue chat {count}</button>
  )
}
let waitForVisual: () => Promise<void>
const loadVisual = vi.fn()
beforeEach(() => {
  vi.resetModules()
  loadVisual.mockClear()
  waitForVisual = () => Promise.resolve()
  vi.doMock("@/components/Common/SplashScreen/SplashOverlay", async () => {
    loadVisual()
    await waitForVisual()
    return {
      default: () => <div role="dialog" aria-label="Requested splash" />
    }
  })
})

describe("extension optional splash visual", () => {
  it("keeps chat usable through a held visual load", async () => {
    let release!: () => void
    waitForVisual = () =>
      new Promise((resolve) => {
        release = resolve
      })
    const { AppShell } = await import("../AppShell")
    render(
      <AppShell router={Router} direction="ltr" emptyDescription="No data">
        <WorkingChat />
      </AppShell>
    )
    await waitFor(() => expect(loadVisual).toHaveBeenCalledOnce())
    fireEvent.click(screen.getByRole("button", { name: "Continue chat 0" }))
    expect(
      screen.getByRole("button", { name: "Continue chat 1" })
    ).toBeEnabled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    await act(async () => release())
    expect(
      await screen.findByRole("dialog", { name: "Requested splash" })
    ).toBeInTheDocument()
  })

  it("keeps chat usable and reports a failed visual load", async () => {
    const failure = new Error("Optional visual unavailable")
    waitForVisual = () => Promise.reject(failure)
    const warning = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined)
    try {
      const { AppShell } = await import("../AppShell")
      render(
        <AppShell router={Router} direction="ltr" emptyDescription="No data">
          <WorkingChat />
        </AppShell>
      )
      await waitFor(() =>
        expect(warning).toHaveBeenCalledWith(
          "Unable to load splash screen",
          expect.objectContaining({ cause: failure })
        )
      )
      fireEvent.click(screen.getByRole("button", { name: "Continue chat 0" }))
      expect(
        screen.getByRole("button", { name: "Continue chat 1" })
      ).toBeEnabled()
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    } finally {
      warning.mockRestore()
    }
  })
})
