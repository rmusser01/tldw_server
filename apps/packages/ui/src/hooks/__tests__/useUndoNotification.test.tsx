import React from "react"
import { App, ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useUndoNotification, type UndoNotificationOptions } from "../useUndoNotification"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback || _key })
}))

function UndoHarness(options: Pick<UndoNotificationOptions, "onUndo" | "onDismiss">) {
  const { showUndoNotification } = useUndoNotification()
  return <button onClick={() => showUndoNotification({ title: "Media deleted", duration: 0, ...options })}>Delete media</button>
}
const renderUndo = (options: Pick<UndoNotificationOptions, "onUndo" | "onDismiss">) => render(
  <ConfigProvider theme={{ token: { colorPrimary: "#123456" } }}>
    <App><UndoHarness {...options} /></App>
  </ConfigProvider>
)

describe("useUndoNotification with the application context", () => {
  afterEach(() => vi.restoreAllMocks())

  it("uses supported actions and restores exactly once even before the notification unmounts", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    let resolveRestore!: () => void
    const onUndo = vi.fn(() => new Promise<void>((resolve) => { resolveRestore = resolve }))
    const onDismiss = vi.fn()
    renderUndo({ onUndo, onDismiss })
    fireEvent.click(screen.getByRole("button", { name: "Delete media" }))
    const undo = await screen.findByRole("button", { name: "Undo", exact: true })
    act(() => { fireEvent.click(undo); fireEvent.click(undo) })
    expect(onUndo).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("Restored successfully")).not.toBeInTheDocument()
    await act(async () => resolveRestore())
    expect(await screen.findByText("Restored successfully")).toBeInTheDocument()
    expect(onDismiss).not.toHaveBeenCalled()
    expect(errors.mock.calls.flat().map(String).join("\n")).not.toMatch(/deprecated|cannot consume context/i)
  })

  it("dismisses without restoring when the visible Close action is used", async () => {
    const onUndo = vi.fn()
    const onDismiss = vi.fn()
    renderUndo({ onUndo, onDismiss })
    fireEvent.click(screen.getByRole("button", { name: "Delete media" }))
    fireEvent.click(await screen.findByRole("button", { name: /Close/i }))
    await waitFor(() => expect(onDismiss).toHaveBeenCalledTimes(1))
    expect(onUndo).not.toHaveBeenCalled()
  })

  it("shows the actual restore failure and never claims success", async () => {
    const onUndo = vi.fn(async () => { throw new Error("Restore request failed") })
    renderUndo({ onUndo })
    fireEvent.click(screen.getByRole("button", { name: "Delete media" }))
    fireEvent.click(await screen.findByRole("button", { name: "Undo", exact: true }))
    expect(await screen.findByText("Failed to restore")).toBeInTheDocument()
    expect(screen.getByText("Restore request failed")).toBeInTheDocument()
    expect(screen.queryByText("Restored successfully")).not.toBeInTheDocument()
    expect(onUndo).toHaveBeenCalledTimes(1)
  })
})
