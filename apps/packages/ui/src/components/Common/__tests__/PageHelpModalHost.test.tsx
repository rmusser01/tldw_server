import React from "react"
import { MemoryRouter } from "react-router-dom"
import { act, fireEvent, render as renderUI, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useTutorialStore } from "@/store/tutorials"
import { PageHelpModalHost } from "../PageHelpModalHost"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

const render = (ui: React.ReactNode) => renderUI(<MemoryRouter initialEntries={["/chat"]}>{ui}</MemoryRouter>)

const Help = () => <div role="dialog" aria-label="Help">Help content</div>
beforeEach(() => { useTutorialStore.getState().closeHelpModal() })
afterEach(() => vi.restoreAllMocks())

it("does not request a help chunk while closed", () => {
  const loadModal = vi.fn().mockResolvedValue({ PageHelpModal: Help })
  render(<PageHelpModalHost loadModal={loadModal} />)
  expect(loadModal).not.toHaveBeenCalled()
})

it.each(["tldw:open-help-modal", "tldw:open-shortcuts-modal"])("opens unloaded help through %s", async (eventName) => {
  const loadModal = vi.fn().mockResolvedValue({ PageHelpModal: Help })
  render(<PageHelpModalHost loadModal={loadModal} />)
  act(() => window.dispatchEvent(new CustomEvent(eventName)))
  expect(await screen.findByRole("dialog", { name: "Help" })).toBeVisible()
  expect(loadModal).toHaveBeenCalledOnce()
  fireEvent.keyDown(document, { key: "Escape" })
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
})

it("offers a confirmed page reload for a cached chunk failure without discarding the current page", async () => {
  // Turbopack retains the same rejected promise after a chunk load fails.
  let failChunk!: (error: Error) => void
  const chunk = new Promise<{ PageHelpModal: React.ComponentType }>((_resolve, reject) => { failChunk = reject })
  const loadModal = vi.fn(() => chunk)
  const reloadPage = vi.fn()
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
  render(<><input aria-label="Settings field" defaultValue="Keep this value" /><PageHelpModalHost loadModal={loadModal} reloadPage={reloadPage} /></>)
  const field = screen.getByRole("textbox", { name: "Settings field" })
  field.focus()
  act(() => useTutorialStore.getState().openHelpModal())
  await act(async () => failChunk(new Error("Failed to load chunk while offline")))
  expect(await screen.findByRole("alert")).toHaveTextContent("Help could not be loaded")
  expect(field).toHaveValue("Keep this value")
  expect(screen.queryByRole("button", { name: "Retry help" })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Reload page" }))
  expect(confirm).toHaveBeenCalledWith(expect.stringContaining("Unsaved edits may be lost"))
  expect(reloadPage).not.toHaveBeenCalled()
  expect(field).toHaveValue("Keep this value")
  confirm.mockReturnValue(true)
  fireEvent.click(screen.getByRole("button", { name: "Reload page" }))
  expect(reloadPage).toHaveBeenCalledOnce()
  expect(loadModal).toHaveBeenCalledOnce()
  fireEvent.keyDown(document, { key: "Escape" })
  expect(field).toHaveFocus()
})

it("dismisses a failed load without requesting another chunk", async () => {
  const loadModal = vi.fn().mockRejectedValue(new Error("offline"))
  render(<PageHelpModalHost loadModal={loadModal} />)
  act(() => useTutorialStore.getState().openHelpModal())
  fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
  expect(loadModal).toHaveBeenCalledOnce()
})


it.each(["resolve", "reject"])(
  "ignores a dismissed lazy load that later %ss and preserves the reopened intent",
  async (result) => {
    let resolveOld!: (value: { PageHelpModal: React.ComponentType }) => void
    let rejectOld!: (error: Error) => void
    let resolveCurrent!: (value: {
      PageHelpModal: React.ComponentType<{ initialTab?: string }>
    }) => void
    const old = new Promise<{ PageHelpModal: React.ComponentType }>(
      (resolve, reject) => {
        resolveOld = resolve
        rejectOld = reject
      }
    )
    const current = new Promise<{
      PageHelpModal: React.ComponentType<{ initialTab?: string }>
    }>((resolve) => {
      resolveCurrent = resolve
    })
    const loadModal = vi
      .fn()
      .mockReturnValueOnce(old)
      .mockReturnValueOnce(current)
    render(<PageHelpModalHost loadModal={loadModal} />)
    act(() => window.dispatchEvent(new CustomEvent("tldw:open-help-modal")))
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    await act(async () => {
      if (result === "resolve") resolveOld({ PageHelpModal: Help })
      else rejectOld(new Error("old offline result"))
      await Promise.resolve()
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    await act(async () =>
      resolveCurrent({
        PageHelpModal: ({ initialTab }) => (
          <div role="dialog" aria-label="Current help">
            {initialTab}
          </div>
        )
      })
    )
    expect(
      await screen.findByRole("dialog", { name: "Current help" })
    ).toHaveTextContent("shortcuts")
    expect(loadModal).toHaveBeenCalledTimes(2)
  }
)
