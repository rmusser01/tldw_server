import React from "react"
import { ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PresentationExportMenu } from "../PresentationExportMenu"
import { usePresentationStudioStore } from "@/store/presentation-studio"

const mocks = vi.hoisted(() => ({ exportPresentation: vi.fn(), downloadBlob: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", async () => ({
  ...(await vi.importActual("@/services/tldw/TldwApiClient")),
  tldwClient: { exportPresentation: mocks.exportPresentation }
}))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: mocks.downloadBlob }))

const menu = (id = "saved-deck", isAuthorityCurrent = () => true) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <PresentationExportMenu
      presentationId={id}
      title="Rowan"
      isAuthorityCurrent={isAuthorityCurrent}
    />
  </ConfigProvider>
)
const renderMenu = (id = "saved-deck", isAuthorityCurrent = () => true) =>
  render(menu(id, isAuthorityCurrent))
const choose = async (label: string) => {
  fireEvent.click(screen.getByRole("button", { name: "Export saved presentation" }))
  fireEvent.click(await screen.findByRole("menuitem", { name: label }))
}
let savedSlides: ReturnType<typeof usePresentationStudioStore.getState>["slides"]

beforeEach(() => {
  vi.clearAllMocks()
  usePresentationStudioStore.getState().reset()
  usePresentationStudioStore.getState().initializeBlankProject()
  savedSlides = usePresentationStudioStore.getState().slides
  usePresentationStudioStore.setState({
    projectId: "saved-deck",
    title: "Rowan",
    etag: 'W/"v3"',
    slides: savedSlides,
    isDirty: false,
    autosaveState: "saved"
  })
  mocks.exportPresentation.mockReset()
  mocks.exportPresentation.mockResolvedValue(new Blob(["export"]))
})

describe("saved structured presentation export", () => {
  it.each([
    ["JSON", "json", "json"],
    ["Markdown", "markdown", "md"],
    ["Reveal.js (ZIP)", "revealjs", "zip"],
    ["PDF", "pdf", "pdf"]
  ])("downloads %s through the existing saved-deck API", async (label, format, ext) => {
    renderMenu()
    await choose(label)
    await waitFor(() =>
      expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), `Rowan.${ext}`)
    )
    expect(mocks.exportPresentation).toHaveBeenCalledExactlyOnceWith("saved-deck", format, {
      abortSignal: expect.any(AbortSignal)
    })
    expect(usePresentationStudioStore.getState().slides).toEqual(savedSlides)
  })

  it.each([{ isDirty: true }, { autosaveState: "saving" as const }])(
    "refuses an unsaved deck: %j",
    (state) => {
      usePresentationStudioStore.setState(state)
      renderMenu()
      expect(screen.getByRole("button", { name: "Export saved presentation" })).toBeDisabled()
      expect(screen.getByRole("status")).toHaveTextContent("Save changes before exporting")
      expect(mocks.exportPresentation).not.toHaveBeenCalled()
    }
  )

  it("blocks duplicate requests, retains slides on failure, and permits retry", async () => {
    let reject!: (error: Error) => void
    mocks.exportPresentation.mockReturnValueOnce(
      new Promise((_resolve, fail) => {
        reject = fail
      })
    )
    renderMenu()
    await choose("JSON")
    expect(screen.getByRole("button", { name: "Export saved presentation" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Export saved presentation" }))
    expect(mocks.exportPresentation).toHaveBeenCalledTimes(1)
    await act(async () => reject(new Error("private transport detail")))
    expect(screen.getByRole("alert")).toHaveTextContent("Your slides are unchanged")
    expect(screen.getByRole("alert")).not.toHaveTextContent("private transport detail")
    expect(usePresentationStudioStore.getState().slides).toEqual(savedSlides)
    await choose("JSON")
    await waitFor(() => expect(mocks.downloadBlob).toHaveBeenCalledTimes(1))
    expect(mocks.exportPresentation).toHaveBeenCalledTimes(2)
  })

  it("aborts on unmount and refuses the late response", async () => {
    let resolve!: (blob: Blob) => void
    mocks.exportPresentation.mockReturnValueOnce(
      new Promise((done) => {
        resolve = done
      })
    )
    const view = renderMenu()
    await choose("JSON")
    const signal = mocks.exportPresentation.mock.calls[0][2].abortSignal
    view.unmount()
    expect(signal.aborted).toBe(true)
    await act(async () => resolve(new Blob(["late private export"])))
    expect(mocks.downloadBlob).not.toHaveBeenCalled()
  })

  it("aborts an export when the mounted project changes", async () => {
    let resolve!: (blob: Blob) => void
    mocks.exportPresentation.mockReturnValueOnce(
      new Promise((done) => {
        resolve = done
      })
    )
    const view = renderMenu()
    await choose("JSON")
    const signal = mocks.exportPresentation.mock.calls[0][2].abortSignal
    act(() => usePresentationStudioStore.setState({ projectId: "different-deck" }))
    view.rerender(menu("different-deck"))
    expect(signal.aborted).toBe(true)
    await act(async () => resolve(new Blob(["old export"])))
    expect(mocks.downloadBlob).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Export saved presentation" })).toBeEnabled()
    await choose("JSON")
    await waitFor(() => expect(mocks.exportPresentation).toHaveBeenCalledTimes(2))
    expect(mocks.exportPresentation.mock.calls[1][0]).toBe("different-deck")
  })

  it("refuses a response after synchronous authority retirement before unmount", async () => {
    let current = true
    let resolve!: (blob: Blob) => void
    mocks.exportPresentation.mockReturnValueOnce(
      new Promise((done) => {
        resolve = done
      })
    )
    renderMenu("saved-deck", () => current)
    await choose("JSON")
    current = false
    await act(async () => resolve(new Blob(["retired private export"])))
    expect(mocks.downloadBlob).not.toHaveBeenCalled()
  })

  it("refuses an export superseded by a changed saved version", async () => {
    let resolve!: (blob: Blob) => void
    mocks.exportPresentation.mockReturnValueOnce(
      new Promise((done) => {
        resolve = done
      })
    )
    renderMenu()
    await choose("JSON")
    act(() => usePresentationStudioStore.setState({ etag: 'W/"v4"' }))
    await act(async () => resolve(new Blob(["old export"])))
    expect(mocks.downloadBlob).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toHaveTextContent("Export failed")
  })
})
