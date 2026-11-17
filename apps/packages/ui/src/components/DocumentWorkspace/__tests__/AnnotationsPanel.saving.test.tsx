import type {} from "@testing-library/jest-dom/vitest"
import React from "react"
import { ConfigProvider } from "antd"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AnnotationsPanel } from "../RightPanel/AnnotationsPanel"
import { useDocumentWorkspaceStore as store } from "@/store/document-workspace"
const hooks = vi.hoisted(() => ({
  remove: vi.fn(),
  update: vi.fn(),
  pending: false
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))
vi.mock("@/hooks/document-workspace/useDocumentOperation", () => ({
  useDocumentOperation: () => owner
}))
const owner = { isCurrent: () => true }
vi.mock("@/hooks/document-workspace", () => ({
  useAnnotations: () => ({ isLoading: false, error: null }),
  useUpdateAnnotation: () => ({
    mutateAsync: hooks.update,
    isPending: hooks.pending
  }),
  useDeleteAnnotation: () => ({ mutateAsync: hooks.remove, isPending: false })
}))
vi.mock("../RightPanel/ExportAnnotationsModal", () => ({
  ExportAnnotationsModal: () => null
}))
const row = (id: string, documentId: number) => ({
  id,
  documentId,
  location: 1,
  text: `Text ${id}`,
  note: `Note ${id}`,
  color: "yellow" as const,
  createdAt: new Date(0),
  updatedAt: new Date(0)
})
const draw = () =>
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <AnnotationsPanel />
    </ConfigProvider>
  )
beforeEach(() => {
  vi.clearAllMocks()
  hooks.pending = false
  store.getState().reset()
  store.setState({
    activeDocumentId: 1,
    activeDocumentType: "pdf",
    annotations: [row("saved", 1)]
  })
})
afterEach(() => {
  cleanup()
  store.getState().reset()
  vi.clearAllTimers()
  vi.useRealTimers()
})
describe("canonical annotation actions", () => {
  it("shows only the active document's annotations", () => {
    store.setState({ annotations: [row("saved", 1), row("other", 2)] })
    draw()
    expect(screen.queryByText('"Text other"')).toBeNull()
  })
  it("disables editing and deletion until creation is acknowledged", () => {
    const pending = row("pending", 1)
    store.setState({ annotations: [pending], pendingAnnotations: [pending] })
    draw()
    expect(
      [
        screen.getByRole("button", { name: "Edit" }),
        screen.getByRole("button", { name: "Delete" })
      ].map((button) => (button as HTMLButtonElement).disabled)
    ).toEqual([true, true])
  })
  it("requires confirmation and retains the row until deletion succeeds", async () => {
    vi.useFakeTimers()
    let finish!: () => void
    hooks.remove.mockReturnValue(
      new Promise<void>((resolve) => {
        finish = resolve
      })
    )
    const view = draw()
    fireEvent.click(
      view.container.querySelector("svg.lucide-trash-2")!.closest("button")!
    )
    expect(store.getState().annotations.map((a) => a.id)).toEqual(["saved"])
    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveTextContent("Delete annotation")
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Delete" })
    )
    expect(hooks.remove).toHaveBeenCalledWith({
      mediaId: 1,
      annotationId: "saved"
    })
    expect(store.getState().annotations).toHaveLength(1)
    await act(async () => {
      finish()
      await Promise.resolve()
    })
  })
  it("protects a held note-save snapshot from further typing", () => {
    const view = draw()
    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    hooks.pending = true
    view.rerender(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <AnnotationsPanel />
      </ConfigProvider>
    )
    expect(
      within(screen.getByRole("dialog")).getByRole("textbox")
    ).toBeDisabled()
  })
  it.each(["success", "failure"])(
    "does not let an old note %s close or alter a replacement dialog",
    async (outcome) => {
      let finish!: (value?: unknown) => void, fail!: (error: Error) => void
      hooks.update.mockReturnValue(
        new Promise((resolve, reject) => {
          finish = resolve
          fail = reject
        })
      )
      draw()
      fireEvent.click(screen.getByRole("button", { name: "Edit" }))
      fireEvent.click(
        within(screen.getByRole("dialog")).getByRole("button", {
          name: "Save",
        })
      )
      act(() =>
        store.setState({
          activeDocumentId: 2,
          annotations: [row("replacement", 2)]
        })
      )
      fireEvent.click(screen.getByRole("button", { name: "Edit" }))
      await act(async () => {
        if (outcome === "success") finish()
        else fail(new Error("old failure"))
        await Promise.resolve()
      })
      expect(
        within(screen.getByRole("dialog")).getByRole("textbox")
      ).toHaveValue("Note replacement")
      expect(screen.queryByRole("alert")).toBeNull()
    }
  )
})
