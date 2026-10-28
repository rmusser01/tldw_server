import type { Annotation } from "../types"
import type {} from "@testing-library/jest-dom/vitest"
import React from "react"
import { ConfigProvider } from "antd"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { ExportAnnotationsModal } from "../RightPanel/ExportAnnotationsModal"
import { useDocumentWorkspaceStore as store } from "@/store/document-workspace"
const api = vi.hoisted(() => ({
  exportPdf: vi.fn(),
  download: vi.fn(),
  annotationDownload: vi.fn(),
  current: true
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))
vi.mock("@/hooks/document-workspace/useDocumentOperation", () => ({
  useDocumentOperation: () => owner
}))
vi.mock("@/services/document-generation", () => ({
  createDocumentClient: () => api
}))
vi.mock("../utils/annotationExport", () => ({
  downloadAnnotations: api.annotationDownload
}))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: api.download }))
const owner = {
  isCurrent: () => api.current,
  signal: new AbortController().signal,
  assertCurrent: vi.fn()
}
const row: Annotation = {
  id: "saved",
  documentId: 1,
  location: 1,
  text: "Selene",
  note: "Friday",
  color: "blue",
  createdAt: new Date(),
  updatedAt: new Date()
}
const form = (type: "pdf" | "epub" = "pdf", open = true) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <ExportAnnotationsModal
      open={open}
      onClose={vi.fn()}
      annotations={[row]}
      documentTitle="Rowan"
      documentType={type}
      documentId={1}
    />
  </ConfigProvider>
)
const draw = (type: "pdf" | "epub" = "pdf") => render(form(type))
beforeEach(() => {
  vi.clearAllMocks()
  api.current = true
  store.getState().reset()
  api.exportPdf.mockResolvedValue(
    new Blob(["%PDF-complete"], { type: "application/pdf" })
  )
})
afterEach(() => {
  cleanup()
  store.getState().reset()
})
it("downloads canonical saved PDF annotations instead of rebuilding them from client rows", async () => {
  draw()
  fireEvent.click(screen.getByText("PDF with saved annotations"))
  fireEvent.click(screen.getByRole("button", { name: "Export" }))
  await waitFor(() => expect(api.download).toHaveBeenCalledTimes(1))
  expect(api.exportPdf.mock.calls[0][0]).toBe(1)
  expect(api.exportPdf.mock.calls[0]).toHaveLength(2)
})
it("blocks saved-PDF export while this document has unsaved annotations", () => {
  store.setState({ pendingAnnotations: [row] })
  draw()
  fireEvent.click(screen.getByText("PDF with saved annotations"))
  expect(screen.getByRole("button", { name: "Export" })).toBeDisabled()
  expect(api.exportPdf).not.toHaveBeenCalled()
})
it("does not download a late PDF after its owner retires", async () => {
  let finish!: (value: Blob) => void
  api.exportPdf.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve
    })
  )
  draw()
  fireEvent.click(screen.getByText("PDF with saved annotations"))
  fireEvent.click(screen.getByRole("button", { name: "Export" }))
  api.current = false
  await act(async () => {
    finish(new Blob(["%PDF"]))
    await Promise.resolve()
  })
  expect(api.download).not.toHaveBeenCalled()
})
it("keeps EPUB annotation-only export available without a misleading PDF option", async () => {
  draw("epub")
  await waitFor(() => expect(screen.getByRole("dialog")).toBeVisible())
  expect(screen.queryByText("PDF with saved annotations")).toBeNull()
  expect(screen.getByText("Markdown")).toBeVisible()
})

it("releases busy state when a held export is closed and reopened", async () => {
  let finish!: (value: Blob) => void
  api.exportPdf.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve
    })
  )
  const view = draw()
  fireEvent.click(screen.getByText("PDF with saved annotations"))
  fireEvent.click(screen.getByRole("button", { name: "Export" }))
  view.rerender(form("pdf", false))
  view.rerender(form("pdf", true))
  await act(async () => {
    finish(new Blob(["%PDF"]))
    await Promise.resolve()
  })
  expect(screen.getByRole("button", { name: /Export$/ })).not.toBeDisabled()
  expect(api.download).not.toHaveBeenCalled()
})

it("refuses local annotation downloads immediately after owner retirement without a rerender", () => {
  draw()
  api.current = false
  fireEvent.click(screen.getByRole("button", { name: "Export" }))
  expect(api.annotationDownload).not.toHaveBeenCalled()
})
