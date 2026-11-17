import { getDocumentUploadProblem } from "../document-utils"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
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
import GenerateDocumentDialog from "../GenerateDocumentDialog"
const api = vi.hoisted(() => ({
  generate: vi.fn(),
  render: vi.fn(),
  upload: vi.fn(),
  download: vi.fn(),
  current: true
}))
vi.mock("@/services/document-generation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/document-generation")>()),
  createDocumentClient: () => api
}))
vi.mock("@/services/tldw", () => ({ tldwClient: { uploadMedia: api.upload } }))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: api.download }))
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "local",
    selectedModelIsLoading: false
  })
}))
vi.mock("@/hooks/useChatModelCatalog", () => ({
  useChatModelCatalog: () => ({ isSuccess: true, isFetching: false })
}))
vi.mock("@/hooks/playground/modelSelectorUtils", () => ({
  isConfiguredUsableModel: () => true,
  resolveModelSelectorSelection: () => ({
    modelId: "local",
    provider: "llamacpp",
    model: {}
  })
}))
vi.mock("@/components/Common/ModelSelect", () => ({
  ModelSelect: () => <span>Configured model</span>
}))
const draft = {
  title: "Rowan",
  pages: [{ heading: "Telescope", text: "The telescope is Selene." }]
}
const operation = {
  authorityKey: "owner",
  authorityRevision: 1,
  isCurrent: () => api.current,
  assertCurrent: vi.fn(),
  signal: new AbortController().signal,
  requestScope: {
    config: { serverUrl: "http://owner.invalid", authMode: "multi-user" },
    userId: "1"
  }
} satisfies QuickIngestOperation
const draw = (onOpenSaved = vi.fn().mockResolvedValue(true)) =>
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <GenerateDocumentDialog
        operation={operation}
        onClose={vi.fn()}
        onLibrary={vi.fn()}
        onOpenSaved={onOpenSaved}
      />
    </ConfigProvider>
  )
const generate = async () => {
  fireEvent.change(screen.getByLabelText("Document instructions"), {
    target: { value: "Rowan observatory" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Generate draft" }))
  await screen.findByLabelText("Document title")
}
beforeEach(() => {
  vi.clearAllMocks()
  api.current = true
  api.generate.mockResolvedValue(structuredClone(draft))
  api.render.mockResolvedValue(
    new Blob(["%PDF-complete"], { type: "application/pdf" })
  )
  api.upload.mockResolvedValue({
    results: [{ media_id: 71, status: "Success", original_file_stored: true }]
  })
})
afterEach(cleanup)
it("saves edited text once and retries only opening after a known save acknowledgement", async () => {
  const opened = vi
    .fn()
    .mockResolvedValueOnce(false)
    .mockResolvedValueOnce(true)
  draw(opened)
  await generate()
  fireEvent.change(screen.getByLabelText("Page 1 text"), {
    target: { value: "The telescope is named Selene." }
  })
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await screen.findByRole("button", { name: "Open saved document" })
  expect(api.render.mock.calls[0][0].pages[0].text).toBe(
    "The telescope is named Selene."
  )
  fireEvent.click(screen.getByRole("button", { name: "Open saved document" }))
  await waitFor(() => expect(opened).toHaveBeenCalledTimes(2))
  expect(api.upload).toHaveBeenCalledTimes(1)
  expect(api.upload.mock.calls[0][1]).toMatchObject({
    keep_original_file: true,
    media_type: "pdf"
  })
})
it("does not replay an uncertain upload and retains the PDF for reconciliation", async () => {
  api.upload.mockRejectedValue(new Error("connection lost"))
  draw()
  await generate()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await screen.findByText(/Check Library before creating another copy/)
  expect(screen.getByRole("button", { name: /Save PDF$/ })).toBeDisabled()
  api.current = false
  fireEvent.click(screen.getByRole("button", { name: "Download retained PDF" }))
  expect(api.download).not.toHaveBeenCalled()
  api.current = true
  fireEvent.click(screen.getByRole("button", { name: "Download retained PDF" }))
  expect(api.upload).toHaveBeenCalledTimes(1)
  expect(api.download).toHaveBeenCalledTimes(1)
})
it("Stop preserves the previous editable draft and ignores late generation", async () => {
  draw()
  await generate()
  let release!: (value: unknown) => void
  api.generate.mockReturnValue(
    new Promise((resolve) => {
      release = resolve
    })
  )
  fireEvent.click(screen.getByRole("button", { name: "Generate draft" }))
  fireEvent.click(screen.getByRole("button", { name: "Stop" }))
  await act(async () => {
    release({ ...draft, title: "Late foreign draft" })
    await Promise.resolve()
  })
  expect(screen.getByLabelText("Document title")).toHaveValue("Rowan")
  expect(api.upload).not.toHaveBeenCalled()
})

it("keeps pending Save protected and sends exactly one upload", async () => {
  let finish!: (value: unknown) => void
  api.upload.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve
    })
  )
  draw()
  await generate()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await waitFor(() => expect(api.upload).toHaveBeenCalledTimes(1))
  expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  expect(api.upload).toHaveBeenCalledTimes(1)
  await act(async () => {
    finish({
      results: [{ media_id: 71, status: "Success", original_file_stored: true }]
    })
    await Promise.resolve()
  })
})
it("retains an editable draft after rendering fails without uploading", async () => {
  api.render.mockRejectedValue(new Error("Shorten the page before saving."))
  draw()
  await generate()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await screen.findByText("Shorten the page before saving.")
  await waitFor(() =>
    expect(screen.getByLabelText("Page 1 text")).not.toBeDisabled()
  )
  expect(screen.getByLabelText("Page 1 text")).toHaveValue(draft.pages[0].text)
  expect(api.upload).not.toHaveBeenCalled()
})
it("keeps a known save with a missing original out of automatic upload/open retries", async () => {
  api.upload.mockResolvedValue({
    results: [{ media_id: 71, status: "Success", original_file_stored: false }]
  })
  const opened = vi.fn()
  draw(opened)
  await generate()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await screen.findByText(/original PDF was not preserved/)
  expect(
    screen.queryByRole("button", { name: "Open saved document" })
  ).toBeNull()
  expect(screen.getByRole("button", { name: /Save PDF$/ })).toBeDisabled()
  expect(api.upload).toHaveBeenCalledTimes(1)
  expect(opened).not.toHaveBeenCalled()
})
it("aborts an abandoned generation before its late result can save anything", async () => {
  let finish!: (value: unknown) => void
  api.generate.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve
    })
  )
  const view = draw()
  fireEvent.change(screen.getByLabelText("Document instructions"), {
    target: { value: "Rowan" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Generate draft" }))
  const signal = api.generate.mock.calls[0][1] as AbortSignal
  view.unmount()
  expect(signal.aborted).toBe(true)
  await act(async () => {
    finish(draft)
    await Promise.resolve()
  })
  expect(api.upload).not.toHaveBeenCalled()
})

it("does not treat a partial Error response with a retained Media ID as successful Save", async () => {
  api.upload.mockResolvedValue({
    results: [
      {
        status: "Error",
        db_id: 71,
        error: "Chunk consistency validation failed",
        warnings: []
      }
    ]
  })
  const opened = vi.fn()
  draw(opened)
  await generate()
  fireEvent.click(screen.getByRole("button", { name: /Save PDF$/ }))
  await screen.findByText(/Import failed/)
  expect(opened).not.toHaveBeenCalled()
  expect(screen.getByRole("button", { name: /Save PDF$/ })).toBeDisabled()
  expect(
    screen.queryByRole("button", { name: "Open saved document" })
  ).toBeNull()
  expect(api.upload).toHaveBeenCalledTimes(1)
})

it.each([
  [{ status: "Warning" }, "warnings"],
  [{ status: "Cancelled" }, "cancelled"],
  [null, "not confirmed"],
  [{ status: "Success", error: "validation failed" }, "error"]
])(
  "retains a non-success import outcome for reconciliation",
  (result, message) => {
    expect(getDocumentUploadProblem(result)).toContain(message)
  }
)
