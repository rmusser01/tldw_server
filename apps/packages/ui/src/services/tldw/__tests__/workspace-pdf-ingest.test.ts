import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { webcrypto } from "node:crypto"
import { ingestWorkspacePdf, listWorkspacePdfUploads } from "../workspace-pdf-ingest"
import { servicePromptAuthorityKey } from "../domains/service-prompts"

const io = vi.hoisted(() => ({ upload: vi.fn(), request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgUpload: io.upload, bgRequest: io.request }))
const requestScope = { config: { serverUrl: "https://owned.test", authMode: "multi-user" as const }, userId: 1 }
const context = () => ({ workspaceId: "workspace", requestScope, assertCurrent: vi.fn() })
const file = () => Object.assign(new File(["PDF"], "book.pdf", { type: "application/pdf" }), { arrayBuffer: async () => new TextEncoder().encode("PDF").buffer })

beforeEach(() => { vi.resetAllMocks(); sessionStorage.clear(); vi.stubGlobal("crypto", webcrypto) })
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

describe("workspace PDF Jobs uploads", () => {
  it.each([["Extracting PDF pages", "Extracting PDF pages"], ["process", "Extracting and preparing PDF passages…"]])("waits beyond 60 seconds, retains originals and chunks, and explains phase %s", async (phase, displayed) => {
    vi.useFakeTimers()
    const start = Date.now()
    io.upload.mockResolvedValue({ batch_id: "batch", jobs: [{ id: 7 }] })
    io.request.mockImplementation(async () => Date.now() - start < 120000
      ? { status: "processing", progress_message: phase, progress_percent: null }
      : { status: "completed", result: { status: "Success", media_id: 92, title: "Book" } })
    const progress = vi.fn()
    const result = ingestWorkspacePdf({ ...context(), file: file(), onProgress: progress })
    await vi.waitFor(() => expect(io.upload).toHaveBeenCalledTimes(1))
    await vi.advanceTimersByTimeAsync(60001)
    expect(io.upload).toHaveBeenCalledTimes(1)
    expect(progress).toHaveBeenCalledWith(expect.objectContaining({ status: "processing", message: displayed }))
    await vi.advanceTimersByTimeAsync(61000)
    await expect(result).resolves.toMatchObject({ mediaId: 92 })
    expect(io.upload.mock.calls[0][0]).toMatchObject({ path: "/api/v1/media/ingest/jobs", fields: { keep_original_file: true, perform_chunking: true, perform_analysis: false, overwrite: false } })
  })

  it("reconnects after a failed status read without uploading again", async () => {
    io.upload.mockResolvedValue({ batch_id: "batch", jobs: [{ id: 7 }] })
    io.request.mockRejectedValueOnce(new Error("offline"))
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).rejects.toThrow("offline")
    const [receipt] = listWorkspacePdfUploads(context())
    io.request.mockResolvedValue({ status: "completed", result: { status: "Success", media_id: 92 } })
    await expect(ingestWorkspacePdf({ ...context(), receipt })).resolves.toMatchObject({ mediaId: 92 })
    expect(io.upload).toHaveBeenCalledTimes(1)
  })

  it("renders old saved raw phase messages in readable language", () => {
    const key = 'tldw:workspace-pdf-ingest:v1:workspace:'
    const receipt = { id: "saved", fileName: "book.pdf", fileSize: 4, status: "processing", message: "process" }
    sessionStorage.setItem(key + servicePromptAuthorityKey(requestScope), JSON.stringify([receipt]))
    expect(listWorkspacePdfUploads(context())[0]?.message).toBe("Extracting and preparing PDF passages…")
  })

  it("does not blindly retry an upload whose acceptance is unknown", async () => {
    io.upload.mockRejectedValue(new Error("connection lost during submit"))
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).rejects.toThrow()
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).rejects.toThrow(/may still be running/)
    expect(io.upload).toHaveBeenCalledTimes(1)
  })

  it("rejects completed jobs containing failed results and permits a fresh upload after known failure", async () => {
    io.upload.mockResolvedValue({ batch_id: "batch", jobs: [{ id: 7 }] })
    io.request.mockResolvedValueOnce({ status: "completed", result: { status: "Error", error: "Extraction failed" } })
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).rejects.toThrow("Extraction failed")
    io.request.mockResolvedValue({ status: "completed", result: { status: "Success", media_id: 92 } })
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).resolves.toMatchObject({ mediaId: 92 })
    expect(io.upload).toHaveBeenCalledTimes(2)
  })

  it("keeps receipts isolated by owner and workspace and blocks stale scope completion", async () => {
    const ctx = context()
    io.upload.mockResolvedValue({ batch_id: "batch", jobs: [{ id: 7 }] })
    io.request.mockImplementation(async () => { ctx.assertCurrent.mockImplementation(() => { throw new DOMException("Changed", "AbortError") }); return { status: "completed", result: { media_id: 92 } } })
    await expect(ingestWorkspacePdf({ ...ctx, file: file() })).rejects.toMatchObject({ name: "AbortError" })
    expect(listWorkspacePdfUploads({ ...context(), workspaceId: "other" })).toEqual([])
    expect(listWorkspacePdfUploads({ ...context(), requestScope: { ...requestScope, userId: 2 } })).toEqual([])
    expect(JSON.stringify(listWorkspacePdfUploads(context()))).not.toContain("accessToken")
  })

  it.each(["server", "owner", "workspace", "job"])("refuses an explicit receipt from another %s before polling", async boundary => {
    io.upload.mockResolvedValue({ batch_id: "batch", jobs: [{ id: 7 }] })
    io.request.mockRejectedValueOnce(new Error("offline"))
    await expect(ingestWorkspacePdf({ ...context(), file: file() })).rejects.toThrow("offline")
    const [saved] = listWorkspacePdfUploads(context())
    const changed = context()
    if (boundary === "server") changed.requestScope = { ...requestScope, config: { ...requestScope.config, serverUrl: "https://other.test" } }
    if (boundary === "owner") changed.requestScope = { ...requestScope, userId: 2 }
    if (boundary === "workspace") changed.workspaceId = "other-workspace"
    const receipt = boundary === "job" ? { ...saved, tracking: { ...saved.tracking!, jobIds: [8] } } : saved
    io.request.mockClear().mockResolvedValue({ status: "completed", result: { media_id: 999 } })
    await expect(ingestWorkspacePdf({ ...changed, receipt })).rejects.toThrow(/different connection or workspace/)
    expect(io.request).not.toHaveBeenCalled()
    expect(io.upload).toHaveBeenCalledTimes(1)
  })
})
