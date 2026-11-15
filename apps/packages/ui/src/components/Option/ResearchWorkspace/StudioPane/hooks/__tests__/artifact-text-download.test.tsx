import type { MessageInstance } from "antd/es/message/interface"
import type { TFunction } from "i18next"
import { renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { GeneratedArtifact } from "@/types/workspace"
import { useArtifactExport } from "../useArtifactExport"

const io = vi.hoisted(() => ({ output: vi.fn(), create: vi.fn(), revoke: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { downloadOutput: io.output } }))
vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: { getState: () => ({ workspaceId: "owned", generatedArtifacts: [] }) },
  hasResearchWorkspaceMigrationTombstone: () => true
}))

const messages = { error: vi.fn(), success: vi.fn() }
const StandardURL = URL
let files: Array<{ filename: string; href: string }>
const artifact = (patch: Partial<GeneratedArtifact> = {}): GeneratedArtifact => ({
  id: "local-card", workspaceArtifactId: "summary-canonical", serverId: "summary-canonical",
  type: "summary", title: "Chapter summaries", status: "completed", content: "Full saved text",
  createdAt: new Date("2026-11-15T00:00:00Z"), ...patch
})
const setup = () => renderHook(() => useArtifactExport({
  workspaceId: "owned", generatedArtifacts: [], messageApi: messages as unknown as MessageInstance,
  isMobile: false, removeArtifact: vi.fn(), restoreArtifact: vi.fn(), captureToCurrentNote: vi.fn(),
  t: ((_key: string, fallback: unknown) => fallback) as unknown as TFunction
}))
const blobText = (blob: Blob) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(String(reader.result))
  reader.onerror = () => reject(reader.error)
  reader.readAsText(blob)
})

beforeEach(() => {
  vi.clearAllMocks()
  files = []
  io.output.mockResolvedValue(new Blob(["Legacy downloaded output"]))
  io.create.mockReturnValue("blob:export")
  class TestURL extends StandardURL {
    static createObjectURL = io.create
    static revokeObjectURL = io.revoke
  }
  vi.stubGlobal("URL", TestURL)
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    files.push({ filename: this.download, href: this.href })
  })
})
afterEach(() => vi.unstubAllGlobals())

describe("Studio text artifact download routing", () => {
  it("exports the full hydrated canonical chapter artifact without calling the legacy Outputs endpoint", async () => {
    const content = "Exact source evidence — printed 319 / PDF 352.\n".repeat(130000)
    const saved = artifact({ content })
    await setup().result.current.handleDownloadArtifact(saved)
    expect(io.output).not.toHaveBeenCalled()
    const downloaded = io.create.mock.calls[0]?.[0] as Blob
    expect(downloaded.size).toBeGreaterThan(5 * 1024 * 1024)
    expect(await blobText(downloaded)).toBe(content)
    expect(files).toEqual([{ filename: "Chapter summaries.md", href: "blob:export" }])
    expect(io.revoke).toHaveBeenCalledExactlyOnceWith("blob:export")
  })

  it.each([undefined, "", "  \n  "])("reports unhydrated canonical text honestly (%j)", async content => {
    await setup().result.current.handleDownloadArtifact(artifact({ content, previewText: "Only a short preview" }))
    expect(io.output).not.toHaveBeenCalled()
    expect(files).toHaveLength(0)
    expect(messages.error).toHaveBeenCalledWith("Saved output text is not loaded. Reopen the output and retry the download.")
  })

  it.each([42, "42"])("keeps legacy numeric Outputs downloads compatible (%j)", async serverId => {
    await setup().result.current.handleDownloadArtifact(artifact({ workspaceArtifactId: undefined, serverId }))
    expect(io.output).toHaveBeenCalledExactlyOnceWith("42")
    expect(await blobText(io.create.mock.calls[0][0])).toBe("Legacy downloaded output")
  })

  it("does not fall back to local content after a legacy download permission error", async () => {
    io.output.mockRejectedValueOnce(Object.assign(new Error("Forbidden"), { status: 403 }))
    await setup().result.current.handleDownloadArtifact(artifact({ workspaceArtifactId: undefined, serverId: 42 }))
    expect(files).toHaveLength(0)
    expect(messages.error).toHaveBeenCalledWith("Download failed")
  })

  it("preserves specialized media output references for canonical artifacts", async () => {
    await setup().result.current.handleDownloadArtifact(artifact({ type: "infographic", exportRefs: [{ format: "png", fileId: 47, status: "ready" }] }))
    expect(io.output).toHaveBeenCalledExactlyOnceWith("47")
    expect(files[0].filename).toBe("Chapter summaries.png")
  })
})
