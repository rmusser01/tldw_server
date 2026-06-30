import { afterEach, describe, expect, it, vi } from "vitest"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"

const originalEnv = { ...process.env }

afterEach(() => {
  process.env = { ...originalEnv }
  vi.resetModules()
  vi.restoreAllMocks()
})

describe("real-server extension launch wrappers", () => {
  it("does not force a default timeout for launchWithExtensionOrSkip", async () => {
    const launchWithExtension = vi.fn().mockResolvedValue({ ok: true })

    vi.doMock("./extension", () => ({
      launchWithExtension,
    }))

    const { launchWithExtensionOrSkip } = await import("./real-server")
    const test = { skip: vi.fn() } as any

    await launchWithExtensionOrSkip(test, "/tmp/ext")

    expect(launchWithExtension).toHaveBeenCalledWith("/tmp/ext", {})
    expect(test.skip).not.toHaveBeenCalled()
  })

  it("does not force a default timeout for launchWithBuiltExtensionOrSkip", async () => {
    const launchWithBuiltExtension = vi.fn().mockResolvedValue({ ok: true })

    vi.doMock("./extension-build", () => ({
      launchWithBuiltExtension,
    }))

    const { launchWithBuiltExtensionOrSkip } = await import("./real-server")
    const test = { skip: vi.fn() } as any

    await launchWithBuiltExtensionOrSkip(test)

    expect(launchWithBuiltExtension).toHaveBeenCalledWith({})
    expect(test.skip).not.toHaveBeenCalled()
  })

  it("includes the manifest path when Knowledge QA live manifest JSON is invalid", async () => {
    const manifestPath = path.join(
      mkdtempSync(path.join(tmpdir(), "knowledge-qa-manifest-")),
      "manifest.json"
    )
    writeFileSync(manifestPath, "{invalid json", "utf8")
    process.env.TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST = manifestPath

    const { loadKnowledgeQaLiveManifest } = await import("./real-server")

    expect(() => loadKnowledgeQaLiveManifest()).toThrow(manifestPath)
    expect(() => loadKnowledgeQaLiveManifest()).toThrow("not valid JSON")
  })
})
