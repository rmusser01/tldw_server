import { beforeEach, describe, expect, it, vi } from "vitest"
import { bgRequest } from "@/services/background-proxy"
import { workspaceApiMethods } from "../workspace-api"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

describe("workspace API skill methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("sends dry-run intent when rendering a skill without execution", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: [],
      model_override: null,
      execution_mode: "fork",
      fork_output: null,
      dry_run: true
    })

    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}/execute"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize/execute")
    }

    const controller = new AbortController()
    await workspaceApiMethods.executeSkill.call(
      clientCore as any,
      "summarize",
      "chapter 1",
      { dryRun: true, signal: controller.signal }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize/execute",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { args: "chapter 1", dry_run: true },
      abortSignal: controller.signal
    })
  })

  it("forwards cancellation when previewing a text skill import", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ valid: true })
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/import/preview")
    }
    const controller = new AbortController()
    const payload = { name: "preview-skill", content: "Body" }

    await workspaceApiMethods.previewSkillImport.call(
      clientCore as any,
      payload,
      { signal: controller.signal }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/import/preview",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      abortSignal: controller.signal
    })
  })

  it("sends the expected version when importing a reviewed file", async () => {
    const upload = vi.fn().mockResolvedValue({ name: "reviewed-skill" })
    const clientCore = { upload }
    const controller = new AbortController()
    const file = {
      name: "reviewed-skill.md",
      type: "text/markdown",
      arrayBuffer: vi.fn().mockResolvedValue(new TextEncoder().encode("Replacement").buffer)
    } as unknown as File

    await workspaceApiMethods.importSkillFile.call(
      clientCore as any,
      file,
      { overwrite: true, expectedVersion: 7, signal: controller.signal }
    )

    expect(upload).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/skills/import/file?overwrite=true&expected_version=7",
      method: "POST",
      abortSignal: controller.signal,
      fileFieldName: "file",
      file: expect.objectContaining({
        name: "reviewed-skill.md",
        type: "text/markdown"
      })
    }))
  })

  it("sends If-Match when deleting a skill with a valid version", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
    }
    const controller = new AbortController()

    await workspaceApiMethods.deleteSkill.call(
      clientCore as any,
      "summarize",
      3,
      { signal: controller.signal }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize",
      method: "DELETE",
      headers: { "If-Match": "3" },
      abortSignal: controller.signal
    })
  })

  it("omits If-Match when deleting a skill without a known version", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
    }

    await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize")

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize",
      method: "DELETE"
    })
  })

  it.each([Number.NaN, 0, -1, 1.5, Number.POSITIVE_INFINITY])(
    "omits If-Match for invalid delete version %s",
    async (version) => {
      vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
      const clientCore = {
        resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
        fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
      }

      await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize", version)

      expect(bgRequest).toHaveBeenCalledWith({
        path: "/api/v1/skills/summarize",
        method: "DELETE"
      })
    }
  )

  it("posts selected skills and valid row versions for bulk delete", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      deleted: ["skill-a", "skill-b"],
      count: 2
    })
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/bulk-delete")
    }

    await workspaceApiMethods.bulkDeleteSkills.call(clientCore as any, [
      { name: "skill-a", version: 2 },
      { name: "skill-b", version: 3 }
    ])

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/bulk-delete",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        skills: [
          { name: "skill-a", version: 2 },
          { name: "skill-b", version: 3 }
        ]
      }
    })
  })

  it("omits invalid bulk delete versions while preserving selected names", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      deleted: ["legacy-skill", "invalid-version"],
      count: 2
    })
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/bulk-delete")
    }

    await workspaceApiMethods.bulkDeleteSkills.call(clientCore as any, [
      { name: "legacy-skill" },
      { name: "invalid-version", version: Number.NaN }
    ])

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/bulk-delete",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        skills: [
          { name: "legacy-skill" },
          { name: "invalid-version" }
        ]
      }
    })
  })

  it("lists Skills Trash with pagination", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      skills: [], count: 0, total: 0, limit: 20, offset: 20
    })
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/trash")
    }

    await workspaceApiMethods.listSkillTrash.call(clientCore as any, {
      limit: 20,
      offset: 20
    })

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/trash?limit=20&offset=20",
      method: "GET",
      abortSignal: undefined
    })
  })

  it("sends If-Match when restoring and permanently deleting Trash items", async () => {
    vi.mocked(bgRequest).mockResolvedValue(undefined)
    const clientCore = {
      resolveApiPath: vi.fn()
        .mockResolvedValueOnce("/api/v1/skills/{name}/restore")
        .mockResolvedValueOnce("/api/v1/skills/{name}/purge"),
      fillPathParams: vi.fn()
        .mockReturnValueOnce("/api/v1/skills/summarize/restore")
        .mockReturnValueOnce("/api/v1/skills/summarize/purge")
    }

    await workspaceApiMethods.restoreSkill.call(clientCore as any, "summarize", 4)
    await workspaceApiMethods.purgeSkill.call(clientCore as any, "summarize", 4)

    expect(bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/skills/summarize/restore",
      method: "POST",
      headers: { "If-Match": "4" }
    })
    expect(bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/skills/summarize/purge",
      method: "DELETE",
      headers: { "If-Match": "4" }
    })
  })

  it("returns exported skill blob with filename metadata from response headers", async () => {
    const payload = new Uint8Array([1, 2, 3]).buffer
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: payload,
      headers: {
        "content-type": "application/zip",
        "content-disposition": 'attachment; filename="server-skill.zip"'
      }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    const result = await workspaceApiMethods.exportSkill.call(
      clientCore as any,
      "client-skill"
    )

    expect(result.filename).toBe("server-skill.zip")
    expect(result.blob).toBeInstanceOf(Blob)
    expect(result.blob.type).toBe("application/zip")
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/client-skill/export",
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true
    })
  })

  it("prefers encoded export filenames from content disposition metadata", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: new Uint8Array([1, 2, 3]).buffer,
      headers: {
        "content-disposition": "attachment; filename=\"plain.zip\"; filename*=UTF-8''encoded-skill.zip"
      }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    const result = await workspaceApiMethods.exportSkill.call(
      clientCore as any,
      "client-skill"
    )

    expect(result.filename).toBe("encoded-skill.zip")
  })

  it("accepts encoded export filenames with RFC 5987 language tags", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: new Uint8Array([1, 2, 3]).buffer,
      headers: {
        "content-disposition": "attachment; filename*=UTF-8'en'encoded-lang-skill.zip"
      }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    const result = await workspaceApiMethods.exportSkill.call(
      clientCore as any,
      "client-skill"
    )

    expect(result.filename).toBe("encoded-lang-skill.zip")
  })

  it("falls back to a safe export filename when response metadata is unsafe", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: new Uint8Array([4, 5, 6]).buffer,
      headers: {
        "content-disposition": 'attachment; filename="../secret.zip"'
      }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    const result = await workspaceApiMethods.exportSkill.call(
      clientCore as any,
      "safe-skill"
    )

    expect(result.filename).toBe("safe-skill.zip")
    expect(result.blob).toBeInstanceOf(Blob)
  })

  it("preserves safe fallback filename characters from user-provided skill names", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: new Uint8Array([4, 5, 6]).buffer
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    const result = await workspaceApiMethods.exportSkill.call(
      clientCore as any,
      "  2_Custom Skill!  "
    )

    expect(result.filename).toBe("2_Custom-Skill.zip")
  })

  it("rejects export responses without contextual response metadata", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    await expect(
      workspaceApiMethods.exportSkill.call(clientCore as any, "client-skill")
    ).rejects.toThrow("Export failed for skill client-skill: missing response")
  })

  it("rejects successful export responses without binary data", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: {
        "content-disposition": 'attachment; filename="server-skill.zip"'
      }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    await expect(
      workspaceApiMethods.exportSkill.call(clientCore as any, "client-skill")
    ).rejects.toThrow("Export failed for skill client-skill: missing export payload")
  })

  it("rejects serialized export responses with invalid binary payloads", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: { lost: "array-buffer" }
    })
    const clientCore = {
      ensureConfigForRequest: vi.fn().mockResolvedValue(undefined)
    }

    await expect(
      workspaceApiMethods.exportSkill.call(clientCore as any, "client-skill")
    ).rejects.toThrow("Export failed for skill client-skill: invalid export payload")
  })
})
