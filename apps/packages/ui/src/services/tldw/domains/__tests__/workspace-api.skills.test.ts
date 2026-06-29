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

    await workspaceApiMethods.executeSkill.call(
      clientCore as any,
      "summarize",
      "chapter 1",
      { dryRun: true }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize/execute",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { args: "chapter 1", dry_run: true }
    })
  })

  it("sends If-Match when deleting a skill with a valid version", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
    }

    await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize", 3)

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize",
      method: "DELETE",
      headers: { "If-Match": "3" }
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
    ).rejects.toThrow("Export failed: missing export payload")
  })
})
