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
})
