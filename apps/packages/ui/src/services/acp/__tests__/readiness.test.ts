import { describe, expect, it } from "vitest"

import { buildACPSetupIssues, normalizeACPHealthStatus } from "@/services/acp/readiness"

describe("ACP readiness normalization", () => {
  it("treats an empty agent inventory as unavailable even when overall is degraded", () => {
    const health = normalizeACPHealthStatus({
      runner: { status: "ok" },
      agents: [],
      overall: "degraded",
      message: "Runner is present but no agents are configured"
    })

    expect(health?.agent).toBe("unavailable")
    expect(buildACPSetupIssues(health).map((issue) => issue.code)).toContain("agent_unavailable")
  })
})
