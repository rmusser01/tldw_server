import { beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import {
  getSetupReadinessStatus,
  previewSetupReadiness
} from "../setup-readiness"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))

describe("setup readiness authorization", () => {
  beforeEach(() => {
    vi.mocked(bgRequest).mockReset()
    vi.mocked(bgRequest).mockResolvedValue({ ok: true, status: 200, data: { lanes: [] } })
  })

  it("can fetch first-run readiness before app credentials are configured", async () => {
    await getSetupReadinessStatus()
    expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/setup/readiness/status", noAuth: true
    }))
  })

  it("uses first-run access for readiness preview", async () => {
    await previewSetupReadiness({ profile_id: "local" })
    expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/setup/readiness/preview", method: "POST", noAuth: true
    }))
  })

  it("retains authenticated access for admin readiness", async () => {
    await getSetupReadinessStatus({ mode: "admin" })
    expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/setup/admin/readiness/status"
    }))
    expect(vi.mocked(bgRequest).mock.calls[0][0].noAuth).not.toBe(true)
  })
})
