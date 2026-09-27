import { afterEach, describe, expect, it, vi } from "vitest"

import { getTldwDeploymentMode } from "../deployment-mode"

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("getTldwDeploymentMode", () => {
  it("defaults to self-hosted mode when process is unavailable", () => {
    vi.stubGlobal("process", undefined)

    expect(getTldwDeploymentMode()).toBe("self_host")
  })
})
