import { describe, expect, it } from "vitest"

import { resolveBuildProfile } from "../../scripts/resolve-build-profile.mjs"
import { shapeWebuiBuildEnv } from "../scripts/build-with-profile.mjs"

describe("resolveBuildProfile", () => {
  it("maps main to production", () => {
    expect(resolveBuildProfile({ branch: "main" })).toBe("production")
  })

  it("maps feature branches to development", () => {
    expect(resolveBuildProfile({ branch: "feat/example" })).toBe("development")
  })

  it("defaults unknown branch state to development", () => {
    expect(resolveBuildProfile({ branch: "" })).toBe("development")
  })

  it("prefers explicit overrides", () => {
    expect(resolveBuildProfile({ branch: "main", override: "development" })).toBe(
      "development"
    )
  })
})

describe("shapeWebuiBuildEnv", () => {
  it("forces quickstart settings for production", () => {
    const env = shapeWebuiBuildEnv("production", {
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
    })

    expect(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE).toBe("quickstart")
    expect(env.NEXT_PUBLIC_API_URL).toBeUndefined()
    expect(env.TLDW_INTERNAL_API_ORIGIN).toBe("http://127.0.0.1:8000")
  })

  it("requires advanced-mode browser api settings for development", () => {
    const env = shapeWebuiBuildEnv("development", {
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
    })

    expect(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE).toBe("advanced")
    expect(env.NEXT_PUBLIC_API_URL).toBe("http://127.0.0.1:8000")
  })
})
