import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const loadSource = (page: string) =>
  readFileSync(path.resolve(__dirname, "../../pages", page), "utf8")

describe("sources Next.js page shims", () => {
  it("loads the shared route modules for user and admin sources pages", () => {
    expect(loadSource("sources.tsx")).toContain(
      'dynamic(() => import("@/routes/option-sources"), { ssr: false })'
    )
    expect(
      loadSource("sources/new.tsx")
    ).toContain(
      'dynamic(() => import("@/routes/option-sources-new"), { ssr: false })'
    )
    expect(
      loadSource("sources/[sourceId].tsx")
    ).toContain(
      'dynamic(() => import("@/routes/option-sources-detail"), { ssr: false })'
    )
    expect(
      loadSource("admin/sources.tsx")
    ).toContain(
      'dynamic(() => import("@/routes/option-admin-sources"), { ssr: false })'
    )
  })
})
