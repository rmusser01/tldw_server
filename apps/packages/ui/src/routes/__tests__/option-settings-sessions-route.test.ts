import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const findSource = (...candidates: string[]) => {
  const found = candidates.find((candidate) => existsSync(candidate))
  if (!found) {
    throw new Error(`Unable to locate source file: ${candidates.join(" | ")}`)
  }
  return readFileSync(found, "utf8")
}

const src = (rel: string) =>
  findSource(
    path.resolve(process.cwd(), `src/${rel}`),
    path.resolve(process.cwd(), `../packages/ui/src/${rel}`),
    path.resolve(process.cwd(), `apps/packages/ui/src/${rel}`)
  )

const extensionRegistry = () =>
  findSource(
    path.resolve(process.cwd(), "../../tldw-frontend/extension/routes/route-registry.tsx"),
    path.resolve(process.cwd(), "../tldw-frontend/extension/routes/route-registry.tsx"),
    path.resolve(process.cwd(), "tldw-frontend/extension/routes/route-registry.tsx"),
    path.resolve(process.cwd(), "apps/tldw-frontend/extension/routes/route-registry.tsx")
  )

describe("active sessions settings route", () => {
  it("is registered in the settings registry, the nav config and the extension registry", () => {
    const registry = src("routes/option-settings-route-registry.tsx")
    expect(registry).toMatch(/path:\s*"\/settings\/sessions"/)
    expect(registry).toContain("ActiveSessionsSettings")
    for (const source of [src("components/Layouts/settings-nav-config.ts"), extensionRegistry()]) {
      expect(source).toMatch(/path:\s*"\/settings\/sessions"/)
      expect(source).toContain("settings:activeSessions.navTitle")
    }
  })
})
