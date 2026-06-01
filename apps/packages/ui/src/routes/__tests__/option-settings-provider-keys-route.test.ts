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

describe("provider key settings shared route", () => {
  it("keeps the shared settings registry and navigation aligned", () => {
    const registrySource = findSource(
      path.resolve(process.cwd(), "src/routes/option-settings-route-registry.tsx"),
      path.resolve(process.cwd(), "../packages/ui/src/routes/option-settings-route-registry.tsx"),
      path.resolve(process.cwd(), "apps/packages/ui/src/routes/option-settings-route-registry.tsx")
    )
    const navSource = findSource(
      path.resolve(process.cwd(), "src/components/Layouts/settings-nav-config.ts"),
      path.resolve(process.cwd(), "../packages/ui/src/components/Layouts/settings-nav-config.ts"),
      path.resolve(process.cwd(), "apps/packages/ui/src/components/Layouts/settings-nav-config.ts")
    )

    expect(registrySource).toMatch(/path:\s*"\/settings\/provider-keys"/)
    expect(registrySource).toContain("OptionProviderKeysSettings")
    expect(registrySource).toContain("ProviderKeysSettings")
    expect(navSource).toMatch(/path:\s*"\/settings\/provider-keys"/)
    expect(navSource).toContain("settings:providerKeys.navTitle")
  })
})
