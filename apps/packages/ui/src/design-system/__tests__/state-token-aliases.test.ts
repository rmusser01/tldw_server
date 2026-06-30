import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const resolveAppsRoot = () => {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const appsRoot = path.resolve(here, "../../../../..")
  if (
    fs.existsSync(path.resolve(appsRoot, "packages/ui")) &&
    fs.existsSync(path.resolve(appsRoot, "tldw-frontend"))
  ) {
    return appsRoot
  }
  throw new Error(`Unable to resolve apps root from ${here}; computed ${appsRoot}`)
}

const appsRoot = resolveAppsRoot()

const sharedCss = fs.readFileSync(
  path.resolve(appsRoot, "packages/ui/src/assets/tailwind-shared.css"),
  "utf8"
)
const frontendTailwindConfig = fs.readFileSync(
  path.resolve(appsRoot, "tldw-frontend/tailwind.config.js"),
  "utf8"
)

describe("state token aliases", () => {
  it("aliases v1 state tokens to existing semantic tokens in light and dark themes", () => {
    const aliases = [
      ["--state-ready", "--color-success"],
      ["--state-unavailable", "--color-danger"],
      ["--state-setup-required", "--color-warn"],
      ["--state-auth-required", "--color-warn"],
      ["--state-permission-denied", "--color-danger"],
      ["--state-degraded", "--color-warn"],
      ["--state-retrying", "--color-primary"],
      ["--state-blocked", "--color-danger"],
      ["--state-empty", "--color-muted"],
      ["--state-loading", "--color-muted"],
      ["--state-error", "--color-danger"]
    ]

    for (const [token, alias] of aliases) {
      expect(sharedCss.match(new RegExp(`${token}: var\\(${alias}\\)`, "g"))).toHaveLength(2)
    }
  })

  it("exposes readable state colors through the WebUI Tailwind config", () => {
    expect(frontendTailwindConfig).toContain("state:")
    expect(frontendTailwindConfig).toContain("--state-ready")
    expect(frontendTailwindConfig).toContain("setupRequired")
    expect(frontendTailwindConfig).toContain("permissionDenied")
  })
})
