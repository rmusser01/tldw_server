import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolveAppsRoot = () => {
  const cwd = process.cwd()
  if (fs.existsSync(path.resolve(cwd, "packages/ui"))) return cwd
  if (fs.existsSync(path.resolve(cwd, "../packages/ui"))) return path.resolve(cwd, "..")
  if (fs.existsSync(path.resolve(cwd, "../../tldw-frontend"))) return path.resolve(cwd, "../..")
  throw new Error(`Unable to resolve apps root from ${cwd}`)
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
