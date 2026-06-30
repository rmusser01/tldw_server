import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const extensionRouteRegistryPathCandidates = [
  "extension/routes/route-registry.tsx",
  "apps/tldw-frontend/extension/routes/route-registry.tsx",
]

const extensionRouteRegistryPath = extensionRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate)
)

if (!extensionRouteRegistryPath) {
  throw new Error("Unable to locate extension route-registry.tsx for stability parity test")
}

const extensionRouteRegistrySource = readFileSync(extensionRouteRegistryPath, "utf8")

describe("extension route registry stability parity", () => {
  it("keeps the image-generation canonical route and alias redirect aligned", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/settings\/image-generation"/)
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/settings\/image-gen"/)
    expect(extensionRouteRegistrySource).toMatch(
      /Navigate to="\/settings\/image-generation" replace/
    )
  })

  it("registers the admin MLX options route", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/admin\/mlx"/)
    expect(extensionRouteRegistrySource).toMatch(/labelToken:\s*"option:header\.adminMlx"/)
  })

  it("registers the quick chat popout options route", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/quick-chat-popout"/)
    expect(extensionRouteRegistrySource).toContain("OptionQuickChatPopout")
  })

  it("registers the provider key settings options route", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/settings\/provider-keys"/)
    expect(extensionRouteRegistrySource).toContain("OptionProviderKeysSettings")
    expect(extensionRouteRegistrySource).toContain("settings:providerKeys.navTitle")
  })
})
