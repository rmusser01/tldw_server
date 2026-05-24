import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import {
  CHAT_PATH,
  LOREBOOK_DEBUG_FOCUS,
  buildChatLorebookDebugPath
} from "../route-paths"

const routeRegistryPathCandidates = [
  "src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for route-path contract test")
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

describe("route-paths lorebook debug entrypoint", () => {
  it("builds chat lorebook diagnostics path with expected query params", () => {
    const href = buildChatLorebookDebugPath()
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get("focus")).toBe(LOREBOOK_DEBUG_FOCUS)
    expect(parsed.searchParams.get("from")).toBeNull()
  })

  it("targets a registered route for workspace diagnostics links", () => {
    const href = buildChatLorebookDebugPath({ from: "research-workspace" })
    const parsed = new URL(href, "https://example.local")

    expect(routeRegistrySource).toContain(`path: "${parsed.pathname}"`)
    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get("focus")).toBe(LOREBOOK_DEBUG_FOCUS)
    expect(parsed.searchParams.get("from")).toBe("research-workspace")
  })
})
