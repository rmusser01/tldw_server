import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import {
  CHAT_PATH,
  LOREBOOK_DEBUG_FOCUS,
  buildChatLorebookDebugPath
} from "../route-paths"

// The option route registry is split into deferred per-area registries
// (see deferred-options-route.tsx); /chat lives in option-chat-route-registry.
const routeRegistryFileNames = [
  "route-registry.tsx",
  "option-chat-route-registry.tsx"
]

const routeRegistryDirCandidates = [
  "src/routes",
  "../packages/ui/src/routes",
  "apps/packages/ui/src/routes"
]

const routeRegistryDir = routeRegistryDirCandidates.find((candidate) =>
  existsSync(`${candidate}/route-registry.tsx`)
)

if (!routeRegistryDir) {
  throw new Error("Unable to locate route-registry.tsx for route-path contract test")
}

const routeRegistrySource = routeRegistryFileNames
  .map((fileName) => readFileSync(`${routeRegistryDir}/${fileName}`, "utf8"))
  .join("\n")

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
