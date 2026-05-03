import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (label: string, ...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing ${label}: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

const sharedRouteRegistrySource = loadSource(
  "shared route registry",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
)
const extensionRouteRegistrySource = loadSource(
  "extension route registry",
  "extension/routes/route-registry.tsx",
  "apps/tldw-frontend/extension/routes/route-registry.tsx"
)
const routePathsSource = loadSource(
  "route paths",
  "../packages/ui/src/routes/route-paths.ts",
  "apps/packages/ui/src/routes/route-paths.ts"
)
const optionRouteVisibilitySource = loadSource(
  "option route visibility",
  "../packages/ui/src/routes/option-route-visibility.ts",
  "apps/packages/ui/src/routes/option-route-visibility.ts"
)
const uiSettingsSource = loadSource(
  "ui settings",
  "../packages/ui/src/services/settings/ui-settings.ts",
  "apps/packages/ui/src/services/settings/ui-settings.ts"
)
const headerShortcutItemsSource = loadSource(
  "header shortcut items",
  "../packages/ui/src/components/Layouts/header-shortcut-items.ts",
  "apps/packages/ui/src/components/Layouts/header-shortcut-items.ts"
)
const nextPageSource = loadSource(
  "Next chat workspace page",
  "pages/chat-workspace.tsx",
  "apps/tldw-frontend/pages/chat-workspace.tsx"
)
const extensionWrapperSource = loadSource(
  "extension chat workspace wrapper",
  "extension/routes/option-chat-workspace.tsx",
  "apps/tldw-frontend/extension/routes/option-chat-workspace.tsx"
)

const getHeaderShortcutAssignments = () => {
  const itemBlocks = Array.from(
    headerShortcutItemsSource.matchAll(/\n\s{6}\{\n[\s\S]*?\n\s{6}\}/g)
  ).map(([block]) => block)

  return itemBlocks
    .map((block) => {
      const id = block.match(/id:\s*"([^"]+)"/)?.[1]
      const shortcutIndex = block.match(/shortcutIndex:\s*(\d+)/)?.[1]

      if (!id || !shortcutIndex) return null

      return {
        id,
        shortcutIndex: Number(shortcutIndex)
      }
    })
    .filter(
      (assignment): assignment is { id: string; shortcutIndex: number } =>
        assignment !== null
    )
}

const getHeaderShortcutIndex = (id: string) =>
  getHeaderShortcutAssignments().find((assignment) => assignment.id === id)
    ?.shortcutIndex

const getDefaultSidebarShortcutSelectionLength = () => {
  const match = uiSettingsSource.match(
    /export const DEFAULT_SIDEBAR_SHORTCUT_SELECTION:\s*SidebarShortcutId\[\]\s*=\s*\[([\s\S]*?)\]/
  )
  if (!match) {
    throw new Error("Missing DEFAULT_SIDEBAR_SHORTCUT_SELECTION")
  }

  return Array.from(match[1].matchAll(/"[^"]+"/g)).length
}

const getSidebarShortcutMaxCount = () => {
  const match = uiSettingsSource.match(/SIDEBAR_SHORTCUT_MAX_COUNT\s*=\s*(\d+)/)
  if (!match) {
    throw new Error("Missing SIDEBAR_SHORTCUT_MAX_COUNT")
  }

  return Number(match[1])
}

describe("chat workspace route registry parity", () => {
  it("registers the shared chat workspace route using the shared path constant", () => {
    expect(sharedRouteRegistrySource).toContain("CHAT_WORKSPACE_PATH")
    expect(sharedRouteRegistrySource).toMatch(/path:\s*CHAT_WORKSPACE_PATH/)
  })

  it("keeps the chat workspace route visible in hosted mode", () => {
    expect(optionRouteVisibilitySource).toContain("CHAT_WORKSPACE_PATH")
    expect(headerShortcutItemsSource).toMatch(
      /HOSTED_VISIBLE_SHORTCUT_PATHS[\s\S]*CHAT_WORKSPACE_PATH/
    )
  })

  it("registers the extension chat workspace route and navigation metadata", () => {
    expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/chat-workspace"/)
    expect(extensionRouteRegistrySource).toMatch(
      /labelToken:\s*"option:header\.chatWorkspace"/
    )
    expect(extensionRouteRegistrySource).toMatch(/group:\s*"workspace"/)
  })

  it("keeps the Next page as an SSR-disabled shared route wrapper", () => {
    expect(nextPageSource).toContain("@/routes/option-chat-workspace")
    expect(nextPageSource).toMatch(/ssr:\s*false/)
  })

  it("keeps the extension wrapper pointed at the shared route", () => {
    expect(extensionWrapperSource).toContain("@/routes/option-chat-workspace")
  })

  it("declares the chat workspace path and constrains its viewport", () => {
    expect(routePathsSource).toMatch(
      /CHAT_WORKSPACE_PATH\s*=\s*"\/chat-workspace"/
    )
    expect(routePathsSource).toMatch(
      /VIEWPORT_CONSTRAINED_PATHS[\s\S]*CHAT_WORKSPACE_PATH/
    )
  })

  it("exposes chat workspace shortcut metadata and defaults", () => {
    expect(uiSettingsSource).toMatch(
      /HEADER_SHORTCUT_IDS\s*=\s*\[[\s\S]*"chat",\s*"chat-workspace"/
    )
    expect(uiSettingsSource).toMatch(
      /DEFAULT_SIDEBAR_SHORTCUT_SELECTION[\s\S]*"chat",\s*"chat-workspace"/
    )
    expect(headerShortcutItemsSource).toContain('id: "chat-workspace"')
    expect(headerShortcutItemsSource).toContain("to: CHAT_WORKSPACE_PATH")
    expect(headerShortcutItemsSource).toContain(
      'labelKey: "option:header.chatWorkspace"'
    )
    expect(headerShortcutItemsSource).toContain('labelDefault: "Chat Workspace"')
    expect(headerShortcutItemsSource).toMatch(
      /descriptionDefault:\s*"Chat-first workspace with staged sources and runtime context"/
    )
    expect(headerShortcutItemsSource).toMatch(
      /id:\s*"chat-workspace"[\s\S]*shortcutIndex:\s*2/
    )
    expect(headerShortcutItemsSource).toContain(
      'descriptionKey: "option:header.chatWorkspaceDesc"'
    )
  })

  it("preserves explicit numeric header shortcuts without duplicates", () => {
    const assignments = getHeaderShortcutAssignments()
    const assignedIndexes = assignments.map(
      (assignment) => assignment.shortcutIndex
    )

    expect(getHeaderShortcutIndex("chat")).toBe(1)
    expect(getHeaderShortcutIndex("chat-workspace")).toBe(2)
    expect(getHeaderShortcutIndex("prompt-studio")).toBe(3)
    expect(new Set(assignedIndexes).size).toBe(assignedIndexes.length)
  })

  it("keeps the default sidebar shortcut selection within the persisted maximum", () => {
    expect(getDefaultSidebarShortcutSelectionLength()).toBeLessThanOrEqual(
      getSidebarShortcutMaxCount()
    )
  })
})
