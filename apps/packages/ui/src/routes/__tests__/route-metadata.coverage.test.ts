import { describe, expect, it } from "vitest"

import {
  AUDITED_ROOT_ROUTE_PATHS,
  getCanonicalRoutePath,
  getCommandPaletteRoutes,
  getCommandPaletteTarget,
  getRouteMetadata,
  ROUTE_METADATA
} from "../route-metadata"

const AUDITED_ROOT_ROUTES = [
  "/",
  "/setup",
  "/login",
  "/signup",
  "/account",
  "/profile",
  "/privileges",
  "/config",
  "/billing",
  "/404",
  "/chat",
  "/quick-chat-popout",
  "/persona",
  "/characters",
  "/companion",
  "/agents",
  "/agent-tasks",
  "/chat-workflows",
  "/chat-workspace",
  "/knowledge",
  "/search",
  "/research",
  "/workspaces",
  "/research-workspace",
  "/document-workspace",
  "/repo2txt",
  "/model-playground",
  "/writing-playground",
  "/presentation-studio",
  "/audiobook-studio",
  "/media",
  "/media-multi",
  "/review",
  "/media-trash",
  "/items",
  "/collections",
  "/reading",
  "/notes",
  "/shared",
  "/chatbooks",
  "/chatbooks-playground",
  "/sources",
  "/connectors",
  "/integrations",
  "/scheduled-tasks",
  "/watchlists",
  "/workflow-editor",
  "/settings",
  "/admin",
  "/mcp-hub",
  "/acp-playground",
  "/prompts",
  "/prompt-studio",
  "/dictionaries",
  "/world-books",
  "/speech",
  "/stt",
  "/tts",
  "/audio",
  "/evaluations",
  "/flashcards",
  "/quiz",
  "/moderation-playground",
  "/content-review",
  "/claims-review",
  "/data-tables",
  "/chunking-playground",
  "/kanban",
  "/skills",
  "/vn-assets",
  "/vn-play",
  "/documentation",
  "/notifications",
  "/composer-variants-preview",
  "/onboarding-test"
] as const

describe("route metadata coverage", () => {
  it("tracks the audited 75-route bootstrap set explicitly", () => {
    expect(AUDITED_ROOT_ROUTE_PATHS).toEqual(AUDITED_ROOT_ROUTES)
    expect(AUDITED_ROOT_ROUTE_PATHS).toHaveLength(75)
  })

  it("defines metadata for every audited root route", () => {
    for (const route of AUDITED_ROOT_ROUTES) {
      expect(getRouteMetadata(route), route).toBeDefined()
    }
  })

  it("defines required user-facing metadata fields", () => {
    for (const metadata of ROUTE_METADATA) {
      expect(metadata.path, metadata.path).toMatch(/^\//)
      expect(metadata.canonicalPath, metadata.path).toMatch(/^\//)
      expect(metadata.label.trim(), metadata.path).not.toHaveLength(0)
      expect(metadata.group, metadata.path).toBeTruthy()
      expect(metadata.surface, metadata.path).toBeTruthy()
      expect(metadata.availability.length, metadata.path).toBeGreaterThan(0)
      expect(metadata.rationale.trim(), metadata.path).not.toHaveLength(0)
    }
  })

  it("resolves compatibility aliases to canonical routes", () => {
    expect(getCanonicalRoutePath("/audio")).toBe("/speech")
    expect(getCanonicalRoutePath("/search")).toBe("/knowledge")
    expect(getCanonicalRoutePath("/prompt-studio")).toBe("/prompts")
    expect(getCanonicalRoutePath("/workspace-playground")).toBeUndefined()
  })

  it("defines research workspace as the canonical workspace research route without legacy aliases", () => {
    const metadata = getRouteMetadata("/research-workspace")

    expect(metadata).toMatchObject({
      path: "/research-workspace",
      canonicalPath: "/research-workspace",
      label: "Research Workspace",
      group: "workspace"
    })
    expect(metadata?.aliases).toBeUndefined()
    expect(metadata?.redirectsTo).toBeUndefined()
    expect(getRouteMetadata("/workspace-playground")).toBeUndefined()
  })

  it("defines Workspaces as the canonical manager route without aliases or redirects", () => {
    const metadata = getRouteMetadata("/workspaces")

    expect(metadata).toMatchObject({
      path: "/workspaces",
      canonicalPath: "/workspaces",
      label: "Workspaces",
      group: "workspace",
      surface: "advanced_self_hosted",
      nav: "secondary",
      requiresBackend: true
    })
    expect(metadata?.availability).toEqual(
      expect.arrayContaining(["web", "extension_options"])
    )
    expect(metadata?.aliases).toBeUndefined()
    expect(metadata?.redirectsTo).toBeUndefined()
  })

  it("labels /notes as the primary Notes workspace destination", () => {
    expect(getRouteMetadata("/notes")).toMatchObject({
      path: "/notes",
      canonicalPath: "/notes",
      label: "Notes",
      group: "media_library",
      nav: "primary",
      commandPalette: "show",
      requiresBackend: true
    })
  })

  it("exposes command palette routes through metadata-owned targets", () => {
    const commandPalettePaths = getCommandPaletteRoutes().map(
      (metadata) => metadata.path
    )

    expect(commandPalettePaths).toEqual(
      expect.arrayContaining([
        "/chat",
        "/knowledge",
        "/media",
        "/notes",
        "/prompts",
        "/flashcards",
        "/documentation",
        "/settings",
        "/mcp-hub"
      ])
    )
    expect(getCommandPaletteTarget("/mcp-hub")).toBe("/mcp-hub")
    expect(getCommandPaletteTarget("/prompt-studio")).toBe("/prompts")
  })

  it("keeps non-product route surfaces hidden from default navigation", () => {
    for (const metadata of ROUTE_METADATA) {
      if (
        metadata.surface === "internal_qa_debug" ||
        metadata.surface === "hosted_only" ||
        metadata.surface === "legacy_alias"
      ) {
        expect(metadata.nav, metadata.path).toBe("hidden")
      }
    }
  })
})
