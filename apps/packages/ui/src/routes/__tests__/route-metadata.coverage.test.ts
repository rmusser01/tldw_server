import { describe, expect, it } from "vitest"

import {
  getCanonicalRoutePath,
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
  "/workspace-playground",
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
  it("tracks the audited root route inventory", () => {
    expect(AUDITED_ROOT_ROUTES).toHaveLength(74)
  })

  it("defines metadata for every audited root route", () => {
    for (const route of AUDITED_ROOT_ROUTES) {
      expect(getRouteMetadata(route), route).toBeDefined()
    }
  })

  it("defines required user-facing metadata fields", () => {
    for (const metadata of ROUTE_METADATA) {
      expect(metadata.path).toMatch(/^\//)
      expect(metadata.canonicalPath).toMatch(/^\//)
      expect(metadata.label.trim()).not.toHaveLength(0)
      expect(metadata.group).toBeTruthy()
      expect(metadata.surface).toBeTruthy()
      expect(metadata.availability.length).toBeGreaterThan(0)
      expect(metadata.rationale.trim()).not.toHaveLength(0)
    }
  })

  it("keeps canonical route paths explicit for legacy or redirect routes", () => {
    expect(getCanonicalRoutePath("/prompt-studio")).toBe("/prompts")
    expect(getRouteMetadata("/prompt-studio")?.redirectsTo).toBe(
      "/prompts?tab=studio"
    )
  })

  it("does not define duplicate route metadata paths", () => {
    const metadataPaths = ROUTE_METADATA.map((metadata) => metadata.path)

    expect(new Set(metadataPaths).size).toBe(metadataPaths.length)
  })
})
