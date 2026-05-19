import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  storedConfig: null as
    | {
        serverUrl: string
        authMode: "single-user" | "multi-user"
        apiKey?: string
        accessToken?: string
      }
    | null
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key?: string) => {
      if (key === "tldwConfig") {
        return mocks.storedConfig
      }
      return null
    }),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import {
  TldwApiClient,
  normalizePersonaProfile
} from "@/services/tldw/TldwApiClient"

const createConfiguredClient = (): TldwApiClient => {
  mocks.storedConfig = {
    serverUrl: "http://127.0.0.1:8000",
    authMode: "single-user",
    apiKey: "test-api-key"
  }
  return new TldwApiClient()
}

describe("TldwApiClient assistant identity helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storedConfig = null
  })

  it("preserves scope metadata and forwards workspace scope when loading a chat", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        if (
          request.path ===
            "/api/v1/chats/chat-7?scope_type=workspace&workspace_id=workspace-7" &&
          request.method === "GET"
        ) {
          return {
            id: "chat-7",
            title: "Workspace thread",
            created_at: "2026-03-08T00:00:00Z",
            scope_type: "workspace",
            workspace_id: "workspace-7"
          }
        }

        throw new Error(`Unexpected request: ${request.method} ${request.path}`)
      }
    )

    const client = createConfiguredClient()
    const chat = await client.getChat("chat-7", {
      scope: { type: "workspace", workspaceId: "workspace-7" }
    })

    expect(chat).toMatchObject({
      id: "chat-7",
      scope_type: "workspace",
      workspace_id: "workspace-7"
    })
  })

  it("normalizes assistant identity fields for persona-backed chats", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        if (
          request.path === "/api/v1/chats/chat-7?scope_type=global" &&
          request.method === "GET"
        ) {
          return {
            id: "chat-7",
            title: "Persona thread",
            created_at: "2026-03-08T00:00:00Z",
            assistant_kind: "persona",
            assistant_id: "garden-helper",
            persona_memory_mode: "read_only"
          }
        }

        throw new Error(`Unexpected request: ${request.method} ${request.path}`)
      }
    )

    const client = createConfiguredClient()
    const chat = await client.getChat("chat-7")

    expect(chat).toMatchObject({
      id: "chat-7",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only"
    })
  })

  it("lists persona profiles from the persona catalog", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        if (
          request.path === "/api/v1/persona/catalog" &&
          request.method === "GET"
        ) {
          return [
            {
              id: 17,
              name: "Garden Helper",
              character_card_id: 42,
              origin_character_id: 42,
              buddy_summary: {
                has_buddy: 1,
                persona_name: " Garden Helper ",
                role_summary: " Research organizer ",
                visual: {
                  species_id: " owl ",
                  silhouette_id: " owl_round ",
                  palette_id: " moss "
                }
              }
            }
          ]
        }

        throw new Error(`Unexpected request: ${request.method} ${request.path}`)
      }
    )

    const client = createConfiguredClient()
    const profiles = await client.listPersonaProfiles()

    expect(profiles).toEqual([
      {
        id: "17",
        name: "Garden Helper",
        character_card_id: 42,
        origin_character_id: 42,
        buddy_summary: {
          has_buddy: true,
          persona_name: "Garden Helper",
          role_summary: "Research organizer",
          visual: {
            species_id: "owl",
            silhouette_id: "owl_round",
            palette_id: "moss"
          }
        }
      }
    ])
  })

  it("gets a persona profile by id", async () => {
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string }) => {
        if (
          request.path === "/api/v1/persona/profiles/garden-helper" &&
          request.method === "GET"
        ) {
          return {
            id: "garden-helper",
            name: "Garden Helper",
            system_prompt: "Stay focused on the garden.",
            use_persona_state_context_default: true,
            buddy_summary: {
              has_buddy: 1,
              persona_name: " Garden Helper ",
              role_summary: " Research organizer ",
              visual: {
                species_id: " owl ",
                silhouette_id: " owl_round ",
                palette_id: " moss "
              }
            }
          }
        }

        throw new Error(`Unexpected request: ${request.method} ${request.path}`)
      }
    )

    const client = createConfiguredClient()
    const profile = await client.getPersonaProfile("garden-helper")

    expect(profile).toEqual({
      id: "garden-helper",
      name: "Garden Helper",
      system_prompt: "Stay focused on the garden.",
      use_persona_state_context_default: true,
      buddy_summary: {
        has_buddy: true,
        persona_name: "Garden Helper",
        role_summary: "Research organizer",
        visual: {
          species_id: "owl",
          silhouette_id: "owl_round",
          palette_id: "moss"
        }
      }
    })
  })

  it("preserves an explicit null buddy_summary over legacy camelCase fallback data", () => {
    const normalized = normalizePersonaProfile({
      id: "garden-helper",
      name: "Garden Helper",
      buddy_summary: null,
      buddySummary: {
        hasBuddy: true,
        personaName: "Stale Buddy",
        roleSummary: "Should not win",
        visual: {
          speciesId: "owl",
          silhouetteId: "perch",
          paletteId: "dawn"
        }
      }
    })

    expect(normalized.buddy_summary).toBeNull()
  })

  it("defaults has_buddy to true when a summary object exists without an explicit flag", () => {
    const normalized = normalizePersonaProfile({
      id: "garden-helper",
      name: "Garden Helper",
      buddy_summary: {
        persona_name: "Garden Helper",
        role_summary: "Keeps the route on track",
        visual: {
          species_id: "owl",
          silhouette_id: "perch",
          palette_id: "dawn"
        }
      }
    })

    expect(normalized.buddy_summary).toMatchObject({
      has_buddy: true,
      persona_name: "Garden Helper",
      visual: {
        species_id: "owl",
        silhouette_id: "perch",
        palette_id: "dawn"
      }
    })
  })
})
