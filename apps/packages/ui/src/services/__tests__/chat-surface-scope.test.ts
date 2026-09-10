import { afterEach, describe, expect, it, vi } from "vitest"

import {
  buildChatSurfaceScopeKey,
  buildChatSurfaceScopeKeyFromConfig,
  deriveSingleUserApiKeyCredentialScope
} from "@/services/chat-surface-scope"
import * as chatSurfaceScope from "@/services/chat-surface-scope"

const JWT_WITH_SUB =
  "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyLTQyIn0.signature"
const REFRESHED_JWT_WITH_SAME_SUB =
  "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyLTQyIiwiaWF0IjoyfQ.refreshed-signature"

describe("chat-surface-scope", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("changes the scope key when the server URL or auth mode changes", () => {
    expect(
      buildChatSurfaceScopeKey({
        serverUrl: "http://localhost:8000",
        authMode: "single-user",
        orgId: null,
        userId: null
      })
    ).not.toBe(
      buildChatSurfaceScopeKey({
        serverUrl: "https://prod.example.com",
        authMode: "multi-user",
        orgId: 7,
        userId: 42
      })
    )
  })

  it("uses access-token identity when an explicit user id is unavailable", () => {
    expect(
      buildChatSurfaceScopeKeyFromConfig({
        serverUrl: "https://prod.example.com",
        authMode: "multi-user",
        orgId: 7,
        accessToken: JWT_WITH_SUB
      })
    ).toContain("user:user-42")
  })

  it("keeps the global surface scope stable when the same subject refreshes credentials", () => {
    const firstScope = buildChatSurfaceScopeKeyFromConfig({
      serverUrl: "https://prod.example.com",
      authMode: "multi-user",
      orgId: 7,
      accessToken: JWT_WITH_SUB
    })
    const refreshedScope = buildChatSurfaceScopeKeyFromConfig({
      serverUrl: "https://prod.example.com",
      authMode: "multi-user",
      orgId: 7,
      accessToken: REFRESHED_JWT_WITH_SAME_SUB
    })

    expect(firstScope).toBe(refreshedScope)
    expect(firstScope).not.toContain(JWT_WITH_SUB)
    expect(refreshedScope).not.toContain(REFRESHED_JWT_WITH_SAME_SUB)
  })

  it("derives a PromptAssist-only authorization revision without exposing credentials", () => {
    const deriveAuthorizationRevision = (
      chatSurfaceScope as typeof chatSurfaceScope & {
        derivePromptAssistAuthorizationRevision?: (config: {
          authMode: string
          accessToken: string
          apiKey?: string
        }) => string
      }
    ).derivePromptAssistAuthorizationRevision

    expect(deriveAuthorizationRevision).toBeTypeOf("function")
    if (!deriveAuthorizationRevision) return

    const firstRevision = deriveAuthorizationRevision({
      authMode: "multi-user",
      accessToken: JWT_WITH_SUB
    })
    const refreshedRevision = deriveAuthorizationRevision({
      authMode: "multi-user",
      accessToken: REFRESHED_JWT_WITH_SAME_SUB
    })

    expect(firstRevision).not.toBe(refreshedRevision)
    expect(firstRevision).not.toContain(JWT_WITH_SUB)
    expect(refreshedRevision).not.toContain(REFRESHED_JWT_WITH_SAME_SUB)
  })

  it("changes single-user scope keys when the API key changes without leaking the raw key", () => {
    const firstScope = buildChatSurfaceScopeKeyFromConfig({
      serverUrl: "https://prod.example.com",
      authMode: "single-user",
      orgId: null,
      apiKey: "alpha-secret-key"
    })
    const secondScope = buildChatSurfaceScopeKeyFromConfig({
      serverUrl: "https://prod.example.com",
      authMode: "single-user",
      orgId: null,
      apiKey: "beta-secret-key"
    })

    expect(firstScope).not.toBe(secondScope)
    expect(firstScope).not.toContain("alpha-secret-key")
    expect(secondScope).not.toContain("beta-secret-key")
  })

  it("derives a collision-resistant credential scope without Web Crypto", () => {
    vi.stubGlobal("crypto", undefined)

    expect(
      deriveSingleUserApiKeyCredentialScope("single-user", "lan-api-key")
    ).toMatch(/^key:sha256:[0-9a-f]{64}$/)
  })
})
