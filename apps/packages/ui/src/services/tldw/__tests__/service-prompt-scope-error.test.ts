import { describe, expect, it } from "vitest"

import {
  createServicePromptScopeChangedError,
  isServicePromptRequestPath,
  servicePromptSingleUserApiKeyScopeMatches,
  servicePromptTargetsMatch
} from "../service-prompt-scope-error"
import {
  buildChatSurfaceScopeKeyFromConfig,
  deriveSingleUserApiKeyCredentialScope,
} from "@/services/chat-surface-scope"

describe("Service Prompt scope policy", () => {
  it.each([
    "/api/v1/writing/manuscripts/scenes/scene-a",
    "/api/v1/writing/manuscripts/projects/project-a/characters?role=protagonist",
    "/api/v1/writing/manuscripts/projects/project-a/world-info?kind=location",
  ])("allows only GET for the bounded context read %s", (path) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(true)
    for (const method of ["POST", "PATCH", "DELETE", "PUT"]) {
      expect(isServicePromptRequestPath(path, method)).toBe(false)
    }
  })

  it.each([
    "/api/v1/writing/manuscripts/projects/project-a",
    "/api/v1/writing/manuscripts/scenes/scene-a/annotations",
    "/api/v1/writing/manuscripts/projects/project-a/characters/relationships",
    "/api/v1/writing/manuscripts/projects/%2e%2e/characters",
    "/api/v1/writing/manuscripts/scenes/a%2fb",
    "/api/v1/writing/manuscripts/scenes/a%5cb",
    "/api/v1/writing/manuscripts/scenes/",
  ])("does not widen scoped access to %s", (path) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(false)
  })

  it("compares only the frozen target keys", () => {
    const current = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      orgId: "org-1",
      accessToken: "current-token"
    }
    const captured = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      orgId: "org-1",
      accessToken: "captured-token"
    }

    expect(servicePromptTargetsMatch(current, captured)).toBe(true)
  })

  it("requires the captured single-user API-key scope to match", () => {
    const expected = deriveSingleUserApiKeyCredentialScope(
      "single-user",
      "captured-account-key"
    )

    expect(servicePromptSingleUserApiKeyScopeMatches({
      authMode: "single-user",
      apiKey: "captured-account-key"
    }, expected)).toBe(true)
    expect(servicePromptSingleUserApiKeyScopeMatches({
      authMode: "single-user",
      apiKey: "different-account-key"
    }, expected)).toBe(false)
    expect(servicePromptSingleUserApiKeyScopeMatches({
      authMode: "single-user",
      apiKey: "captured-account-key"
    }, undefined)).toBe(false)
    expect(servicePromptSingleUserApiKeyScopeMatches({
      authMode: "multi-user"
    }, undefined)).toBe(true)
  })

  it("rejects a changed API key that collides in the UI scope hash", () => {
    const capturedKey = "key-s54895-4z7"
    const changedKey = "key-jiqole-3dcy"
    const expectedScope = deriveSingleUserApiKeyCredentialScope(
      "single-user",
      capturedKey
    )

    expect(buildChatSurfaceScopeKeyFromConfig({
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      apiKey: changedKey
    }, { userId: null })).toBe(
      buildChatSurfaceScopeKeyFromConfig({
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        apiKey: capturedKey
      }, { userId: null })
    )
    expect(
      servicePromptSingleUserApiKeyScopeMatches(
        { authMode: "single-user", apiKey: changedKey },
        expectedScope
      )
    ).toBe(false)
  })

  it("allows only Service Prompt and exact scoped execution routes", () => {
    expect(isServicePromptRequestPath("/api/v1/service-prompts/chat.rag.answer", "GET")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/service-prompts/chat.rag.answer", "PUT")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/service-prompts/chat.rag.answer", "DELETE")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chat/completions", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/rag/search", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/research/websearch", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chats/chat-1/messages", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chats/", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/media/add", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/auth/refresh", "POST")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chat/completions", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/rag/search", "DELETE")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/research/websearch", "PATCH")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats/chat-1/messages", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats/", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats", "POST")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/media/add", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/auth/refresh", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/auth/refresh/extra", "POST")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/media/add/extra", "POST")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chat/completions/extra", "POST")).toBe(false)
  })

  it.each([
    "/api/v1/chats/%2e%2e/messages",
    "/api/v1/chats/./messages",
    "/api/v1/chats/../messages",
    "/api/v1/chats/chat%2fid/messages",
    "/api/v1/chats/chat%5cid/messages",
    "/api/v1/chats/chat\\id/messages",
    "/api/v1/chats//messages"
  ])("rejects ambiguous scoped pathname %s", (path) => {
    expect(isServicePromptRequestPath(path, "POST")).toBe(false)
  })

  it("matches only the pathname while leaving query data inert", () => {
    expect(isServicePromptRequestPath(
      "/api/v1/service-prompts?include=defaults",
      "GET"
    )).toBe(true)
    expect(isServicePromptRequestPath(
      "/api/v1/service-prompts/chat.rag.answer?include=defaults",
      "GET"
    )).toBe(true)
    expect(isServicePromptRequestPath(
      "/api/v1/chat/completions?next=%2fapi%2fv1%2fchats%2f..%2fmessages",
      "POST"
    )).toBe(true)
    expect(isServicePromptRequestPath(
      "/api/v1/chats/chat-1/messages?marker=%5c%2e%2e",
      "POST"
    )).toBe(true)
  })

  it("creates the structured scope-change error", () => {
    expect(createServicePromptScopeChangedError()).toMatchObject({
      status: 412,
      details: {
        detail: {
          code: "request_config_scope_changed"
        }
      }
    })
  })
})
