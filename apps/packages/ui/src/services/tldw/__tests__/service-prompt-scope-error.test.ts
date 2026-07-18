import { describe, expect, it } from "vitest"

import {
  createServicePromptScopeChangedError,
  isServicePromptRequestPath,
  servicePromptTargetsMatch
} from "../service-prompt-scope-error"

describe("Service Prompt scope policy", () => {
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
