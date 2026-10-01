import {
  buildChatSurfaceScopeKeyFromConfig,
  deriveSingleUserApiKeyCredentialScope,
} from "@/services/chat-surface-scope"
import { describe, expect, it } from "vitest"

import {
  createServicePromptScopeChangedError,
  isServicePromptRequestPath,
  servicePromptSingleUserApiKeyScopeMatches,
  servicePromptTargetsMatch
} from "../service-prompt-scope-error"
describe("Service Prompt scope policy", () => {
  it.each([
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc?scope_type=global&include_history_recovery_v1=true", "GET", true],
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc", "POST", false],
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc", "PUT", false],
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc", "DELETE", false],
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc/", "GET", false],
    ["/api/v1/messages/12345678-1234-4123-8123-123456789abc/extra", "GET", false],
    ["/api/v1/messages/not-a-uuid", "GET", false],
    ["/api/v1/messages/a%2fb", "GET", false],
    ["/api/v1/messages/%2e%2e", "GET", false],
  ])("bounds the captured recovery message read %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })
  it.each([
    "/api/v1/workspaces/ws-1", "/api/v1/workspaces/ws-1/sources",
    "/api/v1/workspaces/ws-1/artifacts", "/api/v1/workspaces/ws-1/notes"
  ])("allows only GET for captured activation read %s", (path) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(true)
    for (const method of ["POST", "PUT", "PATCH", "DELETE"]) {
      expect(isServicePromptRequestPath(path, method)).toBe(false)
    }
  })
  it.each(["/api/v1/workspaces", "/api/v1/workspaces/"])(
    "allows only GET for the canonical workspace list %s",
    (path) => {
      expect(isServicePromptRequestPath(path, "GET")).toBe(true)
      for (const method of ["POST", "PUT", "PATCH", "DELETE"]) {
        expect(isServicePromptRequestPath(path, method)).toBe(false)
      }
    }
  )
  it.each([
    "/api/v1/workspaces/ws-1/settings",
    "/api/v1/workspaces/ws-1/sources/status", "/api/v1/workspaces/ws-1/notes/extra",
    "/api/v1/workspaces/a%2fb", "/api/v1/workspaces/a%5cb/notes",
    "/api/v1/workspaces/%2e%2e/artifacts", "/api/v1/workspaces/../sources",
    "/api/v1/workspaces//sources", "/api/v1/workspaces/ws-1/notes/"
  ])("rejects noncanonical or unapproved activation read %s", (path) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(false)
  })
  it.each([
    ["/api/v1/workspaces/ws-1/sources/s1/preview?max_chars=3000&chunk_limit=3", "GET", true],
    ["/api/v1/workspaces/ws%20one/sources/source%20one/preview", "GET", true],
    ["/api/v1/workspaces/ws-1/sources/s1/preview", "POST", false],
    ["/api/v1/workspaces/ws-1/sources/s1/preview", "DELETE", false],
    ["/api/v1/workspaces/ws-1/sources/s1", "GET", false],
    ["/api/v1/workspaces/ws-1/sources/s1/preview/extra", "GET", false],
    ["/api/v1/workspaces/ws-1/sources/a%2fb/preview", "GET", false],
    ["/api/v1/workspaces/%2e%2e/sources/s1/preview", "GET", false],
    ["/api/v1/workspaces/ws-1/sources//preview", "GET", false]
  ])("bounds captured source preview %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })
  it.each([
    ["/api/v1/media/389", "PUT", true],
    ["/api/v1/media/389/metadata", "PATCH", true],
    ["/api/v1/media/389/reprocess", "POST", true],
    ["/api/v1/media/389/metadata", "PUT", false],
    ["/api/v1/media/389/reprocess", "GET", false],
    ["/api/v1/media/389/reprocess/extra", "POST", false],
    ["/api/v1/media/389%2fother", "PUT", false],
    ["/api/v1/media/../settings", "PUT", false],
  ])("bounds owned Content Review commit %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })

  it.each([
    ["/api/v1/chats/owned/complete-v2", "POST", true],
    ["/api/v1/chats/owned/complete-v2?scope_type=workspace&workspace_id=w", "POST", true],
    ["/api/v1/chats/owned/complete-v2", "GET", false],
    ["/api/v1/chats/owned/complete-v2", "PUT", false],
    ["/api/v1/chats/owned/complete", "POST", false],
    ["/api/v1/chats/owned/complete-v2/extra", "POST", false],
    ["/api/v1/chats//complete-v2", "POST", false],
    ["/api/v1/chats/a%2fb/complete-v2", "POST", false],
    ["/api/v1/chats/%2e%2e/complete-v2", "POST", false]
  ])("bounds captured Character generation %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })

  it.each([
    ["/api/v1/chats/owned/completions/persist", "POST", true],
    ["/api/v1/chats/other-owned/completions/persist?scope_type=global", "POST", true],
    ["/api/v1/chats/owned/completions/persist", "GET", false],
    ["/api/v1/chats/owned/completions/persist", "PUT", false],
    ["/api/v1/chats/owned/completions", "POST", false],
    ["/api/v1/chats/owned/completions/persist/extra", "POST", false],
    ["/api/v1/chats//completions/persist", "POST", false],
    ["/api/v1/chats/a%2fb/completions/persist", "POST", false],
    ["/api/v1/chats/%2e%2e/completions/persist", "POST", false]
  ])("bounds character recovery %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })

  it.each([
    ["/api/v1/chats/owned-chat/messages?limit=200&offset=0", true],
    ["/api/v1/chats/other-chat/messages", true],
    ["/api/v1/chats/owned-chat", true],
    ["/api/v1/chats/owned-chat/messages/other-message", false],
    ["/api/v1/chats/a%2fb/messages", false],
    ["/api/v1/chats/%2e%2e/messages", false],
    ["/api/v1/chats//messages", false]
  ])("bounds the scoped message-list route %s", (path, allowed) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(allowed)
  })
  it.each([
    ["/api/v1/notes/", "POST", true],
    ["/api/v1/notes/search/?tokens=workspace%3Aowned", "GET", true],
    ["/api/v1/notes/search/", "POST", false],
    ["/api/v1/notes/search/extra", "GET", false],
    ["/api/v1/notes/private-note", "GET", true],
    ["/api/v1/notes/private-note", "PUT", true],
    ["/api/v1/notes/", "GET", false],
    ["/api/v1/notes/private-note", "DELETE", true],
    ["/api/v1/notes/private-note", "PATCH", false],
    ["/api/v1/notes/private-note/attachments", "POST", false],
    ["/api/v1/notes/%2e%2e", "PUT", false],
    ["/api/v1/notes/a%2fb", "PUT", false],
  ])("bounds Notes request %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })
  it.each([
    "/openapi.json",
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
    "/api/v1/chats",
    "/api/v1/chats/%2e%2e",
    "/api/v1/chats/a%2fb",
    "/api/v1/chats/a%5cb",
    "/api/v1/chats/..",
    "/api/v1/chats//chat-1",
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
    expect(isServicePromptRequestPath("/api/v1/chats/chat-1/messages", "GET")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chats/", "GET")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chats/conversations", "GET")).toBe(true)
    expect(isServicePromptRequestPath("/api/v1/chats", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats/conversations/", "GET")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats/conversations", "POST")).toBe(false)
    expect(isServicePromptRequestPath("/api/v1/chats/conversations/nested", "GET")).toBe(false)
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

describe('H1 scoped routes', () => {
  it.each(['/api/v1/chat/conversations/chat/history/selection', '/api/v1/chat/conversations/chat/history/legacy-projection', '/api/v1/chats/chat/completions/persist'])('allows only POST %s', path => {
    expect(isServicePromptRequestPath(path, 'POST')).toBe(true)
    for (const method of ['GET', 'PUT', 'PATCH', 'DELETE']) expect(isServicePromptRequestPath(path, method)).toBe(false)
  })
  it.each(['/api/v1/chat/conversations//history/selection', '/api/v1/chat/conversations/a%2fb/history/selection', '/api/v1/chat/conversations/a/history/selection/extra', '/api/v1/chat/conversations/a/history/legacy-projection/', '/api/v1/chats/a/completions/persist/extra'])('rejects malformed %s', path => {
    expect(isServicePromptRequestPath(path, 'POST')).toBe(false)
  })
})

it.each([
  ["/api/v1/chats/child", "GET", true],
  ["/api/v1/chats/child/settings?scope_type=workspace", "GET", true],
  ["/api/v1/chats/child/settings", "PUT", true],
  ["/api/v1/chats/child", "PUT", true],
  ["/api/v1/chats/child/settings", "POST", false],
  ["/api/v1/chats/child/settings/extra", "PUT", false],
  ["/api/v1/chats/%2e%2e/settings", "PUT", false],
  ["/api/v1/chats/child%2fother/settings", "GET", false]
])("scoped chat path %s %s has exact access %s", (path, method, expected) => {
  expect(isServicePromptRequestPath(path, method)).toBe(expected)
})


it.each([
  ['/api/v1/media/7', 'DELETE', true],
  ['/api/v1/media/7/keywords', 'PATCH', true],
  ['/api/v1/media/bulk/keyword-update', 'POST', true],
  ['/api/v1/media/7/keywords', 'DELETE', false],
  ['/api/v1/media/7/keywords/extra', 'PATCH', false],
  ['/api/v1/media/7/keywords/', 'PATCH', false],
  ['/api/v1/media/7%2fother', 'DELETE', false],
  ['/api/v1/media/%2e%2e', 'DELETE', false],
  ['/api/v1/media/7/permanent', 'DELETE', false],
  ['/api/v1/media/7/extra', 'DELETE', false],
  ['/api/v1/media/bulk/keyword-update/extra', 'POST', false],
  ['/api/v1/media/bulk/keyword-update', 'DELETE', false],
] as const)('bounds owned Review action %s %s', (path, method, allowed) => {
  expect(isServicePromptRequestPath(path, method)).toBe(allowed)
})

it.each([
  ['/api/v1/users/storage', 'GET', true],
  ['/api/v1/users/storage', 'POST', false],
  ['/api/v1/users/storage/other', 'GET', false],
  ['/api/v1/notes/7', 'DELETE', true],
  ['/api/v1/notes/a9b7-uuid', 'DELETE', true],
  ['/api/v1/notes/a9b7-uuid/restore?expected_version=8', 'POST', true],
  ['/api/v1/media/7/restore', 'POST', true],
  ['/api/v1/notes/tasks', 'DELETE', false],
  ['/api/v1/notes/collections', 'DELETE', false],
  ['/api/v1/notes/trash', 'DELETE', false],
  ['/api/v1/notes/tasks/restore', 'POST', false],
  ['/api/v1/notes/collections/restore', 'POST', false],
  ['/api/v1/notes/7/permanent', 'DELETE', false],
  ['/api/v1/notes/7/restore', 'DELETE', false],
  ['/api/v1/media/word/restore', 'POST', false],
  ['/api/v1/media/7/permanent', 'DELETE', false],
] as const)('bounds Inspector recovery %s %s', (path, method, allowed) => {
  expect(isServicePromptRequestPath(path, method)).toBe(allowed)
})
it.each([
  ["/api/v1/media/ingest/jobs?batch_id=known", "GET", true],
  ["/api/v1/media/ingest/jobs", "DELETE", false],
  ["/api/v1/media/ingest/jobs/", "GET", false],
  ["/api/v1/media/ingest/jobs/extra", "GET", false],
  ["/api/v1/media/ingest/jobs%2fextra", "GET", false]
] as const)("bounds recent import reads %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })

describe("exact registered Media listing route variants", () => {
  it.each(["/api/v1/media?page=2", "/api/v1/media/?page=2"])(
    "accepts GET %s",
    (path) => {
      expect(isServicePromptRequestPath(path, "GET")).toBe(true)
    }
  )
  it.each([
    "/api/v1/media//",
    "/api/v1/media/%2e%2e",
    "/api/v1/media/unrelated",
    "/api/v1/media/%2F"
  ])("rejects GET %s", (path) => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(false)
  })
  it.each(["POST", "DELETE", "PATCH", "PUT"])(
    "rejects %s listing",
    (method) => {
      expect(isServicePromptRequestPath("/api/v1/media/", method)).toBe(false)
    }
  )
})
