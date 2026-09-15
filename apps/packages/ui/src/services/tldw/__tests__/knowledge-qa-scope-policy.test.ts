import { describe, expect, it } from "vitest"
import { isServicePromptRequestPath } from "../service-prompt-scope-error"

describe("bounded Knowledge QA request scope policy", () => {
  it.each([
    ["GET", "/api/v1/characters"], ["GET", "/api/v1/characters/"],
    ["GET", "/api/v1/characters/search?query=Helpful"], ["GET", "/api/v1/characters/search/"],
    ["GET", "/api/v1/chats/owned"], ["DELETE", "/api/v1/chats/owned"],
    ["GET", "/api/v1/chat/conversations?keywords=__knowledge_QA__"],
    ["GET", "/api/v1/chat/conversations/owned"], ["PATCH", "/api/v1/chat/conversations/owned"],
    ["GET", "/api/v1/chat/conversations/owned/messages-with-context?include_rag_context=true"],
    ["POST", "/api/v1/chat/messages/answer/rag-context"],
    ["POST", "/api/v1/chat/conversations/owned/share-links"],
    ["DELETE", "/api/v1/chat/conversations/owned/share-links/share"],
    ["GET", "/api/v1/rag/source-health"], ["POST", "/api/v1/rag/search/stream"],
    ["POST", "/api/v1/chatbooks/export"], ["GET", "/api/v1/chatbooks/download/job"],
    ["POST", "/api/v1/feedback/explicit"],
  ])("permits the captured QA %s %s", (method, path) => {
    expect(isServicePromptRequestPath(path, method)).toBe(true)
  })
  it.each([
    ["POST", "/api/v1/characters"], ["DELETE", "/api/v1/characters/1"],
    ["GET", "/api/v1/characters/1"], ["GET", "/api/v1/characters/search/other"],
    ["GET", "/api/v1/chat/conversations//messages-with-context"],
    ["GET", "/api/v1/chat/conversations/a%2fb/messages-with-context"],
    ["GET", "/api/v1/chat/conversations/%2e%2e/messages-with-context"],
    ["GET", "/api/v1/chat/conversations/owned/messages-with-context/nested"],
    ["POST", "/api/v1/chat/conversations/owned/messages-with-context"],
    ["PUT", "/api/v1/chat/conversations/owned"],
    ["DELETE", "/api/v1/chat/conversations/owned/share-links"],
    ["POST", "/api/v1/chat/conversations/owned/share-links/share"],
    ["GET", "/api/v1/rag/search/stream"], ["POST", "/api/v1/rag/source-health"],
    ["GET", "/api/v1/chatbooks/export"], ["POST", "/api/v1/chatbooks/download/job"],
    ["GET", "/api/v1/chatbooks/download/../job"],
    ["GET", "https://foreign.test/api/v1/chatbooks/download/job"],
    ["GET", "/api/v1/feedback/explicit"], ["POST", "/api/v1/feedback/explicit/"],
    ["POST", "/api/v1/feedback/explicit/nested"], ["POST", "/api/v1/feedback//explicit"],
  ])("rejects expanded QA %s %s", (method, path) => {
    expect(isServicePromptRequestPath(path, method)).toBe(false)
  })
})
