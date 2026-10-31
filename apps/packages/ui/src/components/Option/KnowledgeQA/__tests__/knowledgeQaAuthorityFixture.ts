import { vi } from "vitest"
import { getKnowledgeQaHistoryStorageKey } from "../historyStorage"

export const TEST_HISTORY_STORAGE_KEY = getKnowledgeQaHistoryStorageKey({
  config: { serverUrl: "https://qa.test", authMode: "multi-user" }, userId: "test-owner",
})

// Existing behavior suites use one verified owner; account-boundary tests exercise
// the real authority hook separately instead of this fixed fixture.
vi.mock("../hooks/useKnowledgeQAAuthority", () => {
  const signal = new AbortController().signal
  const authority = {
    key: "test-owner",
    isCurrent: () => true,
    snapshot: {
      scopeKey: "test-owner",
      requestScope: { config: { serverUrl: "https://qa.test", authMode: "multi-user" }, userId: "test-owner" },
      scopeSignal: signal, scopeInvalidatedSignal: signal, release: vi.fn(),
    },
  }
  return { useKnowledgeQAAuthority: () => authority }
})
