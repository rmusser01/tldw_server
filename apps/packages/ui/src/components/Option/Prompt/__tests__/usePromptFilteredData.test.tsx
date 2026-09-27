import React from "react"
import { cleanup, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, expect, it } from "vitest"

import {
  usePromptFilteredData,
  type UsePromptFilteredDataDeps
} from "../hooks/usePromptFilteredData"

afterEach(cleanup)

it("keeps missing creation dates unknown instead of inventing a render timestamp", () => {
  const client = new QueryClient()
  const deps: UsePromptFilteredDataDeps = {
    data: [{ id: "legacy", name: "Legacy prompt", content: "Hello" }],
    isOnline: false,
    normalizedSearchText: "",
    shouldUseServerSearch: false,
    projectFilter: null,
    typeFilter: "all",
    syncFilter: "all",
    usageFilter: "all",
    tagFilter: [],
    tagMatchMode: "any",
    savedView: "all",
    selectedCollection: null,
    currentPage: 1,
    resultsPerPage: 10,
    promptSort: { key: null, order: null },
    getPromptKeywords: () => [],
    getPromptTexts: () => ({ systemText: undefined, userText: "Hello" }),
    getPromptType: () => "quick",
    getPromptModifiedAt: () => 0,
    getPromptUsageCount: () => 0,
    getPromptLastUsedAt: () => null,
    t: (_key, options) => options?.defaultValue || _key
  }
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const { result, unmount } = renderHook(() => usePromptFilteredData(deps), {
    wrapper
  })
  expect(result.current.customPromptRows[0]?.createdAt).toBe(0)
  unmount()
  client.clear()
})
