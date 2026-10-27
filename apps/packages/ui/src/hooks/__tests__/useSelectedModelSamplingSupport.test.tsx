import React from "react"
import { renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { describe, expect, it, vi } from "vitest"

const fetchChatModels = vi.hoisted(() => vi.fn(async () => []))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string, values?: Record<string, string>) =>
      fallback.replace(/\{\{(\w+)\}\}/g, (_m, name) => values?.[name] ?? "")
  })
}))

import { chatModelCatalogQueryKey } from "@/hooks/useChatModelCatalog"
import { getChatModelsCacheGeneration } from "@/services/chat-models-invalidation"
import { useSelectedModelSamplingSupport } from "../useSelectedModelSamplingSupport"

const withCatalog = (models: unknown[] | undefined) => {
  const client = new QueryClient()
  if (models) client.setQueryData(chatModelCatalogQueryKey(getChatModelsCacheGeneration()), models)
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

const catalog = [
  { model: "tldw:or-model", provider: "openrouter", sampling_fields: ["top_k", "min_p"] },
  { model: "tldw:gpt-4o", provider: "openai", sampling_fields: [] },
  { model: "tldw:mistral-large", provider: "mistral", sampling_fields: ["top_k"] }
]

// #3283: the settings say when Top K or Min P won't be sent.
describe("useSelectedModelSamplingSupport", () => {
  it("has no reason to give when the provider accepts both", () => {
    const { result } = renderHook(() => useSelectedModelSamplingSupport("openrouter:or-model"), {
      wrapper: withCatalog(catalog)
    })
    expect(result.current).toEqual({ topKUnavailableReason: undefined, minPUnavailableReason: undefined })
  })

  it("names the provider that doesn't accept a setting", () => {
    const { result } = renderHook(() => useSelectedModelSamplingSupport("mistral:mistral-large"), {
      wrapper: withCatalog(catalog)
    })
    expect(result.current).toEqual({
      topKUnavailableReason: undefined,
      minPUnavailableReason: "Not supported by Mistral"
    })
  })

  it("says the setting isn't sent when the catalog doesn't say, and never fetches it", () => {
    const { result } = renderHook(() => useSelectedModelSamplingSupport("openai:gpt-4o"), {
      wrapper: withCatalog(undefined)
    })
    expect(result.current.topKUnavailableReason).toBe(
      "Not sent: the server doesn't list whether this model accepts it"
    )
    expect(fetchChatModels).not.toHaveBeenCalled()
  })
})
