import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useHomeMilestoneScope } from "../useHomeMilestoneScope"

const config = vi.hoisted(() => ({
  current: null as Record<string, unknown> | null
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    getConfig: async () => config.current
  }
}))
const token = (sub: string) =>
  `header.${btoa(JSON.stringify({ sub }))}.signature`
beforeEach(() => {
  config.current = {
    serverUrl: "http://server-a",
    authMode: "multi-user",
    accessToken: token("alice")
  }
})

it("changes scope on account and server changes and restores the same identity", async () => {
  const { result, unmount } = renderHook(() => useHomeMilestoneScope())
  await waitFor(() => expect(result.current).not.toBeNull())
  const alice = result.current
  await act(async () => {
    config.current = { ...config.current, accessToken: token("bob") }
    window.dispatchEvent(new Event("tldw:config-updated"))
  })
  expect(result.current).not.toBe(alice)
  await act(async () => {
    config.current = {
      ...config.current,
      serverUrl: "http://server-b",
      accessToken: token("alice")
    }
    window.dispatchEvent(new Event("tldw:config-updated"))
  })
  expect(result.current).not.toBe(alice)
  unmount()
  config.current = { ...config.current, serverUrl: "http://server-a" }
  const again = renderHook(() => useHomeMilestoneScope())
  await waitFor(() => expect(again.result.current).toBe(alice))
})

it("does not assign an unknown multi-user token to a shared anonymous scope", async () => {
  config.current = { ...config.current, accessToken: "opaque-unknown" }
  const { result } = renderHook(() => useHomeMilestoneScope())
  await act(async () => {})
  expect(result.current).toBeNull()
})

it("invalidates logout and refreshes account ownership on another tab's config update", async () => {
  const { result } = renderHook(() => useHomeMilestoneScope())
  await waitFor(() => expect(result.current).not.toBeNull())
  const alice = result.current
  await act(async () => {
    window.dispatchEvent(
      new CustomEvent("tldw:auth-principal-changed", {
        detail: { kind: "logout" }
      })
    )
  })
  expect(result.current).toBeNull()
  await act(async () => {
    config.current = { ...config.current, accessToken: token("bob") }
    window.dispatchEvent(
      new StorageEvent("storage", { key: "plasmo-local:tldwConfig" })
    )
  })
  await waitFor(() => expect(result.current).not.toBeNull())
  expect(result.current).not.toBe(alice)
})
