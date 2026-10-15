import { vi } from "vitest"
import type { ChatSettingsTarget } from "../chat-settings"

export const settingsTestOwner = {
  requestScope: { config: { serverUrl: "https://settings-test.example", authMode: "single-user" as const }, userId: 1 },
  validate_lease: () => true,
  draftAddress: { kind: "fresh" as const, id: "fresh-settings", scope: { type: "global" as const }, temporary: false }
} satisfies Partial<ChatSettingsTarget>

export const installSettingsLocks = () => {
  const tails = new Map<string, Promise<unknown>>()
  const locks = {
    request: (key: string, operation: () => unknown) => {
      const next = (tails.get(key) ?? Promise.resolve()).then(operation)
      tails.set(
        key,
        next.catch(() => undefined)
      )
      return next
    }
  }
  vi.stubGlobal(
    "navigator",
    new Proxy(window.navigator, {
      get: (original, key) =>
        key === "locks" ? locks : Reflect.get(original, key, original)
    })
  )
}
