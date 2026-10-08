import { vi } from "vitest"
import type { ChatSettingsTarget } from "../chat-settings"

export const settingsTestOwner = {
  requestScope: { config: { serverUrl: "https://settings-test.example", authMode: "single-user" as const }, userId: 1 },
  validate_lease: () => true,
  draftAddress: { kind: "fresh" as const, id: "fresh-settings", scope: { type: "global" as const }, temporary: false }
} satisfies Partial<ChatSettingsTarget>

export const installSettingsLocks = () => {
  const tails = new Map<string, Promise<unknown>>()
  vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
    request: (key: string, operation: () => unknown) => {
      const next = (tails.get(key) ?? Promise.resolve()).then(operation)
      tails.set(key, next.catch(() => undefined))
      return next
    }
  } } }))
}
