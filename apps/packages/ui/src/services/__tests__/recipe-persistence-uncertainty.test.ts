import { beforeEach, describe, expect, it, vi } from "vitest"
import { buildChatSurfaceScopeKeyFromConfig } from "../chat-surface-scope"

const scope = (serverUrl: string, sub: string, exp = 1) =>
  buildChatSurfaceScopeKeyFromConfig({
    serverUrl,
    authMode: "multi-user",
    accessToken: `header.${btoa(JSON.stringify({ sub, exp }))}.signature`
  })

describe("recipe uncertainty ownership", () => {
  beforeEach(() => vi.resetModules())

  it.each([
    ["backend", scope("https://b.test", "alice")],
    ["principal", scope("https://a.test", "bob")]
  ])("isolates the same local ID across a different %s", async (_, other) => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = scope("https://a.test", "alice")
    registry.markRecipePersistenceUncertain("same-id", owner)
    expect(registry.isRecipePersistenceUncertain("same-id", other)).toBe(false)
    registry.clearRecipePersistenceUncertainty("same-id", other)
    expect(registry.isRecipePersistenceUncertain("same-id", owner)).toBe(true)
  })

  it("retains the owner across same-sub refresh and clears only the matching ID", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const original = scope("https://a.test", "alice", 1)
    const refreshed = scope("https://a.test", "alice", 2)
    registry.markRecipePersistenceUncertain("same-id", original)
    registry.markRecipePersistenceUncertain("other-id", original)
    expect(registry.isRecipePersistenceUncertain("same-id", refreshed)).toBe(
      true
    )
    registry.clearRecipePersistenceUncertainty("same-id", refreshed)
    expect(registry.isRecipePersistenceUncertain("same-id", original)).toBe(
      false
    )
    expect(registry.isRecipePersistenceUncertain("other-id", original)).toBe(
      true
    )
  })

  it("fails closed without an owner and cannot clear an owned marker", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = scope("https://a.test", "alice")
    registry.markRecipePersistenceUncertain("same-id", owner)
    registry.clearRecipePersistenceUncertainty("same-id", null)
    expect(registry.isRecipePersistenceUncertain("same-id", owner)).toBe(true)
    expect(registry.isRecipePersistenceUncertain("unseen-id", null)).toBe(true)
  })

  it("starts with no markers after a fresh module lifecycle", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    expect(
      registry.isRecipePersistenceUncertain(
        "same-id",
        scope("https://a.test", "alice")
      )
    ).toBe(false)
  })
})
