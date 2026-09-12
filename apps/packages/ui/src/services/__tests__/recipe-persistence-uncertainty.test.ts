import { beforeEach, describe, expect, it, vi } from "vitest"

import { deriveRecipePersistenceOwner } from "../recipe-persistence-owner"

const view = (server: string, principal: string, revision = "1") =>
  deriveRecipePersistenceOwner(
    {
      effectiveBase: server,
      authMode: "multi-user",
      authSource: "manual_bearer",
      orgId: null,
      principalKind: "user",
      principal
    },
    revision
  )
describe("async recipe uncertainty authority", () => {
  beforeEach(() => vi.resetModules())
  it.each([
    ["backend", "https://b.test", "alice"],
    ["principal", "https://a.test", "bob"]
  ])("does not clear another %s", async (_label, server, principal) => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = view("https://a.test", "alice").ownerId
    const other = view(server, principal).ownerId
    await registry.markRecipePersistenceScoped("same-id", owner)
    await registry.clearRecipePersistenceScoped("same-id", other)
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", other)
    ).toBe("clear")
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", owner)
    ).toBe("scoped")
  })
  it("same-principal refresh retains owner and cleanup is exact-ID only", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = view("https://a.test", "alice", "1").ownerId
    const refreshed = view("https://a.test", "alice", "2").ownerId
    await registry.markRecipePersistenceScoped("same-id", owner)
    await registry.markRecipePersistenceScoped("other-id", owner)
    await registry.clearRecipePersistenceScoped("same-id", refreshed)
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", owner)
    ).toBe("clear")
    expect(
      await registry.readRecipePersistenceUncertainty("other-id", owner)
    ).toBe("scoped")
  })
  it("reads unknown quarantine even without an owner and only exact Forget clears it", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    await registry.markRecipePersistenceUnknown("same-id")
    await registry.forgetRecipePersistenceUnknown("other-id")
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", null)
    ).toBe("unknown_owner")
    await registry.forgetRecipePersistenceUnknown("same-id")
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", null)
    ).toBe("clear")
  })

  it("reports only a boolean while refusing cross-owner exact reconciliation", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = view("https://a.test", "alice").ownerId
    const other = view("https://a.test", "bob").ownerId
    await registry.markRecipePersistenceScoped("same-id", owner)
    await registry.markRecipePersistenceScoped("same-id", other)

    expect(
      await registry.reconcileRecipePersistenceExact("same-id", owner)
    ).toBe(false)
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", owner)
    ).toBe("scoped")
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", other)
    ).toBe("scoped")
  })

  it("holds and releases an exact unlink lease without clearing uncertainty", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const owner = view("https://a.test", "alice").ownerId
    await registry.markRecipePersistenceScoped("same-id", owner)
    expect(
      await registry.beginRecipePersistenceUnlink(
        "same-id",
        "00000000-0000-4000-8000-000000000001"
      )
    ).toBe(false)
    await registry.clearRecipePersistenceScoped("same-id", owner)
    expect(
      await registry.beginRecipePersistenceUnlink(
        "same-id",
        "00000000-0000-4000-8000-000000000001"
      )
    ).toBe(true)
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", owner)
    ).toBe("unknown_owner")
    await registry.endRecipePersistenceUnlink(
      "same-id",
      "00000000-0000-4000-8000-000000000001"
    )
    expect(
      await registry.readRecipePersistenceUncertainty("same-id", owner)
    ).toBe("clear")
  })
})
