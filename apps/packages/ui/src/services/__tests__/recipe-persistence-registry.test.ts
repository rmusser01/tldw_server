import { RecipePersistenceRegistry } from "@/services/recipe-persistence-registry"
import { describe, expect, it } from "vitest"

describe("recipe uncertainty registry", () => {
  it("provisionally quarantines an exact ID until the matching receipt is acknowledged", () => {
    const registry = new RecipePersistenceRegistry()
    registry.reserve("one", "alice", "operation-1")
    expect(registry.read("one", "bob")).toBe("unknown_owner")
    registry.clearScoped("one", "alice")
    expect(registry.read("one", "bob")).toBe("unknown_owner")
    expect(registry.acknowledge("one", "alice", "operation-1")).toBe(true)
    expect(registry.read("one", "bob")).toBe("clear")
  })

  it.each([
    ["two", "alice", "op-1"],
    ["one", "bob", "op-1"],
    ["one", "alice", "op-old"]
  ])(
    "a mismatched receipt %s/%s/%s cannot acknowledge another operation",
    (id, owner, operation) => {
      const registry = new RecipePersistenceRegistry()
      registry.reserve("one", "alice", "op-1")
      expect(registry.acknowledge(id, owner, operation)).toBe(false)
      expect(registry.read("one", "bob")).toBe("unknown_owner")
    }
  )

  it("acknowledgement keeps scoped/unknown state and old receipts cannot clear a newer reservation", () => {
    const registry = new RecipePersistenceRegistry()
    registry.reserve("one", "alice", "op-1")
    registry.markUnknown("one")
    registry.acknowledge("one", "alice", "op-1")
    expect(registry.read("one", "bob")).toBe("unknown_owner")
    registry.forgetUnknown("one")
    expect(registry.read("one", "alice")).toBe("scoped")
    registry.reserve("one", "bob", "op-2")
    expect(registry.acknowledge("one", "alice", "op-1")).toBe(false)
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    registry.forgetUnknown("two")
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    registry.forgetUnknown("one")
    expect(registry.read("one", "alice")).toBe("scoped")
  })

  it.each(["scoped", "unknown"])(
    "atomically refuses an exact-ID %s reservation without changing other state",
    (state) => {
      const registry = new RecipePersistenceRegistry()
      if (state === "scoped") registry.markScoped("one", "alice")
      else registry.markUnknown("one")
      expect(() => registry.reserve("one", "alice")).toThrow()
      expect(registry.read("two", "alice")).toBe("clear")
      expect(registry.read("one", "alice")).toBe(
        state === "scoped" ? "scoped" : "unknown_owner"
      )
    }
  )
  it("allows one reservation per owner and exact ID, without blocking a different owner", () => {
    const registry = new RecipePersistenceRegistry()
    registry.reserve("one", "alice")
    expect(() => registry.reserve("one", "alice")).toThrow()
    registry.reserve("one", "bob")
    registry.reserve("two", "alice")
    expect(registry.read("one", "bob")).toBe("scoped")
    expect(registry.read("two", "alice")).toBe("scoped")
  })
  it("separates exact IDs and owners and only clears the matching owner", () => {
    const registry = new RecipePersistenceRegistry()
    registry.markScoped(" recipe ", "alice")
    registry.clearScoped(" recipe ", "bob")
    expect(registry.read(" recipe ", "alice")).toBe("scoped")
    expect(registry.read("recipe", "alice")).toBe("clear")
    expect(registry.read(" recipe ", "bob")).toBe("clear")
    registry.clearScoped(" recipe ", "alice")
    expect(registry.read(" recipe ", "alice")).toBe("clear")
  })

  it("quarantines an unknown exact ID for every owner until Forget", () => {
    const registry = new RecipePersistenceRegistry()
    registry.markScoped("one", "alice")
    registry.markUnknown("one")
    for (const owner of ["alice", "bob", null]) {
      expect(registry.read("one", owner)).toBe("unknown_owner")
    }
    registry.clearScoped("one", "bob")
    expect(registry.read("one", "bob")).toBe("unknown_owner")
    expect(registry.read("two", "bob")).toBe("clear")
    registry.forgetUnknown("one")
    expect(registry.read("one", "alice")).toBe("scoped")
    expect(registry.read("one", "bob")).toBe("clear")
  })

  it("starts clean after an application authority restart", () => {
    const old = new RecipePersistenceRegistry()
    old.markScoped("one", "alice")
    old.markUnknown("two")
    const restarted = new RecipePersistenceRegistry()
    expect(restarted.read("one", "alice")).toBe("clear")
    expect(restarted.read("two", "alice")).toBe("clear")
  })

  it("holds exact reconciliation exclusively through its committed release", () => {
    const registry = new RecipePersistenceRegistry()
    registry.markScoped("one", "alice")
    registry.markScoped("two", "alice")

    expect(registry.reconcileExact("one", "alice", "reconcile-1")).toBe(true)
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    expect(() => registry.reserve("one", "alice")).toThrow()
    expect(() => registry.reserve("one", "bob")).toThrow()
    expect(
      registry.finishReconcileExact("one", "alice", "reconcile-1", true)
    ).toBe(true)
    expect(registry.read("one", "alice")).toBe("clear")
    expect(registry.read("two", "alice")).toBe("scoped")
  })

  it("retains matching scoped evidence after an aborted or mismatched reconciliation release", () => {
    const registry = new RecipePersistenceRegistry()

    expect(registry.reconcileExact("one", "alice", "reconcile-1")).toBe(true)
    expect(
      registry.finishReconcileExact("one", "alice", "wrong-token", true)
    ).toBe(false)
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    registry.clearScoped("one", "alice")
    expect(
      registry.finishReconcileExact("one", "alice", "reconcile-1", false)
    ).toBe(true)
    expect(registry.read("one", "alice")).toBe("scoped")
  })

  it.each(["other owner", "unknown", "provisional"])(
    "refuses exact reconciliation while %s uncertainty remains",
    (state) => {
      const registry = new RecipePersistenceRegistry()
      if (state !== "provisional") registry.markScoped("one", "alice")
      if (state === "other owner") registry.markScoped("one", "bob")
      if (state === "unknown") registry.markUnknown("one")
      if (state === "provisional")
        registry.reserve("one", "alice", "operation-1")

      expect(registry.reconcileExact("one", "alice", "reconcile-1")).toBe(
        false
      )
      expect(registry.read("one", "alice")).not.toBe("clear")
      if (state === "other owner")
        expect(registry.read("one", "bob")).toBe("scoped")
    }
  )

  it("holds an exact-ID unlink lease across the local write", () => {
    const registry = new RecipePersistenceRegistry()

    expect(registry.beginExclusive("one", "operation-1")).toBe(true)
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    expect(() => registry.reserve("one", "alice")).toThrow()
    expect(registry.endExclusive("one", "operation-other")).toBe(false)
    expect(registry.read("one", "alice")).toBe("unknown_owner")
    expect(registry.endExclusive("one", "operation-1")).toBe(true)
    expect(registry.read("one", "alice")).toBe("clear")
  })

  it.each(["scoped", "unknown", "provisional"])(
    "refuses an unlink lease while exact-ID %s uncertainty remains",
    (state) => {
      const registry = new RecipePersistenceRegistry()
      if (state === "scoped") registry.markScoped("one", "alice")
      if (state === "unknown") registry.markUnknown("one")
      if (state === "provisional")
        registry.reserve("one", "alice", "operation-1")

      expect(registry.beginExclusive("one", "unlink-operation")).toBe(false)
      expect(registry.read("one", "alice")).not.toBe("clear")
    }
  )
})
