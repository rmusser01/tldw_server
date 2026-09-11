import { RecipePersistenceRegistry } from "@/services/recipe-persistence-registry"
import { describe, expect, it } from "vitest"

describe("recipe uncertainty registry", () => {
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
})
