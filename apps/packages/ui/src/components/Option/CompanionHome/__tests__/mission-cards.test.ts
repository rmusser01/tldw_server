import { describe, expect, it } from "vitest"
import { MISSION_CARDS } from "../mission-cards"

describe("mission card registry", () => {
  it("has cards for all three personas", () => {
    const familyCards = MISSION_CARDS.filter((c) =>
      c.persona === "all" || (Array.isArray(c.persona) && c.persona.includes("family"))
    )
    const researcherCards = MISSION_CARDS.filter((c) =>
      c.persona === "all" || (Array.isArray(c.persona) && c.persona.includes("researcher"))
    )
    const explorerCards = MISSION_CARDS.filter((c) =>
      c.persona === "all" || (Array.isArray(c.persona) && c.persona.includes("explorer"))
    )
    expect(familyCards.length).toBeGreaterThanOrEqual(3)
    expect(researcherCards.length).toBeGreaterThanOrEqual(3)
    expect(explorerCards.length).toBeGreaterThanOrEqual(2)
  })

  it("all cards have required fields", () => {
    for (const card of MISSION_CARDS) {
      expect(card.id).toBeTruthy()
      expect(card.title).toBeTruthy()
      expect(card.href).toBeTruthy()
      expect(card.icon).toBeDefined()
      expect(card.category).toBeTruthy()
      expect(typeof card.priority).toBe("number")
    }
  })

  it("every card has at least one prerequisite milestone", () => {
    for (const card of MISSION_CARDS) {
      expect(card.prerequisiteMilestones.length).toBeGreaterThanOrEqual(1)
    }
  })

  it("cards are unique by id", () => {
    const ids = MISSION_CARDS.map((c) => c.id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})
