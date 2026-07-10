import { describe, expect, it } from "vitest"
import {
  buildShuffledIndexOrder,
  buildShuffledOptionEntries,
  drawDeterministicGroupedQuestionPool,
  drawDeterministicQuestionPool
} from "../optionShuffle"

describe("optionShuffle", () => {
  it("builds a deterministic permutation for a given seed", () => {
    const first = buildShuffledIndexOrder(5, 12345)
    const second = buildShuffledIndexOrder(5, 12345)

    expect(first).toEqual(second)
    expect([...first].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4])
  })

  it("returns stable shuffled options per question and session seed", () => {
    const options = ["A", "B", "C", "D"]
    const first = buildShuffledOptionEntries(options, 7, 42)
    const second = buildShuffledOptionEntries(options, 7, 42)

    expect(first).toEqual(second)
    expect(first).toHaveLength(4)
    expect(first.map((entry) => entry.label).sort()).toEqual(["A", "B", "C", "D"])
  })

  it("uses question id to vary shuffles within the same session", () => {
    const options = ["A", "B", "C", "D"]
    const one = buildShuffledOptionEntries(options, 11, 2026).map((entry) => entry.originalIndex)
    const two = buildShuffledOptionEntries(options, 12, 2026).map((entry) => entry.originalIndex)

    expect(one).not.toEqual(two)
  })

  it("handles empty and single-option inputs", () => {
    expect(buildShuffledOptionEntries([], 1, 1)).toEqual([])
    expect(buildShuffledOptionEntries(["Only"], 1, 1)).toEqual([
      { originalIndex: 0, label: "Only" }
    ])
  })

  it("draws deterministic question pools from larger sets", () => {
    const questions = ["q1", "q2", "q3", "q4", "q5", "q6"]
    const first = drawDeterministicQuestionPool(questions, 3, 1234)
    const second = drawDeterministicQuestionPool(questions, 3, 1234)

    expect(first).toEqual(second)
    expect(first).toHaveLength(3)
    expect(new Set(first).size).toBe(3)
    first.forEach((entry) => {
      expect(questions).toContain(entry)
    })
  })

  it("returns full set when draw count exceeds question count", () => {
    const questions = ["q1", "q2", "q3"]
    expect(drawDeterministicQuestionPool(questions, 10, 2026)).toEqual(questions)
  })

  it("draws grouped stems atomically while preserving member order", () => {
    const questions = [
      { id: "a-1", groupId: "group-a" },
      { id: "b-1", groupId: "group-b" },
      { id: "a-2", groupId: "group-a" },
      { id: "b-2", groupId: "group-b" }
    ]
    const first = drawDeterministicGroupedQuestionPool(questions, 1, 2026, (question) => question.groupId)
    const second = drawDeterministicGroupedQuestionPool(questions, 1, 2026, (question) => question.groupId)
    const selectedGroup = first[0]?.groupId

    expect(first).toEqual(second)
    expect(first).toHaveLength(2)
    expect(first.every((question) => question.groupId === selectedGroup)).toBe(true)
    expect(first.map((question) => question.id)).toEqual(
      questions
        .filter((question) => question.groupId === selectedGroup)
        .map((question) => question.id)
    )
  })

  it("keeps ungrouped draws compatible with the existing deterministic pool", () => {
    const questions = ["q1", "q2", "q3", "q4", "q5"]

    expect(drawDeterministicGroupedQuestionPool(questions, 3, 77, () => null)).toEqual(
      drawDeterministicQuestionPool(questions, 3, 77)
    )
  })
})
