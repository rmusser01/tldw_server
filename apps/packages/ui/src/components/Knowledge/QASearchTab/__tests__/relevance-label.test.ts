// CC-10 review: direct table tests for the plain relevance label.
import { describe, expect, it } from "vitest"

import { chunkScore, relevanceLevel, topChunkScore } from "../relevance-label"

describe("relevance labels", () => {
  it.each([
    ["one result", [0.42], ["best"]],
    ["equal scores", [0.5, 0.5, 0.5], ["best", "best", "best"]],
    ["the last of two distinct results", [0.9, 0.3], ["best", "weaker"]],
    ["a close second and a far third", [1, 0.7, 0.2], ["best", "good", "weaker"]],
    ["a zero score next to a positive one", [0.8, 0], ["best", "weaker"]]
  ])("labels %s", (_name, scores, expected) => {
    const docs = scores.map((score) => ({ score }))
    const top = topChunkScore(docs)
    expect(docs.map((doc) => relevanceLevel(chunkScore(doc), top))).toEqual(
      expected
    )
  })

  it.each([
    ["all zero", [0, 0]],
    ["all negative", [-1, -2]],
    ["not a number", [Number.NaN]],
    ["missing", [undefined]]
  ])("shows no label when scores are %s", (_name, scores) => {
    const docs = scores.map((score) => ({ score }))
    const top = topChunkScore(docs)
    expect(top).toBeNull()
    expect(docs.map((doc) => relevanceLevel(chunkScore(doc), top))).toEqual(
      scores.map(() => null)
    )
  })

  it("does not label a negative score in a positive list as a match", () => {
    expect(relevanceLevel(-0.2, 0.9)).toBe("weaker")
    expect(relevanceLevel(Number.NaN, 0.9)).toBeNull()
  })

  it("falls back to relevance when score is missing", () => {
    expect(chunkScore({ relevance: 0.6 })).toBe(0.6)
    expect(chunkScore({ score: Number.POSITIVE_INFINITY })).toBeNull()
  })
})
