import { describe, expect, it } from "vitest"
import {
  computeRolloutBucket,
  isFlagEnabledForRollout,
  normalizeRolloutPercentage,
  resolveRolloutPercentageFromCandidates
} from "@/utils/feature-rollout"

describe("feature-rollout utilities", () => {
  it("normalizes rollout percentages and clamps out-of-range values", () => {
    expect(normalizeRolloutPercentage("50.8")).toBe(50)
    expect(normalizeRolloutPercentage(-10)).toBe(0)
    expect(normalizeRolloutPercentage(200)).toBe(100)
    expect(normalizeRolloutPercentage("bad", 30)).toBe(30)
  })

  it("resolves rollout percentage from highest-priority valid candidate", () => {
    expect(
      resolveRolloutPercentageFromCandidates([null, "", undefined, "25", "80"], 100)
    ).toBe(25)
    expect(resolveRolloutPercentageFromCandidates(["invalid", "45"], 100)).toBe(45)
    expect(resolveRolloutPercentageFromCandidates([undefined, null], 60)).toBe(60)
  })

  it("produces deterministic buckets", () => {
    const firstBucket = computeRolloutBucket(
      "research_workspace_provenance_v1",
      "subject-a"
    )
    const secondBucket = computeRolloutBucket(
      "research_workspace_provenance_v1",
      "subject-a"
    )
    const differentBucket = computeRolloutBucket(
      "research_workspace_provenance_v1",
      "subject-b"
    )

    expect(firstBucket).toBe(secondBucket)
    expect(firstBucket).toBeGreaterThanOrEqual(0)
    expect(firstBucket).toBeLessThan(100)
    expect(differentBucket).toBeGreaterThanOrEqual(0)
    expect(differentBucket).toBeLessThan(100)
  })

  it("enforces rollout gates consistently for the same subject", () => {
    const flagKey = "research_workspace_status_guardrails_v1"
    const subjectId = "subject-fixed"
    const enabledAt10 = isFlagEnabledForRollout({
      flagKey,
      subjectId,
      rolloutPercentage: 10
    })
    const enabledAt50 = isFlagEnabledForRollout({
      flagKey,
      subjectId,
      rolloutPercentage: 50
    })
    const enabledAt100 = isFlagEnabledForRollout({
      flagKey,
      subjectId,
      rolloutPercentage: 100
    })

    expect(enabledAt10 && !enabledAt50).toBe(false)
    expect(enabledAt100).toBe(true)
  })
})
