import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS,
  FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
  isFlagEnabledForRollout
} from "@/utils/feature-rollout"
import { FEATURE_FLAGS, useFeatureFlag } from "../useFeatureFlags"

const useStorageMock = vi.hoisted(() => vi.fn())

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => useStorageMock(...args)
}))

describe("useFeatureFlag rollout controls", () => {
  beforeEach(() => {
    useStorageMock.mockReset()
    useStorageMock.mockImplementation(
      (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()] as const
    )
    window.localStorage.clear()
    delete (
      window as Window & { __TLDW_RESEARCH_WORKSPACE_ROLLOUT__?: unknown }
    ).__TLDW_RESEARCH_WORKSPACE_ROLLOUT__
  })

  it("respects persisted manual disable even when rollout assignment passes", () => {
    useStorageMock.mockImplementation(
      (key: string, defaultValue: unknown) => [
        key === FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1 ? false : defaultValue,
        vi.fn()
      ] as const
    )
    window.localStorage.setItem(
      FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
      "manual-disable-subject"
    )
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1,
      "100"
    )

    const { result } = renderHook(() =>
      useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1)
    )
    expect(result.current[0]).toBe(false)
  })

  it("blocks a rollout-gated flag when percentage is 0", () => {
    useStorageMock.mockImplementation(() => [true, vi.fn()] as const)
    window.localStorage.setItem(FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY, "subject-a")
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_workspace_status_guardrails_v1,
      "0"
    )

    const { result } = renderHook(() =>
      useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1)
    )
    expect(result.current[0]).toBe(false)
  })

  it("applies deterministic cohort assignment at intermediate percentages", () => {
    useStorageMock.mockImplementation(() => [true, vi.fn()] as const)
    const subjectId = "cohort-subject-42"
    const percentage = 35
    window.localStorage.setItem(FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY, subjectId)
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_workspace_status_guardrails_v1,
      String(percentage)
    )

    const expected = isFlagEnabledForRollout({
      flagKey: FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1,
      subjectId,
      rolloutPercentage: percentage
    })

    const { result } = renderHook(() =>
      useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1)
    )
    expect(result.current[0]).toBe(expected)
  })

  it("uses runtime window override before storage/env rollout values", () => {
    useStorageMock.mockImplementation(() => [true, vi.fn()] as const)
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1,
      "0"
    )
    ;(
      window as Window & {
        __TLDW_RESEARCH_WORKSPACE_ROLLOUT__?: Record<string, unknown>
      }
    ).__TLDW_RESEARCH_WORKSPACE_ROLLOUT__ = {
      [FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1]: 100
    }

    const { result } = renderHook(() =>
      useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1)
    )
    expect(result.current[0]).toBe(true)
  })

  it("does not apply rollout gating to unrelated flags", () => {
    useStorageMock.mockImplementation(() => [true, vi.fn()] as const)
    window.localStorage.setItem(
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_workspace_status_guardrails_v1,
      "0"
    )

    const { result } = renderHook(() => useFeatureFlag(FEATURE_FLAGS.NEW_CHAT))
    expect(result.current[0]).toBe(true)
  })
})
