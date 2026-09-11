import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  OSCE_DRAFT_TTL_MS,
  clearOsceDraft,
  createOsceSaveQueue,
  osceDraftKey,
  readOsceDraft,
  saveOsceDraft
} from "../osceDraftStore"
import type { OsceCandidateAttempt } from "@/services/osce"

const candidateAttempt = (version: number, notes = ""): OsceCandidateAttempt => ({
  id: 7,
  quiz_id: 3,
  station_id: 11,
  client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  state: "in_progress",
  version,
  notes,
  started_at: "2026-09-11T10:00:00Z",
  last_modified_at: "2026-09-11T10:00:00Z",
  server_time: "2026-09-11T10:02:00Z",
  station: {
    schema_version: "osce.station.v1",
    title: "Medicine safety",
    candidate_instructions: "Speak with the simulated patient.",
    candidate_task: "Explain safe medicine use.",
    patient_context: { text: "A fictional adult recently started treatment.", citations: [] },
    recommended_duration_seconds: 480
  }
})

describe("OSCE local drafts", () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it("stores only writable fields and recursively strips marking-guide content", () => {
    saveOsceDraft("user-42", {
      attemptId: 7,
      version: 2,
      notes: "private",
      checklistSelections: {
        safe: "met",
        nested: { expected_key_points: [{ text: "must not persist" }] }
      },
      rubricSelections: {
        communication: "effective",
        poisoned: { citations: [{ quote: "must not persist" }] }
      },
      expected_key_points: [{ text: "must not persist" }],
      station: { checklist_items: [{ rationale: "must not persist" }] },
      savedAt: 1_000
    } as never)

    const raw = window.localStorage.getItem(osceDraftKey("user-42", 7))
    expect(raw).not.toBeNull()
    expect(raw).not.toContain("expected_key_points")
    expect(raw).not.toContain("checklist_items")
    expect(raw).not.toContain("citations")
    expect(raw).not.toContain("rationale")
    expect(JSON.parse(raw!)).toEqual({
      attemptId: 7,
      version: 2,
      notes: "private",
      checklistSelections: { safe: "met" },
      rubricSelections: { communication: "effective" },
      savedAt: 1_000
    })
  })

  it("scopes keys by user and expires drafts after 24 hours", () => {
    saveOsceDraft("user-a", {
      attemptId: 7,
      version: 2,
      notes: "private",
      checklistSelections: {},
      rubricSelections: {},
      savedAt: 5_000
    })

    expect(readOsceDraft("user-b", 7, { now: 5_001 })).toBeNull()
    expect(readOsceDraft("user-a", 7, { now: 5_000 + OSCE_DRAFT_TTL_MS - 1 })?.notes).toBe("private")
    expect(readOsceDraft("user-a", 7, { now: 5_000 + OSCE_DRAFT_TTL_MS })).toBeNull()
    expect(window.localStorage.getItem(osceDraftKey("user-a", 7))).toBeNull()
  })

  it("serializes patches, writes locally first, and reconciles returned versions", async () => {
    const resolutions: Array<(attempt: OsceCandidateAttempt) => void> = []
    const update = vi.fn((_attemptId: number, patch: unknown) =>
      new Promise<OsceCandidateAttempt>((resolve) => resolutions.push(resolve))
    )
    const acknowledged: number[] = []
    const queue = createOsceSaveQueue({
      attempt: candidateAttempt(2),
      userScope: "user-42",
      update,
      onAcknowledged: (attempt) => acknowledged.push(attempt.version)
    })

    const first = queue.enqueue({ notes: "first" })
    const second = queue.enqueue({ notes: "second" })

    expect(readOsceDraft("user-42", 7)?.notes).toBe("second")
    await Promise.resolve()
    expect(update).toHaveBeenCalledTimes(1)
    expect(update).toHaveBeenNthCalledWith(1, 7, expect.objectContaining({
      expected_version: 2,
      notes: "first"
    }))

    resolutions.shift()?.(candidateAttempt(3, "first"))
    await first
    expect(update).toHaveBeenCalledTimes(2)
    expect(update).toHaveBeenNthCalledWith(2, 7, expect.objectContaining({
      expected_version: 3,
      notes: "second"
    }))
    expect(readOsceDraft("user-42", 7)?.version).toBe(3)

    resolutions.shift()?.(candidateAttempt(4, "second"))
    await second
    await queue.flush()

    expect(acknowledged).toEqual([3, 4])
    expect(queue.getAcknowledgedVersion()).toBe(4)
    expect(queue.hasPending()).toBe(false)
    expect(window.localStorage.getItem(osceDraftKey("user-42", 7))).toBeNull()
  })

  it.each([
    ["disconnect", Object.assign(new Error("offline"), { status: 0 })],
    ["conflict", Object.assign(new Error("conflict"), { status: 409 })]
  ])("retains the latest local draft after %s", async (_label, failure) => {
    const queue = createOsceSaveQueue({
      attempt: candidateAttempt(2),
      userScope: "user-42",
      update: vi.fn().mockRejectedValue(failure)
    })

    await expect(queue.enqueue({ notes: "unsent" })).rejects.toBe(failure)
    await expect(queue.flush()).rejects.toBe(failure)
    expect(readOsceDraft("user-42", 7)?.notes).toBe("unsent")
    expect(queue.hasPending()).toBe(true)

    clearOsceDraft("user-42", 7)
    expect(readOsceDraft("user-42", 7)).toBeNull()
  })
})
