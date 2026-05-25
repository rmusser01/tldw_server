import { describe, expect, it } from "vitest"

import {
  applyAttachedResearchContextEdits,
  buildResearchFollowUpPrompt,
  clearAttachedResearchContext,
  deriveAttachedResearchContext,
  isDeepResearchCompletionMetadata,
  pinAttachedResearchContext,
  resetAttachedResearchContext,
  restorePinnedResearchContext,
  setAttachedResearchContextActive,
  sanitizeAttachedResearchContext,
  unpinAttachedResearchContext,
  type AttachedResearchContext,
  toChatResearchContext
} from "../research-chat-context"

describe("research-chat-context", () => {
  const buildContext = (
    runId: string,
    overrides: Partial<AttachedResearchContext> = {}
  ): AttachedResearchContext => ({
    attached_at: "2026-03-08T20:00:00Z",
    run_id: runId,
    query: `Query for ${runId}`,
    question: `Question for ${runId}`,
    outline: [{ title: `Outline ${runId}` }],
    key_claims: [{ text: `Claim ${runId}` }],
    unresolved_questions: [`Question ${runId}`],
    verification_summary: { unsupported_claim_count: 0 },
    source_trust_summary: { high_trust_count: 1 },
    research_url: `/research?run=${runId}`,
    ...overrides
  })

  it("derives a bounded attached context from a completed research bundle", () => {
    const context = deriveAttachedResearchContext(
      {
        question: "What changed in the battery recycling market?",
        outline: {
          sections: [
            { title: "Overview" },
            { title: "Regional shifts" }
          ]
        },
        claims: [
          { text: "Claim 1" },
          { text: "Claim 2" },
          { text: "Claim 3" },
          { text: "Claim 4" },
          { text: "Claim 5" },
          { text: "Claim 6" }
        ],
        unresolved_questions: [
          "Question 1",
          "Question 2",
          "Question 3",
          "Question 4",
          "Question 5",
          "Question 6"
        ],
        verification_summary: {
          unsupported_claim_count: 2
        },
        source_trust: [
          { source_id: "src_high_1", trust_tier: "high" },
          { source_id: "src_medium", trust_tier: "medium" },
          { source_id: "src_high_2", trust_tier: "HIGH" }
        ],
        report_markdown: "# Full report",
        source_inventory: [{ source_id: "src_high_1" }]
      },
      "run_123",
      "battery recycling supply chain"
    )

    expect(context).toMatchObject({
      run_id: "run_123",
      query: "battery recycling supply chain",
      question: "What changed in the battery recycling market?",
      outline: [{ title: "Overview" }, { title: "Regional shifts" }],
      key_claims: [
        { text: "Claim 1" },
        { text: "Claim 2" },
        { text: "Claim 3" },
        { text: "Claim 4" },
        { text: "Claim 5" }
      ],
      unresolved_questions: [
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5"
      ],
      verification_summary: {
        unsupported_claim_count: 2
      },
      source_trust_summary: {
        high_trust_count: 2
      },
      research_url: "/research?run=run_123"
    })
    expect(context.attached_at).toEqual(expect.any(String))
    expect("report_markdown" in context).toBe(false)
    expect("source_inventory" in context).toBe(false)
    expect(context.key_claims).toHaveLength(5)
    expect(context.unresolved_questions).toHaveLength(5)
  })

  it("recognizes deep research completion handoff metadata", () => {
    expect(
      isDeepResearchCompletionMetadata({
        run_id: "run_123",
        query: "battery recycling supply chain",
        kind: "completion_handoff"
      })
    ).toBe(true)
    expect(
      isDeepResearchCompletionMetadata({
        run_id: "run_123",
        query: "battery recycling supply chain",
        kind: "other"
      })
    ).toBe(false)
    expect(isDeepResearchCompletionMetadata(null)).toBe(false)
  })

  it("builds a deterministic follow-up prompt from the run query", () => {
    expect(buildResearchFollowUpPrompt("Battery recycling supply chain")).toBe(
      "Follow up on this research: Battery recycling supply chain"
    )
    expect(
      buildResearchFollowUpPrompt("  Battery   recycling   supply chain  ")
    ).toBe("Follow up on this research: Battery recycling supply chain")
    expect(buildResearchFollowUpPrompt("   ")).toBe("Follow up on this research")
  })

  it("sanitizes and applies attached research context edits without mutating identity fields", () => {
    const active = {
      attached_at: "2026-03-08T20:00:00Z",
      run_id: "run_123",
      query: "Battery recycling supply chain",
      question: "Battery recycling supply chain",
      outline: [{ title: "Overview" }],
      key_claims: [{ text: "Claim one" }],
      unresolved_questions: ["What changed in Europe?"],
      verification_summary: { unsupported_claim_count: 0 },
      source_trust_summary: { high_trust_count: 2 },
      research_url: "/research?run=run_123"
    }

    const edited = applyAttachedResearchContextEdits(active, {
      question: "  Edited question  ",
      outline: [{ title: " Updated overview " }, { title: "   " }],
      key_claims: [{ text: " Edited claim " }, { text: " " }],
      unresolved_questions: ["  Follow-up  ", ""],
      verification_summary: { unsupported_claim_count: 3 },
      source_trust_summary: { high_trust_count: 5 },
      run_id: "run_mutated",
      query: "mutated query",
      research_url: "/research?run=mutated"
    })

    expect(edited).toMatchObject({
      run_id: "run_123",
      query: "Battery recycling supply chain",
      research_url: "/research?run=run_123",
      question: "Edited question",
      outline: [{ title: "Updated overview" }],
      key_claims: [{ text: "Edited claim" }],
      unresolved_questions: ["Follow-up"],
      verification_summary: { unsupported_claim_count: 3 },
      source_trust_summary: { high_trust_count: 5 }
    })
    expect(toChatResearchContext(edited)).toMatchObject({
      run_id: "run_123",
      question: "Edited question",
      outline: [{ title: "Updated overview" }]
    })
  })

  it("sanitizes a draft and resets back to the run-derived baseline", () => {
    const baseline = {
      attached_at: "2026-03-08T20:00:00Z",
      run_id: "run_123",
      query: "Battery recycling supply chain",
      question: "Battery recycling supply chain",
      outline: [{ title: "Overview" }],
      key_claims: [{ text: "Claim one" }],
      unresolved_questions: ["What changed in Europe?"],
      verification_summary: { unsupported_claim_count: 0 },
      source_trust_summary: { high_trust_count: 2 },
      research_url: "/research?run=run_123"
    }

    expect(
      sanitizeAttachedResearchContext({
        ...baseline,
        question: "   ",
        outline: [{ title: " " }, { title: "Regional shifts" }],
        key_claims: [{ text: " " }, { text: "Claim two" }],
        unresolved_questions: [" ", "Question two"]
      })
    ).toMatchObject({
      question: "Battery recycling supply chain",
      outline: [{ title: "Regional shifts" }],
      key_claims: [{ text: "Claim two" }],
      unresolved_questions: ["Question two"]
    })

    expect(resetAttachedResearchContext(baseline)).toEqual(baseline)
    expect(resetAttachedResearchContext(null)).toBeNull()
  })

  it("canonicalizes the research link from run_id during sanitization", () => {
    expect(
      sanitizeAttachedResearchContext(
        buildContext("run_123", {
          research_url: "https://example.com/not-research"
        })
      )
    ).toMatchObject({
      run_id: "run_123",
      research_url: "/research?run=run_123"
    })
  })

  it("attaching a different run pushes the old active attachment into history and resets baseline", () => {
    const active = buildContext("run_active")
    const prior = buildContext("run_prior", {
      attached_at: "2026-03-08T19:00:00Z"
    })
    const next = buildContext("run_next", {
      attached_at: "2026-03-08T21:00:00Z"
    })

    const transitioned = setAttachedResearchContextActive({
      active,
      baseline: active,
      history: [prior],
      pinned: null,
      nextActive: next
    })

    expect(transitioned.active).toEqual(next)
    expect(transitioned.baseline).toEqual(next)
    expect(transitioned.history.map((entry) => entry.run_id)).toEqual([
      "run_active",
      "run_prior"
    ])
  })

  it("attaching the same run_id updates active and baseline without churning history", () => {
    const active = buildContext("run_same", {
      question: "Original question"
    })
    const history = [buildContext("run_prior")]
    const replacement = buildContext("run_same", {
      question: "Replacement question"
    })

    const transitioned = setAttachedResearchContextActive({
      active,
      baseline: active,
      history,
      pinned: null,
      nextActive: replacement
    })

    expect(transitioned.active?.question).toBe("Replacement question")
    expect(transitioned.baseline?.question).toBe("Replacement question")
    expect(transitioned.history.map((entry) => entry.run_id)).toEqual([
      "run_prior"
    ])
  })

  it("restoring a history entry swaps it into active immediately and keeps history deduped", () => {
    const active = buildContext("run_active")
    const history = [
      buildContext("run_restore", { attached_at: "2026-03-08T21:00:00Z" }),
      buildContext("run_old", { attached_at: "2026-03-08T19:00:00Z" })
    ]

    const transitioned = setAttachedResearchContextActive({
      active,
      baseline: active,
      history,
      pinned: null,
      nextActive: history[0]
    })

    expect(transitioned.active?.run_id).toBe("run_restore")
    expect(transitioned.baseline?.run_id).toBe("run_restore")
    expect(transitioned.history.map((entry) => entry.run_id)).toEqual([
      "run_active",
      "run_old"
    ])
  })

  it("removing the active attachment preserves history and clears baseline", () => {
    const active = buildContext("run_active")
    const history = [buildContext("run_prior")]

    const cleared = clearAttachedResearchContext({
      active,
      baseline: active,
      history,
      pinned: null
    })

    expect(cleared.active).toBeNull()
    expect(cleared.baseline).toBeNull()
    expect(cleared.history.map((entry) => entry.run_id)).toEqual(["run_prior"])
  })

  it("caps history at three entries after repeated swaps", () => {
    const active = buildContext("run_active")
    const history = [
      buildContext("run_hist_1", { attached_at: "2026-03-08T19:00:00Z" }),
      buildContext("run_hist_2", { attached_at: "2026-03-08T18:00:00Z" }),
      buildContext("run_hist_3", { attached_at: "2026-03-08T17:00:00Z" })
    ]
    const next = buildContext("run_next", { attached_at: "2026-03-08T21:00:00Z" })

    const transitioned = setAttachedResearchContextActive({
      active,
      baseline: active,
      history,
      pinned: null,
      nextActive: next
    })

    expect(transitioned.history.map((entry) => entry.run_id)).toEqual([
      "run_active",
      "run_hist_1",
      "run_hist_2"
    ])
  })

  it("pinning the active attachment copies it into the pinned slot without changing active", () => {
    const active = buildContext("run_active")
    const history = [buildContext("run_hist_1")]

    const pinned = pinAttachedResearchContext({
      active,
      baseline: active,
      history,
      pinned: null
    })

    expect(pinned.active?.run_id).toBe("run_active")
    expect(pinned.pinned?.run_id).toBe("run_active")
    expect(pinned.history.map((entry) => entry.run_id)).toEqual(["run_hist_1"])
  })

  it("pinning a history entry copies it into pinned without forcing it active", () => {
    const active = buildContext("run_active")
    const history = [buildContext("run_hist_pin"), buildContext("run_hist_keep")]

    const pinned = pinAttachedResearchContext({
      active,
      baseline: active,
      history,
      pinned: null,
      nextPinned: history[0]
    })

    expect(pinned.active?.run_id).toBe("run_active")
    expect(pinned.pinned?.run_id).toBe("run_hist_pin")
    expect(pinned.history.map((entry) => entry.run_id)).toEqual(["run_hist_keep"])
  })

  it("unpin clears only the pinned slot", () => {
    const active = buildContext("run_active")
    const pinnedAttachment = buildContext("run_pinned")
    const history = [buildContext("run_hist_1")]

    const next = unpinAttachedResearchContext({
      active,
      baseline: active,
      history,
      pinned: pinnedAttachment
    })

    expect(next.active?.run_id).toBe("run_active")
    expect(next.pinned).toBeNull()
    expect(next.history.map((entry) => entry.run_id)).toEqual(["run_hist_1"])
  })

  it("restoring pinned makes it active immediately and resets baseline", () => {
    const active = buildContext("run_active")
    const pinnedAttachment = buildContext("run_pinned")
    const history = [buildContext("run_hist_1")]

    const next = restorePinnedResearchContext({
      active,
      baseline: active,
      history,
      pinned: pinnedAttachment
    })

    expect(next.active?.run_id).toBe("run_pinned")
    expect(next.baseline?.run_id).toBe("run_pinned")
    expect(next.pinned?.run_id).toBe("run_pinned")
    expect(next.history.map((entry) => entry.run_id)).toEqual([
      "run_active",
      "run_hist_1"
    ])
  })

  it("restoring history while pinned exists preserves slot precedence and dedupe", () => {
    const active = buildContext("run_active")
    const pinnedAttachment = buildContext("run_pinned")
    const history = [
      buildContext("run_restore"),
      buildContext("run_pinned"),
      buildContext("run_hist_old")
    ]

    const transitioned = setAttachedResearchContextActive({
      active,
      baseline: active,
      history,
      pinned: pinnedAttachment,
      nextActive: history[0]
    })

    expect(transitioned.active?.run_id).toBe("run_restore")
    expect(transitioned.pinned?.run_id).toBe("run_pinned")
    expect(transitioned.history.map((entry) => entry.run_id)).toEqual([
      "run_active",
      "run_hist_old"
    ])
  })
})
