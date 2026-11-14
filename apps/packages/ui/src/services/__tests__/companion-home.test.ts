import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchCompanionWorkspaceSnapshot: vi.fn(),
  getReadingList: vi.fn(),
  listNotes: vi.fn()
}))

vi.mock("../companion", () => ({
  fetchCompanionWorkspaceSnapshot: (...args: unknown[]) =>
    mocks.fetchCompanionWorkspaceSnapshot(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getReadingList: (...args: unknown[]) => mocks.getReadingList(...args),
    listNotes: (...args: unknown[]) => mocks.listNotes(...args)
  }
}))

import { fetchCompanionHomeSnapshot } from "../companion-home"

const buildWorkspaceSnapshot = (overrides: Record<string, unknown> = {}) => ({
  activity: [
    {
      id: "activity-1",
      event_type: "reading.saved",
      source_type: "reading_item",
      source_id: "reading-1",
      surface: "reading",
      tags: ["research"],
      provenance: {},
      metadata: {
        title: "Inbox capture",
        summary: "Captured while researching."
      },
      created_at: "2026-03-18T09:00:00Z"
    }
  ],
  activityTotal: 1,
  knowledge: [],
  knowledgeTotal: 0,
  goals: [
    {
      id: "goal-1",
      title: "Finish queue review",
      description: "Review the saved queue.",
      goal_type: "reading_backlog",
      config: {},
      progress: {},
      status: "active",
      created_at: "2026-03-01T09:00:00Z",
      updated_at: "2026-03-01T09:00:00Z"
    }
  ],
  activeGoalCount: 1,
  reflections: [
    {
      id: "reflection-1",
      cadence: "daily",
      title: "Daily reflection",
      summary: "You should revisit the reading queue.",
      evidence: [{ source_id: "reading-1" }],
      follow_up_prompts: [],
      provenance: {},
      created_at: "2026-03-18T10:00:00Z"
    }
  ],
  reflectionNotifications: [],
  inbox: [
    {
      id: 201,
      kind: "companion_reflection",
      title: "Daily reflection",
      message: "You should revisit the reading queue.",
      severity: "info",
      link_type: "companion_reflection",
      link_id: "reflection-1",
      created_at: "2026-03-18T10:05:00Z",
      read_at: null,
      dismissed_at: null
    }
  ],
  ...overrides
})

describe("fetchCompanionHomeSnapshot", () => {
  it("preserves reading and notes without requesting an absent Companion router", async () => {
    const snapshot = await fetchCompanionHomeSnapshot("options", { companionAvailable: false })
    expect(mocks.fetchCompanionWorkspaceSnapshot).not.toHaveBeenCalled()
    expect(mocks.getReadingList).toHaveBeenCalledTimes(1)
    expect(mocks.listNotes).toHaveBeenCalledTimes(1)
    expect(snapshot.readingQueue).toEqual(expect.arrayContaining([expect.objectContaining({ entityId: "reading-1" })]))
    expect(snapshot.resumeWork).toEqual(expect.arrayContaining([expect.objectContaining({ entityId: "note-1" })]))
    expect(snapshot.goalsFocus).toEqual([])
    expect(snapshot.degradedSources).toContain("workspace")
  })

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-20T12:00:00Z"))

    mocks.fetchCompanionWorkspaceSnapshot.mockResolvedValue(buildWorkspaceSnapshot())
    mocks.getReadingList.mockResolvedValue({
      items: [
        {
          id: "reading-1",
          title: "Queue article",
          url: "https://example.com/queue",
          status: "saved",
          favorite: false,
          tags: ["research"],
          created_at: "2026-03-05T08:00:00Z",
          updated_at: "2026-03-05T08:00:00Z"
        }
      ],
      total: 1,
      page: 1,
      size: 25
    })
    mocks.listNotes.mockResolvedValue({
      items: [
        {
          id: "note-1",
          title: "Draft outline",
          content: "Turn the queue review into a checklist.",
          status: "draft",
          updated_at: "2026-03-04T12:00:00Z"
        }
      ],
      total: 1
    })
  })

  it("aggregates inbox, goals, reading, notes, and activity into a home snapshot", async () => {
    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot).toMatchObject({
      inbox: expect.any(Array),
      needsAttention: expect.any(Array),
      resumeWork: expect.any(Array),
      goalsFocus: expect.any(Array),
      recentActivity: expect.any(Array),
      readingQueue: expect.any(Array)
    })
    expect(snapshot.inbox).toEqual([
      expect.objectContaining({
        entityId: "reflection-1",
        source: "canonical_inbox"
      })
    ])
    expect(snapshot.needsAttention).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "goal-1", source: "goal" }),
        expect.objectContaining({ entityId: "reading-1", source: "reading" }),
        expect.objectContaining({ entityId: "note-1", source: "note" })
      ])
    )
    expect(snapshot.resumeWork).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "goal-1", source: "goal" }),
        expect.objectContaining({ entityId: "reading-1", source: "reading" }),
        expect.objectContaining({ entityId: "note-1", source: "note" })
      ])
    )
    expect(snapshot.goalsFocus).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "goal-1", source: "goal" })
      ])
    )
    expect(snapshot.recentActivity).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "reading-1", source: "reading" })
      ])
    )
    expect(snapshot.readingQueue).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "reading-1", source: "reading" })
      ])
    )
  })

  it("dedupes needs-attention items when a canonical inbox item already exists", async () => {
    mocks.fetchCompanionWorkspaceSnapshot.mockResolvedValueOnce(
      buildWorkspaceSnapshot({
        inbox: [
          {
            id: 301,
            kind: "goal_prompt",
            title: "Finish queue review",
            message: "Goal needs attention.",
            severity: "info",
            link_type: "companion_goal",
            link_id: "goal-1",
            created_at: "2026-03-19T10:00:00Z",
            read_at: null,
            dismissed_at: null
          }
        ]
      })
    )

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.needsAttention).not.toContainEqual(
      expect.objectContaining({ entityId: "goal-1" })
    )
  })

  it("dedupes resume-work items when a canonical inbox item already exists", async () => {
    mocks.fetchCompanionWorkspaceSnapshot.mockResolvedValueOnce(
      buildWorkspaceSnapshot({
        inbox: [
          {
            id: 302,
            kind: "goal_prompt",
            title: "Finish queue review",
            message: "Goal is waiting for attention.",
            severity: "info",
            link_type: "companion_goal",
            link_id: "goal-1",
            created_at: "2026-03-19T10:00:00Z",
            read_at: null,
            dismissed_at: null
          }
        ]
      })
    )

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.resumeWork).not.toContainEqual(
      expect.objectContaining({ entityId: "goal-1" })
    )
  })

  it("does not suppress unrelated derived items when inbox ids overlap across types", async () => {
    mocks.fetchCompanionWorkspaceSnapshot.mockResolvedValueOnce(
      buildWorkspaceSnapshot({
        inbox: [
          {
            id: 303,
            kind: "goal_prompt",
            title: "Goal with shared id",
            message: "A goal uses the shared entity id.",
            severity: "info",
            link_type: "companion_goal",
            link_id: "shared-42",
            created_at: "2026-03-19T10:00:00Z",
            read_at: null,
            dismissed_at: null
          }
        ]
      })
    )
    mocks.getReadingList.mockResolvedValueOnce({
      items: [
        {
          id: "shared-42",
          title: "Shared id article",
          url: "https://example.com/shared",
          status: "saved",
          favorite: false,
          tags: ["research"],
          created_at: "2026-03-05T08:00:00Z",
          updated_at: "2026-03-05T08:00:00Z"
        }
      ],
      total: 1,
      page: 1,
      size: 25
    })
    mocks.listNotes.mockResolvedValueOnce({
      items: [
        {
          id: "shared-42",
          title: "Shared id note",
          content: "A note reuses the same raw id as the goal.",
          status: "draft",
          updated_at: "2026-03-04T12:00:00Z"
        }
      ],
      total: 1
    })

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.resumeWork).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "shared-42",
          entityType: "reading_item"
        }),
        expect.objectContaining({
          entityId: "shared-42",
          entityType: "note"
        })
      ])
    )
  })

  it("normalizes notes from a bare array payload", async () => {
    mocks.listNotes.mockResolvedValueOnce([
      {
        note_id: "note-2",
        name: "Loose payload note",
        text: "Picked up from a bare array response.",
        updated_at: "2026-03-03T12:00:00Z"
      }
    ])

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.needsAttention).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "note-2",
          source: "note",
          title: "Loose payload note"
        })
      ])
    )
    expect(snapshot.resumeWork).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "note-2",
          source: "note"
        })
      ])
    )
  })

  it("falls back to an empty notes collection for unknown payloads", async () => {
    mocks.listNotes.mockResolvedValueOnce({ note: "unexpected-shape" })

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.degradedSources).not.toContain("notes")
    expect(snapshot.needsAttention).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ source: "note" })])
    )
    expect(snapshot.resumeWork).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ source: "note" })])
    )
  })

  it("degrades by source instead of failing the whole snapshot", async () => {
    mocks.getReadingList.mockRejectedValueOnce(new Error("reading unavailable"))

    const snapshot = await fetchCompanionHomeSnapshot("options")

    expect(snapshot.degradedSources).toContain("reading")
    expect(snapshot.inbox).toEqual([
      expect.objectContaining({ entityId: "reflection-1" })
    ])
    expect(snapshot.resumeWork).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ entityId: "goal-1" }),
        expect.objectContaining({ entityId: "note-1" })
      ])
    )
    expect(snapshot.resumeWork).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ entityId: "reading-1" })])
    )
    expect(snapshot.readingQueue).toEqual([])
  })

  it("omits unsupported sidepanel links for reading and note entities", async () => {
    const snapshot = await fetchCompanionHomeSnapshot("sidepanel")

    expect(snapshot.readingQueue).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "reading-1",
          href: undefined
        })
      ])
    )
    expect(snapshot.resumeWork).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "note-1",
          href: undefined
        })
      ])
    )
    expect(snapshot.goalsFocus).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "goal-1",
          href: "/companion"
        })
      ])
    )
    expect(snapshot.inbox).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "reflection-1",
          href: "/companion"
        })
      ])
    )
  })
})
