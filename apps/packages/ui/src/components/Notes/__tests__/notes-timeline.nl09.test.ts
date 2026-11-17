import { afterEach, describe, expect, it } from "vitest"
import { buildNotesTimelineSections } from "../notes-timeline"
import type { NoteListItem } from "../notes-manager-types"

// NL-09: Timeline months follow the reader's local time zone.

const originalTimezone = process.env.TZ

const note = (id: string, updatedAt: string | null): NoteListItem =>
  ({ id, title: id, updated_at: updatedAt }) as NoteListItem

describe("buildNotesTimelineSections (NL-09)", () => {
  afterEach(() => {
    if (originalTimezone === undefined) delete process.env.TZ
    else process.env.TZ = originalTimezone
  })

  it("files an early-October note under October for a reader west of UTC", () => {
    process.env.TZ = "America/Los_Angeles"
    // 10:00 local on Oct 2 in Los Angeles (17:00 UTC)
    const sections = buildNotesTimelineSections([note("oct", "2026-10-02T17:00:00.000Z")], {
      unknownLabel: "Unknown date",
      locale: "en-US"
    })
    expect(sections).toHaveLength(1)
    expect(sections[0].key).toBe("2026-10")
    expect(sections[0].label).toBe("October 2026")
  })

  it("uses the local month when the UTC month differs", () => {
    process.env.TZ = "America/Los_Angeles"
    // 20:00 on Sep 30 in Los Angeles, already Oct 1 in UTC
    const sections = buildNotesTimelineSections([note("late-sep", "2026-10-01T03:00:00.000Z")], {
      unknownLabel: "Unknown date",
      locale: "en-US"
    })
    expect(sections[0].key).toBe("2026-09")
    expect(sections[0].label).toBe("September 2026")
  })

  it("files a note by the local month east of UTC", () => {
    process.env.TZ = "Asia/Tokyo"
    // 08:00 on Nov 1 in Tokyo, still Oct 31 in UTC
    const sections = buildNotesTimelineSections([note("nov", "2026-10-31T23:00:00.000Z")], {
      unknownLabel: "Unknown date",
      locale: "en-US"
    })
    expect(sections[0].key).toBe("2026-11")
    expect(sections[0].label).toBe("November 2026")
  })

  it("orders months newest first, notes newest first, unknown dates last", () => {
    process.env.TZ = "UTC"
    const sections = buildNotesTimelineSections(
      [
        note("a", "2026-09-10T10:00:00.000Z"),
        note("b", null),
        note("c", "2026-10-05T10:00:00.000Z"),
        note("d", "2026-10-20T10:00:00.000Z")
      ],
      { unknownLabel: "Unknown date", locale: "en-US" }
    )
    expect(sections.map((section) => section.key)).toEqual(["2026-10", "2026-09", "unknown"])
    expect(sections[0].notes.map((entry) => entry.id)).toEqual(["d", "c"])
    expect(sections[2].label).toBe("Unknown date")
  })
})
