// @vitest-environment jsdom
import { describe, expect, it } from "vitest"
import { adminDateTime, formatAdminDateTime } from "../admin-format"

describe("admin-format shared date formatter (C-S5)", () => {
  it("builds one shared medium-date/short-time Intl formatter at module scope", () => {
    expect(adminDateTime).toBeInstanceOf(Intl.DateTimeFormat)
    const resolved = adminDateTime.resolvedOptions()
    expect(resolved.dateStyle).toBe("medium")
    expect(resolved.timeStyle).toBe("short")
  })

  it("formats a known ISO timestamp through the shared formatter", () => {
    const iso = "2026-10-06T12:34:56Z"
    expect(formatAdminDateTime(iso)).toBe(adminDateTime.format(new Date(iso)))

    // Snapshot the Intl result structurally: both date and time segments must
    // be present, whatever the runtime locale/timezone is.
    const partTypes = new Set(
      adminDateTime.formatToParts(new Date(iso)).map((part) => part.type)
    )
    for (const expected of ["year", "month", "day", "hour", "minute"]) {
      expect(partTypes.has(expected as Intl.DateTimeFormatPartTypes)).toBe(true)
    }
  })

  it("accepts epoch numbers and Date instances alongside ISO strings", () => {
    const epoch = Date.UTC(2026, 9, 6, 12, 34, 56)
    expect(formatAdminDateTime(epoch)).toBe(adminDateTime.format(new Date(epoch)))
    expect(formatAdminDateTime(new Date(epoch))).toBe(
      adminDateTime.format(new Date(epoch))
    )
  })

  it("does not construct a fresh formatter per call (stable instance)", () => {
    const before = formatAdminDateTime("2026-01-01T00:00:00Z")
    const after = formatAdminDateTime("2026-01-01T00:00:00Z")
    expect(after).toBe(before)
  })
})
