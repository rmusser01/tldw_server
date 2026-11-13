import { describe, expect, it } from "vitest"

import { readNotesTotal } from "@/services/notes-page-request"

describe("readNotesTotal", () => {
  it("reads pagination.total, then the top-level total", () => {
    expect(readNotesTotal({ pagination: { total: 105 }, total: 7 }, 20)).toBe(105)
    expect(readNotesTotal({ total: 42 }, 20)).toBe(42)
  })

  it("estimates an uncounted total from the page's offset and rows, plus one when more follow", () => {
    // The server could not count: page 3 of 20 with more rows after it.
    const pageThree = { items: [], total: null, offset: 40, pagination: { total: null, offset: 40, has_more: true } }
    expect(readNotesTotal(pageThree, 20)).toBe(61)
    // The last page, short and with nothing after it.
    const lastPage = { total: null, offset: 60, pagination: { total: null, offset: 60, has_more: false } }
    expect(readNotesTotal(lastPage, 5)).toBe(65)
  })

  it("falls back to the row count when the response says nothing about paging", () => {
    expect(readNotesTotal({}, 3)).toBe(3)
    expect(readNotesTotal(null, 0)).toBe(0)
  })
})
