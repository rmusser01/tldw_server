import { describe, expect, it } from "vitest"

import { mapApiListToUi } from "@/services/tldw/data-tables"

describe("data tables pagination compatibility", () => {
  it("falls back to canonical pagination.total when legacy total is absent", () => {
    const mapped = mapApiListToUi({
      tables: [
        {
          uuid: "table-1",
          name: "Table 1",
          prompt: "Prompt",
          status: "ready",
          row_count: 3,
          column_count: 2,
          source_count: 1,
          created_at: "2026-04-29T00:00:00Z",
          updated_at: "2026-04-29T00:00:00Z",
        },
      ],
      count: 1,
      limit: 20,
      offset: 0,
      pagination: {
        mode: "offset",
        limit: 20,
        offset: 0,
        total: 7,
        has_more: true,
        next_offset: 20,
      },
    })

    expect(mapped.tables).toHaveLength(1)
    expect(mapped.total).toBe(7)
  })
})
