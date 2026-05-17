import { describe, expect, it } from "vitest"

import { buildSourcesNewPath } from "@/routes/route-paths"

describe("sources route path builders", () => {
  it("builds the notes folder sync preset path", () => {
    expect(buildSourcesNewPath({ preset: "notes-folder-sync" })).toBe(
      "/sources/new?preset=notes-folder-sync"
    )
  })

  it("builds the plain new source path without options", () => {
    expect(buildSourcesNewPath()).toBe("/sources/new")
  })
})
