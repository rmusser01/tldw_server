import { describe, expect, it } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import {
  buildStagedSourceFromWorkspaceSource,
  formatStagedSourceInsertText,
  getReadyStagedMediaIds,
  stageWorkspaceSources
} from "../staging"

const source = (overrides: Partial<WorkspaceSource> = {}): WorkspaceSource => ({
  id: "source-1",
  mediaId: 101,
  title: "Operator Notes",
  type: "document",
  status: "ready",
  addedAt: new Date("2026-05-03T00:00:00Z"),
  ...overrides
})

describe("chat workspace staging", () => {
  it("builds explicit staged source metadata from a workspace source", () => {
    expect(buildStagedSourceFromWorkspaceSource(source(), "Default workspace")).toMatchObject({
      sourceId: "source-1",
      mediaId: 101,
      title: "Operator Notes",
      type: "document",
      scopeLabel: "Default workspace",
      availability: "ready"
    })
  })

  it("deduplicates staged sources by source id", () => {
    const staged = stageWorkspaceSources(
      [buildStagedSourceFromWorkspaceSource(source(), "A")],
      [source({ title: "Renamed" })],
      "A"
    )

    expect(staged).toHaveLength(1)
    expect(staged[0].title).toBe("Renamed")
  })

  it("formats insert text and leaves sending to the user", () => {
    const staged = [buildStagedSourceFromWorkspaceSource(source(), "Default workspace")]
    expect(formatStagedSourceInsertText(staged)).toContain("Context sources")
    expect(formatStagedSourceInsertText(staged)).toContain("Operator Notes")
  })

  it("returns only ready positive media ids for structured RAG", () => {
    const ready = buildStagedSourceFromWorkspaceSource(source(), "A")
    const error = buildStagedSourceFromWorkspaceSource(
      source({ id: "source-2", mediaId: 202, status: "error" }),
      "A"
    )
    expect(getReadyStagedMediaIds([ready, error])).toEqual([101])
  })
})
