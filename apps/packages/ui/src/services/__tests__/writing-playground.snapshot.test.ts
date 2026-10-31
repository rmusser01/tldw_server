import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import {
  exportWritingSnapshot,
  getWritingDefaults,
  importWritingSnapshot,
  listManuscriptCharacters,
  listManuscriptWorldInfo
} from "@/services/writing-playground"

describe("writing-playground snapshot service wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it.each([
    [listManuscriptCharacters, "characters", { id: "character-1", project_id: "project-1", name: "Ari", role: "hero" }],
    [listManuscriptWorldInfo, "items", { id: "world-1", project_id: "project-1", name: "Harbor", kind: "location" }]
  ] as const)("normalizes manuscript list payloads for %s", async (list, key, item) => {
    mocks.bgRequest.mockResolvedValueOnce([item])
    expect(await list("project-1")).toEqual({ [key]: [item], total: 1 })
    mocks.bgRequest.mockResolvedValueOnce([])
    expect(await list("project-1")).toEqual({ [key]: [], total: 0 })
    const wrapped = { [key]: [item], total: 4 }
    mocks.bgRequest.mockResolvedValueOnce(wrapped)
    expect(await list("project-1")).toEqual(wrapped)
  })

  it("calls snapshot export endpoint", async () => {
    const response = {
      version: 1,
      counts: { sessions: 1, templates: 1, themes: 1 },
      sessions: [{ id: "s1", name: "Session 1", payload: {}, schema_version: 1 }],
      templates: [{ name: "Template 1", payload: {}, schema_version: 1, is_default: false }],
      themes: [{ name: "Theme 1", schema_version: 1, is_default: false, order: 0 }]
    }
    mocks.bgRequest.mockResolvedValueOnce(response)

    const result = await exportWritingSnapshot()

    expect(result).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/writing/snapshot/export",
        method: "GET"
      })
    )
  })

  it("calls snapshot import endpoint with mode and snapshot payload", async () => {
    const payload = {
      mode: "replace" as const,
      snapshot: {
        sessions: [
          {
            id: "session-restore-1",
            name: "Restored Session",
            payload: { text: "restored" },
            schema_version: 1
          }
        ],
        templates: [
          {
            name: "Restored Template",
            payload: { inst_pre: "<U>" },
            schema_version: 1,
            is_default: true
          }
        ],
        themes: [
          {
            name: "Restored Theme",
            class_name: "restored-theme",
            css: ".restored-theme { color: #123; }",
            schema_version: 1,
            is_default: true,
            order: 0
          }
        ]
      }
    }
    const response = {
      mode: "replace" as const,
      imported: { sessions: 1, templates: 1, themes: 1 }
    }
    mocks.bgRequest.mockResolvedValueOnce(response)

    const result = await importWritingSnapshot(payload)

    expect(result).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/writing/snapshot/import",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })
    )
  })

  it("calls defaults catalog endpoint", async () => {
    const response = {
      version: 1,
      templates: [{ name: "default", payload: {}, schema_version: 1, is_default: true }],
      themes: [
        {
          name: "default",
          class_name: "",
          css: "",
          schema_version: 1,
          is_default: true,
          order: 0
        }
      ]
    }
    mocks.bgRequest.mockResolvedValueOnce(response)

    const result = await getWritingDefaults()

    expect(result).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/writing/defaults",
        method: "GET"
      })
    )
  })
})
