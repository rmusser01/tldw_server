import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  buildFakeServerPrompt,
  createFakePromptsApi
} from "./server-prompts-api-fixture"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  SERVER_PROMPT_LIBRARY_PAGE_SIZE,
  getServerPromptDetail,
  isServerPromptLibraryUnconfiguredError,
  listAllServerPrompts
} from "../server-prompt-library"

const serveFakeApi = (api: ReturnType<typeof createFakePromptsApi>) => {
  mocks.bgRequest.mockImplementation(async (request) => api.handle(request))
}

describe("server prompt library client", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  it("pages through every server prompt with the list endpoint's real params", async () => {
    const prompts = Array.from({ length: 250 }, (_, index) =>
      buildFakeServerPrompt(index + 1)
    )
    const api = createFakePromptsApi(prompts)
    serveFakeApi(api)

    const library = await listAllServerPrompts()

    expect(library.prompts).toHaveLength(250)
    expect(library.totalItems).toBe(250)
    expect(library.truncated).toBe(false)
    expect(library.prompts.map((prompt) => prompt.id)).toEqual(
      prompts.map((prompt) => prompt.id)
    )
    expect(api.listRequests.map((params) => params.get("page"))).toEqual([
      "1",
      "2",
      "3"
    ])
    for (const params of api.listRequests) {
      expect(params.get("per_page")).toBe(
        String(SERVER_PROMPT_LIBRARY_PAGE_SIZE)
      )
      expect(params.get("sort_by")).toBe("id")
      expect(params.get("sort_order")).toBe("asc")
    }
  })

  it("keeps only the brief fields the list endpoint returns", async () => {
    serveFakeApi(
      createFakePromptsApi([
        buildFakeServerPrompt(4, {
          name: "Daily standup",
          author: "Robert",
          system_prompt: "Never listed"
        })
      ])
    )

    const library = await listAllServerPrompts()

    expect(library.prompts).toEqual([
      {
        id: 4,
        uuid: "00000000-0000-4000-8000-000000000004",
        name: "Daily standup",
        author: "Robert",
        last_modified: "2026-09-30T12:00:00Z",
        usage_count: 0,
        last_used_at: null
      }
    ])
  })

  it("lists an empty library from a single request", async () => {
    const api = createFakePromptsApi([])
    serveFakeApi(api)

    const library = await listAllServerPrompts()

    expect(library).toEqual({ prompts: [], totalItems: 0, truncated: false })
    expect(api.listRequests).toHaveLength(1)
  })

  it("falls back to total_pages when a server omits the pagination block", async () => {
    const api = createFakePromptsApi(
      Array.from({ length: 3 }, (_, index) => buildFakeServerPrompt(index + 1))
    )
    mocks.bgRequest.mockImplementation(async (request) => {
      const { pagination: _omitted, ...legacy } = api.handle(request) as Record<
        string,
        unknown
      >
      return legacy
    })

    const library = await listAllServerPrompts({ pageSize: 2 })

    expect(library.prompts.map((prompt) => prompt.id)).toEqual([1, 2, 3])
    expect(api.listRequests).toHaveLength(2)
  })

  it("drops a prompt repeated across pages", async () => {
    const repeated = buildFakeServerPrompt(2)
    mocks.bgRequest
      .mockResolvedValueOnce({
        items: [buildFakeServerPrompt(1), repeated].map(({ id, uuid, name }) => ({
          id,
          uuid,
          name,
          author: null,
          last_modified: "2026-09-30T12:00:00Z",
          usage_count: 0,
          last_used_at: null
        })),
        total_pages: 2,
        current_page: 1,
        total_items: 3,
        pagination: { mode: "page", page: 1, per_page: 2, total: 3, total_pages: 2, has_more: true }
      })
      .mockResolvedValueOnce({
        items: [repeated].map(({ id, uuid, name }) => ({
          id,
          uuid,
          name,
          author: null,
          last_modified: "2026-09-30T12:00:00Z",
          usage_count: 0,
          last_used_at: null
        })),
        total_pages: 2,
        current_page: 2,
        total_items: 3,
        pagination: { mode: "page", page: 2, per_page: 2, total: 3, total_pages: 2, has_more: false }
      })

    const library = await listAllServerPrompts({ pageSize: 2 })

    expect(library.prompts.map((prompt) => prompt.id)).toEqual([1, 2])
  })

  it("marks the listing truncated only when the runaway page guard stops it", async () => {
    const api = createFakePromptsApi(
      Array.from({ length: 5 }, (_, index) => buildFakeServerPrompt(index + 1))
    )
    serveFakeApi(api)

    const library = await listAllServerPrompts({ pageSize: 2, maxPages: 2 })

    expect(library.prompts).toHaveLength(4)
    expect(library.totalItems).toBe(5)
    expect(library.truncated).toBe(true)
  })

  it("rejects a response that is not a prompt list", async () => {
    mocks.bgRequest.mockResolvedValue({})

    await expect(listAllServerPrompts()).rejects.toThrow(
      "Unexpected response from the server prompt library"
    )
  })

  it("fetches one prompt's text from the detail endpoint", async () => {
    const api = createFakePromptsApi([
      buildFakeServerPrompt(9, {
        name: "Socratic tutor",
        system_prompt: "Ask guiding questions.",
        user_prompt: null,
        keywords: ["teaching"]
      })
    ])
    serveFakeApi(api)

    const detail = await getServerPromptDetail(9)

    expect(api.detailRequests).toEqual(["9"])
    expect(detail).toMatchObject({
      id: 9,
      name: "Socratic tutor",
      system_prompt: "Ask guiding questions.",
      user_prompt: null,
      keywords: ["teaching"],
      prompt_format: "legacy"
    })
  })

  it("recognises the not-configured transport error", () => {
    expect(
      isServerPromptLibraryUnconfiguredError(
        new Error("tldw server not configured")
      )
    ).toBe(true)
    expect(
      isServerPromptLibraryUnconfiguredError(new Error("Request failed: 500"))
    ).toBe(false)
  })
})
