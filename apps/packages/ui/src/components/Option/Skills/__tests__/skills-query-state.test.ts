import { describe, expect, it } from "vitest"
import {
  DEFAULT_SKILLS_QUERY_STATE,
  parseSkillsQueryState,
  serializeSkillsQueryState
} from "../skills-query-state"

describe("skills-query-state", () => {
  it("parses a complete shareable skills view", () => {
    expect(
      parseSkillsQueryState(
        new URLSearchParams(
          "view=trash&q=research&mode=fork&visibility=hidden&tools=with-tools&model=gpt&sort=context&order=desc&page=3&pageSize=20"
        )
      )
    ).toEqual({
      view: "trash",
      search: "research",
      context: "fork",
      visibility: "hidden",
      tools: "with-tools",
      model: "gpt",
      sort: "context",
      order: "desc",
      page: 3,
      pageSize: 20
    })
  })

  it("falls back safely when query values are unsupported", () => {
    expect(
      parseSkillsQueryState(
        new URLSearchParams(
          "mode=remote&visibility=maybe&tools=all&sort=description&order=sideways&page=-1&pageSize=500"
        )
      )
    ).toEqual(DEFAULT_SKILLS_QUERY_STATE)
  })

  it("serializes only non-default view state", () => {
    expect(
      serializeSkillsQueryState({
        ...DEFAULT_SKILLS_QUERY_STATE,
        view: "trash",
        search: "  report  ",
        context: "inline",
        page: 2,
        pageSize: 50
      }).toString()
    ).toBe("view=trash&q=report&mode=inline&page=2&pageSize=50")
  })

  it("round trips supported state", () => {
    const expected = {
      ...DEFAULT_SKILLS_QUERY_STATE,
      search: "summarize",
      tools: "without-tools" as const,
      sort: "name" as const,
      order: "asc" as const
    }

    expect(parseSkillsQueryState(serializeSkillsQueryState(expected))).toEqual(expected)
  })

  it.each(["created_at", "last_modified"] as const)(
    "round trips the %s timestamp sort",
    (sort) => {
      const expected = {
        ...DEFAULT_SKILLS_QUERY_STATE,
        sort,
        order: "desc" as const
      }

      expect(parseSkillsQueryState(serializeSkillsQueryState(expected))).toEqual(expected)
    }
  )

  it("ignores and omits incomplete sort pairs", () => {
    expect(parseSkillsQueryState(new URLSearchParams("sort=name"))).toEqual(
      DEFAULT_SKILLS_QUERY_STATE
    )
    expect(parseSkillsQueryState(new URLSearchParams("order=desc"))).toEqual(
      DEFAULT_SKILLS_QUERY_STATE
    )
    expect(
      serializeSkillsQueryState({
        ...DEFAULT_SKILLS_QUERY_STATE,
        sort: "name"
      }).toString()
    ).toBe("")
  })
})
