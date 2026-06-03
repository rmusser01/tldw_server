import { describe, expect, it } from "vitest"
import {
  QUICK_CHAT_WORKFLOW_GUIDES,
  filterQuickChatWorkflowGuides,
  normalizeQuickChatRoutePath,
  parseQuickChatWorkflowGuidesJson,
  recommendQuickChatWorkflowGuides
} from "../workflow-guides"

describe("quick chat workflow guides", () => {
  it("returns all guides when query is empty", () => {
    const results = filterQuickChatWorkflowGuides("")
    expect(results).toHaveLength(QUICK_CHAT_WORKFLOW_GUIDES.length)
  })

  it("matches guides by tag and question text", () => {
    const ragResults = filterQuickChatWorkflowGuides("rag")
    expect(ragResults.some((guide) => guide.id === "docs-research-qa")).toBe(true)

    const workflowResults = filterQuickChatWorkflowGuides("which page")
    expect(
      workflowResults.some((guide) => guide.id === "find-tools-for-goal")
    ).toBe(true)
  })

  it("returns empty when no guide matches", () => {
    const results = filterQuickChatWorkflowGuides("totally-unmatched-search-token")
    expect(results).toEqual([])
  })

  it("normalizes options/hash routes for recommendation routing", () => {
    expect(normalizeQuickChatRoutePath("#/research-workspace?tab=chat")).toBe(
      "/research-workspace"
    )
    expect(
      normalizeQuickChatRoutePath(
        "chrome-extension://abc/options.html#/settings/health"
      )
    ).toBe("/settings/health")
  })

  it("recommends matching workflow pages with route awareness", () => {
    const recommendations = recommendQuickChatWorkflowGuides({
      query:
        "I am in research workspace. How do I benchmark model quality and compare providers?",
      answer: "Use evaluations to compare metrics.",
      currentRoute: "/research-workspace",
      maxResults: 8
    })
    expect(recommendations.length).toBeGreaterThan(0)
    expect(
      recommendations.some(
        (item) =>
          item.route === "/research-workspace" && item.isCurrentRoute === true
      )
    ).toBe(true)
  })

  it("parses valid custom workflow guide JSON", () => {
    const parsed = parseQuickChatWorkflowGuidesJson(
      JSON.stringify([
        {
          id: "custom-guide",
          title: "Custom Guide",
          question: "How do I start?",
          answer: "Use Research Workspace.",
          route: "research-workspace",
          routeLabel: "Research Workspace",
          tags: ["custom", "workflow"]
        }
      ])
    )

    expect(parsed.error).toBeUndefined()
    expect(parsed.guides).toEqual([
      {
        id: "custom-guide",
        title: "Custom Guide",
        question: "How do I start?",
        answer: "Use Research Workspace.",
        route: "/research-workspace",
        routeLabel: "Research Workspace",
        tags: ["custom", "workflow"]
      }
    ])
  })

  it("accepts cards without routeLabel and derives one from route", () => {
    const parsed = parseQuickChatWorkflowGuidesJson(
      JSON.stringify([
        {
          id: "custom-no-route-label",
          title: "Custom Guide",
          question: "Where do I go?",
          answer: "Open workspace.",
          route: "/research-workspace",
          tags: ["workflow"]
        }
      ])
    )

    expect(parsed.error).toBeUndefined()
    expect(parsed.guides).toEqual([
      {
        id: "custom-no-route-label",
        title: "Custom Guide",
        question: "Where do I go?",
        answer: "Open workspace.",
        route: "/research-workspace",
        routeLabel: "Research Workspace",
        tags: ["workflow"]
      }
    ])
  })

  it("returns validation error for invalid JSON payload", () => {
    const parsed = parseQuickChatWorkflowGuidesJson("{not-json")
    expect(parsed.guides).toBeNull()
    expect(parsed.error).toContain("not valid")
  })
})
