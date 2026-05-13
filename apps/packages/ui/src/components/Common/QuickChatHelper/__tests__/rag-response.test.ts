import { describe, expect, it } from "vitest"
import { buildQuickChatRagReply } from "../rag-response"

describe("buildQuickChatRagReply", () => {
  it("returns generated answer with formatted references", () => {
    const response = {
      generated_answer: "Use the Media page first, then move to Knowledge.",
      citations: [
        {
          title: "Media workflow",
          source: "Docs",
          url: "https://example.com/media-workflow"
        }
      ]
    }

    const result = buildQuickChatRagReply(response)
    expect(result.hasContext).toBe(true)
    expect(result.message).toContain("Use the Media page first")
    expect(result.message).toContain("### References")
    expect(result.message).toContain("[Media workflow](https://example.com/media-workflow)")
  })

  it("builds snippet fallback when answer is missing but docs exist", () => {
    const response = {
      results: [
        {
          content:
            "Research Studio helps discover tools and route users to specialized pages for each workflow.",
          metadata: {
            title: "Research Studio guide",
            source: "Documentation"
          }
        }
      ]
    }

    const result = buildQuickChatRagReply(response)
    expect(result.hasContext).toBe(true)
    expect(result.message).toContain("I found relevant documentation snippets")
    expect(result.message).toContain("Research Studio guide")
  })

  it("returns no-context fallback when response has no answer/docs", () => {
    const result = buildQuickChatRagReply({})
    expect(result.hasContext).toBe(false)
    expect(result.message).toContain("could not find relevant indexed documentation")
  })

  it("adds suggested pages based on query and response context", () => {
    const response = {
      generated_answer:
        "Use the Evaluations page to run quality checks and compare model outputs."
    }

    const result = buildQuickChatRagReply(response, {
      query: "How do I benchmark model quality?",
      currentRoute: "/workspace-playground"
    })

    expect(result.hasContext).toBe(true)
    expect(result.message).toContain("### Suggested Pages")
    expect(result.message).toContain("`/evaluations`")
  })

  it("uses provided custom guides for suggested pages", () => {
    const response = {
      generated_answer: "Open setup diagnostics first."
    }

    const result = buildQuickChatRagReply(response, {
      query: "How do I troubleshoot setup?",
      guides: [
        {
          id: "custom-setup",
          title: "Custom setup guide",
          question: "How do I setup?",
          answer: "Use diagnostics.",
          route: "/settings/health",
          routeLabel: "Health & Diagnostics",
          tags: ["setup", "diagnostics"]
        }
      ]
    })

    expect(result.message).toContain("### Suggested Pages")
    expect(result.message).toContain("`/settings/health`")
    expect(result.message).not.toContain("`/evaluations`")
  })
})
