import { describe, expect, it } from "vitest"
import { buildQuickChatDocsRagProfile } from "../docs-rag-profile"

describe("buildQuickChatDocsRagProfile", () => {
  it("uses synopsis-focused retrieval settings for summary-style queries", () => {
    const profile = buildQuickChatDocsRagProfile({
      query: "What's the synopsis of this paper?"
    })

    expect(profile.options.include_parent_document).toBe(true)
    expect(profile.options.fts_level).toBe("document")
    expect(profile.options.top_k).toBe(10)
    expect(profile.options.sources).toEqual(["media_db"])
    expect(profile.options.index_namespace).toBe("project_docs")
    expect(profile.options.corpus).toBe("media_db")
  })

  it("injects current route context when query references this page", () => {
    const profile = buildQuickChatDocsRagProfile({
      query: "How do I do this on this page?",
      currentRoute: "#/research-workspace?tab=chat"
    })

    expect(profile.query).toContain("Current page context:")
    expect(profile.query).toContain("Research Workspace")
    expect(profile.query).toContain("/research-workspace")
    expect(profile.options.include_parent_document).toBe(false)
  })

  it("relaxes minimum score for troubleshooting requests", () => {
    const profile = buildQuickChatDocsRagProfile({
      query: "This workflow is not working, how do I debug it?"
    })
    expect(profile.options.min_score).toBe(0.1)
  })

  it("accepts explicit docs media id restrictions in strict mode", () => {
    const profile = buildQuickChatDocsRagProfile({
      query: "Where is auth setup documented?",
      scope: {
        projectDocsNamespace: "official_docs",
        projectDocsMediaIds: [21, 55, 89]
      }
    })

    expect(profile.options.index_namespace).toBe("official_docs")
    expect(profile.options.include_media_ids).toEqual([21, 55, 89])
    expect(profile.options.sources).toEqual(["media_db"])
  })

  it("allows non-strict mode fallback to broader sources", () => {
    const profile = buildQuickChatDocsRagProfile({
      query: "How do I perform workflow discovery?",
      scope: {
        strictProjectDocsOnly: false
      }
    })

    expect(profile.options.sources).toEqual(["media_db", "notes"])
    expect(profile.options.index_namespace).toBeUndefined()
    expect(profile.options.include_media_ids).toBeUndefined()
  })
})
