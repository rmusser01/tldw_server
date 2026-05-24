import { describe, expect, it } from "vitest"
import {
  buildWorkspaceGlobalSearchResults,
  getWorkspaceChatSearchMessageId,
  parseWorkspaceGlobalSearchQuery
} from "../workspace-global-search"

describe("workspace global search", () => {
  it("parses optional domain filters", () => {
    expect(parseWorkspaceGlobalSearchQuery("chat: retrieval quality")).toEqual({
      raw: "chat: retrieval quality",
      normalized: "retrieval quality",
      terms: ["retrieval", "quality"],
      filter: "chat"
    })

    expect(parseWorkspaceGlobalSearchQuery("notes: confidence").filter).toBe("note")
    expect(parseWorkspaceGlobalSearchQuery("unknown: confidence").filter).toBeNull()
  })

  it("ranks mixed-domain results and keeps only matching records", () => {
    const results = buildWorkspaceGlobalSearchResults({
      query: "transition report",
      sources: [
        {
          id: "source-1",
          mediaId: 1,
          title: "Climate transition report",
          type: "pdf",
          addedAt: new Date("2026-02-18T09:00:00.000Z")
        },
        {
          id: "source-2",
          mediaId: 2,
          title: "Unrelated transcript",
          type: "text",
          addedAt: new Date("2026-02-18T09:05:00.000Z")
        }
      ],
      chatMessages: [
        {
          isBot: false,
          name: "You",
          message: "Can we map the transition timeline to quarterly milestones?",
          sources: []
        },
        {
          isBot: true,
          name: "Assistant",
          message: "Sure, here is a structured report outline.",
          sources: []
        }
      ],
      currentNote: {
        id: 11,
        title: "Transition checklist",
        content: "Capture report references and key dates.",
        keywords: ["planning"],
        version: 3,
        isDirty: false
      },
      workspaceNotes: []
    })

    expect(results.length).toBeGreaterThanOrEqual(3)
    expect(results[0].domain).toBe("source")
    expect(results.map((entry) => entry.domain)).toEqual(
      expect.arrayContaining(["source", "chat", "note"])
    )
    expect(results.every((entry) => entry.score > 0)).toBe(true)
  })

  it("applies chat-only filtering via query prefix", () => {
    const results = buildWorkspaceGlobalSearchResults({
      query: "chat: confidence",
      sources: [
        {
          id: "source-1",
          mediaId: 1,
          title: "Confidence intervals",
          type: "pdf",
          addedAt: new Date("2026-02-18T09:00:00.000Z")
        }
      ],
      chatMessages: [
        {
          isBot: true,
          name: "Assistant",
          message: "Confidence remains low for source B.",
          sources: []
        }
      ],
      currentNote: {
        title: "Confidence notes",
        content: "Need follow-up sample size analysis",
        keywords: [],
        isDirty: false
      },
      workspaceNotes: []
    })

    expect(results).toHaveLength(1)
    expect(results[0].domain).toBe("chat")
  })

  it("generates deterministic chat message ids", () => {
    const direct = getWorkspaceChatSearchMessageId(
      {
        id: "message-123",
        serverMessageId: "server-1",
        createdAt: 123
      },
      0
    )
    const fallback = getWorkspaceChatSearchMessageId(
      {
        id: "",
        serverMessageId: "",
        createdAt: 987654
      },
      7
    )

    expect(direct).toBe("msg:message-123")
    expect(fallback).toBe("idx:7:987654")
  })

  it("indexes workspace note corpus and includes note ids in results", () => {
    const results = buildWorkspaceGlobalSearchResults({
      query: "confidence tracker",
      sources: [],
      chatMessages: [],
      currentNote: {
        id: undefined,
        title: "Draft scratchpad",
        content: "Unrelated draft content",
        keywords: [],
        isDirty: true
      },
      workspaceNotes: [
        {
          id: 77,
          title: "Confidence tracker",
          content: "Track weekly confidence by source",
          keywords: ["confidence"]
        }
      ]
    })

    const noteResult = results.find((result) => result.noteId === 77)
    expect(noteResult).toBeTruthy()
    expect(noteResult?.domain).toBe("note")
    expect(noteResult?.noteField).toBe("title")
  })
})
