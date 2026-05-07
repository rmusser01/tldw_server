import { describe, expect, it } from "vitest"

import {
  buildChatToolFilterState,
  getMcpToolGroupLabel,
  normalizeChatToolName,
  normalizeChatToolsForRequest,
  resolveMcpToolIdentity
} from "../chat-tools"

describe("chat tool normalization", () => {
  it("normalizes MCP tool names using the same chat-safe identity", () => {
    expect(normalizeChatToolName("ext.docs.docs.search")).toBe(
      "ext_docs_docs_search"
    )
    expect(normalizeChatToolName(" docs_search ")).toBe("docs_search")
    expect(normalizeChatToolName("...")).toBeNull()
  })

  it("resolves identity from MCP and OpenAI-style tool shapes", () => {
    expect(
      resolveMcpToolIdentity({
        name: "notes.search",
        description: "Search notes"
      })
    ).toMatchObject({
      rawName: "notes.search",
      chatName: "notes_search",
      displayName: "notes.search",
      description: "Search notes"
    })

    expect(
      resolveMcpToolIdentity({
        type: "function",
        function: {
          name: "media-search",
          description: "Search media"
        }
      })
    ).toMatchObject({
      rawName: "media-search",
      chatName: "media-search",
      displayName: "media-search",
      description: "Search media"
    })
  })

  it("filters disabled and unexecutable tools while keeping discovered tools available for display", () => {
    const state = buildChatToolFilterState({
      tools: [
        { name: "notes.search", canExecute: true },
        { name: "media.search", canExecute: false },
        { name: "slides.list", canExecute: true }
      ],
      disabledToolNames: ["slides_list"]
    })

    expect(state.discoveredTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "media.search",
      "slides.list"
    ])
    expect(state.availableTools.map((tool) => tool.rawName)).toEqual([
      "notes.search",
      "slides.list"
    ])
    expect(state.chatTools.map((tool) => tool.rawName)).toEqual(["notes.search"])
    expect(state.counts).toMatchObject({
      discovered: 3,
      executable: 2,
      disabled: 1,
      colliding: 0,
      chatEnabled: 1
    })
  })

  it("excludes normalized name collisions from chat tools", () => {
    const state = buildChatToolFilterState({
      tools: [
        { name: "docs.search", canExecute: true },
        { name: "docs_search", canExecute: true },
        { name: "notes.search", canExecute: true }
      ],
      disabledToolNames: []
    })

    expect(state.collisionToolNames).toEqual(["docs_search"])
    expect(state.chatTools.map((tool) => tool.rawName)).toEqual(["notes.search"])
    expect(state.counts).toMatchObject({
      discovered: 3,
      executable: 3,
      disabled: 0,
      colliding: 2,
      chatEnabled: 1
    })
  })

  it("normalizes request tools and omits empty tool payloads", () => {
    expect(
      normalizeChatToolsForRequest([
        { name: "notes.search", description: "Search notes" },
        { name: "docs.search" },
        { name: "docs_search" }
      ])
    ).toEqual([
      {
        type: "function",
        function: {
          name: "notes_search",
          description: "Search notes",
          parameters: {
            type: "object",
            properties: {}
          }
        }
      }
    ])

    expect(normalizeChatToolsForRequest([])).toBeUndefined()
  })

  it("normalizes already resolved chat tools for request construction", () => {
    const resolved = buildChatToolFilterState({
      tools: [
        {
          name: "notes.search",
          description: "Search notes",
          parameters: { type: "object", properties: { q: { type: "string" } } },
          canExecute: true
        }
      ]
    })

    expect(normalizeChatToolsForRequest(resolved.chatTools)).toEqual([
      {
        type: "function",
        function: {
          name: "notes_search",
          description: "Search notes",
          parameters: {
            type: "object",
            properties: { q: { type: "string" } }
          }
        }
      }
    ])
  })
})

describe("MCP tool grouping", () => {
  it("uses external server metadata, ext server id, module, then MCP fallback", () => {
    expect(
      getMcpToolGroupLabel({
        name: "ext.docs.search",
        metadata: { server_name: "Docs Server", server_id: "docs" }
      })
    ).toBe("Docs Server")

    expect(
      getMcpToolGroupLabel({
        name: "ext.docs.search",
        metadata: { server_id: "docs" }
      })
    ).toBe("docs")

    expect(
      getMcpToolGroupLabel({
        name: "ext.browser.search"
      })
    ).toBe("browser")

    expect(
      getMcpToolGroupLabel({
        name: "notes.search",
        module: "notes"
      })
    ).toBe("notes")

    expect(getMcpToolGroupLabel({ name: "unknown" })).toBe("MCP")
  })
})
