import { describe, expect, it } from "vitest"

import {
  buildChatToolFilterState,
  getMcpToolGroupLabel,
  normalizeChatToolName,
  normalizeChatToolsForRequest,
  resolveChatToolRequest,
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

describe("chat tool request resolver", () => {
  const executableTool = {
    name: "notes.search",
    description: "Search notes",
    parameters: { type: "object", properties: { q: { type: "string" } } },
    canExecute: true
  }

  it.each([
    [
      "selected none",
      { tools: [executableTool], toolChoice: "none" as const },
      "tool_choice_none"
    ],
    [
      "model lacks tools",
      {
        tools: [executableTool],
        toolChoice: "auto" as const,
        modelSupportsTools: false
      },
      "model_lacks_tool_capability"
    ],
    [
      "MCP absent",
      {
        tools: [executableTool],
        toolChoice: "auto" as const,
        hasMcp: false
      },
      "mcp_absent"
    ],
    [
      "MCP unavailable",
      {
        tools: [executableTool],
        toolChoice: "auto" as const,
        mcpHealthState: "unavailable"
      },
      "mcp_unavailable"
    ],
    [
      "MCP unhealthy",
      {
        tools: [executableTool],
        toolChoice: "auto" as const,
        mcpHealthState: "unhealthy"
      },
      "mcp_unhealthy"
    ],
    [
      "all executable tools filtered out",
      {
        tools: [
          { name: "docs.search", canExecute: true },
          { name: "docs_search", canExecute: true }
        ],
        toolChoice: "auto" as const
      },
      "no_enabled_executable_tools"
    ],
    [
      "no normalized request tools",
      {
        tools: [{ name: "..." }],
        toolChoice: "auto" as const
      },
      "no_normalized_request_tools"
    ]
  ])("omits tools when %s", (_label, input, omittedReason) => {
    const resolved = resolveChatToolRequest({
      modelSupportsTools: true,
      hasMcp: true,
      mcpHealthState: "healthy",
      ...input
    })

    expect(resolved.tools).toBeUndefined()
    expect(resolved.toolChoice).toBeUndefined()
    expect(resolved.omittedReason).toBe(omittedReason)
    expect(resolved.counts).toHaveProperty("chatEnabled")
  })

  it("includes normalized request tools and only effective tool choices", () => {
    const resolved = resolveChatToolRequest({
      tools: [executableTool],
      toolChoice: "required",
      modelSupportsTools: true,
      hasMcp: true,
      mcpHealthState: "healthy"
    })

    expect(resolved.omittedReason).toBeUndefined()
    expect(resolved.toolChoice).toBe("required")
    expect(resolved.counts).toMatchObject({
      discovered: 1,
      executable: 1,
      disabled: 0,
      colliding: 0,
      chatEnabled: 1
    })
    expect(resolved.tools).toEqual([
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
