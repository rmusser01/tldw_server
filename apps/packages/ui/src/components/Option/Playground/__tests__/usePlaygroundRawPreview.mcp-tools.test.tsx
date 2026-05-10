// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { usePlaygroundRawPreview } from "../hooks/usePlaygroundRawPreview"

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "openai")
}))

const buildDeps = (overrides: Partial<any> = {}) => ({
  composerModels: [{ id: "tool-model", capabilities: ["tools"] }],
  selectedModel: "tool-model",
  compareModeActive: false,
  compareSelectedModels: [],
  compareMaxModels: 4,
  currentChatModelSettings: {
    apiProvider: "openai"
  },
  history: [],
  systemPrompt: undefined,
  hasMcp: true,
  mcpHealthState: "healthy",
  mcpTools: [],
  toolChoice: "auto",
  temporaryChat: true,
  serverChatId: null,
  serverChatState: null,
  serverChatSource: null,
  selectedCharacter: null,
  messageSteeringMode: "none",
  messageSteeringForceNarrate: false,
  ragMediaIds: null,
  selectedKnowledge: null,
  ragPinnedResults: [],
  fileRetrievalEnabled: false,
  contextFiles: [],
  documentContext: [],
  selectedDocuments: [],
  imageBackendDefaultTrimmed: "",
  resolveSubmissionIntent: (message: string) => ({
    message,
    isImageCommand: false
  }),
  formImage: "",
  formMessage: "hello",
  researchContext: undefined,
  notificationApi: {
    error: vi.fn(),
    success: vi.fn()
  },
  t: (_key: string, defaultValueOrOptions?: any) =>
    typeof defaultValueOrOptions === "string" ? defaultValueOrOptions : _key,
  setToolsPopoverOpen: vi.fn(),
  ...overrides
})

describe("usePlaygroundRawPreview MCP tools", () => {
  it("normalizes chat tools in the raw request preview", async () => {
    const { result } = renderHook(() =>
      usePlaygroundRawPreview(
        buildDeps({
          mcpTools: [
            {
              name: "notes.search",
              description: "Search notes",
              parameters: {
                type: "object",
                properties: { q: { type: "string" } }
              },
              canExecute: true
            }
          ]
        })
      )
    )

    await act(async () => {
      await result.current.refreshRawRequestSnapshot()
    })

    const body = result.current.rawRequestSnapshot?.body as any
    expect(result.current.rawRequestSnapshot?.metadata).toMatchObject({
      toolChoice: "auto",
      toolCounts: {
        discovered: 1,
        executable: 1,
        disabled: 0,
        colliding: 0,
        chatEnabled: 1
      }
    })
    expect(body.tool_choice).toBe("auto")
    expect(body.tools).toEqual([
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

  it("omits tool fields when all candidate tools collide", async () => {
    const { result } = renderHook(() =>
      usePlaygroundRawPreview(
        buildDeps({
          mcpTools: [
            { name: "docs.search", canExecute: true },
            { name: "docs_search", canExecute: true }
          ]
        })
      )
    )

    await act(async () => {
      await result.current.refreshRawRequestSnapshot()
    })

    const body = result.current.rawRequestSnapshot?.body as any
    expect(body).not.toHaveProperty("tool_choice")
    expect(body).not.toHaveProperty("tools")
    expect(result.current.rawRequestSnapshot?.metadata).toMatchObject({
      toolOmissionReason: "no_enabled_executable_tools",
      toolCounts: {
        discovered: 2,
        executable: 2,
        disabled: 0,
        colliding: 2,
        chatEnabled: 0
      }
    })
  })

  it("keeps selected none out of the raw body and explains the omission in metadata", async () => {
    const { result } = renderHook(() =>
      usePlaygroundRawPreview(
        buildDeps({
          toolChoice: "none",
          mcpTools: [{ name: "notes.search", canExecute: true }]
        })
      )
    )

    await act(async () => {
      await result.current.refreshRawRequestSnapshot()
    })

    const body = result.current.rawRequestSnapshot?.body as any
    expect(body).not.toHaveProperty("tool_choice")
    expect(body).not.toHaveProperty("tools")
    expect(result.current.rawRequestSnapshot?.metadata).toMatchObject({
      toolOmissionReason: "tool_choice_none"
    })
  })

  it("omits tools for models without tool support and reports metadata", async () => {
    const { result } = renderHook(() =>
      usePlaygroundRawPreview(
        buildDeps({
          composerModels: [{ id: "plain-model", capabilities: [] }],
          selectedModel: "plain-model",
          mcpTools: [{ name: "notes.search", canExecute: true }]
        })
      )
    )

    await act(async () => {
      await result.current.refreshRawRequestSnapshot()
    })

    const body = result.current.rawRequestSnapshot?.body as any
    expect(body).not.toHaveProperty("tool_choice")
    expect(body).not.toHaveProperty("tools")
    expect(result.current.rawRequestSnapshot?.metadata).toMatchObject({
      toolOmissionReason: "model_lacks_tool_capability"
    })
  })

  it("uses the same resolver metadata for comparison previews", async () => {
    const { result } = renderHook(() =>
      usePlaygroundRawPreview(
        buildDeps({
          compareModeActive: true,
          selectedModel: null,
          compareSelectedModels: ["tool-model", "plain-model"],
          composerModels: [
            { id: "tool-model", capabilities: ["tools"] },
            { id: "plain-model", capabilities: [] }
          ],
          mcpTools: [{ name: "notes.search", canExecute: true }]
        })
      )
    )

    await act(async () => {
      await result.current.refreshRawRequestSnapshot()
    })

    const body = result.current.rawRequestSnapshot?.body as any
    expect(body.requests).toHaveLength(2)
    expect(body.requests[0].tool_choice).toBe("auto")
    expect(body.requests[0].tools).toHaveLength(1)
    expect(body.requests[1]).not.toHaveProperty("tool_choice")
    expect(body.requests[1]).not.toHaveProperty("tools")
    expect(result.current.rawRequestSnapshot?.metadata).toMatchObject({
      toolRequests: [
        {
          model: "tool-model",
          toolChoice: "auto"
        },
        {
          model: "plain-model",
          toolOmissionReason: "model_lacks_tool_capability"
        }
      ]
    })
  })
})
