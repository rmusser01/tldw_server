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
  })
})
