import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useQuickChat } from "@/hooks/useQuickChat"
import { useQuickChatStore } from "@/store/quick-chat"

const remote = vi.hoisted(() => ({
  selectedModel: "",
  models: vi.fn(),
  stream: vi.fn()
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({ selectedModel: remote.selectedModel })
}))
vi.mock("wxt/browser", () => ({
  browser: {
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } }
  }
}))
vi.mock("@/services/tldw", () => ({
  tldwModels: { getModels: (...args: unknown[]) => remote.models(...args) }
}))
// Keep the actual helper, provider resolver, Chat service and request builder.
// Replace only catalog/config I/O and the final stream transport.
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    getConfig: async () => null,
    streamChatCompletion: (...args: unknown[]) => remote.stream(...args)
  }
}))

const held = new Set<() => void>()
const pending = new Set<Promise<void>>()
beforeEach(() => {
  remote.models.mockReset().mockResolvedValue([])
  remote.stream.mockReset().mockImplementation(async function* () {
    yield { choices: [{ delta: { content: "Current answer" } }] }
  })
  useQuickChatStore.setState({
    isOpen: true,
    messages: [],
    isStreaming: false,
    modelOverride: null,
    assistantMode: "chat"
  })
})
afterEach(async () => {
  cleanup()
  for (const release of held) release()
  await Promise.allSettled([...pending])
  held.clear()
  pending.clear()
})

describe("Quick Chat selected model reaches the real Chat request", () => {
  it.each([
    [
      "tldw:unsloth/Qwen3.8-27B-GGUF:Q8_0",
      "unsloth/Qwen3.8-27B-GGUF:Q8_0",
      "llama.cpp"
    ],
    [
      "tldw:anthropic/claude-4.5-sonnet",
      "anthropic/claude-4.5-sonnet",
      "openrouter"
    ],
    ["tldw:family:model:Q4", "family:model:Q4", "llama.cpp"],
    ["family:model:Q4", "family:model:Q4", "llama.cpp"],
    ["llamacpp:Vendor/Model:Q4", "Vendor/Model:Q4", "llama.cpp"]
  ])(
    "routes %s without changing the provider's model ID",
    async (selection, model, provider) => {
      remote.selectedModel = selection
      remote.models.mockResolvedValue([
        { id: model, name: model, provider, type: "chat" }
      ])
      const { result } = renderHook(() => useQuickChat())
      await act(async () => {
        await result.current.sendMessage("Exact current question")
      })
      expect(remote.stream).toHaveBeenCalledOnce()
      expect(remote.stream.mock.calls[0][0]).toMatchObject({
        model,
        api_provider: provider,
        stream: true,
        messages: [{ role: "user", content: "Exact current question" }]
      })
      expect(
        useQuickChatStore.getState().messages.map((row) => row.content)
      ).toEqual(["Exact current question", "Current answer"])
    }
  )

  it.each(["account", "cancel", "unmount"])(
    "does not dispatch after a pending catalog lookup is retired by %s",
    async (retirement) => {
      remote.selectedModel = "tldw:family:model:Q4"
      let release!: (value: unknown[]) => void
      remote.models.mockReturnValue(
        new Promise<unknown[]>((resolve) => {
          release = resolve
        })
      )
      held.add(() => release([]))
      const hook = renderHook(() => useQuickChat())
      let request!: Promise<void>
      act(() => {
        request = hook.result.current.sendMessage("Alice private question")
        pending.add(request)
      })
      await waitFor(() => expect(remote.models).toHaveBeenCalledOnce())
      act(() => {
        if (retirement === "account")
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        else if (retirement === "cancel") hook.result.current.cancelStream()
        else hook.unmount()
      })
      await act(async () => {
        release([
          { id: "family:model:Q4", provider: "llama.cpp", type: "chat" }
        ])
        await request
      })
      expect(remote.stream).not.toHaveBeenCalled()
      expect(
        useQuickChatStore
          .getState()
          .messages.some((row) => row.content === "Current answer")
      ).toBe(false)
    }
  )
})
