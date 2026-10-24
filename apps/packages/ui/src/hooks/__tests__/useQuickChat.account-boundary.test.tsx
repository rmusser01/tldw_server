import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import { ConfigProvider } from "antd"
import i18n from "i18next"
import { initReactI18next } from "react-i18next"
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi
} from "vitest"
import { QuickChatInput } from "@/components/Common/QuickChatHelper/QuickChatInput"
import { useQuickChat } from "@/hooks/useQuickChat"
import { useQuickChatStore } from "@/store/quick-chat"

const transport = vi.hoisted(() => ({
  stream: vi.fn(),
  cancel: vi.fn(),
  rag: vi.fn(),
  initialize: vi.fn()
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({ selectedModel: "llamacpp:fixture" })
}))
vi.mock("wxt/browser", () => ({
  browser: {
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } }
  }
}))
vi.mock("@/services/tldw/TldwChat", () => ({
  TldwChatService: class {
    streamMessage = transport.stream
    cancelStream = transport.cancel
  }
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { initialize: transport.initialize, ragSearch: transport.rag }
}))

const held = new Set<() => void>()
const running = new Set<Promise<void>>()
function track(promise: Promise<void>) {
  running.add(promise)
  return promise
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  held.add(() => resolve(undefined as T))
  return { promise, resolve, reject }
}
function boundary() {
  window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
}
beforeAll(async () => {
  await i18n
    .use(initReactI18next)
    .init({ lng: "en", fallbackLng: "en", resources: { en: { option: {} } } })
})
beforeEach(() => {
  transport.stream.mockReset()
  transport.cancel.mockReset()
  transport.rag.mockReset()
  transport.initialize.mockReset().mockResolvedValue(undefined)
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
  for (const settle of held) settle()
  await Promise.allSettled([...running])
  held.clear()
  running.clear()
})

describe("Quick Chat account boundary", () => {
  it("retires visible private conversation and helper settings at a principal change", () => {
    renderHook(() => useQuickChat())
    act(() => {
      const store = useQuickChatStore.getState()
      store.addMessage("user", "Alice private question")
      store.addMessage("assistant", "Alice private answer")
      store.setModelOverride("Alice private model")
      store.setAssistantMode("docs_rag")
      boundary()
    })
    const state = useQuickChatStore.getState()
    expect(state.messages).toEqual([])
    expect(state.isOpen).toBe(false)
    expect(state.modelOverride).toBeNull()
    expect(state.assistantMode).toBe("chat")
    expect(transport.cancel).toHaveBeenCalled()
  })

  it("clears an unsent helper input without requiring a close/unmount", () => {
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QuickChatInput
          onSend={vi.fn()}
          onCancel={vi.fn()}
          isStreaming={false}
        />
      </ConfigProvider>
    )
    const input = screen.getByRole("textbox")
    fireEvent.change(input, {
      target: { value: "Alice unsent private draft" }
    })
    act(boundary)
    expect(input).toHaveValue("")
  })

  it("keeps a conversation and unsent draft during an authority-preserving refresh", () => {
    renderHook(() => useQuickChat())
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QuickChatInput
          onSend={vi.fn()}
          onCancel={vi.fn()}
          isStreaming={false}
        />
      </ConfigProvider>
    )
    act(() =>
      useQuickChatStore.getState().addMessage("user", "Alice current question")
    )
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Alice current draft" }
    })
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false }
        })
      )
    )
    expect(useQuickChatStore.getState().messages[0].content).toBe(
      "Alice current question"
    )
    expect(screen.getByRole("textbox")).toHaveValue("Alice current draft")
    expect(transport.cancel).not.toHaveBeenCalled()
  })

  it.each(["chunk", "error"])(
    "does not let a retired stream %s modify Bob's active turn",
    async (mode) => {
      const old = deferred<string>(),
        current = deferred<string>()
      transport.stream.mockImplementationOnce(async function* () {
        yield await old.promise
      })
      transport.stream.mockImplementationOnce(async function* () {
        yield await current.promise
      })
      const { result } = renderHook(() => useQuickChat())
      let first!: Promise<void>, second!: Promise<void>
      act(() => {
        first = track(result.current.sendMessage("Alice private question"))
      })
      await waitFor(() => expect(transport.stream).toHaveBeenCalledTimes(1))
      act(boundary)
      // Bob opens a fresh helper; retired work must not mutate its new slot.
      act(() => {
        useQuickChatStore.getState().clearMessages()
        useQuickChatStore.getState().setIsOpen(true)
      })
      act(() => {
        second = track(result.current.sendMessage("Bob current question"))
      })
      await waitFor(() => expect(transport.stream).toHaveBeenCalledTimes(2))
      await act(async () => {
        if (mode === "error") old.reject(new Error("Alice private failure"))
        else old.resolve("Alice private late output")
        await first
      })
      expect(useQuickChatStore.getState().isStreaming).toBe(true)
      expect(
        useQuickChatStore.getState().messages.map((row) => row.content)
      ).toEqual(["Bob current question", ""])
      await act(async () => {
        current.resolve("Bob current answer")
        await second
      })
      expect(
        useQuickChatStore.getState().messages.map((row) => row.content)
      ).toEqual(["Bob current question", "Bob current answer"])
      expect(useQuickChatStore.getState().isStreaming).toBe(false)
    }
  )

  it("does not publish a late documentation reply after a principal change", async () => {
    const old = deferred<unknown>()
    transport.rag.mockReturnValueOnce(old.promise)
    const { result } = renderHook(() => useQuickChat())
    let pending!: Promise<void>
    act(() => {
      pending = track(
        result.current.sendMessage("Alice docs question", {
          mode: "docs_rag"
        })
      )
    })
    await waitFor(() => expect(transport.rag).toHaveBeenCalledOnce())
    act(boundary)
    act(() => {
      useQuickChatStore.getState().clearMessages()
      useQuickChatStore.getState().addMessage("user", "Bob current question")
      useQuickChatStore.getState().addMessage("assistant", "")
    })
    await act(async () => {
      old.resolve({
        generated_answer: "Alice private documentation",
        documents: []
      })
      await pending
    })
    expect(
      useQuickChatStore.getState().messages.map((row) => row.content)
    ).toEqual(["Bob current question", ""])
  })

  it("does not start an old documentation request after initialization crosses accounts", async () => {
    const init = deferred<void>()
    transport.initialize.mockReturnValueOnce(init.promise)
    const { result } = renderHook(() => useQuickChat())
    let pending!: Promise<void>
    act(() => {
      pending = track(
        result.current.sendMessage("Alice not-yet-dispatched question", {
          mode: "docs_rag"
        })
      )
    })
    await waitFor(() => expect(transport.initialize).toHaveBeenCalledOnce())
    act(boundary)
    await act(async () => {
      init.resolve()
      await pending
    })
    expect(transport.rag).not.toHaveBeenCalled()
    expect(useQuickChatStore.getState().messages).toEqual([])
  })

  it("keeps a newly mounted helper's turn intact when an unmounted helper finishes", async () => {
    const old = deferred<unknown>(),
      current = deferred<string>()
    transport.rag.mockReturnValueOnce(old.promise)
    transport.stream.mockImplementationOnce(async function* () {
      yield await current.promise
    })
    const first = renderHook(() => useQuickChat())
    let pending!: Promise<void>, fresh!: Promise<void>
    act(() => {
      pending = track(
        first.result.current.sendMessage("Alice old docs", {
          mode: "docs_rag"
        })
      )
    })
    await waitFor(() => expect(transport.rag).toHaveBeenCalledOnce())
    first.unmount()
    act(() => useQuickChatStore.getState().clearMessages())
    const second = renderHook(() => useQuickChat())
    act(() => {
      fresh = track(second.result.current.sendMessage("Bob new question"))
    })
    await waitFor(() => expect(transport.stream).toHaveBeenCalledOnce())
    await act(async () => {
      old.resolve({ generated_answer: "Alice old output", documents: [] })
      await pending
    })
    expect(
      useQuickChatStore.getState().messages.map((row) => row.content)
    ).toEqual(["Bob new question", ""])
    expect(useQuickChatStore.getState().isStreaming).toBe(true)
    await act(async () => {
      current.resolve("Bob answer")
      await fresh
    })
    expect(
      useQuickChatStore.getState().messages.map((row) => row.content)
    ).toEqual(["Bob new question", "Bob answer"])
  })
  it("retires retained helper settings when accounts change while the helper is closed", () => {
    const first = renderHook(() => useQuickChat())
    act(() => {
      useQuickChatStore.getState().setModelOverride("Alice private model")
      useQuickChatStore.getState().setAssistantMode("docs_rag")
      first.result.current.closeModal()
    })
    first.unmount()
    act(boundary)
    renderHook(() => useQuickChat())
    expect(useQuickChatStore.getState().modelOverride).toBeNull()
    expect(useQuickChatStore.getState().assistantMode).toBe("chat")
  })
})
