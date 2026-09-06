import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  state: { activeProjectId: "project-a", activeNodeId: "scene-a" },
  load: vi.fn(),
  send: vi.fn(),
  request: vi.fn(),
  configListeners: new Set<() => void>(),
}))
vi.mock("@/store/writing-playground", () => ({
  useWritingPlaygroundStore: (selector: (state: typeof mocks.state) => unknown) => selector(mocks.state),
}))
vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: { apiProvider: string }) => unknown) => selector({ apiProvider: "openai" }),
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => ["test-model"] }))
vi.mock("@/services/tldw/TldwChat", () => ({
  TldwChatService: class { sendMessage = mocks.send },
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => mocks.request(...args) }))
vi.mock("@/services/service-prompts", async () => ({
  ...await vi.importActual<typeof import("@/services/service-prompts")>("@/services/service-prompts"),
  loadServicePromptSnapshot: (...args: unknown[]) => mocks.load(...args),
  subscribeToServicePromptConfigChanges: (listener: () => void) => {
    mocks.configListeners.add(listener)
    return () => mocks.configListeners.delete(listener)
  },
}))

import { AIAgentTab } from "../AIAgentTab"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const requestScope = {
  config: { serverUrl: "https://server-a.test", authMode: "multi-user" as const },
  userId: "owner-a",
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

let scope: AbortController
let snapshots: ServicePromptSnapshot[]

function send(text = "Help my story") {
  const input = screen.getByPlaceholderText("Type a message...")
  fireEvent.change(input, { target: { value: text } })
  fireEvent.keyDown(input, { key: "Enter" })
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.state = { activeProjectId: "project-a", activeNodeId: "scene-a" }
  scope = new AbortController()
  snapshots = []
  mocks.load.mockImplementation(async ([id]: string[], options: { signal?: AbortSignal } = {}) => {
    const snapshot: ServicePromptSnapshot = {
      scopeKey: "scope-a", requestScope, capability: "supported",
      definitions: { [id]: {
        definition: { id, parts: [{ key: "system", mode: "literal", required_variables: [] }] },
        parts: { system: `Custom ${id} {literal}` }, source: "user", revision: "revision-a",
      } },
      scopeSignal: options.signal ?? scope.signal,
      scopeInvalidatedSignal: scope.signal,
      release: vi.fn(),
    }
    snapshots.push(snapshot)
    return snapshot
  })
  mocks.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.includes("/scenes/")) return { title: "Opening", content_plain: "Scene text" }
    if (path.endsWith("/characters")) return { characters: [{ name: "Ari", role: "hero" }] }
    if (path.endsWith("/world-info")) return { items: [{ name: "Harbor", kind: "location" }] }
    throw new Error(`Unexpected request ${path}`)
  })
  mocks.send.mockResolvedValue("A useful answer")
})
afterEach(cleanup)

describe("Writing Agent service prompts", () => {
  it.each([
    ["Quick", "quick", 0.3, 256],
    ["Planning", "planning", 0.6, 1024],
    ["Brainstorm", "brainstorm", 0.9, 1024],
  ] as const)("uses the captured %s instruction with unchanged context and provider settings", async (label, mode, temperature, maxTokens) => {
    render(<AIAgentTab isOnline />)
    fireEvent.click(screen.getByText(label))
    send()
    await screen.findByText("A useful answer")
    expect(mocks.load).toHaveBeenCalledWith([`writing.agent.${mode}`], { signal: expect.any(AbortSignal) })
    expect(mocks.send).toHaveBeenCalledWith([{ role: "user", content: "Help my story" }], expect.objectContaining({
      systemPrompt: `Custom writing.agent.${mode} {literal}\n\n--- Manuscript Context ---\n[Current Scene: Opening]\nScene text\n\n[Characters]\n- Ari (hero)\n\n[World Info]\n- Harbor (location)`,
      model: "test-model", apiProvider: "openai", temperature, maxTokens,
      requestScope, signal: expect.any(AbortSignal),
    }))
    for (const [request] of mocks.request.mock.calls) {
      expect(request).toMatchObject({
        servicePromptConfig: { ...requestScope.config, expectedUserId: "owner-a" },
        headers: { "X-TLDW-Expected-User-ID": "owner-a" },
        abortSignal: expect.any(AbortSignal),
      })
    }
  })

  it("retains bounded manuscript snippets", async () => {
    mocks.request.mockImplementation(async ({ path }: { path: string }) => path.includes("/scenes/")
      ? { title: "Long scene", content_plain: "x".repeat(2001) }
      : path.endsWith("/characters")
        ? { characters: Array.from({ length: 11 }, (_, i) => ({ name: `Person${i}`, role: "hero" })) }
        : { items: Array.from({ length: 11 }, (_, i) => ({ name: `Place${i}`, kind: "location" })) })
    render(<AIAgentTab isOnline />)
    send()
    await screen.findByText("A useful answer")
    const prompt = mocks.send.mock.calls[0][1].systemPrompt
    expect(prompt).toContain("x".repeat(2000) + "...")
    expect(prompt).not.toContain("Person10")
    expect(prompt).not.toContain("Place10")
  })

  it.each(["mode", "project", "scope"] as const)("discards pending replies after a %s change and clears old history", async (boundary) => {
    const response = deferred<string>()
    mocks.send.mockReturnValue(response.promise)
    const view = render(<AIAgentTab isOnline />)
    send("Old owner question")
    await waitFor(() => expect(mocks.send).toHaveBeenCalledTimes(1))
    if (boundary === "mode") fireEvent.click(screen.getByText("Planning"))
    if (boundary === "project") {
      mocks.state = { activeProjectId: "project-b", activeNodeId: "scene-b" }
      view.rerender(<AIAgentTab isOnline />)
    }
    if (boundary === "scope") act(() => scope.abort())
    await act(async () => response.resolve("Old owner answer"))
    expect(screen.queryByText("Old owner answer")).not.toBeInTheDocument()
    expect(screen.queryByText("Old owner question")).not.toBeInTheDocument()
  })

  it("does not dispatch generation after scope changes during context loading", async () => {
    const scene = deferred<unknown>()
    mocks.request.mockReturnValueOnce(scene.promise)
    render(<AIAgentTab isOnline />)
    send()
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
    act(() => scope.abort())
    await act(async () => scene.resolve({ title: "Old", content_plain: "Old private scene" }))
    expect(mocks.send).not.toHaveBeenCalled()
    expect(mocks.request).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("Help my story")).not.toBeInTheDocument()
  })

  it("invalidates completed history when the account changes between requests", async () => {
    render(<AIAgentTab isOnline />)
    send()
    await screen.findByText("A useful answer")
    act(() => scope.abort())
    expect(screen.queryByText("A useful answer")).not.toBeInTheDocument()
    expect(screen.queryByText("Help my story")).not.toBeInTheDocument()
  })

  it("preserves ordinary context-error fallback but stops invalid prompt loads", async () => {
    mocks.request.mockRejectedValue(new Error("Scene unavailable"))
    const view = render(<AIAgentTab isOnline />)
    send()
    await screen.findByText("A useful answer")
    expect(mocks.send.mock.calls[0][1].systemPrompt).toBe("Custom writing.agent.quick {literal}")
    view.unmount()
    mocks.send.mockClear()
    mocks.request.mockClear()
    mocks.load.mockRejectedValue(new Error("Invalid saved prompt"))
    render(<AIAgentTab isOnline />)
    send()
    await screen.findByText("Error: Invalid saved prompt")
    expect(mocks.request).not.toHaveBeenCalled()
    expect(mocks.send).not.toHaveBeenCalled()
  })

  it("does not publish stale errors or let an old request clear a newer loading state", async () => {
    const old = deferred<string>()
    const next = deferred<string>()
    mocks.send.mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise)
    render(<AIAgentTab isOnline />)
    send()
    await waitFor(() => expect(mocks.send).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByText("Planning"))
    send("New question")
    await waitFor(() => expect(mocks.send).toHaveBeenCalledTimes(2))
    await act(async () => old.reject(new Error("Old private error")))
    expect(screen.queryByText("Error: Old private error")).not.toBeInTheDocument()
    expect(screen.getByPlaceholderText("Type a message...")).toBeDisabled()
    await act(async () => next.resolve("New answer"))
    expect(screen.getByText("New answer")).toBeVisible()
  })

  it("releases its retained scope and aborts pending work on unmount", async () => {
    const response = deferred<string>()
    mocks.send.mockReturnValue(response.promise)
    const view = render(<AIAgentTab isOnline />)
    send()
    await waitFor(() => expect(mocks.send).toHaveBeenCalledTimes(1))
    view.unmount()
    expect(mocks.send.mock.calls[0][1].signal.aborted).toBe(true)
    expect(snapshots[0].release).toHaveBeenCalled()
    await act(async () => response.resolve("Too late"))
  })

  it("does not carry a question from an unverified failed prompt load into another account", async () => {
    mocks.load.mockRejectedValueOnce(new Error("Prompt server unavailable"))
    render(<AIAgentTab isOnline />)
    send("Private question before identity was resolved")
    await screen.findByText("Error: Prompt server unavailable")
    send("New account question")
    await screen.findByText("A useful answer")
    expect(mocks.send.mock.calls[0][0]).toEqual([{ role: "user", content: "New account question" }])
  })

  it.each(["config", "credentials"])("clears an unbound lookup error on a %s boundary", async (boundary) => {
    mocks.load.mockRejectedValueOnce(new Error("Old account lookup error"))
    render(<AIAgentTab isOnline />)
    send()
    await screen.findByText("Error: Old account lookup error")
    act(() => {
      if (boundary === "config") mocks.configListeners.forEach((listener) => listener())
      else window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    })
    expect(screen.queryByText("Error: Old account lookup error")).not.toBeInTheDocument()
  })

  it("releases a late prompt snapshot without requesting context after a project change", async () => {
    const snapshot = await mocks.load(["writing.agent.quick"])
    const pending = deferred<ServicePromptSnapshot>()
    mocks.load.mockReturnValueOnce(pending.promise)
    const view = render(<AIAgentTab isOnline />)
    send()
    mocks.state = { activeProjectId: "project-b", activeNodeId: "scene-b" }
    view.rerender(<AIAgentTab isOnline />)
    await act(async () => pending.resolve(snapshot))
    expect(mocks.request).not.toHaveBeenCalled()
    expect(snapshot.release).toHaveBeenCalled()
  })

  it("captures edits only for the next send while retaining same-scope history", async () => {
    const scene = deferred<unknown>()
    mocks.request.mockReturnValueOnce(scene.promise)
    render(<AIAgentTab isOnline />)
    send("First question")
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
    const next = await mocks.load(["writing.agent.quick"])
    mocks.load.mockResolvedValueOnce({ ...next, definitions: {
      ...next.definitions,
      "writing.agent.quick": { ...next.definitions["writing.agent.quick"]!, parts: { system: "Edited instructions" } },
    } })
    await act(async () => scene.resolve({ title: "Opening", content_plain: "Scene text" }))
    await screen.findByText("A useful answer")
    expect(mocks.send.mock.calls[0][1].systemPrompt).toMatch(/^Custom writing.agent.quick/)
    send("Second question")
    await waitFor(() => expect(mocks.send).toHaveBeenCalledTimes(2))
    expect(mocks.send.mock.calls[1][1].systemPrompt).toMatch(/^Edited instructions/)
    expect(mocks.send.mock.calls[1][0]).toEqual([
      { role: "user", content: "First question" },
      { role: "assistant", content: "A useful answer" },
      { role: "user", content: "Second question" },
    ])
  })
})
