/**
 * Playground integration harness.
 *
 * Mounts the real chat `Playground` — its own `HistorySelectionProvider`, the
 * real `PlaygroundForm` composer, `PlaygroundChat` transcript, the
 * `useChatActions` send path and `useClearChat` — inside the providers the app
 * shell supplies (router, antd App, React Query, PageAssist and demo-mode
 * contexts). Only process boundaries are replaced:
 *   - network: `globalThis.fetch` -> an in-memory fake tldw server that records
 *     every request (see ./fake-tldw-server.ts)
 *   - IndexedDB: `@/db/dexie/schema` -> an in-memory Dexie stand-in
 *     (jsdom has no IndexedDB and the repo does not ship fake-indexeddb)
 *   - extension APIs: `@plasmohq/storage(/hook)` and `wxt/browser` -> the
 *     web build's shims, exactly what the Next.js WebUI aliases them to
 *   - i18n: `react-i18next` -> English-locale translator (./harness-i18n.tsx)
 *
 * Import this module before anything else in a test file so the mocks below
 * are registered before the Playground module graph loads, and call
 * `resetPlaygroundHarness()` in `beforeEach`.
 */
import React from "react"
import { expect, vi } from "vitest"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { App as AntdApp, ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"

import { Playground } from "../../Playground"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { DemoModeProvider } from "@/context/demo-mode"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { useStoreMessageOption } from "@/store/option"
import { useStoreMessage } from "@/store"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreChatModelSettings } from "@/store/model"
import { useConnectionStore } from "@/store/connection"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { resetHistoryTurnRegistry } from "@/services/history-turn-keep"
import { resetServerChatSaveStatus } from "@/store/server-chat-save-status"
import { tldwModels } from "@/services/tldw"
import { clearChatModelsCache } from "@/services/tldw-server"
import { ConnectionPhase } from "@/types/connection"
import type { MemoryDexie } from "./memory-dexie"
import {
  FAKE_API_KEY,
  FAKE_SELECTED_MODEL,
  FAKE_SERVER_URL,
  type FakeTldwServer,
  type JsonBody
} from "./fake-tldw-server"

const harnessDb = vi.hoisted(() => ({ current: null as unknown }))

vi.mock("@/db/dexie/schema", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/db/dexie/schema")>()
  const { createMemoryDexie } = await import("./memory-dexie")
  harnessDb.current = harnessDb.current ?? createMemoryDexie()
  return { ...actual, db: harnessDb.current }
})

vi.mock("@plasmohq/storage", () =>
  import("../../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@plasmohq/storage/hook", () =>
  import("../../../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook")
)
vi.mock("wxt/browser", () =>
  import("../../../../../../../../tldw-frontend/extension/shims/wxt-browser")
)

vi.mock("react-i18next", async () => (await import("./harness-i18n")).reactI18nextStandIn)

export * from "./fake-tldw-server"

/** antd-heavy jsdom trees take 0.6-2 s per interaction; give async queries room. */
export const HARNESS_WAIT = { timeout: 15_000 } as const

export const getHarnessDb = () => harnessDb.current as MemoryDexie

const installDomPolyfills = () => {
  // jsdom lacks element scrolling; the transcript auto-scrolls on every turn.
  if (typeof Element.prototype.scrollTo !== "function") {
    Element.prototype.scrollTo = function scrollTo() {}
  }
  if (typeof Element.prototype.scrollIntoView !== "function") {
    Element.prototype.scrollIntoView = function scrollIntoView() {}
  }
}

const seedBrowserState = (persistedModel: string | null) => {
  localStorage.clear()
  sessionStorage.clear()
  localStorage.setItem(
    "tldwConfig",
    JSON.stringify({
      serverUrl: FAKE_SERVER_URL,
      authMode: "single-user",
      apiKey: FAKE_API_KEY
    })
  )
  if (persistedModel) localStorage.setItem("selectedModel", JSON.stringify(persistedModel))
}

/**
 * Restore a clean world between tests: in-memory IndexedDB, browser storage,
 * the shared Zustand stores the chat surface writes, client caches keyed on
 * chats, and the fetch stub.
 */
const resetAppStores = () => {
  for (const store of [
    useStoreMessageOption,
    useStoreMessage,
    usePlaygroundSessionStore,
    useStoreChatModelSettings,
    useConnectionStore,
    useChatSurfaceCoordinatorStore
  ] as Array<{ setState: (state: unknown, replace: boolean) => void; getInitialState: () => unknown }>) {
    store.setState(store.getInitialState(), true)
  }
  const client = tldwClient as unknown as {
    chatMessagesCache?: Map<unknown, unknown>
    chatMessagesInFlight?: Map<unknown, unknown>
  }
  client.chatMessagesCache?.clear()
  client.chatMessagesInFlight?.clear()
  // Page-lifetime chat state: running turns and acknowledged server writes.
  resetHistoryTurnRegistry()
  resetServerChatSaveStatus()
}

export const resetPlaygroundHarness = async () => {
  vi.unstubAllGlobals()
  getHarnessDb()?.resetAll()
  // Model catalogs are cached per module; a test's catalog must not leak into the next.
  clearChatModelsCache()
  await tldwModels.clearCache()
  localStorage.clear()
  sessionStorage.clear()
  resetAppStores()
}

const snapshotStorage = (storage: Storage) =>
  Object.fromEntries(
    Array.from({ length: storage.length }, (_, index) => storage.key(index) as string).map(
      (key) => [key, storage.getItem(key) as string]
    )
  )

const restoreStorage = (storage: Storage, entries: Record<string, string>) => {
  storage.clear()
  for (const [key, value] of Object.entries(entries)) storage.setItem(key, value)
}

/**
 * Model a browser reload of the tab: the mounted app goes away and every
 * in-memory store starts over, while IndexedDB, localStorage and the tab's
 * sessionStorage survive. Requests the old page left in flight are abandoned
 * (never answered), as they are when the page unloads. Remount with
 * `renderPlayground({ server, keepBrowserState: true })`.
 */
export const simulatePageReload = async (view: PlaygroundView) => {
  view.unmount()
  const local = snapshotStorage(localStorage)
  const session = snapshotStorage(sessionStorage)
  resetAppStores()
  restoreStorage(localStorage, local)
  restoreStorage(sessionStorage, session)
  await usePlaygroundSessionStore.persist.rehydrate()
}

export type RenderPlaygroundOptions = {
  server: FakeTldwServer
  /** UI rendered beside the Playground, outside its HistorySelectionProvider (like the app header). */
  extras?: React.ReactNode
  initialPath?: string
  /** The model persisted from an earlier session; `null` simulates a first run. */
  persistedModel?: string | null
  /** Remount over the browser storage left by an earlier mount (reload, return to /chat). */
  keepBrowserState?: boolean
}

// Vitest resolves the real react-router-dom; the Next.js typecheck maps it to the web
// shim, whose MemoryRouter type omits `initialEntries`.
const Router = MemoryRouter as React.ComponentType<{
  initialEntries?: string[]
  children?: React.ReactNode
}>

/** Mount the real Playground against the fake server and wait until it is connected. */
export const renderPlayground = async ({
  server,
  extras,
  initialPath = "/chat",
  persistedModel = FAKE_SELECTED_MODEL,
  keepBrowserState = false
}: RenderPlaygroundOptions) => {
  installDomPolyfills()
  if (!keepBrowserState) seedBrowserState(persistedModel)
  vi.stubGlobal("fetch", server.fetch)
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } }
  })
  const user = userEvent.setup()
  const view = render(
    <Router initialEntries={[initialPath]}>
      <ConfigProvider>
        <AntdApp>
          <QueryClientProvider client={queryClient}>
            <PageAssistProvider>
              <DemoModeProvider>
                {extras}
                <Playground />
              </DemoModeProvider>
            </PageAssistProvider>
          </QueryClientProvider>
        </AntdApp>
      </ConfigProvider>
    </Router>
  )
  // The app shell's connection bootstrap (header/status probes) runs checkOnce;
  // the Playground itself only reads the resulting phase.
  await act(async () => {
    await useConnectionStore.getState().checkOnce({ force: true })
  })
  await waitFor(() => {
    const state = useConnectionStore.getState().state
    expect(state.isConnected && state.phase === ConnectionPhase.CONNECTED).toBe(true)
  }, HARNESS_WAIT)
  const composer = await screen.findByPlaceholderText(/Type a message/i, {}, HARNESS_WAIT)
  return { ...view, user, queryClient, server, composer }
}

export type PlaygroundView = Awaited<ReturnType<typeof renderPlayground>>

/** Wait until no turn is streaming or processing. */
export const waitForChatIdle = async () => {
  await waitFor(() => {
    const state = useStoreMessageOption.getState()
    expect(state.streaming || state.isProcessing).toBe(false)
  }, HARNESS_WAIT)
}

/**
 * Type a message into the real composer and press Enter, then wait until the
 * turn settles (streaming ends). Returns the requests made during the turn.
 */
export const sendFromComposer = async (view: PlaygroundView, text: string) => {
  const before = view.server.requests.length
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(() => expect(view.composer).toHaveValue(""), HARNESS_WAIT)
  await waitForChatIdle()
  return view.server.requests.slice(before)
}

/** The chat messages a completion request sent, as role/content pairs. */
export const completionMessages = (request: { body?: JsonBody }) =>
  ((request.body?.messages ?? []) as Array<{ role: string; content: unknown }>).map((message) => ({
    role: message.role,
    content: typeof message.content === "string" ? message.content : JSON.stringify(message.content)
  }))

/**
 * Stand-in for the app header's "New chat" button (Header.tsx `startSavedChat`):
 * rendered outside the Playground's HistorySelectionProvider and wired to the
 * real `useClearChat` hook, exactly like the header and sidebar "+" are.
 */
export const HeaderNewChatButton = () => {
  const clearChat = useClearChat()
  const setTemporaryChat = useStoreMessageOption((state) => state.setTemporaryChat)
  return (
    <button
      type="button"
      onClick={() => {
        if (clearChat() === false) return
        setTemporaryChat(false)
      }}>
      Harness header: New chat
    </button>
  )
}

/**
 * Stand-in for a sidebar row: opens a saved server chat through the real
 * `useSelectServerChat` hook, as ServerChatList does on click.
 */
export const SidebarServerChatButton = ({ chatId, title }: { chatId: string; title: string }) => {
  const select = useSelectServerChat()
  return (
    <button
      type="button"
      onClick={() =>
        select({
          id: chatId,
          title,
          state: "in-progress",
          version: 1,
          source: "webui-chat"
        } as Parameters<typeof select>[0])
      }>
      Harness sidebar: open {title}
    </button>
  )
}

/** Click the header stand-in rendered via `extras: <HeaderNewChatButton />`. */
export const clickHeaderNewChat = async (view: PlaygroundView) => {
  await view.user.click(screen.getByRole("button", { name: "Harness header: New chat" }))
}

/**
 * Open a saved server chat through the sidebar stand-in and wait until its
 * last message is rendered and the history selection has captured it.
 */
export const openSavedServerChat = async (
  view: PlaygroundView,
  { title, lastMessage }: { title: string; lastMessage: string }
) => {
  await view.user.click(screen.getByRole("button", { name: `Harness sidebar: open ${title}` }))
  await screen.findByText(lastMessage, {}, HARNESS_WAIT)
  await waitFor(() => {
    expect(
      view.server.find("POST", /^\/api\/v1\/chat\/conversations\/[^/]+\/history\/selection$/).length
    ).toBeGreaterThan(0)
  }, HARNESS_WAIT)
  await waitForChatIdle()
}

/** Text of every visible antd notification (titles and descriptions). */
export const visibleNotifications = () =>
  [...document.querySelectorAll(".ant-notification-notice")].map((node) =>
    (node.textContent ?? "").trim()
  )

/**
 * Some defects surface as raw errors rejected from click handlers. Vitest
 * would report those as run errors even though the reproduction is the
 * expected failure, so record them for the duration of `run` and restore
 * Vitest's own listeners afterwards.
 */
export const withUnhandledRejectionsRecorded = async <T,>(
  run: (reasons: unknown[]) => Promise<T>
): Promise<T> => {
  const reasons: unknown[] = []
  const previous = process.listeners("unhandledRejection")
  const record = (reason: unknown) => {
    reasons.push(reason)
  }
  process.removeAllListeners("unhandledRejection")
  process.on("unhandledRejection", record)
  try {
    return await run(reasons)
  } finally {
    // Let Node deliver rejections raised during the last interaction.
    await new Promise((resolve) => setTimeout(resolve, 0))
    process.off("unhandledRejection", record)
    for (const listener of previous) {
      process.on("unhandledRejection", listener as NodeJS.UnhandledRejectionListener)
    }
  }
}

export { act, screen, waitFor, within } from "@testing-library/react"
