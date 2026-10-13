/**
 * Drive the built tldw browser extension's side panel against the live-tier
 * backend (ux-regression project).
 *
 * The live-tier runner builds the extension before the run
 * (`bun run build:chrome:prod` in apps/extension) and passes its output
 * directory as TLDW_UXR_EXTENSION_DIR. A missing or dev-server build fails the
 * test with instructions; it is never skipped.
 *
 * Chromium loads the unpacked extension into a fresh persistent profile. The
 * default headless shell cannot load extensions, so the context uses the full
 * Chromium build (`channel: "chromium"`) in new headless mode, which works on
 * macOS and on Linux CI without a display.
 *
 * The side panel is opened as `sidepanel.html#/chat` in an ordinary tab. A real
 * side panel has no sender tab, so its chat state lives under the ":global"
 * storage key; an init script makes `tldw:get-tab-id` answer null the same
 * way, so state survives closing and reopening the panel page.
 */
import { chromium, expect, type BrowserContext, type Locator, type Page, type Worker } from "@playwright/test"
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { test as base } from "./fixtures"
import { TEST_CONFIG } from "./helpers"

export const EXTENSION_DIR_ENV = "TLDW_UXR_EXTENSION_DIR"

/** Where `bun run build:chrome:prod` in apps/extension writes the unpacked Chrome build. */
const DEFAULT_EXTENSION_DIR = path.resolve(__dirname, "../../../extension/.output/chrome-mv3")

/** Roughly the width Chrome gives a docked side panel. */
const SIDE_PANEL_VIEWPORT = { width: 420, height: 900 }

const CONNECTED_STATUS = /^Connected to your tldw server/

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

/** The unpacked extension to load, or a clear error saying how to build it. */
export function resolveBuiltExtensionDir(dir = process.env[EXTENSION_DIR_ENV] || DEFAULT_EXTENSION_DIR): string {
  const missing = ["manifest.json", "background.js", "sidepanel.html"].filter(
    (file) => !existsSync(path.join(dir, file))
  )
  if (missing.length) {
    throw new Error(
      `No built tldw extension at ${dir} (missing ${missing.join(", ")}). ` +
        "Run the suite through `bun run e2e:ux-regression`, which builds it, or build it with " +
        `\`bun run build:chrome:prod\` in apps/extension and set ${EXTENSION_DIR_ENV}.`
    )
  }
  const sidepanelHtml = readFileSync(path.join(dir, "sidepanel.html"), "utf8")
  if (sidepanelHtml.includes("/@vite/client") || sidepanelHtml.includes("http://localhost:")) {
    throw new Error(`${dir} is a WXT dev-server build; the side-panel specs need a production build.`)
  }
  return dir
}

export type ExtensionSession = {
  context: BrowserContext
  extensionId: string
  /** Open the side-panel chat in a new page and wait until it reports a server connection. */
  openSidePanel: () => Promise<Page>
  close: () => Promise<void>
}

const isExtensionWorker = (worker: Worker) => worker.url().startsWith("chrome-extension://")

/** Launch Chromium with the built extension, configured for `serverUrl` with an API key. */
export async function launchExtension({ serverUrl, apiKey }: { serverUrl: string; apiKey: string }): Promise<ExtensionSession> {
  const extensionDir = resolveBuiltExtensionDir()
  const userDataDir = mkdtempSync(path.join(tmpdir(), "tldw-uxr-extension-"))
  let context: BrowserContext | null = null
  const close = async () => {
    await context?.close().catch(() => undefined)
    rmSync(userDataDir, { recursive: true, force: true })
  }
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      channel: "chromium",
      headless: true,
      viewport: SIDE_PANEL_VIEWPORT,
      ignoreDefaultArgs: ["--disable-extensions"],
      args: [`--disable-extensions-except=${extensionDir}`, `--load-extension=${extensionDir}`],
    })
    const worker =
      context.serviceWorkers().find(isExtensionWorker) ??
      (await context.waitForEvent("serviceworker", { predicate: isExtensionWorker, timeout: 30_000 }))
    const extensionId = new URL(worker.url()).host

    await context.addInitScript(() => {
      if (!location.pathname.endsWith("/sidepanel.html") || !globalThis.chrome?.runtime?.sendMessage) return
      const send = chrome.runtime.sendMessage.bind(chrome.runtime) as (...args: unknown[]) => unknown
      ;(chrome.runtime as { sendMessage: unknown }).sendMessage = (message: { type?: string } | undefined, ...rest: unknown[]) =>
        message?.type === "tldw:get-tab-id" ? Promise.resolve({ ok: false, tabId: null }) : send(message, ...rest)
    })

    // Configure the server in chrome.storage.local only: the extension keeps
    // JSON-encoded values in chrome.storage.sync, and a raw object there fails
    // to parse.
    await worker.evaluate(
      async ({ serverUrl, apiKey }) => {
        await chrome.storage.local.set({
          tldwConfig: {
            serverUrl,
            authMode: "single-user",
            apiKey,
            authSource: "manual",
            credentialSource: "manual",
            apiKeyPersistence: "device",
            apiKeyServerOrigin: new URL(serverUrl).origin,
          },
          __tldw_first_run_complete: true,
          tldw_skip_landing_hub: true,
        })
      },
      { serverUrl, apiKey }
    )

    const openContext = context
    const openSidePanel = async () => {
      const page = await openContext.newPage()
      await page.goto(`chrome-extension://${extensionId}/sidepanel.html#/chat`)
      await expect(page.getByTestId("status-dot")).toHaveAttribute("aria-label", CONNECTED_STATUS, { timeout: 30_000 })
      return page
    }
    return { context, extensionId, openSidePanel, close }
  } catch (error) {
    await close()
    throw error
  }
}

/** `test` with an `extension` fixture: a fresh profile per test, screenshots on failure. */
export const test = base.extend<{ extension: ExtensionSession }>({
  extension: async ({}, use, testInfo) => {
    const session = await launchExtension({ serverUrl: TEST_CONFIG.serverUrl, apiKey: TEST_CONFIG.apiKey })
    try {
      await use(session)
      if (testInfo.status !== testInfo.expectedStatus) {
        for (const [index, page] of session.context.pages().entries()) {
          if (page.url().startsWith("chrome-extension://")) {
            await testInfo.attach(`extension-page-${index}.png`, {
              body: await page.screenshot().catch(() => Buffer.alloc(0)),
              contentType: "image/png",
            })
          }
        }
      }
    } finally {
      await session.close()
    }
  },
})

export { expect }

export type SidePanelTab = {
  id: string
  label: string
  /** The server chat the tab is bound to (tab metadata, else its snapshot). */
  serverChatId: string | null
  /** Text of the messages saved in the tab's snapshot, in order. */
  messages: string[]
}

export type SidePanelTabsState = { activeTabId: string | null; tabs: SidePanelTab[] }

/** Page object for the side-panel chat (`sidepanel.html#/chat`). */
export class SidePanelChat {
  readonly sidebar: Locator
  readonly searchInput: Locator
  readonly transcript: Locator

  constructor(readonly page: Page) {
    this.sidebar = page.getByTestId("sidepanel-chat-sidebar")
    this.searchInput = page.getByTestId("sidepanel-sidebar-search")
    this.transcript = page.getByRole("log", { name: "Chat messages" })
  }

  async openSidebar() {
    if (!(await this.sidebar.isVisible())) await this.page.getByRole("button", { name: "Expand sidebar" }).click()
    await expect(this.sidebar).toBeVisible()
  }

  /**
   * Close the sidebar to read the chat: at side-panel width the open sidebar
   * leaves the transcript too narrow to show its messages.
   */
  async closeSidebar() {
    if (!(await this.sidebar.isVisible())) return
    // The overlay sidebar sits over a click-to-dismiss backdrop (no test id or
    // name). Tap it beside the sidebar, as a user does; a docked sidebar has
    // no backdrop and collapses from the header toggle instead.
    const backdrop = this.page.locator('div.fixed.inset-0[aria-hidden="true"]')
    const viewport = this.page.viewportSize() ?? SIDE_PANEL_VIEWPORT
    if (await backdrop.isVisible()) {
      await backdrop.click({ position: { x: viewport.width - 10, y: Math.round(viewport.height / 2) } })
    } else {
      await this.page.getByRole("button", { name: "Collapse sidebar" }).click()
    }
    await expect(this.sidebar).toBeHidden()
  }

  async search(term: string) {
    await this.openSidebar()
    await this.searchInput.fill(term)
  }

  async clearSearch() {
    await this.openSidebar()
    await this.searchInput.fill("")
  }

  /**
   * A history-search result: a server chat or a local copy of one. A result
   * found by what was said in the chat shows that text between the title and
   * the source label (CS-02), and it is part of the button's name.
   */
  searchResult(title: string): Locator {
    return this.sidebar.getByRole("button", { name: new RegExp(`^${escapeRegExp(title)}(?: .+)? (Server|Local)\\b`) })
  }

  /** An open tab's row in the sidebar's tab list (shown while the search is empty). */
  tabRow(label: string): Locator {
    return this.sidebar.getByRole("button", { name: label, exact: true })
  }

  /**
   * Search the history for `title` and open its server result, as a user
   * looking up an old chat. Returns once the chat's tab is active; the sidebar
   * stays open.
   */
  async openServerChatFromSearch(title: string, chatId: string) {
    await this.search(title)
    const result = this.sidebar.getByRole("button", { name: new RegExp(`^${escapeRegExp(title)}(?: .+)? Server$`) })
    await expect(result).toBeVisible()
    await result.click()
    await expect
      .poll(async () => {
        const state = await this.readTabs()
        return state.tabs.find((tab) => tab.id === state.activeTabId)?.serverChatId ?? null
      }, { message: `the side panel should switch to a tab for chat ${chatId}` })
      .toBe(chatId)
  }

  /** Right-click a tab row and pick an item from its context menu. */
  async chooseTabMenuItem(label: string, item: string) {
    await this.tabRow(label).click({ button: "right" })
    await this.page.getByRole("menuitem", { name: item }).click()
  }

  /** The side panel's saved tabs, as persisted in chrome.storage.local. */
  async readTabs(): Promise<SidePanelTabsState> {
    return this.page.evaluate(async () => {
      type Snapshot = { serverChatId?: string | null; messages?: Array<{ message?: unknown }> }
      type Saved = {
        activeTabId?: string | null
        tabs?: Array<{ id: string; label: string; serverChatId?: string | null }>
        snapshotsById?: Record<string, Snapshot>
      }
      const all = await chrome.storage.local.get(null)
      const key = Object.keys(all).find((name) => name.startsWith("sidepanelChatTabsState:v2:") && name.endsWith(":global"))
      const raw = key ? all[key] : null
      const saved: Saved | null = typeof raw === "string" ? JSON.parse(raw) : (raw as Saved | null)
      if (!saved) return { activeTabId: null, tabs: [] }
      return {
        activeTabId: saved.activeTabId ?? null,
        tabs: (saved.tabs ?? []).map((tab) => {
          const snapshot = saved.snapshotsById?.[tab.id]
          return {
            id: tab.id,
            label: tab.label,
            serverChatId: tab.serverChatId ?? snapshot?.serverChatId ?? null,
            messages: (snapshot?.messages ?? []).map((message) => String(message?.message ?? "")),
          }
        }),
      }
    })
  }
}
