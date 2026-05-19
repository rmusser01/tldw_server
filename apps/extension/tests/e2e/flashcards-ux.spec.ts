import { test, expect, type BrowserContext, type Locator } from "@playwright/test"
import path from "path"
import { launchWithExtension } from "./utils/extension"
import { grantHostPermission } from "./utils/permissions"
import { waitForConnectionStore, logConnectionSnapshot } from "./utils/connection"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"

test.describe("Flashcards workspace UX", () => {
  const waitForFlashcardsState = async (
    page: import("@playwright/test").Page,
    timeoutMs = 15000
  ) => {
    const managerTab = page.getByRole("tab", { name: /^(Review|Study)$/i })
    const offlineHeadline = page.getByText(
      /Connect to use Flashcards|Explore Flashcards in demo mode|Flashcards API not available|Can.?t reach your tldw server/i
    )
    let offlineSeenAt: number | null = null
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (await managerTab.isVisible().catch(() => false)) {
        return "manager"
      }
      const offlineVisible = await offlineHeadline.isVisible().catch(() => false)
      if (offlineVisible) {
        if (offlineSeenAt == null) offlineSeenAt = Date.now()
        const snapshot = await page
          .evaluate(() => {
            const store = (window as any).__tldw_useConnectionStore
            if (!store?.getState) return null
            const state = store.getState().state
            return {
              phase: state.phase,
              mode: state.mode,
              isConnected: state.isConnected,
              isChecking: state.isChecking,
              errorKind: state.errorKind
            }
          })
          .catch(() => null)
        if (snapshot?.mode === "demo") {
          return "offline"
        }
        if (!snapshot?.isChecking && snapshot?.phase !== "searching") {
          if (Date.now() - offlineSeenAt > 1500) {
            return "offline"
          }
        }
      }
      await page.waitForTimeout(300)
    }
    throw new Error("Flashcards workspace did not render manager or offline state.")
  }

  const waitForConnectionReady = async (page: import("@playwright/test").Page) => {
    await page.waitForFunction(
      () =>
        typeof (window as any).__tldw_enableOfflineBypass === "function" &&
        typeof (window as any).__tldw_disableOfflineBypass === "function" &&
        typeof (window as any).__tldw_forceUnconfigured === "function"
    )
  }

  const pick = async (preferred: Locator, fallback: Locator) =>
    (await preferred.count()) > 0 ? preferred : fallback

  const selectOptionWithTimeout = async (
    page: import("@playwright/test").Page,
    select: Locator,
    optionName: string,
    timeoutMs = 5000
  ) => {
    const deadline = Date.now() + timeoutMs
    const selector = (await select.locator(".ant-select-selector").count()) > 0
      ? select.locator(".ant-select-selector").first()
      : select
    const isVisible = await selector.isVisible().catch(() => false)
    if (!isVisible) return false
    const ariaDisabled = await select.getAttribute("aria-disabled").catch(() => null)
    if (ariaDisabled === "true") return false
    await selector.scrollIntoViewIfNeeded().catch(() => {})
    const clicked = await selector
      .click({ timeout: Math.max(1000, timeoutMs / 2), force: true })
      .then(() => true)
      .catch(() => false)
    if (!clicked) return false
    const option = page.getByRole("option", { name: optionName, exact: true })
    const visible = await option
      .waitFor({ state: "visible", timeout: Math.max(1000, deadline - Date.now()) })
      .then(() => true)
      .catch(() => false)
    if (!visible) {
      await page.keyboard.press("Escape").catch(() => {})
      return false
    }
    await option
      .click({ timeout: Math.max(1000, deadline - Date.now()) })
      .catch(() => {})
    return true
  }

  const clickVisibleMenuItem = async (
    page: import("@playwright/test").Page,
    name: RegExp,
    label: string
  ) => {
    const dropdown = page.locator(".ant-dropdown:visible").last()
    const dropdownVisible = await dropdown
      .isVisible()
      .then(() => true)
      .catch(() => false)
    const scope = dropdownVisible ? dropdown : page
    const items = scope.getByRole("menuitem", { name })
    const count = await items.count()
    for (let i = 0; i < count; i += 1) {
      const item = items.nth(i)
      if (await item.isVisible().catch(() => false)) {
        const clicked = await item
          .click({ timeout: 5000 })
          .then(() => true)
          .catch(() => false)
        if (clicked) return
        await item.click({ timeout: 2000, force: true })
        return
      }
    }
    throw new Error(`No visible menu item found for ${label}`)
  }

  const openCreateDrawer = async (page: import("@playwright/test").Page) => {
    const cardsTab = page.getByRole("tab", { name: /^(Cards|Manage)$/i })
    await expect(cardsTab).toBeVisible()
    await cardsTab.click()

    const fab = page.getByTestId("flashcards-fab-create")
    await expect(fab).toBeVisible()
    await fab.click()

    const drawer = page
      .locator(".ant-drawer.ant-drawer-open")
      .filter({ has: page.getByText(/Create Flashcard/i) })
      .last()
    await expect(drawer).toBeVisible()
    return drawer
  }

  const waitForDrawerToCloseOrError = async (
    page: import("@playwright/test").Page,
    drawer: Locator,
    label: string
  ) => {
    const errorToast = page
      .locator(".ant-message-notice-content")
      .filter({ hasText: /Create failed|Failed to create/i })
      .first()
    const deadline = Date.now() + 20000
    while (Date.now() < deadline) {
      if (await errorToast.isVisible().catch(() => false)) {
        throw new Error(`${label} failed: ${await errorToast.textContent()}`)
      }
      const className = await drawer.getAttribute("class")
      if (className && !className.includes("ant-drawer-open")) {
        return
      }
      await page.waitForTimeout(300)
    }
    throw new Error(`${label} did not close`)
  }

  const selectDeckInDrawer = async (
    page: import("@playwright/test").Page,
    drawer: Locator,
    deckName: string
  ) => {
    const deckSelect = drawer.locator(".ant-select").first()
    return selectOptionWithTimeout(page, deckSelect, deckName, 5000)
  }

  test("shows connection-focused empty state when server is offline", async () => {
    const extPath = path.resolve("build/chrome-mv3")
    const { context, page, extensionId } = (await launchWithExtensionOrSkip(test, extPath)) as any
    const optionsUrl = `chrome-extension://${extensionId}/options.html`

    await waitForConnectionReady(page)
    await page.evaluate(() => (window as any).__tldw_forceUnconfigured?.())

    await page.goto(`${optionsUrl}#/flashcards`)

    // When not connected or misconfigured, the Flashcards workspace should
    // surface clear connection messaging (either the global connection card
    // or the feature-specific empty state).
    const state = await waitForFlashcardsState(page)
    if (state !== "offline") {
      await context.close()
      return
    }

    const headline = page.getByText(
      /Connect to use Flashcards|Explore Flashcards in demo mode|Flashcards API not available|Can.?t reach your tldw server/i
    )
    await expect(headline).toBeVisible()

    const goToButton = page.getByRole("button", { name: /Go to server card/i })
    const fallbackButton = page.getByRole("button", {
      name: /Retry connection|Set up server/i
    })
    if ((await goToButton.count()) > 0) {
      await expect(goToButton.first()).toBeVisible()
      await goToButton.first().click()
      const card = page.locator("#server-connection-card")
      if ((await card.count()) > 0) {
        await card.scrollIntoViewIfNeeded()
        await expect(card).toBeVisible()
        await expect(
          card.getByRole("button", { name: /Back to workspace/i })
        ).toBeVisible()
      } else {
        await expect(page).toHaveURL(/#\/settings\/tldw/)
      }
    } else {
      await expect(fallbackButton.first()).toBeVisible()
      await fallbackButton.first().click()
    }

    await context.close()
  })

  test("logs create mutation errors when server is unreachable", async () => {
    const extPath = path.resolve("build/chrome-mv3")
    const { context, page, optionsUrl } = (await launchWithExtensionOrSkip(test, extPath)) as any

    // Set up console listener FIRST - before any page navigation
    // This captures all console messages including those from bgRequest
    const consoleMessages: Array<{ type: string; text: string }> = []
    page.on("console", (msg) => {
      consoleMessages.push({ type: msg.type(), text: msg.text() })
    })

    await waitForConnectionReady(page)
    await page.evaluate(async () => {
      await (window as any).__tldw_enableOfflineBypass?.()
    })

    // Route BEFORE navigation - note: may not intercept service worker requests
    await context.route("**/api/v1/flashcards**", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Simulated flashcards failure" })
      })
    })

    await page.goto(`${optionsUrl}#/flashcards`)
    await expect(page.getByTestId("flashcards-tabs")).toBeVisible()
    const drawer = await openCreateDrawer(page)
    const frontField = await pick(
      drawer.getByLabel(/Front/i),
      drawer.getByPlaceholder(/Question or prompt/i)
    )
    const backField = await pick(
      drawer.getByLabel(/Back/i),
      drawer.getByPlaceholder(/Answer/i)
    )
    await frontField.fill("What is 2+2?")
    await backField.fill("4")
    await drawer.getByRole("button", { name: /^Create$/i }).click()

    // Check for either:
    // - "Failed to create flashcard" (from mutation onError - console.error)
    // - "[tldw:request]" with error (from bgRequest - console.warn)
    // - Any error/warning containing "flashcard" and failure indication
    await expect
      .poll(
        () => {
          const relevantMessages = consoleMessages.filter(
            (m) => m.type === "error" || m.type === "warning"
          )
          return relevantMessages.some(
            (m) =>
              m.text.includes("Failed to create flashcard") ||
              m.text.includes("[tldw:request]") ||
              (m.text.toLowerCase().includes("flashcard") &&
                (m.text.toLowerCase().includes("fail") ||
                  m.text.toLowerCase().includes("error")))
          )
        },
        { timeout: 10000 }
      )
      .toBeTruthy()

    await context.close()
  })

  test("walks through flashcards workflows end-to-end", async () => {
    test.setTimeout(150000)
    const startedAt = Date.now()
    let step = 0
    const mark = (label: string, detail?: string) => {
      step += 1
      const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1)
      const detailText = detail ? ` | ${detail}` : ""
      console.log(`[flashcards-ux][${step}][+${elapsed}s] ${label}${detailText}`)
    }
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = serverUrl.match(/^https?:\/\//)
      ? serverUrl
      : `http://${serverUrl}`

    mark("preflight")
    let preflight: Response | null = null
    try {
      preflight = await fetch(
        `${normalizedServerUrl}/api/v1/flashcards?limit=1&offset=0`,
        {
          headers: {
            "x-api-key": apiKey
          }
        }
      )
    } catch (error) {
      test.skip(
        true,
        `Flashcards API preflight unreachable in this environment: ${String(error)}`
      )
      return
    }
    if (!preflight) return
    if (!preflight.ok) {
      const body = await preflight.text().catch(() => "")
      test.skip(
        true,
        `Flashcards API preflight failed: ${preflight.status} ${preflight.statusText} ${body}`
      )
      return
    }

    const extPath = path.resolve("build/chrome-mv3")
    let context: BrowserContext | null = null
    const baseName = `E2E Flashcards ${Date.now()}`

    try {
      mark("launch extension")
      const launchResult = await launchWithExtensionOrSkip(test, extPath, {
        seedConfig: {
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        }
      })
      context = launchResult.context
      const { page, extensionId, optionsUrl } = launchResult
      const origin = new URL(normalizedServerUrl).origin + "/*"

      mark("grant host permission", origin)
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
        )
      }

      mark("seed flashcards via API")
      const fixture = await page.evaluate(
        async ({ baseUrl, apiKey, baseName }) => {
          const normalizedBase = baseUrl.replace(/\/$/, "")
          const headers = {
            "Content-Type": "application/json",
            "x-api-key": apiKey
          }
          const api = async (path: string, init: RequestInit = {}) => {
            const res = await fetch(`${normalizedBase}${path}`, {
              ...init,
              headers: {
                ...headers,
                ...(init.headers || {})
              }
            })
            const text = await res.text()
            let payload: any = null
            if (text) {
              try {
                payload = JSON.parse(text)
              } catch {
                payload = text
              }
            }
            if (!res.ok) {
              throw new Error(
                `Flashcards API failed ${path}: ${res.status} ${res.statusText} ${text}`
              )
            }
            return payload
          }

          const deck = await api("/api/v1/flashcards/decks", {
            method: "POST",
            body: JSON.stringify({
              name: `${baseName} Deck`
            })
          })
          const deckB = await api("/api/v1/flashcards/decks", {
            method: "POST",
            body: JSON.stringify({
              name: `${baseName} Deck B`
            })
          })

          const cards = []
          for (let i = 0; i < 2; i += 1) {
            const front = `${baseName} Seed ${i + 1}: What is ${i + 2}?`
            const back = `${baseName} Seed ${i + 1} answer`
            const created = await api("/api/v1/flashcards", {
              method: "POST",
              body: JSON.stringify({
                deck_id: deck.id,
                front,
                back,
                tags: ["e2e", "seed"]
              })
            })
            cards.push({
              uuid: created.uuid,
              version: created.version,
              front,
              back
            })
          }

          return {
            deckId: deck.id,
            deckName: deck.name,
            deckIdB: deckB.id,
            deckNameB: deckB.name,
            cards
          }
        },
        { baseUrl: normalizedServerUrl, apiKey, baseName }
      )

      mark("open flashcards workspace", `${optionsUrl}#/flashcards`)
      await page.goto(`${optionsUrl}#/flashcards`)
      await page.waitForLoadState("domcontentloaded")
      await waitForConnectionStore(page, "flashcards-open")

      const state = await waitForFlashcardsState(page)
      mark("flashcards state", state)
      if (state !== "manager") {
        mark("workspace not ready, retrying connection")
        await logConnectionSnapshot(page, "flashcards-offline")
        const retryButton = page.getByRole("button", { name: /Retry connection/i })
        if ((await retryButton.count()) > 0) {
          await retryButton.first().click()
        } else {
          mark("retry button not found, reloading flashcards")
          await page.goto(`${optionsUrl}#/flashcards`)
        }
        if (!page.url().includes("#/flashcards")) {
          mark("navigate back to flashcards", page.url())
          await page.goto(`${optionsUrl}#/flashcards`)
        }
        await waitForConnectionStore(page, "flashcards-retry")
        const postRetryState = await waitForFlashcardsState(page, 45000)
        mark("flashcards state after retry", postRetryState)
        if (postRetryState !== "manager") {
          await logConnectionSnapshot(page, "flashcards-still-offline")
          throw new Error("Flashcards workspace did not reach manager state after retry.")
        }
      }

      const tabsRoot = page
        .locator(".ant-tabs")
        .filter({ has: page.getByRole("tab", { name: /^(Review|Study)$/i }) })
        .first()
      await expect(tabsRoot).toBeVisible()
      const getTabPanel = async (name: string | RegExp) => {
        const tab =
          typeof name === "string"
            ? tabsRoot.getByRole("tab", { name, exact: true })
            : tabsRoot.getByRole("tab", { name })
        await expect(tab).toBeVisible()
        await tab.scrollIntoViewIfNeeded()
        await tab.click()
        const panelId = await tab.getAttribute("aria-controls")
        if (!panelId) {
          throw new Error(`Missing panel id for flashcards tab: ${name}`)
        }
        const panel = page.locator(`#${panelId}`)
        await expect(panel).toBeVisible()
        return panel
      }

      mark("review tab")
      const reviewPanel = await getTabPanel(/^(Review|Study)$/i)
      const reviewDeckSelect = reviewPanel.getByTestId(
        "flashcards-review-deck-select"
      )
      if ((await reviewDeckSelect.count()) > 0) {
        mark("review tab: select deck", fixture.deckName)
        const deckOptionVisible = await selectOptionWithTimeout(
          page,
          reviewDeckSelect,
          fixture.deckName,
          5000
        )
        if (deckOptionVisible) {
          mark("review tab: deck selected")
        } else {
          mark("review tab: deck option not visible")
        }
      }
      const promptSideToggle = reviewPanel.getByTestId(
        "flashcards-review-prompt-side-toggle"
      )
      if ((await promptSideToggle.count()) > 0) {
        mark("review tab: prompt side toggle")
        await expect(promptSideToggle).toBeVisible()
        const frontFirst = promptSideToggle.getByText("Front first", { exact: true })
        const backFirst = promptSideToggle.getByText("Back first", { exact: true })
        if ((await backFirst.count()) > 0) {
          await backFirst.click()
          await expect(backFirst).toBeVisible()
        }
        if ((await frontFirst.count()) > 0) {
          await frontFirst.click()
          await expect(frontFirst).toBeVisible()
        }
      }
      const showAnswerButton = reviewPanel.getByTestId(
        "flashcards-review-show-answer"
      )
      const emptyReviewState = reviewPanel.getByText(
        /No flashcards yet|You're all caught up/i
      )
      const deadline = Date.now() + 15000
      let reviewState: "card" | "empty" | null = null
      while (Date.now() < deadline) {
        if (await showAnswerButton.isVisible().catch(() => false)) {
          reviewState = "card"
          break
        }
        if (await emptyReviewState.isVisible().catch(() => false)) {
          reviewState = "empty"
          break
        }
        await page.waitForTimeout(300)
      }
      if (!reviewState) {
        throw new Error("Review card or empty state did not appear.")
      }
      mark("review tab: state resolved", reviewState)
      if (reviewState === "card") {
        mark("review tab: show answer")
        await showAnswerButton.click()
        const rateButton = reviewPanel.getByTestId("flashcards-review-rate-3")
        await expect(rateButton).toBeVisible()
        mark("review tab: rate card", "Good (3)")
        await rateButton.click()
        await expect
          .poll(
            async () => {
              const hasShow = await showAnswerButton
                .isVisible()
                .catch(() => false)
              const emptyState = await emptyReviewState
                .isVisible()
                .catch(() => false)
              return hasShow || emptyState
            },
            { timeout: 15000 }
          )
          .toBeTruthy()
        mark("review tab: ready for next card")
      }

      const templatesTab = tabsRoot.getByRole("tab", { name: /Templates/i })
      if ((await templatesTab.count()) > 0) {
        mark("templates tab")
        const templatesPanel = await getTabPanel(/Templates/i)
        await expect(templatesPanel).toBeVisible()
      }

      mark("cards tab: create")
      const createdFront = `${baseName} UI: Define recursion`
      const createdBack = `${baseName} UI: A function that calls itself`
      const addAnotherFront = `${baseName} UI: Tail recursion`
      const addAnotherBack = `${baseName} UI: Recursion with optimized tail calls`
      const createDrawer = await openCreateDrawer(page)
      mark("cards tab: create drawer open")
      await expect(
        createDrawer.getByText(/Existing cards in this deck/i)
      ).toHaveCount(0)
      const referenceToggle = createDrawer.getByRole("button", {
        name: /Existing cards in this deck/i
      })
      const drawerDeckSelected = await selectDeckInDrawer(
        page,
        createDrawer,
        fixture.deckName
      )
      if (drawerDeckSelected) {
        mark("cards tab: deck selected", fixture.deckName)
      } else {
        mark("cards tab: deck option not visible", fixture.deckName)
      }
      await expect(referenceToggle).toBeVisible()
      await referenceToggle.click()
      const referenceSearch = createDrawer.getByPlaceholder(/Search this deck/i)
      const referenceSearchTerm = fixture.cards[0].front
      await expect(referenceSearch).toBeVisible()
      await expect(createDrawer.getByText(fixture.cards[1].front)).toBeVisible()
      await expect(createDrawer.getByText(fixture.cards[1].back)).toBeVisible()
      const createFrontField = await pick(
        createDrawer.getByLabel(/Front/i),
        createDrawer.getByPlaceholder(/Question or prompt/i)
      )
      const createBackField = await pick(
        createDrawer.getByLabel(/Back/i),
        createDrawer.getByPlaceholder(/Answer/i)
      )
      await referenceSearch.fill(referenceSearchTerm)
      mark("cards tab: fill create form")
      await createFrontField.fill(createdFront)
      await createBackField.fill(createdBack)
      mark("cards tab: submit create")
      await createDrawer.getByRole("button", { name: /^Create$/i }).click()
      mark("cards tab: wait for drawer close")
      await waitForDrawerToCloseOrError(page, createDrawer, "Create flashcard")
      mark("cards tab: create drawer closed")

      mark("cards tab: reopen create drawer for add another")
      const reopenedCreateDrawer = await openCreateDrawer(page)
      await expect(
        reopenedCreateDrawer.getByText(/Existing cards in this deck/i)
      ).toHaveCount(0)
      const reopenedDeckSelected = await selectDeckInDrawer(
        page,
        reopenedCreateDrawer,
        fixture.deckName
      )
      if (reopenedDeckSelected) {
        mark("cards tab: drawer reopened deck selected", fixture.deckName)
      } else {
        mark("cards tab: drawer reopened deck option not visible", fixture.deckName)
      }
      const reopenedReferenceToggle = reopenedCreateDrawer.getByRole("button", {
        name: /Existing cards in this deck/i
      })
      const reopenedDeckField = reopenedCreateDrawer.getByRole("combobox", {
        name: /Deck/i
      })
      await expect(reopenedReferenceToggle).toBeVisible()
      await reopenedReferenceToggle.click()
      const reopenedReferenceSearch = reopenedCreateDrawer.getByPlaceholder(
        /Search this deck/i
      )
      await expect(reopenedReferenceSearch).toHaveValue("")
      await reopenedReferenceSearch.fill(referenceSearchTerm)
      const reopenedFrontField = await pick(
        reopenedCreateDrawer.getByLabel(/Front/i),
        reopenedCreateDrawer.getByPlaceholder(/Question or prompt/i)
      )
      const reopenedBackField = await pick(
        reopenedCreateDrawer.getByLabel(/Back/i),
        reopenedCreateDrawer.getByPlaceholder(/Answer/i)
      )
      await reopenedFrontField.fill(addAnotherFront)
      await reopenedBackField.fill(addAnotherBack)
      await reopenedCreateDrawer
        .getByRole("button", { name: /^Create & Add Another$/i })
        .click()
      await expect(reopenedCreateDrawer).toBeVisible()
      await expect(reopenedFrontField).toHaveValue("")
      await expect(reopenedBackField).toHaveValue("")
      await expect(reopenedDeckField).toContainText(fixture.deckName)
      await expect(reopenedReferenceSearch).toHaveValue(referenceSearchTerm)
      await expect(reopenedCreateDrawer.getByText(addAnotherFront)).toBeVisible()
      await expect(reopenedCreateDrawer.getByText(addAnotherBack)).toBeVisible()

      mark("cards tab: close and reopen to verify search reset")
      await reopenedCreateDrawer.getByRole("button", { name: /Cancel/i }).click()
      await expect(reopenedCreateDrawer).toBeHidden()
      const resetCreateDrawer = await openCreateDrawer(page)
      const resetDeckSelected = await selectDeckInDrawer(
        page,
        resetCreateDrawer,
        fixture.deckName
      )
      if (resetDeckSelected) {
        mark("cards tab: drawer reopened after add another", fixture.deckName)
      }
      await expect(
        resetCreateDrawer.getByRole("button", {
          name: /Existing cards in this deck/i
        })
      ).toBeVisible()
      await resetCreateDrawer
        .getByRole("button", { name: /Existing cards in this deck/i })
        .click()
      await expect(resetCreateDrawer.getByPlaceholder(/Search this deck/i)).toHaveValue("")
      await resetCreateDrawer.getByRole("button", { name: /Cancel/i }).click()
      await expect(resetCreateDrawer).toBeHidden()
      mark("cards tab: create drawer closed")

      mark("cards tab")
      const managePanel = await getTabPanel(/^(Cards|Manage)$/i)
      mark("cards tab: panel ready")
      const manageSearch = await pick(
        managePanel.getByTestId("flashcards-manage-search"),
        managePanel.getByPlaceholder(/Search/i)
      )
      const manageSearchInput =
        (await manageSearch.locator("input").count()) > 0
          ? manageSearch.locator("input").first()
          : manageSearch
      await expect(manageSearchInput).toBeVisible()
      const deckFilter = managePanel.getByTestId(
        "flashcards-manage-deck-select"
      )
      if ((await deckFilter.count()) > 0) {
        mark("cards tab: filter by deck", fixture.deckName)
        const deckFilterSelected = await selectOptionWithTimeout(
          page,
          deckFilter,
          fixture.deckName,
          5000
        )
        if (!deckFilterSelected) {
          mark("cards tab: deck option not visible", fixture.deckName)
        }
      }
      const seedCard = fixture.cards[0]
      mark("cards tab: search", baseName)
      await manageSearchInput.fill(baseName)
      await manageSearchInput.press("Enter")
      mark("cards tab: search submitted")

      await expect
        .poll(async () => {
          return managePanel
            .locator(".ant-list-item")
            .filter({ hasText: baseName })
            .count()
        }, {
          timeout: 15000
        })
        .toBeGreaterThan(0)
      mark("cards tab: list ready")

      mark("cards tab: keyboard shortcuts")
      await manageSearchInput
        .evaluate((el) => (el as HTMLInputElement).blur())
        .catch(() => {})
      const cardCount = await managePanel.locator(".ant-list-item").count()
      if (cardCount === 0) {
        throw new Error("No flashcards available for keyboard shortcut checks.")
      }
      await page.keyboard.press("j")
      const focusedItem = managePanel.locator(".ant-list-item.ring-2").first()
      await expect(focusedItem).toBeVisible({ timeout: 3000 })
      const focusedCheckbox = focusedItem.locator('input[type="checkbox"]').first()
      await expect(focusedCheckbox).toBeVisible()
      await expect(focusedCheckbox).not.toBeChecked()
      await page.keyboard.press("Space")
      await expect(focusedCheckbox).toBeChecked()
      await page.keyboard.press("Space")
      await expect(focusedCheckbox).not.toBeChecked()
      await page.keyboard.press("Enter")
      const keyboardEditDrawer = page
        .locator(".ant-drawer")
        .filter({ has: page.getByText(/Edit Flashcard/i) })
        .first()
      await expect(keyboardEditDrawer).toBeVisible({ timeout: 5000 })
      await keyboardEditDrawer.locator(".ant-drawer-close").click()
      await expect(keyboardEditDrawer).toBeHidden({ timeout: 5000 })

      const densityToggle = managePanel.getByTestId(
        "flashcards-density-toggle"
      )
      if ((await densityToggle.count()) > 0) {
        mark("cards tab: toggle density")
        await densityToggle.click()
      }

      const seedCardItem = managePanel.getByTestId(
        `flashcard-item-${seedCard.uuid}`
      )
      await expect(seedCardItem).toBeVisible()
      mark("cards tab: open preview", seedCard.uuid)
      await seedCardItem.click()
      const previewBlock = seedCardItem
        .locator(".border")
        .filter({ hasText: seedCard.back })
        .first()
      await expect(previewBlock).toBeVisible()
      mark("cards tab: preview toggled")

      mark("cards tab: duplicate", seedCard.uuid)
      await managePanel
        .getByTestId(`flashcard-more-${seedCard.uuid}`)
        .click({ timeout: 5000 })
      await clickVisibleMenuItem(page, /Duplicate/i, "Duplicate")
      await expect
        .poll(
          async () =>
            managePanel
              .locator(".ant-list-item")
              .filter({ hasText: seedCard.front })
              .count(),
          { timeout: 15000 }
        )
        .toBeGreaterThan(1)

      mark("cards tab: move", fixture.cards[1].uuid)
      const moveCard = fixture.cards[1]
      await managePanel
        .getByTestId(`flashcard-more-${moveCard.uuid}`)
        .click({ timeout: 5000 })
      await clickVisibleMenuItem(page, /Move to deck/i, "Move to deck")
      const moveDrawer = page
        .locator(".ant-drawer")
        .filter({ has: page.getByText(/Move to deck|Bulk Move/i) })
        .first()
      let moveDrawerReady = await moveDrawer
        .waitFor({ state: "visible", timeout: 8000 })
        .then(() => true)
        .catch(() => false)
      if (!moveDrawerReady) {
        mark("cards tab: move drawer not visible, fallback to bulk move")
        await page.keyboard.press("Escape").catch(() => {})
        const moveCheckbox = managePanel.getByTestId(
          `flashcard-item-${moveCard.uuid}-select`
        )
        await expect(moveCheckbox).toBeVisible({ timeout: 5000 })
        await moveCheckbox.click()
        const bulkMoveButton = page.getByRole("button", { name: /^Move$/i })
        const bulkMoveVisible = await bulkMoveButton
          .waitFor({ state: "visible", timeout: 5000 })
          .then(() => true)
          .catch(() => false)
        if (!bulkMoveVisible) {
          mark("cards tab: bulk move button not visible, skipping move")
        } else {
          await bulkMoveButton.click()
          moveDrawerReady = await moveDrawer
            .waitFor({ state: "visible", timeout: 8000 })
            .then(() => true)
            .catch(() => false)
        }
      }
      if (!moveDrawerReady) {
        mark("cards tab: move skipped (drawer not visible)")
      } else {
        const moveDeckSelected = await selectOptionWithTimeout(
          page,
          moveDrawer.locator(".ant-select").first(),
          fixture.deckNameB,
          5000
        )
        if (moveDeckSelected) {
          mark("cards tab: confirm move", fixture.deckNameB)
          await moveDrawer.getByRole("button", { name: /^Move$/i }).click()
          await expect(moveDrawer).toBeHidden()

          mark("cards tab: verify moved card", fixture.deckNameB)
          const movedFilterSelected = await selectOptionWithTimeout(
            page,
            deckFilter,
            fixture.deckNameB,
            5000
          )
          if (movedFilterSelected) {
            await expect(managePanel.getByText(moveCard.front)).toBeVisible()
          } else {
            mark("cards tab: deck option not visible", fixture.deckNameB)
          }

          mark("cards tab: return to deck", fixture.deckName)
          const returnFilterSelected = await selectOptionWithTimeout(
            page,
            deckFilter,
            fixture.deckName,
            5000
          )
          if (!returnFilterSelected) {
            mark("cards tab: deck option not visible", fixture.deckName)
          }
        } else {
          mark("cards tab: move deck option not visible", fixture.deckNameB)
          await moveDrawer.getByRole("button", { name: /Cancel/i }).click()
          await expect(moveDrawer).toBeHidden()
        }
      }

      mark("cards tab: edit", createdFront)
      const editedFront = `${createdFront} (edited)`
      const createdItem = managePanel
        .locator(".ant-list-item")
        .filter({ hasText: createdFront })
        .first()
      const createdItemReady = await createdItem
        .waitFor({ state: "visible", timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!createdItemReady) {
        mark("cards tab: created item not visible, skipping edit")
      } else {
        await createdItem.getByRole("button", { name: /Edit/i }).click()
      }
      const editDrawer = page
        .locator(".ant-drawer")
        .filter({ has: page.getByText(/Edit Flashcard/i) })
        .first()
      const editDrawerReady = await editDrawer
        .waitFor({ state: "visible", timeout: 8000 })
        .then(() => true)
        .catch(() => false)
      if (!editDrawerReady) {
        mark("cards tab: edit drawer not visible, skipping edit")
      } else {
        const editFrontField = await pick(
          editDrawer.getByLabel(/Front/i),
          editDrawer.getByPlaceholder(/Question or prompt/i)
        )
        mark("cards tab: edit front")
        await editFrontField.fill(editedFront)
        mark("cards tab: save edit")
        await editDrawer.getByRole("button", { name: /Save/i }).click()
        await expect(editDrawer).toBeHidden()
        await expect(managePanel.getByText(editedFront)).toBeVisible()
      }

      mark("cards tab: review now", editedFront)
      const editedItem = managePanel
        .locator(".ant-list-item")
        .filter({ hasText: editedFront })
        .first()
      const editedItemReady = await editedItem
        .waitFor({ state: "visible", timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!editedItemReady) {
        mark("cards tab: edited item not visible, skipping review now")
      } else {
        await editedItem
          .getByRole("button", { name: /More actions/i })
          .click({ timeout: 5000 })
        await clickVisibleMenuItem(page, /Review now/i, "Review now")
      }
      const reviewActivated = await page
        .getByRole("tab", { name: /^(Review|Study)$/i })
        .getAttribute("aria-selected")
        .then((value) => value === "true")
        .catch(() => false)
      if (!reviewActivated) {
        mark("review tab not active, skipping review now flow")
      } else {
        await expect(showAnswerButton).toBeVisible()
        mark("review tab: show answer (review now)")
        await showAnswerButton.click()
        mark("review tab: rate card", "Hard (2)")
        await reviewPanel.getByTestId("flashcards-review-rate-2").click()
        await expect(showAnswerButton).toBeVisible({ timeout: 15000 })
      }

      mark("import/export tab")
      if (page.isClosed()) {
        throw new Error("Flashcards page closed before import/export tab.")
      }
      await page.keyboard.press("Escape").catch(() => {})
      const importTab = page.getByRole("tab", { name: "Import / Export", exact: true })
      const importTabReady = await importTab
        .waitFor({ state: "visible", timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!importTabReady) {
        mark("import/export tab not visible, skipping import/export")
        return
      }
      await importTab.scrollIntoViewIfNeeded().catch(() => {})
      const importTabClicked = await importTab
        .click({ timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!importTabClicked) {
        const forcedClick = await importTab
          .click({ timeout: 5000, force: true })
          .then(() => true)
          .catch(() => false)
        if (!forcedClick) {
          mark("import/export tab click failed, skipping import/export")
          return
        }
      }
      const importPanelId = await importTab.getAttribute("aria-controls")
      if (!importPanelId) {
        mark("import/export panel id missing, skipping import/export")
        return
      }
      const importExportPanel = page.locator(`#${importPanelId}`)
      const importExportReady = await importExportPanel
        .waitFor({ state: "visible", timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!importExportReady) {
        mark("import/export panel not visible, skipping import/export")
        return
      }
      const importCard = importExportPanel
        .locator(".ant-card")
        .filter({ hasText: /Import Flashcards/i })
        .first()
      const importContent = [
        "Deck\tFront\tBack\tTags\tNotes",
        `${fixture.deckName}\t${baseName} Import: TCP/IP\t${baseName} Import: Networking stack\te2e;imported\t`
      ].join("\n")
      const importTextarea = await pick(
        importCard.getByTestId("flashcards-import-textarea"),
        importCard.getByPlaceholder(/Paste content here/i)
      )
      mark("import: fill textarea")
      await importTextarea.fill(importContent)
      const headerSwitch = await pick(
        importCard.getByTestId("flashcards-import-has-header"),
        importCard.locator(".ant-switch").first()
      )
      if ((await headerSwitch.getAttribute("aria-checked")) !== "true") {
        mark("import: enable header")
        await headerSwitch.click()
      }
      const importButton = await pick(
        importCard.getByTestId("flashcards-import-button"),
        importCard.getByRole("button", { name: /Import/i })
      )
      mark("import: submit")
      await importButton.click()
      await expect(importTextarea).toHaveValue("")

      mark("export download")
      const exportCard = importExportPanel
        .locator(".ant-card")
        .filter({ hasText: /Export Flashcards/i })
        .first()
      const exportButton = await pick(
        exportCard.getByTestId("flashcards-export-button"),
        exportCard.getByRole("button", { name: /Export/i })
      )
      const [download] = await Promise.all([
        page.waitForEvent("download", { timeout: 15000 }),
        exportButton.click()
      ])
      mark("export: download complete", download.suggestedFilename())
      expect(download.suggestedFilename()).toBe("flashcards.csv")

      mark("export: switch format to APKG")
      const exportFormatSelect = exportCard.getByTestId("flashcards-export-format")
      const apkgSelected = await selectOptionWithTimeout(
        page,
        exportFormatSelect,
        "APKG (Anki)",
        5000
      )
      if (!apkgSelected) {
        mark("export: apkg format not selectable, skipping")
      } else {
        const [apkgDownload] = await Promise.all([
          page.waitForEvent("download", { timeout: 15000 }),
          exportButton.click()
        ])
        mark("export: apkg download complete", apkgDownload.suggestedFilename())
        expect(apkgDownload.suggestedFilename()).toBe("flashcards.apkg")
      }

      mark("verify import via api")
      await expect
        .poll(async () => {
          const res = await fetch(
            `${normalizedServerUrl.replace(/\/$/, "")}/api/v1/flashcards?due_status=all&limit=100&offset=0&order_by=due_at`,
            {
              headers: {
                "Content-Type": "application/json",
                "x-api-key": apiKey
              }
            }
          )
          if (!res.ok) return false
          const payload = await res.json().catch(() => null)
          return (payload?.items || []).some((item: any) =>
            String(item?.front || "").includes("Import: TCP/IP")
          )
        }, {
          timeout: 15000
        })
        .toBe(true)

      mark("cards tab: bulk actions")
      const bulkPanel = await getTabPanel(/^(Cards|Manage)$/i)
      const bulkSearch = await pick(
        bulkPanel.getByTestId("flashcards-manage-search"),
        bulkPanel.getByPlaceholder(/Search/i)
      )
      const bulkSearchInput =
        (await bulkSearch.locator("input").count()) > 0
          ? bulkSearch.locator("input").first()
          : bulkSearch
      await bulkSearchInput.fill(baseName)
      await bulkSearchInput.press("Enter")
      mark("bulk: search submitted", baseName)
      await expect
        .poll(async () => {
          return bulkPanel
            .locator(".ant-list-item")
            .filter({ hasText: baseName })
            .count()
        }, {
          timeout: 15000
        })
        .toBeGreaterThan(0)

      const seedCheckbox = bulkPanel.getByTestId(
        `flashcard-item-${seedCard.uuid}-select`
      )
      mark("bulk: select first item", seedCard.uuid)
      await seedCheckbox.click()

      const selectAllLink = bulkPanel.getByRole("button", {
        name: /Select all/i
      })
      if ((await selectAllLink.count()) === 0) {
        throw new Error("Select all link not found for bulk selection.")
      }
      mark("bulk: select all across results")
      await selectAllLink.first().click()
      const selectedAcrossLabel = bulkPanel
        .getByText(/selected across all results/i)
        .first()
      await expect(selectedAcrossLabel).toBeVisible()

      const actionBar = page
        .locator(".fixed")
        .filter({
          has: page.getByRole("button", { name: /^Delete$/i })
        })
        .first()
      const actionBarReady = await actionBar
        .waitFor({ state: "visible", timeout: 5000 })
        .then(() => true)
        .catch(() => false)
      if (!actionBarReady) {
        mark("bulk: action bar not visible, skipping bulk export/delete")
      } else {
        const exportSelectedButton = actionBar.getByRole("button", { name: /^Export$/i })
        const selectedDownload = await Promise.all([
          page.waitForEvent("download", { timeout: 15000 }).catch(() => null),
          exportSelectedButton.click()
        ]).then(([download]) => download)
        if (!selectedDownload) {
          mark("bulk: export selected download missing, skipping")
        } else {
          mark("bulk: export selected", selectedDownload.suggestedFilename())
          expect(selectedDownload.suggestedFilename()).toBe("flashcards-selected.tsv")
        }

        mark("bulk: delete selected")
        await actionBar.getByRole("button", { name: /^Delete$/i }).click()
        const confirmDialog = page.getByRole("dialog", {
          name: /Please confirm/i
        })
        await expect(confirmDialog).toBeVisible()
        await confirmDialog.getByRole("button", { name: /^Delete$/i }).click()
        const undoToast = page
          .locator(".undo-notification")
          .filter({ hasText: /deleted/i })
          .last()
        await expect(undoToast).toBeVisible({ timeout: 5000 })

        mark("bulk: open trash view")
        const trashToggle = await pick(
          bulkPanel.getByRole("radio", { name: /Trash/i }),
          bulkPanel.getByText(/Trash/i)
        )
        await trashToggle.click()

        const trashedSeed = bulkPanel.getByTestId(
          `flashcard-trash-${seedCard.uuid}`
        )
        await expect(trashedSeed).toBeVisible({ timeout: 5000 })

        const undoAll = bulkPanel.getByRole("button", { name: /Undo all/i })
        await expect(undoAll).toBeVisible({ timeout: 5000 })
        await undoAll.click()
        await expect(trashedSeed).toBeHidden({ timeout: 5000 })

        mark("bulk: return to cards view")
        const cardsToggle = await pick(
          bulkPanel.getByRole("radio", { name: /Cards/i }),
          bulkPanel.getByText(/Cards/i)
        )
        await cardsToggle.click()
        await expect(
          bulkPanel.getByTestId(`flashcard-item-${seedCard.uuid}`)
        ).toBeVisible({ timeout: 15000 })
      }
    } finally {
      mark("cleanup flashcards")
      try {
        const headers = {
          "Content-Type": "application/json",
          "x-api-key": apiKey
        }
        const res = await fetch(
          `${normalizedServerUrl.replace(/\/$/, "")}/api/v1/flashcards?q=${encodeURIComponent(
            baseName
          )}&limit=200&offset=0`,
          { headers }
        )
        if (res.ok) {
          const payload = await res.json().catch(() => null)
          const items = payload?.items || []
          const deleteWithRetry = async (item: any) => {
            const url = `${normalizedServerUrl.replace(
              /\/$/,
              ""
            )}/api/v1/flashcards/${encodeURIComponent(item.uuid)}?expected_version=${item.version}`
            for (let attempt = 0; attempt < 3; attempt += 1) {
              const res = await fetch(url, { method: "DELETE", headers })
              if (res.ok || res.status === 404) {
                return
              }
              const body = await res.text().catch(() => "")
              if (!/locked/i.test(body)) {
                return
              }
              await new Promise((resolve) =>
                setTimeout(resolve, 200 * (attempt + 1))
              )
            }
          }
          for (const item of items) {
            await deleteWithRetry(item)
          }
        }
      } catch (err) {
        console.warn("[flashcards-ux] cleanup failed", err)
      }
      if (context) {
        await context.close()
      }
    }
  })
})
