import fs from "node:fs"
import { expect, test } from "@playwright/test"
import { grantHostPermission } from "./utils/permissions"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"
import {
  forceConnected,
  setSelectedModel,
  waitForConnectionStore
} from "./utils/connection"
import { collectGreetings } from "@tldw/ui/utils/character-greetings"

const DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY =
  "preferences.chat.default_character_id"

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value : `http://${value}`

const fetchWithTimeout = async (
  url: string,
  init: RequestInit | undefined,
  timeoutMs = 15000
) => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal
    })
  } finally {
    clearTimeout(timeout)
  }
}

const requestJson = async (
  serverUrl: string,
  apiKey: string,
  path: string,
  init?: RequestInit
) => {
  const response = await fetchWithTimeout(`${serverUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      ...(init?.headers || {})
    }
  })
  const text = await response.text().catch(() => "")
  if (!response.ok) {
    throw new Error(
      `Request failed ${response.status} ${response.statusText}: ${text}`
    )
  }
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

const parseListPayload = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload
  if (!payload || typeof payload !== "object") return []
  return (
    payload.items ||
    payload.results ||
    payload.data ||
    payload.chats ||
    payload.characters ||
    []
  )
}

const getFirstModelId = (payload: any): string | null => {
  const modelsList = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.models)
      ? payload.models
      : []
  const candidate =
    modelsList.find((m: any) => m?.id || m?.model || m?.name) || null
  const id = candidate?.id || candidate?.model || candidate?.name
  return id ? String(id) : null
}

const closeBlockingDrawerIfVisible = async (page: any) => {
  const drawerMask = page.locator(".ant-drawer-mask").first()
  const maskVisible = await drawerMask
    .isVisible({ timeout: 1000 })
    .then(() => true)
    .catch(() => false)
  if (!maskVisible) return

  const closeCandidates = [
    page.locator(".ant-drawer-close").first(),
    page.getByRole("button", { name: /close|x/i }).first()
  ]

  for (const candidate of closeCandidates) {
    const visible = await candidate
      .isVisible({ timeout: 750 })
      .then(() => true)
      .catch(() => false)
    if (!visible) continue
    await candidate.click({ timeout: 3000 }).catch(() => {})
    break
  }

  await page.keyboard.press("Escape").catch(() => {})
  await drawerMask.waitFor({ state: "hidden", timeout: 10000 }).catch(() => {})
}

const ensureComposerActionBarVisible = async (page: any) => {
  await closeBlockingDrawerIfVisible(page)
  const composerInput = page.locator("#textarea-message")
  await expect(composerInput).toBeVisible({ timeout: 15000 })
  await composerInput.click({ timeout: 5000 })
  await composerInput.hover().catch(() => {})
  await page.waitForTimeout(100)
}

const normalizeText = (value: string) => value.replace(/\s+/g, " ").trim()

const findVisibleButton = async (locator: any) => {
  const count = Math.min(await locator.count(), 10)
  for (let index = 0; index < count; index += 1) {
    const candidate = locator.nth(index)
    const visible = await candidate
      .isVisible({ timeout: 500 })
      .then(() => true)
      .catch(() => false)
    if (visible) return candidate
  }
  return null
}

const findVisibleSearchInput = async (page: any) => {
  const locator = page.getByPlaceholder(/Search characters(?: by name)?/i)
  return findVisibleButton(locator)
}

const findCharacterMenuItem = async (page: any, characterName: string) => {
  const target = normalizeText(characterName)
  const candidateLists = [
    page.locator("[role='menuitem']"),
    page.locator(".ant-dropdown-menu-item")
  ]

  for (const candidateList of candidateLists) {
    const count = Math.min(await candidateList.count(), 40)
    let partialMatch: any = null
    for (let index = 0; index < count; index += 1) {
      const candidate = candidateList.nth(index)
      const visible = await candidate
        .isVisible({ timeout: 500 })
        .then(() => true)
        .catch(() => false)
      if (!visible) continue

      const label = normalizeText(await candidate.innerText().catch(() => ""))
      if (!label) continue
      if (label === target) return candidate
      if (!partialMatch && label.includes(target)) {
        partialMatch = candidate
      }
    }

    if (partialMatch) return partialMatch
  }

  return null
}

const collectVisibleCharacterMenuLabels = async (page: any) =>
  page.evaluate(() =>
    Array.from(
      document.querySelectorAll("[role='menuitem'], .ant-dropdown-menu-item")
    )
      .map((node) => {
        const element = node as HTMLElement
        const style = window.getComputedStyle(element)
        const rect = element.getBoundingClientRect()
        const visible =
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || "1") > 0 &&
          rect.width > 0 &&
          rect.height > 0
        return {
          visible,
          text: (element.innerText || element.textContent || "").trim()
        }
      })
      .filter((entry) => entry.visible)
      .map((entry) => entry.text)
      .filter(Boolean)
      .slice(0, 40)
  )

const openCharacterSelector = async (page: any) => {
  await ensureComposerActionBarVisible(page)

  const candidates = [
    page.getByRole("button", { name: /select( a)? character/i }),
    page.getByRole("button", { name: /clear character/i }),
    page.getByTestId("chat-character-select")
  ]

  for (const locator of candidates) {
    const visibleButton = await findVisibleButton(locator)
    if (!visibleButton) continue
    await visibleButton.click({ timeout: 5000 })
    return
  }

  const visibleButtons = await page.evaluate(() =>
    Array.from(document.querySelectorAll("button"))
      .map((button) => {
        const style = window.getComputedStyle(button)
        const rect = button.getBoundingClientRect()
        const visible =
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity || "1") > 0 &&
          rect.width > 0 &&
          rect.height > 0
        return {
          ariaLabel: button.getAttribute("aria-label") || null,
          title: button.getAttribute("title") || null,
          text: (button.textContent || "").trim().slice(0, 80),
          visible
        }
      })
      .filter((entry) => entry.visible)
      .slice(0, 50)
  )

  throw new Error(
    `Character trigger button was not visible. Visible buttons: ${JSON.stringify(visibleButtons)}`
  )
}

const dismissTutorialPromptIfVisible = async (page: any) => {
  const dismissButton = await findVisibleButton(
    page.getByRole("button", { name: /Not now/i })
  )
  if (dismissButton) {
    await dismissButton.click({ timeout: 5000 }).catch(() => {})
    await page.waitForTimeout(150)
  }
}

const createEphemeralCharacters = async (
  serverUrl: string,
  apiKey: string,
  count = 2
) => {
  const results: { id: string; name: string; greetings: string[] }[] = []
  const createdIds: string[] = []
  const batchSeed = Date.now()

  for (let index = 0; index < count; index += 1) {
    const suffix = `${batchSeed}-${index + 1}-${Math.floor(Math.random() * 10000)}`
    const name = `E2E Character ${suffix}`
    const created = await createCharacter(serverUrl, apiKey, name)
    createdIds.push(created.id)
    results.push(created)
  }

  return { characters: results, createdIds }
}

const confirmCharacterSwitchIfNeeded = async (
  page: any,
  { expectModal = false }: { expectModal?: boolean } = {}
) => {
  const modal = page.locator(".ant-modal")
  const confirmButton = modal.locator(".ant-btn-dangerous").first()
  const isVisible = await confirmButton
    .isVisible({ timeout: expectModal ? 15000 : 2000 })
    .then(() => true)
    .catch(() => false)
  if (!isVisible) {
    if (expectModal) {
      throw new Error("Expected switch character modal, but it was not shown.")
    }
    return
  }
  await confirmButton.click({ timeout: 5000 }).catch(() => {})
  await expect(modal).toBeHidden({ timeout: 15000 })
}

// Each test launches a fresh extension profile for one account. Require exactly
// one owned assistant record so another account can never satisfy this probe.
const readSelectedCharacterFromStorage = async (page: any) =>
  page.evaluate(async () => {
    const area = (window as any).chrome?.storage?.local
    if (!area?.get) return null
    const items = await new Promise<Record<string, unknown>>(resolve => area.get(null, resolve))
    const records = Object.entries(items).filter(([key]) => key.startsWith("selectedAssistant:owner:"))
    if (records.length !== 1) return null
    const [key, raw] = records[0]
    let record: any = raw
    if (typeof raw === "string") {
      try { record = JSON.parse(raw) } catch { return null }
    }
    if (!record?.ownerKey || key !== `selectedAssistant:owner:${record.ownerKey}`) return null
    return record.selection?.kind === "character" ? record.selection : null
  })

const waitForGreeting = async (page: any, characterName: string) => {
  const greetingMessage = page
    .locator(
      "[data-message-type='character:greeting'], [data-message-type='greeting']"
    )
    .first()
  const visible = await greetingMessage
    .waitFor({ state: "visible", timeout: 15000 })
    .then(() => true)
    .catch(() => false)
  if (!visible) {
    const debug = await page.evaluate(() => {
      const w = window as any
      const store = w.__tldw_useStoreMessageOption?.getState?.()
      const messages = store?.messages || []
      return {
        serverChatId: store?.serverChatId ?? null,
        serverChatCharacterId: store?.serverChatCharacterId ?? null,
        historyId: store?.historyId ?? null,
        selectedCharacter: store?.selectedCharacter ?? null,
        messages: messages.map((msg: any) => ({
          isBot: msg?.isBot,
          name: msg?.name,
          messageType: msg?.messageType ?? msg?.message_type,
          message: msg?.message
        }))
      }
    })
    throw new Error(
      `Character greeting not visible. Debug: ${JSON.stringify(debug)}`
    )
  }
  const hasName = await greetingMessage
    .getByText(new RegExp(characterName, "i"))
    .first()
    .isVisible()
    .catch(() => false)
  if (!hasName) {
    const text = await greetingMessage.textContent()
    throw new Error(
      `Greeting did not mention ${characterName}. Content: ${text || ""}`
    )
  }
  return greetingMessage
}

const dismissWorkflowHubIfVisible = async (page: any) => {
  const startChatButton = await findVisibleButton(
    page.getByRole("button", { name: /Start chatting/i })
  )
  if (startChatButton) {
    await startChatButton.click({ timeout: 5000 }).catch(() => {})
    await page.waitForTimeout(150)
  }

  const legacyHubHeading = page.getByText(/What would you like to do\?/i).first()
  const legacyHubDialog = page
    .locator("[role='dialog']")
    .filter({ has: legacyHubHeading })
    .first()
  const legacyVisible = await legacyHubDialog
    .isVisible({ timeout: 1500 })
    .then(() => true)
    .catch(() => false)
  if (legacyVisible) {
    const closeButton = legacyHubDialog
      .getByRole("button", { name: /close|x/i })
      .first()
    const closeVisible = await closeButton
      .isVisible({ timeout: 1000 })
      .then(() => true)
      .catch(() => false)
    if (closeVisible) {
      await closeButton.click().catch(() => {})
    } else {
      await page.keyboard.press("Escape").catch(() => {})
    }
    await legacyHubDialog.waitFor({ state: "hidden", timeout: 10000 }).catch(() => {})
  }

  await dismissTutorialPromptIfVisible(page)
}

const enterCharacterChatModeIfAvailable = async (page: any) => {
  const candidates = [
    page.getByRole("button", { name: /^Character chat$/i }),
    page.getByRole("button", { name: /new character chat/i }),
    page.getByRole("button", { name: /^Character$/i })
  ]

  for (const locator of candidates) {
    const visibleButton = await findVisibleButton(locator)
    if (!visibleButton) continue
    await visibleButton.click({ timeout: 5000 }).catch(() => {})
    await closeBlockingDrawerIfVisible(page)
    await page.waitForTimeout(150)
    return
  }
}

const createCharacter = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const greeting = [
    `Hello from ${name}!`,
    "",
    "```",
    `${name} ready`,
    "```",
    "",
    "- /start",
    "- /help"
  ].join("\n")
  const alternateGreetings = [
    [
      `Hello from ${name}!`,
      "",
      "```",
      `${name} alternate 1`,
      "```",
      "",
      "- /alt1"
    ].join("\n"),
    [
      `Hello from ${name}!`,
      "",
      "```",
      `${name} alternate 2`,
      "```",
      "",
      "- /alt2"
    ].join("\n")
  ]
  const created = await requestJson(serverUrl, apiKey, "/api/v1/characters/", {
    method: "POST",
    body: JSON.stringify({
      name,
      greeting,
      first_message: greeting,
      alternate_greetings: alternateGreetings
    })
  }).catch(() =>
    requestJson(serverUrl, apiKey, "/api/v1/characters", {
      method: "POST",
      body: JSON.stringify({
        name,
        greeting,
        first_message: greeting,
        alternate_greetings: alternateGreetings
      })
    })
  )
  if (created?.id == null) {
    throw new Error("Failed to create character for playground test.")
  }
  return {
    id: String(created.id),
    name,
    greetings: [greeting, ...alternateGreetings]
  }
}

const ensureCharacters = async (
  serverUrl: string,
  apiKey: string,
  count = 2
) => {
  const list = await requestJson(serverUrl, apiKey, "/api/v1/characters/").catch(
    () => requestJson(serverUrl, apiKey, "/api/v1/characters")
  )
  const characters = parseListPayload(list)
  const existing = characters.filter(
    (c: any) =>
      c?.id &&
      c?.name &&
      !/^default assistant$/i.test(String(c.name).trim()) &&
      collectGreetings(c).length > 0
  )
  const results: { id: string; name: string; greetings: string[] }[] = []
  const createdIds: string[] = []
  const usedIds = new Set<string>()

  for (const entry of existing) {
    if (results.length >= count) break
    const id = String(entry.id)
    if (usedIds.has(id)) continue
    usedIds.add(id)
    results.push({
      id,
      name: String(entry.name),
      greetings: collectGreetings(entry)
    })
  }

  while (results.length < count) {
    const suffix = `${Date.now()}-${results.length + 1}`
    const name = `E2E Character ${suffix}`
    const created = await createCharacter(serverUrl, apiKey, name)
    createdIds.push(created.id)
    results.push(created)
  }

  return { characters: results, createdIds }
}

const deleteCharacter = async (
  serverUrl: string,
  apiKey: string,
  characterId: string
) => {
  await requestJson(serverUrl, apiKey, `/api/v1/characters/${characterId}`, {
    method: "DELETE"
  }).catch(() => null)
}

const setDefaultCharacterPreference = async (
  serverUrl: string,
  apiKey: string,
  characterId: string | null
) =>
  requestJson(serverUrl, apiKey, "/api/v1/users/me/profile", {
    method: "PATCH",
    body: JSON.stringify({
      updates: [
        {
          key: DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY,
          value: characterId
        }
      ]
    })
  })

test.describe("Playground character selection", () => {
  test("sends character_id when chatting as a selected character", async () => {
    test.setTimeout(120000)
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    let modelsResponse: Response | null = null
    try {
      modelsResponse = await fetchWithTimeout(
        `${normalizedServerUrl}/api/v1/llm/models/metadata`,
        { headers: { "x-api-key": apiKey } }
      )
    } catch (error) {
      test.skip(
        true,
        `Chat models preflight unreachable in this environment: ${String(error)}`
      )
      return
    }
    if (!modelsResponse) return
    if (!modelsResponse.ok) {
      const body = await modelsResponse.text().catch(() => "")
      test.skip(
        true,
        `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
      )
      return
    }
    const modelId = getFirstModelId(
      await modelsResponse.json().catch(() => [])
    )
    if (!modelId) {
      test.skip(true, "No chat models returned from tldw_server.")
    }
    const selectedModelId = modelId.startsWith("tldw:")
      ? modelId
      : `tldw:${modelId}`

    let characters: { id: string; name: string; greetings: string[] }[] = []
    let createdIds: string[] = []
    try {
      const provisioned = await createEphemeralCharacters(
        normalizedServerUrl,
        apiKey,
        2
      )
      characters = provisioned.characters
      createdIds = provisioned.createdIds
    } catch {
      const fallback = await ensureCharacters(normalizedServerUrl, apiKey, 2)
      characters = fallback.characters
      createdIds = fallback.createdIds
    }
    const [character, secondCharacter] = characters
    const createdCharacterIds = [...createdIds]

    const { context, page, extensionId, optionsUrl } =
      await launchWithExtensionOrSkip(test, "", {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldw_skip_landing_hub: true,
          "tldw:workflow:landing-config": {
            showOnFirstRun: true,
            dismissedAt: Date.now(),
            completedWorkflows: []
          },
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        }
      })
    page.setDefaultTimeout(15000)
    page.setDefaultNavigationTimeout(20000)

    const origin = new URL(normalizedServerUrl).origin + "/*"
    const granted = await grantHostPermission(context, extensionId, origin)
    if (!granted) {
      await context.close()
      for (const createdId of createdCharacterIds) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdId)
      }
      test.skip(
        true,
        "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
      )
    }

    try {
      await page.goto(`${optionsUrl}#/`, {
        waitUntil: "domcontentloaded"
      })
      await waitForConnectionStore(page, "character-playground:before-check")
      await forceConnected(
        page,
        { serverUrl: normalizedServerUrl },
        "character-playground:force-connect"
      )
      await setSelectedModel(page, selectedModelId)
      await dismissWorkflowHubIfVisible(page)
      await enterCharacterChatModeIfAvailable(page)
      const composerInput = page.locator("#textarea-message")
      await expect(composerInput).toBeVisible({ timeout: 15000 })

      await openCharacterSelector(page)

      const searchInput = await findVisibleSearchInput(page)
      if (searchInput) {
        await searchInput.fill(character.name)
        await page.waitForTimeout(100)
      }
      const menuItem = await findCharacterMenuItem(page, character.name)
      if (!menuItem) {
        const visibleMenuLabels = await collectVisibleCharacterMenuLabels(page)
        throw new Error(
          `Unable to find character menu item for "${character.name}". Visible menu labels: ${JSON.stringify(
            visibleMenuLabels
          )}`
        )
      }
      await expect(menuItem).toBeVisible({ timeout: 15000 })
      await menuItem.click()
      await confirmCharacterSwitchIfNeeded(page)
      await expect
        .poll(
          async () => {
            const selection = await readSelectedCharacterFromStorage(page)
            return selection?.id ? String(selection.id) : ""
          },
          { timeout: 15000 }
        )
        .toBe(String(character.id))

      const storageSnapshot = await readSelectedCharacterFromStorage(page)
      if (!storageSnapshot?.id) {
        const debugPayload = {
          selectedCharacter: storageSnapshot
        }
        const debugPath = test
          .info()
          .outputPath("character-storage-missing.json")
        fs.writeFileSync(debugPath, JSON.stringify(debugPayload, null, 2))
        throw new Error("An owned Character assistant was not stored after selection.")
      }
      expect(String(storageSnapshot.id)).toBe(String(character.id))

      await waitForGreeting(page, character.name)

      await openCharacterSelector(page)
      const secondSearchInput = await findVisibleSearchInput(page)
      if (secondSearchInput) {
        await secondSearchInput.fill(secondCharacter.name)
        await page.waitForTimeout(100)
      }
      const secondItem = await findCharacterMenuItem(page, secondCharacter.name)
      if (!secondItem) {
        const visibleMenuLabels = await collectVisibleCharacterMenuLabels(page)
        throw new Error(
          `Unable to find second character menu item for "${secondCharacter.name}". Visible menu labels: ${JSON.stringify(
            visibleMenuLabels
          )}`
        )
      }
      await expect(secondItem).toBeVisible({ timeout: 15000 })
      await secondItem.click()
      await confirmCharacterSwitchIfNeeded(page)
      await expect
        .poll(
          async () => {
            const selection = await readSelectedCharacterFromStorage(page)
            return selection?.id ? String(selection.id) : ""
          },
          { timeout: 15000 }
        )
        .toBe(String(secondCharacter.id))

      await waitForGreeting(page, secondCharacter.name)
      await expect(
        page.locator(
          "[data-message-type='character:greeting'], [data-message-type='greeting']",
          { hasText: character.name }
        )
      ).toHaveCount(0)
    } finally {
      if (!page.isClosed()) {
        const finalScreenshotPath = test
          .info()
          .outputPath("final-before-close.png")
        await page
          .screenshot({ path: finalScreenshotPath, fullPage: true })
          .then(() => {
            test.info().attach("final-before-close", {
              path: finalScreenshotPath,
              contentType: "image/png"
            })
          })
          .catch(() => null)
      }
      await context.close()
      for (const createdId of createdCharacterIds) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdId)
      }
    }
  })

  test("preselects server default after entering character chat and preserves manual override across reload", async () => {
    test.setTimeout(120000)
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    let modelsResponse: Response | null = null
    try {
      modelsResponse = await fetchWithTimeout(
        `${normalizedServerUrl}/api/v1/llm/models/metadata`,
        { headers: { "x-api-key": apiKey } }
      )
    } catch (error) {
      test.skip(
        true,
        `Chat models preflight unreachable in this environment: ${String(error)}`
      )
      return
    }
    if (!modelsResponse) return
    if (!modelsResponse.ok) {
      const body = await modelsResponse.text().catch(() => "")
      test.skip(
        true,
        `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
      )
      return
    }
    const modelId = getFirstModelId(
      await modelsResponse.json().catch(() => [])
    )
    if (!modelId) {
      test.skip(true, "No chat models returned from tldw_server.")
    }
    const selectedModelId = modelId.startsWith("tldw:")
      ? modelId
      : `tldw:${modelId}`

    let characters: { id: string; name: string; greetings: string[] }[] = []
    let createdIds: string[] = []
    try {
      const provisioned = await createEphemeralCharacters(
        normalizedServerUrl,
        apiKey,
        2
      )
      characters = provisioned.characters
      createdIds = provisioned.createdIds
    } catch {
      const fallback = await ensureCharacters(normalizedServerUrl, apiKey, 2)
      characters = fallback.characters
      createdIds = fallback.createdIds
    }
    const [defaultCharacter, manualCharacter] = characters
    const createdCharacterIds = [...createdIds]

    const clearDefaultPreference = async () => {
      await setDefaultCharacterPreference(normalizedServerUrl, apiKey, null).catch(
        () => null
      )
    }

    try {
      await setDefaultCharacterPreference(
        normalizedServerUrl,
        apiKey,
        String(defaultCharacter.id)
      )
    } catch (error) {
      await clearDefaultPreference()
      for (const createdId of createdCharacterIds) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdId)
      }
      test.skip(
        true,
        `Unable to set profile default character preference: ${String(error)}`
      )
      return
    }

    const { context, page, extensionId, optionsUrl } =
      await launchWithExtensionOrSkip(test, "", {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldw_skip_landing_hub: true,
          "tldw:workflow:landing-config": {
            showOnFirstRun: true,
            dismissedAt: Date.now(),
            completedWorkflows: []
          },
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        }
      })
    page.setDefaultTimeout(15000)
    page.setDefaultNavigationTimeout(20000)

    const origin = new URL(normalizedServerUrl).origin + "/*"
    const granted = await grantHostPermission(context, extensionId, origin)
    if (!granted) {
      await context.close()
      await clearDefaultPreference()
      for (const createdId of createdCharacterIds) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdId)
      }
      test.skip(
        true,
        "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
      )
    }

    try {
      await page.goto(`${optionsUrl}#/`, {
        waitUntil: "domcontentloaded"
      })
      await waitForConnectionStore(page, "character-default-bootstrap:before-check")
      await forceConnected(
        page,
        { serverUrl: normalizedServerUrl },
        "character-default-bootstrap:force-connect"
      )
      await setSelectedModel(page, selectedModelId)
      await dismissWorkflowHubIfVisible(page)
      await enterCharacterChatModeIfAvailable(page)

      await expect
        .poll(
          async () => {
            const selection = await readSelectedCharacterFromStorage(page)
            return selection?.id ? String(selection.id) : ""
          },
          { timeout: 20000 }
        )
        .toBe(String(defaultCharacter.id))

      await openCharacterSelector(page)
      const searchInput = await findVisibleSearchInput(page)
      if (searchInput) {
        await searchInput.fill(manualCharacter.name)
        await page.waitForTimeout(100)
      }
      const manualItem = await findCharacterMenuItem(page, manualCharacter.name)
      if (!manualItem) {
        const visibleMenuLabels = await collectVisibleCharacterMenuLabels(page)
        throw new Error(
          `Unable to find manual character menu item for "${manualCharacter.name}". Visible menu labels: ${JSON.stringify(
            visibleMenuLabels
          )}`
        )
      }
      await expect(manualItem).toBeVisible({ timeout: 15000 })
      await manualItem.click()
      await confirmCharacterSwitchIfNeeded(page)

      await expect
        .poll(
          async () => {
            const selection = await readSelectedCharacterFromStorage(page)
            return selection?.id ? String(selection.id) : ""
          },
          { timeout: 15000 }
        )
        .toBe(String(manualCharacter.id))

      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "character-default-bootstrap:after-reload")
      await forceConnected(
        page,
        { serverUrl: normalizedServerUrl },
        "character-default-bootstrap:after-reload-force-connect"
      )
      await setSelectedModel(page, selectedModelId)
      await dismissWorkflowHubIfVisible(page)
      await enterCharacterChatModeIfAvailable(page)

      await expect
        .poll(
          async () => {
            const selection = await readSelectedCharacterFromStorage(page)
            return selection?.id ? String(selection.id) : ""
          },
          { timeout: 20000 }
        )
        .toBe(String(manualCharacter.id))

      const postReloadSelection = await readSelectedCharacterFromStorage(page)
      expect(String(postReloadSelection?.id || "")).not.toBe(
        String(defaultCharacter.id)
      )
    } finally {
      if (!page.isClosed()) {
        const finalScreenshotPath = test
          .info()
          .outputPath("default-preselect-before-close.png")
        await page
          .screenshot({ path: finalScreenshotPath, fullPage: true })
          .then(() => {
            test.info().attach("default-preselect-before-close", {
              path: finalScreenshotPath,
              contentType: "image/png"
            })
          })
          .catch(() => null)
      }
      await context.close()
      await clearDefaultPreference()
      for (const createdId of createdCharacterIds) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdId)
      }
    }
  })
})
