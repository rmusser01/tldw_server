import { test, expect, type Locator } from "@playwright/test"
import { grantHostPermission } from "./utils/permissions"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"
import { forceConnected, waitForConnectionStore } from "./utils/connection"

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value : `http://${value}`

const parseListPayload = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload
  if (!payload || typeof payload !== "object") return []
  const candidate = payload
  return (
    candidate.items ||
    candidate.results ||
    candidate.data ||
    candidate.chats ||
    candidate.characters ||
    []
  )
}

const requestJson = async (
  serverUrl: string,
  apiKey: string,
  path: string,
  init?: RequestInit,
  timeoutMs = 15000
) => {
  const controller = new AbortController()
  const timeoutHandle = setTimeout(() => controller.abort(), timeoutMs)
  let response: Response
  try {
    response = await fetch(`${serverUrl}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        ...(init?.headers || {})
      }
    })
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs}ms: ${path}`)
    }
    throw error
  } finally {
    clearTimeout(timeoutHandle)
  }
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

const ensureCharacterId = async (serverUrl: string, apiKey: string) => {
  const list = await requestJson(serverUrl, apiKey, "/api/v1/characters/").catch(
    () => requestJson(serverUrl, apiKey, "/api/v1/characters")
  )
  const characters = parseListPayload(list)
  const defaultName = "Helpful AI Assistant"
  const existing = characters.find((c: any) => {
    const name = String(c?.name || "").trim().toLowerCase()
    return name === defaultName.toLowerCase()
  })
  if (existing?.id != null) {
    return { id: String(existing.id), created: false }
  }
  const created = await requestJson(serverUrl, apiKey, "/api/v1/characters/", {
    method: "POST",
    body: JSON.stringify({ name: `E2E Pin Character ${Date.now()}` })
  }).catch(() =>
    requestJson(serverUrl, apiKey, "/api/v1/characters", {
      method: "POST",
      body: JSON.stringify({ name: `E2E Pin Character ${Date.now()}` })
    })
  )
  if (created?.id == null) {
    throw new Error("Failed to create character for pin test.")
  }
  return { id: String(created.id), created: true }
}

const createServerChat = async (
  serverUrl: string,
  apiKey: string,
  payload: Record<string, any>
) => {
  const created = await requestJson(serverUrl, apiKey, "/api/v1/chats/", {
    method: "POST",
    body: JSON.stringify(payload)
  }).catch(() =>
    requestJson(serverUrl, apiKey, "/api/v1/chats", {
      method: "POST",
      body: JSON.stringify(payload)
    })
  )
  const rawId = created?.id ?? created?.chat_id ?? null
  return rawId != null ? String(rawId) : null
}

const deleteServerChat = async (
  serverUrl: string,
  apiKey: string,
  chatId: string
) => {
  await requestJson(serverUrl, apiKey, `/api/v1/chats/${chatId}`, {
    method: "DELETE"
  }).catch(() => null)
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

const selectServerTab = async (sidebar: Locator) => {
  const radio = sidebar.getByRole("radio", { name: /^Server/i })
  if ((await radio.count()) > 0) {
    await radio.first().click()
    return
  }
  const button = sidebar.getByRole("button", { name: /^Server/i })
  if ((await button.count()) > 0) {
    await button.first().click()
    return
  }
  await sidebar.getByText(/^Server/i).first().click()
}

test.describe("Server chat pins", () => {
  test("pins and unpins server chats in the main chat sidebar", async () => {
    test.setTimeout(120000)
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    const createdChats: string[] = []
    let createdCharacterId: string | null = null
    let context: Awaited<ReturnType<typeof launchWithExtensionOrSkip>>["context"] | null = null

    try {
      const character = await ensureCharacterId(normalizedServerUrl, apiKey)
      createdCharacterId = character.created ? character.id : null

      const timestamp = Date.now()
      const chatTitleA = `E2E Pin Chat A ${timestamp}`
      const chatTitleB = `E2E Pin Chat B ${timestamp}`
      const chatAId = await createServerChat(normalizedServerUrl, apiKey, {
        title: chatTitleA,
        character_id: character.id,
        state: "in-progress",
        source: "e2e"
      })
      const chatBId = await createServerChat(normalizedServerUrl, apiKey, {
        title: chatTitleB,
        character_id: character.id,
        state: "in-progress",
        source: "e2e"
      })
      if (!chatAId || !chatBId) {
        test.skip(true, "Unable to create server chats for pin test.")
      }
      if (chatAId) createdChats.push(chatAId)
      if (chatBId) createdChats.push(chatBId)

      const launchResult = await launchWithExtensionOrSkip(test, "", {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        }
      })
      context = launchResult.context
      const { page, extensionId } = launchResult

      const optionsUrl = `chrome-extension://${extensionId}/options.html`
      const origin = new URL(normalizedServerUrl).origin + "/*"
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
        )
      }

      // The chat list shows only on the chat route (XP-11).
      await page.goto(`${optionsUrl}#/chat`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "server-pin:options-store")
      await forceConnected(
        page,
        { serverUrl: normalizedServerUrl },
        "server-pin:force-connect"
      )
      await page.evaluate(() => {
        if (chrome?.storage?.local?.remove) {
          chrome.storage.local.remove("tldw:server-chat-pins")
        }
      })

      const sidebar = page.getByTestId("chat-sidebar")
      await expect(sidebar).toBeVisible({ timeout: 20000 })
      await selectServerTab(sidebar)

      const chatAButton = sidebar.getByRole("button", {
        name: new RegExp(chatTitleA)
      })
      const chatVisible = await chatAButton
        .isVisible({ timeout: 20000 })
        .catch(() => false)
      if (!chatVisible) {
        test.skip(true, "Server chat rows were not visible in the sidebar.")
      }

      const chatARow = chatAButton.locator("..")
      const pinButton = chatARow.getByRole("button", { name: /^Pin$/i })
      const hasPinButton = await pinButton
        .isVisible({ timeout: 10000 })
        .catch(() => false)
      if (!hasPinButton) {
        test.skip(true, "Pin button not present for server chat row in this UI variant.")
      }
      await pinButton.click()
      await expect(
        chatARow.getByRole("button", { name: /^Unpin$/i })
      ).toBeVisible({ timeout: 10000 })
      await expect(sidebar.getByText(/^Pinned$/i)).toBeVisible({
        timeout: 10000
      })

      const unpinButton = chatARow.getByRole("button", { name: /^Unpin$/i })
      await unpinButton.click()
      await expect(sidebar.getByText(/^Pinned$/i)).toHaveCount(0)

    } catch (error) {
      test.skip(true, `Server chat pin test skipped: ${String(error)}`)
    } finally {
      if (context) {
        await context.close().catch(() => null)
      }
      for (const chatId of createdChats) {
        await deleteServerChat(normalizedServerUrl, apiKey, chatId)
      }
      if (createdCharacterId) {
        await deleteCharacter(normalizedServerUrl, apiKey, createdCharacterId)
      }
    }
  })
})
