import type { Storage } from "@plasmohq/storage"
import { browser } from "wxt/browser"
import { getInitialConfig } from "@/services/action"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"

export type BackgroundCapabilities = {
  sendToTldw: boolean
  processLocal: boolean
  transcribe: boolean
  openApiCheck: boolean
}

export type BackgroundInitOptions = {
  storage: Storage
  contextMenuId: { webui: string; sidePanel: string }
  saveToClipperMenuId: string
  saveToCompanionMenuId: string
  saveToNotesMenuId: string
  narrateSelectionMenuId: string
  transcribeMenuId: { transcribe: string; transcribeAndSummarize: string }
  warmModels: (force?: boolean, throwOnError?: boolean) => Promise<any[] | null>
  capabilities: BackgroundCapabilities
  onActionIconClickChange: (value: string) => void
  onContextMenuClickChange: (value: string) => void
}

export type BackgroundInitResult = {
  modelWarmAlarmName: string
}

export const MODEL_WARM_ALARM_NAME = "tldw:model-warm"
const MODEL_WARM_INTERVAL_MINUTES = 60
const OPENAPI_DRIFT_LAST_CHECK_KEY = "__tldwOpenApiDriftLastCheckV1"
const OPENAPI_DRIFT_MIN_INTERVAL_MS = 15 * 60 * 1000

type OpenApiDriftLastCheck = {
  base: string
  checkedAt: number
}

const getServerUrl = (cfg: any): string => {
  if (!cfg || typeof cfg !== "object") return ""
  return String(cfg.serverUrl || "").trim()
}

const scheduleModelWarmAlarm = async (enabled: boolean) => {
  try {
    if (!browser?.alarms) return
    if (!enabled) {
      await browser.alarms.clear(MODEL_WARM_ALARM_NAME)
      return
    }
    await browser.alarms.clear(MODEL_WARM_ALARM_NAME)
    await browser.alarms.create(MODEL_WARM_ALARM_NAME, {
      periodInMinutes: MODEL_WARM_INTERVAL_MINUTES
    })
  } catch (error) {
    console.debug(
      "[tldw] model warm alarm setup failed:",
      (error as any)?.message || error
    )
  }
}

const syncWebClipperContextMenu = async (menuId: string) => {
  try {
    await browser.contextMenus.remove(menuId)
  } catch {
    // best effort removal before re-creating the menu item
  }

  let hasWebClipper = false
  try {
    const capabilities = await getServerCapabilities({ forceRefresh: true })
    hasWebClipper = Boolean(capabilities.hasWebClipper)
  } catch {
    hasWebClipper = false
  }

  if (!hasWebClipper) {
    return
  }

  browser.contextMenus.create({
    id: menuId,
    title: browser.i18n.getMessage("contextSaveToClipper"),
    contexts: ["page", "selection"]
  })
}

const shouldSkipOpenApiDriftCheck = async (
  storage: Storage,
  base: string
): Promise<boolean> => {
  try {
    const raw = await storage.get<OpenApiDriftLastCheck | null>(
      OPENAPI_DRIFT_LAST_CHECK_KEY
    )
    if (!raw || typeof raw !== "object") return false

    const previousBase = String((raw as any).base || "")
    const checkedAt = Number((raw as any).checkedAt || 0)
    if (!previousBase || previousBase !== base) return false
    if (!Number.isFinite(checkedAt) || checkedAt <= 0) return false
    return Date.now() - checkedAt < OPENAPI_DRIFT_MIN_INTERVAL_MS
  } catch {
    return false
  }
}

const markOpenApiDriftCheck = async (
  storage: Storage,
  base: string
): Promise<void> => {
  try {
    await storage.set(OPENAPI_DRIFT_LAST_CHECK_KEY, {
      base,
      checkedAt: Date.now()
    } satisfies OpenApiDriftLastCheck)
  } catch {
    // best-effort cache write only
  }
}

const checkOpenApiDrift = async (storage: Storage) => {
  let timeout: ReturnType<typeof setTimeout> | null = null
  try {
    const cfg = await storage.get<any>("tldwConfig")
    const base = String(cfg?.serverUrl || "").replace(/\/$/, "")
    if (!base) return
    if (await shouldSkipOpenApiDriftCheck(storage, base)) return

    const controller = new AbortController()
    timeout = setTimeout(() => controller.abort(), 10000)
    const headers: Record<string, string> = {}

    if (cfg?.authMode === "single-user") {
      const key = String(cfg?.apiKey || "").trim()
      if (key) headers["X-API-KEY"] = key
    } else if (cfg?.authMode === "multi-user") {
      const token = String(cfg?.accessToken || "").trim()
      if (token) headers["Authorization"] = `Bearer ${token}`
    }

    const res = await fetch(`${base}/openapi.json`, {
      headers,
      signal: controller.signal
    })
    await markOpenApiDriftCheck(storage, base)

    if (!res.ok) return

    const spec = await res.json().catch(() => null)
    const paths = spec && spec.paths ? spec.paths : {}
    const required = [
      "/api/v1/chat/completions",
      "/api/v1/rag/search",
      "/api/v1/rag/search/stream",
      "/api/v1/media/ingest/jobs",
      "/api/v1/media/process-videos",
      "/api/v1/media/process-audios",
      "/api/v1/media/process-pdfs",
      "/api/v1/media/process-ebooks",
      "/api/v1/media/process-documents",
      "/api/v1/media/process-web-scraping",
      "/api/v1/reading/save",
      "/api/v1/reading/items",
      "/api/v1/audio/transcriptions",
      "/api/v1/audio/speech",
      "/api/v1/llm/models",
      "/api/v1/llm/models/metadata",
      "/api/v1/llm/providers",
      // Workspace features
      "/api/v1/notes/",
      "/api/v1/notes/search/",
      "/api/v1/flashcards",
      "/api/v1/flashcards/decks",
      "/api/v1/characters/world-books",
      "/api/v1/chat/dictionaries",
      "/api/v1/web-clipper/save"
    ]
    const missing = required.filter((path) => !(path in paths))
    if (missing.length > 0) {
      console.warn("[tldw] OpenAPI drift detected - missing endpoints:", missing)
      try {
        await browser.runtime.sendMessage({
          type: "tldw:openapi-warn",
          payload: { missing }
        })
      } catch {}
    }
  } catch (error) {
    console.debug(
      "[tldw] OpenAPI check skipped:",
      (error as any)?.message || error
    )
  } finally {
    if (timeout) {
      clearTimeout(timeout)
    }
  }
}

export const initBackground = async (
  options: BackgroundInitOptions
): Promise<BackgroundInitResult> => {
  const {
    storage,
    contextMenuId,
    saveToClipperMenuId,
    saveToCompanionMenuId,
    saveToNotesMenuId,
    narrateSelectionMenuId,
    transcribeMenuId,
    warmModels,
    capabilities,
    onActionIconClickChange,
    onContextMenuClickChange
  } = options

  try {
    await browser.contextMenus.removeAll()
  } catch (error) {
    console.debug(
      "[tldw] contextMenus.removeAll failed:",
      (error as any)?.message || error
    )
  }

  const contextMenuTitle = {
    webui: browser.i18n.getMessage("openOptionToChat"),
    sidePanel: browser.i18n.getMessage("openSidePanelToChat")
  }

  storage.watch({
    actionIconClick: (value) => {
      const oldValue = value?.oldValue || "webui"
      const newValue = value?.newValue || "webui"
      if (oldValue !== newValue) {
        onActionIconClickChange(newValue)
      }
    },
    contextMenuClick: (value) => {
      const oldValue = value?.oldValue || "sidePanel"
      const newValue = value?.newValue || "sidePanel"
      if (oldValue !== newValue) {
        onContextMenuClickChange(newValue)
        browser.contextMenus.remove(contextMenuId[oldValue])
        browser.contextMenus.create({
          id: contextMenuId[newValue],
          title: contextMenuTitle[newValue],
          contexts: ["page", "selection"]
        })
      }
    },
    tldwConfig: (value) => {
      const nextUrl = getServerUrl(value?.newValue)
      const prevUrl = getServerUrl(value?.oldValue)
      const hasServer = nextUrl.length > 0
      void scheduleModelWarmAlarm(hasServer)
      void syncWebClipperContextMenu(saveToClipperMenuId)
      if (hasServer && nextUrl !== prevUrl) {
        void warmModels(true)
      }
    }
  })

  const data = await getInitialConfig()
  onContextMenuClickChange(data.contextMenuClick)
  onActionIconClickChange(data.actionIconClick)

  // Top-level: Open Sidebar / Open Web UI (dynamic)
  browser.contextMenus.create({
    id: contextMenuId[data.contextMenuClick],
    title: contextMenuTitle[data.contextMenuClick],
    contexts: ["page", "selection"]
  })

  // Parent: AI Actions
  browser.contextMenus.create({
    id: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextMenuAIActions") || "AI Actions",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "summarize-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextSummarize") || "Summarize",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "explain-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextExplain") || "Explain",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "rephrase-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextRephrase") || "Rephrase",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "translate-pg",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextTranslate") || "Translate",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "custom-pg",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextCustom") || "Custom",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "contextual-popup-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextCopilotPopup") || "AI Popup",
    contexts: ["selection"]
  })

  // Parent: Save
  browser.contextMenus.create({
    id: "tldw-save",
    title: browser.i18n.getMessage("contextMenuSave") || "Save",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: saveToNotesMenuId,
    parentId: "tldw-save",
    title: browser.i18n.getMessage("contextSaveToNotes") || "Save to Notes",
    contexts: ["selection"]
  })
  await syncWebClipperContextMenu(saveToClipperMenuId)
  browser.contextMenus.create({
    id: saveToCompanionMenuId,
    parentId: "tldw-save",
    title:
      browser.i18n.getMessage("contextSaveToCompanion") || "Save to Companion",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: narrateSelectionMenuId,
    parentId: "tldw-save",
    title: browser.i18n.getMessage("contextNarrateSelection") || "Read Aloud",
    contexts: ["selection"]
  })

  // Parent: Process (only created if at least one capability is enabled)
  if (capabilities.sendToTldw || capabilities.processLocal || capabilities.transcribe) {
    browser.contextMenus.create({
      id: "tldw-process",
      title: browser.i18n.getMessage("contextMenuProcess") || "Process",
      contexts: ["page", "link"]
    })

    if (capabilities.sendToTldw) {
      browser.contextMenus.create({
        id: "send-to-tldw",
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextSendToTldw") || "Save to Library",
        contexts: ["page", "link"]
      })
    }

    if (capabilities.processLocal) {
      browser.contextMenus.create({
        id: "process-local-tldw",
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextProcessLocalTldw") || "Analyze without Saving",
        contexts: ["page", "link"]
      })
    }

    if (capabilities.transcribe) {
      browser.contextMenus.create({
        id: transcribeMenuId.transcribe,
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextTranscribeMedia") || "Transcribe Video/Audio",
        contexts: ["page", "link"]
      })
      browser.contextMenus.create({
        id: transcribeMenuId.transcribeAndSummarize,
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextTranscribeAndSummarizeMedia") || "Transcribe + Summarize",
        contexts: ["page", "link"]
      })
    }
  }

  if (capabilities.openApiCheck) {
    await checkOpenApiDrift(storage)
  }

  let hasServer = false
  try {
    const cfg = await storage.get<any>("tldwConfig")
    hasServer = getServerUrl(cfg).length > 0
  } catch {
    hasServer = false
  }

  if (hasServer) {
    await warmModels(true)
  }
  await scheduleModelWarmAlarm(hasServer)

  return { modelWarmAlarmName: MODEL_WARM_ALARM_NAME }
}
