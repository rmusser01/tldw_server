import type { PendingClipDraft } from "./pending-draft"
import type { WebClipperSaveResponse } from "./types"

export const WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY =
  "tldw:web-clipper:pendingAgentTask"

const EXTRACT_PREVIEW_LIMIT = 1200

export type PendingWebClipAgentTaskRequest = {
  id: string
  clipId: string
  noteId: string
  workspaceId: string
  workspaceNoteId: number
  pageUrl: string
  pageTitle: string
  extractPreview: string
  hasScreenshot: boolean
  createdAt: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const readString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : ""

const readPositiveNumber = (value: unknown): number | null => {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? Math.trunc(parsed) : null
}

const truncatePreview = (value: string): string =>
  value.length <= EXTRACT_PREVIEW_LIMIT
    ? value
    : `${value.slice(0, EXTRACT_PREVIEW_LIMIT).trimEnd()}\n[Truncated for agent task handoff.]`

type ExtensionStorageArea = {
  get?: (
    key: string,
    callback?: (items: Record<string, unknown>) => void
  ) => Promise<Record<string, unknown>> | void
  set?: (
    items: Record<string, unknown>,
    callback?: () => void
  ) => Promise<void> | void
  remove?: (key: string, callback?: () => void) => Promise<void> | void
}

const getExtensionStorageArea = (): ExtensionStorageArea | null => {
  const storage = globalThis.chrome?.storage
  return storage?.session ?? storage?.local ?? null
}

const hasChromeRuntimeError = (): boolean =>
  Boolean(globalThis.chrome?.runtime?.lastError)

const readExtensionStorageValue = async (key: string): Promise<unknown> => {
  const storage = getExtensionStorageArea()
  if (!storage?.get) return undefined

  return new Promise((resolve) => {
    let settled = false
    const settle = (items: unknown) => {
      if (settled) return
      settled = true
      resolve(isRecord(items) ? items[key] : undefined)
    }

    try {
      const maybePromise = storage.get(key, (items) => {
        settle(hasChromeRuntimeError() ? undefined : items)
      })
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(settle).catch(() => settle(undefined))
      }
    } catch {
      settle(undefined)
    }
  })
}

const writeExtensionStorageValue = async (
  key: string,
  value: unknown
): Promise<boolean> => {
  const storage = getExtensionStorageArea()
  if (!storage?.set) return false

  return new Promise((resolve) => {
    let settled = false
    const settle = (success: boolean) => {
      if (settled) return
      settled = true
      resolve(success)
    }

    try {
      const maybePromise = storage.set({ [key]: value }, () =>
        settle(!hasChromeRuntimeError())
      )
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(() => settle(true)).catch(() => settle(false))
      }
    } catch {
      settle(false)
    }
  })
}

const removeExtensionStorageValue = async (key: string): Promise<boolean> => {
  const storage = getExtensionStorageArea()
  if (!storage?.remove) return false

  return new Promise((resolve) => {
    let settled = false
    const settle = (success: boolean) => {
      if (settled) return
      settled = true
      resolve(success)
    }

    try {
      const maybePromise = storage.remove(key, () =>
        settle(!hasChromeRuntimeError())
      )
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(() => settle(true)).catch(() => settle(false))
      }
    } catch {
      settle(false)
    }
  })
}

const readBrowserStorageValue = (key: string): unknown => {
  if (typeof window === "undefined") return undefined

  for (const storage of [window.localStorage, window.sessionStorage]) {
    try {
      const raw = storage.getItem(key)
      if (!raw) continue
      return JSON.parse(raw)
    } catch {
      continue
    }
  }

  return undefined
}

const writeBrowserStorageValue = (key: string, value: unknown): void => {
  if (typeof window === "undefined") return
  const serialized = JSON.stringify(value)

  try {
    window.localStorage.setItem(key, serialized)
    return
  } catch {
    // Fall back to same-context storage for non-extension tests.
  }

  try {
    window.sessionStorage.setItem(key, serialized)
  } catch {
    // Ignore transient storage failures.
  }
}

const removeBrowserStorageValue = (key: string): void => {
  if (typeof window === "undefined") return
  for (const storage of [window.localStorage, window.sessionStorage]) {
    try {
      storage.removeItem(key)
    } catch {
      // Ignore transient storage failures.
    }
  }
}

export const buildPendingWebClipAgentTaskRequest = ({
  draft,
  response
}: {
  draft: PendingClipDraft
  response: WebClipperSaveResponse
}): PendingWebClipAgentTaskRequest | null => {
  const placement = response.workspace_placement
  const workspaceId = readString(placement?.workspace_id)
  const noteId = readString(response.note?.id ?? response.note_id)
  const workspaceNoteId = readPositiveNumber(placement?.workspace_note_id)
  if (!workspaceId || !noteId || workspaceNoteId == null) {
    return null
  }

  const extractPreview = truncatePreview(
    readString(draft.selectionText) ||
      readString(draft.fullExtract) ||
      readString(draft.visibleBody)
  )

  return {
    id: crypto.randomUUID(),
    clipId: response.clip_id || draft.clipId,
    noteId,
    workspaceId,
    workspaceNoteId,
    pageUrl: draft.pageUrl,
    pageTitle: draft.pageTitle,
    extractPreview,
    hasScreenshot: Boolean(draft.captureMetadata.screenshotDataUrl?.trim()),
    createdAt: new Date().toISOString()
  }
}

export const readPendingWebClipAgentTaskRequest =
  async (): Promise<PendingWebClipAgentTaskRequest | null> => {
    const raw = await readExtensionStorageValue(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY
    )
    const storedValue =
      raw == null
        ? readBrowserStorageValue(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)
        : raw
    try {
      if (!storedValue) return null
      const parsed =
        typeof storedValue === "string" ? JSON.parse(storedValue) : storedValue
      if (!parsed || typeof parsed !== "object") return null
      const record = parsed as Record<string, unknown>
      const workspaceId = readString(record.workspaceId)
      const noteId = readString(record.noteId)
      const workspaceNoteId = readPositiveNumber(record.workspaceNoteId)
      const pageUrl = readString(record.pageUrl)
      const pageTitle = readString(record.pageTitle)
      if (!workspaceId || !noteId || workspaceNoteId == null || !pageUrl) {
        return null
      }
      return {
        id: readString(record.id) || crypto.randomUUID(),
        clipId: readString(record.clipId),
        noteId,
        workspaceId,
        workspaceNoteId,
        pageUrl,
        pageTitle,
        extractPreview: truncatePreview(readString(record.extractPreview)),
        hasScreenshot: Boolean(record.hasScreenshot),
        createdAt: readString(record.createdAt) || new Date().toISOString()
      }
    } catch {
      return null
    }
  }

export const writePendingWebClipAgentTaskRequest = (
  request: PendingWebClipAgentTaskRequest
): Promise<void> => {
  return writeExtensionStorageValue(
    WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
    request
  ).then((storedInExtension) => {
    if (!storedInExtension) {
      writeBrowserStorageValue(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY, request)
    }
  })
}

export const clearPendingWebClipAgentTaskRequest = async (): Promise<void> => {
  const removed = await removeExtensionStorageValue(
    WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY
  )
  if (!removed) {
    await writeExtensionStorageValue(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY, null)
  }
  removeBrowserStorageValue(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)
}
