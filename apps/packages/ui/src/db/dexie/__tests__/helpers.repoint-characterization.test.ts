import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

/**
 * WP-18 / TASK-13522 Stage 4 characterization: the Dexie implementations that
 * production code repointed to (`db/dexie/helpers.ts` getPromptById /
 * getSessionFiles, backed by `db.prompts.get` / `db.sessionFiles.get`) return
 * the same shapes and semantics as the legacy chrome.storage blob DB oracle
 * (pinned in db/__tests__/legacy-blob-db.quarantine.test.ts):
 * existing / missing / empty-id / soft-deleted prompts and the session-files
 * shape, plus the Firefox-private-mode DatabaseClosedError fallback wiring.
 */

const memory = vi.hoisted(() => {
  const prompts = new Map<string, any>()
  const sessionFiles = new Map<string, any>()
  const state = {
    failPromptGet: null as null | Error
  }
  return {
    __prompts: prompts,
    __sessionFiles: sessionFiles,
    __state: state,
    prompts: {
      get: async (id: string) => {
        if (state.failPromptGet) throw state.failPromptGet
        const record = prompts.get(id)
        return record === undefined ? undefined : structuredClone(record)
      },
      put: async (value: any) => {
        prompts.set(value.id, structuredClone(value))
      },
      update: async (id: string, patch: any) => {
        prompts.set(id, { ...prompts.get(id), ...structuredClone(patch) })
      }
    },
    sessionFiles: {
      get: async (id: string) => {
        const record = sessionFiles.get(id)
        return record === undefined ? undefined : structuredClone(record)
      },
      put: async (value: any) => {
        sessionFiles.set(value.sessionId, structuredClone(value))
      }
    }
  }
})

vi.mock("../schema", () => ({ db: memory }))

import { getPromptById, getSessionFiles, getSessionFilesInfo } from "../helpers"

const promptFixture = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  title: `Title ${id}`,
  name: `Title ${id}`,
  content: `Content ${id}`,
  is_system: false,
  keywords: ["k"],
  createdAt: 1234,
  ...extra
})

const chromeWithPrompts = (prompts: unknown[]) => {
  const store = new Map<string, unknown>([["prompts", prompts]])
  vi.stubGlobal("chrome", {
    storage: {
      local: {
        get: (keys: any, callback: (items: any) => void) => {
          const out: Record<string, unknown> = {}
          for (const key of Array.isArray(keys) ? keys : [keys]) {
            if (store.has(key)) out[key] = structuredClone(store.get(key))
          }
          setTimeout(() => callback(out), 0)
        },
        set: (items: Record<string, unknown>, callback: () => void) => {
          Object.entries(items).forEach(([key, value]) =>
            store.set(key, structuredClone(value))
          )
          setTimeout(() => callback(), 0)
        },
        remove: (keys: any, callback?: () => void) => {
          for (const key of Array.isArray(keys) ? keys : [keys]) {
            store.delete(key)
          }
          setTimeout(() => callback?.(), 0)
        }
      }
    }
  })
  return store
}

beforeEach(() => {
  memory.__prompts.clear()
  memory.__sessionFiles.clear()
  memory.__state.failPromptGet = null
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("Dexie getPromptById characterization (mirrors legacy oracle)", () => {
  it("returns the stored record with the same shape for an existing prompt", async () => {
    const existing = promptFixture("p1", {
      favorite: true,
      deletedAt: null
    })
    await memory.prompts.put(existing)
    await memory.prompts.put(promptFixture("p2"))

    await expect(getPromptById("p1")).resolves.toEqual(existing)
  })

  it("returns undefined for a missing prompt (legacy: undefined)", async () => {
    await memory.prompts.put(promptFixture("p1"))

    await expect(getPromptById("nope")).resolves.toBeUndefined()
  })

  it("returns null for an empty or whitespace id (legacy: null)", async () => {
    await expect(getPromptById("")).resolves.toBeNull()
    await expect(getPromptById("   ")).resolves.toBeNull()
  })

  it("still surfaces soft-deleted prompts (legacy: visible via deletedAt)", async () => {
    const softDeleted = promptFixture("p1", { deletedAt: 9999 })
    await memory.prompts.put(softDeleted)
    // Soft delete is a deletedAt-stamping update, never a row removal.
    await memory.prompts.update("p1", { deletedAt: 11111, updatedAt: 11111 })

    await expect(getPromptById("p1")).resolves.toMatchObject({
      id: "p1",
      deletedAt: 11111
    })
  })

  it("falls back to the legacy chrome.storage blob on DatabaseClosedError", async () => {
    const fbPrompt = promptFixture("fb-1")
    chromeWithPrompts([fbPrompt])
    memory.__state.failPromptGet = Object.assign(
      new Error("IndexedDB is closed"),
      { name: "DatabaseClosedError" }
    )

    // Firefox-private-mode fallback map entry: helpers -> getPromptByIdFB
    // -> legacy PageAssitDatabase -> chrome.storage.local.
    await expect(getPromptById("fb-1")).resolves.toEqual(fbPrompt)
    await expect(getPromptById("never-stored")).resolves.toBeUndefined()
  })
})

describe("Dexie session-files characterization (mirrors legacy oracle)", () => {
  const file = {
    id: "f1",
    filename: "a.txt",
    type: "text/plain",
    content: "hello",
    size: 5,
    uploadedAt: 10,
    processed: true
  }

  it("returns the files array for a known session and [] for an unknown one", async () => {
    await memory.sessionFiles.put({
      sessionId: "s1",
      files: [file],
      retrievalEnabled: true,
      createdAt: 1
    })

    await expect(getSessionFiles("s1")).resolves.toEqual([file])
    await expect(getSessionFiles("missing")).resolves.toEqual([])
    await expect(getSessionFilesInfo("missing")).resolves.toBeNull()
    await expect(getSessionFilesInfo("s1")).resolves.toEqual({
      sessionId: "s1",
      files: [file],
      retrievalEnabled: true,
      createdAt: 1
    })
  })
})
