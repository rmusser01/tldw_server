import { beforeEach, describe, expect, it, vi } from "vitest"

import { getPromptById, upsertServerLibraryPromptCopy } from "../helpers"

const mocks = vi.hoisted(() => ({
  prompts: [] as Array<Record<string, unknown>>,
  addPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  savePromptFB: vi.fn(),
  updatePromptFB: vi.fn(),
  getPromptByIdFB: vi.fn(),
  closed: false
}))

vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => ({ ownerId: "recipe-owner:sha256:" + "a".repeat(64), authorizationRevision: "test-owner" })
}))

vi.mock("../chat", () => ({
  PageAssistDatabase: class {
    async getAllPrompts() {
      return mocks.prompts.filter((prompt) => !prompt.deletedAt)
    }
    async getPromptById(id: string) {
      if (mocks.closed) throw Object.assign(new Error("closed"), { name: "DatabaseClosedError" })
      return mocks.prompts.find((prompt) => prompt.id === id)
    }
    async addPrompt(prompt: Record<string, unknown>) {
      mocks.addPrompt(prompt)
      mocks.prompts.push(prompt)
    }
    async updatePrompt(id: string, updates: Record<string, unknown>) {
      mocks.updatePrompt(id, updates)
      // Like chat.ts updatePrompt, fields left undefined keep their value.
      const defined = Object.fromEntries(
        Object.entries(updates).filter(([, value]) => value !== undefined)
      )
      const index = mocks.prompts.findIndex((prompt) => prompt.id === id)
      mocks.prompts[index] = { ...mocks.prompts[index], ...defined }
    }
  }
}))

vi.mock("../..", () => ({
  deletePromptByIdFB: vi.fn(),
  getAllPromptsFB: vi.fn(async () => []),
  getPromptByIdFB: mocks.getPromptByIdFB,
  restorePromptSnapshotFB: vi.fn(),
  savePromptFB: mocks.savePromptFB,
  updatePromptFB: mocks.updatePromptFB
}))

const serverPrompt = {
  id: 6,
  uuid: "00000000-0000-4000-8000-000000000006",
  name: "Socratic tutor",
  author: "Robert",
  details: "Teaching aid",
  system_prompt: "Ask guiding questions.",
  user_prompt: null,
  keywords: ["teaching"]
}

describe("upsertServerLibraryPromptCopy", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.prompts = []
    mocks.closed = false
  })

  it("saves a system prompt copy linked to the server library prompt", async () => {
    const copy = await upsertServerLibraryPromptCopy(serverPrompt)

    expect(copy).toMatchObject({
      title: "Socratic tutor",
      content: "Ask guiding questions.",
      is_system: true,
      system_prompt: "Ask guiding questions.",
      author: "Robert",
      details: "Teaching aid",
      keywords: ["teaching"],
      serverLibraryId: 6,
      serverLibraryUuid: serverPrompt.uuid,
      syncStatus: "local"
    })
    expect(mocks.addPrompt).toHaveBeenCalledWith(copy)
    expect(mocks.savePromptFB).toHaveBeenCalledWith(copy)
  })

  it("saves a prompt without system text as a quick prompt", async () => {
    const copy = await upsertServerLibraryPromptCopy({
      ...serverPrompt,
      system_prompt: "   ",
      user_prompt: "Rewrite this plainly"
    })

    expect(copy).toMatchObject({
      content: "Rewrite this plainly",
      is_system: false,
      user_prompt: "Rewrite this plainly"
    })
  })

  it("refreshes the linked copy instead of creating a second one", async () => {
    mocks.prompts = [
      {
        id: "copy-6",
        ownerId: "recipe-owner:sha256:" + "a".repeat(64),
        title: "Old name",
        name: "Old name",
        content: "Old text",
        is_system: true,
        favorite: true,
        createdAt: 1,
        serverLibraryId: 6,
        serverLibraryUuid: serverPrompt.uuid
      }
    ]

    const copy = await upsertServerLibraryPromptCopy(serverPrompt)

    expect(mocks.addPrompt).not.toHaveBeenCalled()
    expect(mocks.updatePrompt).toHaveBeenCalledWith(
      "copy-6",
      expect.objectContaining({
        title: "Socratic tutor",
        content: "Ask guiding questions.",
        is_system: true
      })
    )
    expect(copy).toMatchObject({
      id: "copy-6",
      title: "Socratic tutor",
      favorite: true,
      serverLibraryUuid: serverPrompt.uuid
    })
  })

  it("returns a Firefox fallback row for the captured current owner", async () => {
    mocks.closed = true
    const row = { id: "owned", title: "Private", ownerId: "recipe-owner:sha256:" + "a".repeat(64) }
    mocks.getPromptByIdFB.mockResolvedValue(row)
    await expect(getPromptById("owned")).resolves.toEqual(row)
  })

  it("hides a foreign Firefox fallback row", async () => {
    mocks.closed = true
    mocks.getPromptByIdFB.mockResolvedValue({ id: "foreign", ownerId: "recipe-owner:sha256:" + "b".repeat(64) })
    await expect(getPromptById("foreign")).resolves.toBeUndefined()
  })

  it("rejects an empty id without reading the Firefox fallback", async () => {
    await expect(getPromptById(" ")).resolves.toBeNull()
    expect(mocks.getPromptByIdFB).not.toHaveBeenCalled()
  })
})
