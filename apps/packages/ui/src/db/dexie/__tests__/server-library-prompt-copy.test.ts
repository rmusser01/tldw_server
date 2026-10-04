import { beforeEach, describe, expect, it, vi } from "vitest"

import { upsertServerLibraryPromptCopy } from "../helpers"

const mocks = vi.hoisted(() => ({
  prompts: [] as Array<Record<string, unknown>>,
  addPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  savePromptFB: vi.fn(),
  updatePromptFB: vi.fn()
}))

vi.mock("../chat", () => ({
  PageAssistDatabase: class {
    async getAllPrompts() {
      return mocks.prompts.filter((prompt) => !prompt.deletedAt)
    }
    async getPromptById(id: string) {
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
  getPromptByIdFB: vi.fn(),
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
})
