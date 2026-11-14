import { beforeEach, expect, it, vi } from "vitest"

const state = vi.hoisted(() => ({
  owner: "recipe-owner:sha256:" + "a".repeat(64),
  settings: new Map<string, Record<string, unknown>>()
}))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => state.owner ? ({ ownerId: state.owner }) : null
}))
vi.mock("@/services/settings/registry", () => ({
  defineSetting: (key: string, defaults: unknown, normalize: (value: unknown) => unknown) => ({ key, defaults, normalize }),
  getSetting: async (setting: any) => setting.normalize(state.settings.get(setting.key) ?? setting.defaults),
  setSetting: async (setting: any, value: Record<string, unknown>) => { state.settings.set(setting.key, value) }
}))
import { getPromptStudioDefaults, setPromptStudioDefaults } from "@/services/prompt-studio-settings"

beforeEach(() => {
  state.owner = "recipe-owner:sha256:" + "a".repeat(64)
  state.settings.clear()
  state.settings.set("promptStudioDefaults", { defaultProjectId: 9, autoSyncWorkspacePrompts: false })
})

it("does not assign an ambiguous device-wide project ID to the current account", async () => {
  expect(await getPromptStudioDefaults()).toMatchObject({ defaultProjectId: null, autoSyncWorkspacePrompts: false })
  expect(state.settings.get("promptStudioDefaults")?.defaultProjectId).toBe(9)
})

it("keeps default project IDs separate while preserving general preferences", async () => {
  const alice = state.owner
  await setPromptStudioDefaults({ defaultProjectId: 1 })
  state.owner = "recipe-owner:sha256:" + "b".repeat(64)
  expect((await getPromptStudioDefaults()).defaultProjectId).toBeNull()
  await setPromptStudioDefaults({ defaultProjectId: 2 })
  state.owner = alice
  expect(await getPromptStudioDefaults()).toMatchObject({ defaultProjectId: 1, autoSyncWorkspacePrompts: false })
})
it("saves ordinary offline preferences with a null project without claiming historical defaults", async () => {
  state.owner = ""
  await expect(setPromptStudioDefaults({ defaultProjectId: null, pageSize: 20 })).resolves.toMatchObject({ defaultProjectId: null, pageSize: 20 })
  expect(state.settings.get("promptStudioDefaults")?.defaultProjectId).toBe(9)
})
it("rejects a non-null offline project before changing general preferences", async () => {
  state.owner = ""
  const before = structuredClone(state.settings.get("promptStudioDefaults"))
  await expect(setPromptStudioDefaults({ defaultProjectId: 17, pageSize: 20 })).rejects.toThrow(/owner/)
  expect(state.settings.get("promptStudioDefaults")).toEqual(before)
})
