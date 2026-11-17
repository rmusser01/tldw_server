import { defineSetting, getSetting, setSetting } from "@/services/settings/registry"
import { resolvePromptOwnerId } from "@/services/prompt-local-owner"

export type PromptStudioDefaults = {
  defaultProjectId?: number | null
  autoSyncWorkspacePrompts?: boolean
  executeProvider?: string
  executeModel?: string
  executeTemperature?: number
  executeMaxTokens?: number
  evalModelName?: string
  evalTemperature?: number
  evalMaxTokens?: number
  pageSize?: number
  warnSeconds?: number
}

const STORAGE_KEY = "promptStudioDefaults"

const DEFAULTS: Required<PromptStudioDefaults> = {
  defaultProjectId: null,
  autoSyncWorkspacePrompts: true,
  executeProvider: "openai",
  executeModel: "gpt-3.5-turbo",
  executeTemperature: 0.2,
  executeMaxTokens: 256,
  evalModelName: "gpt-3.5-turbo",
  evalTemperature: 0.2,
  evalMaxTokens: 512,
  pageSize: 10,
  warnSeconds: 30
}

const PROMPT_STUDIO_DEFAULTS_SETTING = defineSetting(
  STORAGE_KEY,
  DEFAULTS,
  (value) => ({
    ...DEFAULTS,
    ...(value && typeof value === "object" ? value : {})
  })
)

/** Project IDs belong to the server/account; general editor preferences stay local. */
export async function getPromptStudioDefaultsForOwner(expectedOwnerId?: string | null): Promise<PromptStudioDefaults> {
  const ownerId = await resolvePromptOwnerId(expectedOwnerId)
  const general = await getSetting(PROMPT_STUDIO_DEFAULTS_SETTING)
  const scoped = ownerId ? await getSetting({
    ...PROMPT_STUDIO_DEFAULTS_SETTING,
    key: `${STORAGE_KEY}:${ownerId}`
  }) : null
  return { ...general, defaultProjectId: scoped?.defaultProjectId ?? null }
}

export async function getPromptStudioDefaults(): Promise<PromptStudioDefaults> {
  return await getPromptStudioDefaultsForOwner()
}

export async function setPromptStudioDefaults(
  updates: Partial<PromptStudioDefaults>, expectedOwnerId?: string | null
): Promise<PromptStudioDefaults> {
  const ownerId = await resolvePromptOwnerId(expectedOwnerId)
  const { defaultProjectId, ...preferences } = updates
  if (!ownerId && defaultProjectId != null) throw new Error("Prompt project owner is unavailable")
  const general = await getSetting(PROMPT_STUDIO_DEFAULTS_SETTING)
  if (Object.keys(preferences).length)
    await setSetting(PROMPT_STUDIO_DEFAULTS_SETTING, { ...general, ...preferences })
  if (ownerId && Object.prototype.hasOwnProperty.call(updates, "defaultProjectId")) {
    const setting = { ...PROMPT_STUDIO_DEFAULTS_SETTING, key: `${STORAGE_KEY}:${ownerId}` }
    const current = await getSetting(setting)
    await setSetting(setting, { ...current, defaultProjectId: defaultProjectId ?? null })
  }
  return await getPromptStudioDefaultsForOwner(ownerId)
}
