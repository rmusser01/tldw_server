import {
  ACTOR_SETTINGS_VERSION,
  ActorSettings,
  ActorAspect,
  createDefaultActorSettings,
  normalizeActorAspectKey
} from "@/types/actor"

import {
  createLocalRegistryBucket,
  withRecordLock,
  type LocalRegistryRecord
} from "./settings/local-bucket"
import {
  captureActorProfileTarget,
  type ActorSettingsTarget,
  type ActorProfileTarget
} from "@/utils/actor-settings-target"

const scenes = createLocalRegistryBucket<ActorSettings>({
  prefix: "actorSettings:owned:"
})
const profiles = createLocalRegistryBucket<ActorSettings>({
  prefix: "actorProfile:owned:"
})
export type ActorSceneRecord = LocalRegistryRecord<ActorSettings>
export type ActorSceneRead = {
  target: ActorSettingsTarget
  originalRecord: ActorSceneRecord | null
  settings: ActorSettings
  origin: "scene" | "profile" | "default"
  profileUnavailable: boolean
}

const isValidActorAspect = (value: any): value is ActorAspect => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false

  const {
    id,
    key,
    target,
    name,
    source,
    value: aspectValue,
    lorebookId,
    entryId
  } = value as {
    id: unknown
    key: unknown
    target: unknown
    name: unknown
    source: unknown
    value: unknown
    lorebookId?: unknown
    entryId?: unknown
  }

  const isString = (v: unknown): v is string => typeof v === "string"

  if (
    !isString(id) ||
    !isString(key) ||
    !isString(target) ||
    !isString(name) ||
    !isString(source) ||
    !isString(aspectValue)
  ) {
    return false
  }

  if (lorebookId !== undefined && !isString(lorebookId)) return false
  if (entryId !== undefined && !isString(entryId)) return false

  // Basic enum checks; if these ever expand, this guard can be relaxed.
  const validTargets = ["user", "char", "world"]
  const validSources = ["free", "lore"]

  if (!validTargets.includes(target) || !validSources.includes(source)) {
    return false
  }

  return true
}

const isValidActorSettings = (raw: any): raw is ActorSettings => {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return false
  }

  const isNumber = (v: unknown): v is number =>
    typeof v === "number" && Number.isFinite(v)
  const isBoolean = (v: unknown): v is boolean => typeof v === "boolean"
  const isString = (v: unknown): v is string => typeof v === "string"

  const {
    version,
    isEnabled,
    aspects,
    notes,
    notesGmOnly,
    chatPosition,
    chatDepth,
    chatRole,
    appendable,
    templateMode
  } = raw as Partial<ActorSettings>

  if (!isNumber(version)) return false
  if (!isBoolean(isEnabled)) return false
  if (
    !Array.isArray(aspects) ||
    !Array.from(aspects).every(isValidActorAspect)
  ) {
    return false
  }
  if (!isString(notes)) return false

  if (notesGmOnly !== undefined && !isBoolean(notesGmOnly)) {
    return false
  }

  if (!isString(chatPosition)) return false
  const validPositions = ["before", "after", "depth"]
  if (!validPositions.includes(chatPosition)) return false

  if (!isNumber(chatDepth)) return false

  if (!isString(chatRole)) return false
  const validRoles = ["system", "user", "assistant"]
  if (!validRoles.includes(chatRole)) return false

  if (appendable !== undefined && !isBoolean(appendable)) {
    return false
  }

  if (templateMode !== undefined) {
    if (!isString(templateMode)) return false
    const validTemplateModes = ["merge", "override", "ignore"]
    if (!validTemplateModes.includes(templateMode)) return false
  }

  return true
}

const migrateSettings = (raw: any): ActorSettings => {
  if (!raw || typeof raw !== "object") {
    return createDefaultActorSettings()
  }

  const version: number = Number(raw.version ?? 0)

  if (version < ACTOR_SETTINGS_VERSION) {
    // No legacy prompts to preserve; reset to new neutral defaults.
    return createDefaultActorSettings()
  }

  if (!isValidActorSettings(raw)) {
    console.warn(
      "Invalid Actor settings found in storage; resetting to defaults.",
      raw
    )
    return createDefaultActorSettings()
  }

  const settings = raw as ActorSettings
  let changed = false
  const aspects = (settings.aspects || []).map((aspect) => {
    const normalizedKey = normalizeActorAspectKey({
      raw: aspect.key,
      target: aspect.target,
      fallback: aspect.key || undefined
    })
    if (!normalizedKey || normalizedKey === aspect.key) {
      return aspect
    }
    changed = true
    return {
      ...aspect,
      key: normalizedKey
    }
  })

  if (!changed) {
    if (typeof settings.appendable === "boolean") {
      return settings
    }
    return {
      ...settings,
      appendable: settings.appendable ?? false
    }
  }

  return {
    ...settings,
    aspects,
    appendable: settings.appendable ?? false
  }
}

const isValidSceneRecord = (raw: unknown): raw is ActorSceneRecord => {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return false
  const record = raw as ActorSceneRecord
  return (
    isValidActorSettings(record.value) &&
    record.value.version === ACTOR_SETTINGS_VERSION &&
    typeof record.updatedAt === "number" &&
    Number.isFinite(record.updatedAt) &&
    Number.isFinite(new Date(record.updatedAt).getTime())
  )
}

const verifyOriginalLocalProfile = async (
  target: ActorProfileTarget & { localProfileId?: string }
) => {
  if (!target.isCurrent()) throw new Error("actor_settings_scope_changed")
  if (target.localProfileId !== undefined) {
    const { ensureLocalProfileId } = await import(
      "@/db/dexie/history-selection"
    )
    await ensureLocalProfileId(target.localProfileId)
    if (!target.isCurrent()) throw new Error("actor_settings_scope_changed")
  }
}

const readOwnedRecord = async (
  bucket: typeof scenes,
  target: ActorProfileTarget & { localProfileId?: string }
): Promise<ActorSceneRecord | null> => {
  if (!target.isCurrent()) throw new Error("actor_settings_scope_changed")
  if (target.localProfileId !== undefined)
    await verifyOriginalLocalProfile(target)
  const record = await bucket.get(target.key, { failClosed: true })
  if (target.localProfileId !== undefined) {
    await verifyOriginalLocalProfile(target)
    if (!bucket.isDurable()) throw new Error("actor_settings_unavailable")
  }
  if (!target.isCurrent()) throw new Error("actor_settings_scope_changed")
  if (record !== null && !isValidSceneRecord(record))
    throw new Error("actor_settings_invalid")
  return record ? structuredClone(record) : null
}

export const getActorSceneForChat = async (params: {
  target: ActorSettingsTarget
  characterId?: string | number | null
  validate_lease?: () => boolean
}): Promise<ActorSceneRead> => {
  const { target, characterId } = params
  try {
    const readTarget = {
      key: target.key,
      localProfileId: target.localProfileId,
      isCurrent: () => target.isCurrent() && params.validate_lease?.() !== false
    }
    const originalRecord = await readOwnedRecord(scenes, readTarget)
    if (originalRecord)
      return {
        target,
        originalRecord,
        settings: migrateSettings(structuredClone(originalRecord.value)),
        origin: "scene",
        profileUnavailable: false
      }
    const empty = {
      target,
      originalRecord: null,
      settings: createDefaultActorSettings(),
      origin: "default" as const,
      profileUnavailable: false
    }
    if (characterId == null) return empty
    const profileTarget = captureActorProfileTarget(
      target.accountKey,
      characterId,
      () => readTarget.isCurrent() && target.profileIsCurrent()
    )
    if (!profileTarget) return { ...empty, profileUnavailable: true }
    let inherited: ActorSceneRead = empty
    try {
      const profile = await readOwnedRecord(profiles, profileTarget)
      if (profile)
        inherited = {
          ...empty,
          settings: migrateSettings(structuredClone(profile.value)),
          origin: "profile"
        }
    } catch {
      if (!readTarget.isCurrent())
        throw new Error("actor_settings_scope_changed")
      inherited = { ...empty, profileUnavailable: true }
    }
    if (target.localProfileId !== undefined) {
      await verifyOriginalLocalProfile(readTarget)
      if (!scenes.isDurable()) throw new Error("actor_settings_unavailable")
    }
    if (!readTarget.isCurrent()) throw new Error("actor_settings_scope_changed")
    return inherited
  } catch (error) {
    console.error("Failed to load owned Actor scene", error)
    throw new Error("actor_settings_unavailable")
  }
}

/** Copy only an original saved fresh scene after its caller accepted the destination bind. */
export const copyActorSceneToBoundChat = async (
  source: ActorSceneRead,
  destination: ActorSettingsTarget,
  validate_lease: () => boolean = () => true
): Promise<boolean> => {
  const fresh = source.target.address
  const boundAddress = destination.address
  const account = source.target.account
  const current = () =>
    account?.isCurrent() === true && destination.isCurrent() && validate_lease()
  if (
    fresh?.kind !== "fresh" ||
    fresh.temporary ||
    !account ||
    !source.target.accountKey ||
    source.origin !== "scene" ||
    !source.originalRecord ||
    !isValidSceneRecord(source.originalRecord) ||
    !boundAddress ||
    boundAddress.kind === "fresh" ||
    destination.key === source.target.key ||
    (boundAddress.kind === "native" &&
      (boundAddress.scope.type !== fresh.scope.type ||
        (boundAddress.scope.type === "workspace" &&
          fresh.scope.type === "workspace" &&
          boundAddress.scope.workspaceId !== fresh.scope.workspaceId))) ||
    (destination.localProfileId === undefined &&
      destination.accountKey !== source.target.accountKey) ||
    !current() ||
    typeof navigator === "undefined" ||
    !navigator.locks?.request
  )
    return false
  const original = structuredClone(source.originalRecord)
  try {
    return await withRecordLock(
      scenes.buildKey(source.target.key),
      async () => {
        const retained = await readOwnedRecord(scenes, {
          key: source.target.key,
          isCurrent: current
        })
        if (JSON.stringify(retained) !== JSON.stringify(original)) return false
        const bound = await readOwnedRecord(scenes, {
          ...destination,
          isCurrent: current
        })
        if (
          bound !== null &&
          JSON.stringify(bound) !== JSON.stringify(original)
        )
          return false
        // The same conditional write also certifies an exact prior copy under its destination lock.
        const saved = await scenes.set(
          destination.key,
          original.value,
          original.updatedAt,
          bound,
          current
        )
        if (saved && destination.localProfileId !== undefined) {
          await verifyOriginalLocalProfile({
            ...destination,
            isCurrent: current
          })
          if (!scenes.isDurable()) return false
        }
        return saved && current()
      }
    )
  } catch {
    return false
  }
}

export const getActorSettingsForChat = async (params: {
  historyId: string | null
  serverChatId: string | null
  target?: ActorSettingsTarget
}): Promise<ActorSettings> => {
  if (!params.target) throw new Error("actor_settings_unavailable")
  return (await getActorSceneForChat({ target: params.target })).settings
}

export const saveActorSettingsForChat = async (params: {
  historyId: string | null
  serverChatId: string | null
  settings: ActorSettings
  target?: ActorSettingsTarget
  expectedRecord?: ActorSceneRecord | null
  validate_lease?: () => boolean
}): Promise<boolean> => {
  try {
    if (
      !isValidActorSettings(params.settings) ||
      params.settings.version !== ACTOR_SETTINGS_VERSION
    )
      return false
    const { target, expectedRecord } = params
    if (
      !target ||
      expectedRecord === undefined ||
      !target.isCurrent() ||
      params.validate_lease?.() === false
    )
      return false
    if (expectedRecord !== null && !isValidSceneRecord(expectedRecord))
      return false
    const proposal = structuredClone(params.settings)
    const original =
      expectedRecord === null ? null : structuredClone(expectedRecord)
    const updatedAt = Math.max(Date.now(), (original?.updatedAt ?? -1) + 1)
    if (!Number.isFinite(new Date(updatedAt).getTime())) return false
    if (target.localProfileId !== undefined)
      await verifyOriginalLocalProfile({
        key: target.key,
        localProfileId: target.localProfileId,
        isCurrent: () =>
          target.isCurrent() && params.validate_lease?.() !== false
      })
    const saved = await scenes.set(
      target.key,
      proposal,
      updatedAt,
      original,
      () => target.isCurrent() && params.validate_lease?.() !== false
    )
    if (saved && target.localProfileId !== undefined) {
      await verifyOriginalLocalProfile({
        key: target.key,
        localProfileId: target.localProfileId,
        isCurrent: () =>
          target.isCurrent() && params.validate_lease?.() !== false
      })
      if (!scenes.isDurable()) return false
    }
    return saved && target.isCurrent() && params.validate_lease?.() !== false
  } catch (error) {
    console.error("Failed to save Actor settings", error)
    return false
  }
}

export const getActorProfileForCharacter = async (
  characterId: string | number | null | undefined,
  options?: {
    throwOnError?: boolean
    validate_lease?: () => boolean
    target?: ActorProfileTarget
  }
): Promise<ActorSettings | null> => {
  if (!characterId) return null
  try {
    const target = options?.target
    if (!target) throw new Error("actor_settings_unavailable")
    const record = await readOwnedRecord(profiles, {
      key: target.key,
      isCurrent: () =>
        target.isCurrent() && options.validate_lease?.() !== false
    })
    return record ? migrateSettings(structuredClone(record.value)) : null
  } catch (error) {
    console.error("Failed to load Actor profile for character", error)
    throw new Error("actor_settings_unavailable")
  }
}

export const saveActorProfileForCharacter = async (params: {
  characterId: string | number
  settings: ActorSettings
  target?: ActorProfileTarget
  validate_lease?: () => boolean
}): Promise<boolean> => {
  try {
    if (params.validate_lease?.() === false) return false
    if (
      !isValidActorSettings(params.settings) ||
      params.settings.version !== ACTOR_SETTINGS_VERSION
    )
      return false
    const captured = params.target
    if (!captured) return false
    const target = {
      key: captured.key,
      isCurrent: () =>
        captured.isCurrent() && params.validate_lease?.() !== false
    }
    const proposal = structuredClone(params.settings)
    const original = await readOwnedRecord(profiles, target)
    const updatedAt = Math.max(Date.now(), (original?.updatedAt ?? -1) + 1)
    if (!Number.isFinite(new Date(updatedAt).getTime())) return false
    return await profiles.set(
      target.key,
      proposal,
      updatedAt,
      original,
      target.isCurrent
    )
  } catch (error) {
    console.error("Failed to save Actor profile for character", error)
    return false
  }
}

export const getActorSettingsForChatWithCharacterFallback = async (params: {
  historyId: string | null
  serverChatId: string | null
  characterId?: string | number | null
  target?: ActorSettingsTarget
}): Promise<ActorSettings> => {
  if (!params.target) throw new Error("actor_settings_unavailable")
  const read = await getActorSceneForChat({
    target: params.target,
    characterId: params.characterId
  })
  if (read.profileUnavailable) throw new Error("actor_settings_unavailable")
  return read.settings
}
