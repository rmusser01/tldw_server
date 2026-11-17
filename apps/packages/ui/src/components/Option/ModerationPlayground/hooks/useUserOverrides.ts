import React from "react"
import { useQuery } from "@tanstack/react-query"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

import { useServerOnline } from "@/hooks/useServerOnline"
import {
  deleteUserOverride,
  getUserOverride,
  listUserOverrides,
  setUserOverride,
  type ModerationOverrideRule,
  type ModerationUserOverride
} from "@/services/moderation"
import {
  areRulesEquivalent,
  buildOverridePayload,
  createRuleId,
  getErrorStatus,
  isEqualJson,
  normalizeCategories,
  normalizeOverrideForCompare,
  normalizeOverrideRules,
  PRESET_PROFILES
} from "../moderation-utils"

export interface UserOverridesState {
  draft: ModerationUserOverride
  setDraft: React.Dispatch<React.SetStateAction<ModerationUserOverride>>
  baseline: ModerationUserOverride | null
  loaded: boolean
  loading: boolean
  userIdError: string | null
  isDirty: boolean
  rules: ModerationOverrideRule[]
  bannedRules: ModerationOverrideRule[]
  notifyRules: ModerationOverrideRule[]
  overridesQuery: ReturnType<typeof useQuery>
  updateDraft: (partial: Partial<ModerationUserOverride>) => void
  reset: () => void
  save: () => Promise<void>
  remove: (userId?: string | null) => Promise<void>
  bulkDelete: (userIds: string[]) => Promise<string[]>
  addRule: (rule: Omit<ModerationOverrideRule, "id">) => boolean
  removeRule: (ruleId: string) => void
  applyPreset: (key: string) => Promise<void>
}

export function useUserOverrides(activeUserId: string | null): UserOverridesState {
  const online = useServerOnline()
  const lifetime = React.useRef({ active: true, revision: 0 })
  React.useEffect(() => {
    lifetime.current.active = true
    const retire = () => {
      lifetime.current.active = false
      lifetime.current.revision += 1
    }
    const stop = watchChatAccountChanges((invalidated) => {
      if (invalidated) retire()
    })
    return () => {
      retire()
      stop()
    }
  }, [])

  const [draft, setDraft] = React.useState<ModerationUserOverride>({})
  const [baseline, setBaseline] = React.useState<ModerationUserOverride | null>(null)
  const [loaded, setLoaded] = React.useState(false)
  const [loading, setLoading] = React.useState(false)
  const [userIdError, setUserIdError] = React.useState<string | null>(null)

  const overridesQuery = useQuery({
    queryKey: ["moderation-overrides"],
    queryFn: listUserOverrides,
    enabled: online
  })

  // Load override when activeUserId changes
  React.useEffect(() => {
    if (!activeUserId) {
      setDraft({})
      setLoaded(false)
      setUserIdError(null)
      setBaseline({})
      return
    }
    let cancelled = false
    const load = async () => {
      setLoading(true)
      setUserIdError(null)
      try {
        const result = await getUserOverride(activeUserId)
        if (cancelled) return
        const data = result.override ?? {}
        if (!result.exists) {
          setDraft({})
          setLoaded(false)
          setUserIdError(`No override found for "${activeUserId}". You can create a new one.`)
          setBaseline({})
          return
        }
        const normalizedCategories =
          typeof data.categories_enabled === "undefined"
            ? undefined
            : normalizeCategories(data.categories_enabled)
        const normalized: ModerationUserOverride = {
          enabled: data.enabled,
          input_enabled: data.input_enabled,
          output_enabled: data.output_enabled,
          input_action: data.input_action,
          output_action: data.output_action,
          redact_replacement: data.redact_replacement,
          categories_enabled: normalizedCategories,
          rules: normalizeOverrideRules(data.rules)
        }
        setDraft(normalized)
        setLoaded(true)
        setUserIdError(null)
        setBaseline(normalized)
      } catch (err: unknown) {
        if (cancelled) return
        const status = getErrorStatus(err)
        if (status === 404) {
          setDraft({})
          setLoaded(false)
          setUserIdError(`No override found for "${activeUserId}". You can create a new one.`)
          setBaseline({})
        } else {
          console.error("[useUserOverrides] Failed to load override:", err)
          setUserIdError(`Failed to load override for "${activeUserId}". Please try again.`)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [activeUserId])

  const normalizedDraft = normalizeOverrideForCompare(draft)
  const normalizedBaseline = normalizeOverrideForCompare(baseline ?? {})
  const isDirty = Boolean(activeUserId) && !isEqualJson(normalizedDraft, normalizedBaseline)

  const rules = React.useMemo(() => normalizeOverrideRules(draft.rules), [draft.rules])
  const bannedRules = React.useMemo(() => rules.filter((r) => r.action === "block"), [rules])
  const notifyRules = React.useMemo(() => rules.filter((r) => r.action === "warn"), [rules])

  const updateDraft = React.useCallback((partial: Partial<ModerationUserOverride>) => {
    setDraft((prev) => ({ ...prev, ...partial }))
  }, [])

  const reset = React.useCallback(() => {
    const base = baseline ?? {}
    const normalized: ModerationUserOverride = {
      ...base,
      categories_enabled:
        base.categories_enabled !== undefined
          ? normalizeCategories(base.categories_enabled)
          : undefined,
      rules: normalizeOverrideRules(base.rules)
    }
    setDraft(normalized)
  }, [baseline])

  const save = React.useCallback(async () => {
    if (!activeUserId) throw new Error("No active user")
    const payload = buildOverridePayload(draft)
    await setUserOverride(activeUserId, payload)
    setLoaded(true)
    setBaseline(normalizeOverrideForCompare(draft))
    await overridesQuery.refetch()
  }, [activeUserId, draft, overridesQuery])

  const remove = React.useCallback(async (userId?: string | null) => {
    const targetId = userId || activeUserId
    if (!targetId) return
    await deleteUserOverride(targetId)
    if (targetId === activeUserId) {
      setDraft({})
      setLoaded(false)
      setUserIdError(null)
      setBaseline({})
    }
    await overridesQuery.refetch()
  }, [activeUserId, overridesQuery])

  const bulkDelete = React.useCallback(async (userIds: string[]) => {
    if (!userIds.length) return []
    const owner = lifetime.current.revision
    const assertCurrent = () => {
      if (!lifetime.current.active || owner !== lifetime.current.revision) {
        throw new DOMException("Moderation workspace retired", "AbortError")
      }
    }
    const failed: string[] = []
    for (const userId of userIds) {
      assertCurrent()
      try {
        await deleteUserOverride(userId)
      } catch {
        failed.push(userId)
      }
    }
    assertCurrent()
    if (activeUserId && userIds.includes(activeUserId)) {
      setDraft({})
      setLoaded(false)
      setUserIdError(null)
      setBaseline({})
    }
    await overridesQuery.refetch()
    assertCurrent()
    return failed
  }, [activeUserId, overridesQuery])

  const addRule = React.useCallback((rule: Omit<ModerationOverrideRule, "id">) => {
    const nextRule: ModerationOverrideRule = { ...rule, id: createRuleId() }
    const currentRules = normalizeOverrideRules(draft.rules)
    if (currentRules.some((existing) => areRulesEquivalent(existing, nextRule))) {
      return false
    }
    setDraft((prev) => ({
      ...prev,
      rules: [...normalizeOverrideRules(prev.rules), nextRule]
    }))
    return true
  }, [draft.rules])

  const removeRule = React.useCallback((ruleId: string) => {
    setDraft((prev) => ({
      ...prev,
      rules: normalizeOverrideRules(prev.rules).filter((r) => r.id !== ruleId)
    }))
  }, [])

  const applyPreset = React.useCallback(async (key: string) => {
    if (!activeUserId) throw new Error("No active user")
    const preset = PRESET_PROFILES[key]
    if (!preset) throw new Error(`Unknown preset: ${key}`)
    const payload = buildOverridePayload(preset.payload)
    await setUserOverride(activeUserId, payload)
    const normalizedPayload: ModerationUserOverride = {
      ...payload,
      categories_enabled:
        payload.categories_enabled !== undefined
          ? normalizeCategories(payload.categories_enabled)
          : undefined
    }
    setDraft(normalizedPayload)
    setLoaded(true)
    setUserIdError(null)
    setBaseline(normalizeOverrideForCompare(normalizedPayload))
    await overridesQuery.refetch()
  }, [activeUserId, overridesQuery])

  return {
    draft,
    setDraft,
    baseline,
    loaded,
    loading,
    userIdError,
    isDirty,
    rules,
    bannedRules,
    notifyRules,
    overridesQuery,
    updateDraft,
    reset,
    save,
    remove,
    bulkDelete,
    addRule,
    removeRule,
    applyPreset
  }
}
