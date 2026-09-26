import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { DEFAULT_CHARACTER_STORAGE_KEY, defaultCharacterStorage } from "@/utils/default-character-preference"
import type { Character } from "@/types/character"

type OwnedDefault = { ownerKey: string; selection: Character | null; localOnly?: boolean }
const prefix = `${DEFAULT_CHARACTER_STORAGE_KEY}:owner:`
// An empty, owner-stamped read also proves that this key finished hydrating.
const ownedStorage = {
  get: async (key: string) => {
    const saved = await defaultCharacterStorage.get<OwnedDefault>(key)
    const ownerKey = key.slice(prefix.length)
    return saved?.ownerKey === ownerKey ? saved : { ownerKey, selection: null }
  },
  set: (key: string, value: OwnedDefault) => defaultCharacterStorage.set(key, value),
  remove: (key: string) => defaultCharacterStorage.remove(key),
  watch: (callbacks: Parameters<typeof defaultCharacterStorage.watch>[0]) => defaultCharacterStorage.watch(callbacks),
  unwatch: (callbacks: Parameters<typeof defaultCharacterStorage.unwatch>[0]) => defaultCharacterStorage.unwatch(callbacks)
} as typeof defaultCharacterStorage

/** Preserve per-device defaults only for the account that selected them. */
export function useDefaultCharacterSelection() {
  const queryClient = useQueryClient()
  const { ownerKey, isCurrent } = useChatDraftOwner(() => {})
  const [record, setRecord, meta] = useStorage<OwnedDefault | null>(
    { key: `${prefix}${ownerKey ?? "unresolved"}`, instance: ownedStorage }, null
  )
  const owned = isCurrent() && record?.ownerKey === ownerKey ? record : null
  React.useEffect(() => { void defaultCharacterStorage.remove(DEFAULT_CHARACTER_STORAGE_KEY).catch(() => {}) }, [])
  const assertCurrent = React.useCallback(() => {
    if (!ownerKey || !isCurrent()) throw new Error("The default Character account changed. Try again after account verification.")
  }, [ownerKey, isCurrent])
  const captureScope = React.useCallback(async () => {
    assertCurrent()
    const scope = await resolveServicePromptScope()
    assertCurrent()
    return { config: scope.config, userId: scope.userId }
  }, [assertCurrent])
  const { data } = useQuery({
    queryKey: ["tldw:defaultCharacterPreference", ownerKey],
    enabled: Boolean(ownerKey),
    queryFn: async () => {
      const requestScope = await captureScope()
      const defaultCharacterId = await tldwClient.getDefaultCharacterPreference({ requestScope })
      assertCurrent()
      return { ownerKey, defaultCharacterId }
    },
    staleTime: 60 * 1000,
    throwOnError: false
  })
  const setSelection = React.useCallback(async (selection: Character | null, options: { localOnly?: boolean } = {}) => {
    assertCurrent()
    await setRecord({ ownerKey: ownerKey!, selection, localOnly: Boolean(options.localOnly) })
    assertCurrent()
  }, [assertCurrent, setRecord, ownerKey])
  const writePreference = React.useCallback(async (id: string | null) => {
    const requestScope = await captureScope()
    const result = await tldwClient.setDefaultCharacterPreference(id, { requestScope })
    assertCurrent()
    await queryClient.cancelQueries({ queryKey: ["tldw:defaultCharacterPreference", ownerKey], exact: true })
    assertCurrent()
    queryClient.setQueryData(["tldw:defaultCharacterPreference", ownerKey], { ownerKey, defaultCharacterId: id })
    return result
  }, [captureScope, assertCurrent, queryClient, ownerKey])
  return [owned?.selection ?? null, setSelection, {
    isLoading: !owned || Boolean(meta?.isLoading),
    preference: owned && !meta?.isLoading && data?.ownerKey === ownerKey && !owned.localOnly ? data : undefined,
    writePreference
  }] as const
}
