import { useQuery } from "@tanstack/react-query"
import { getAllPrompts, getDeletedPrompts, getSharedLocalPrompts } from "@/db/dexie/helpers"
import { isPromptOwnedBy } from "@/services/prompt-local-owner"
import { usePromptLocalAuthority } from "./usePromptLocalAuthority"

/** Each account has a separate query cache, including palette/chat consumers. */
export function usePromptInventory(key: string, options: { deleted?: boolean; enabled?: boolean; staleTime?: number } = {}) {
  const authority = usePromptLocalAuthority()
  const query = useQuery({
    queryKey: [key, authority.ownerId, authority.ready ? "resolved" : "resolving"],
    queryFn: () => options.deleted ? getDeletedPrompts(authority.ownerId) : getAllPrompts(authority.ownerId),
    enabled: authority.ready && options.enabled !== false,
    staleTime: options.staleTime
  })
  const legacy = useQuery({
    queryKey: [key, "shared-local-legacy"],
    queryFn: () => getSharedLocalPrompts(options.deleted),
    enabled: !authority.ready && options.enabled !== false,
    staleTime: options.staleTime
  })
  return authority.ready
    ? { ...query, data: query.data?.filter(prompt => isPromptOwnedBy(prompt, authority.ownerId)) }
    : { ...legacy, data: legacy.data?.filter(prompt => prompt.ownerId == null && isPromptOwnedBy(prompt, null)) }
}
