import { useQuery } from "@tanstack/react-query"

import { apiSend } from "@/services/api-send"
import type { ArchetypePreview } from "@/types/archetype"

type UseArchetypePreviewResult = {
  preview: ArchetypePreview | null
  loading: boolean
  error: string | null
}

/**
 * Fetches a full archetype preview from
 * `GET /api/v1/persona/archetypes/{key}/preview`.
 *
 * Skips the fetch when `key` is null, and re-fetches whenever the key
 * changes.
 */
export function useArchetypePreview(
  key: string | null
): UseArchetypePreviewResult {
  const query = useQuery({
    queryKey: ["archetype-preview", key],
    enabled: Boolean(key),
    retry: false,
    queryFn: async () => {
      const encodedKey = encodeURIComponent(String(key))
      const path: `/api/v1/persona/archetypes/${string}/preview` =
        `/api/v1/persona/archetypes/${encodedKey}/preview`
      const res = await apiSend<ArchetypePreview>({
        path,
        method: "GET"
      })
      if (res.ok && res.data) {
        return res.data
      }
      throw new Error(res.error ?? "Failed to load archetype preview")
    }
  })

  return {
    preview: key ? query.data ?? null : null,
    loading: key ? query.isLoading : false,
    error: query.error instanceof Error ? query.error.message : null
  }
}
