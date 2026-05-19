import React from "react"

import { apiSend } from "@/services/api-send"
import type { ArchetypeSummary } from "@/types/archetype"

type UseArchetypeCatalogResult = {
  archetypes: ArchetypeSummary[]
  loading: boolean
  error: string | null
}

let cachedArchetypes: ArchetypeSummary[] | null = null

/**
 * Fetches the archetype catalog from `GET /api/v1/persona/archetypes`.
 * Results are cached in module scope so the request is made only once
 * across all consumers.
 */
export function useArchetypeCatalog(): UseArchetypeCatalogResult {
  const [archetypes, setArchetypes] = React.useState<ArchetypeSummary[]>(
    cachedArchetypes ?? []
  )
  const [loading, setLoading] = React.useState(cachedArchetypes === null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (cachedArchetypes !== null) {
      setArchetypes(cachedArchetypes)
      setLoading(false)
      return
    }

    let cancelled = false

    const fetchArchetypes = async () => {
      setLoading(true)
      setError(null)
      try {
        const path = "/api/v1/persona/archetypes" as const
        const res = await apiSend<ArchetypeSummary[]>({
          path,
          method: "GET"
        })
        if (cancelled) return
        if (res.ok && Array.isArray(res.data)) {
          cachedArchetypes = res.data
          setArchetypes(res.data)
        } else {
          setError(res.error ?? "Failed to load archetype catalog")
        }
      } catch {
        if (!cancelled) {
          setError("Failed to load archetype catalog")
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void fetchArchetypes()

    return () => {
      cancelled = true
    }
  }, [])

  return { archetypes, loading, error }
}
