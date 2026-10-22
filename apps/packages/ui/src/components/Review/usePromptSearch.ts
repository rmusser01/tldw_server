import React from "react"
import { usePromptLocalAuthority } from "@/hooks/usePromptLocalAuthority"

import { getAllPrompts } from "@/db/dexie/helpers"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type PromptSearchResult = {
  id?: string
  title: string
  content: string
}

const dedupePromptResults = (
  items: PromptSearchResult[]
): PromptSearchResult[] => {
  const seen = new Set<string>()
  return items.filter((item) => {
    const key = `${item.title}:${item.content.slice(0, 64)}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export const usePromptSearch = () => {
  const authority = usePromptLocalAuthority()
  const scopeKey = JSON.stringify([authority.boundaryKey, authority.ownerId, authority.ready])
  const currentScope = React.useRef(scopeKey)
  currentScope.current = scopeKey
  const sequence = React.useRef(0)
  React.useEffect(() => () => { currentScope.current = "retired" }, [])
  const [query, setQuery] = React.useState("")
  const [results, setResults] = React.useState<PromptSearchResult[]>([])
  const [loading, setLoading] = React.useState(false)
  const [resultScope, setResultScope] = React.useState(scopeKey)
  const [includeLocal, setIncludeLocal] = React.useState(true)
  const [includeServer, setIncludeServer] = React.useState(true)

  const clearResults = React.useCallback(() => {
    setResults([])
  }, [])

  const search = React.useCallback(
    async (q: string) => {
      if (!authority.ready) return
      const request = ++sequence.current
      const isCurrent = () => currentScope.current === scopeKey && sequence.current === request
      const trimmed = q.trim()
      if (!trimmed) {
        setResults([])
        return
      }

      setLoading(true)
      try {
        let merged: PromptSearchResult[] = []
        if (includeLocal) {
          const locals = await getAllPrompts(authority.ownerId)
          if (!isCurrent()) return
          const filtered = (locals || [])
            .filter(
              (prompt) =>
                prompt.title?.toLowerCase().includes(trimmed.toLowerCase()) ||
                prompt.content?.toLowerCase().includes(trimmed.toLowerCase())
            )
            .map((prompt) => ({
              id: prompt.id,
              title: prompt.title,
              content: prompt.content
            }))
          merged = merged.concat(filtered)
        }

        if (includeServer) {
          await tldwClient.initialize().catch(() => null)
          if (!isCurrent()) return
          const res = await tldwClient.searchPrompts(trimmed, { expectedOwnerId: authority.ownerId }).catch(() => [])
          if (!isCurrent()) return
          const list: any[] = Array.isArray(res)
            ? res
            : res?.results || res?.prompts || []
          merged = merged.concat(
            list.map((entry) => ({
              id: entry.id,
              title: String(entry.title || entry.name || "Untitled"),
              content: String(entry.content || entry.prompt || "")
            }))
          )
        }

        if (isCurrent()) {
          setResultScope(scopeKey)
          setResults(dedupePromptResults(merged).slice(0, 50))
        }
      } finally {
        if (isCurrent()) setLoading(false)
      }
    },
    [authority.ownerId, authority.ready, scopeKey, includeLocal, includeServer]
  )

  return {
    query,
    setQuery,
    results: authority.ready && resultScope === scopeKey ? results : [],
    loading: authority.ready && resultScope === scopeKey && loading,
    includeLocal,
    setIncludeLocal,
    includeServer,
    setIncludeServer,
    search,
    clearResults
  }
}
