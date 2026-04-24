import { useMemo, createElement } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Cloud, NotebookPen } from "lucide-react"
import { getAllPrompts } from "@/db/dexie/helpers"
import { searchPromptsServer, type PromptSearchField } from "@/services/prompts-api"
import { useServerOnline } from "@/hooks/useServerOnline"
import type { CommandItem } from "@/components/Common/CommandPalette"

const SEARCH_FIELDS: PromptSearchField[] = [
  "name",
  "author",
  "details",
  "system_prompt",
  "user_prompt",
  "keywords"
]

/**
 * Returns CommandItem[] for prompts to inject into the global CommandPalette.
 *
 * - No query (or < 2 chars): shows recent/favorite prompts (static, ~10 items)
 * - Query >= 2 chars: searches local + server prompts in parallel, returns
 *   unified deduplicated results with source labels
 */
export function usePromptPaletteCommands(
  query = "",
  enabled = true
): CommandItem[] {
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const trimmedQuery = query.trim()
  const isSearching = enabled && trimmedQuery.length >= 2

  // Local prompts — always loaded (used for both static list and local search)
  const { data: prompts } = useQuery({
    queryKey: ["promptPaletteAll"],
    queryFn: getAllPrompts,
    staleTime: 30_000,
    enabled
  })

  // Server search — only when query is long enough and online
  const { data: serverResults } = useQuery({
    queryKey: ["promptPaletteSearch", trimmedQuery],
    queryFn: () =>
      searchPromptsServer({
        searchQuery: trimmedQuery,
        searchFields: SEARCH_FIELDS,
        page: 1,
        resultsPerPage: 15,
        includeDeleted: false
      }),
    enabled: enabled && isSearching && isOnline,
    staleTime: 10_000
  })

  return useMemo(() => {
    if (!enabled) {
      return []
    }
    const active = (prompts || []).filter((p) => !p.deletedAt)

    // --- Static mode: no query → favorites + recent ---
    if (!isSearching) {
      const getLastUsed = (p: (typeof active)[number]): number =>
        typeof p.lastUsedAt === "number" && Number.isFinite(p.lastUsedAt)
          ? p.lastUsedAt
          : 0

      const favorites = active
        .filter((p) => p.favorite)
        .sort((a, b) => getLastUsed(b) - getLastUsed(a))
        .slice(0, 5)

      const recent = active
        .filter((p) => getLastUsed(p) > 0 && !p.favorite)
        .sort((a, b) => getLastUsed(b) - getLastUsed(a))
        .slice(0, 5)

      const combined = [...favorites, ...recent]
      const seen = new Set<string>()
      const unique = combined.filter((p) => {
        if (seen.has(p.id)) return false
        seen.add(p.id)
        return true
      })

      return unique.map((p) => ({
        id: `prompt-${p.id}`,
        label: p.name || p.title || "Untitled Prompt",
        description: (p.system_prompt || p.content || "").slice(0, 80),
        icon: createElement(NotebookPen, { className: "size-4" }),
        action: () => {
          navigate(`/prompts?edit=${p.id}`)
        },
        category: "prompt" as const,
        keywords: [
          "prompt",
          "template",
          ...(p.keywords ?? []),
          ...(p.tags ?? [])
        ]
      }))
    }

    // --- Search mode: query >= 2 chars → local + server results ---
    const queryLower = trimmedQuery.toLowerCase()

    // Local matches
    const localMatches = active.filter((p) => {
      const name = (p.name || p.title || "").toLowerCase()
      const content = (p.content || "").toLowerCase()
      const system = (p.system_prompt || "").toLowerCase()
      const user = (p.user_prompt || "").toLowerCase()
      const tags = (p.keywords || p.tags || []).join(" ").toLowerCase()
      return (
        name.includes(queryLower) ||
        content.includes(queryLower) ||
        system.includes(queryLower) ||
        user.includes(queryLower) ||
        tags.includes(queryLower)
      )
    })

    // Build local result items
    const localServerIds = new Set<number>()
    const results: CommandItem[] = localMatches.slice(0, 15).map((p) => {
      if (typeof p.serverId === "number") {
        localServerIds.add(p.serverId)
      }
      return {
        id: `prompt-search-local-${p.id}`,
        label: p.name || p.title || "Untitled Prompt",
        description: "Local prompt",
        icon: createElement(NotebookPen, { className: "size-4" }),
        action: () => {
          navigate(`/prompts?prompt=${p.id}`)
        },
        category: "prompt" as const,
        keywords: [
          "prompt",
          ...(p.keywords ?? []),
          ...(p.tags ?? [])
        ]
      }
    })

    // Server matches (deduplicated against local)
    const serverItems = serverResults?.items ?? []
    for (const item of serverItems) {
      // Skip if we already have this prompt locally
      if (localServerIds.has(item.id)) continue

      results.push({
        id: `prompt-search-server-${item.id}`,
        label: item.name || "Untitled Prompt",
        description: "Server prompt",
        icon: createElement(Cloud, { className: "size-4" }),
        action: () => {
          navigate(`/prompts?prompt=${item.id}&source=studio`)
        },
        category: "prompt" as const,
        keywords: [
          "prompt",
          "server",
          "studio",
          ...(item.keywords ?? [])
        ]
      })

      if (results.length >= 20) break
    }

    return results
  }, [enabled, prompts, isSearching, trimmedQuery, serverResults, navigate])
}
