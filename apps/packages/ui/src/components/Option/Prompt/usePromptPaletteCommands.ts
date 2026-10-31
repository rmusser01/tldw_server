import { usePromptInventory } from "@/hooks/usePromptInventory"
import { useMemo, createElement } from "react"
import { useNavigate } from "react-router-dom"
import { NotebookPen } from "lucide-react"
import type { CommandItem } from "@/components/Common/CommandPalette"
import { matchesPromptSearchText } from "./custom-prompts-utils"
import { classifyPromptRecipe } from "./prompt-recipe-library"

/**
 * Returns CommandItem[] for prompts to inject into the global CommandPalette.
 *
 * - No query (or < 2 chars): shows recent/favorite prompts (static, ~10 items)
 * - Query >= 2 chars: searches the hydrated local inventory and opens local UUIDs
 */
export function usePromptPaletteCommands(
  query = "",
  enabled = true
): CommandItem[] {
  const navigate = useNavigate()
  const trimmedQuery = query.trim()
  const isSearching = enabled && trimmedQuery.length >= 2

  // Local prompts — always loaded (used for both static list and local search)
  const { data: prompts } = usePromptInventory("fetchAllPrompts", { staleTime: 30_000, enabled })

  return useMemo(() => {
    if (!enabled) {
      return []
    }
    const active = (prompts || []).filter(
      (p) => !p.deletedAt && classifyPromptRecipe(p).kind !== "quarantined_recipe"
    )

    // --- Static mode: no query → favorites + recent ---
    if (!isSearching) {
      const getLastUsed = (p: (typeof active)[number]): number => {
        const lastUsedAt = (p as { lastUsedAt?: unknown }).lastUsedAt
        return typeof lastUsedAt === "number" && Number.isFinite(lastUsedAt)
          ? lastUsedAt
          : 0
      }

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
        targetPath: `/prompts?edit=${p.id}`,
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

    // Search the same inventory as the workspace, independent of sync catalog.
    const queryLower = trimmedQuery.toLowerCase()

    // Local matches
    const localMatches = active.filter((p) =>
      matchesPromptSearchText(p, queryLower, (prompt) => prompt.keywords ?? prompt.tags ?? [])
    )

    // Build local result items
    return localMatches.slice(0, 15).map((p) => {
      return {
        id: `prompt-search-local-${p.id}`,
        label: p.name || p.title || "Untitled Prompt",
        description: "Local prompt",
        icon: createElement(NotebookPen, { className: "size-4" }),
        targetPath: `/prompts?prompt=${p.id}`,
        action: () => {
          navigate(`/prompts?prompt=${p.id}`)
        },
        category: "prompt" as const,
        keywords: [
          "prompt",
          query,
          ...(p.keywords ?? []),
          ...(p.tags ?? [])
        ]
      }
    })

  }, [enabled, prompts, isSearching, trimmedQuery, query, navigate])
}
