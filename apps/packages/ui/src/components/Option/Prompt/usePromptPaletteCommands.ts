import { useMemo, createElement } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { NotebookPen } from "lucide-react"
import { getAllPrompts } from "@/db/dexie/helpers"
import type { CommandItem } from "@/components/Common/CommandPalette"

/**
 * Returns CommandItem[] for prompts to inject into the global CommandPalette.
 * Shows recent/favorite prompts when no query, all matching prompts when queried.
 */
export function usePromptPaletteCommands(): CommandItem[] {
  const navigate = useNavigate()
  const { data: prompts } = useQuery({
    queryKey: ["promptPaletteAll"],
    queryFn: getAllPrompts,
    staleTime: 30_000,
  })

  return useMemo(() => {
    if (!prompts?.length) return []

    // Filter out deleted prompts
    const active = prompts.filter((p) => !p.deletedAt)

    const getLastUsed = (p: (typeof active)[number]): number =>
      ((p as Record<string, unknown>).lastUsedAt as number) ?? 0

    // Build a combined list: favorites first, then recently used, capped at 10
    const favorites = active
      .filter((p) => p.favorite)
      .sort((a, b) => getLastUsed(b) - getLastUsed(a))
      .slice(0, 5)

    const recent = active
      .filter((p) => getLastUsed(p) > 0 && !p.favorite)
      .sort((a, b) => getLastUsed(b) - getLastUsed(a))
      .slice(0, 5)

    const combined = [...favorites, ...recent]
    // Deduplicate
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
        ...(p.tags ?? []),
      ],
    }))
  }, [prompts, navigate])
}
