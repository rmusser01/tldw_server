import { useState, useCallback } from "react"
import type { PromptSavedView } from "./prompt-workspace-types"
import type { TagMatchMode } from "./custom-prompts-utils"

const STORAGE_KEY = "tldw-prompt-filter-presets-v1"

export type FilterPreset = {
  id: string
  name: string
  typeFilter: string
  syncFilter: string
  usageFilter: "all" | "used" | "unused"
  tagFilter: string[]
  tagMatchMode: TagMatchMode
  savedView: PromptSavedView
}

const loadPresets = (): FilterPreset[] => {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map((preset) => ({
      ...preset,
      usageFilter:
        preset?.usageFilter === "used" || preset?.usageFilter === "unused"
          ? preset.usageFilter
          : "all"
    }))
  } catch {
    return []
  }
}

const persistPresets = (presets: FilterPreset[]) => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(presets))
  } catch {
    // Ignore
  }
}

export const useFilterPresets = () => {
  const [presets, setPresets] = useState<FilterPreset[]>(loadPresets)

  const savePreset = useCallback(
    (name: string, filters: Omit<FilterPreset, "id" | "name">) => {
      setPresets((prev) => {
        const next = [
          ...prev,
          { ...filters, id: crypto.randomUUID(), name },
        ]
        persistPresets(next)
        return next
      })
    },
    []
  )

  const deletePreset = useCallback((id: string) => {
    setPresets((prev) => {
      const next = prev.filter((p) => p.id !== id)
      persistPresets(next)
      return next
    })
  }, [])

  return { presets, savePreset, deletePreset }
}
