import { useMemo, useRef, useState } from "react"
import { Storage } from "@plasmohq/storage"
import { useStorage } from "@plasmohq/storage/hook"
import { resolvePresetMap, type PresetMap } from "@/components/Common/QuickIngest/presets"

const key = "quickIngestPresetConfigs"
export const presetDefaults = resolvePresetMap()

/** Keep typing responsive while extension storage commits changes asynchronously. */
export function useQuickIngestPresetSettings() {
  const [storage] = useState(() => new Storage())
  const [stored, , { isLoading, setRenderValue }] = useStorage<PresetMap>(
    { key, instance: storage },
    presetDefaults
  )
  const saved = useMemo(() => resolvePresetMap(stored), [stored])
  const [pending, setPending] = useState<PresetMap | null>(null)
  const latest = useRef<PresetMap | null>(null)
  const writes = useRef(Promise.resolve())

  function write(update: (map: PresetMap) => PresetMap) {
    const preview = update(latest.current ?? saved)
    latest.current = preview
    setPending(preview)

    // Apply each edit to the latest persisted map so a slow write doesn't erase later input.
    const commit = writes.current.then(async () => {
      const before = resolvePresetMap(await storage.get<PresetMap>(key))
      const after = update(before)
      await storage.set(key, after)
      setRenderValue(after)
      return before
    })
    writes.current = commit.then(
      () => undefined,
      () => undefined
    )
    return commit.finally(() => {
      if (latest.current === preview) {
        latest.current = null
        setPending(null)
      }
    })
  }

  return { presets: pending ?? saved, isLoading, write }
}
