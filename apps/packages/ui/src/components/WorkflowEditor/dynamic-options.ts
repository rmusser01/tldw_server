import { useEffect, useMemo, useState } from "react"
import type { ConfigFieldSchema } from "@/types/workflow-editor"
import { tldwClient, tldwModels } from "@/services/tldw"
import { fetchTldwTtsModels } from "@/services/tldw/audio-models"
import { fetchTtsProviders } from "@/services/tldw/audio-providers"
import { fetchTldwVoices, fetchTldwVoiceCatalog } from "@/services/tldw/audio-voices"
import { fetchEmbeddingCollections } from "@/services/tldw/embedding-collections"
import { fetchFolders } from "@/services/folder-api"
import {
  listDatasets,
  listEvaluations,
  listRuns
} from "@/services/evaluations"

type Option = { value: string; label: string }

type OptionSourceKey =
  | "chatModels"
  | "embeddingModels"
  | "imageModels"
  | "ttsModels"
  | "transcriptionModels"
  | "ttsProviders"
  | "llmProviders"
  | "embeddingCollections"
  | "noteCollections"
  | "prompts"
  | "evaluations"
  | "evaluationDatasets"
  | "evaluationRuns"
  | `evaluationRuns:${string}`
  | `ttsVoices:${string}`
  | "readingItems"
  | "outputs"

type DynamicOptionsState = {
  optionsByKey: Record<string, Option[]>
  loadingByKey: Record<string, boolean>
}

const CACHE_TTL_MS = 5 * 60 * 1000
const optionsCache = new Map<OptionSourceKey, { options: Option[]; ts: number }>()
const inFlight = new Map<OptionSourceKey, Promise<Option[]>>()

export const __resetWorkflowDynamicOptionsCacheForTests = () => {
  optionsCache.clear()
  inFlight.clear()
}

const normalizeKey = (key: string) =>
  key.toLowerCase().replace(/[^a-z0-9]/g, "")

const isPromptIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "promptid" ||
    compact.endsWith("promptid") ||
    compact === "promptidentifier" ||
    compact.endsWith("promptidentifier")
  )
}

const isEvaluationIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "evalid" ||
    compact.endsWith("evalid") ||
    compact === "evaluationid" ||
    compact.endsWith("evaluationid")
  )
}

const isDatasetIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return compact === "datasetid" || compact.endsWith("datasetid")
}

const isRunIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return compact === "runid" || compact.endsWith("runid")
}

const isItemIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return compact === "itemid" || compact.endsWith("itemid")
}

const isOutputIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "outputid" ||
    compact.endsWith("outputid") ||
    compact === "artifactid" ||
    compact.endsWith("artifactid")
  )
}

const isFileIdKey = (key: string) => {
  const compact = normalizeKey(key)
  return compact === "fileid" || compact.endsWith("fileid")
}

const isCollectionKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "collection" ||
    compact.endsWith("collection") ||
    compact === "collectionid" ||
    compact.endsWith("collectionid")
  )
}

const isModelKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "model" ||
    compact.endsWith("model") ||
    compact.endsWith("modelid")
  )
}

const isProviderKey = (key: string) => {
  const compact = normalizeKey(key)
  return (
    compact === "provider" ||
    compact.endsWith("provider") ||
    compact === "apiname" ||
    compact.endsWith("apiname")
  )
}

const isVoiceKey = (key: string) => {
  const compact = normalizeKey(key)
  return compact === "voice" || compact.endsWith("voice")
}

const resolveProviderFromConfig = (
  config?: Record<string, unknown>
): string => {
  if (!config) return ""
  const raw =
    config.provider ??
    config.api_name ??
    config.apiName ??
    config.tts_provider ??
    config.ttsProvider ??
    config.voice_provider ??
    config.voiceProvider
  if (raw == null) return ""
  return String(raw).trim()
}

const resolveEvaluationIdFromConfig = (
  config?: Record<string, unknown>
): string => {
  if (!config) return ""
  const raw =
    config.eval_id ??
    config.evaluation_id ??
    config.evaluationId ??
    config.evalId
  if (raw == null) return ""
  return String(raw).trim()
}

const resolveCollectionSource = (stepType?: string, key?: string) => {
  const normalizedStep = String(stepType || "").toLowerCase()
  const normalizedKey = String(key || "").toLowerCase()
  if (
    normalizedStep === "notes" ||
    normalizedStep === "collections" ||
    normalizedStep === "prompts" ||
    normalizedKey.includes("notebook") ||
    normalizedKey.includes("folder")
  ) {
    return "noteCollections"
  }
  return "embeddingCollections"
}

const resolveModelSource = (stepType?: string, key?: string): OptionSourceKey => {
  const normalizedStep = String(stepType || "").toLowerCase()
  const normalizedKey = String(key || "").toLowerCase()
  if (normalizedStep === "tts" || normalizedKey.includes("tts")) {
    return "ttsModels"
  }
  if (
    normalizedStep === "stt_transcribe" ||
    normalizedKey.includes("transcription") ||
    normalizedKey.includes("stt")
  ) {
    return "transcriptionModels"
  }
  if (
    normalizedStep === "image_gen" ||
    normalizedKey.includes("image")
  ) {
    return "imageModels"
  }
  if (normalizedStep === "embed" || normalizedKey.includes("embedding")) {
    return "embeddingModels"
  }
  return "chatModels"
}

const normalizeOptions = (options: Option[]): Option[] => {
  const seen = new Set<string>()
  return options
    .map((option) => ({
      value: String(option.value ?? "").trim(),
      label: String(option.label ?? option.value ?? "").trim()
    }))
    .filter((option) => {
      if (!option.value) return false
      if (seen.has(option.value)) return false
      seen.add(option.value)
      return true
    })
}

const collectOptions = (source: unknown, output: Option[]) => {
  if (!source) return
  if (Array.isArray(source)) {
    for (const entry of source) {
      collectOptions(entry, output)
    }
    return
  }
  if (typeof source === "string" || typeof source === "number") {
    output.push({ value: String(source), label: String(source) })
    return
  }
  if (typeof source !== "object") return

  const record = source as Record<string, unknown>
  const value =
    record.value ??
    record.id ??
    record.model ??
    record.name ??
    record.label ??
    record.key
  if (value != null) {
    const label =
      record.label ??
      record.name ??
      record.title ??
      record.value ??
      record.id ??
      value
    output.push({ value: String(value), label: String(label) })
    return
  }

  for (const entry of Object.values(record)) {
    collectOptions(entry, output)
  }
}

const extractOptions = (source: unknown): Option[] => {
  const output: Option[] = []
  collectOptions(source, output)
  return normalizeOptions(output)
}

const cacheGet = (key: OptionSourceKey): Option[] | null => {
  const entry = optionsCache.get(key)
  if (!entry) return null
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    optionsCache.delete(key)
    return null
  }
  return entry.options
}

const cacheSet = (key: OptionSourceKey, options: Option[]) => {
  optionsCache.set(key, { options, ts: Date.now() })
}

const loadOptions = async (key: OptionSourceKey): Promise<Option[]> => {
  const cached = cacheGet(key)
  if (cached) return cached
  const inflight = inFlight.get(key)
  if (inflight) return inflight

  const promise = (async () => {
    try {
      let options: Option[] = []
      if (key === "chatModels") {
        const models = await tldwModels.getChatModels()
        options = normalizeOptions(
          models.map((model) => ({
            value: model.id,
            label: model.name
              ? `${model.name}${model.provider ? ` (${model.provider})` : ""}`
              : model.id
          }))
        )
      } else if (key === "embeddingModels") {
        const models = await tldwModels.getEmbeddingModels()
        options = normalizeOptions(
          models.map((model) => ({
            value: model.id,
            label: model.name
              ? `${model.name}${model.provider ? ` (${model.provider})` : ""}`
              : model.id
          }))
        )
      } else if (key === "imageModels") {
        const models = await tldwModels.getImageModels()
        options = normalizeOptions(
          models.map((model) => ({
            value: model.id,
            label: model.name
              ? `${model.name}${model.provider ? ` (${model.provider})` : ""}`
              : model.id
          }))
        )
      } else if (key === "ttsModels") {
        const models = await fetchTldwTtsModels()
        options = normalizeOptions(
          models.map((model) => ({
            value: model.id,
            label: model.label || model.id
          }))
        )
      } else if (key === "transcriptionModels") {
        const res = await tldwClient.getTranscriptionModels()
        options = extractOptions(res)
      } else if (key === "ttsProviders") {
        const providers = await fetchTtsProviders()
        const names = new Set<string>()
        if (providers?.providers) {
          Object.keys(providers.providers).forEach((name) => {
            names.add(name)
          })
        }
        if (providers?.voices) {
          Object.keys(providers.voices).forEach((name) => {
            names.add(name)
          })
        }
        options = normalizeOptions(
          Array.from(names).map((name) => ({ value: name, label: name }))
        )
      } else if (key === "llmProviders") {
        const res = await tldwClient.getProviders()
        if (Array.isArray(res)) {
          options = normalizeOptions(
            res.map((entry) => ({
              value: String((entry as any)?.name ?? (entry as any)?.id ?? entry),
              label: String((entry as any)?.name ?? (entry as any)?.id ?? entry)
            }))
          )
        } else if (res && typeof res === "object") {
          options = normalizeOptions(
            Object.keys(res).map((name) => ({ value: name, label: name }))
          )
        }
      } else if (key === "embeddingCollections") {
        const collections = await fetchEmbeddingCollections()
        options = normalizeOptions(
          collections.map((col) => ({ value: col.name, label: col.name }))
        )
      } else if (key === "noteCollections") {
        const result = await fetchFolders()
        if (result.ok && Array.isArray(result.data)) {
          options = normalizeOptions(
            result.data.map((folder) => ({
              value: String(folder.id),
              label: folder.name
            }))
          )
        }
      } else if (key === "prompts") {
        const res = await tldwClient.getPrompts()
        const items = Array.isArray(res)
          ? res
          : Array.isArray(res?.items)
            ? res.items
            : Array.isArray(res?.data)
              ? res.data
              : Array.isArray(res?.prompts)
                ? res.prompts
                : []
        options = normalizeOptions(
          items.map((item: any) => {
            const value =
              item?.id ??
              item?.prompt_id ??
              item?.uuid ??
              item?.prompt_uuid ??
              item?.name ??
              item?.title
            const label =
              item?.name ??
              item?.title ??
              item?.prompt_name ??
              item?.prompt_title ??
              value
            return { value: String(value ?? ""), label: String(label ?? "") }
          })
        )
      } else if (key === "evaluations") {
        const res = await listEvaluations({ limit: 100 })
        if (res.ok && res.data) {
          const items = Array.isArray(res.data)
            ? res.data
            : Array.isArray(res.data.data)
              ? res.data.data
              : []
          options = normalizeOptions(
            items.map((item: any) => ({
              value: String(item?.id ?? ""),
              label: String(item?.name ?? item?.id ?? "")
            }))
          )
        }
      } else if (key === "evaluationDatasets") {
        const res = await listDatasets({ limit: 100 })
        if (res.ok && res.data) {
          const items = Array.isArray(res.data)
            ? res.data
            : Array.isArray(res.data.data)
              ? res.data.data
              : []
          options = normalizeOptions(
            items.map((item: any) => ({
              value: String(item?.id ?? ""),
              label: String(item?.name ?? item?.id ?? "")
            }))
          )
        }
      } else if (key === "evaluationRuns" || key.startsWith("evaluationRuns:")) {
        const evalId = key.startsWith("evaluationRuns:")
          ? key.slice("evaluationRuns:".length)
          : ""
        if (!evalId) {
          options = []
        } else {
          const res = await listRuns(evalId, { limit: 100 })
          if (res.ok && res.data) {
          const items = Array.isArray(res.data)
            ? res.data
            : Array.isArray(res.data.data)
              ? res.data.data
              : []
          options = normalizeOptions(
            items.map((item: any) => ({
              value: String(item?.id ?? ""),
              label: String(item?.id ?? "")
            }))
          )
        }
        }
      } else if (key.startsWith("ttsVoices:")) {
        const provider = key.slice("ttsVoices:".length)
        const voices =
          provider && provider !== "all"
            ? await fetchTldwVoiceCatalog(provider)
            : await fetchTldwVoices()
        options = normalizeOptions(
          voices.map((voice) => {
            const value = voice.voice_id || voice.id || voice.name || ""
            const label =
              voice.name ||
              voice.voice_id ||
              voice.id ||
              ""
            return {
              value: String(value),
              label: voice.provider ? `${label} (${voice.provider})` : String(label)
            }
          })
        )
      } else if (key === "readingItems") {
        const res = await tldwClient.getReadingList({ page: 1, size: 50 })
        const items = Array.isArray(res?.items) ? res.items : []
        options = normalizeOptions(
          items.map((item: any) => ({
            value: String(item?.id ?? ""),
            label: String(item?.title ?? item?.url ?? item?.id ?? "")
          }))
        )
      } else if (key === "outputs") {
        const res = await tldwClient.listOutputs({ page: 1, size: 50 })
        const items = Array.isArray(res?.items) ? res.items : []
        options = normalizeOptions(
          items.map((item: any) => ({
            value: String(item?.id ?? ""),
            label: String(item?.title ?? item?.name ?? item?.id ?? "Output")
          }))
        )
      }

      cacheSet(key, options)
      return options
    } catch {
      return []
    }
  })()

  inFlight.set(key, promise)
  promise.finally(() => {
    inFlight.delete(key)
  })

  return promise
}

const resolveOptionSource = (
  field: ConfigFieldSchema,
  stepType?: string,
  config?: Record<string, unknown>
): OptionSourceKey | null => {
  if (field.options && field.options.length > 0) return null
  const key = field.key

  if (isPromptIdKey(key)) return "prompts"
  if (isEvaluationIdKey(key)) return "evaluations"
  if (isDatasetIdKey(key)) return "evaluationDatasets"
  if (isRunIdKey(key)) {
    const evalId = resolveEvaluationIdFromConfig(config)
    return evalId ? `evaluationRuns:${evalId}` : "evaluationRuns"
  }
  if (isItemIdKey(key)) return "readingItems"
  if (isOutputIdKey(key) || isFileIdKey(key)) return "outputs"

  if (field.type === "collection-picker" || isCollectionKey(key)) {
    return resolveCollectionSource(stepType, key)
  }

  if (field.type === "model-picker" || isModelKey(key)) {
    return resolveModelSource(stepType, key)
  }

  if (isVoiceKey(key)) {
    const provider = resolveProviderFromConfig(config)
    return `ttsVoices:${provider || "all"}`
  }

  if (isProviderKey(key)) {
    if (String(stepType || "").toLowerCase() === "tts") return "ttsProviders"
    return "llmProviders"
  }

  return null
}

export const useWorkflowDynamicOptions = (params: {
  fields: ConfigFieldSchema[]
  stepType?: string
  config?: Record<string, unknown>
}): DynamicOptionsState => {
  const { fields, stepType, config } = params
  const resolvedSources = useMemo(() => {
    const fieldToSource = new Map<string, OptionSourceKey>()
    const sources = new Set<OptionSourceKey>()
    for (const field of fields) {
      const source = resolveOptionSource(field, stepType, config)
      if (!source) continue
      fieldToSource.set(field.key, source)
      sources.add(source)
    }
    return {
      fieldToSource,
      sources: Array.from(sources)
    }
  }, [fields, stepType, config])

  const [optionsBySource, setOptionsBySource] = useState<
    Record<string, Option[]>
  >({})
  const [loadingBySource, setLoadingBySource] = useState<
    Record<string, boolean>
  >({})
  const sourceSignature = useMemo(
    () => resolvedSources.sources.join("|"),
    [resolvedSources.sources]
  )

  useEffect(() => {
    let isActive = true
    const pendingSources = resolvedSources.sources.filter(
      (source) => !optionsBySource[source] && !loadingBySource[source]
    )

    if (pendingSources.length === 0) {
      return () => {
        isActive = false
      }
    }

    const load = async () => {
      await Promise.all(
        pendingSources.map(async (source) => {
          if (!isActive) return
          setLoadingBySource((prev) =>
            prev[source] ? prev : { ...prev, [source]: true }
          )
          try {
            const options = await loadOptions(source)
            if (!isActive) return
            setOptionsBySource((prev) => ({ ...prev, [source]: options }))
          } finally {
            if (isActive) {
              setLoadingBySource((prev) => ({ ...prev, [source]: false }))
            }
          }
        })
      )
    }
    void load()
    return () => {
      isActive = false
    }
  }, [sourceSignature])

  const optionsByKey = useMemo(() => {
    const output: Record<string, Option[]> = {}
    resolvedSources.fieldToSource.forEach((source, key) => {
      if (optionsBySource[source]) {
        output[key] = optionsBySource[source]
      }
    })
    return output
  }, [resolvedSources.fieldToSource, optionsBySource])

  const loadingByKey = useMemo(() => {
    const output: Record<string, boolean> = {}
    resolvedSources.fieldToSource.forEach((source, key) => {
      if (loadingBySource[source]) {
        output[key] = true
      }
    })
    return output
  }, [resolvedSources.fieldToSource, loadingBySource])

  return { optionsByKey, loadingByKey }
}
