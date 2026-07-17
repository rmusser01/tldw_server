import { bgRequestClient } from "@/services/background-proxy"

export type TldwVoice = {
  id?: string
  voice_id?: string
  name?: string
  description?: string | null
  provider?: string | null
  duration_seconds?: number | null
  tags?: string[] | null
}

const isVoiceRecord = (value: unknown): value is TldwVoice => {
  if (!value || typeof value !== "object") return false
  const record = value as Record<string, unknown>
  if ("voices" in record || "data" in record) return false
  return (
    "voice_id" in record ||
    "id" in record ||
    "name" in record ||
    "provider" in record
  )
}

const collectVoices = (source: unknown, output: TldwVoice[]) => {
  if (!source) return
  if (Array.isArray(source)) {
    for (const item of source) {
      collectVoices(item, output)
    }
    return
  }
  if (typeof source === "string") {
    output.push({ id: source, name: source })
    return
  }
  if (typeof source !== "object") return

  const record = source as Record<string, unknown>
  if ("voices" in record) {
    collectVoices(record.voices, output)
    return
  }
  if ("data" in record) {
    collectVoices(record.data, output)
    return
  }
  if (isVoiceRecord(record)) {
    output.push(record as TldwVoice)
    return
  }
  for (const value of Object.values(record)) {
    collectVoices(value, output)
  }
}

const extractVoices = (source: unknown): TldwVoice[] => {
  const output: TldwVoice[] = []
  collectVoices(source, output)
  const seen = new Set<string>()
  return output.filter((voice) => {
    const key = voice.voice_id || voice.id || voice.name
    if (!key) return false
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

type FetchTldwVoicesOptions = {
  throwOnError?: boolean
  model?: string
}

export const fetchTldwVoices = async (
  options?: FetchTldwVoicesOptions
): Promise<TldwVoice[]> => {
  try {
    const res = await bgRequestClient<any>({
      path: "/api/v1/audio/voices",
      method: "GET"
    })
    if (!res) return []
    return extractVoices(res)
  } catch (error) {
    if (options?.throwOnError) {
      throw error
    }
    return []
  }
}

export const fetchTldwVoiceCatalog = async (
  provider: string,
  options?: FetchTldwVoicesOptions
): Promise<TldwVoice[]> => {
  const p = String(provider || "").trim()
  if (!p) return []
  const model = String(options?.model || "").trim()
  try {
    const res = await bgRequestClient<any>({
      path: `/api/v1/audio/voices/catalog?provider=${encodeURIComponent(p)}${
        model ? `&model=${encodeURIComponent(model)}` : ""
      }`,
      method: "GET"
    })
    if (!res) return []
    return extractVoices(res)
  } catch (error) {
    if (options?.throwOnError) {
      throw error
    }
    return []
  }
}
