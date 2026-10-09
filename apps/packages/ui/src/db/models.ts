import { getAllModelNicknames } from "./nickname"
import type { Model } from "@/db/dexie/types"
import {
  getLLamaCppModelId,
  getLMStudioModelId,
  getLlamafileModelId,
  getVLLMModelId,
  isCustomModel,
  isLLamaCppModel,
  isLMStudioModel,
  isLlamafileModel,
  isVLLMModel,
  removeModelSuffix,
  type DynamicFetchParams,
  type DynamicModelListing,
  dynamicFetchLMStudio,
  dynamicFetchLLamaCpp,
  dynamicFetchLlamafile,
  dynamicFetchVLLM
} from "./model-provider-utils"

interface CustomModelView extends Model {
  nickname: string
  avatar?: string
}
export const generateID = () => {
  return "model-xxxx-xxxx-xxx-xxxx".replace(/[x]/g, () => {
    const r = Math.floor(Math.random() * 16)
    return r.toString(16)
  })
}

/**
 * Model records live under a `model:` key prefix so reads can be scoped to
 * exactly the catalog instead of dumping ALL of chrome.storage.local. The
 * index key tracks the record ids; it is backfilled once from legacy
 * unprefixed records so existing persisted data keeps working.
 */
const MODEL_KEY_PREFIX = "model:"
const MODEL_INDEX_KEY = "__tldwModelDbIndexV1"

const toStorageKey = (id: string) => `${MODEL_KEY_PREFIX}${id}`

const isLegacyModelRecord = (key: string, value: unknown): value is Model => {
  if (!value || typeof value !== "object") return false
  const record = value as Partial<Model> & { id?: unknown }
  return (
    record.id === key &&
    typeof record.model_id === "string" &&
    typeof record.provider_id === "string"
  )
}

const readIndex = async (
  db: chrome.storage.StorageArea
): Promise<{ ids: string[]; migrated: boolean } | null> => {
  return new Promise((resolve, reject) => {
    db.get(MODEL_INDEX_KEY, (result) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError)
        return
      }
      const raw = (result as Record<string, unknown>)[MODEL_INDEX_KEY]
      if (Array.isArray(raw)) {
        resolve({
          ids: raw.filter((id): id is string => typeof id === "string"),
          migrated: true
        })
        return
      }
      resolve(null)
    })
  })
}

const writeIndex = async (
  db: chrome.storage.StorageArea,
  ids: string[]
): Promise<void> => {
  return new Promise((resolve, reject) => {
    db.set({ [MODEL_INDEX_KEY]: ids }, () => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError)
      } else {
        resolve()
      }
    })
  })
}

const migrateLegacyRecords = async (
  db: chrome.storage.StorageArea
): Promise<string[]> => {
  // One-time full read: move legacy unprefixed model records under the
  // `model:` prefix and seed the index. Later reads never use get(null).
  const everything = await new Promise<Record<string, unknown>>(
    (resolve, reject) => {
      db.get(null, (result) => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve(result as Record<string, unknown>)
        }
      })
    }
  )

  const legacy: Model[] = []
  for (const [key, value] of Object.entries(everything)) {
    if (key.startsWith(MODEL_KEY_PREFIX) || key === MODEL_INDEX_KEY) continue
    if (isLegacyModelRecord(key, value)) {
      legacy.push(value)
    }
  }

  const existing: Model[] = []
  for (const [key, value] of Object.entries(everything)) {
    if (!key.startsWith(MODEL_KEY_PREFIX)) continue
    if (value && typeof value === "object" && typeof (value as Model).id === "string") {
      existing.push(value as Model)
    }
  }

  const byId = new Map<string, Model>()
  for (const record of [...existing, ...legacy]) {
    byId.set(record.id, record)
  }
  const ids = Array.from(byId.keys())

  const writes: Record<string, unknown> = { [MODEL_INDEX_KEY]: ids }
  const legacyKeys: string[] = []
  for (const [id, record] of byId) {
    writes[toStorageKey(id)] = record
    if (!existing.some((candidate) => candidate.id === id)) {
      legacyKeys.push(id)
    }
  }

  await new Promise<void>((resolve, reject) => {
    db.set(writes, () => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError)
      } else {
        resolve()
      }
    })
  })
  if (legacyKeys.length > 0) {
    await new Promise<void>((resolve, reject) => {
      db.remove(legacyKeys, () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve()
        }
      })
    })
  }
  return ids
}

export class ModelDb {
  db: chrome.storage.StorageArea
  private indexIds: string[] | null = null

  constructor() {
    this.db = chrome.storage.local
  }

  private ensureIndex = async (): Promise<string[]> => {
    if (this.indexIds) return this.indexIds
    const stored = await readIndex(this.db).catch(() => null)
    this.indexIds =
      stored?.ids ?? (await migrateLegacyRecords(this.db).catch(() => []))
    return this.indexIds
  }

  private persistRecords = async (
    records: Model[],
    options?: { removeIds?: string[]; nextIndexIds?: string[] }
  ): Promise<void> => {
    const current = await this.ensureIndex()
    const indexSet = new Set(options?.nextIndexIds ?? current)
    for (const record of records) {
      indexSet.add(record.id)
    }
    for (const removedId of options?.removeIds ?? []) {
      indexSet.delete(removedId)
    }
    const ids = Array.from(indexSet)
    const writes: Record<string, unknown> = {
      [MODEL_INDEX_KEY]: ids,
      ...Object.fromEntries(records.map((record) => [toStorageKey(record.id), record]))
    }
    await new Promise<void>((resolve, reject) => {
      this.db.set(writes, () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve()
        }
      })
    })
    this.indexIds = ids
  }

  getAll = async (): Promise<Model[]> => {
    const ids = await this.ensureIndex()
    if (ids.length === 0) return []
    return new Promise((resolve, reject) => {
      this.db.get(ids.map(toStorageKey), (result) => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          const items = result as Record<string, Model>
          resolve(
            ids
              .map((id) => items[toStorageKey(id)])
              .filter((item): item is Model => Boolean(item))
          )
        }
      })
    })
  }

  create = async (model: Model): Promise<void> => {
    await this.createMany([model])
  }

  /** Batched write: all records plus the index update land in ONE set(). */
  createMany = async (models: Model[]): Promise<void> => {
    if (models.length === 0) return
    await this.persistRecords(models)
  }

  getById = async (id: string): Promise<Model | undefined> => {
    await this.ensureIndex()
    return new Promise((resolve, reject) => {
      this.db.get(toStorageKey(id), (result) => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve((result as Record<string, Model>)[toStorageKey(id)])
        }
      })
    })
  }

  update = async (model: Model): Promise<void> => {
    await this.persistRecords([model])
  }

  delete = async (id: string): Promise<void> => {
    await this.deleteMany([id])
  }

  /** Batched delete: one remove() for the records plus one index write. */
  deleteMany = async (ids: string[]): Promise<void> => {
    if (ids.length === 0) return
    const current = await this.ensureIndex()
    const removed = new Set(ids)
    await new Promise<void>((resolve, reject) => {
      // Also drop legacy unprefixed keys if any somehow remain.
      const keys = ids.flatMap((id) => [toStorageKey(id), id])
      this.db.remove(keys, () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve()
        }
      })
    })
    const nextIndexIds = current.filter((candidate) => !removed.has(candidate))
    if (nextIndexIds.length !== current.length) {
      await writeIndex(this.db, nextIndexIds).catch(() => undefined)
    }
    this.indexIds = nextIndexIds
  }

  deleteAll = async (): Promise<void> => {
    const ids = await this.ensureIndex()
    const keys = [...ids.map(toStorageKey), ...ids, MODEL_INDEX_KEY]
    await new Promise<void>((resolve, reject) => {
      this.db.remove(keys, () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve()
        }
      })
    })
    this.indexIds = []
  }
}

export const createManyModels = async (
  data: {
    model_id: string
    name: string
    provider_id: string
    model_type: string
  }[]
) => {
  const db = new ModelDb()

  const models = data.map((item) => {
    return {
      ...item,
      lookup: `${item.model_id}_${item.provider_id}`,
      id: `${item.model_id}_${generateID()}`,
      db_type: "openai_model",
      name: item.name.replaceAll(/accounts\/[^\/]+\/models\//g, "")
    }
  })

  const existing = await db.getAll()
  const existingLookups = new Set(
    existing.map((entry) => entry?.lookup).filter(Boolean)
  )
  const missing = models.filter((model) => !existingLookups.has(model.lookup))
  if (missing.length === 0) return
  await db.createMany(missing)
}

export const createModelFB = async (model: Model): Promise<boolean> => {
  try {
    const db = new ModelDb()
    await db.create(model)
    return true
  } catch (e) {
    // Surface storage failures instead of silently swallowing them
    // eslint-disable-next-line no-console
    console.error("Failed to create model", e)
    return false
  }
}

export const getAllModelsExT = async () => {
  const db = new ModelDb()
  const allData = await db.getAll()
  return allData?.filter((d) => d?.db_type === "openai_model") || []
}

export const getModelInfoFB = async (id: string) => {
  const db = new ModelDb()

  if (isLMStudioModel(id)) {
    const lmstudioId = getLMStudioModelId(id)
    if (!lmstudioId) {
      throw new Error("Invalid LMStudio model ID")
    }
    return {
      model_id: id.replace(
        /_lmstudio_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      ),
      provider_id: `openai-${lmstudioId.provider_id}`,
      name: id.replace(
        /_lmstudio_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      )
    }
  }

  if (isLlamafileModel(id)) {
    const llamafileId = getLlamafileModelId(id)
    if (!llamafileId) {
      throw new Error("Invalid Llamafile model ID")
    }
    return {
      model_id: id.replace(
        /_llamafile_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      ),
      provider_id: `openai-${llamafileId.provider_id}`,
      name: id.replace(
        /_llamafile_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      )
    }
  }

  if (isLLamaCppModel(id)) {
    const llamaCppId = getLLamaCppModelId(id)
    if (!llamaCppId) {
      throw new Error("Invalid LlamaCpp model ID")
    }

    return {
      model_id: id.replace(
        /_llamacpp_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      ),
      provider_id: `openai-${llamaCppId.provider_id}`,
      name: id.replace(
        /_llamacpp_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      )
    }
  }

  if (isVLLMModel(id)) {
    const vllmId = getVLLMModelId(id)
    if (!vllmId) {
      throw new Error("Invalid Vllm model ID")
    }
    return {
      model_id: id.replace(
        /_vllm_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/,
        ""
      ),
      provider_id: `openai-${vllmId.provider_id}`,
      name: id.replace(/_vllm_openai-[a-f0-9]{4}-[a-f0-9]{3}-[a-f0-9]{4}/, "")
    }
  }

  const model = await db.getById(id)
  return model
}

export const getAllCustomModelsFB = async (): Promise<CustomModelView[]> => {
  const db = new ModelDb()
  const modelNicknames = await getAllModelNicknames()
  const models = (await db.getAll()).filter(
    (model) => model?.db_type === "openai_model"
  )
  return models.map((model) => {
    return {
      ...model,
      nickname: modelNicknames[model.model_id]?.model_name || model.model_id,
      avatar: modelNicknames[model.model_id]?.model_avatar || undefined
    }
  })
}

export const deleteModelFB = async (id: string) => {
  const db = new ModelDb()
  await db.delete(id)
}

export const deleteAllModelsByProviderId = async (provider_id: string) => {
  const db = new ModelDb()
  const models = await db.getAll()
  const modelsToDelete = models.filter(
    (model) => model.provider_id === provider_id
  )
  await db.deleteMany(modelsToDelete.map((model) => model.id))
}

export const bulkAddModelsFB = async (models: Model[]) => {
  const db = new ModelDb()
  // delete all exist models
  const modelsToDelete = (await db.getAll()).filter(
    (model) => model?.db_type === "openai_model"
  )
  await db.deleteMany(modelsToDelete.map((model) => model.id))
  // add new models
  await db.createMany(models)
}

export const isLookupExist = async (lookup: string) => {
  const db = new ModelDb()
  const models = await db.getAll()
  const model = models.find((model) => model?.lookup === lookup)
  return !!model
}

export type { DynamicFetchParams, DynamicModelListing }
export {
  removeModelSuffix,
  isLMStudioModel,
  isLlamafileModel,
  isLLamaCppModel,
  isVLLMModel,
  getLMStudioModelId,
  getLlamafileModelId,
  getLLamaCppModelId,
  getVLLMModelId,
  isCustomModel,
  dynamicFetchLMStudio,
  dynamicFetchLLamaCpp,
  dynamicFetchVLLM,
  dynamicFetchLlamafile
}
