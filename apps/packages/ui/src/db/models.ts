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
 * Prefixed records and legacy unprefixed records remain authoritative. The old
 * shared index is only metadata: concurrent contexts can overwrite its ids.
 */
const MODEL_KEY_PREFIX = "model:"

const toStorageKey = (id: string) => `${MODEL_KEY_PREFIX}${id}`

const isModelRecord = (key: string, value: unknown): value is Model => {
  if (!value || typeof value !== "object") return false
  const record = value as Partial<Model> & { id?: unknown }
  return (
    record.id === key &&
    typeof record.model_id === "string" &&
    typeof record.provider_id === "string"
  )
}

const readStorage = (
  db: chrome.storage.StorageArea,
  keys: string[] | null
): Promise<Record<string, unknown>> => {
  return new Promise((resolve, reject) => {
    db.get(keys, (result) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError)
      } else {
        resolve(result as Record<string, unknown>)
      }
    })
  })
}

export class ModelDb {
  db: chrome.storage.StorageArea

  constructor() {
    this.db = chrome.storage.local
  }

  getAll = async (): Promise<Model[]> => {
    const getKeys = (this.db as chrome.storage.StorageArea & {
      getKeys?: (callback: (keys: string[]) => void) => void
    }).getKeys
    let items: Record<string, unknown>
    if (typeof getKeys === "function") {
      const keys = await new Promise<string[]>((resolve, reject) => {
        getKeys.call(this.db, (result) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError)
          } else {
            resolve(result)
          }
        })
      })
      const currentKeys = keys.filter((key) => key.startsWith(MODEL_KEY_PREFIX))
      // Legacy ids are unrestricted; inspect candidates without migrating them.
      const legacyKeys = keys.filter(
        (key) => !key.startsWith(MODEL_KEY_PREFIX)
      )
      items = {
        ...(legacyKeys.length ? await readStorage(this.db, legacyKeys) : {}),
        ...(currentKeys.length ? await readStorage(this.db, currentKeys) : {})
      }
    } else {
      // Older browsers lack getKeys; retain the read-only full-read fallback.
      items = await readStorage(this.db, null)
    }

    const byId = new Map<string, Model>()
    for (const [key, value] of Object.entries(items)) {
      if (isModelRecord(key, value)) byId.set(value.id, value)
    }
    // Current records win over legacy duplicates regardless of enumeration order.
    for (const [key, value] of Object.entries(items)) {
      if (key.startsWith(MODEL_KEY_PREFIX)) {
        if (isModelRecord(key.slice(MODEL_KEY_PREFIX.length), value)) {
          byId.set(value.id, value)
        }
      }
    }
    return Array.from(byId.values())
  }

  create = async (model: Model): Promise<void> => {
    await this.createMany([model])
  }

  /** Batched write: all records land in one set(), without shared index RMW. */
  createMany = async (models: Model[]): Promise<void> => {
    if (models.length === 0) return
    const writes = Object.fromEntries(
      models.map((record) => [toStorageKey(record.id), record])
    )
    const keys = Object.keys(writes)
    let items = await readStorage(this.db, keys)
    // Keyed web-shim reads use null for absence; distinguish stored null.
    if (keys.some((key) => items[key] === null)) {
      items = await readStorage(this.db, null)
    }
    for (const model of models) {
      const key = toStorageKey(model.id)
      if (
        Object.prototype.hasOwnProperty.call(items, key) &&
        !isModelRecord(model.id, items[key])
      ) {
        throw new Error(`Cannot overwrite occupied model storage key: ${key}`)
      }
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
  }

  getById = async (id: string): Promise<Model | undefined> => {
    const items = await readStorage(this.db, [toStorageKey(id), id])
    const current = items[toStorageKey(id)]
    if (isModelRecord(id, current)) return current
    const legacy = items[id]
    return isModelRecord(id, legacy) ? legacy : undefined
  }

  update = async (model: Model): Promise<void> => {
    await this.createMany([model])
  }

  delete = async (id: string): Promise<void> => {
    await this.deleteMany([id])
  }

  /** Batched delete: one remove() for current and legacy aliases. */
  deleteMany = async (ids: string[]): Promise<void> => {
    if (ids.length === 0) return
    await this.removeRecords(ids)
  }

  deleteAll = async (): Promise<void> => {
    const ids = (await this.getAll()).map((model) => model.id)
    await this.removeRecords(ids)
  }

  private removeRecords = async (ids: string[]): Promise<void> => {
    const items = ids.length
      ? await readStorage(this.db, ids.flatMap((id) => [toStorageKey(id), id]))
      : {}
    // An alias may hold unrelated data or another model with a prefix-shaped id.
    const keys = ids.flatMap((id) =>
      [toStorageKey(id), id].filter((key) => isModelRecord(id, items[key]))
    )
    if (keys.length === 0) return
    await new Promise<void>((resolve, reject) => {
      this.db.remove(keys, () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError)
        } else {
          resolve()
        }
      })
    })
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
