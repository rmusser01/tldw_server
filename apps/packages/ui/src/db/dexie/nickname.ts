import { db } from "./schema"
import { ModelNicknames, ModelNickname as MNick } from "./types"

/**
 * In-memory cache for the nickname map (WP-28 / TASK-13523):
 * getAllModelNicknames() runs per history load and per model-list render, so
 * repeat reads are served from memory. The cache is invalidated by every
 * nickname mutation in this module (saveModelNickname, importDataV2) and by
 * whole-database clears (PageAssistDatabase.clear/clearDB).
 *
 * Limitation: the cache is per JS context. Other browser contexts (popup,
 * sidepanel, page) hold their own module instance, so there is no
 * cross-context invalidation; a same-context write that bypasses this module
 * stays unobserved until an invalidation or reload.
 */
let nicknameCache: Record<string, MNick> | undefined

export const invalidateModelNicknameCache = () => {
  nicknameCache = undefined
}

export class ModelNickname {
  async saveModelNickname(
    id: string,
    model_name: string,
    model_avatar?: string
  ): Promise<void> {
    await db.modelNickname.put({
      id,
      model_name,
      model_id: id,
      model_avatar
    })
    invalidateModelNicknameCache()
    console.log({
      id,
      model_name,
      model_id: id,
      model_avatar
    })
  }

  async getModelNicknameByID(model_id: string) {
    return await db.modelNickname.get(model_id)
  }

  async getAllModelNicknames() {
    return await db.modelNickname.reverse().toArray()
  }

  async importDataV2(
    data: ModelNicknames,
    options: {
      replaceExisting?: boolean
      mergeData?: boolean
    } = {}
  ): Promise<void> {
    const { replaceExisting = false, mergeData = true } = options

    await db.modelNickname.bulkPut(
      data.map((oai) => ({
        id: oai.model_id,
        model_id: oai.model_id,
        model_name: oai.model_name,
        model_avatar: oai.model_avatar
      }))
    )
    invalidateModelNicknameCache()
  }
}

export const getAllModelNicknames = async () => {
  if (nicknameCache !== undefined) return nicknameCache
  try {
    const modelNickname = new ModelNickname()
    const data = await modelNickname.getAllModelNicknames()
    const result: Record<string, MNick> = {}
    for (const d of data) {
      result[d.model_id] = d
    }
    // Only successful reads are cached; a failed read (e.g. Firefox private
    // mode) retries on the next call.
    nicknameCache = result
    return result
  } catch (e) {
    console.error("Firefox Private Mode Error", e)
    return {}
  }
}

export const getModelNicknameByID = async (
  model_id: string
): Promise<{ model_name: string; model_avatar?: string } | null> => {
  try {
    const modelNickname = new ModelNickname()
    return await modelNickname.getModelNicknameByID(model_id)
  } catch (e) {
    return null
  }
}

export const saveModelNickname = async ({
  model_id,
  model_name,
  model_avatar
}: {
  model_id: string
  model_name: string
  model_avatar?: string
}) => {
  const modelNickname = new ModelNickname()
  return await modelNickname.saveModelNickname(
    model_id,
    model_name,
    model_avatar
  )
}
