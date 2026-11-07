import {
  getLastUsedChatModel,
  getLastUsedChatSystemPrompt
} from "@/services/model-settings"
import { PageAssitDatabase as ChromeDB } from "../index"
import { PageAssistDatabase as DexieDB } from "./chat"
import { db } from "./schema"
import type { HistoryInfo } from "./types"
import { ModelNickname as DexieDBNick } from "./nickname"
import { ModelDb as DexieDBM } from "./models"
import { getAllModelNicknamesMig } from "../nickname"

const hasHistoryOwner = (history?: HistoryInfo) => history && (
  history.local_owner_key != null || history.server_scope_key != null ||
  history.server_chat_id != null || history.message_source === "server"
)

export class DatabaseMigration {
  private chromeDB: ChromeDB
  private dexieDB: DexieDB
  private dexieDBM: DexieDBM
  private dexieNick: DexieDBNick

  constructor() {
    this.chromeDB = new ChromeDB()
    this.dexieDB = new DexieDB()
    this.dexieDBM = new DexieDBM()
    this.dexieNick = new DexieDBNick()
  }

  async migrateAllData(): Promise<{
    success: boolean
    migratedCounts: {
      chatHistories: number
      messages: number
      prompts: number
      webshares: number
      sessionFiles: number
      userSettings: number
    }
    errors: string[]
  }> {
    const errors: string[] = []
    const migratedCounts = {
      chatHistories: 0,
      messages: 0,
      prompts: 0,
      webshares: 0,
      sessionFiles: 0,
      userSettings: 0,
    }

    try {
      // Migrate user settings
      try {
        const userId = await this.chromeDB.getUserID()
        if (userId && typeof userId === "string" && userId.trim() !== "") {
          await db.transaction("rw", db.userSettings, async () => {
            if (!(await this.dexieDB.getUserID())) await this.dexieDB.setUserID(userId)
          })
          migratedCounts.userSettings = 1
        }
      } catch (error) {
        errors.push(`Failed to migrate user settings: ${error}`)
      }



      // Migrate chat histories and messages
      try {
        const chatHistories = await this.chromeDB.getChatHistories()
        for (const history of chatHistories) {
          try {
            const model = await getLastUsedChatModel(history.id)
            const prompt = await getLastUsedChatSystemPrompt(history.id)
            const messages = await this.chromeDB.getChatHistory(history.id)
            if (messages.some((message) => message.history_id !== history.id)) {
              throw new Error("Message belongs to a different history")
            }
            await db.transaction("rw", [db.chatHistories, db.messages, db.compareStates], async () => {
              const existing = await db.chatHistories.get(history.id)
              const saved = await db.messages.bulkGet(messages.map((message) => message.id))
              if (saved.some((message) => message && message.history_id !== history.id)) {
                throw new Error("Message ID belongs to a different history")
              }
              const missing = messages.filter((_, index) => !saved[index])
              if (hasHistoryOwner(existing) && missing.length) {
                throw new Error("Cannot add legacy messages to a claimed history")
              }
              await this.dexieDB.importChatHistoryV2([{
                history: existing ? undefined : { ...history, model_id: model, last_used_prompt: prompt },
                messages: missing
              }])
            })
            migratedCounts.chatHistories++
            migratedCounts.messages += messages.length
          } catch (historyError) {
            errors.push(
              `Failed to migrate chat history ${history.id}: ${historyError}`
            )
          }
        }
        // Keep the legacy copy available after an interrupted or conflicting import.
      } catch (error) {
        errors.push(`Failed to migrate chat histories: ${error}`)
      }


      // Skip migration of legacy OpenAI configs and custom models; the
      // extension is now tldw_server-only.

     // Migrate nickname
      try {
        console.log("Saving Nickname")
        const nicknames = await getAllModelNicknamesMig()
        await db.transaction("rw", db.modelNickname, async () => {
          const saved = await db.modelNickname.bulkGet(nicknames.map((nickname) => nickname.model_id))
          await this.dexieNick.importDataV2(nicknames.filter((_, index) => !saved[index]))
        })
      } catch (error) {
        errors.push(`Failed to migrate nick: ${error}`)
      }



      // Migrate prompts
      try {
        const prompts = await this.chromeDB.getAllPrompts()
        for (const prompt of prompts) {
          try {
            await db.transaction("rw", db.prompts, async () => {
              if (!(await db.prompts.get(prompt.id))) await this.dexieDB.migrateLegacyPrompt(prompt)
            })
            migratedCounts.prompts++
          } catch (promptError) {
            errors.push(`Failed to migrate prompt ${prompt.id}: ${promptError}`)
          }
        }
      } catch (error) {
        errors.push(`Failed to migrate prompts: ${error}`)
      }

      // Migrate webshares
      try {
        const webshares = await this.chromeDB.getAllWebshares()
        for (const webshare of webshares) {
          try {
            await db.transaction("rw", db.webshares, async () => {
              if (!(await this.dexieDB.getWebshare(webshare.id))) await this.dexieDB.addWebshare(webshare)
            })
            migratedCounts.webshares++
          } catch (webshareError) {
            errors.push(
              `Failed to migrate webshare ${webshare.id}: ${webshareError}`
            )
          }
        }
      } catch (error) {
        errors.push(`Failed to migrate webshares: ${error}`)
      }

      // Migrate session files - this is tricky since Chrome storage doesn't have a way to enumerate all keys
      // We'll need to handle this separately or ask the user to provide session IDs
      console.warn(
        "Session files migration requires manual handling of session IDs"
      )

      return {
        success: errors.length === 0,
        migratedCounts,
        errors
      }
    } catch (error) {
      errors.push(`General migration error: ${error}`)
      return {
        success: false,
        migratedCounts,
        errors
      }
    }
  }

  async migrateSessionFiles(sessionIds: string[]): Promise<{
    success: boolean
    migratedCount: number
    errors: string[]
  }> {
    const errors: string[] = []
    let migratedCount = 0

    for (const sessionId of sessionIds) {
      try {
        const sessionFiles = await this.chromeDB.getSessionFilesInfo(sessionId)
        if (sessionFiles) {
          await db.transaction("rw", [db.chatHistories, db.sessionFiles], async () => {
            if (hasHistoryOwner(await db.chatHistories.get(sessionId))) {
              const existing = await db.sessionFiles.get(sessionId)
              const savedIds = new Set(existing?.files.map((file) => file.id))
              if (!existing || sessionFiles.files.some((file) => !savedIds.has(file.id))) {
                throw new Error("Cannot add legacy session files to a claimed history")
              }
              return
            }
            await this.dexieDB.importSessionFilesV2([{ ...sessionFiles, sessionId }])
          })
          migratedCount++
        }
      } catch (error) {
        errors.push(
          `Failed to migrate session files for ${sessionId}: ${error}`
        )
      }
    }

    return {
      success: errors.length === 0,
      migratedCount,
      errors
    }
  }

  async verifyMigration(): Promise<{
    isValid: boolean
    counts: {
      chrome: {
        chatHistories: number
        prompts: number
        webshares: number
      }
      dexie: {
        chatHistories: number
        prompts: number
        webshares: number
      }
    }
    issues: string[]
  }> {
    const issues: string[] = []

    try {
      // Count Chrome storage data
      const chromeChatHistories = await this.chromeDB.getChatHistories()
      const chromePrompts = await this.chromeDB.getAllPrompts()
      const chromeWebshares = await this.chromeDB.getAllWebshares()

      // Count Dexie data
      const dexieChatHistories = await this.dexieDB.getChatHistories()
      const dexiePromptsCount = await this.dexieDB.countStoredPromptsForMigration()
      const dexieWebshares = await this.dexieDB.getAllWebshares()

      const counts = {
        chrome: {
          chatHistories: chromeChatHistories.length,
          prompts: chromePrompts.length,
          webshares: chromeWebshares.length
        },
        dexie: {
          chatHistories: dexieChatHistories.length,
          prompts: dexiePromptsCount,
          webshares: dexieWebshares.length
        }
      }

      // Check for discrepancies
      if (counts.chrome.chatHistories !== counts.dexie.chatHistories) {
        issues.push(
          `Chat histories count mismatch: Chrome(${counts.chrome.chatHistories}) vs Dexie(${counts.dexie.chatHistories})`
        )
      }

      if (counts.chrome.prompts !== counts.dexie.prompts) {
        issues.push(
          `Prompts count mismatch: Chrome(${counts.chrome.prompts}) vs Dexie(${counts.dexie.prompts})`
        )
      }

      if (counts.chrome.webshares !== counts.dexie.webshares) {
        issues.push(
          `Webshares count mismatch: Chrome(${counts.chrome.webshares}) vs Dexie(${counts.dexie.webshares})`
        )
      }

      return {
        isValid: issues.length === 0,
        counts,
        issues
      }
    } catch (error) {
      issues.push(`Verification failed: ${error}`)
      return {
        isValid: false,
        counts: {
          chrome: { chatHistories: 0, prompts: 0, webshares: 0 },
          dexie: { chatHistories: 0, prompts: 0, webshares: 0 }
        },
        issues
      }
    }
  }

  async clearDexieDatabase(): Promise<void> {
    await this.dexieDB.clear()
  }
}

/**
 * Detect whether there is any legacy data in the old Chrome storage
 * that would require running the migration.
 */
export const hasLegacyData = async (): Promise<boolean> => {
  const chromeDB = new ChromeDB()

  try {
    const [chatHistories, prompts, webshares] = await Promise.all([
      chromeDB.getChatHistories(),
      chromeDB.getAllPrompts(),
      chromeDB.getAllWebshares()
    ])

    const sessionIds = await getAllSessionIds()

    return (
      chatHistories.length > 0 ||
      prompts.length > 0 ||
      webshares.length > 0 ||
      sessionIds.length > 0
    )
  } catch (error) {
    console.warn("[migration] Failed to check legacy data presence:", error)
    // In doubt, err on the side of running the migration.
    return true
  }
}

export const runMigration = async (): Promise<void> => {
  const migration = new DatabaseMigration()

  console.log("Starting database migration...")
  const result = await migration.migrateAllData()

  if (result.success) {
    console.log("Migration completed successfully!")
    console.log("Migrated counts:", result.migratedCounts)
  } else {
    console.error("Migration completed with errors:")
    console.error("Errors:", result.errors)
    console.log("Partial migration counts:", result.migratedCounts)
    throw new Error("Local database migration did not finish")
  }

  // Verify migration
  console.log("Verifying migration...")
  const verification = await migration.verifyMigration()

  if (verification.isValid) {
    console.log("Migration verification passed!")
  } else {
    console.warn("Migration verification found issues:")
    console.warn(verification.issues)
  }

  console.log("Data counts:", verification.counts)
}

// Helper function to get all session IDs from Chrome storage
export const getAllSessionIds = async (): Promise<string[]> => {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(null, (items) => {
      if (chrome.runtime?.lastError || !items) {
        reject(new Error(chrome.runtime?.lastError?.message || "Storage unavailable"))
        return
      }
      const sessionIds: string[] = []
      for (const key in items) {
        if (key.startsWith("session_files_")) {
          sessionIds.push(key.replace("session_files_", ""))
        }
      }
      resolve(sessionIds)
    })
  })
}

export const runSessionFilesMigration = async (): Promise<void> => {
  const migration = new DatabaseMigration()
  const sessionIds = await getAllSessionIds()

  if (sessionIds.length > 0) {
    console.log(`Found ${sessionIds.length} session files to migrate`)
    const result = await migration.migrateSessionFiles(sessionIds)

    if (result.success) {
      console.log(`Successfully migrated ${result.migratedCount} session files`)
    } else {
      console.error("Session files migration completed with errors:")
      console.error(result.errors)
      throw new Error("Session file migration did not finish")
    }
  } else {
    console.log("No session files found to migrate")
  }
}

export const runAllMigrations = async (): Promise<void> => {
  await runMigration()
  await runSessionFilesMigration()
}
