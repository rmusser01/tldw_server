/**
 * Folder System Zustand Store
 *
 * Manages folder and keyword state, syncing with tldw_server.
 * UI preferences (expand state, colors) are persisted locally.
 */

import { createWithEqualityFn } from "zustand/traditional"
import { persist, createJSONStorage, type StateStorage } from "zustand/middleware"
import { useShallow } from "zustand/react/shallow"
import type { Folder, Keyword, FolderKeywordLink, ConversationKeywordLink } from "@/db/dexie/types"
import { db } from "@/db/dexie/schema"
import {
  fetchFolders,
  fetchKeywords,
  fetchFolderKeywordLinks,
  fetchConversationKeywordLinks,
  createFolder as apiCreateFolder,
  updateFolder as apiUpdateFolder,
  deleteFolder as apiDeleteFolder,
  createKeyword as apiCreateKeyword,
  deleteKeyword as apiDeleteKeyword,
  linkKeywordToFolder as apiLinkKeywordToFolder,
  unlinkKeywordFromFolder as apiUnlinkKeywordFromFolder,
  linkKeywordToConversation as apiLinkKeywordToConversation,
  unlinkKeywordFromConversation as apiUnlinkKeywordFromConversation
} from "@/services/folder-api"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface FolderUIPrefs {
  isOpen: boolean
  color?: string
}

export interface FolderTreeNode {
  key: number
  title: string
  folder: Folder
  children: FolderTreeNode[]
  keywords: Keyword[]
  itemCount: number
}

const normalizeKeywordValue = (value: string): string =>
  value.trim().toLowerCase()

const findKeywordByName = (keywords: Keyword[], value: string): Keyword | null => {
  const normalized = normalizeKeywordValue(value)
  if (!normalized) return null
  return (
    keywords.find(
      (keyword) =>
        !keyword.deleted &&
        normalizeKeywordValue(keyword.keyword) === normalized
    ) || null
  )
}

interface FolderState {
  // Server data (cached)
  folders: Folder[]
  keywords: Keyword[]
  folderKeywordLinks: FolderKeywordLink[]
  conversationKeywordLinks: ConversationKeywordLink[]

  // Loading state
  isLoading: boolean
  lastSynced: number | null
  error: string | null

  // Feature availability (cached to avoid repeated 404s)
  folderApiAvailable: boolean | null // null = unknown, true/false = checked

  // View state
  viewMode: 'folders' | 'date'

  // Local UI preferences (persisted)
  uiPrefs: Record<number, FolderUIPrefs>

  // Actions - Data
  setFolders: (folders: Folder[]) => void
  setKeywords: (keywords: Keyword[]) => void
  setFolderKeywordLinks: (links: FolderKeywordLink[]) => void
  setConversationKeywordLinks: (links: ConversationKeywordLink[]) => void

  // Actions - UI
  setViewMode: (mode: 'folders' | 'date') => void
  toggleFolderOpen: (folderId: number) => void
  setFolderColor: (folderId: number, color: string | undefined) => void
  expandAllFolders: () => void
  collapseAllFolders: () => void

  // Actions - Server operations
  refreshFromServer: () => Promise<void>
  createFolder: (name: string, parentId?: number | null) => Promise<Folder | null>
  updateFolder: (id: number, data: { name?: string; parent_id?: number | null }) => Promise<boolean>
  deleteFolder: (id: number) => Promise<boolean>
  createKeyword: (keyword: string) => Promise<Keyword | null>
  ensureKeyword: (keyword: string) => Promise<Keyword | null>
  addKeywordToConversation: (conversationId: string, keywordId: number) => Promise<boolean>
  removeKeywordFromConversation: (conversationId: string, keywordId: number) => Promise<boolean>
  addKeywordToFolder: (folderId: number, keywordId: number) => Promise<boolean>
  addConversationToFolder: (conversationId: string, folderId: number) => Promise<boolean>
  removeConversationFromFolder: (conversationId: string, folderId: number) => Promise<boolean>

  // Computed helpers
  getFolderTree: () => FolderTreeNode[]
  getKeywordsForFolder: (folderId: number) => Keyword[]
  getKeywordsForConversation: (conversationId: string) => Keyword[]
  getFoldersForConversation: (conversationId: string) => Folder[]
  getConversationsForFolder: (folderId: number) => string[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Build folder tree from flat list
// ─────────────────────────────────────────────────────────────────────────────

export const buildFolderTree = (
  folders: Folder[],
  keywords: Keyword[],
  folderKeywordLinks: FolderKeywordLink[],
  conversationKeywordLinks: ConversationKeywordLink[]
): FolderTreeNode[] => {
  const activeFolders = folders.filter(f => !f.deleted)
  const activeKeywords = keywords.filter(k => !k.deleted)
  const keywordById = new Map<number, Keyword>()
  activeKeywords.forEach(k => keywordById.set(k.id, k))

  // Build keyword lookup by folder
  const keywordsByFolder = new Map<number, Keyword[]>()
  const keywordIdsByFolder = new Map<number, Set<number>>()
  folderKeywordLinks.forEach(link => {
    const kw = keywordById.get(link.keyword_id)
    if (!kw) return

    let seen = keywordIdsByFolder.get(link.folder_id)
    if (!seen) {
      seen = new Set<number>()
      keywordIdsByFolder.set(link.folder_id, seen)
    }

    if (seen.has(kw.id)) return
    seen.add(kw.id)

    let existing = keywordsByFolder.get(link.folder_id)
    if (!existing) {
      existing = []
      keywordsByFolder.set(link.folder_id, existing)
    }
    existing.push(kw)
  })

  // Count conversations per folder (via keywords)
  const conversationsByKeyword = new Map<number, Set<string>>()
  conversationKeywordLinks.forEach(link => {
    const existing = conversationsByKeyword.get(link.keyword_id) || new Set()
    existing.add(link.conversation_id)
    conversationsByKeyword.set(link.keyword_id, existing)
  })

  // Create nodes
  const nodeMap = new Map<number, FolderTreeNode>()
  activeFolders.forEach(folder => {
    const folderKeywords = keywordsByFolder.get(folder.id) || []
    const conversationIds = new Set<string>()
    folderKeywords.forEach(kw => {
      const convs = conversationsByKeyword.get(kw.id)
      if (convs) convs.forEach(c => conversationIds.add(c))
    })

    nodeMap.set(folder.id, {
      key: folder.id,
      title: folder.name,
      folder,
      children: [],
      keywords: folderKeywords,
      itemCount: conversationIds.size
    })
  })

  // Build hierarchy
  const roots: FolderTreeNode[] = []
  activeFolders.forEach(folder => {
    const node = nodeMap.get(folder.id)!
    if (folder.parent_id != null && nodeMap.has(folder.parent_id)) {
      nodeMap.get(folder.parent_id)!.children.push(node)
    } else {
      roots.push(node)
    }
  })

  // Sort children by name
  const sortChildren = (nodes: FolderTreeNode[]) => {
    nodes.sort((a, b) => a.title.localeCompare(b.title))
    nodes.forEach(n => sortChildren(n.children))
  }
  sortChildren(roots)

  return roots
}

// ─────────────────────────────────────────────────────────────────────────────
// Throttled Storage Adapter
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates a throttled localStorage adapter that batches writes to avoid
 * exceeding browser storage quota limits (MAX_WRITE_OPERATIONS_PER_MINUTE = 120 in Chrome).
 */
const createThrottledLocalStorage = (delay = 1000): StateStorage => {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  let pendingValue: string | null = null

  return {
    getItem: (name) => localStorage.getItem(name),
    setItem: (name, value) => {
      pendingValue = value
      if (!timeoutId) {
        timeoutId = setTimeout(() => {
          if (pendingValue !== null) {
            try {
              localStorage.setItem(name, pendingValue)
            } catch (e) {
              // Ignore storage quota errors silently
              // Ignore storage quota errors silently
            }
            pendingValue = null
          }
          timeoutId = null
        }, delay)
      }
    },
    removeItem: (name) => localStorage.removeItem(name)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useFolderStore = createWithEqualityFn<FolderState>()(
  persist(
    (set, get) => ({
      // Initial state
      folders: [],
      keywords: [],
      folderKeywordLinks: [],
      conversationKeywordLinks: [],
      isLoading: false,
      lastSynced: null,
      error: null,
      folderApiAvailable: null, // null = not yet checked
      viewMode: 'date',
      uiPrefs: {},

      // Setters
      setFolders: (folders) => set({ folders }),
      setKeywords: (keywords) => set({ keywords }),
      setFolderKeywordLinks: (links) => set({ folderKeywordLinks: links }),
      setConversationKeywordLinks: (links) => set({ conversationKeywordLinks: links }),

      // UI actions
      setViewMode: (mode) => set({ viewMode: mode }),

      toggleFolderOpen: (folderId) => set((state) => ({
        uiPrefs: {
          ...state.uiPrefs,
          [folderId]: {
            ...state.uiPrefs[folderId],
            isOpen: !(state.uiPrefs[folderId]?.isOpen ?? true)
          }
        }
      })),

      setFolderColor: (folderId, color) =>
        set((state) => {
          const prev = state.uiPrefs[folderId]
          return {
            uiPrefs: {
              ...state.uiPrefs,
              [folderId]: {
                ...prev,
                isOpen: prev?.isOpen ?? true,
                color
              }
            }
          }
        }),

      expandAllFolders: () => set((state) => {
        const newPrefs = { ...state.uiPrefs }
        state.folders.filter(f => !f.deleted).forEach(f => {
          newPrefs[f.id] = { ...newPrefs[f.id], isOpen: true }
        })
        return { uiPrefs: newPrefs }
      }),

      collapseAllFolders: () => set((state) => {
        const newPrefs = { ...state.uiPrefs }
        state.folders.filter(f => !f.deleted).forEach(f => {
          newPrefs[f.id] = { ...newPrefs[f.id], isOpen: false }
        })
        return { uiPrefs: newPrefs }
      }),

      // Server sync
      refreshFromServer: async () => {
        const state = get()

        // Skip if we already know the folder API is not available
        if (state.folderApiAvailable === false) {
          return
        }

        set({ isLoading: true, error: null })
        const controller = new AbortController()
        const timeoutMs = 15000
        const timeoutId = setTimeout(() => {
          try {
            controller.abort()
          } catch {
            // ignore abort errors
          }
        }, timeoutMs)

        try {
          const [foldersRes, keywordsRes, folderKeywordLinksRes, conversationKeywordLinksRes] = await Promise.all([
            fetchFolders({ abortSignal: controller.signal, timeoutMs }),
            fetchKeywords({ abortSignal: controller.signal, timeoutMs }),
            fetchFolderKeywordLinks({ abortSignal: controller.signal, timeoutMs }),
            fetchConversationKeywordLinks(undefined, { abortSignal: controller.signal, timeoutMs })
          ])

          const responses = [
            foldersRes,
            keywordsRes,
            folderKeywordLinksRes,
            conversationKeywordLinksRes
          ]
          const anyError = responses.some((res) => !res.ok)
          const hasNotFound = responses.some((res) => res.status === 404)

          if (hasNotFound) {
            // Folder API not available (404) - disabling folder sync
            set({
              isLoading: false,
              folderApiAvailable: false
            })
            clearTimeout(timeoutId)
            return
          }

          if (anyError) {
            const firstError =
              foldersRes.error ||
              keywordsRes.error ||
              folderKeywordLinksRes.error ||
              conversationKeywordLinksRes.error
            throw new Error(firstError || 'Failed to sync')
          }

          const folders = foldersRes.data || []
          const keywords = keywordsRes.data || []
          const folderKeywordLinks = folderKeywordLinksRes.data || []
          const conversationKeywordLinks = conversationKeywordLinksRes.data || []

          // Cache to Dexie (atomic)
          await db.transaction(
            "rw",
            db.folders,
            db.keywords,
            db.folderKeywordLinks,
            db.conversationKeywordLinks,
            async () => {
              await Promise.all([
                db.folders.clear(),
                db.keywords.clear(),
                db.folderKeywordLinks.clear(),
                db.conversationKeywordLinks.clear()
              ])

              await Promise.all([
                db.folders.bulkPut(folders),
                db.keywords.bulkPut(keywords),
                db.folderKeywordLinks.bulkPut(folderKeywordLinks),
                db.conversationKeywordLinks.bulkPut(conversationKeywordLinks)
              ])
            }
          )

          set({
            folders,
            keywords,
            folderKeywordLinks,
            conversationKeywordLinks,
            lastSynced: Date.now(),
            isLoading: false,
            folderApiAvailable: true // Mark API as available
          })
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error)
          const errorStatus =
            typeof (error as { status?: number | string })?.status !== "undefined"
              ? Number((error as { status?: number | string }).status)
              : undefined
          const is404 = errorStatus === 404 || errorMsg.includes('404')

          if (is404) {
            // Folder API not available (404) - disabling folder sync
            set({
              isLoading: false,
              folderApiAvailable: false
            })
            clearTimeout(timeoutId)
            return
          }

          console.error('Failed to refresh folders from server:', error)
          set({
            error: errorMsg,
            isLoading: false
          })

          // Fall back to cached data
          try {
            const [folders, keywords, folderKeywordLinks, conversationKeywordLinks] = await Promise.all([
              db.folders.toArray(),
              db.keywords.toArray(),
              db.folderKeywordLinks.toArray(),
              db.conversationKeywordLinks.toArray()
            ])
            set((state) => {
              const prevLastSynced = state.lastSynced
              if (prevLastSynced) {
                return { folders, keywords, folderKeywordLinks, conversationKeywordLinks }
              }

              const hasCachedData =
                folders.length > 0 ||
                keywords.length > 0 ||
                folderKeywordLinks.length > 0 ||
                conversationKeywordLinks.length > 0
              if (!hasCachedData) {
                return { folders, keywords, folderKeywordLinks, conversationKeywordLinks }
              }

              const candidates: number[] = []
              const addTs = (value?: string | null) => {
                if (!value) return
                const ts = Date.parse(value)
                if (Number.isFinite(ts)) candidates.push(ts)
              }

              for (const item of folders) {
                addTs(item.last_modified)
                addTs(item.created_at)
              }
              for (const item of keywords) {
                addTs(item.last_modified)
                addTs(item.created_at)
              }

              const guessedLastSynced = candidates.length > 0 ? Math.max(...candidates) : Date.now()

              return {
                folders,
                keywords,
                folderKeywordLinks,
                conversationKeywordLinks,
                lastSynced: guessedLastSynced
              }
            })
          } catch {
            // Ignore cache read errors
          }
        } finally {
          clearTimeout(timeoutId)
        }
      },

      // CRUD operations
      createFolder: async (name, parentId) => {
        const result = await apiCreateFolder(name, parentId)
        if (result.ok && result.data) {
          const folder = result.data
          set((state) => ({
            folders: [...state.folders, folder],
            uiPrefs: {
              ...state.uiPrefs,
              [folder.id]: { isOpen: true }
            }
          }))
          await db.folders.put(folder)
          return folder
        }
        return null
      },

      updateFolder: async (id, data) => {
        const result = await apiUpdateFolder(id, data)
        if (result.ok && result.data) {
          const folder = result.data
          set((state) => ({
            folders: state.folders.map(f => f.id === id ? folder : f)
          }))
          await db.folders.put(folder)
          return true
        }
        return false
      },

      deleteFolder: async (id) => {
        const result = await apiDeleteFolder(id)
        if (result.ok) {
          set((state) => {
            const { [id]: _removed, ...restPrefs } = state.uiPrefs
            return {
              folders: state.folders.map(f =>
                f.id === id ? { ...f, deleted: true } : f
              ),
              uiPrefs: restPrefs
            }
          })
          await db.folders.update(id, { deleted: true })
          return true
        }
        return false
      },

      createKeyword: async (keyword) => {
        const result = await apiCreateKeyword(keyword)
        if (result.ok && result.data) {
          const kw = result.data
          set((state) => ({
            keywords: [...state.keywords, kw]
          }))
          await db.keywords.put(kw)
          return kw
        }
        return null
      },

      ensureKeyword: async (keyword) => {
        const trimmed = keyword.trim()
        if (!trimmed) return null
        const existing = findKeywordByName(get().keywords, trimmed)
        if (existing) return existing
        return await get().createKeyword(trimmed)
      },

      addKeywordToConversation: async (conversationId, keywordId) => {
        const result = await apiLinkKeywordToConversation(
          conversationId,
          keywordId
        )
        if (result.ok) {
          const link: ConversationKeywordLink = {
            conversation_id: conversationId,
            keyword_id: keywordId
          }
          set((state) => {
            const exists = state.conversationKeywordLinks.some(
              (l) =>
                l.conversation_id === link.conversation_id &&
                l.keyword_id === link.keyword_id
            )
            return exists
              ? state
              : { conversationKeywordLinks: [...state.conversationKeywordLinks, link] }
          })
          await db.conversationKeywordLinks.put(link)
          return true
        }
        return false
      },

      removeKeywordFromConversation: async (conversationId, keywordId) => {
        const result = await apiUnlinkKeywordFromConversation(
          conversationId,
          keywordId
        )
        if (result.ok) {
          const linkKey = `${conversationId}::${keywordId}`
          const removeLinkFromState = () => {
            set((state) => ({
              conversationKeywordLinks: state.conversationKeywordLinks.filter(
                (l) => `${l.conversation_id}::${l.keyword_id}` !== linkKey
              )
            }))
          }
          try {
            await db.conversationKeywordLinks
              .where('[conversation_id+keyword_id]')
              .equals([conversationId, keywordId])
              .delete()
            removeLinkFromState()
          } catch (error) {
            console.error(
              'Failed to delete conversation-keyword link from cache:',
              error
            )
            removeLinkFromState()
          }
          return true
        }
        return false
      },

      addKeywordToFolder: async (folderId, keywordId) => {
        const result = await apiLinkKeywordToFolder(folderId, keywordId)
        if (result.ok) {
          const link: FolderKeywordLink = {
            folder_id: folderId,
            keyword_id: keywordId
          }
          set((state) => {
            const exists = state.folderKeywordLinks.some(
              (l) => l.folder_id === link.folder_id && l.keyword_id === link.keyword_id
            )
            return exists
              ? state
              : { folderKeywordLinks: [...state.folderKeywordLinks, link] }
          })
          await db.folderKeywordLinks.put(link)
          return true
        }
        return false
      },

      addConversationToFolder: async (conversationId, folderId) => {
        const state = get()
        // Find a keyword associated with this folder, or create one
        let keyword = state.keywords.find(
          (k) =>
            !k.deleted &&
            state.folderKeywordLinks.some(
              (l) => l.folder_id === folderId && l.keyword_id === k.id
            )
        )

        let createdKeywordForFolder = false

        if (!keyword) {
          // Create a keyword with the folder name
          const folder = state.folders.find(f => f.id === folderId)
          if (!folder) return false
          keyword = await get().createKeyword(folder.name)
          if (!keyword) return false
          createdKeywordForFolder = true

          const linked = await get().addKeywordToFolder(folderId, keyword.id)
          if (!linked) {
            await apiDeleteKeyword(keyword.id)
            set((state) => ({
              keywords: state.keywords.map((k) =>
                k.id === keyword.id ? { ...k, deleted: true } : k
              )
            }))
            await db.keywords.update(keyword.id, { deleted: true })
            return false
          }
        }

        const keywordId = keyword.id

        const linkResult = await apiLinkKeywordToConversation(conversationId, keywordId)
        const success = linkResult.ok

        if (!success && createdKeywordForFolder) {
          await apiUnlinkKeywordFromFolder(folderId, keywordId)

          set((state) => ({
            folderKeywordLinks: state.folderKeywordLinks.filter(
              (l) => !(l.folder_id === folderId && l.keyword_id === keywordId)
            )
          }))

          await db.folderKeywordLinks
            .where('[folder_id+keyword_id]')
            .equals([folderId, keywordId])
            .delete()

          await apiDeleteKeyword(keywordId)
          set((state) => ({
            keywords: state.keywords.map((k) =>
              k.id === keywordId ? { ...k, deleted: true } : k
            )
          }))
          await db.keywords.update(keywordId, { deleted: true })
        }
        if (success) {
          const link: ConversationKeywordLink = {
            conversation_id: conversationId,
            keyword_id: keywordId
          }
          set((state) => {
            const exists = state.conversationKeywordLinks.some(
              (l) => l.conversation_id === link.conversation_id && l.keyword_id === link.keyword_id
            )
            return exists
              ? state
              : { conversationKeywordLinks: [...state.conversationKeywordLinks, link] }
          })
          await db.conversationKeywordLinks.put(link)
        }
        return success
      },

      removeConversationFromFolder: async (conversationId, folderId) => {
        const state = get()
        // Find keywords associated with this folder
        const folderKeywordIds = new Set(
          state.folderKeywordLinks
            .filter((l) => l.folder_id === folderId)
            .map((l) => l.keyword_id)
        )

        // Find which of these keywords are linked to the conversation
        const linksToRemove = state.conversationKeywordLinks.filter(
          (l) => l.conversation_id === conversationId && folderKeywordIds.has(l.keyword_id)
        )

        if (linksToRemove.length === 0) return true

        const results = await Promise.all(
          linksToRemove.map(async (link) => {
            const result = await apiUnlinkKeywordFromConversation(conversationId, link.keyword_id)
            return {
              ok: result.ok,
              link
            }
          })
        )

        if (!results.every((r) => r.ok)) {
          return false
        }

        try {
          await db.transaction("rw", db.conversationKeywordLinks, async () => {
            await Promise.all(
              linksToRemove.map((link) =>
                db.conversationKeywordLinks
                  .where('[conversation_id+keyword_id]')
                  .equals([link.conversation_id, link.keyword_id])
                  .delete()
              )
            )
          })
        } catch (error) {
          console.error('Failed to delete conversation-keyword links from cache:', error)
          // Continue to update in-memory state even if DB sync fails
        }

        const removedKeySet = new Set(
          linksToRemove.map((link) => `${link.conversation_id}::${link.keyword_id}`)
        )

        set((state) => ({
          conversationKeywordLinks: state.conversationKeywordLinks.filter(
            (l) => !removedKeySet.has(`${l.conversation_id}::${l.keyword_id}`)
          )
        }))

        return true
      },

      // Computed helpers
      getFolderTree: () => {
        const state = get()
        return buildFolderTree(
          state.folders,
          state.keywords,
          state.folderKeywordLinks,
          state.conversationKeywordLinks
        )
      },

      getKeywordsForFolder: (folderId) => {
        const state = get()
        const keywordIds = state.folderKeywordLinks
          .filter(l => l.folder_id === folderId)
          .map(l => l.keyword_id)
        return state.keywords.filter(k => keywordIds.includes(k.id) && !k.deleted)
      },

      getKeywordsForConversation: (conversationId) => {
        const state = get()
        const keywordIds = state.conversationKeywordLinks
          .filter((l) => l.conversation_id === conversationId)
          .map((l) => l.keyword_id)
        return state.keywords.filter(
          (k) => keywordIds.includes(k.id) && !k.deleted
        )
      },

      getFoldersForConversation: (conversationId) => {
        const state = get()
        // Get keyword IDs for this conversation
        const keywordIds = state.conversationKeywordLinks
          .filter(l => l.conversation_id === conversationId)
          .map(l => l.keyword_id)

        // Get folder IDs that contain these keywords
        const folderIds = new Set(
          state.folderKeywordLinks
            .filter(l => keywordIds.includes(l.keyword_id))
            .map(l => l.folder_id)
        )

        return state.folders.filter(f => folderIds.has(f.id) && !f.deleted)
      },

      getConversationsForFolder: (folderId) => {
        const state = get()
        // Get keyword IDs for this folder
        const keywordIds = state.folderKeywordLinks
          .filter(l => l.folder_id === folderId)
          .map(l => l.keyword_id)

        // Get conversation IDs that have these keywords
        const conversationIds = new Set(
          state.conversationKeywordLinks
            .filter(l => keywordIds.includes(l.keyword_id))
            .map(l => l.conversation_id)
        )

        return Array.from(conversationIds)
      }
    }),
    {
      name: 'tldw-folder-store',
      // Use throttled storage to avoid exceeding browser write quota limits
      storage: createJSONStorage(() => createThrottledLocalStorage(1000)),
      // Persist UI prefs + cache metadata (not the server data itself).
      // NOTE: `folderApiAvailable` is intentionally NOT persisted. It is a
      // transient runtime signal: a single 404 sets it false to avoid hammering
      // a server that lacks the folder API for the rest of the session, but it
      // must reset to `null` (unknown, retryable) on the next session so one
      // transient 404 cannot disable folder sync forever.
      partialize: (state) => ({
        uiPrefs: state.uiPrefs,
        viewMode: state.viewMode,
        lastSynced: state.lastSynced
      }),
      // Never rehydrate `folderApiAvailable` from storage. Besides not persisting
      // it going forward (see partialize), this strips any stale `false` written
      // by an earlier build so users already "stuck" with folder sync disabled
      // recover to the retryable `null` default on their next load. Behaves like
      // the default shallow merge otherwise.
      merge: (persistedState, currentState) => {
        const persisted = (persistedState ?? {}) as Partial<FolderState>
        const { folderApiAvailable: _legacyFolderApiAvailable, ...rest } =
          persisted
        return {
          ...currentState,
          ...rest
        }
      }
    }
  )
)

// ─────────────────────────────────────────────────────────────────────────────
// Selectors (for performance - avoid re-renders)
// ─────────────────────────────────────────────────────────────────────────────

export const useFolders = () => useFolderStore((s) => s.folders)
export const useKeywords = () => useFolderStore((s) => s.keywords)
export const useFolderViewMode = () => useFolderStore((s) => s.viewMode)
export const useFolderIsLoading = () => useFolderStore((s) => s.isLoading)
export const useFolderUIPrefs = () => useFolderStore((s) => s.uiPrefs)

export const useFolderActions = () =>
  useFolderStore(
    useShallow((s) => ({
      setViewMode: s.setViewMode,
      toggleFolderOpen: s.toggleFolderOpen,
      expandAllFolders: s.expandAllFolders,
      collapseAllFolders: s.collapseAllFolders,
      refreshFromServer: s.refreshFromServer,
      createFolder: s.createFolder,
      updateFolder: s.updateFolder,
      deleteFolder: s.deleteFolder,
      ensureKeyword: s.ensureKeyword,
      addKeywordToConversation: s.addKeywordToConversation,
      removeKeywordFromConversation: s.removeKeywordFromConversation,
      addConversationToFolder: s.addConversationToFolder,
      removeConversationFromFolder: s.removeConversationFromFolder
    }))
  )
