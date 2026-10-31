import React, { useEffect, useRef, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Empty, Skeleton, message, Modal, Input } from "antd"
import { useQuery } from "@tanstack/react-query"
import { FolderPlus } from "lucide-react"

import { useConnectionState } from "@/hooks/useConnectionState"
import { useServerChatHistory } from "@/hooks/useServerChatHistory"
import { isRecoverableAuthConfigError } from "@/services/auth-errors"
import { FolderTree } from "@/components/Folders"
import {
  useFolderStore,
  useFolderActions
} from "@/store/folder"
import { useConnectionStore } from "@/store/connection"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { cn } from "@/libs/utils"

interface FolderChatListProps {
  className?: string
  onConversationSelected?: () => void
}

const MISSING_CHAT_BATCH_SIZE = 5

export function FolderChatList({ className, onConversationSelected }: FolderChatListProps) {
  const { t } = useTranslation(["common"])
  const { isConnected } = useConnectionState()
  const checkConnection = useConnectionStore((state) => state.checkOnce)
  const { refreshFromServer, createFolder } = useFolderActions()
  const folderRefreshInFlightRef = useRef<Promise<void> | null>(null)
  const [newFolderModalOpen, setNewFolderModalOpen] = useState(false)
  const [newFolderName, setNewFolderName] = useState("")
  const [isCreating, setIsCreating] = useState(false)

  const selectServerChat = useSelectServerChat()

  // Folder data
  const conversationKeywordLinks = useFolderStore((s) => s.conversationKeywordLinks)
  const folders = useFolderStore((s) => s.folders)
  const isFolderLoading = useFolderStore((s) => s.isLoading)

  // Load folders when component mounts
  useEffect(() => {
    if (!isConnected) return
    if (folderRefreshInFlightRef.current) return

    const refreshPromise = refreshFromServer().finally(() => {
      if (folderRefreshInFlightRef.current === refreshPromise) {
        folderRefreshInFlightRef.current = null
      }
    })
    folderRefreshInFlightRef.current = refreshPromise
  }, [isConnected, refreshFromServer])

  // Get all conversation IDs from folder links
  const folderConversationIds = useMemo(() => {
    const ids = new Set<string>()
    conversationKeywordLinks.forEach((link) => {
      if (link.conversation_id) {
        ids.add(link.conversation_id)
      }
    })
    return Array.from(ids).sort((a, b) => a.localeCompare(b))
  }, [conversationKeywordLinks])

  const { data: serverChatData, isLoading: isServerChatsLoading } =
    useServerChatHistory("")
  const serverChats = serverChatData || []
  const serverChatById = useMemo(
    () => new Map(serverChats.map((chat) => [chat.id, chat])),
    [serverChats]
  )

  const missingFolderConversationIds = useMemo(() => {
    return folderConversationIds.filter(
      (conversationId) => !serverChatById.has(conversationId)
    )
  }, [folderConversationIds, serverChatById])

  const { data: missingFolderChats = [], isLoading: isMissingChatsLoading } = useQuery({
    queryKey: ["folderConversationMissingChats", missingFolderConversationIds],
    queryFn: async () => {
      try {
        await tldwClient.initialize()

        // No batch chat lookup endpoint yet; fetch missing chats in parallel and cache.
        const results: ServerChatSummary[] = []
        for (
          let index = 0;
          index < missingFolderConversationIds.length;
          index += MISSING_CHAT_BATCH_SIZE
        ) {
          const batch = missingFolderConversationIds.slice(
            index,
            index + MISSING_CHAT_BATCH_SIZE
          )
          const batchResults = await Promise.all(
            batch.map(async (conversationId) => {
              try {
                return await tldwClient.getChat(conversationId)
              } catch (error) {
                if (isRecoverableAuthConfigError(error)) {
                  return null
                }
                console.error(
                  "Failed to load server chat info for folder conversation:",
                  conversationId,
                  error
                )
                return null
              }
            })
          )
          results.push(...(batchResults.filter(Boolean) as ServerChatSummary[]))
        }
        return results
      } catch (error) {
        if (isRecoverableAuthConfigError(error)) {
          void checkConnection().catch(() => null)
          return []
        }
        console.error("Failed to initialize tldw client for folder chats:", error)
        throw error
      }
    },
    enabled: isConnected && missingFolderConversationIds.length > 0,
    staleTime: 60_000,
    retry: (failureCount, error) =>
      !isRecoverableAuthConfigError(error) && failureCount < 1
  })

  const missingFolderChatById = useMemo(
    () => new Map(missingFolderChats.map((chat) => [chat.id, chat])),
    [missingFolderChats]
  )

  const isTitlesLoading = isServerChatsLoading || isMissingChatsLoading

  // Build conversations list for folder tree
  const folderTreeConversations = useMemo(() => {
    const titleById = new Map<string, string>()
    serverChats.forEach((chat) => {
      titleById.set(chat.id, chat.title || "")
    })
    missingFolderChats.forEach((chat) => {
      titleById.set(chat.id, chat.title || "")
    })
    return folderConversationIds.map((conversationId) => ({
      id: conversationId,
      title: titleById.get(conversationId) || conversationId
    }))
  }, [folderConversationIds, serverChats, missingFolderChats])

  const hasFolders = useMemo(
    () => folders.some((folder) => !folder.deleted),
    [folders]
  )

  const loadServerChatById = React.useCallback(
    async (conversationId: string) => {
      const cachedChat =
        serverChatById.get(conversationId) ||
        missingFolderChatById.get(conversationId)
      if (cachedChat) {
        selectServerChat(cachedChat)
        onConversationSelected?.()
        return
      }

      try {
        await tldwClient.initialize()
        const chat = await tldwClient.getChat(conversationId)
        selectServerChat(chat)
        onConversationSelected?.()
      } catch (error) {
        console.error(
          "Failed to load server chat for folder conversation:",
          conversationId,
          error
        )
        message.error(
          t("common:error.friendlyGenericSummary", {
            defaultValue: "Something went wrong while talking to your tldw server."
          })
        )
      }
    },
    [missingFolderChatById, onConversationSelected, selectServerChat, serverChatById, t]
  )

  const handleCreateFolder = async () => {
    const trimmedName = newFolderName.trim()
    if (!trimmedName || isCreating) return

    setIsCreating(true)
    try {
      await createFolder(trimmedName)
      setNewFolderName("")
      setNewFolderModalOpen(false)
      message.success(
        t("common:success.folderCreated", {
          defaultValue: "Folder created successfully"
        })
      )
    } catch (error) {
      console.error("Failed to create folder:", error)
      message.error(
        (error as { message?: string })?.message ||
          t("common:error.createFolder", {
            defaultValue: "Failed to create folder"
          })
      )
    } finally {
      setIsCreating(false)
    }
  }

  // Not connected state
  if (!isConnected) {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <Empty
          description={t("common:chatSidebar.foldersRequireConnection", {
            defaultValue: "Folders require a server connection"
          })}
        />
      </div>
    )
  }

  // Loading state
  if (isTitlesLoading || isFolderLoading) {
    return (
      <div className={cn("flex justify-center items-center py-8", className)}>
        <Skeleton active paragraph={{ rows: 4 }} />
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col", className)}>
      <div className="flex items-center justify-end px-3 py-2">
        <button
          type="button"
          onClick={() => setNewFolderModalOpen(true)}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-subtle hover:bg-surface hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)]"
        >
          <FolderPlus className="size-3.5" />
          <span>{t("common:newFolder", { defaultValue: "New Folder" })}</span>
        </button>
      </div>

      {hasFolders ? (
        <FolderTree
          onConversationSelect={(conversationId) => {
            void loadServerChatById(conversationId)
          }}
          conversations={folderTreeConversations}
          showConversations
        />
      ) : (
        <div className="flex justify-center items-center py-8">
          <Empty
            description={t("common:noFolders", {
              defaultValue: "No folders yet. Create a folder to organize your chats."
            })}
          />
        </div>
      )}

      <Modal
        open={newFolderModalOpen}
        onCancel={() => {
          setNewFolderModalOpen(false)
          setNewFolderName("")
        }}
        onOk={handleCreateFolder}
        okText={t("common:create", { defaultValue: "Create" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={isCreating}
        okButtonProps={{ disabled: !newFolderName.trim() }}
        title={t("common:newFolder", { defaultValue: "New Folder" })}
      >
        <Input
          placeholder={t("common:folderName", { defaultValue: "Folder name" })}
          value={newFolderName}
          onChange={(e) => setNewFolderName(e.target.value)}
          onPressEnter={handleCreateFolder}
          autoFocus
        />
      </Modal>
    </div>
  )
}

export default FolderChatList
