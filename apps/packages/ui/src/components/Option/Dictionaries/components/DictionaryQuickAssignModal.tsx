import { Button, Input, Modal, Skeleton } from "antd"
import React from "react"

type QuickAssignChatOption = {
  chat: any
  chatId: string
  title: string
  state: string
  disabledReason?: string
}

type DictionaryQuickAssignModalProps = {
  dictionaryName?: string
  open: boolean
  onCancel: () => void
  onConfirm: () => void
  selectedChatIds: string[]
  assignSaving: boolean
  searchValue: string
  onSearchChange: (value: string) => void
  chatsStatus: "pending" | "error" | "success" | string
  chatsError: unknown
  chatOptions: QuickAssignChatOption[]
  onRetry: () => void
  onToggleChatSelection: (chatId: string) => void
  onOpenChat: (chat: any) => void
}

export const DictionaryQuickAssignModal: React.FC<
  DictionaryQuickAssignModalProps
> = ({
  dictionaryName,
  open,
  onCancel,
  onConfirm,
  selectedChatIds,
  assignSaving,
  searchValue,
  onSearchChange,
  chatsStatus,
  chatsError,
  chatOptions,
  onRetry,
  onToggleChatSelection,
  onOpenChat
}) => {
  return (
    <Modal
      title={
        dictionaryName
          ? `Quick assign: ${dictionaryName}`
          : "Quick assign dictionary"
      }
      open={open}
      onCancel={onCancel}
      onOk={onConfirm}
      okText="Assignment unavailable"
      okButtonProps={{ disabled: true }}
      cancelButtonProps={{ disabled: assignSaving }}>
      <div className="space-y-3">
        <p className="text-xs text-text-muted">
          This server version does not support dictionary assignment to saved
          chats across the chat interfaces. Existing links remain unchanged;
          Open lets you inspect a conversation.
        </p>
        <Input
          value={searchValue}
          onChange={(event) => onSearchChange(event.target.value)}
          allowClear
          placeholder="Search chats by title or ID"
          aria-label="Search chats for quick assign"
        />

        {chatsStatus === "pending" && (
          <Skeleton active paragraph={{ rows: 3 }} />
        )}

        {chatsStatus === "error" && (
          <div className="space-y-2 rounded-md border border-danger/30 bg-danger/5 p-3">
            <p className="text-xs text-danger">
              {chatsError instanceof Error
                ? chatsError.message
                : "Unable to load chat sessions for assignment."}
            </p>
            <Button size="small" onClick={onRetry}>
              Retry
            </Button>
          </div>
        )}

        {chatsStatus === "success" && chatOptions.length === 0 && (
          <div className="rounded-md border border-border bg-surface2/40 px-3 py-2 text-xs text-text-muted">
            No chat sessions match your search.
          </div>
        )}

        {chatsStatus === "success" && chatOptions.length > 0 && (
          <div className="max-h-72 space-y-1 overflow-y-auto rounded-md border border-border bg-surface2/20 p-2">
            {chatOptions.map((option) => {
              const checked = selectedChatIds.includes(option.chatId)
              return (
                <div
                  key={`assign-chat-${option.chatId}`}
                  className="flex min-h-11 items-center gap-2 rounded-md px-2 py-2 text-sm hover:bg-surface2/60">
                  <input
                    disabled
                    aria-describedby={`assign-chat-reason-${option.chatId}`}
                    type="checkbox"
                    checked={checked}
                    onChange={() => onToggleChatSelection(option.chatId)}
                    aria-label={`Select chat ${option.title}`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="break-words">{option.title}</div>
                    <p
                      id={`assign-chat-reason-${option.chatId}`}
                      className="text-xs text-text-muted">
                      {option.disabledReason ||
                        "Dictionary assignment is unavailable for saved chats on this server version."}
                    </p>
                    <div className="text-[11px] text-text-muted">
                      {option.chatId} · {option.state}
                    </div>
                  </div>
                  <Button
                    type="link"
                    className="min-h-11"
                    aria-label={`Open chat ${option.title}`}
                    size="small"
                    onClick={(event) => {
                      event.preventDefault()
                      event.stopPropagation()
                      onOpenChat(option.chat)
                    }}>
                    Open
                  </Button>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </Modal>
  )
}
