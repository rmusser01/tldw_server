import type { ChatDraftHandle } from "@/utils/chat-draft-address"
import {
  acknowledgeChatCreation,
  createChatWithRecovery,
  markChatCreationBound,
  markChatCreationSettled,
  readChatCreationRecovery
} from "@/services/pending-chat-promotion"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw"
import type { DocumentChatRequestOptions, DocumentChatTarget } from "@/services/tldw/domains/media"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"

type Receipt = { id: string; record: string; title: string }
type PendingChat = { key: string; receipt?: Receipt }

const storageKey = (target: DocumentChatTarget, options: DocumentChatRequestOptions) =>
  `document-chat-creation:${chatDraftOwnerKey(options.requestScope)}:${target.mediaUuid}`

const draftHandle = (key: string, options: DocumentChatRequestOptions): ChatDraftHandle => ({
  ownerHash: chatDraftOwnerKey(options.requestScope),
  address: { kind: "fresh", id: key, scope: { type: "global" }, temporary: false }
})

export function readPendingDocumentChats(target: DocumentChatTarget, options: DocumentChatRequestOptions): PendingChat[] {
  const value: unknown = JSON.parse(sessionStorage.getItem(storageKey(target, options)) ?? "[]")
  return Array.isArray(value) ? value.filter(row => typeof row?.key === "string") : []
}

function writePendingChat(target: DocumentChatTarget, options: DocumentChatRequestOptions, pending: PendingChat) {
  const rows = readPendingDocumentChats(target, options).filter(row => row.key !== pending.key)
  sessionStorage.setItem(storageKey(target, options), JSON.stringify([...rows, pending]))
}

export async function settleDocumentChatCreation(
  target: DocumentChatTarget, options: DocumentChatRequestOptions, key: string, chatId: string
) {
  await markChatCreationSettled(draftHandle(key, options), chatId)
  const rows = readPendingDocumentChats(target, options).filter(row => row.key !== key)
  sessionStorage.setItem(storageKey(target, options), JSON.stringify(rows))
}

export async function readPendingDocumentDraft(key: string, options: DocumentChatRequestOptions) {
  return (await readChatCreationRecovery(draftHandle(key, options)))?.inputText ?? ""
}

/** Persist the create address and exact acknowledgement before attempting to attach. */
export async function createLinkedDocumentChat({
  key, question, target, options, isCurrent
}: {
  key: string
  question: string
  target: DocumentChatTarget
  options: DocumentChatRequestOptions
  isCurrent: () => boolean
}): Promise<Receipt> {
  const handle = draftHandle(key, options)
  let receipt = readPendingDocumentChats(target, options).find(row => row.key === key)?.receipt
  if (!receipt) {
    writePendingChat(target, options, { key })
    const create = (id: string, input: string) => tldwClient.createChat(
      { id, title: input.slice(0, 80), source: "web-ui" },
      { ...options, documentTarget: target },
    )
    const remember = (id: string, chat: Awaited<ReturnType<typeof create>>) => {
      if (chat.id !== id || !chat.document_chat_record) throw new Error("The server did not confirm this chat.")
      const confirmed = { id, title: chat.title, record: chat.document_chat_record }
      writePendingChat(target, options, { key, receipt: confirmed })
      return confirmed
    }
    const attempt = await readChatCreationRecovery(handle)
    if (attempt?.clientChatId && attempt.state === "unconfirmed") {
      const id = attempt.clientChatId
      if (!isCurrent()) throw new Error("This conversation changed. Your draft is still available.")
      // Replay the original create so its fingerprint verifies the original intent.
      receipt = remember(id, await create(id, attempt.inputText))
      if (!(await acknowledgeChatCreation(attempt, id))) throw new Error("Could not confirm this chat save. Retry to check the same conversation.")
    } else {
      const snapshot = await loadServicePromptSnapshot([], options)
      try {
        await createChatWithRecovery({
          snapshot, handle, surface: "playground", isCurrent, inputText: question, setup: {},
          create: async id => {
            const chat = await create(id, question)
            receipt = remember(id, chat)
            return chat
          }
        })
      } finally {
        snapshot.release()
      }
    }
  }
  if (!receipt || !isCurrent()) throw new Error("This conversation changed. Your draft is still available.")
  await tldwClient.attachDocumentChat(target, receipt.id, receipt.record, options)
  if (!isCurrent()) throw new Error("This conversation changed. Your draft is still available.")
  await markChatCreationBound(handle, receipt.id)
  return receipt
}
