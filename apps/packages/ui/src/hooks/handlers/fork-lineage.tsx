/**
 * Lineage for a branch of a saved server chat (CM-12).
 *
 * New Branch used to create an orphan called "Forked conversation". The
 * branch now records its parent chat and the message it starts from, takes a
 * title that says so, and opens with a notice that leads back to the parent.
 */
import React from "react"
import i18n from "i18next"
import { Button } from "antd"
import type { NotificationInstance } from "antd/es/notification/interface"
import type { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { UploadedFile } from "@/db/dexie/types"
import type { ChatScope } from "@/types/chat-scope"

const BRANCH_TITLE_DEFAULT = "{{title}} — branch at message {{number}}"
const TITLE_BASE_LIMIT = 60
const SAMPLE_TITLE = "\u0001"
const SAMPLE_NUMBER = 987654321

/**
 * The localized " — branch at message N" ending this module adds, so a
 * branch of a branch replaces it instead of piling up endings.
 */
const branchSuffixPattern = (): RegExp | null => {
  const sample = i18n.t("playground:historySelection.branchTitle", {
    defaultValue: BRANCH_TITLE_DEFAULT,
    title: SAMPLE_TITLE,
    number: SAMPLE_NUMBER
  })
  if (typeof sample !== "string") return null
  const at = sample.indexOf(SAMPLE_TITLE)
  const suffix = at >= 0 ? sample.slice(at + SAMPLE_TITLE.length) : ""
  if (!suffix.trim() || !suffix.includes(String(SAMPLE_NUMBER))) return null
  const escaped = suffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(String(SAMPLE_NUMBER), "\\d+")
  return new RegExp(`${escaped}$`)
}

/**
 * "<parent title> — branch at message N", or "Branch at message N" for an
 * untitled chat. N counts the chat's questions and replies up to the branch
 * point. The parent title loses its own branch ending and is cut to 60
 * characters, so titles don't grow with each branch. `null` when the branch
 * point is unknown, so the server keeps its own default.
 */
export const nativeBranchTitle = (
  parentTitle: string | null | undefined,
  messageCount: number
): string | null => {
  if (!Number.isSafeInteger(messageCount) || messageCount < 1) return null
  let base = parentTitle?.trim() ?? ""
  const suffix = branchSuffixPattern()
  while (suffix && suffix.test(base)) base = base.replace(suffix, "").trim()
  base = base.slice(0, TITLE_BASE_LIMIT).trim()
  const title = base
    ? i18n.t("playground:historySelection.branchTitle", {
        defaultValue: BRANCH_TITLE_DEFAULT,
        title: base,
        number: messageCount
      })
    : i18n.t("playground:historySelection.branchTitleUntitled", {
        defaultValue: "Branch at message {{number}}",
        number: messageCount
      })
  return typeof title === "string" && title.trim() ? title : null
}

/** The questions and replies a branch copies, which its title counts. */
export const countBranchMessages = (rows: readonly { readonly role?: unknown }[]): number =>
  rows.filter((row) => row.role === "user" || row.role === "assistant").length

/** The draft fields an opened saved server chat resets, as New Branch sets them. */
export type NativeChatOpenSetters = {
  setHistoryId: (id: string | null) => void
  setContext?: (context: UploadedFile[]) => void
  setSelectedSystemPrompt?: (prompt: string) => void
  setSystemPrompt?: (prompt: string) => void
  setServerChatMetaLoaded?: (loaded: boolean) => void
  setServerChatId?: (id: string | null) => void
}

/**
 * Open a saved server chat in place through the history selection. The chat
 * is shown only for the exact load receipt of that chat, in the same scope,
 * while the draft the caller captured is still current.
 */
export const openNativeChat = (
  controller: HistorySelectionController,
  target: { chatId: string; ownerKey: string; scope?: ChatScope },
  acceptDraft: ReturnType<typeof captureChatDraftLoad>,
  setters: NativeChatOpenSetters,
  onOpened?: (chatId: string) => void
) =>
  controller.loadConversation(
    { serverChatId: target.chatId, scope: target.scope, isCurrent: acceptDraft.isCurrent },
    undefined,
    (receipt) => {
      const live = controller.getCurrent()
      if (
        live.status !== "ready" ||
        live.owner !== receipt.owner ||
        live.view !== receipt.view ||
        receipt.owner.kind !== "native" ||
        receipt.owner.conversation_id !== target.chatId ||
        receipt.view.conversation_id !== target.chatId ||
        receipt.view.owner_key !== target.ownerKey ||
        !receipt.owner.validate_lease()
      )
        return
      const expected = target.scope ?? { type: "global" }
      const actual = receipt.owner.scope ?? { type: "global" }
      if (
        actual.type !== expected.type ||
        (actual.type === "workspace" &&
          expected.type === "workspace" &&
          actual.workspaceId !== expected.workspaceId)
      )
        return
      if (!acceptDraft(receipt)) return
      setters.setHistoryId(null)
      setters.setContext?.([])
      setters.setSelectedSystemPrompt?.("")
      setters.setSystemPrompt?.("")
      setters.setServerChatMetaLoaded?.(false)
      setters.setServerChatId?.(target.chatId)
      onOpened?.(target.chatId)
    }
  )

let announcement = 0

/** Tell the user the branch is open, and offer the way back to its parent. */
export const announceNativeBranch = (
  notification: Pick<NotificationInstance, "success" | "destroy">,
  { branchTitle, parentTitle, backToParent }: {
    branchTitle: string | null
    parentTitle: string | null | undefined
    backToParent: () => void
  }
) => {
  const key = `native-branch-${++announcement}`
  const parent = parentTitle?.trim()
  notification.success({
    key,
    message: i18n.t("playground:historySelection.branchCreated", "Branch created"),
    description: parent
      ? i18n.t("playground:historySelection.branchCreatedFrom", {
          defaultValue: "This chat continues from \"{{title}}\". The original chat is unchanged.",
          title: parent
        })
      : branchTitle ?? undefined,
    actions: (
      <Button
        size="small"
        onClick={() => {
          // A test or minimal host may not provide destroy; the notice then times out.
          notification.destroy?.(key)
          backToParent()
        }}
      >
        {i18n.t("playground:historySelection.backToParent", "Back to parent")}
      </Button>
    )
  })
}
