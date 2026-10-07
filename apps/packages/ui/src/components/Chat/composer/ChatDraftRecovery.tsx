import type { ChatCreationRecovery } from "@/services/pending-chat-promotion"
import { Button } from "@/components/Common/Button"
import { useStoreMessageOption } from "@/store/option"
import type { ChatDraftAddress } from "@/utils/chat-draft-address"
import { Modal } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"

import type {
  ChatDraftRecoveryListing,
  RecoveredChatDraft,
  UseComposerTextResult
} from "./hooks/useComposerText"

export type RestoreChatDraft = (
  draft: RecoveredChatDraft,
  copy: (
    surfaceKey: string,
    destination?: Extract<ChatDraftAddress, { kind: "fresh" }>
  ) => Promise<boolean>
) => boolean | Promise<boolean>

export function ChatDraftRecovery({
  listDrafts,
  copyDraft,
  onRestore,
  creationRecovery,
  onReviewCreatedChat,
  onCopyCreationText,
  onFindConversations,
  draftOwnerVerificationState,
  retryDraftOwnerVerification,
  disabled = false
}: {
  listDrafts: () => Promise<ChatDraftRecoveryListing>
  copyDraft: (
    draft: RecoveredChatDraft,
    surfaceKey: string,
    destination?: Extract<ChatDraftAddress, { kind: "fresh" }>
  ) => Promise<boolean>
  onRestore?: RestoreChatDraft
  creationRecovery?: { active: ChatCreationRecovery | null; unavailable: boolean; copyInputRequired?: boolean }
  onReviewCreatedChat?: () => Promise<boolean>
  onCopyCreationText?: () => Promise<boolean>
  onFindConversations?: () => void
  disabled?: boolean
  draftOwnerVerificationState?: UseComposerTextResult["draftOwnerVerificationState"]
  retryDraftOwnerVerification?: () => void
}) {
  const { t } = useTranslation("playground")
  const text = (key: string, fallback: string) =>
    t(`draftRecovery.${key}`, fallback)
  const revision = useStoreMessageOption((state) => state.chatDraftRevision)
  const generation = React.useRef(0)
  const [open, setOpen] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [listing, setListing] = React.useState<ChatDraftRecoveryListing | null>(
    null
  )
  const [error, setError] = React.useState<string | null>(null)
  const invalidate = React.useCallback(() => {
    generation.current++
  }, [])
  const close = React.useCallback(() => {
    invalidate()
    setOpen(false)
    setListing(null)
    setError(null)
    setBusy(false)
  }, [invalidate])
  React.useLayoutEffect(() => {
    close()
    return invalidate
  }, [revision, close, invalidate])
  React.useLayoutEffect(() => {
    if (disabled) close()
  }, [disabled, close])
  const refresh = async () => {
    const token = ++generation.current
    setBusy(true)
    setError(null)
    try {
      const result = await listDrafts()
      if (token === generation.current) setListing(result)
    } catch {
      if (token === generation.current)
        setListing({ drafts: [], unavailable: true })
    } finally {
      if (token === generation.current) setBusy(false)
    }
  }
  const recover = async (draft: RecoveredChatDraft) => {
    if (!onRestore) return
    const token = ++generation.current
    setBusy(true)
    setError(null)
    try {
      const current = (await listDrafts()).drafts.find(
        (item) =>
          item.key === draft.key &&
          item.handle.address.temporary === draft.handle.address.temporary
      )
      if (token !== generation.current) return
      if (!current) {
        setError(
          text(
            "unavailable",
            "This draft is no longer available. Your current text is unchanged."
          )
        )
        return
      }
      const accepted = await onRestore(
        current,
        async (surface, destination) => {
          if (token !== generation.current) return false
          const copied = await copyDraft(current, surface, destination)
          return token === generation.current && copied
        }
      )
      if (token !== generation.current) return
      if (accepted) close()
      else
        setError(
          text(
            "notOpened",
            "Could not open this draft. Your current text and saved draft are kept."
          )
        )
    } catch {
      if (token === generation.current)
        setError(
          text(
            "notOpened",
            "Could not open this draft. Your current text and saved draft are kept."
          )
        )
    } finally {
      if (token === generation.current) setBusy(false)
    }
  }
  const review = async () => {
    if (!onReviewCreatedChat || busy || disabled) return
    const token = ++generation.current
    setBusy(true)
    setError(null)
    try {
      const accepted = await onReviewCreatedChat()
      if (token === generation.current && !accepted) setError(text("creationReviewFailed", "Could not verify the created chat. Your draft and recovery record are kept."))
    } catch {
      if (token === generation.current) setError(text("creationReviewFailed", "Could not verify the created chat. Your draft and recovery record are kept."))
    } finally { if (token === generation.current) setBusy(false) }
  }
  return (
    <div className="w-full">
      {draftOwnerVerificationState && draftOwnerVerificationState !== "ready" && (
        <div className="mb-2 rounded border border-border p-3 text-sm">
          <p role="status">
            {draftOwnerVerificationState === "pending"
              ? text(
                  "accountPending",
                  "Checking your account before restoring or saving drafts…"
                )
              : text(
                  "accountVerificationFailed",
                  "Your account could not be verified. Draft saving, recovery, and imports are paused. Retry or check your connection settings."
                )}
          </p>
          {draftOwnerVerificationState === "error" &&
            retryDraftOwnerVerification && (
              <Button
                variant="outline"
                size="lg"
                className="mt-2"
                onClick={retryDraftOwnerVerification}
              >
                {text("retryAccountVerification", "Retry account verification")}
              </Button>
            )}
        </div>
      )}
      {!disabled && creationRecovery?.active && <div role="status" className="mb-2 rounded border border-border p-3 text-sm">
        <p className="font-medium">{["acknowledged", "bound", "settled"].includes(creationRecovery.active.state) ? text("creationBound", "Chat already created") : creationRecovery.active.state === "prepared" ? text("creationPrepared", "Chat creation not started") : text("creationUnconfirmed", "Server save unconfirmed")}</p>
        <p>{creationRecovery.active.state === "prepared" ? text("creationPreparedHelp", "No chat creation was dispatched. Review your setup and send when ready; your attempted text is retained.") : ["acknowledged", "bound", "settled"].includes(creationRecovery.active.state) ? text("creationBoundHelp", "This chat was already created. Review it before sending retained text again; starting another chat may create a duplicate.") : text("creationHelp", "The server may have created this chat. Review conversations before starting again; another chat may create a duplicate. Your original attempted text is retained.")}</p>
        {creationRecovery.active.state !== "settled" && <details className="mt-2">
          <summary>{text("creationOriginalText", "Original attempted text")}</summary>
          <p className="max-h-40 overflow-auto whitespace-pre-wrap break-words">{creationRecovery.active.inputText || text("creationEmptyText", "No user message was submitted with this creation.")}</p>
          {creationRecovery.active.inputText && onCopyCreationText && <Button size="sm" disabled={busy} onClick={async () => {
            const token = ++generation.current
            setBusy(true)
            try { await onCopyCreationText() }
            catch { if (token === generation.current) setError(text("creationCopyFailed", "Could not copy the text. You can select and copy it above.")) }
            finally { if (token === generation.current) setBusy(false) }
          }}>{text("creationCopy", "Copy attempted text")}</Button>}
        </details>}
        <div className="mt-2 flex flex-wrap gap-2">
          {onFindConversations && <Button size="sm" onClick={onFindConversations}>{text("findConversations", "Find conversations")}</Button>}
          {creationRecovery.active.chatId && onReviewCreatedChat && <Button size="sm" disabled={busy} onClick={() => void review()}>{(creationRecovery.copyInputRequired || creationRecovery.active.state === "settled") ? text("copyAndReviewCreated", "Copy draft and review saved chat") : text("reviewCreated", "Review created chat")}</Button>}
        </div>
        {error && <p role="alert" className="mt-2 text-danger">{error}</p>}
      </div>}
      {!disabled && creationRecovery?.unavailable && <p role="status" className="mb-2 text-sm">{text("creationStorageUnavailable", "Creation recovery could not be read. Keep your draft and check retained drafts before starting again.")}</p>}
      <div className="flex justify-end">
      <Button
        variant="text"
        size="sm"
        disabled={disabled || !onRestore}
        onClick={() => {
          setOpen(true)
          void refresh()
        }}>
        {text("open", "Recover drafts")}
      </Button>
      </div>
      <Modal
        open={open && !disabled}
        title={text("title", "Retained drafts")}
        footer={null}
        onCancel={close}
        destroyOnHidden>
        <p className="mb-3 text-sm text-text-muted">
          {text(
            "help",
            "Opening a draft starts a fresh conversation in its original mode, using your current chat setup. Your current unsent text stays available. Review retained text before sending it again. Attachments and conversation messages are not included."
          )}
        </p>
        {busy && <p role="status">{text("loading", "Loading drafts…")}</p>}
        {error && (
          <p role="alert" className="mb-2 text-danger">
            {error}
          </p>
        )}
        {listing?.unavailable && (
          <div className="mb-3" role="status">
            <p>
              {text(
                "partial",
                "Some saved drafts could not be read. Retry to check again."
              )}
            </p>
            <Button size="sm" disabled={busy} onClick={() => void refresh()}>
              {text("retry", "Retry")}
            </Button>
          </div>
        )}
        {listing && !listing.unavailable && !listing.drafts.length && !busy && (
          <p>{text("empty", "No other retained drafts on this browser.")}</p>
        )}
        <ul className="max-h-[50vh] space-y-2 overflow-y-auto">
          {!disabled &&
            listing?.drafts.map((draft) => (
              <li key={draft.key}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void recover(draft)}
                  className="w-full rounded border border-border p-3 text-left text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-50">
                  <span className="block text-sm font-medium">
                    {draft.creationAttempt?.state === "settled"
                      ? text("savedChatDraft", "Retained message for a saved chat")
                      : draft.handle.address.temporary
                      ? text("temporary", "Temporary draft")
                      : text("saved", "New chat draft")}
                  </span>
                  {draft.creationAttempt && <span className="block text-sm text-text-muted">{["acknowledged", "bound", "settled"].includes(draft.creationAttempt.state) ? text("creationBound", "Chat already created") : draft.creationAttempt.state === "prepared" ? text("creationPrepared", "Chat creation not started") : text("creationUnconfirmed", "Server save unconfirmed")}. {text("creationDraftHelp", "Open this draft to review its creation before sending. Starting another chat may create a duplicate.")}</span>}
                  <span className="block break-words text-sm">
                    {draft.content.slice(0, 160)}
                    {draft.content.length > 160 ? "…" : ""}
                  </span>
                  <time
                    className="block text-xs text-text-muted"
                    dateTime={new Date(draft.updatedAt).toISOString()}>
                    {new Date(draft.updatedAt).toLocaleString()}
                  </time>
                </button>
              </li>
            ))}
        </ul>
      </Modal>
    </div>
  )
}
