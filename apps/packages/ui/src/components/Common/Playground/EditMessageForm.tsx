import React from "react"
import { useTranslation } from "react-i18next"
import useDynamicTextareaSize from "~/hooks/useDynamicTextareaSize"
import { isFirefoxTarget } from "@/config/platform"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { Button } from "@/components/Common/Button"
import { getMessageEditFeedback } from "@/utils/message-edit-feedback"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"

type Props = {
  value: string
  onSumbit: (
    value: string,
    isSend: boolean,
    isCurrent?: () => boolean,
    isOriginCurrent?: () => boolean
  ) => unknown
  onClose: () => void
  isBot: boolean
}

export const EditMessageForm = (props: Props) => {
  const selection = useHistorySelectionContext()
  const composing = React.useRef(false)
  const mounted = React.useRef(true)
  const attempt = React.useRef<object | null>(null)
  const source = React.useRef({ value: props.value, revision: 0 })
  const baseline = React.useRef(0)
  const expectedAcknowledgement = React.useRef<string | null>(null)
  const [pending, setPending] = React.useState(false)
  const [feedback, setFeedback] = React.useState<string | null>(null)
  const copyAttempt = React.useRef<{ isCurrent: () => boolean } | null>(null)
  const [copying, setCopying] = React.useState(false)
  const [copyFeedback, setCopyFeedback] = React.useState<{
    text: string
    inputRevision: number
    sourceRevision: number
    isSourceCurrent: () => boolean
  } | null>(null)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const feedbackId = React.useId()
  const { t } = useTranslation("common")
  const form = useSimpleForm({ initialValues: { message: props.value } })
  const attachTextarea = useDynamicTextareaSize(
    textareaRef,
    form.values.message,
    300
  )

  if (source.current.value !== props.value) {
    source.current = {
      value: props.value,
      revision: source.current.revision + 1
    }
    // A confirmed write can reach the parent after the user has continued typing.
    if (expectedAcknowledgement.current === props.value) {
      baseline.current = source.current.revision
    }
    expectedAcknowledgement.current = null
  }

  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      attempt.current = null
      copyAttempt.current = null
    }
  }, [])

  React.useEffect(() => {
    if (copyAttempt.current && !copyAttempt.current.isCurrent()) {
      copyAttempt.current = null
      setCopying(false)
    }
    if (
      copyFeedback &&
      (copyFeedback.inputRevision !== form.getRevision() ||
        copyFeedback.sourceRevision !== source.current.revision ||
        !copyFeedback.isSourceCurrent())
    )
      setCopyFeedback(null)
  })

  const issueMessage = (issue: unknown) => {
    const copy = getMessageEditFeedback(issue)
    return t(copy.key, copy.fallback)
  }
  const changedMessage = issueMessage({ reason: "message_changed" })
  const close = () => {
    attempt.current = null
    copyAttempt.current = null
    setPending(false)
    setCopying(false)
    setCopyFeedback(null)
    props.onClose()
  }

  const copySource = captureChatDraftSource(selection?.fence())
  const copyEdit = async () => {
    if (copyAttempt.current || !copySource.isCurrent()) return
    const inputRevision = form.getRevision(),
      sourceRevision = source.current.revision
    const currentCopy: { isCurrent: () => boolean } = {
      isCurrent: () =>
        mounted.current &&
        copyAttempt.current === currentCopy &&
        form.getRevision() === inputRevision &&
        source.current.revision === sourceRevision &&
        copySource.isCurrent()
    }
    const isCurrent = currentCopy.isCurrent
    copyAttempt.current = currentCopy
    setCopying(true)
    setCopyFeedback(null)
    try {
      // Authored edits copy verbatim; saved-reply preferences may strip markup.
      await navigator.clipboard.writeText(form.getValues().message)
      if (isCurrent())
        setCopyFeedback({
          inputRevision,
          sourceRevision,
          isSourceCurrent: copySource.isCurrent,
          text: t("editMessage.copied", "Edit copied")
        })
    } catch {
      if (isCurrent())
        setCopyFeedback({
          inputRevision,
          sourceRevision,
          isSourceCurrent: copySource.isCurrent,
          text: t(
            "editMessage.copyFailed",
            "Copy failed. Select your edited text and use your browser's Copy command."
          )
        })
    } finally {
      if (mounted.current && copyAttempt.current === currentCopy) {
        copyAttempt.current = null
        setCopying(false)
      }
    }
  }

  const save = async (isSend: boolean) => {
    if (composing.current || attempt.current) return
    if (source.current.revision !== baseline.current) {
      setFeedback(changedMessage)
      return
    }
    const currentAttempt = {}
    const originalSource = source.current
    const revision = form.getRevision()
    const value = form.getValues().message
    attempt.current = currentAttempt
    setPending(true)
    setFeedback(null)
    try {
      const isOriginCurrent = () =>
        mounted.current &&
        attempt.current === currentAttempt &&
        (source.current.revision === originalSource.revision ||
          (source.current.revision === originalSource.revision + 1 &&
            source.current.value === value))
      const isCurrent = () =>
        isOriginCurrent() && form.getRevision() === revision
      const outcome = props.onSumbit(value, isSend, isCurrent, isOriginCurrent)
      const asynchronous =
        outcome !== null &&
        (typeof outcome === "object" || typeof outcome === "function") &&
        "then" in outcome &&
        typeof outcome.then === "function"
      const result = await outcome
      if (!mounted.current || attempt.current !== currentAttempt) return
      const unchangedSource =
        source.current.revision === originalSource.revision
      const acknowledgedSource =
        source.current.revision === originalSource.revision + 1 &&
        source.current.value === value
      if (!unchangedSource && !acknowledgedSource) {
        setFeedback(changedMessage)
        return
      }
      if (
        (result === undefined && asynchronous) ||
        (result !== undefined &&
          result !== true &&
          !(
            result &&
            typeof result === "object" &&
            "status" in result &&
            result.status === "submitted"
          ))
      ) {
        setFeedback(issueMessage(result))
        return
      }
      baseline.current = source.current.revision
      expectedAcknowledgement.current =
        unchangedSource && originalSource.value !== value ? value : null
      if (form.getRevision() === revision) close()
    } catch (error) {
      if (mounted.current && attempt.current === currentAttempt)
        setFeedback(issueMessage(error))
    } finally {
      if (mounted.current && attempt.current === currentAttempt) {
        attempt.current = null
        setPending(false)
      }
    }
  }

  const alert =
    !pending && source.current.revision !== baseline.current
      ? changedMessage
      : feedback

  return (
    <form
      data-chat-message-editor
      aria-busy={pending}
      onSubmit={form.onSubmit(() => save(!props.isBot))}
      className="flex flex-col gap-2">
      <textarea
        {...form.getInputProps("message")}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault()
            close()
          }
        }}
        onCompositionStart={() => {
          if (!isFirefoxTarget) composing.current = true
        }}
        onCompositionEnd={() => {
          if (!isFirefoxTarget) composing.current = false
        }}
        aria-label={t("editMessage.label", "Edit message")}
        aria-describedby={alert ? feedbackId : undefined}
        required
        autoFocus
        rows={1}
        style={{ minHeight: "60px" }}
        tabIndex={0}
        placeholder={t("editMessage.placeholder")}
        ref={attachTextarea}
        className="w-full bg-transparent text-body text-text placeholder:text-text-muted focus-within:outline-none focus:ring-0 focus-visible:ring-0 ring-0 border-0"
      />
      {pending && (
        <p role="status" className="text-sm text-text-muted">
          {t("saving")}
        </p>
      )}
      {alert && (
        <p id={feedbackId} role="alert" className="text-sm text-text">
          {alert}
        </p>
      )}
      {!pending && copying && (
        <p role="status" className="text-sm text-text-muted">
          {t("editMessage.copying", "Copying edit...")}
        </p>
      )}
      {!pending &&
        copyFeedback?.inputRevision === form.getRevision() &&
        copyFeedback?.sourceRevision === source.current.revision &&
        copyFeedback.isSourceCurrent() && (
          <p role="status" className="text-sm text-text-muted">
            {copyFeedback.text}
          </p>
        )}
      <div
        className={`mt-2 flex w-full flex-wrap gap-2 ${props.isBot ? "justify-end" : "justify-between"}`}>
        {!props.isBot && (
          <Button
            variant="outline"
            disabled={pending}
            onClick={() => {
              void save(false)
            }}
            ariaLabel={t("save")}>
            {t("save")}
          </Button>
        )}
        {alert && !pending && (
          <Button
            variant="outline"
            disabled={copying}
            onClick={() => {
              void copyEdit()
            }}
            ariaLabel={t("editMessage.copy", "Copy edit")}>
            {t("editMessage.copy", "Copy edit")}
          </Button>
        )}
        <div className="flex gap-2">
          <Button
            variant="primary"
            htmlType="submit"
            disabled={pending}
            ariaLabel={props.isBot ? t("save") : t("saveAndSubmit")}>
            {props.isBot ? t("save") : t("saveAndSubmit")}
          </Button>
          <Button variant="outline" onClick={close} ariaLabel={t("cancel")}>
            {t("cancel")}
          </Button>
        </div>
      </div>
    </form>
  )
}
