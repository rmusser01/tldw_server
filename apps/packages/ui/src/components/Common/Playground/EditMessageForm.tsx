import React from "react"
import { useTranslation } from "react-i18next"
import useDynamicTextareaSize from "~/hooks/useDynamicTextareaSize"
import { isFirefoxTarget } from "@/config/platform"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { Button } from "@/components/Common/Button"

type Props = {
  value: string
  onSumbit: (
    value: string,
    isSend: boolean,
    isCurrent?: () => boolean
  ) => unknown
  onClose: () => void
  isBot: boolean
}

export const EditMessageForm = (props: Props) => {
  const composing = React.useRef(false)
  const mounted = React.useRef(true)
  const attempt = React.useRef<object | null>(null)
  const source = React.useRef({ value: props.value, revision: 0 })
  const baseline = React.useRef(0)
  const expectedAcknowledgement = React.useRef<string | null>(null)
  const [pending, setPending] = React.useState(false)
  const [feedback, setFeedback] = React.useState<string | null>(null)
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
    }
  }, [])

  const changedMessage = t(
    "editMessage.changed",
    "This message changed while you were editing. Your edited text is kept. Copy it, then cancel and reopen the message to review the latest version."
  )
  const failedSave = t(
    "editMessage.failed",
    "Your edit could not be confirmed. Your edited text is kept. Try saving again."
  )
  const close = () => {
    attempt.current = null
    setPending(false)
    props.onClose()
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
      const isCurrent = () =>
        mounted.current &&
        attempt.current === currentAttempt &&
        form.getRevision() === revision &&
        (source.current.revision === originalSource.revision ||
          (source.current.revision === originalSource.revision + 1 &&
            source.current.value === value))
      const outcome = props.onSumbit(value, isSend, isCurrent)
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
        setFeedback(failedSave)
        return
      }
      baseline.current = source.current.revision
      expectedAcknowledgement.current =
        unchangedSource && originalSource.value !== value ? value : null
      if (form.getRevision() === revision) props.onClose()
    } catch {
      if (mounted.current && attempt.current === currentAttempt)
        setFeedback(failedSave)
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
