import React from "react"
import { useTranslation } from "react-i18next"
import useDynamicTextareaSize from "~/hooks/useDynamicTextareaSize"
import { isFirefoxTarget } from "@/config/platform"
import { useSimpleForm } from "@/hooks/useSimpleForm"

type Props = {
  value: string
  onSumbit: (value: string, isSend: boolean) => void
  onClose: () => void
  isBot: boolean
}

export const EditMessageForm = (props: Props) => {
  const [isComposing, setIsComposing] = React.useState(false)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const { t } = useTranslation("common")

  const form = useSimpleForm({
    initialValues: {
      message: props.value
    }
  })
  useDynamicTextareaSize(textareaRef, form.values.message, 300)

  React.useEffect(() => {
    form.setFieldValue("message", props.value)
  }, [props.value])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      e.preventDefault()
      props.onClose()
    }
  }

  return (
    <form
      data-chat-message-editor
      onSubmit={form.onSubmit((data) => {
        if (isComposing) return
        props.onClose()
        props.onSumbit(data.message, true)
      })}
      className="flex flex-col gap-2">
      <textarea
        {...form.getInputProps("message")}
        onKeyDown={handleKeyDown}
        onCompositionStart={() => {
          if (!isFirefoxTarget) {
            setIsComposing(true)
          }
        }}
        onCompositionEnd={() => {
          if (!isFirefoxTarget) {
            setIsComposing(false)
          }
        }}
        required
        rows={1}
        style={{ minHeight: "60px" }}
        tabIndex={0}
        placeholder={t("editMessage.placeholder")}
        ref={textareaRef}
        className="w-full bg-transparent text-body text-text placeholder:text-text-muted focus-within:outline-none focus:ring-0 focus-visible:ring-0 ring-0 border-0"
      />
      <div className="flex flex-wrap gap-2 mt-2">
        <div
          className={`w-full flex ${
            !props.isBot ? "justify-between" : "justify-end"
          }`}>
          {!props.isBot && (
            <button
              type="button"
              onClick={() => {
                props.onSumbit(form.values.message, false)
                props.onClose()
              }}
              aria-label={t("save")}
              title={t("save")}
              className="rounded-lg border border-border bg-surface px-2 py-2 text-sm text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus hover:bg-surface2">
              {t("save")}
            </button>
          )}
          <div className="flex space-x-2">
            <button
              aria-label={props.isBot ? t("save") : t("saveAndSubmit")}
              title={props.isBot ? t("save") : t("saveAndSubmit")}
              className="rounded-lg bg-primary px-2 py-2 text-sm text-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus hover:bg-primaryStrong">
              {props.isBot ? t("save") : t("saveAndSubmit")}
            </button>

            <button
              type="button"
              onClick={props.onClose}
              aria-label={t("cancel")}
              title={t("cancel")}
              className="rounded-lg border border-border bg-surface px-2 py-2 text-sm text-text-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-focus hover:bg-surface2 hover:text-text">
              {t("cancel")}
            </button>
          </div>
        </div>
      </div>{" "}
    </form>
  )
}
