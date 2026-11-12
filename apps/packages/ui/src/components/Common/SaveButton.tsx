import { useEffect, useRef, useState } from "react"
import { CheckIcon, Loader2 } from "lucide-react"
import { useTranslation } from "react-i18next"
type Props = {
  onClick?: () => void
  disabled?: boolean
  loading?: boolean
  className?: string
  text?: string
  textOnSave?: string
  btnType?: "button" | "submit" | "reset"
  /** Duration in ms to show the "saved" state. Defaults to 2000ms */
  savedDuration?: number
}

export const SaveButton = ({
  onClick,
  disabled,
  loading,
  className,
  text = "save",
  textOnSave = "saved",
  btnType = "button",
  savedDuration = 2000
}: Props) => {
  const [clickedSave, setClickedSave] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { t } = useTranslation("common")

  const showSaved = clickedSave && !loading

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!clickedSave || loading) {
      return
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    timeoutRef.current = setTimeout(() => {
      setClickedSave(false)
      timeoutRef.current = null
    }, savedDuration)

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [clickedSave, loading, savedDuration])

  return (
    <button
      type={btnType}
      onClick={() => {
        if (loading) return
        setClickedSave(true)
        if (onClick) {
          onClick()
        }
      }}
      disabled={disabled || loading}
      className={`inline-flex mt-4 items-center rounded-md border border-transparent bg-primary px-3 py-2 min-h-[40px] text-sm font-medium leading-4 text-surface shadow-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50 ${className}`}>
      {loading ? (
        <Loader2 className="icon mr-2 animate-spin" />
      ) : showSaved ? (
        <CheckIcon className="icon mr-2" />
      ) : null}
      {loading ? t("saving", "Saving...") : showSaved ? t(textOnSave) : t(text)}
    </button>
  )
}
