import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"

export type SidepanelDirection = "ltr" | "rtl"

export interface UseSidepanelInitOptions {
  titleDefaultValue?: string
}

export function useSidepanelInit(options: UseSidepanelInitOptions = {}) {
  const { titleDefaultValue = "tldw" } = options
  const { t, i18n } = useTranslation()
  const [direction, setDirection] = useState<SidepanelDirection>("ltr")

  const language = i18n.language

  useEffect(() => {
    if (typeof document === "undefined") return
    if (!language) return

    document.documentElement.lang = language
    const dir = i18n.dir(language)
    const resolvedDirection: SidepanelDirection = dir === "rtl" ? "rtl" : "ltr"
    document.documentElement.dir = resolvedDirection
    setDirection(resolvedDirection)
  }, [language])

  useEffect(() => {
    if (typeof document === "undefined") return
    document.title = i18n.t("common:titles.sidepanel", {
      defaultValue: titleDefaultValue
    })
  }, [language, titleDefaultValue])

  return { direction, t, i18n }
}
