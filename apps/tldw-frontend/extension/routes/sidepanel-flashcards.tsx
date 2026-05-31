import { useCallback, useEffect, useRef } from "react"
import { Button } from "antd"
import { ExternalLink } from "lucide-react"
import { useTranslation } from "react-i18next"
import { browser } from "wxt/browser"

const SidepanelFlashcards = () => {
  const { t } = useTranslation(["sidepanel"])
  const autoOpenedRef = useRef(false)

  const openOptionsHashRoute = useCallback((route: string) => {
    const normalizedRoute = route.startsWith("/") ? route : `/${route}`
    const optionsPath =
      normalizedRoute === "/flashcards"
        ? "/options.html#/flashcards"
        : `/options.html#${normalizedRoute}`
    const url = browser.runtime.getURL(optionsPath)
    const openFallbackWindow = () => {
      window.open(url, "_blank", "noopener,noreferrer")
    }

    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch(openFallbackWindow)
      return
    }

    openFallbackWindow()
  }, [])

  useEffect(() => {
    if (autoOpenedRef.current) return
    autoOpenedRef.current = true
    openOptionsHashRoute("/flashcards")
  }, [openOptionsHashRoute])

  return (
    <main className="flex min-h-screen items-center justify-center bg-neutral-50 px-4 py-6 text-text dark:bg-surface">
      <section className="w-full max-w-sm rounded-lg border border-border bg-bg p-4 shadow-sm">
        <div className="space-y-3">
          <div>
            <h1 className="text-base font-semibold">
              {t("sidepanel:flashcards.opening", "Opening Flashcards")}
            </h1>
            <p className="mt-1 text-sm text-text-muted">
              {t(
                "sidepanel:flashcards.openedInTab",
                "Flashcards opens in the full extension workspace so review, import, and deck management have enough room."
              )}
            </p>
          </div>
          <Button
            type="primary"
            icon={<ExternalLink className="size-4" />}
            onClick={() => openOptionsHashRoute("/flashcards")}
          >
            {t("sidepanel:flashcards.openAgain", "Open Flashcards")}
          </Button>
        </div>
      </section>
    </main>
  )
}

export default SidepanelFlashcards
