import { useCallback, useEffect, useRef, useState } from "react"
import type { SplashCard } from "../components/Common/SplashScreen/engine/types"
import { useSetting } from "@/hooks/useSetting"
import {
  SPLASH_DISABLED_SETTING,
  SPLASH_ENABLED_CARD_NAMES_SETTING,
  SPLASH_DURATION_SECONDS_MAX,
  SPLASH_DURATION_SECONDS_MIN,
  SPLASH_DURATION_SECONDS_SETTING
} from "@/services/settings/ui-settings"

interface SplashScreenState {
  /** Whether to show the splash overlay right now. */
  visible: boolean
  /** The selected card for this session. */
  card: SplashCard | null
  /** Random loading message. */
  message: string
  /** Dismiss the splash (click / keypress / timeout). */
  dismiss: () => void
  /** Trigger a fresh random splash (used after successful login action). */
  show: (options?: { force?: boolean }) => void
  /** Whether splashes are disabled in user prefs. */
  disabled: boolean
  /** Toggle splash preference. */
  setDisabled: (v: boolean) => void
}

/**
 * Manages splash screen state.
 * Splash is event-driven and shown only when `show()` is called.
 */
export function useSplashScreen(): SplashScreenState {
  const [visible, setVisible] = useState(false)
  const [card, setCard] = useState<SplashCard | null>(null)
  const [message, setMessage] = useState("")
  const [disabled, setDisabledPreference] = useSetting(SPLASH_DISABLED_SETTING)
  const [enabledCardNames] = useSetting(SPLASH_ENABLED_CARD_NAMES_SETTING)
  const [durationSeconds] = useSetting(SPLASH_DURATION_SECONDS_SETTING)
  const request = useRef(0)
  const preferences = useRef({ disabled, enabledCardNames, durationSeconds })
  useEffect(() => {
    const previous = preferences.current
    if (
      previous.disabled !== disabled ||
      previous.durationSeconds !== durationSeconds ||
      previous.enabledCardNames.length !== enabledCardNames.length ||
      previous.enabledCardNames.some(
        (name, index) => name !== enabledCardNames[index]
      )
    ) {
      request.current++
      setVisible(false)
    }
    preferences.current = { disabled, enabledCardNames, durationSeconds }
  }, [disabled, enabledCardNames, durationSeconds])
  useEffect(
    () => () => {
      request.current++
    },
    []
  )

  const resolveDurationSeconds = useCallback(() => {
    const fallback = durationSeconds
    if (typeof window === "undefined") return fallback
    const localStorageKey = SPLASH_DURATION_SECONDS_SETTING.localStorageKey
    if (!localStorageKey) return fallback
    try {
      const raw = window.localStorage.getItem(localStorageKey)
      if (raw === null || raw === "") return fallback
      const parsed = Math.round(Number(raw))
      if (!Number.isFinite(parsed)) return fallback
      return Math.min(
        SPLASH_DURATION_SECONDS_MAX,
        Math.max(SPLASH_DURATION_SECONDS_MIN, parsed)
      )
    } catch {
      return fallback
    }
  }, [durationSeconds])

  const dismiss = useCallback(() => {
    request.current++
    setVisible(false)
  }, [])

  const show = useCallback(
    (options?: { force?: boolean }) => {
      const original = {
        ...preferences.current,
        enabledCardNames: [...preferences.current.enabledCardNames]
      }
      const force = options?.force === true
      if (original.disabled && !force) return
      const generation = ++request.current
      const effectiveDurationSeconds = resolveDurationSeconds()
      setVisible(false)
      void Promise.all([
        import("../data/splash-cards"),
        import("../data/splash-messages")
      ])
        .then(([cards, messages]) => {
          if (request.current !== generation) return
          const selected = cards.randomSplashCard({
            enabledNames: original.enabledCardNames
          })
          setCard({
            ...selected,
            duration: Math.round(effectiveDurationSeconds * 1000)
          })
          setMessage(selected.subtitle || messages.randomSplashMessage())
          setVisible(true)
        })
        .catch((error) => console.warn("Unable to load splash content", error))
    },
    [resolveDurationSeconds]
  )

  const setDisabled = useCallback(
    (value: boolean) => {
      preferences.current = { ...preferences.current, disabled: value }
      if (value) dismiss()
      void setDisabledPreference(value)
    },
    [dismiss, setDisabledPreference]
  )

  return { visible, card, message, dismiss, show, disabled, setDisabled }
}
