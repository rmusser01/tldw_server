export type OnboardingEntryIntent = "character-chat"

export const CHARACTER_CHAT_ONBOARDING_INTENT: OnboardingEntryIntent =
  "character-chat"

type LocationLike = {
  pathname?: string
  search?: string
  hash?: string
}

const SAFE_URL_ORIGIN = "https://tldw.local"

export const getSafeOnboardingReturnTo = (
  value: string | null | undefined
): string | null => {
  const rawValue = value?.trim()
  if (!rawValue) return null
  if (!rawValue.startsWith("/") || rawValue.startsWith("//")) return null

  try {
    const parsed = new URL(rawValue, SAFE_URL_ORIGIN)
    if (parsed.origin !== SAFE_URL_ORIGIN) return null
    return `${parsed.pathname}${parsed.search}${parsed.hash}`
  } catch {
    return null
  }
}

export const getOnboardingReturnToFromSearch = (
  search: string | undefined
): string | null => {
  const params = new URLSearchParams(search || "")
  return getSafeOnboardingReturnTo(params.get("returnTo"))
}

export const resolveOnboardingEntryIntent = ({
  pathname = "",
  search = ""
}: LocationLike): OnboardingEntryIntent | null => {
  const params = new URLSearchParams(search)
  if (params.get("intent") === CHARACTER_CHAT_ONBOARDING_INTENT) {
    return CHARACTER_CHAT_ONBOARDING_INTENT
  }

  const returnTo = getSafeOnboardingReturnTo(params.get("returnTo"))
  if (returnTo?.startsWith("/characters")) {
    return CHARACTER_CHAT_ONBOARDING_INTENT
  }

  if (pathname === "/characters" || pathname.startsWith("/characters/")) {
    return CHARACTER_CHAT_ONBOARDING_INTENT
  }

  return null
}

export const buildFirstRunOnboardingRoute = ({
  pathname = "",
  search = "",
  hash = ""
}: LocationLike): string => {
  if (
    resolveOnboardingEntryIntent({
      pathname,
      search
    }) !== CHARACTER_CHAT_ONBOARDING_INTENT
  ) {
    return "/"
  }

  const returnTo = getSafeOnboardingReturnTo(`${pathname}${search}${hash}`)
  const params = new URLSearchParams({
    intent: CHARACTER_CHAT_ONBOARDING_INTENT
  })
  if (returnTo) {
    params.set("returnTo", returnTo)
  }
  return `/?${params.toString()}`
}

export const buildCharacterOnboardingRoute = ({
  returnTo,
  action
}: {
  returnTo?: string | null
  action: "create" | "import"
}): string => {
  const safeReturnTo = getSafeOnboardingReturnTo(returnTo)
  const fallback =
    action === "create"
      ? "/characters?from=onboarding&create=true"
      : "/characters?from=onboarding&import=true"
  const baseRoute = safeReturnTo?.startsWith("/characters")
    ? safeReturnTo
    : fallback

  try {
    const parsed = new URL(baseRoute, SAFE_URL_ORIGIN)
    if (!parsed.searchParams.has("from")) {
      parsed.searchParams.set("from", "onboarding")
    }
    parsed.searchParams.delete(action === "create" ? "import" : "create")
    parsed.searchParams.set(action, "true")
    return `${parsed.pathname}${parsed.search}${parsed.hash}`
  } catch {
    return fallback
  }
}
