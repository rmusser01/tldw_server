import { browser } from "wxt/browser"
import { getBrowserRuntime, isExtensionRuntime } from "@/utils/browser-runtime"

export type CharactersDestinationMode =
  | "options-in-place"
  | "options-tab"
  | "web-route"

export type BuildCharactersRouteOptions = {
  from: string
  create?: boolean
  focus?: "expressions"
  characterId?: string | number | null
}

export const buildCharactersRoute = ({
  from,
  create = false,
  focus,
  characterId
}: BuildCharactersRouteOptions): string => {
  const params = new URLSearchParams({ from })
  if (create) {
    params.set("create", "true")
  }
  if (focus) {
    params.set("focus", focus)
  }
  if (characterId !== null && typeof characterId !== "undefined") {
    params.set("characterId", String(characterId))
  }
  return `/characters?${params.toString()}`
}

export const buildCharactersHash = (
  options: BuildCharactersRouteOptions
): string => `#${buildCharactersRoute(options)}`

export const resolveCharactersDestinationMode = ({
  pathname,
  extensionRuntime
}: {
  pathname?: string
  extensionRuntime: boolean
}): CharactersDestinationMode => {
  if ((pathname || "").includes("options.html")) {
    return "options-in-place"
  }
  return extensionRuntime ? "options-tab" : "web-route"
}

export const openCharactersWorkspace = async (
  options: BuildCharactersRouteOptions
): Promise<void> => {
  if (typeof window === "undefined") return

  const route = buildCharactersRoute(options)
  const hash = buildCharactersHash(options)
  const optionsPath = `/options.html${hash}`
  const runtime = getBrowserRuntime()
  const mode = resolveCharactersDestinationMode({
    pathname: window.location.pathname || "",
    extensionRuntime: isExtensionRuntime(runtime)
  })

  if (mode === "options-in-place") {
    const base = window.location.href.replace(/#.*$/, "")
    window.location.href = `${base}${hash}`
    return
  }

  if (mode === "options-tab") {
    const url = runtime?.getURL ? runtime.getURL(optionsPath) : optionsPath
    try {
      if (browser.tabs?.create) {
        await browser.tabs.create({ url })
        return
      }
    } catch (error) {
      console.debug("[characters-route] Failed to open characters tab:", error)
    }

    window.open(url, "_blank")
    return
  }

  try {
    window.open(route, "_blank")
  } catch (error) {
    console.debug("[characters-route] Failed to open characters route:", error)
  }
}
