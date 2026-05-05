type TldwClientLike = {
  getConfig: () => Promise<
    | {
        authMode?: unknown
        accessToken?: unknown
        apiKey?: unknown
      }
    | null
    | undefined
  >
}

type TldwAuthLike = {
  getCurrentUser: () => Promise<unknown>
}

export const loadTldwClient = async (): Promise<TldwClientLike> => {
  const clientModule = await import("@/services/tldw/TldwApiClient")
  const candidate = clientModule.tldwClient
  if (!candidate || typeof candidate.getConfig !== "function") {
    throw new TypeError("Configured tldw client does not expose getConfig")
  }
  return candidate
}

export const loadTldwAuth = async (): Promise<TldwAuthLike> => {
  const authModule = await import("@/services/tldw/TldwAuth")
  return authModule.tldwAuth as TldwAuthLike
}
