import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import React from "react"

import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"

type NotesGraphAuthorityConfig = Pick<
  TldwConfig,
  "serverUrl" | "authMode" | "authSource" | "orgId" | "apiKey" | "accessToken"
>

type NotesGraphAuthorityResolution = {
  boundaryKey: string
  scope: string
}

const normalizeServerOrigin = (serverUrl: string): string | null => {
  try {
    return new URL(serverUrl.trim()).origin
  } catch {
    return null
  }
}

const buildBoundaryKey = (
  config: NotesGraphAuthorityConfig | null
): { key: string; origin: string; isCookieSession: boolean } | null => {
  if (!config) return null
  const origin = normalizeServerOrigin(config.serverUrl)
  if (!origin) return null
  const credential =
    config.authMode === "single-user" ? config.apiKey : config.accessToken
  const credentialBoundary = credential
    ? bytesToHex(
        sha256(
          utf8ToBytes(
            `tldw:notes-graph-config-boundary:v1\0${credential.trim()}`
          )
        )
      )
    : "none"
  return {
    key: JSON.stringify([
      origin,
      String(config.authMode || "unknown"),
      config.orgId ?? null,
      credentialBoundary
    ]),
    origin,
    isCookieSession: config.authSource === "cookie-session"
  }
}

export const createNotesGraphAuthorityScope = (
  serverOrigin: string,
  principalId: string | number
): string => {
  const tuple = JSON.stringify([
    normalizeServerOrigin(serverOrigin) ?? serverOrigin.trim(),
    String(principalId).trim()
  ])
  const digest = sha256(utf8ToBytes(`tldw:notes-graph-authority:v1\0${tuple}`))
  return `notes-graph:sha256:${bytesToHex(digest)}`
}

export const useNotesGraphAuthorityScope = ({
  config,
  loading
}: {
  config: NotesGraphAuthorityConfig | null
  loading: boolean
}): string | null => {
  const boundary = React.useMemo(() => buildBoundaryKey(config), [config])
  const [resolution, setResolution] =
    React.useState<NotesGraphAuthorityResolution | null>(null)
  const [revision, setRevision] = React.useState(0)
  const epochRef = React.useRef(0)

  React.useEffect(() => {
    const epoch = ++epochRef.current
    setResolution(null)
    if (loading || !boundary) return

    void tldwAuth
      .getCurrentUser()
      .then((user) => {
        if (epoch !== epochRef.current) return
        const principalId = String(user?.id ?? "").trim()
        if (!principalId || user?.is_active !== true) return
        setResolution({
          boundaryKey: boundary.key,
          scope: createNotesGraphAuthorityScope(boundary.origin, principalId)
        })
      })
      .catch(() => {
        if (epoch === epochRef.current) setResolution(null)
      })

    return () => {
      epochRef.current += 1
    }
  }, [boundary, loading, revision])

  React.useEffect(() => {
    const invalidate = () => {
      epochRef.current += 1
      setResolution(null)
      setRevision((current) => current + 1)
    }
    const revalidateCookieSession = () => {
      if (!boundary?.isCookieSession) return
      invalidate()
    }
    const revalidateVisibleCookieSession = () => {
      if (document.visibilityState === "visible") revalidateCookieSession()
    }
    window.addEventListener("tldw:config-updated", invalidate)
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    window.addEventListener("focus", revalidateCookieSession)
    window.addEventListener("pageshow", revalidateCookieSession)
    document.addEventListener(
      "visibilitychange",
      revalidateVisibleCookieSession
    )
    return () => {
      window.removeEventListener("tldw:config-updated", invalidate)
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
      window.removeEventListener("focus", revalidateCookieSession)
      window.removeEventListener("pageshow", revalidateCookieSession)
      document.removeEventListener(
        "visibilitychange",
        revalidateVisibleCookieSession
      )
      epochRef.current += 1
    }
  }, [boundary?.isCookieSession])

  if (loading || !boundary || resolution?.boundaryKey !== boundary.key) {
    return null
  }
  return resolution.scope
}
