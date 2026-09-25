const QUICKSTART_MODE = "quickstart"
const MANAGED_MODE = "managed"

function isAbsoluteUrl(value) {
  try {
    const parsed = new URL(value)
    return /^https?:$/i.test(parsed.protocol) && parsed.origin.length > 0
  } catch {
    return false
  }
}

function canonicalBareHttpOrigin(value) {
  try {
    const parsed = new URL(value)
    const origin = parsed.origin
    if (
      /^https?:$/i.test(parsed.protocol) &&
      origin !== "null" &&
      !parsed.username &&
      !parsed.password &&
      parsed.pathname === "/" &&
      !parsed.search &&
      !parsed.hash &&
      (value === origin || value === `${origin}/`)
    ) {
      return origin
    }
    return null
  } catch {
    return null
  }
}

export function validateNetworkingConfig(env = process.env) {
  const deploymentMode = String(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || "").trim() || "advanced"
  const internalApiOrigin = String(env.TLDW_INTERNAL_API_ORIGIN || "")
  const canonicalInternalApiOrigin = canonicalBareHttpOrigin(internalApiOrigin)
  const publicApiUrl = String(env.NEXT_PUBLIC_API_URL || "").trim()
  const sameOriginMode = deploymentMode === QUICKSTART_MODE || deploymentMode === MANAGED_MODE

  if (deploymentMode === QUICKSTART_MODE && !canonicalInternalApiOrigin) {
    throw new Error(
      "Invalid WebUI networking config: quickstart mode requires TLDW_INTERNAL_API_ORIGIN to be an absolute HTTP(S) origin."
    )
  }

  if (
    sameOriginMode &&
    publicApiUrl.length > 0 &&
    isAbsoluteUrl(publicApiUrl)
  ) {
    throw new Error(
      "Invalid WebUI networking config: same-origin mode must not set NEXT_PUBLIC_API_URL to an absolute browser API URL."
    )
  }

  if (deploymentMode === MANAGED_MODE && internalApiOrigin && !canonicalInternalApiOrigin) {
    throw new Error(
      "Invalid WebUI networking config: managed mode requires TLDW_INTERNAL_API_ORIGIN to be a canonical HTTP(S) origin when supplied."
    )
  }

  if (!sameOriginMode && !isAbsoluteUrl(publicApiUrl)) {
    throw new Error(
      "Invalid WebUI networking config: advanced mode requires NEXT_PUBLIC_API_URL to be an absolute browser API URL."
    )
  }

  return {
    deploymentMode,
    internalApiOrigin: canonicalInternalApiOrigin || internalApiOrigin,
    publicApiUrl
  }
}

const isEntrypoint = process.argv[1] === new URL(import.meta.url).pathname

if (isEntrypoint) {
  validateNetworkingConfig(process.env)
}
