const QUICKSTART_MODE = "quickstart"

function isAbsoluteUrl(value) {
  try {
    const parsed = new URL(value)
    return /^https?:$/i.test(parsed.protocol) && parsed.origin.length > 0
  } catch {
    return false
  }
}

function isBareHttpOrigin(value) {
  try {
    const parsed = new URL(value)
    return (
      /^https?:$/i.test(parsed.protocol) &&
      parsed.origin !== "null" &&
      !parsed.username &&
      !parsed.password &&
      parsed.pathname === "/" &&
      !parsed.search &&
      !parsed.hash
    )
  } catch {
    return false
  }
}

export function validateNetworkingConfig(env = process.env) {
  const deploymentMode =
    String(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || "").trim() || "advanced"
  const internalApiOrigin = String(env.TLDW_INTERNAL_API_ORIGIN || "").trim()
  const publicApiUrl = String(env.NEXT_PUBLIC_API_URL || "").trim()

  if (
    deploymentMode === QUICKSTART_MODE &&
    !isBareHttpOrigin(internalApiOrigin)
  ) {
    throw new Error(
      "Invalid WebUI networking config: quickstart mode requires TLDW_INTERNAL_API_ORIGIN to be an absolute HTTP(S) origin."
    )
  }

  if (
    deploymentMode === QUICKSTART_MODE &&
    publicApiUrl.length > 0 &&
    isAbsoluteUrl(publicApiUrl)
  ) {
    throw new Error(
      "Invalid WebUI networking config: quickstart mode must not set NEXT_PUBLIC_API_URL to an absolute browser API URL."
    )
  }

  if (
    deploymentMode !== QUICKSTART_MODE &&
    !isAbsoluteUrl(publicApiUrl)
  ) {
    throw new Error(
      "Invalid WebUI networking config: advanced mode requires NEXT_PUBLIC_API_URL to be an absolute browser API URL."
    )
  }

  return {
    deploymentMode,
    internalApiOrigin,
    publicApiUrl
  }
}

const isEntrypoint = process.argv[1] === new URL(import.meta.url).pathname

if (isEntrypoint) {
  validateNetworkingConfig(process.env)
}
