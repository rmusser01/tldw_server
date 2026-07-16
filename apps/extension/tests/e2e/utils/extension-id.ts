import type { BrowserContext } from "@playwright/test"
import { createHash } from "node:crypto"
import fs from "node:fs"
import path from "node:path"

type ResolveExtensionIdOptions = {
  userDataDir?: string
  extensionPath?: string
}

function resolveExtensionIdFromUserDataDir(
  userDataDir?: string
): string | null {
  if (!userDataDir) {
    return null
  }

  const extensionsRoot = path.join(userDataDir, "Default", "Extensions")
  if (!fs.existsSync(extensionsRoot)) {
    return null
  }

  try {
    const candidates = fs
      .readdirSync(extensionsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && /^[a-p]{32}$/.test(entry.name))
      .map((entry) => entry.name)

    return candidates[0] || null
  } catch {
    return null
  }
}

function resolveExtensionIdFromManifestKey(
  extensionPath?: string
): string | null {
  if (!extensionPath) {
    return null
  }

  try {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(extensionPath, "manifest.json"), "utf8")
    ) as { key?: unknown }
    const manifestKey =
      typeof manifest.key === "string" ? manifest.key.trim() : ""
    if (!manifestKey) {
      return null
    }

    const keyBytes = Buffer.from(manifestKey, "base64")
    if (!keyBytes.length) {
      return null
    }

    const hash = createHash("sha256").update(keyBytes).digest()
    return Array.from(hash.subarray(0, 16))
      .flatMap((byte) => [byte >> 4, byte & 15])
      .map((nibble) => String.fromCharCode(97 + nibble))
      .join("")
  } catch {
    return null
  }
}

export async function resolveExtensionId(
  context: BrowserContext,
  options: ResolveExtensionIdOptions = {}
): Promise<string> {
  let targetUrl =
    context.backgroundPages()[0]?.url() ||
    context.serviceWorkers()[0]?.url() ||
    ""

  if (!targetUrl) {
    let createdProbePage:
      | Awaited<ReturnType<BrowserContext["newPage"]>>
      | undefined
    let session:
      | Awaited<ReturnType<BrowserContext["newCDPSession"]>>
      | undefined
    try {
      const existingPage = context.backgroundPages()[0] || context.pages()[0]
      const page = existingPage || (await context.newPage())
      createdProbePage = existingPage ? undefined : page
      session = await context.newCDPSession(page)
      const { targetInfos } = await session.send("Target.getTargets")
      const extTarget =
        targetInfos.find(
          (t: any) =>
            typeof t.url === "string" &&
            t.url.startsWith("chrome-extension://") &&
            (t.type === "background_page" || t.type === "service_worker")
        ) ||
        targetInfos.find(
          (t: any) =>
            typeof t.url === "string" && t.url.startsWith("chrome-extension://")
        )

      if (extTarget?.url) {
        targetUrl = extTarget.url
      }
    } catch {
      // Best-effort only; fall through to error below if we still
      // cannot determine the extension id.
    } finally {
      try {
        await session?.detach()
      } catch {
        // Best-effort probe cleanup must not hide id fallbacks.
      }
      try {
        await createdProbePage?.close()
      } catch {
        // Best-effort probe cleanup must not hide id fallbacks.
      }
    }
  }

  const match = targetUrl.match(/chrome-extension:\/\/([a-p]{32})/)
  if (match) {
    return match[1]
  }

  const extensionIdFromProfile = resolveExtensionIdFromUserDataDir(
    options.userDataDir
  )
  if (extensionIdFromProfile) {
    return extensionIdFromProfile
  }

  const extensionIdFromManifestKey = resolveExtensionIdFromManifestKey(
    options.extensionPath
  )
  if (extensionIdFromManifestKey) {
    return extensionIdFromManifestKey
  }

  const activeTargets = context
    .backgroundPages()
    .concat(context.serviceWorkers())
    .map((target) => target.url())
    .filter(Boolean)

  const targetSummary = activeTargets.length
    ? activeTargets.join(", ")
    : "[no extension targets]"
  throw new Error(
    `Could not determine extension id from ${targetUrl || targetSummary}`
  )
}
