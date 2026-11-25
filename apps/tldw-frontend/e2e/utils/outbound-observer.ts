import { writeFile } from "node:fs/promises"

import { chromium, type BrowserContext, type Request } from "@playwright/test"

export const loopbackOrigin = (port: number): string => `http://127.0.0.1:${port}`

export function fixtureSetting(name: string): string {
  const value = process.env[name]
  if (!value) throw new Error(`Missing browser fixture setting ${name}`)
  return value
}

export function chromiumFixtureOptions({ trace = true } = {}) {
  const wrapper = process.env.OUTBOUND_CHROMIUM_WRAPPER
  if (!wrapper) return { args: [] }
  const binary = chromium.executablePath()
  return {
    executablePath: trace ? wrapper : binary,
    env: trace ? { ...process.env, OUTBOUND_CHROMIUM_BINARY: binary } : undefined,
    args: [
      "--no-proxy-server",
      "--host-resolver-rules=EXCLUDE 127.0.0.1, MAP * ~NOTFOUND",
      // These scenarios do not use physical audio devices or the desktop sound server.
      "--disable-audio-output",
      "--use-fake-device-for-media-stream"
    ]
  }
}

export async function observeOutbound(
  context: BrowserContext,
  fixtureOrigins: readonly string[],
  expectedWorkerPost?: string
) {
  const allowedOrigins = new Set(fixtureOrigins)
  let allowedRequests = 0
  let workerRequests = 0
  let unexpectedRequests = 0
  let tripwirePassed = false

  const isAllowed = (url: string) => allowedOrigins.has(new URL(url).origin)
  const onRequest = (request: Request) => {
    if (!/^https?:/i.test(request.url())) return
    if (!isAllowed(request.url())) {
      unexpectedRequests++
      return
    }
    allowedRequests++
    if (request.serviceWorker() && request.method() === "POST" && request.url() === expectedWorkerPost) {
      workerRequests++
    }
  }
  context.on("request", onRequest)
  await context.route(/^https?:\/\//, async (route) => {
    if (isAllowed(route.request().url())) await route.fallback()
    else await route.abort("blockedbyclient")
  })
  await context.routeWebSocket(/./, (socket) => {
    const url = new URL(socket.url())
    url.protocol = url.protocol === "wss:" ? "https:" : "http:"
    if (allowedOrigins.has(url.origin)) {
      socket.connectToServer()
    } else {
      unexpectedRequests++
      void socket.close()
    }
  })

  return {
    sawWorkerRequest: () => workerRequests > 0,
    unexpectedRequests: () => unexpectedRequests,
    noteTripwire: () => { tripwirePassed = true },
    async finish() {
      await context.close()
      await writeFile(
        fixtureSetting("OUTBOUND_BROWSER_OBSERVER_FILE"),
        JSON.stringify({ allowedRequests, workerRequests, unexpectedRequests, tripwirePassed }),
        { flag: "wx", mode: 0o600 }
      )
    }
  }
}
