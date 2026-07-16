import { describe, expect, it } from "vitest"

import { createSkillsRelayObserver } from "./skills-certification-relay"

type Listener = (...args: any[]) => void

class ContextDouble {
  private readonly listeners = new Map<string, Set<Listener>>()

  on(event: string, listener: Listener) {
    const listeners = this.listeners.get(event) || new Set<Listener>()
    listeners.add(listener)
    this.listeners.set(event, listeners)
    return this
  }

  off(event: string, listener: Listener) {
    this.listeners.get(event)?.delete(listener)
    return this
  }

  emit(event: string, value: any) {
    for (const listener of this.listeners.get(event) || []) listener(value)
  }
}

const workerUrl = "chrome-extension://strict-worker/background.js"

const makeRequest = ({
  url,
  method = "GET",
  worker = workerUrl,
  postData
}: {
  url: string
  method?: string
  worker?: string | null
  postData?: string
}) => ({
  url: () => url,
  method: () => method,
  serviceWorker: () => (worker === null ? null : { url: () => worker }),
  postData: () => postData || null
})

const respond = (
  context: ContextDouble,
  request: ReturnType<typeof makeRequest>,
  status: number
) => context.emit("response", { request: () => request, status: () => status })

const emitRequest = (
  context: ContextDouble,
  request: ReturnType<typeof makeRequest>
) => context.emit("request", request)

const successfulLifecycle = (context: ContextDouble, dryRun = true) => {
  const events = [
    ["POST", "/api/v1/skills", 201],
    [
      "POST",
      "/api/v1/skills/widget/execute",
      200,
      JSON.stringify({ dry_run: dryRun, arguments: { secret: "nope" } })
    ],
    ["DELETE", "/api/v1/skills/widget", 204],
    ["POST", "/api/v1/skills/widget/restore", 200],
    ["DELETE", "/api/v1/skills/widget", 204],
    ["DELETE", "/api/v1/skills/widget/purge", 204]
  ] as const
  for (const [method, suffix, status, postData] of events) {
    const request = makeRequest({
      url: `https://server.test${suffix}`,
      method,
      postData
    })
    emitRequest(context, request)
    respond(context, request, status)
  }
}

describe("createSkillsRelayObserver", () => {
  it("canonicalizes root and dynamic Skills routes", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)

    for (const suffix of [
      "/api/v1/skills/",
      "/api/v1/skills/widget",
      "/api/v1/skills/widget/execute",
      "/api/v1/skills/widget/restore",
      "/api/v1/skills/widget/purge",
      "/api/v1/skills/trash"
    ]) {
      emitRequest(context, makeRequest({ url: `https://server.test${suffix}` }))
    }

    expect(observer.entries.map((entry) => entry.path)).toEqual([
      "/api/v1/skills",
      "/api/v1/skills/:name",
      "/api/v1/skills/:name/execute",
      "/api/v1/skills/:name/restore",
      "/api/v1/skills/:name/purge",
      "/api/v1/skills/trash"
    ])
  })

  it("ignores non-Skills URLs", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)

    emitRequest(
      context,
      makeRequest({ url: "https://server.test/api/v1/media" })
    )

    expect(observer.entries).toEqual([])
  })

  it("records unsupported Skills paths with a sanitized sentinel and rejects them", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)

    for (const suffix of [
      "/api/v1/skills/widget/unknown",
      "/api/v1/skills/widget/execute/again",
      "/api/v1/skills/trash/widget",
      "/api/v1/skills//widget"
    ]) {
      const request = makeRequest({ url: `https://server.test${suffix}` })
      emitRequest(context, request)
      respond(context, request, 200)
    }

    expect(observer.entries.map((entry) => entry.path)).toEqual([
      "/api/v1/skills/:unexpected",
      "/api/v1/skills/:unexpected",
      "/api/v1/skills/:unexpected",
      "/api/v1/skills/:unexpected"
    ])
    expect(() => observer.assertValid()).toThrow(/unexpected Skills route/i)
  })

  it("uses exact worker URL strings for ownership", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)

    emitRequest(
      context,
      makeRequest({
        url: "https://server.test/api/v1/skills",
        worker: `${workerUrl}?stale`
      })
    )
    emitRequest(
      context,
      makeRequest({
        url: "https://server.test/api/v1/skills",
        worker: workerUrl
      })
    )

    expect(observer.entries.map((entry) => entry.worker_owned)).toEqual([
      false,
      true
    ])
  })

  it("rejects a page-owned request after a prior worker success", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    const workerRequest = makeRequest({
      url: "https://server.test/api/v1/skills"
    })
    const pageRequest = makeRequest({
      url: "https://server.test/api/v1/skills",
      worker: null
    })
    emitRequest(context, workerRequest)
    respond(context, workerRequest, 200)
    emitRequest(context, pageRequest)
    respond(context, pageRequest, 200)

    expect(() => observer.assertValid()).toThrow(/page-owned/i)
  })

  it("rejects failed and HTTP-error Skills requests", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    const failed = makeRequest({ url: "https://server.test/api/v1/skills" })
    const error = makeRequest({
      url: "https://server.test/api/v1/skills/trash"
    })
    emitRequest(context, failed)
    context.emit("requestfailed", failed)
    emitRequest(context, error)
    respond(context, error, 500)

    expect(() => observer.assertValid()).toThrow(/failed|HTTP/i)
  })

  it("retains redirects while excluding them from terminal mutation counts", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    const redirect = makeRequest({
      url: "https://server.test/api/v1/skills/widget",
      method: "DELETE"
    })
    emitRequest(context, redirect)
    respond(context, redirect, 302)

    expect(observer.entries.at(-1)).toMatchObject({
      outcome: "redirect",
      status: 302
    })
    expect(() => observer.assertValid()).not.toThrow()
  })

  it("permits arbitrary successful GET counts", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    for (let index = 0; index < 4; index += 1) {
      const request = makeRequest({
        url: `https://server.test/api/v1/skills?query=${index}`
      })
      emitRequest(context, request)
      respond(context, request, 200)
    }

    expect(() => observer.assertValid()).not.toThrow()
  })

  it("does not include successful GET execute routes in dry-run accounting", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    const getExecute = makeRequest({
      url: "https://server.test/api/v1/skills/widget/execute"
    })
    emitRequest(context, getExecute)
    respond(context, getExecute, 200)

    expect(() => observer.assertValid()).not.toThrow()
  })

  it("rejects pending tracked Skills requests, including GETs", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    emitRequest(
      context,
      makeRequest({ url: "https://server.test/api/v1/skills" })
    )

    expect(() => observer.assertValid()).toThrow(/pending/i)
  })

  it("requires the exact create, dry execute, trash, restore, and purge mutations", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)

    expect(() => observer.assertValid()).not.toThrow()
  })

  it("rejects missing required mutation status", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    const badCreate = observer.entries.find(
      (entry) => entry.method === "POST" && entry.path === "/api/v1/skills"
    )
    if (badCreate) badCreate.status = 200

    expect(() => observer.assertValid()).toThrow(
      /POST \/api\/v1\/skills status 201 x1/
    )
  })

  it("requires dry_run true for execute requests without retaining its body", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context, false)

    expect(() => observer.assertValid()).toThrow(/dry_run/i)
    expect(JSON.stringify(observer.entries)).not.toContain("dry_run")
  })

  it("rejects extra successful mutations", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    successfulLifecycle(context)
    const extra = makeRequest({
      url: "https://server.test/api/v1/skills/widget",
      method: "PATCH"
    })
    emitRequest(context, extra)
    respond(context, extra, 200)

    expect(() => observer.assertValid()).toThrow(/extra successful mutation/i)
  })

  it("exposes only the sanitized ledger key allowlist", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    const request = makeRequest({
      url: "https://server.test/api/v1/skills/widget?api_key=top-secret",
      method: "POST",
      postData:
        '{"dry_run":true,"arguments":{"token":"top-secret"},"content":"skill content"}'
    })
    emitRequest(context, request)
    respond(context, request, 200)

    expect(Object.keys(observer.entries[0]).sort()).toEqual([
      "method",
      "outcome",
      "path",
      "status",
      "worker_owned"
    ])
    const ledger = JSON.stringify(observer.entries)
    for (const forbidden of [
      "top-secret",
      "arguments",
      "skill content",
      "headers",
      "https://",
      "api_key",
      "body"
    ]) {
      expect(ledger).not.toContain(forbidden)
    }
  })

  it("removes its listeners idempotently", () => {
    const context = new ContextDouble()
    const observer = createSkillsRelayObserver(context as any, workerUrl)
    observer.dispose()
    observer.dispose()
    emitRequest(
      context,
      makeRequest({ url: "https://server.test/api/v1/skills" })
    )

    expect(observer.entries).toEqual([])
  })
})
