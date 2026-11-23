import { afterEach, describe, expect, it, vi } from "vitest"
import { tldwRequest, type TldwRequestPayload } from "@/services/tldw/request-core"

const path = "https://api.example.com/api/v1/llm/models/metadata"
const getConfig = async () => ({ serverUrl: "https://api.example.com" })
type ResponseType = TldwRequestPayload["responseType"]
const readModes: {
  name: string
  contentType: string
  responseType?: ResponseType
}[] = [
  { name: "inferred JSON", contentType: "application/json" },
  { name: "inferred text", contentType: "text/plain" },
  { name: "explicit JSON", contentType: "text/plain", responseType: "json" },
  { name: "explicit text", contentType: "application/json", responseType: "text" },
  {
    name: "binary",
    contentType: "application/octet-stream",
    responseType: "arrayBuffer"
  }
]

describe("tldwRequest response body transfer", () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  describe.each([200, 429])("HTTP %i", (status) => {
    it.each(readModes)(
      "fails interrupted $name reads without replaying",
      async ({ contentType, responseType }) => {
        const fetchFn = vi.fn(
          async () =>
            new Response(
              new ReadableStream({
                start(controller) {
                  controller.enqueue(new TextEncoder().encode('{"partial":'))
                },
                pull(controller) {
                  controller.error(new TypeError("Body transfer interrupted"))
                }
              }),
              { status, headers: { "content-type": contentType, "retry-after": "5" } }
            )
        )

        const result = await tldwRequest({ path, responseType }, { getConfig, fetchFn })

        expect(result).toMatchObject({
          ok: false,
          status: 0,
          error: "Body transfer interrupted"
        })
        expect(fetchFn).toHaveBeenCalledTimes(1)
      }
    )
  })

  it("does not mistake a stream SyntaxError for completed invalid JSON", async () => {
    const fetchFn = vi.fn(
      async () =>
        new Response(
          new ReadableStream({
            pull(controller) {
              controller.error(new SyntaxError("Stream decoding failed"))
            }
          }),
          { headers: { "content-type": "application/json" } }
        )
    )

    await expect(tldwRequest({ path }, { getConfig, fetchFn })).resolves.toMatchObject({
      ok: false,
      status: 0,
      error: "Stream decoding failed"
    })
  })

  it.each(readModes)(
    "preserves caller cancellation during $name reads",
    async ({ contentType, responseType }) => {
      const abort = new AbortController()
      let notifyReading!: () => void
      const reading = new Promise<void>((resolve) => {
        notifyReading = resolve
      })
      const fetchFn = vi.fn(
        async (_url: RequestInfo | URL, init?: RequestInit) =>
          new Response(
            new ReadableStream(
              {
                start(controller) {
                  init?.signal?.addEventListener("abort", () =>
                    controller.error(init.signal?.reason)
                  )
                },
                pull() {
                  notifyReading()
                }
              },
              { highWaterMark: 0 }
            ),
            { headers: { "content-type": contentType } }
          )
      )

      const pending = tldwRequest(
        { path, responseType, abortSignal: abort.signal },
        { getConfig, fetchFn }
      )
      await reading
      abort.abort()

      await expect(pending).resolves.toMatchObject({
        ok: false,
        status: 0,
        error: expect.stringMatching(/abort/i)
      })
      expect(fetchFn).toHaveBeenCalledTimes(1)
    }
  )

  it.each([200, 503])("fails a timed-out HTTP %i body after auth refresh", async (status) => {
    vi.useFakeTimers()
    let notifyReading!: () => void
    const reading = new Promise<void>((resolve) => {
      notifyReading = resolve
    })
    let calls = 0
    const fetchFn = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      if (++calls === 1) return new Response(null, { status: 401 })
      return new Response(
        new ReadableStream(
          {
            start(controller) {
              init?.signal?.addEventListener("abort", () => controller.error(init.signal?.reason))
            },
            pull() {
              notifyReading()
            }
          },
          { highWaterMark: 0 }
        ),
        { status, headers: { "content-type": "application/json" } }
      )
    })
    const pending = tldwRequest(
      { path, timeoutMs: 100 },
      {
        getConfig: async () => ({
          serverUrl: "https://api.example.com",
          authMode: "multi-user",
          accessToken: "access-token",
          refreshToken: "refresh-token"
        }),
        refreshAuth: async () => {},
        fetchFn
      }
    )
    await reading
    await vi.advanceTimersByTimeAsync(101)

    await expect(pending).resolves.toMatchObject({
      ok: false,
      status: 0,
      error: "Request timed out.",
      code: "REQUEST_TIMEOUT"
    })
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe("tldwRequest completed bodies", () => {
  it.each([
    {
      name: "204 JSON",
      status: 204,
      body: null,
      contentType: "application/json",
      responseType: undefined,
      expected: null
    },
    {
      name: "205 JSON",
      status: 205,
      body: null,
      contentType: "application/json",
      responseType: "json",
      expected: null
    },
    {
      name: "empty JSON",
      status: 200,
      body: "",
      contentType: "application/json",
      responseType: undefined,
      expected: null
    },
    {
      name: "empty explicit JSON",
      status: 200,
      body: "",
      contentType: "text/plain",
      responseType: "json",
      expected: null
    },
    {
      name: "whitespace JSON",
      status: 200,
      body: " \n",
      contentType: "application/json",
      responseType: undefined,
      expected: null
    },
    {
      name: "invalid JSON",
      status: 200,
      body: "not json",
      contentType: "application/json",
      responseType: undefined,
      expected: null
    },
    {
      name: "JSON null",
      status: 200,
      body: "null",
      contentType: "application/json",
      responseType: undefined,
      expected: null
    },
    {
      name: "empty text",
      status: 200,
      body: "",
      contentType: "text/plain",
      responseType: undefined,
      expected: ""
    },
    {
      name: "empty explicit text",
      status: 204,
      body: null,
      contentType: "application/json",
      responseType: "text",
      expected: ""
    },
    {
      name: "JSON data",
      status: 200,
      body: '{"models":[]}',
      contentType: "application/json",
      responseType: undefined,
      expected: { models: [] }
    },
    {
      name: "explicit JSON data",
      status: 200,
      body: '{"models":[]}',
      contentType: "text/plain",
      responseType: "json",
      expected: { models: [] }
    },
    {
      name: "explicit text data",
      status: 200,
      body: "null",
      contentType: "application/json",
      responseType: "text",
      expected: "null"
    }
  ])("preserves $name", async ({ status, body, contentType, responseType, expected }) => {
    const fetchFn = vi.fn(
      async () =>
        new Response(body, {
          status,
          headers: { "content-type": contentType }
        })
    )

    await expect(
      tldwRequest({ path, responseType: responseType as ResponseType }, { getConfig, fetchFn })
    ).resolves.toMatchObject({ ok: true, status, data: expected })
  })

  it("preserves an empty HEAD response with a nonzero representation length", async () => {
    const fetchFn = vi.fn(
      async () =>
        new Response(null, {
          headers: { "content-type": "application/json", "content-length": "256" }
        })
    )

    await expect(
      tldwRequest({ path, method: "HEAD" }, { getConfig, fetchFn })
    ).resolves.toMatchObject({ ok: true, status: 200, data: null })
  })

  it.each(["", "binary data"])("preserves completed binary data %j", async (body) => {
    const fetchFn = vi.fn(async () => new Response(body))
    const result = await tldwRequest({ path, responseType: "arrayBuffer" }, { getConfig, fetchFn })

    expect(result).toMatchObject({ ok: true, status: 200 })
    expect(new TextDecoder().decode(result.data as ArrayBuffer)).toBe(body)
  })

  it.each([
    {
      name: "JSON detail",
      body: '{"detail":"Slow down"}',
      contentType: "application/json",
      responseType: undefined,
      data: { detail: "Slow down" },
      error: "Slow down"
    },
    {
      name: "binary error fallback",
      body: '{"detail":"Slow down"}',
      contentType: "application/json",
      responseType: "arrayBuffer",
      data: { detail: "Slow down" },
      error: "Slow down"
    },
    {
      name: "empty JSON",
      body: "",
      contentType: "application/json",
      responseType: undefined,
      data: null,
      error: "Too Many Requests"
    },
    {
      name: "invalid JSON",
      body: "not json",
      contentType: "application/json",
      responseType: "json",
      data: null,
      error: "Too Many Requests"
    },
    {
      name: "empty text",
      body: "",
      contentType: "text/plain",
      responseType: undefined,
      data: "",
      error: "Too Many Requests"
    }
  ])(
    "preserves HTTP error status and retry metadata for $name",
    async ({ body, contentType, responseType, data, error }) => {
      const fetchFn = vi.fn(
        async () =>
          new Response(body, {
            status: 429,
            statusText: "Too Many Requests",
            headers: {
              "content-type": contentType,
              "retry-after": "5",
              "x-request-id": "body-test"
            }
          })
      )

      await expect(
        tldwRequest({ path, responseType: responseType as ResponseType }, { getConfig, fetchFn })
      ).resolves.toMatchObject({
        ok: false,
        status: 429,
        data,
        error,
        retryAfterMs: 5000,
        headers: { "x-request-id": "body-test" }
      })
    }
  )
})
