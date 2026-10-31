import type { Page, Request, Response } from "@playwright/test"
import { describe, expect, it, vi } from "vitest"
import { captureAllApiCalls } from "../e2e/utils/api-assertions"

describe("captureAllApiCalls", () => {
  it.each([
    ["application/json; charset=utf-8", 200],
    ["application/problem+json", 422],
    ["Application/Vnd.Api+Json; charset=UTF-8", 400],
  ])("waits for in-flight %s response parsing before returning captured calls", async (contentType, status) => {
    let requestHandler: ((request: Request) => void) | null = null
    let resolveResponseBody: ((value: unknown) => void) | null = null

    const response = {
      status: () => status,
      headers: () => ({ "content-type": contentType }),
      json: () =>
        new Promise((resolve) => {
          resolveResponseBody = resolve
        }),
    } as unknown as Response
    const request = {
      url: () => "http://127.0.0.1:8000/api/v1/chat/completions",
      method: () => "POST",
      response: async () => response,
      postDataJSON: () => ({ model: "local-uat-chat" }),
    } as unknown as Request
    const page = {
      on: (_event: string, handler: (request: Request) => void) => {
        requestHandler = handler
      },
      removeListener: vi.fn(),
    } as unknown as Page

    const capture = captureAllApiCalls(page)
    requestHandler?.(request)
    await Promise.resolve()

    const stopping = capture.stop()
    let settled = false
    void stopping.then(() => {
      settled = true
    })
    await Promise.resolve()
    expect(settled).toBe(false)

    resolveResponseBody?.({ result: "complete" })
    await expect(stopping).resolves.toEqual([
      expect.objectContaining({
        method: "POST",
        requestBody: { model: "local-uat-chat" },
        responseBody: { result: "complete" },
        status,
      }),
    ])
  })

  it.each([
    ["GET", "/notifications/stream?after=0", null],
    ["POST", "/chat/completions", { model: "local-uat-chat", stream: true }],
  ])("returns %s %s stream metadata without waiting for its body", async (method, path, requestBody) => {
    let requestHandler: ((request: Request) => void) | null = null
    const page = {
      on: (_event: string, handler: (request: Request) => void) => {
        requestHandler = handler
      },
      removeListener: vi.fn(),
    } as unknown as Page
    const request = {
      url: () => `http://127.0.0.1:8000/api/v1${path}`,
      method: () => method,
      response: async () => ({
        status: () => 200,
        headers: () => ({ "content-type": "text/event-stream; charset=utf-8" }),
        // A live SSE subscription never finishes its response body.
        json: () => new Promise<never>(() => {}),
      }),
      postDataJSON: () => requestBody,
    } as unknown as Request

    const capture = captureAllApiCalls(page)
    requestHandler?.(request)
    await expect(capture.stop()).resolves.toEqual([
      expect.objectContaining({
        url: request.url(),
        method,
        requestBody,
        status: 200,
        responseBody: null,
      }),
    ])
  }, 500)
})
