import { cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const useStorageMock = vi.hoisted(() => vi.fn())
const getConfigMock = vi.hoisted(() => vi.fn())

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => useStorageMock(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => getConfigMock(...args)
  }
}))

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

const createUseStorageImplementation = ({
  serverUrl = "",
  authMode = "single-user",
  apiKey = "",
  accessToken = "",
  refreshRotation
}: {
  serverUrl?: string
  authMode?: string
  apiKey?: string
  accessToken?: string
  refreshRotation?: unknown
}) => {
  return (key: string, defaultValue: unknown) => {
    if (key === "serverUrl") {
      return [serverUrl || defaultValue, vi.fn()] as const
    }
    if (key === "authMode") {
      return [authMode, vi.fn()] as const
    }
    if (key === "apiKey") {
      return [apiKey, vi.fn()] as const
    }
    if (key === "accessToken") {
      return [accessToken, vi.fn()] as const
    }
    if (key === "tldwRefreshRotation") {
      return [refreshRotation, vi.fn()] as const
    }
    return [defaultValue, vi.fn()] as const
  }
}

describe("useCanonicalConnectionConfig", () => {
  const originalWindow = globalThis.window

  beforeEach(() => {
    useStorageMock.mockReset()
    getConfigMock.mockReset()
    useStorageMock.mockImplementation(createUseStorageImplementation({}))
  })

  afterEach(() => {
    cleanup()
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "window", {
      value: originalWindow,
      configurable: true
    })
  })

  it("canonicalizes quickstart webui config hydration to the current page origin", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    useStorageMock.mockImplementation(
      createUseStorageImplementation({
        serverUrl: "http://127.0.0.1:8000",
        apiKey: "frontend-key"
      })
    )
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "frontend-key"
    })

    const mockWindow = Object.create(originalWindow)
    Object.defineProperty(mockWindow, "location", {
      value: {
        origin: "http://192.168.5.184:3000",
        protocol: "http:",
        hostname: "192.168.5.184"
      },
      configurable: true
    })
    Object.defineProperty(globalThis, "window", {
      value: mockWindow,
      configurable: true
    })

    const { result } = renderHook(() => useCanonicalConnectionConfig())

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.config).toMatchObject({
      serverUrl: "http://192.168.5.184:3000",
      authMode: "single-user",
      apiKey: "frontend-key"
    })
  })

  it("keeps explicit hosts intact for extension quickstart contexts", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    useStorageMock.mockImplementation(
      createUseStorageImplementation({
        serverUrl: "http://192.168.5.186:8000",
        apiKey: "frontend-key"
      })
    )
    getConfigMock.mockResolvedValue({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "frontend-key"
    })

    const mockWindow = Object.create(originalWindow)
    Object.defineProperty(mockWindow, "location", {
      value: {
        origin: "chrome-extension://abcdefghijklmnop",
        protocol: "chrome-extension:",
        hostname: "abcdefghijklmnop"
      },
      configurable: true
    })
    Object.defineProperty(globalThis, "window", {
      value: mockWindow,
      configurable: true
    })

    const { result } = renderHook(() => useCanonicalConnectionConfig())

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.config).toMatchObject({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "frontend-key"
    })
  })

  it("rehydrates mounted consumers when scoped refresh rotation changes", async () => {
    let refreshRotation: unknown
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) =>
      createUseStorageImplementation({ refreshRotation })(key, defaultValue)
    )
    getConfigMock
      .mockResolvedValueOnce({
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        accessToken: "stale-access"
      })
      .mockResolvedValue({
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        accessToken: "rotated-access"
      })

    const { result, rerender } = renderHook(() =>
      useCanonicalConnectionConfig()
    )
    await waitFor(() => {
      expect(result.current.config?.accessToken).toBe("stale-access")
    })

    refreshRotation = { version: 1, accessToken: "rotated-access" }
    rerender()

    await waitFor(() => {
      expect(result.current.config?.accessToken).toBe("rotated-access")
    })
    expect(getConfigMock).toHaveBeenCalledTimes(2)
  })

})
