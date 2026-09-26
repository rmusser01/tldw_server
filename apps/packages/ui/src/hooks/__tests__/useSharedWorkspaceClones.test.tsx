import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useSharedWorkspaceClones } from "../useSharedWorkspaceClones"
import {
  CLONE_RECOVERY_KEY,
  CLONE_RECOVERY_TTL
} from "@/services/shared-clone-recovery"
import { clonePayload } from "@/services/tldw/domains/__tests__/shared-workspace-clone.fixture"
import { TldwApiError } from "@/services/tldw/api-error"

const mocks = vi.hoisted(() => ({
  storageListeners: new Set<
    (changes: Record<string, unknown>, area: string) => void
  >(),
  url: vi.fn(),
  user: vi.fn(),
  clone: vi.fn(),
  cloneStatus: vi.fn()
}))
vi.mock("wxt/browser", () => ({
  browser: {
    storage: {
      onChanged: {
        addListener: (
          listener: (changes: Record<string, unknown>, area: string) => void
        ) => mocks.storageListeners.add(listener),
        removeListener: (
          listener: (changes: Record<string, unknown>, area: string) => void
        ) => mocks.storageListeners.delete(listener)
      }
    }
  }
}))
vi.mock("@/services/tldw-server", () => ({ getTldwServerURL: mocks.url }))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: mocks.user }
}))
vi.mock("@/services/tldw/domains/shared-workspaces", () => ({
  createSharedWorkspaceCloneContext: async () => {
    const url = new URL(await mocks.url())
    const user = await mocks.user()
    return {
      scope: JSON.stringify([
        url.origin,
        url.pathname.replace(/\/+$/, ""),
        String(user.id)
      ]),
      api: { clone: mocks.clone, cloneStatus: mocks.cloneStatus }
    }
  }
}))

describe("clone session scope", () => {
  beforeEach(() => {
    window.localStorage.clear()
    const tails = new Map<string, Promise<unknown>>()
    vi.stubGlobal("navigator", {
      locks: {
        request: (name: string, options: unknown, callback?: () => unknown) => {
          const run = (tails.get(name) ?? Promise.resolve()).then(
            callback ?? (options as () => unknown)
          )
          tails.set(
            name,
            run.catch(() => undefined)
          )
          return run
        }
      }
    })
    mocks.url.mockReset().mockResolvedValue("https://server.example")
    mocks.user.mockReset().mockResolvedValue({ id: 42 })
    mocks.clone.mockReset().mockResolvedValue(clonePayload())
    mocks.cloneStatus.mockReset().mockResolvedValue(clonePayload())
  })
  afterEach(() => vi.unstubAllGlobals())

  it("waits for authenticated scope before accepting commands", async () => {
    mocks.user.mockRejectedValue(new Error("unauthenticated"))
    const { result } = renderHook(() => useSharedWorkspaceClones())
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.status).toBe("auth_required"))
    expect(mocks.clone).not.toHaveBeenCalled()
  })

  it("clears rows and recovery immediately at logout, ignoring a late response", async () => {
    let resolve!: (value: unknown) => void
    mocks.clone.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done
        })
    )
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(mocks.clone).toHaveBeenCalled())
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" }
        })
      )
    )
    await act(async () => resolve(clonePayload("succeeded")))
    expect(result.current.rows).toEqual([])
    expect(result.current.scope).toBeNull()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("preserves the previous server's recovery and resumes only after switching back", async () => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.rows[0]?.operation).toBeTruthy())
    const raw = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    mocks.url.mockResolvedValue("https://other.example")
    act(() => window.dispatchEvent(new Event("tldw:config-updated")))
    await waitFor(() => expect(result.current.status).toBe("recovery_conflict"))
    expect(result.current.scope).toBeNull()
    expect(result.current.rows).toEqual([])
    act(() => result.current.begin(43, "Blocked"))
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    mocks.url.mockResolvedValue("https://server.example")
    act(() => result.current.refresh())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    await waitFor(() => expect(result.current.rows[0]?.operation).toBeTruthy())
    expect(mocks.clone).toHaveBeenCalledTimes(1)
  })

  it.each(["before event", "after event"])(
    "blocks a queued admission on foreign recovery %s without discarding it",
    async (timing) => {
      const { result } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      let release!: () => void
      const held = navigator.locks.request(
        CLONE_RECOVERY_KEY,
        () =>
          new Promise<void>((done) => {
            release = done
          })
      )
      await act(async () => {
        await Promise.resolve()
      })
      act(() => result.current.begin(42, "Queued"))
      const raw = JSON.stringify({ version: 1, scope: "B", records: [] })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      if (timing === "after event")
        act(() =>
          window.dispatchEvent(
            new StorageEvent("storage", {
              key: CLONE_RECOVERY_KEY,
              newValue: raw
            })
          )
        )
      await act(async () => {
        release()
        await held
      })
      await waitFor(() =>
        expect(result.current.status).toBe("recovery_conflict")
      )
      expect(result.current.scope).toBeNull()
      expect(result.current.rows).toEqual([])
      act(() => result.current.refresh())
      await waitFor(() =>
        expect(result.current.status).toBe("recovery_conflict")
      )
      act(() => result.current.begin(43, "Blocked"))
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
      expect(mocks.clone).not.toHaveBeenCalled()
      expect(mocks.cloneStatus).not.toHaveBeenCalled()
    }
  )

  it("does not become ready before the verified scope's recovery read completes", async () => {
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          release = done
        })
    )
    await Promise.resolve()
    const raw = JSON.stringify({ version: 1, scope: "B", records: [] })
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await act(async () => {
      await Promise.resolve()
    })
    const pendingStatus = result.current.status
    await act(async () => {
      release()
      await held
    })
    expect(pendingStatus).toBe("loading")
    await waitFor(() => expect(result.current.status).toBe("recovery_conflict"))
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
  })

  it.each([
    "tldwConfig",
    "tldwManualSessionApiKey",
    "tldwRefreshRotation",
    "tldwCookieSessionConfig"
  ])("synchronously blocks commands on cross-tab %s changes", async (key) => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    mocks.user.mockImplementation(() => new Promise(() => undefined))
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key, newValue: "changed" })
      )
      result.current.begin(42, "Stale click")
    })
    expect(result.current.scope).toBeNull()
    expect(result.current.status).toBe("loading")
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.clone).not.toHaveBeenCalled()
  })

  it.each(["local", "session", "sync"])(
    "invalidates on extension %s storage changes and removes its listener",
    async (area) => {
      const { result, unmount } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      mocks.user.mockImplementation(() => new Promise(() => undefined))
      act(() => {
        for (const listener of mocks.storageListeners)
          listener({ tldwConfig: { newValue: {} } }, area)
        result.current.begin(42, "Stale click")
      })
      expect(result.current.scope).toBeNull()
      unmount()
      expect(mocks.storageListeners.size).toBe(0)
      expect(mocks.clone).not.toHaveBeenCalled()
    }
  )

  it("preserves the same operation across a same-principal token refresh", async () => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.rows[0]?.operation).toBeTruthy())
    act(() => window.dispatchEvent(new Event("tldw:config-updated")))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(result.current.rows[0].entry.operation_id).toBeTruthy()
    expect(mocks.clone).toHaveBeenCalledTimes(1)
  })

  it.each([
    "logout",
    "account",
    "tldwConfig",
    "tldwManualSessionApiKey",
    "tldwRefreshRotation",
    "tldwCookieSessionConfig"
  ])("cancels queued persistence at a %s boundary", async (boundary) => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          release = done
        })
    )
    await act(async () => {
      await Promise.resolve()
    })
    act(() => result.current.begin(42, "Research"))
    if (boundary !== "logout") mocks.user.mockResolvedValue({ id: 99 })
    act(() =>
      window.dispatchEvent(
        boundary.startsWith("tldw")
          ? new StorageEvent("storage", { key: boundary, newValue: "changed" })
          : new CustomEvent("tldw:auth-principal-changed", {
              detail: { kind: boundary }
            })
      )
    )
    await act(async () => {
      release()
      await held
    })
    await waitFor(() =>
      expect(result.current.status).toBe(
        boundary === "logout" ? "auth_required" : "ready"
      )
    )
    expect(mocks.clone).not.toHaveBeenCalled()
    expect(result.current.rows).toEqual([])
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it.each(["poll", "uncertain retry"])(
    "cancels %s on a cross-tab account change and ignores a late receipt",
    async (phase) => {
      let complete!: (value: unknown) => void
      const pending = new Promise((resolve) => {
        complete = resolve
      })
      if (phase === "uncertain retry")
        mocks.clone.mockRejectedValueOnce(new TypeError("offline"))
      const { result } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      act(() => result.current.begin(42, "Research"))
      await waitFor(() => expect(result.current.rows[0]?.pending).toBe(false))
      if (phase === "poll") {
        mocks.cloneStatus.mockReturnValue(pending)
        act(() => result.current.refresh(42))
        await waitFor(() => expect(mocks.cloneStatus).toHaveBeenCalled())
      } else {
        expect(result.current.rows[0].issue).toBe("uncertain")
      }
      const originalCalls = mocks.clone.mock.calls.length
      mocks.user.mockImplementation(() => new Promise(() => undefined))
      act(() => {
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: "tldwConfig",
            newValue: "changed"
          })
        )
        result.current.refresh(42)
        result.current.begin(42, "Stale retry")
      })
      if (phase === "poll") {
        expect(mocks.cloneStatus.mock.calls[0][2].aborted).toBe(true)
        await act(async () => complete(clonePayload("succeeded")))
      }
      expect(result.current.scope).toBeNull()
      expect(result.current.rows).toEqual([])
      expect(mocks.clone).toHaveBeenCalledTimes(originalCalls)
    }
  )

  it("refreshes admission transport only after the same principal is reverified", async () => {
    mocks.clone.mockRejectedValueOnce(new TypeError("Response lost"))
    const originalClone = mocks.clone
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.rows[0]?.issue).toBe("uncertain"))
    const newCredentialClone = vi.fn().mockResolvedValue(clonePayload())
    mocks.clone = newCredentialClone
    try {
      act(() =>
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: "tldwRefreshRotation",
            newValue: "changed"
          })
        )
      )
      await waitFor(() => expect(result.current.status).toBe("ready"))
      vi.useFakeTimers()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      await act(async () => {
        result.current.refresh(42)
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(newCredentialClone).toHaveBeenCalledTimes(1)
      expect(newCredentialClone.mock.calls[0].slice(0, 3)).toEqual(
        originalClone.mock.calls[0].slice(0, 3)
      )
      expect(originalClone).toHaveBeenCalledTimes(1)
      expect(result.current.rows[0].entry.idempotency_key).toBeTruthy()
    } finally {
      vi.useRealTimers()
      mocks.clone = originalClone
    }
  })

  it("does not reuse a disposed manager when principal verification changes during locked cleanup", async () => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          release = done
        })
    )
    await act(async () => {
      await Promise.resolve()
    })
    mocks.user.mockResolvedValue({ id: 99 })
    await act(async () => {
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    mocks.user.mockResolvedValue({ id: 42 })
    await act(async () => {
      window.dispatchEvent(new Event("focus"))
    })
    await act(async () => {
      release()
      await held
    })
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.rows[0]?.operation).toBeTruthy())
  })

  it("requires explicit re-verification after a scope-change 412 and never substitutes another account", async () => {
    mocks.clone.mockRejectedValueOnce(
      new TldwApiError("Account changed", 412, {
        code: "request_config_scope_changed",
        retryable: false,
        recovery_action: "refresh"
      })
    )
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.status).toBe("auth_required"))
    expect(result.current.scope).toBeNull()
    expect(
      JSON.parse(window.localStorage.getItem(CLONE_RECOVERY_KEY)!).records[0]
        .idempotency_key
    ).toBeTruthy()
    expect(mocks.user).toHaveBeenCalledTimes(1)
    act(() => result.current.begin(42, "Stale retry"))
    expect(mocks.clone).toHaveBeenCalledTimes(1)
    const raw = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    mocks.user.mockResolvedValue({ id: 99 })
    act(() => result.current.refresh())
    await waitFor(() => expect(result.current.status).toBe("recovery_conflict"))
    expect(result.current.scope).toBeNull()
    expect(result.current.rows).toEqual([])
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    expect(mocks.clone).toHaveBeenCalledTimes(1)
  })

  it.each([
    { status: 401, detail: {}, label: "Expired token" },
    {
      status: 403,
      detail: "CSRF token validation failed",
      label: "CSRF token validation failed"
    },
    {
      status: 403,
      detail: { code: "sharing_permission_required" },
      label: "Permission revoked"
    }
  ])(
    "preserves the key after replay $label for reverified same-scope credentials",
    async ({ status, detail, label }) => {
      mocks.clone.mockRejectedValueOnce(new TypeError("Accepted response lost"))
      const originalClone = mocks.clone
      const { result } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      act(() => result.current.begin(42, "Research"))
      await waitFor(() =>
        expect(result.current.rows[0]?.issue).toBe("uncertain")
      )
      mocks.clone.mockRejectedValueOnce(new TldwApiError(label, status, detail))
      vi.useFakeTimers()
      try {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(5000)
        })
        await act(async () => {
          result.current.refresh(42)
          await vi.advanceTimersByTimeAsync(0)
        })
        expect(result.current.status).toBe("auth_required")
        expect(result.current.scope).toBeNull()
        const rotated = vi.fn().mockResolvedValue(clonePayload())
        mocks.clone = rotated
        await act(async () => {
          result.current.refresh()
          await vi.advanceTimersByTimeAsync(0)
        })
        expect(result.current.status).toBe("ready")
        expect(rotated.mock.calls[0].slice(0, 3)).toEqual(
          originalClone.mock.calls[0].slice(0, 3)
        )
        expect(originalClone).toHaveBeenCalledTimes(2)
        expect(mocks.user).toHaveBeenCalledTimes(2)
      } finally {
        vi.useRealTimers()
        mocks.clone = originalClone
      }
    }
  )

  it("reverifies plain-string CSRF 403 before explicit refresh uses fresh transport with the same key", async () => {
    mocks.clone.mockRejectedValueOnce(
      new TldwApiError("Forbidden", 403, "CSRF token validation failed")
    )
    const original = mocks.clone
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => result.current.begin(42, "Research"))
    await waitFor(() => expect(result.current.status).toBe("auth_required"))
    expect(result.current.scope).toBeNull()
    act(() => result.current.begin(42, "Blocked"))
    expect(original).toHaveBeenCalledTimes(1)
    const rotated = vi.fn().mockResolvedValue(clonePayload())
    mocks.clone = rotated
    try {
      act(() => result.current.refresh())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      await waitFor(() => expect(rotated).toHaveBeenCalledTimes(1))
      expect(rotated.mock.calls[0].slice(0, 3)).toEqual(
        original.mock.calls[0].slice(0, 3)
      )
      expect(mocks.user).toHaveBeenCalledTimes(2)
      expect(original).toHaveBeenCalledTimes(1)
    } finally {
      mocks.clone = original
    }
  })

  it("preserves a new account's cross-tab recovery written while old-scope cleanup waits", async () => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    await act(async () => {
      await Promise.resolve()
    })
    mocks.user.mockResolvedValue({ id: 99 })
    await act(async () => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: "tldwConfig", newValue: "changed" })
      )
    })
    const scope = '["https://server.example","","99"]'
    const record = {
      share_id: 43,
      operation_id: clonePayload().operation_id,
      expires_at: Date.now() + CLONE_RECOVERY_TTL
    }
    window.localStorage.setItem(
      CLONE_RECOVERY_KEY,
      JSON.stringify({ version: 1, scope, records: [record] })
    )
    mocks.cloneStatus.mockResolvedValue(clonePayload("running", 43))
    await act(async () => {
      release()
      await held
    })
    await waitFor(() => expect(result.current.scope).toBe(scope))
    await waitFor(() =>
      expect(result.current.rows[0]?.operation?.status).toBe("running")
    )
    expect(
      JSON.parse(window.localStorage.getItem(CLONE_RECOVERY_KEY)!).records
    ).toEqual([record])
    expect(mocks.clone).not.toHaveBeenCalled()
  })

  it.each(["read", "write"])(
    "preserves a clicked admission waiting for its %s lock when another tab updates same-scope recovery",
    async (phase) => {
      mocks.clone.mockImplementation(async (shareId: number) =>
        clonePayload("succeeded", shareId)
      )
      const { result } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      let release!: () => void
      let held = navigator.locks.request(
        CLONE_RECOVERY_KEY,
        () =>
          new Promise<void>((done) => {
            release = done
          })
      )
      await act(async () => {
        await Promise.resolve()
      })
      act(() => result.current.begin(42, "My click"))
      if (phase === "write") {
        const first = held
        const releaseFirst = release
        held = navigator.locks.request(
          CLONE_RECOVERY_KEY,
          () =>
            new Promise<void>((done) => {
              release = done
            })
        )
        await act(async () => {
          releaseFirst()
          await first
        })
      }
      const otherTabRecord = {
        share_id: 43,
        expires_at: Date.now() + CLONE_RECOVERY_TTL,
        name: "Other tab (Copy)",
        idempotency_key: "other-tab-key-00000001"
      }
      const raw = JSON.stringify({
        version: 1,
        scope: result.current.scope,
        records: [otherTabRecord]
      })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      act(() =>
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: CLONE_RECOVERY_KEY,
            newValue: raw
          })
        )
      )
      const statusAfterUpdate = result.current.status
      await act(async () => {
        release()
        await held
      })
      await waitFor(() =>
        expect(
          result.current.rows.find((row) => row.entry.share_id === 42)
            ?.operation?.status
        ).toBe("succeeded")
      )
      await waitFor(() =>
        expect(
          result.current.rows.find((row) => row.entry.share_id === 43)
            ?.operation?.status
        ).toBe("succeeded")
      )
      expect(statusAfterUpdate).toBe("ready")
      expect(mocks.user).toHaveBeenCalledTimes(1)
      expect(new Set(mocks.clone.mock.calls.map(([id]) => id))).toEqual(
        new Set([42, 43])
      )
      const stored = JSON.parse(
        window.localStorage.getItem(CLONE_RECOVERY_KEY)!
      )
      expect(
        stored.records.map((entry: { share_id: number }) => entry.share_id)
      ).toEqual([42, 43])
    }
  )

  it.each([null, CLONE_RECOVERY_KEY])(
    "revalidates auth and cancels a queued click when storage key %s is cleared",
    async (key) => {
      const { result } = renderHook(() => useSharedWorkspaceClones())
      await waitFor(() => expect(result.current.status).toBe("ready"))
      let release!: () => void
      const held = navigator.locks.request(
        CLONE_RECOVERY_KEY,
        () =>
          new Promise<void>((done) => {
            release = done
          })
      )
      await act(async () => {
        await Promise.resolve()
      })
      act(() => result.current.begin(42, "Research"))
      mocks.user.mockRejectedValue(new Error("logged out"))
      act(() =>
        window.dispatchEvent(
          new StorageEvent("storage", { key, newValue: null })
        )
      )
      await act(async () => {
        release()
        await held
      })
      await waitFor(() => expect(result.current.status).toBe("auth_required"))
      expect(mocks.clone).not.toHaveBeenCalled()
      expect(result.current.rows).toEqual([])
    }
  )

  it("revalidates a non-null recovery update belonging to another principal", async () => {
    const { result } = renderHook(() => useSharedWorkspaceClones())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    mocks.user.mockResolvedValue({ id: 99 })
    const raw = JSON.stringify({
      version: 1,
      scope: JSON.stringify(["https://server.example", "", "99"]),
      records: []
    })
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    act(() =>
      window.dispatchEvent(
        new StorageEvent("storage", { key: CLONE_RECOVERY_KEY, newValue: raw })
      )
    )
    await waitFor(() =>
      expect(result.current.scope).toBe(
        JSON.stringify(["https://server.example", "", "99"])
      )
    )
    expect(result.current.rows).toEqual([])
  })
})
