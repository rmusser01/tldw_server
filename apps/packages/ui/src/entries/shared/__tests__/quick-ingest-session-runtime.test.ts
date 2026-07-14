import { afterEach, describe, expect, it, vi } from "vitest"

import {
  createQuickIngestSessionRuntime,
  parseQuickIngestCompactRunSession,
} from "@/entries/shared/quick-ingest-session-runtime"

describe("quick ingest session runtime", () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it("returns an ack only after the durable start marker is stored", async () => {
    const run = vi.fn(async () => ({ results: [{ id: "1", status: "ok" }] }))
    const emit = vi.fn()
    let releaseMarker!: () => void
    const markerGate = new Promise<void>((resolve) => {
      releaseMarker = resolve
    })
    const saveRunSession = vi.fn(async (record: any) => {
      if (record?.kind === "start") await markerGate
    })
    const runtime = createQuickIngestSessionRuntime({ run, emit, saveRunSession })

    const ackPromise = Promise.resolve(runtime.start({ entries: [], files: [] }))
    let acknowledged = false
    void ackPromise.then(() => {
      acknowledged = true
    })

    await Promise.resolve()
    expect(acknowledged).toBe(false)
    expect(run).not.toHaveBeenCalled()

    releaseMarker()
    const ack = await ackPromise
    expect(ack.ok).toBe(true)
    expect(typeof ack.sessionId).toBe("string")

    await vi.waitFor(() => {
      expect(run).toHaveBeenCalledTimes(1)
    })
  })

  it("emits completed event keyed by session id on success", async () => {
    const run = vi.fn(async () => ({ results: [{ id: "1", status: "ok" }] }))
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({ run, emit })

    const ack = await runtime.start({ entries: [], files: [] })

    await vi.waitFor(() => {
      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/completed",
        expect.objectContaining({
          sessionId: ack.sessionId,
          results: expect.any(Array)
        })
      )
    })
  })

  it("marks cancelled sessions immediately and suppresses completed emission", async () => {
    let release: (() => void) | null = null
    let registeredController: AbortController | null = null
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const run = vi.fn(async (_payload: any, context: any) => {
      registeredController = new AbortController()
      context.registerAbortController(registeredController)
      await gate
      return { results: [] }
    })
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({ run, emit })

    const ack = await runtime.start({ entries: [], files: [] })
    await vi.waitFor(() => {
      expect(registeredController).toBeTruthy()
    })
    const cancelResp = await runtime.cancel(ack.sessionId, "user_cancelled")

    expect(cancelResp).toEqual({ ok: true })
    expect(registeredController?.signal.aborted).toBe(true)
    expect(
      emit.mock.calls.some(
        ([type, payload]) =>
          type === "tldw:quick-ingest/cancelled" &&
          payload?.sessionId === ack.sessionId
      )
    ).toBe(true)

    release?.()
    await Promise.resolve()
    await Promise.resolve()

    expect(
      emit.mock.calls.some(
        ([type, payload]) =>
          type === "tldw:quick-ingest/completed" &&
          payload?.sessionId === ack.sessionId
      )
    ).toBe(false)
  })

  it("persists only the run id and compact occurrence mappings for v2 recovery", async () => {
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const payload = {
      pendingRunRequest: {
        contractVersion: 2,
        inputs: [{ occurrenceId: "occ-1" }],
      },
      entries: [],
      files: [],
    }
    const saveRunSession = vi.fn()
    const run = vi.fn(async (receivedPayload: any, context: any) => {
      await context.setRunTracking({
        mode: "extension-runtime",
        runId: "run-1",
        submissionState: "submitting",
        submissionOccurrenceIds: ["occ-1"],
        submittedItemIds: ["occ-1"],
        plannedItemIds: ["collection-item-not-runtime-state"],
        batchIds: ["batch-1"],
        jobIds: [71],
        jobIdToItemId: { "71": "occ-1" },
        startedAt: 123,
      })
      await gate
      return { results: [] }
    })
    const runtime = createQuickIngestSessionRuntime({
      run,
      emit: vi.fn(),
      saveRunSession,
      createSessionId: () => "session-1",
    } as any)

    runtime.start(payload)

    await vi.waitFor(() => {
      expect(saveRunSession).toHaveBeenCalledTimes(2)
    })
    expect(run.mock.calls[0]?.[0]).toBe(payload)
    expect(saveRunSession.mock.calls[0]?.[0]).toEqual({
      version: 1,
      kind: "start",
      sessionId: "session-1",
      generation: expect.any(String),
      attemptToken: expect.any(String),
      occurrenceIds: ["occ-1"],
      startedAt: expect.any(Number),
    })
    expect(saveRunSession.mock.calls[1]?.[0]).toEqual({
      version: 1,
      kind: "run",
      sessionId: "session-1",
      runId: "run-1",
      generation: expect.any(String),
      attemptToken: expect.any(String),
      submissionState: "submitting",
      occurrenceIds: ["occ-1"],
      jobIdToItemId: { "71": "occ-1" },
      startedAt: 123,
    })

    release()
  })

  it("records a caller-owned start marker before submission and rejects conflicting identity reuse", async () => {
    let releaseMarker!: () => void
    const markerGate = new Promise<void>((resolve) => {
      releaseMarker = resolve
    })
    const saveRunSession = vi.fn(async (record: any) => {
      if (record?.kind === "start") await markerGate
    })
    const run = vi.fn(async () => ({ results: [] }))
    const runtime = createQuickIngestSessionRuntime({
      run,
      emit: vi.fn(),
      saveRunSession,
    } as any)
    const payload = {
      pendingRunRequest: {
        inputs: [{ occurrenceId: "occ-stable", inputKind: "direct_url" }],
      },
      entries: [{ id: "occ-stable", url: "https://example.com/stable" }],
      files: [],
    }

    const firstPromise = Promise.resolve(
      runtime.start(payload, {
        sessionId: "qi-caller-stable",
        attemptToken: "attempt-caller-stable",
      } as any)
    )
    const duplicatePromise = Promise.resolve(
      runtime.start(payload, {
        sessionId: "qi-caller-stable",
        attemptToken: "attempt-caller-stable",
      } as any)
    )
    const conflict = await runtime.start(
      {
        ...payload,
        pendingRunRequest: {
          inputs: [{ occurrenceId: "occ-conflict", inputKind: "direct_url" }],
        },
      },
      {
        sessionId: "qi-caller-stable",
        attemptToken: "attempt-conflicting-reuse",
      } as any
    )

    let firstAcknowledged = false
    void firstPromise.then(() => {
      firstAcknowledged = true
    })
    await Promise.resolve()
    expect(firstAcknowledged).toBe(false)
    expect(conflict).toMatchObject({
      ok: false,
      sessionId: "qi-caller-stable",
      error: expect.stringMatching(/conflict|attempt|reuse/i),
    })
    expect(run).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(saveRunSession).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "start",
          sessionId: "qi-caller-stable",
          generation: expect.any(String),
          attemptToken: "attempt-caller-stable",
          occurrenceIds: ["occ-stable"],
        })
      )
    })

    releaseMarker()
    const [first, duplicate] = await Promise.all([firstPromise, duplicatePromise])
    expect(first).toEqual({ ok: true, sessionId: "qi-caller-stable" })
    expect(duplicate).toEqual(first)
    await vi.waitFor(() => expect(run).toHaveBeenCalledTimes(1))
  })

  it("persists only a caller-owned opaque attempt token and never fingerprints secret-bearing payload fields", async () => {
    let release!: () => void
    const runGate = new Promise<void>((resolve) => {
      release = resolve
    })
    const saveRunSession = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async () => {
        await runGate
        return { results: [] }
      }),
      emit: vi.fn(),
      saveRunSession,
    } as any)

    await runtime.start(
      {
        entries: [{ id: "occ-secret" }],
        custom_headers: { Authorization: "Bearer should-never-be-fingerprinted" },
        custom_cookies: "session=should-never-be-fingerprinted",
        advancedValues: Object.fromEntries(
          Array.from({ length: 600 }, (_, index) => [`unbounded-${index}`, "x".repeat(500)])
        ),
      },
      {
        sessionId: "qi-secret-safe",
        attemptToken: "attempt-secret-safe",
      } as any
    )

    const marker = saveRunSession.mock.calls[0]?.[0]
    expect(marker).toMatchObject({
      kind: "start",
      sessionId: "qi-secret-safe",
      attemptToken: "attempt-secret-safe",
    })
    expect(marker).not.toHaveProperty("requestFingerprint")
    expect(JSON.stringify(marker)).not.toMatch(/Bearer|cookie|should-never|unbounded-/i)
    release()
  })

  it("restores a durable pre-create marker as interrupted without rerunning submission", async () => {
    const marker = {
      version: 1,
      kind: "start",
      sessionId: "session-pre-create",
      generation: "generation-pre-create",
      attemptToken: "attempt-pre-create",
      occurrenceIds: ["occ-pre-create"],
      startedAt: Date.now(),
    }
    const run = vi.fn()
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run,
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([marker]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn(),
    } as any)

    await runtime.restore()

    expect(run).not.toHaveBeenCalled()
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/interrupted",
      expect.objectContaining({
        sessionId: "session-pre-create",
        recoverable: true,
        error: expect.stringMatching(/before.*run|create|interrupted/i),
      })
    )
    await expect(runtime.replay("session-pre-create")).resolves.toMatchObject({
      ok: true,
      active: true,
      event: expect.objectContaining({ type: "tldw:quick-ingest/interrupted" }),
    })
  })

  it("polls a compact run after worker recreation and emits occurrence-aware progress", async () => {
    const record = {
      version: 1 as const,
      sessionId: "session-restored",
      runId: "run-restored",
      occurrenceIds: ["occ-restored"],
      jobIdToItemId: { "81": "occ-restored" },
      startedAt: 456,
    }
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 81,
          attempt: 4,
          status: "running",
          sourceItemId: "occ-restored",
        },
      ],
      errorMessage: null,
    })
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([record]),
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    await runtime.restore()

    expect(reattachRun).toHaveBeenCalledTimes(1)
    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "extension-runtime",
        runId: "run-restored",
        submissionOccurrenceIds: ["occ-restored"],
        jobIdToItemId: { "81": "occ-restored" },
      }),
      { transportPreference: "poll" }
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/progress",
      expect.objectContaining({
        sessionId: "session-restored",
        runId: "run-restored",
        occurrenceId: "occ-restored",
        jobId: 81,
        attempt: 4,
        status: "running",
        result: expect.objectContaining({
          id: "occ-restored",
          status: "running",
        }),
      })
    )
  })

  it("restores compact submission state for authoritative unsent cleanup", async () => {
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          sessionId: "session-partial",
          runId: "run-partial",
          submissionState: "run_created",
          occurrenceIds: ["occ-partial"],
          jobIdToItemId: {},
          startedAt: 460,
        },
      ]),
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    await runtime.restore()

    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-partial",
        submissionState: "run_created",
      }),
      { transportPreference: "poll" }
    )
  })

  it("keeps cleanup-required extension recovery active until authoritative cleanup succeeds", async () => {
    const emit = vi.fn()
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "cancelled",
      jobs: [
        {
          jobId: null,
          status: "cancelled",
          sourceItemId: "occ-cleanup-extension",
          error: "Cancelled after cleanup retry.",
        },
      ],
      errorMessage: "Cancelled after cleanup retry.",
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-cleanup-extension",
          submissionState: "run_created",
          submissionOccurrenceIds: ["occ-cleanup-extension"],
        })
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-cleanup-extension",
          submissionState: "cleanup_required",
          submissionOccurrenceIds: ["occ-cleanup-extension"],
        })
        throw new Error("Initial extension cleanup failed.")
      }),
      emit,
      saveRunSession: vi.fn(),
      reattachRun,
      createSessionId: () => "session-cleanup-extension",
    } as any)

    runtime.start({
      pendingRunRequest: {
        inputs: [{ occurrenceId: "occ-cleanup-extension" }],
      },
    })

    await vi.waitFor(() => {
      expect(reattachRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runId: "run-cleanup-extension",
          submissionState: "cleanup_required",
        }),
        { transportPreference: "poll" }
      )
    })
    expect(emit).not.toHaveBeenCalledWith(
      "tldw:quick-ingest/failed",
      expect.objectContaining({ sessionId: "session-cleanup-extension" })
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/cancelled",
      expect.objectContaining({
        sessionId: "session-cleanup-extension",
        results: [
          expect.objectContaining({
            id: "occ-cleanup-extension",
            status: "error",
          }),
        ],
      })
    )
  })

  it("retains interrupted restores and emits a recoverable interruption", async () => {
    const saveRunSession = vi.fn()
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          sessionId: "session-auth",
          runId: "run-auth",
          submissionState: "acknowledged",
          occurrenceIds: ["occ-auth"],
          jobIdToItemId: {},
          startedAt: 461,
        },
      ]),
      saveRunSession,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "interrupted",
        jobs: [],
        errorMessage: "Authorization required.",
      }),
    } as any)

    await runtime.restore()

    expect(saveRunSession).not.toHaveBeenCalledWith(
      null,
      "session-auth",
      "run-auth"
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/interrupted",
      expect.objectContaining({
        sessionId: "session-auth",
        runId: "run-auth",
        recoverable: true,
        error: "Authorization required.",
      })
    )
    expect(runtime.hasSession("session-auth")).toBe(true)
  })

  it("ignores late run tracking after a local cancel wins the create race", async () => {
    let release!: () => void
    let finishRun!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const runFinished = new Promise<void>((resolve) => {
      finishRun = resolve
    })
    const saveRunSession = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        try {
          await gate
          await context.setRunTracking({
            mode: "extension-runtime",
            runId: "run-too-late",
            submissionState: "run_created",
            submissionOccurrenceIds: ["occ-too-late"],
          })
          return { results: [] }
        } finally {
          finishRun()
        }
      }),
      emit: vi.fn(),
      saveRunSession,
      createSessionId: () => "session-create-race",
    } as any)

    runtime.start({ pendingRunRequest: { contractVersion: 2 } })
    await Promise.resolve()
    await runtime.cancel("session-create-race")
    release()
    await runFinished
    expect(
      saveRunSession.mock.calls.some(([record]) => record?.kind === "run")
    ).toBe(false)
  })

  it("waits for delayed restoration before cancelling a restored run", async () => {
    let resolveLoad!: (records: unknown[]) => void
    const loadGate = new Promise<unknown[]>((resolve) => {
      resolveLoad = resolve
    })
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(() => loadGate),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      }),
      cancelRun,
    } as any)

    const restoring = runtime.restore()
    const cancelling = runtime.cancel("session-delayed", "user_cancelled")
    resolveLoad([
      {
        version: 1,
        sessionId: "session-delayed",
        runId: "run-delayed",
        submissionState: "acknowledged",
        occurrenceIds: ["occ-delayed"],
        jobIdToItemId: {},
        startedAt: 462,
      },
    ])

    await restoring
    await expect(cancelling).resolves.toEqual({ ok: true })
    expect(cancelRun).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-delayed" }),
      "user_cancelled"
    )
  })

  it("rejects a stale cancellation generation before it can touch the current run", async () => {
    vi.useFakeTimers()
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          kind: "run",
          sessionId: "session-cancel-generation",
          runId: "run-cancel-generation",
          generation: "generation-cancel-current-g2",
          attemptToken: "attempt-cancel-generation",
          submissionState: "acknowledged",
          occurrenceIds: ["occ-cancel-generation"],
          jobIdToItemId: {},
          startedAt: 463,
        },
      ]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      }),
      cancelRun,
    } as any)

    try {
      await runtime.restore()
      const response = await (runtime as any).cancel(
        "session-cancel-generation",
        "user_cancelled",
        undefined,
        "generation-cancel-stale-g1",
      )

      expect(response).toMatchObject({
        ok: false,
        error: expect.stringMatching(/generation|stale|superseded/i),
      })
      expect(cancelRun).not.toHaveBeenCalled()
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("includes occurrence results when an authoritative restore fails", async () => {
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          sessionId: "session-failed-results",
          runId: "run-failed-results",
          submissionState: "acknowledged",
          occurrenceIds: ["occ-failed-results"],
          jobIdToItemId: { "93": "occ-failed-results" },
          startedAt: 463,
        },
      ]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "failed",
        jobs: [
          {
            jobId: 93,
            status: "failed",
            sourceItemId: "occ-failed-results",
            error: "Transcription failed.",
          },
        ],
        errorMessage: "One item failed.",
      }),
    } as any)

    await runtime.restore()

    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/failed",
      expect.objectContaining({
        results: [
          expect.objectContaining({
            id: "occ-failed-results",
            status: "error",
            error: "Transcription failed.",
          }),
        ],
      })
    )
  })

  it("retains an accepted cancellation until polling confirms terminal cancellation", async () => {
    vi.useFakeTimers()
    try {
      const saveRunSession = vi.fn()
      const emit = vi.fn()
      const reattachRun = vi
        .fn()
        .mockResolvedValueOnce({
          lifecycle: "processing",
          jobs: [],
          errorMessage: null,
        })
        .mockResolvedValueOnce({
          lifecycle: "cancelled",
          jobs: [
            {
              jobId: 94,
              status: "cancelled",
              sourceItemId: "occ-cancel-confirmed",
              error: "Cancelled by user.",
            },
          ],
          errorMessage: "Cancelled by user.",
        })
      const runtime = createQuickIngestSessionRuntime({
        run: vi.fn(),
        emit,
        loadRunSessions: vi.fn().mockResolvedValue([
          {
            version: 1,
            sessionId: "session-cancel-confirmed",
            runId: "run-cancel-confirmed",
            submissionState: "acknowledged",
            occurrenceIds: ["occ-cancel-confirmed"],
            jobIdToItemId: { "94": "occ-cancel-confirmed" },
            startedAt: 464,
          },
        ]),
        saveRunSession,
        reattachRun,
        cancelRun: vi.fn().mockResolvedValue({ ok: true }),
      } as any)

      await runtime.restore()
      await runtime.cancel("session-cancel-confirmed")

      expect(saveRunSession).not.toHaveBeenCalledWith(
        null,
        "session-cancel-confirmed",
        "run-cancel-confirmed"
      )
      expect(emit).not.toHaveBeenCalledWith(
        "tldw:quick-ingest/cancelled",
        expect.anything()
      )

      await vi.advanceTimersByTimeAsync(1_500)

      expect(saveRunSession).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "terminal",
          sessionId: "session-cancel-confirmed",
          runId: "run-cancel-confirmed",
        }),
        "session-cancel-confirmed",
        "run-cancel-confirmed",
        expect.any(String)
      )
      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/cancelled",
        expect.objectContaining({
          sessionId: "session-cancel-confirmed",
          results: [
            expect.objectContaining({
              id: "occ-cancel-confirmed",
              status: "error",
              error: "Cancelled by user.",
            }),
          ],
        })
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("keeps polling a restored run until its occurrence reaches a terminal state", async () => {
    vi.useFakeTimers()
    try {
      const saveRunSession = vi.fn()
      const emit = vi.fn()
      const reattachRun = vi
        .fn()
        .mockResolvedValueOnce({
          lifecycle: "processing",
          jobs: [
            {
              jobId: 82,
              status: "running",
              sourceItemId: "occ-polled",
            },
          ],
          errorMessage: null,
        })
        .mockResolvedValueOnce({
          lifecycle: "completed",
          jobs: [
            {
              jobId: 82,
              status: "completed",
              sourceItemId: "occ-polled",
              result: { outcome: "completed" },
            },
          ],
          errorMessage: null,
        })
      const runtime = createQuickIngestSessionRuntime({
        run: vi.fn(),
        emit,
        loadRunSessions: vi.fn().mockResolvedValue([
          {
            version: 1,
            sessionId: "session-polled",
            runId: "run-polled",
            occurrenceIds: ["occ-polled"],
            jobIdToItemId: { "82": "occ-polled" },
            startedAt: 457,
          },
        ]),
        saveRunSession,
        reattachRun,
      } as any)

      await runtime.restore()
      expect(reattachRun).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(1_500)

      expect(reattachRun).toHaveBeenCalledTimes(2)
      expect(saveRunSession).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "terminal",
          sessionId: "session-polled",
          runId: "run-polled",
        }),
        "session-polled",
        "run-polled",
        expect.any(String)
      )
      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/completed",
        expect.objectContaining({
          sessionId: "session-polled",
          runId: "run-polled",
        })
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("replays a bounded terminal tombstone after worker recreation until explicit acknowledgement", async () => {
    let storedRecords: any[] = [
      {
        version: 1,
        kind: "run",
        sessionId: "session-terminal-replay",
        runId: "run-terminal-replay",
        generation: "generation-terminal-replay",
        requestFingerprint: "request-terminal-replay",
        occurrenceIds: ["occ-terminal-replay"],
        jobIdToItemId: {},
        startedAt: Date.now(),
      },
    ]
    const saveRunSession = vi.fn(
      async (
        record: any,
        sessionId?: string,
        expectedRunId?: string,
        expectedGeneration?: string
      ) => {
        const id = String(record?.sessionId || sessionId || "")
        storedRecords = storedRecords.filter((stored) => {
          if (stored.sessionId !== id) return true
          if (expectedRunId && stored.runId !== expectedRunId) return true
          if (expectedGeneration && stored.generation !== expectedGeneration) {
            return true
          }
          return false
        })
        if (record) storedRecords.push(record)
      }
    )
    const firstEmit = vi.fn()
    const firstWorker = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: firstEmit,
      loadRunSessions: vi.fn(async () => storedRecords),
      saveRunSession,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "completed",
        jobs: [
          {
            jobId: null,
            status: "completed",
            sourceItemId: "occ-terminal-replay",
            result: { media_id: 42, outcome: "included_existing" },
          },
        ],
        errorMessage: null,
      }),
    } as any)

    await firstWorker.restore()

    expect(storedRecords).toEqual([
      expect.objectContaining({
        version: 1,
        kind: "terminal",
        sessionId: "session-terminal-replay",
        runId: "run-terminal-replay",
        generation: "generation-terminal-replay",
        expiresAt: expect.any(Number),
        event: expect.objectContaining({
          type: "tldw:quick-ingest/completed",
          payload: expect.objectContaining({
            results: [expect.objectContaining({ id: "occ-terminal-replay" })],
          }),
        }),
      }),
    ])

    const recreatedEmit = vi.fn()
    const recreatedReattach = vi.fn()
    const recreatedWorker = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: recreatedEmit,
      loadRunSessions: vi.fn(async () => storedRecords),
      saveRunSession,
      reattachRun: recreatedReattach,
    } as any)
    await recreatedWorker.restore()

    expect(recreatedReattach).not.toHaveBeenCalled()
    const replay = await recreatedWorker.replay("session-terminal-replay")
    expect(replay).toMatchObject({
      ok: true,
      active: false,
      event: expect.objectContaining({
        type: "tldw:quick-ingest/completed",
      }),
      replayAck: {
        runId: "run-terminal-replay",
        generation: "generation-terminal-replay",
      },
    })
    expect(storedRecords).toHaveLength(1)

    expect(typeof (recreatedWorker as any).acknowledgeReplay).toBe("function")
    if (typeof (recreatedWorker as any).acknowledgeReplay !== "function") return
    await (recreatedWorker as any).acknowledgeReplay(
      "session-terminal-replay",
      "run-terminal-replay",
      "generation-terminal-replay"
    )
    expect(storedRecords).toEqual([])
  })

  it("retains terminal replay when durable acknowledgement loses its generation CAS", async () => {
    const terminal = {
      version: 1,
      kind: "terminal",
      sessionId: "session-replay-ack-cas",
      runId: "run-replay-ack-cas",
      generation: "generation-replay-ack-cas",
      attemptToken: "attempt-replay-ack-cas",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "session-replay-ack-cas",
          runId: "run-replay-ack-cas",
          results: [],
        },
      },
    }
    const saveRunSession = vi.fn().mockResolvedValue(false)
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([terminal]),
      saveRunSession,
    } as any)

    await runtime.restore()
    const acknowledgement = await runtime.acknowledgeReplay(
      terminal.sessionId,
      terminal.runId,
      terminal.generation
    )

    expect(acknowledgement).toEqual({
      ok: false,
      error: expect.stringMatching(/superseded|retained/i),
    })
    expect(await runtime.replay(terminal.sessionId)).toMatchObject({
      ok: true,
      event: terminal.event,
    })
  })

  it("expires terminal tombstones with generation-safe cleanup", async () => {
    const saveRunSession = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          kind: "terminal",
          sessionId: "session-terminal-expired",
          runId: "run-terminal-expired",
          generation: "generation-terminal-expired",
          requestFingerprint: "request-terminal-expired",
          expiresAt: Date.now() - 1,
          event: {
            type: "tldw:quick-ingest/completed",
            payload: {
              sessionId: "session-terminal-expired",
              runId: "run-terminal-expired",
              results: [],
            },
          },
        },
      ]),
      saveRunSession,
      reattachRun: vi.fn(),
    } as any)

    await runtime.restore()

    expect(saveRunSession).toHaveBeenCalledWith(
      null,
      "session-terminal-expired",
      "run-terminal-expired",
      "generation-terminal-expired"
    )
    await expect(runtime.replay("session-terminal-expired")).resolves.toMatchObject({
      ok: false,
      error: expect.stringMatching(/not found|expired/i),
    })
  })

  it("expires durable review tombstones with generation-safe cleanup", async () => {
    const review = {
      version: 1,
      kind: "review",
      sessionId: "session-review-expired",
      generation: "generation-review-expired",
      attemptToken: "attempt-review-expired",
      expiresAt: Date.now() - 1,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "session-review-expired",
          reviewRequired: [
            {
              occurrenceId: "occ-review-expired",
              reason: "duplicate_action_required",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "overwrite"],
            },
          ],
        },
      },
    }
    const saveRunSession = vi.fn().mockResolvedValue(true)
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([review]),
      saveRunSession,
    } as any)

    await runtime.restore()

    expect(saveRunSession).toHaveBeenCalledWith(
      null,
      review.sessionId,
      undefined,
      review.generation
    )
    await expect(runtime.replay(review.sessionId)).resolves.toMatchObject({
      ok: false,
      error: expect.stringMatching(/not found|expired/i),
    })
  })

  it("delegates restored run cancellation instead of cancelling legacy job ids", async () => {
    const record = {
      version: 1 as const,
      sessionId: "session-cancel",
      runId: "run-cancel",
      occurrenceIds: ["occ-cancel"],
      jobIdToItemId: { "91": "occ-cancel" },
      startedAt: 789,
    }
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([record]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [
          {
            jobId: 91,
            status: "running",
            sourceItemId: "occ-cancel",
          },
        ],
        errorMessage: null,
      }),
      cancelRun,
    } as any)

    await runtime.restore()
    const response = await runtime.cancel("session-cancel", "user_cancelled")

    expect(response).toEqual({ ok: true })
    expect(cancelRun).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "extension-runtime",
        runId: "run-cancel",
        submissionOccurrenceIds: ["occ-cancel"],
      }),
      "user_cancelled"
    )
  })

  it("makes cancellation observable while the server run cancel is pending", async () => {
    let releaseRun!: () => void
    let resolveCancel!: (value: { ok: boolean; error?: string }) => void
    let runtimeContext: any = null
    const runGate = new Promise<void>((resolve) => {
      releaseRun = resolve
    })
    const cancelGate = new Promise<{ ok: boolean; error?: string }>((resolve) => {
      resolveCancel = resolve
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        runtimeContext = context
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-cancel-pending",
          submissionOccurrenceIds: ["occ-cancel-pending"],
        })
        await runGate
        return { results: [] }
      }),
      emit: vi.fn(),
      saveRunSession: vi.fn(),
      cancelRun: vi.fn(() => cancelGate),
      createSessionId: () => "session-cancel-pending",
    } as any)

    runtime.start({ pendingRunRequest: { contractVersion: 2 } })
    await vi.waitFor(() => {
      expect(runtimeContext).toBeTruthy()
    })

    const cancelling = runtime.cancel(
      "session-cancel-pending",
      "user_cancelled"
    )

    expect(runtimeContext.isCancelled()).toBe(true)
    resolveCancel({ ok: false, error: "cancel unavailable" })
    await expect(cancelling).resolves.toEqual({
      ok: false,
      error: "cancel unavailable",
    })
    expect(runtimeContext.isCancelled()).toBe(false)

    releaseRun()
  })

  it("restores through one shared poll delegate without rerunning playlist submission", async () => {
    const run = vi.fn()
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run,
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          sessionId: "session-thin",
          runId: "run-thin",
          occurrenceIds: ["occ-thin"],
          jobIdToItemId: {},
          startedAt: 999,
        },
      ]),
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    await runtime.restore()

    expect(run).not.toHaveBeenCalled()
    expect(reattachRun).toHaveBeenCalledTimes(1)
  })

  it("ignores malformed or oversized compact records during restoration", async () => {
    const validRecord = {
      version: 1 as const,
      sessionId: "session-valid",
      runId: "run-valid",
      occurrenceIds: ["occ-valid"],
      jobIdToItemId: {},
      startedAt: 1000,
    }
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        { ...validRecord, version: 99 },
        { ...validRecord, sessionId: "s".repeat(256) },
        { ...validRecord, runId: "r".repeat(256) },
        { ...validRecord, occurrenceIds: Array.from({ length: 501 }, (_, index) => `occ-${index}`) },
        {
          ...validRecord,
          occurrenceIds: ["occ-valid"],
          jobIdToItemId: Object.fromEntries(
            Array.from({ length: 501 }, (_, index) => [
              String(index + 1),
              "occ-valid",
            ])
          ),
        },
        validRecord,
      ]),
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    await runtime.restore()

    expect(reattachRun).toHaveBeenCalledTimes(1)
    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-valid" }),
      { transportPreference: "poll" }
    )
  })

  it("accepts bounded terminal tombstones and rejects oversized terminal results", () => {
    const bounded = {
      version: 1,
      kind: "terminal",
      sessionId: "session-terminal-bounded",
      runId: "run-terminal-bounded",
      generation: "generation-terminal-bounded",
      attemptToken: "attempt-terminal-bounded",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "session-terminal-bounded",
          runId: "run-terminal-bounded",
          results: [
            {
              id: "occ-terminal-bounded",
              status: "ok",
              type: "video",
              data: { media_id: 42, outcome: "completed", title: "Bounded" },
            },
          ],
        },
      },
    }

    expect(parseQuickIngestCompactRunSession(bounded)).toEqual(bounded)
    expect(
      parseQuickIngestCompactRunSession({
        ...bounded,
        event: {
          ...bounded.event,
          payload: {
            ...bounded.event.payload,
            results: Array.from({ length: 501 }, (_, index) => ({
              id: `occ-${index}`,
              status: "ok",
              type: "video",
            })),
          },
        },
      })
    ).toBeNull()
  })

  it("fails closed and retains recovery when the first post-create storage write fails", async () => {
    const emit = vi.fn()
    const saveRunSession = vi
      .fn()
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error("storage unavailable"))
      .mockResolvedValue(undefined)
    const cancelRun = vi
      .fn()
      .mockResolvedValue({ ok: false, error: "Cancellation unconfirmed." })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: null,
          status: "processing",
          sourceItemId: "occ-storage-failed",
        },
      ],
      errorMessage: "Run status is temporarily unavailable. Quick ingest will retry.",
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-storage-failed",
          submissionState: "run_created",
          submissionOccurrenceIds: ["occ-storage-failed"],
        })
        return { results: [] }
      }),
      emit,
      saveRunSession,
      cancelRun,
      reattachRun,
      createSessionId: () => "session-storage-failed",
    } as any)

    runtime.start({
      pendingRunRequest: {
        contractVersion: 2,
        inputs: [{ occurrenceId: "occ-storage-failed" }],
      },
    })

    await vi.waitFor(() => {
      expect(cancelRun).toHaveBeenCalledWith(
        expect.objectContaining({ runId: "run-storage-failed" }),
        "tracking_persistence_failed"
      )
    })
    expect(saveRunSession).toHaveBeenCalledTimes(3)
    expect(saveRunSession.mock.calls[2]?.[0]).toEqual(
      expect.objectContaining({
        kind: "run",
        sessionId: "session-storage-failed",
        runId: "run-storage-failed",
      })
    )
    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-storage-failed" }),
      { transportPreference: "poll" }
    )
    expect(emit).toHaveBeenCalledWith(
      "tldw:quick-ingest/interrupted",
      expect.objectContaining({
        sessionId: "session-storage-failed",
        runId: "run-storage-failed",
        recoverable: true,
        error: expect.stringMatching(/storage|persist|recovery/i),
      })
    )
    expect(emit).not.toHaveBeenCalledWith(
      "tldw:quick-ingest/failed",
      expect.objectContaining({ sessionId: "session-storage-failed" })
    )
    expect(runtime.hasSession("session-storage-failed")).toBe(true)
  })

  it("retains post-create recovery when its first reconciliation poll also fails", async () => {
    const emit = vi.fn()
    const reattachRun = vi.fn().mockRejectedValue(new Error("poll unavailable"))
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-storage-poll-failed",
          submissionState: "run_created",
          submissionOccurrenceIds: ["occ-storage-poll-failed"],
        })
        return { results: [] }
      }),
      emit,
      saveRunSession: vi
        .fn()
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error("storage unavailable"))
        .mockResolvedValue(undefined),
      cancelRun: vi
        .fn()
        .mockResolvedValue({ ok: false, error: "Cancellation unconfirmed." }),
      reattachRun,
      createSessionId: () => "session-storage-poll-failed",
    } as any)

    runtime.start({
      pendingRunRequest: {
        inputs: [{ occurrenceId: "occ-storage-poll-failed" }],
      },
    })

    await vi.waitFor(() => {
      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/interrupted",
        expect.objectContaining({
          sessionId: "session-storage-poll-failed",
          recoverable: true,
        })
      )
    })
    await vi.waitFor(() => expect(reattachRun).toHaveBeenCalledTimes(1))
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(
      emit.mock.calls.some(
        ([type, payload]) =>
          type === "tldw:quick-ingest/failed" &&
          payload?.sessionId === "session-storage-poll-failed"
      )
    ).toBe(false)
    expect(runtime.hasSession("session-storage-poll-failed")).toBe(true)
  })

  it("surfaces a startup storage read failure and retries it on replay", async () => {
    const record = {
      version: 1,
      sessionId: "session-storage-read-retry",
      runId: "run-storage-read-retry",
      occurrenceIds: ["occ-storage-read-retry"],
      jobIdToItemId: {},
      startedAt: Date.now(),
    }
    const loadRunSessions = vi
      .fn()
      .mockRejectedValueOnce(new Error("storage read unavailable"))
      .mockResolvedValue([record])
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions,
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    await expect(runtime.restore()).rejects.toThrow("storage read unavailable")
    await expect(runtime.replay("session-storage-read-retry")).resolves.toMatchObject({
      ok: true,
      active: true,
    })
    expect(loadRunSessions).toHaveBeenCalledTimes(2)
    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-storage-read-retry" }),
      { transportPreference: "poll" }
    )
  })

  it("isolates restored run poll failures and retries without blocking later records", async () => {
    vi.useFakeTimers()
    const makeRecord = (suffix: string) => ({
      version: 1,
      kind: "run",
      sessionId: `session-restore-isolation-${suffix}`,
      runId: `run-restore-isolation-${suffix}`,
      generation: `generation-restore-isolation-${suffix}`,
      attemptToken: `attempt-restore-isolation-${suffix}`,
      occurrenceIds: [`occ-restore-isolation-${suffix}`],
      jobIdToItemId: {},
      startedAt: Date.now(),
    })
    const first = makeRecord("first")
    const second = makeRecord("second")
    const attempts = new Map<string, number>()
    const reattachRun = vi.fn(async (tracking: { runId?: string }) => {
      const runId = String(tracking.runId || "")
      const attempt = (attempts.get(runId) || 0) + 1
      attempts.set(runId, attempt)
      if (runId === first.runId && attempt === 1) {
        throw new Error("first restored poll failed")
      }
      return {
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      }
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([first, second]),
      saveRunSession: vi.fn(),
      reattachRun,
    } as any)

    try {
      await expect(runtime.restore()).resolves.toBeUndefined()
      expect(attempts.get(first.runId)).toBe(1)
      expect(attempts.get(second.runId)).toBe(1)

      await vi.advanceTimersByTimeAsync(1_500)

      expect(attempts.get(first.runId)).toBe(2)
      expect(attempts.get(second.runId)).toBeGreaterThanOrEqual(1)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("isolates expired replay cleanup failures and retries them without blocking later runs", async () => {
    vi.useFakeTimers()
    const now = Date.now()
    const expiredTerminal = {
      version: 1,
      kind: "terminal",
      sessionId: "session-expired-cleanup-terminal",
      runId: "run-expired-cleanup-terminal",
      generation: "generation-expired-cleanup-terminal",
      attemptToken: "attempt-expired-cleanup-terminal",
      expiresAt: now - 1,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "session-expired-cleanup-terminal",
          runId: "run-expired-cleanup-terminal",
          results: [],
        },
      },
    }
    const laterActive = {
      version: 1,
      kind: "run",
      sessionId: "session-after-expired-cleanup",
      runId: "run-after-expired-cleanup",
      generation: "generation-after-expired-cleanup",
      attemptToken: "attempt-after-expired-cleanup",
      occurrenceIds: ["occ-after-expired-cleanup"],
      jobIdToItemId: {},
      startedAt: now,
    }
    const expiredReview = {
      version: 1,
      kind: "review",
      sessionId: "session-expired-cleanup-review",
      generation: "generation-expired-cleanup-review",
      attemptToken: "attempt-expired-cleanup-review",
      expiresAt: now - 1,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "session-expired-cleanup-review",
          reviewRequired: [
            {
              occurrenceId: "occ-expired-cleanup-review",
              reason: "duplicate_action_required",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "overwrite"],
            },
          ],
        },
      },
    }
    const cleanupAttempts = new Map<string, number>()
    const saveRunSession = vi.fn(async (record: any, sessionId?: string) => {
      if (record !== null || !sessionId?.includes("expired-cleanup")) return true
      const attempt = (cleanupAttempts.get(sessionId) || 0) + 1
      cleanupAttempts.set(sessionId, attempt)
      if (attempt === 1) throw new Error(`cleanup failed for ${sessionId}`)
      return true
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi
        .fn()
        .mockResolvedValue([expiredTerminal, laterActive, expiredReview]),
      saveRunSession,
      reattachRun,
    } as any)

    try {
      await expect(runtime.restore()).resolves.toBeUndefined()
      expect(cleanupAttempts.get(expiredTerminal.sessionId)).toBe(1)
      expect(cleanupAttempts.get(expiredReview.sessionId)).toBe(1)
      expect(reattachRun).toHaveBeenCalledWith(
        expect.objectContaining({ runId: laterActive.runId }),
        { transportPreference: "poll" }
      )

      await vi.advanceTimersByTimeAsync(1_500)

      expect(cleanupAttempts.get(expiredTerminal.sessionId)).toBe(2)
      expect(cleanupAttempts.get(expiredReview.sessionId)).toBe(2)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("clears a scheduled replay cleanup after a later restore succeeds", async () => {
    vi.useFakeTimers()
    const expiredReview = {
      version: 1,
      kind: "review",
      sessionId: "session-expired-cleanup-recovered",
      generation: "generation-expired-cleanup-recovered",
      attemptToken: "attempt-expired-cleanup-recovered",
      expiresAt: Date.now() - 1,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "session-expired-cleanup-recovered",
          reviewRequired: [
            {
              occurrenceId: "occ-expired-cleanup-recovered",
              reason: "duplicate_action_required",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "overwrite"],
            },
          ],
        },
      },
    }
    let cleanupAttempts = 0
    const saveRunSession = vi.fn(async () => {
      cleanupAttempts += 1
      if (cleanupAttempts === 1) throw new Error("cleanup failed once")
      return true
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([expiredReview]),
      saveRunSession,
    } as any)

    try {
      await expect(runtime.restore()).resolves.toBeUndefined()
      expect(cleanupAttempts).toBe(1)
      expect(vi.getTimerCount()).toBe(1)

      await expect(runtime.restore()).resolves.toBeUndefined()
      expect(cleanupAttempts).toBe(2)

      await vi.advanceTimersByTimeAsync(1_500)

      expect(cleanupAttempts).toBe(2)
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("clears a delayed active poll after one terminal tombstone wins authority", async () => {
    vi.useFakeTimers()
    const activeRecord = {
      version: 1,
      kind: "run",
      sessionId: "session-terminal-once",
      runId: "run-terminal-once",
      generation: "generation-terminal-once",
      attemptToken: "attempt-terminal-once",
      occurrenceIds: ["occ-terminal-once"],
      jobIdToItemId: {},
      startedAt: Date.now(),
    }
    let storedRecord: any = activeRecord
    const saveRunSession = vi.fn(async (record: any) => {
      if (record?.kind === "terminal" && storedRecord?.kind === "terminal") {
        return false
      }
      storedRecord = record
      return true
    })
    const reattachRun = vi
      .fn()
      .mockResolvedValueOnce({
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      })
      .mockResolvedValue({
        lifecycle: "completed",
        jobs: [
          {
            jobId: 91,
            status: "completed",
            sourceItemId: "occ-terminal-once",
            result: { outcome: "processed" },
            error: null,
          },
        ],
        errorMessage: null,
      })
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn(async () => (storedRecord ? [storedRecord] : [])),
      saveRunSession,
      reattachRun,
    } as any)

    try {
      await runtime.restore()
      await runtime.replay(activeRecord.sessionId)
      const winningTombstone = JSON.stringify(storedRecord)
      expect(
        emit.mock.calls.filter(([type]) => type === "tldw:quick-ingest/completed")
      ).toHaveLength(1)

      await vi.advanceTimersByTimeAsync(1_500)

      expect(reattachRun).toHaveBeenCalledTimes(2)
      expect(JSON.stringify(storedRecord)).toBe(winningTombstone)
      expect(
        emit.mock.calls.filter(([type]) => type === "tldw:quick-ingest/completed")
      ).toHaveLength(1)
      expect(emit).not.toHaveBeenCalledWith(
        "tldw:quick-ingest/interrupted",
        expect.anything()
      )
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("stops without rescheduling when terminal persistence is superseded", async () => {
    vi.useFakeTimers()
    const activeRecord = {
      version: 1,
      kind: "run",
      sessionId: "session-terminal-superseded",
      runId: "run-terminal-superseded",
      generation: "generation-terminal-superseded",
      attemptToken: "attempt-terminal-superseded",
      occurrenceIds: ["occ-terminal-superseded"],
      jobIdToItemId: {},
      startedAt: Date.now(),
    }
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "completed",
      jobs: [
        {
          jobId: 92,
          status: "completed",
          sourceItemId: "occ-terminal-superseded",
          result: { outcome: "processed" },
          error: null,
        },
      ],
      errorMessage: null,
    })
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([activeRecord]),
      saveRunSession: vi.fn(async (record: any) =>
        record?.kind === "terminal" ? false : true
      ),
      reattachRun,
    } as any)

    try {
      await runtime.restore()
      await vi.advanceTimersByTimeAsync(4_500)

      expect(reattachRun).toHaveBeenCalledTimes(1)
      expect(emit).not.toHaveBeenCalledWith(
        "tldw:quick-ingest/interrupted",
        expect.anything()
      )
      expect(emit).not.toHaveBeenCalledWith(
        "tldw:quick-ingest/completed",
        expect.anything()
      )
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("retries failed active-run persistence on a later reconciliation poll", async () => {
    vi.useFakeTimers()
    let runWriteAttempts = 0
    const saveRunSession = vi.fn(async (record: any) => {
      if (record?.kind === "run") {
        runWriteAttempts += 1
        if (runWriteAttempts <= 2) throw new Error("storage unavailable")
      }
      return true
    })
    const reattachRun = vi
      .fn()
      .mockResolvedValueOnce({
        lifecycle: "processing",
        jobs: [{ jobId: null, status: "processing", sourceItemId: "occ-retry-persist" }],
        errorMessage: null,
      })
      .mockResolvedValueOnce({
        lifecycle: "completed",
        jobs: [{ jobId: null, status: "completed", sourceItemId: "occ-retry-persist" }],
        errorMessage: null,
      })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (_payload: any, context: any) => {
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-retry-persist",
          submissionState: "run_created",
          submissionOccurrenceIds: ["occ-retry-persist"],
        })
        return { results: [] }
      }),
      emit: vi.fn(),
      saveRunSession,
      cancelRun: vi.fn().mockResolvedValue({
        ok: false,
        error: "Cancellation unconfirmed.",
      }),
      reattachRun,
      createSessionId: () => "session-retry-persist",
    } as any)

    await runtime.start({
      pendingRunRequest: { inputs: [{ occurrenceId: "occ-retry-persist" }] },
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(runWriteAttempts).toBe(2)
    expect(reattachRun).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1_500)

    expect(runWriteAttempts).toBeGreaterThanOrEqual(3)
    expect(saveRunSession).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "run",
        sessionId: "session-retry-persist",
        runId: "run-retry-persist",
      }),
      "session-retry-persist",
      undefined,
      expect.any(String)
    )
    vi.useRealTimers()
  })

  it("propagates structured review-required recovery instead of emitting a generic failure", async () => {
    const reviewRequired = [
      {
        occurrenceId: "occ-review-runtime",
        reason: "duplicate_action_required",
        evidence: {
          kind: "library",
          existingMediaId: 42,
          duplicateOfOccurrenceId: null,
        },
        allowedActions: ["skip", "overwrite"],
      },
    ]
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn().mockResolvedValue({ results: [], reviewRequired }),
      emit,
      saveRunSession: vi.fn().mockResolvedValue(true),
      createSessionId: () => "session-review-runtime",
    } as any)

    await runtime.start({ entries: [], files: [] })

    await vi.waitFor(() => {
      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/review-required",
        expect.objectContaining({
          sessionId: "session-review-runtime",
          reviewRequired,
        })
      )
    })
    expect(emit).not.toHaveBeenCalledWith(
      "tldw:quick-ingest/failed",
      expect.anything()
    )
  })

  it("replays durable review recovery after the start response is lost and the worker restarts", async () => {
    const reviewRequired = [
      {
        occurrenceId: "occ-review-recreated",
        reason: "duplicate_action_required",
        evidence: {
          kind: "library",
          existingMediaId: 42,
          duplicateOfOccurrenceId: null,
        },
        allowedActions: ["skip", "overwrite"],
      },
    ]
    let storedRecords: any[] = []
    const saveRunSession = vi.fn(
      async (
        record: any,
        sessionId?: string,
        expectedRunId?: string,
        expectedGeneration?: string
      ) => {
        const id = String(record?.sessionId || sessionId || "")
        const matching = storedRecords.filter((stored) => stored.sessionId === id)
        if (
          expectedGeneration &&
          (matching.length === 0 ||
            matching.some((stored) => stored.generation !== expectedGeneration))
        ) {
          return false
        }
        if (
          expectedRunId &&
          matching.some(
            (stored) => stored.kind !== "start" && stored.runId !== expectedRunId
          )
        ) {
          return false
        }
        storedRecords = storedRecords.filter((stored) => stored.sessionId !== id)
        if (record) storedRecords.push(record)
        return true
      }
    )
    const firstRuntime = createQuickIngestSessionRuntime({
      run: vi.fn().mockResolvedValue({ results: [], reviewRequired }),
      emit: vi.fn(),
      saveRunSession,
    } as any)

    void firstRuntime.start(
      { entries: [], files: [] },
      {
        sessionId: "session-review-recreated",
        attemptToken: "attempt-review-recreated",
      }
    )
    await vi.waitFor(() => {
      expect(storedRecords).toEqual([
        expect.objectContaining({
          kind: "review",
          sessionId: "session-review-recreated",
          generation: expect.any(String),
          attemptToken: "attempt-review-recreated",
          expiresAt: expect.any(Number),
          event: {
            type: "tldw:quick-ingest/review-required",
            payload: {
              sessionId: "session-review-recreated",
              reviewRequired,
            },
          },
        }),
      ])
    })
    expect(saveRunSession).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "review" }),
      "session-review-recreated",
      undefined,
      expect.any(String)
    )

    const recreatedRuntime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue(storedRecords),
      saveRunSession,
    } as any)
    await recreatedRuntime.restore()

    expect(await recreatedRuntime.replay("session-review-recreated")).toEqual({
      ok: true,
      active: false,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "session-review-recreated",
          reviewRequired,
        },
      },
    })
  })

  it("rejects a terminal tombstone whose aggregate serialized event exceeds 512 KiB", () => {
    const oversizedTerminal = {
      version: 1,
      kind: "terminal",
      sessionId: "session-terminal-byte-limit",
      runId: "run-terminal-byte-limit",
      generation: "generation-terminal-byte-limit",
      requestFingerprint: "legacy-request-terminal-byte-limit",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-terminal-byte-limit",
          runId: "run-terminal-byte-limit",
          results: Array.from({ length: 300 }, (_, index) => ({
            id: `occ-byte-${index}`,
            status: "error",
            type: "video",
            error: "x".repeat(2_000),
          })),
        },
      },
    }

    expect(JSON.stringify(oversizedTerminal.event).length).toBeGreaterThan(512 * 1_024)
    expect(parseQuickIngestCompactRunSession(oversizedTerminal)).toBeNull()
  })

  it("applies the 512 KiB terminal limit to the complete persisted record", () => {
    const sessionId = "s".repeat(255)
    const runId = "r".repeat(255)
    const terminal = {
      version: 1,
      kind: "terminal",
      sessionId,
      runId,
      generation: "g".repeat(255),
      attemptToken: "a".repeat(255),
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId,
          runId,
          results: Array.from({ length: 254 }, (_, index) => ({
            id: `occ-${index}`,
            status: "error",
            type: "video",
            error: "x".repeat(2_000),
          })),
        },
      },
    }
    const encoder = new TextEncoder()

    expect(encoder.encode(JSON.stringify(terminal.event)).byteLength).toBeLessThan(
      512 * 1_024
    )
    expect(encoder.encode(JSON.stringify(terminal)).byteLength).toBeGreaterThan(
      512 * 1_024
    )
    expect(parseQuickIngestCompactRunSession(terminal)).toBeNull()
  })

  it("compacts oversized 500-item terminal results without losing occurrence outcomes", async () => {
    vi.useFakeTimers()
    const occurrenceIds = Array.from(
      { length: 500 },
      (_, index) => `occ-oversized-terminal-${index}`
    )
    const activeRecord = {
      version: 1,
      kind: "run",
      sessionId: "session-oversized-terminal",
      runId: "run-oversized-terminal",
      generation: "generation-oversized-terminal",
      attemptToken: "attempt-oversized-terminal",
      occurrenceIds,
      jobIdToItemId: {},
      startedAt: Date.now(),
    }
    const terminalWrites: any[] = []
    const saveRunSession = vi.fn(async (record: any) => {
      if (record?.kind === "terminal") terminalWrites.push(record)
      return true
    })
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([activeRecord]),
      saveRunSession,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "failed",
        jobs: occurrenceIds.map((occurrenceId, index) => ({
          jobId: null,
          status: "failed",
          sourceItemId: occurrenceId,
          result: {
            outcome: "failed",
            title: `Oversized result ${index} ${"t".repeat(2_000)}`,
          },
          error: `Oversized failure ${index} ${"e".repeat(2_000)}`,
        })),
        errorMessage: `Oversized terminal failure ${"m".repeat(2_000)}`,
      }),
    } as any)

    try {
      await runtime.restore()

      expect(terminalWrites).toHaveLength(1)
      const terminal = terminalWrites[0]
      expect(
        new TextEncoder().encode(JSON.stringify(terminal)).byteLength
      ).toBeLessThanOrEqual(512 * 1_024)
      expect(terminal.event.payload.results).toHaveLength(500)
      expect(terminal.event.payload.results.map((item: any) => item.id)).toEqual(
        occurrenceIds
      )
      expect(terminal.event.payload.results).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: occurrenceIds[0],
            status: "error",
            data: { outcome: "failed" },
          }),
          expect.objectContaining({
            id: occurrenceIds[499],
            status: "error",
            data: { outcome: "failed" },
          }),
        ])
      )
      expect(
        emit.mock.calls.filter(([type]) => type === "tldw:quick-ingest/failed")
      ).toHaveLength(1)
      expect(emit).not.toHaveBeenCalledWith(
        "tldw:quick-ingest/interrupted",
        expect.anything()
      )
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("restores only the newest generation when duplicate same-session records are encountered", async () => {
    const oldRecord = {
      version: 1,
      kind: "run",
      sessionId: "session-duplicate-generation",
      runId: "run-shared-generation",
      generation: "generation-old",
      requestFingerprint: "legacy-old",
      occurrenceIds: ["occ-duplicate-generation"],
      jobIdToItemId: {},
      startedAt: 1_000,
    }
    const newRecord = {
      ...oldRecord,
      generation: "generation-new",
      requestFingerprint: "legacy-new",
      startedAt: 2_000,
    }
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "completed",
      jobs: [
        {
          jobId: null,
          status: "completed",
          sourceItemId: "occ-duplicate-generation",
        },
      ],
      errorMessage: null,
    })
    const saveRunSession = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([oldRecord, newRecord]),
      saveRunSession,
      reattachRun,
    } as any)

    await runtime.restore()

    expect(reattachRun).toHaveBeenCalledTimes(1)
    expect(saveRunSession).toHaveBeenCalledWith(
      expect.objectContaining({ generation: "generation-new" }),
      "session-duplicate-generation",
      "run-shared-generation",
      "generation-new"
    )
  })

  it("replaces a terminal run with a replay tombstone for the same run generation", async () => {
    const saveRunSession = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          sessionId: "session-terminal",
          runId: "run-terminal-old",
          occurrenceIds: ["occ-terminal"],
          jobIdToItemId: {},
          startedAt: 1001,
        },
      ]),
      saveRunSession,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "completed",
        jobs: [
          {
            jobId: null,
            status: "completed",
            sourceItemId: "occ-terminal",
            result: { outcome: "included_existing" },
          },
        ],
        errorMessage: null,
      }),
    } as any)

    await runtime.restore()

    expect(saveRunSession).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "terminal",
        sessionId: "session-terminal",
        runId: "run-terminal-old",
        generation: expect.any(String),
        expiresAt: expect.any(Number),
      }),
      "session-terminal",
      "run-terminal-old",
      expect.any(String)
    )
  })

  it("preserves canonical lifecycle and progress evidence in restored progress events", async () => {
    vi.useFakeTimers()
    const record = {
      version: 1,
      kind: "run",
      sessionId: "session-canonical-progress",
      runId: "run-canonical-progress",
      generation: "generation-canonical-progress",
      attemptToken: "attempt-canonical-progress",
      occurrenceIds: ["occ-canonical-progress"],
      jobIdToItemId: { "71": "occ-canonical-progress" },
      startedAt: Date.now(),
    }
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([record]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [
          {
            jobId: 71,
            status: "awaiting_upload",
            lifecycleState: "awaiting_upload",
            progressPercent: 23,
            progressMessage: "Waiting for replacement file",
            retryable: true,
            sourceItemId: "occ-canonical-progress",
          },
        ],
        errorMessage: null,
      }),
    } as any)

    try {
      await runtime.restore()

      expect(emit).toHaveBeenCalledWith(
        "tldw:quick-ingest/progress",
        expect.objectContaining({
          sessionId: "session-canonical-progress",
          runId: "run-canonical-progress",
          occurrenceId: "occ-canonical-progress",
          lifecycleState: "awaiting_upload",
          progressPercentage: 23,
          progressMessage: "Waiting for replacement file",
          retryable: true,
        })
      )
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("forwards occurrence-scoped cancellation to the restored run authority", async () => {
    vi.useFakeTimers()
    const record = {
      version: 1,
      kind: "run",
      sessionId: "session-row-cancel",
      runId: "run-row-cancel",
      generation: "generation-row-cancel",
      attemptToken: "attempt-row-cancel",
      occurrenceIds: ["occ-row-cancel", "occ-other"],
      jobIdToItemId: {},
      startedAt: Date.now(),
    }
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([record]),
      saveRunSession: vi.fn(),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      }),
      cancelRun,
    } as any)

    try {
      await runtime.restore()
      await (runtime.cancel as any)(
        "session-row-cancel",
        "user_cancelled",
        ["occ-row-cancel"]
      )

      expect(cancelRun).toHaveBeenCalledWith(
        expect.objectContaining({ runId: "run-row-cancel" }),
        "user_cancelled",
        ["occ-row-cancel"]
      )
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("preserves canonical retry eligibility in terminal persistence and replay", async () => {
    vi.useFakeTimers()
    const record = {
      version: 1,
      kind: "run",
      sessionId: "session-terminal-retryable",
      runId: "run-terminal-retryable",
      generation: "generation-terminal-retryable",
      attemptToken: "attempt-terminal-retryable",
      occurrenceIds: ["occ-terminal-retryable"],
      jobIdToItemId: { "91": "occ-terminal-retryable" },
      startedAt: Date.now(),
    }
    const emit = vi.fn()
    const saveRunSession = vi.fn().mockResolvedValue(true)
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([record]),
      saveRunSession,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "failed",
        jobs: [
          {
            jobId: 91,
            status: "failed",
            sourceItemId: "occ-terminal-retryable",
            lifecycleState: "terminal",
            terminalOutcome: "processing_failed",
            progressPercent: 100,
            retryable: true,
            error: "Temporary worker outage",
          },
        ],
        errorMessage: "Temporary worker outage",
      }),
    } as any)

    try {
      await runtime.restore()

      const terminalEmission = emit.mock.calls.find(
        ([type]) => type === "tldw:quick-ingest/failed"
      )
      expect(terminalEmission?.[1]?.results).toEqual([
        expect.objectContaining({
          id: "occ-terminal-retryable",
          retryable: true,
        }),
      ])

      const replay = await runtime.replay("session-terminal-retryable")
      expect((replay as any).event?.payload?.results).toEqual([
        expect.objectContaining({
          id: "occ-terminal-retryable",
          retryable: true,
        }),
      ])
      expect(saveRunSession).toHaveBeenCalledWith(
        expect.objectContaining({ kind: "terminal" }),
        "session-terminal-retryable",
        "run-terminal-retryable",
        "generation-terminal-retryable"
      )
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("cancels one pre-run occurrence without cancelling or submitting its siblings", async () => {
    let releasePreparation!: () => void
    const preparationGate = new Promise<void>((resolve) => {
      releasePreparation = resolve
    })
    let runContext: any
    const submittedOccurrenceIds: string[] = []
    const emit = vi.fn()
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(async (payload: any, context: any) => {
        runContext = context
        await preparationGate
        const inputs = payload.pendingRunRequest.inputs as Array<{
          occurrenceId: string
        }>
        const isOccurrenceCancelled = context.isOccurrenceCancelled
        for (const input of inputs) {
          if (
            typeof isOccurrenceCancelled === "function" &&
            isOccurrenceCancelled(input.occurrenceId)
          ) {
            continue
          }
          submittedOccurrenceIds.push(input.occurrenceId)
        }
        return { results: [] }
      }),
      emit,
      saveRunSession: vi.fn().mockResolvedValue(true),
      createSessionId: () => "session-pre-run-row-cancel",
    } as any)

    const ack = await runtime.start({
      pendingRunRequest: {
        inputs: [
          { occurrenceId: "occ-pre-run-cancel" },
          { occurrenceId: "occ-pre-run-continue" },
        ],
      },
    })
    expect(ack.ok).toBe(true)
    await vi.waitFor(() => expect(runContext).toBeTruthy())

    const response = await (runtime.cancel as any)(
      "session-pre-run-row-cancel",
      "user_cancelled",
      ["occ-pre-run-cancel"]
    )
    expect(response).toEqual({ ok: true })
    expect(runContext.isCancelled()).toBe(false)
    expect(runContext.isOccurrenceCancelled).toBeTypeOf("function")

    releasePreparation()
    await vi.waitFor(() =>
      expect(submittedOccurrenceIds).toEqual(["occ-pre-run-continue"])
    )
    expect(
      emit.mock.calls.some(
        ([type]) => type === "tldw:quick-ingest/cancelled"
      )
    ).toBe(false)
  })

  it("re-arms a retained terminal session after an accepted retry instead of replaying its tombstone", async () => {
    vi.useFakeTimers()
    const retainedExpiresAt = Date.now() + 60_000
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-rearm",
      runId: "run-extension-retry-rearm",
      generation: "generation-extension-retry-old",
      attemptToken: "attempt-extension-retry-rearm",
      expiresAt: retainedExpiresAt,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-rearm",
          runId: "run-extension-retry-rearm",
          results: [
            {
              id: "occ-extension-retry-rearm",
              status: "error",
              retryable: true,
              error: "Temporary worker outage",
            },
          ],
        },
      },
    }
    const emit = vi.fn()
    const callOrder: string[] = []
    const saveRunSession = vi.fn(async (record: any) => {
      callOrder.push(`save:${record.kind}`)
      return true
    })
    const retryRun = vi.fn(async () => {
      callOrder.push("retry")
      return { ok: true }
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 92,
          status: "queued",
          sourceItemId: "occ-extension-retry-rearm",
          lifecycleState: "queued",
          progressPercent: 0,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit,
      loadRunSessions: vi.fn().mockResolvedValue([terminalRecord]),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    try {
      await runtime.restore()
      const retry = (runtime as any).retry
      expect(retry).toBeTypeOf("function")
      if (typeof retry !== "function") return

      const response = await retry(
        "session-extension-retry-rearm",
        "run-extension-retry-rearm",
        ["occ-extension-retry-rearm"]
      )

      expect(response).toMatchObject({
        ok: true,
        generation: expect.not.stringMatching(/old/),
      })
      expect(retryRun).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionId: "session-extension-retry-rearm",
          runId: "run-extension-retry-rearm",
        }),
        ["occ-extension-retry-rearm"]
      )
      expect(saveRunSession).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          kind: "retrying",
          sessionId: "session-extension-retry-rearm",
          runId: "run-extension-retry-rearm",
          generation: expect.not.stringMatching(/old/),
        }),
        "session-extension-retry-rearm",
        "run-extension-retry-rearm",
        "generation-extension-retry-old"
      )
      const reservedGeneration = saveRunSession.mock.calls[0]?.[0]?.generation
      expect(saveRunSession.mock.calls[0]?.[0]?.expiresAt).toBeGreaterThan(
        retainedExpiresAt,
      )
      expect(saveRunSession).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          kind: "run",
          sessionId: "session-extension-retry-rearm",
          runId: "run-extension-retry-rearm",
          generation: reservedGeneration,
        }),
        "session-extension-retry-rearm",
        "run-extension-retry-rearm",
        reservedGeneration
      )
      expect(callOrder).toEqual(["save:retrying", "retry", "save:run"])
      expect(reattachRun).toHaveBeenCalled()
      expect(
        emit.mock.calls.slice(0).some(
          ([type, payload]) =>
            type === "tldw:quick-ingest/failed" &&
            payload?.error === "Temporary worker outage"
        )
      ).toBe(false)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it("allows only one concurrent retry caller to reserve authority and mutate the backend", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-concurrent",
      runId: "run-extension-retry-concurrent",
      generation: "generation-extension-retry-concurrent-old",
      attemptToken: "attempt-extension-retry-concurrent",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-concurrent",
          runId: "run-extension-retry-concurrent",
          results: [
            {
              id: "occ-extension-retry-concurrent",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    let stored: any = terminalRecord
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        if (stored.generation !== expected) return false
        stored = record
        return true
      }
    )
    const retryRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "processing",
        jobs: [],
        errorMessage: null,
      }),
    } as any)

    await runtime.restore()
    const responses = await Promise.all([
      (runtime as any).retry(
        "session-extension-retry-concurrent",
        "run-extension-retry-concurrent",
        ["occ-extension-retry-concurrent"]
      ),
      (runtime as any).retry(
        "session-extension-retry-concurrent",
        "run-extension-retry-concurrent",
        ["occ-extension-retry-concurrent"]
      ),
    ])

    expect(retryRun).toHaveBeenCalledTimes(1)
    expect(responses.filter((response) => response.ok)).toHaveLength(1)
    expect(responses.filter((response) => !response.ok)).toEqual([
      expect.objectContaining({
        error: expect.stringMatching(/retry|reserved|superseded/i),
      }),
    ])
  })

  it("does not let restore promote a retry reservation while its live owner awaits the backend", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-staggered",
      runId: "run-extension-retry-staggered",
      generation: "generation-extension-retry-staggered-old",
      attemptToken: "attempt-extension-retry-staggered",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-staggered",
          runId: "run-extension-retry-staggered",
          results: [
            {
              id: "occ-extension-retry-staggered",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    let stored: any = terminalRecord
    let reservationSaved!: () => void
    const reservationGate = new Promise<void>((resolve) => {
      reservationSaved = resolve
    })
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        if (stored.generation !== expected) return false
        stored = record
        if (record?.kind === "retrying") reservationSaved()
        return true
      },
    )
    let releaseRetry!: () => void
    const retryGate = new Promise<void>((resolve) => {
      releaseRetry = resolve
    })
    const retryRun = vi.fn(async () => {
      await retryGate
      return { ok: true }
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    await runtime.restore()
    const first = (runtime as any).retry(
      "session-extension-retry-staggered",
      "run-extension-retry-staggered",
      ["occ-extension-retry-staggered"],
    )
    await reservationGate
    await vi.waitFor(() => expect(retryRun).toHaveBeenCalledTimes(1))

    try {
      const second = await (runtime as any).retry(
        "session-extension-retry-staggered",
        "run-extension-retry-staggered",
        ["occ-extension-retry-staggered"],
      )

      expect(second).toMatchObject({
        ok: false,
        error: expect.stringMatching(/busy|reserved|retry.*progress|live owner/i),
      })
      expect(retryRun).toHaveBeenCalledTimes(1)
      expect(reattachRun).not.toHaveBeenCalled()
      expect(
        saveRunSession.mock.calls.some(([record]) => record?.kind === "run"),
      ).toBe(false)
    } finally {
      releaseRetry()
      await first
    }
  })

  it("restores a retryable terminal record in the reserved generation when the backend rejects retry", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-reject",
      runId: "run-extension-retry-reject",
      generation: "generation-extension-retry-reject-old",
      attemptToken: "attempt-extension-retry-reject",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-reject",
          runId: "run-extension-retry-reject",
          results: [
            {
              id: "occ-extension-retry-reject",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    let stored: any = terminalRecord
    const callOrder: string[] = []
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        callOrder.push(`save:${record.kind}`)
        if (stored.generation !== expected) return false
        stored = record
        return true
      }
    )
    const retryRun = vi.fn(async () => {
      callOrder.push("retry")
      return { ok: false, error: "Backend rejected retry" }
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun: vi.fn(),
    } as any)

    await runtime.restore()
    await expect(
      (runtime as any).retry(
        "session-extension-retry-reject",
        "run-extension-retry-reject",
        ["occ-extension-retry-reject"]
      )
    ).resolves.toMatchObject({ ok: false, error: "Backend rejected retry" })

    expect(callOrder).toEqual(["save:retrying", "retry", "save:terminal"])
    const reservedGeneration = saveRunSession.mock.calls[0]?.[0]?.generation
    expect(stored).toMatchObject({
      kind: "terminal",
      generation: reservedGeneration,
      event: {
        payload: {
          results: [
            expect.objectContaining({
              id: "occ-extension-retry-reject",
              retryable: true,
            }),
          ],
        },
      },
    })
  })

  it("keeps ambiguous retry authority reserved until staged occurrences are reconciled and resubmitted", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-ambiguous",
      runId: "run-extension-retry-ambiguous",
      generation: "generation-extension-retry-ambiguous-old",
      attemptToken: "attempt-extension-retry-ambiguous",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-ambiguous",
          runId: "run-extension-retry-ambiguous",
          results: [
            {
              id: "occ-extension-retry-ambiguous",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    let stored: any = terminalRecord
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        if (stored.generation !== expected) return false
        stored = record
        return true
      },
    )
    const retryRun = vi.fn().mockResolvedValue({
      ok: false,
      indeterminate: true,
      error: "Retry was accepted but its response timed out.",
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 815,
          status: "processing",
          sourceItemId: "occ-extension-retry-ambiguous",
          lifecycleState: "running",
          progressPercent: 17,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    await runtime.restore()
    const first = await (runtime as any).retry(
      "session-extension-retry-ambiguous",
      "run-extension-retry-ambiguous",
      ["occ-extension-retry-ambiguous"],
    )
    expect(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/old/),
    })
    expect(stored).toMatchObject({
      kind: "retrying",
      generation: first.generation,
      occurrenceIds: ["occ-extension-retry-ambiguous"],
    })
    expect(reattachRun).not.toHaveBeenCalled()
    expect(retryRun).toHaveBeenCalledTimes(1)
  })

  it("reconciles a restored retry reservation before promoting it to active polling", async () => {
    let stored: any = {
      version: 1,
      kind: "retrying",
      sessionId: "session-extension-retry-restore-reconcile",
      runId: "run-extension-retry-restore-reconcile",
      generation: "generation-extension-retry-restore-reconcile-g2",
      attemptToken: "attempt-extension-retry-restore-reconcile",
      occurrenceIds: ["occ-extension-retry-restore-reconcile"],
      startedAt: Date.now(),
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-restore-reconcile",
          runId: "run-extension-retry-restore-reconcile",
          results: [
            {
              id: "occ-extension-retry-restore-reconcile",
              status: "error",
              retryable: true,
              error: "Retry response was lost",
            },
          ],
        },
      },
    }
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        if (stored.generation !== expected) return false
        stored = record
        return true
      },
    )
    const retryRun = vi.fn().mockResolvedValue({ ok: true })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 816,
          status: "queued",
          sourceItemId: "occ-extension-retry-restore-reconcile",
          lifecycleState: "queued",
          progressPercent: 0,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    await runtime.restore()

    expect(retryRun).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "extension-runtime",
        sessionId: "session-extension-retry-restore-reconcile",
        runId: "run-extension-retry-restore-reconcile",
        generation: "generation-extension-retry-restore-reconcile-g2",
      }),
      ["occ-extension-retry-restore-reconcile"],
      { reconcileOnly: true },
    )
    expect(stored).toMatchObject({
      kind: "run",
      generation: "generation-extension-retry-restore-reconcile-g2",
    })
    expect(reattachRun).toHaveBeenCalledTimes(1)
  })

  it("returns pending g2 across repeated restore reconciliation without issuing a second backend retry", async () => {
    let stored: any = {
      version: 1,
      kind: "retrying",
      sessionId: "session-extension-retry-pending-reconcile",
      runId: "run-extension-retry-pending-reconcile",
      generation: "generation-extension-retry-pending-reconcile-g2",
      attemptToken: "attempt-extension-retry-pending-reconcile",
      occurrenceIds: ["occ-extension-retry-pending-reconcile"],
      startedAt: Date.now(),
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-pending-reconcile",
          runId: "run-extension-retry-pending-reconcile",
          results: [
            {
              id: "occ-extension-retry-pending-reconcile",
              status: "error",
              retryable: true,
              error: "Retry response was lost",
            },
          ],
        },
      },
    }
    let authoritativeManifestAvailable = false
    const retryRun = vi.fn(async (_tracking, _occurrenceIds, options) => {
      expect(options).toEqual({ reconcileOnly: true })
      return authoritativeManifestAvailable
        ? { ok: true }
        : {
            ok: false,
            indeterminate: true,
            error: "Authoritative run manifest is temporarily unavailable.",
          }
    })
    const saveRunSession = vi.fn(
      async (record: any, _sessionId: string, _runId: string, expected: string) => {
        if (stored.generation !== expected) return false
        stored = record
        return true
      },
    )
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 817,
          status: "queued",
          sourceItemId: "occ-extension-retry-pending-reconcile",
          lifecycleState: "queued",
          progressPercent: 0,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    await runtime.restore()
    const pending = await runtime.retry(
      "session-extension-retry-pending-reconcile",
      "run-extension-retry-pending-reconcile",
      ["occ-extension-retry-pending-reconcile"],
    )

    expect(pending).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: "generation-extension-retry-pending-reconcile-g2",
    })
    expect(stored).toMatchObject({
      kind: "retrying",
      generation: "generation-extension-retry-pending-reconcile-g2",
    })
    expect(reattachRun).not.toHaveBeenCalled()

    authoritativeManifestAvailable = true
    const reconciled = await runtime.retry(
      "session-extension-retry-pending-reconcile",
      "run-extension-retry-pending-reconcile",
      ["occ-extension-retry-pending-reconcile"],
    )

    expect(reconciled).toEqual({
      ok: true,
      generation: "generation-extension-retry-pending-reconcile-g2",
    })
    expect(retryRun).toHaveBeenCalledTimes(3)
    expect(
      retryRun.mock.calls.every(([, , options]) => options?.reconcileOnly),
    ).toBe(true)
    expect(stored).toMatchObject({
      kind: "run",
      generation: "generation-extension-retry-pending-reconcile-g2",
    })
    expect(reattachRun).toHaveBeenCalledTimes(1)
  })

  it("reconciles newer active authority when an accepted retry loses its active-transition CAS", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-cas-loss",
      runId: "run-extension-retry-cas-loss",
      generation: "generation-extension-retry-cas-loss-old",
      attemptToken: "attempt-extension-retry-cas-loss",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-cas-loss",
          runId: "run-extension-retry-cas-loss",
          results: [
            {
              id: "occ-extension-retry-cas-loss",
              status: "error",
              retryable: true,
              error: "Retained failure",
            },
          ],
        },
      },
    }
    const newerActive = {
      version: 1,
      kind: "run",
      sessionId: "session-extension-retry-cas-loss",
      runId: "run-extension-retry-cas-loss",
      generation: "generation-extension-retry-newer-authority",
      attemptToken: "attempt-extension-retry-cas-loss",
      submissionState: "acknowledged",
      occurrenceIds: ["occ-extension-retry-cas-loss"],
      jobIdToItemId: {},
      startedAt: 777,
    }
    let stored: any = terminalRecord
    let saveCalls = 0
    const saveRunSession = vi.fn(async (record: any) => {
      saveCalls += 1
      if (saveCalls === 1) {
        stored = record
        return true
      }
      stored = newerActive
      return false
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun: vi.fn().mockResolvedValue({ ok: true }),
      reattachRun,
    } as any)

    await runtime.restore()
    await (runtime as any).retry(
      "session-extension-retry-cas-loss",
      "run-extension-retry-cas-loss",
      ["occ-extension-retry-cas-loss"]
    )

    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: "session-extension-retry-cas-loss",
        runId: "run-extension-retry-cas-loss",
        startedAt: 777,
      }),
      { transportPreference: "poll" }
    )
    await expect(
      runtime.replay("session-extension-retry-cas-loss")
    ).resolves.toMatchObject({ ok: true, active: true })
  })

  it("keeps an accepted retry polling in memory when active persistence throws", async () => {
    const terminalRecord = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-persist-throw",
      runId: "run-extension-retry-persist-throw",
      generation: "generation-extension-retry-persist-throw-old",
      attemptToken: "attempt-extension-retry-persist-throw",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-persist-throw",
          runId: "run-extension-retry-persist-throw",
          results: [
            {
              id: "occ-extension-retry-persist-throw",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    let saveCalls = 0
    const saveRunSession = vi.fn(async () => {
      saveCalls += 1
      if (saveCalls === 1) return true
      throw new Error("storage unavailable")
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 814,
          status: "processing",
          sourceItemId: "occ-extension-retry-persist-throw",
          lifecycleState: "running",
          progressPercent: 43,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn().mockResolvedValue([terminalRecord]),
      saveRunSession,
      retryRun: vi.fn().mockResolvedValue({ ok: true }),
      reattachRun,
    } as any)

    await runtime.restore()
    await expect(
      (runtime as any).retry(
        "session-extension-retry-persist-throw",
        "run-extension-retry-persist-throw",
        ["occ-extension-retry-persist-throw"]
      )
    ).resolves.toMatchObject({
      ok: false,
      generation: expect.not.stringMatching(/old/),
      error: expect.stringMatching(/persist|recover|storage/i),
    })

    expect(reattachRun).toHaveBeenCalled()
    await expect(
      runtime.replay("session-extension-retry-persist-throw")
    ).resolves.toMatchObject({ ok: true, active: true })
  })

  it("retries once after reconciliation proves a lost retry request never advanced the backend", async () => {
    let stored: any = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-retry-lost-before-delivery",
      runId: "run-extension-retry-lost-before-delivery",
      generation: "generation-extension-retry-lost-before-delivery-g1",
      attemptToken: "attempt-extension-retry-lost-before-delivery",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-retry-lost-before-delivery",
          runId: "run-extension-retry-lost-before-delivery",
          results: [
            {
              id: "occ-extension-retry-lost-before-delivery",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    const saveRunSession = vi.fn(
      async (record: any, _sessionId?: string, _runId?: string, expected?: string) => {
        if (expected && stored?.generation !== expected) return false
        stored = record
        return true
      },
    )
    let retryPosts = 0
    let reconcileCalls = 0
    let authoritativeSubmissions = 0
    const retryRun = vi.fn(async (_tracking, _occurrenceIds, options) => {
      if (options?.reconcileOnly) {
        reconcileCalls += 1
        return {
          ok: false,
          indeterminate: true,
          notAdvanced: true,
          error: "The selected URL has not advanced beyond attempt 1.",
        }
      }
      retryPosts += 1
      if (retryPosts === 1) {
        return {
          ok: false,
          indeterminate: true,
          notAdvanced: true,
          error: "Retry request timed out before backend delivery.",
        }
      }
      authoritativeSubmissions += 1
      return { ok: true }
    })
    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 940,
          status: "queued",
          sourceItemId: "occ-extension-retry-lost-before-delivery",
          lifecycleState: "queued",
          progressPercent: 0,
          retryable: false,
        },
      ],
      errorMessage: null,
    })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => (stored ? [stored] : [])),
      saveRunSession,
      retryRun,
      reattachRun,
    } as any)

    await runtime.restore()
    const first = await runtime.retry(
      "session-extension-retry-lost-before-delivery",
      "run-extension-retry-lost-before-delivery",
      ["occ-extension-retry-lost-before-delivery"],
    )
    expect(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const second = await runtime.retry(
      "session-extension-retry-lost-before-delivery",
      "run-extension-retry-lost-before-delivery",
      ["occ-extension-retry-lost-before-delivery"],
    )

    expect(second).toEqual({ ok: true, generation: first.generation })
    expect(reconcileCalls).toBeGreaterThanOrEqual(1)
    expect(retryPosts).toBe(2)
    expect(authoritativeSubmissions).toBe(1)
    expect(stored).toMatchObject({
      kind: "run",
      generation: first.generation,
    })
    expect(reattachRun).toHaveBeenCalledTimes(1)
  })

  it("routes pending retry occurrence cancellation through g2 and rejects stale g1 without mutation", async () => {
    const terminalGeneration = "generation-extension-pending-cancel-g1"
    let stored: any = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-pending-cancel",
      runId: "run-extension-pending-cancel",
      generation: terminalGeneration,
      attemptToken: "attempt-extension-pending-cancel",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-pending-cancel",
          runId: "run-extension-pending-cancel",
          results: [
            {
              id: "occ-extension-pending-cancel",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    const saveRunSession = vi.fn(
      async (record: any, _sessionId?: string, _runId?: string, expected?: string) => {
        if (expected && stored?.generation !== expected) return false
        stored = record
        return true
      },
    )
    const retryRun = vi.fn(async (_tracking, _occurrenceIds, options) => ({
      ok: false,
      indeterminate: true,
      ...(options?.reconcileOnly ? { notAdvanced: true } : {}),
      error: "Retry authority is awaiting reconciliation.",
    }))
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      cancelRun,
    } as any)

    await runtime.restore()
    const retry = await runtime.retry(
      "session-extension-pending-cancel",
      "run-extension-pending-cancel",
      ["occ-extension-pending-cancel"],
    )
    expect(retry).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const stale = await runtime.cancel(
      "session-extension-pending-cancel",
      "user_cancelled",
      ["occ-extension-pending-cancel"],
      terminalGeneration,
    )
    expect(stale).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|superseded/i),
    })
    expect(cancelRun).not.toHaveBeenCalled()

    const current = await runtime.cancel(
      "session-extension-pending-cancel",
      "user_cancelled",
      ["occ-extension-pending-cancel"],
      retry.generation,
    )

    expect(current).toEqual({ ok: true })
    expect(cancelRun).toHaveBeenCalledTimes(1)
    expect(cancelRun).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "extension-runtime",
        sessionId: "session-extension-pending-cancel",
        runId: "run-extension-pending-cancel",
        generation: retry.generation,
      }),
      "user_cancelled",
      ["occ-extension-pending-cancel"],
    )
    expect(stored).toMatchObject({
      kind: "retrying",
      generation: retry.generation,
      occurrenceIds: ["occ-extension-pending-cancel"],
    })
  })

  it("routes pending g2 Cancel All without occurrence ids and retires its fence only at terminal authority", async () => {
    const terminalGeneration = "generation-extension-pending-cancel-all-g1"
    let stored: any = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-pending-cancel-all",
      runId: "run-extension-pending-cancel-all",
      generation: terminalGeneration,
      attemptToken: "attempt-extension-pending-cancel-all",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: "session-extension-pending-cancel-all",
          runId: "run-extension-pending-cancel-all",
          results: [
            {
              id: "occ-extension-pending-cancel-all",
              status: "error",
              retryable: true,
              error: "Retryable failure",
            },
          ],
        },
      },
    }
    const saveRunSession = vi.fn(
      async (record: any, _sessionId?: string, _runId?: string, expected?: string) => {
        if (expected && stored?.generation !== expected) return false
        stored = record
        return true
      },
    )
    const retryRun = vi.fn(async (_tracking, _occurrenceIds, options) => ({
      ok: false,
      indeterminate: true,
      ...(options?.reconcileOnly ? { notAdvanced: true } : {}),
      error: "Retry authority is awaiting reconciliation.",
    }))
    const cancelRun = vi.fn().mockResolvedValue({ ok: true })
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => [stored]),
      saveRunSession,
      retryRun,
      cancelRun,
    } as any)

    await runtime.restore()
    const retry = await runtime.retry(
      "session-extension-pending-cancel-all",
      "run-extension-pending-cancel-all",
      ["occ-extension-pending-cancel-all"],
    )
    expect(retry).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const stale = await runtime.cancel(
      "session-extension-pending-cancel-all",
      "user_cancelled",
      undefined,
      terminalGeneration,
    )
    expect(stale).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|superseded/i),
    })
    expect(cancelRun).not.toHaveBeenCalled()

    const current = await runtime.cancel(
      "session-extension-pending-cancel-all",
      "user_cancelled",
      undefined,
      retry.generation,
    )

    expect(current).toEqual({ ok: true })
    expect(cancelRun).toHaveBeenCalledTimes(1)
    expect(cancelRun).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "extension-runtime",
        sessionId: "session-extension-pending-cancel-all",
        runId: "run-extension-pending-cancel-all",
        generation: retry.generation,
      }),
      "user_cancelled",
    )
    expect(stored).toMatchObject({
      kind: "retrying",
      generation: retry.generation,
      occurrenceIds: ["occ-extension-pending-cancel-all"],
    })

    stored = {
      version: 1,
      kind: "terminal",
      sessionId: "session-extension-pending-cancel-all",
      runId: "run-extension-pending-cancel-all",
      generation: retry.generation,
      attemptToken: "attempt-extension-pending-cancel-all",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/cancelled",
        payload: {
          sessionId: "session-extension-pending-cancel-all",
          runId: "run-extension-pending-cancel-all",
          generation: retry.generation,
          results: [],
        },
      },
    }
    await runtime.restore()

    const afterTerminal = await runtime.cancel(
      "session-extension-pending-cancel-all",
      "user_cancelled",
      undefined,
      retry.generation,
    )
    expect(afterTerminal).toMatchObject({
      ok: false,
      error: expect.stringMatching(/not found|terminal/i),
    })
    expect(cancelRun).toHaveBeenCalledTimes(1)
  })
})
