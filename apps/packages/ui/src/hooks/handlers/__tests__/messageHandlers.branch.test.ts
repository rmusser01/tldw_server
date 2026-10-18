import { QueryObserver } from "@tanstack/react-query"
import i18n from 'i18next';
import ICUWithInterpolation from '@/i18n/icu-format';
import enPlayground from '@/assets/locale/en/playground.json';
import { finishForkOperation, recordForkCandidate } from '@/db/dexie/fork-operations';
import { createQueryClient, invalidateServerChatHistory } from '@/services/query-client';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
const forks = vi.hoisted(() => ({ prepare: vi.fn(), commit: vi.fn() }));
const nativeForks = vi.hoisted(() => ({ prepare: vi.fn(), commit: vi.fn() }));
vi.mock('@/services/chat-history-selection', () => ({
  prepareNativeFork: nativeForks.prepare,
  commitNativeFork: nativeForks.commit,
}));
vi.mock('@/db/dexie/fork-operations', () => ({
  prepareForkOperation: async (request: any, context: any) => ({
    ...request,
    request,
    context,
    state: 'prepared',
  }),
  claimForkOperation: async (record: any) => ({ record }),
  recordForkCandidate: vi.fn(async () => {}),
  finishForkOperation: vi.fn(async () => {}),
  forkOperationResult: (record: any) => record.result,
}));
const edits = vi.hoisted(() => ({ update: vi.fn() }));
vi.mock('@/db/dexie/branch', () => ({
  prepareLocalFork: forks.prepare,
  commitLocalFork: forks.commit
}))
vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: (rows: any[]) =>
    rows.map((row) => ({ content: row.content })),
  formatToMessage: (rows: any[]) =>
    rows.map((row) => ({ id: row.id, message: row.content })),
  updateMessageById: edits.update
}))
import {
  createBranchMessage,
  createEditMessage,
  createRegenerateLastMessage
} from "../messageHandlers"
const request: any = {
  operation_id: "op",
  owner_key: "local",
  destination_owner_key: "local",
  request_digest: "digest",
  input: { kind: "normal", selection: { conversation_id: "source" } }
}
const committed = {
  state: "committed",
  operation_id: "op",
  owner_key: "local",
  child_id: "child",
  message_map: { u: "child-u" }
}
const testController = () => {
  const state: any = { status: "ready" }
  return {
    getCurrent: () => state,
    loadConversation: vi.fn(async ({ historyId }, _ref, loaded) => {
      state.owner = {
        kind: "local",
        profile_id: "declared-local-profile",
        owner_key: "local",
        conversation_id: historyId
      }
      state.view = { owner_key: "local", conversation_id: historyId }
      loaded({ owner: state.owner, view: state.view })
      return true
    })
  } as any
}
const setup = (extra = {}) => ({
  historySelection: testController(),
  notification: { error: vi.fn(), warning: vi.fn() } as any,
  historyId: "source",
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setHistoryId: vi.fn(),
  setContext: vi.fn(),
  setSystemPrompt: vi.fn(),
  ...extra
})
beforeAll(async () => {
  await i18n.use(ICUWithInterpolation).init({lng: "en", fallbackLng: false, resources: {en: {playground: enPlayground}}, interpolation: {escapeValue: false}})
})
beforeEach(() => {
  vi.clearAllMocks()
  forks.prepare.mockResolvedValue({
    history: { id: "child", last_used_prompt: { prompt_content: "literal" } },
    messages: [{ id: "child-u", content: "hello" }],
    files: { files: [] }
  })
  forks.commit.mockResolvedValue(committed)
})
describe("typed fork handler", () => {
  it("returns committed owner binding and child map after one copy", async () => {
    const options = setup()
    expect(await createBranchMessage(options)(request)).toEqual(committed)
    expect(options.setHistoryId).toHaveBeenCalledWith("child")
    expect(options.setSystemPrompt).toHaveBeenCalledWith("literal")
    expect(forks.commit).toHaveBeenCalledTimes(1)
  })
  it("retains a committed result after navigation without replacing the new view", async () => {
    let current = true
    let release!: (result: any) => void
    forks.commit.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve
        })
    )
    const options = setup({ captureViewFence: () => () => current })
    const pending = createBranchMessage(options)(request)
    await vi.waitFor(() => expect(forks.commit).toHaveBeenCalledTimes(1))
    current = false
    release(committed)
    expect(await pending).toEqual(committed)
    expect(options.setHistoryId).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
  })
  it.each(["partial", "unknown", "rejected"])(
    "preserves %s without navigation or fallback",
    async (state) => {
      const result = {
        state,
        operation_id: "op",
        owner_key: "local",
        code: "failed",
        candidate_child_id: "candidate"
      }
      forks.commit.mockResolvedValue(result)
      const options = setup()
      expect(await createBranchMessage(options)(request)).toEqual(result)
      expect(options.setHistoryId).not.toHaveBeenCalled()
      expect(forks.prepare).toHaveBeenCalledTimes(1)
    }
  )
  it("blocks unverified native replay before any copy", async () => {
    const options = setup({ serverChatId: "native" })
    expect(await createBranchMessage(options)(request)).toEqual({
      state: "blocked",
      operation_id: "op",
      owner_key: "local",
      code: "native_fork_projection_unavailable"
    })
    expect(forks.prepare).not.toHaveBeenCalled()
  })
  it("reports preflight rejection without a snapshot fallback", async () => {
    forks.prepare.mockRejectedValue(new Error("unsupported_asset"))
    const options = setup()
    expect(await createBranchMessage(options)(request)).toMatchObject({
      state: "rejected",
      code: "unsupported_asset",
      owner_key: "local",
      operation_id: "op"
    })
    expect(forks.commit).not.toHaveBeenCalled()
  })
})
it("edit uses the stable selected ID despite greeting offset and updates display only on success", async () => {
  const messages: any = [
    { id: "greeting", message: "hi", isBot: true },
    { id: "selected", message: "same", isBot: true }
  ]
  const options: any = {
    ...setup(),
    messages,
    history: [],
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn()
  }
  await createEditMessage(options)(1, "edited", false, false)
  expect(edits.update).toHaveBeenCalledWith("source", "selected", "edited")
  expect(options.setMessages).toHaveBeenCalledWith([
    messages[0],
    { ...messages[1], message: "edited" }
  ])
  edits.update.mockRejectedValue(new Error("message_owner_mismatch"))
  options.setMessages.mockClear()
  await expect(
    createEditMessage(options)(1, "bad", false, false)
  ).rejects.toThrow("message_owner_mismatch")
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("regenerate gate leaves owner inputs and visible user/assistant untouched", async () => {
  const options: any = {
    ...setup(),
    messages: [
      { id: "u", message: "input", isBot: false },
      { id: "a", message: "answer", isBot: true }
    ],
    history: [{ role: "user", content: "input" }],
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn()
  }
  await expect(createRegenerateLastMessage(options)()).rejects.toThrow(
    "unsupported_history_regeneration"
  )
  expect(options.onSubmit).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("view application failure cannot relabel an acknowledged commit as rejected", async () => {
  const options = setup()
  options.setHistory.mockImplementation(() => {
    throw new Error("view unmounted")
  })
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(forks.commit).toHaveBeenCalledTimes(1)
})

it("waits for an exact child load receipt before displaying a committed child", async () => {
  const state: any = {
    status: "ready",
    owner: { kind: "local", owner_key: "local", conversation_id: "source" },
    view: {}
  }
  const controller = {
    getCurrent: () => state,
    fence: () => () => true,
    loadConversation: vi.fn(async (_target, _reference, loaded) => {
      state.owner = {
        kind: "local",
        profile_id: "declared-local-profile",
        owner_key: "local",
        conversation_id: "child"
      }
      state.view = { owner_key: "local", conversation_id: "child" }
      loaded({ owner: state.owner, view: state.view })
      return true
    })
  }
  const options = setup({ historySelection: controller })
  await createBranchMessage(options)(request)
  expect(controller.loadConversation).toHaveBeenCalledWith(
    { historyId: "child", isCurrent: expect.any(Function) },
    undefined,
    expect.any(Function)
  )
  expect(options.setHistoryId).toHaveBeenCalledWith("child")
})
it("a settled load without a receipt never presents the committed child", async () => {
  const options = setup({
    historySelection: { loadConversation: vi.fn(async () => true) }
  })
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.setHistoryId).not.toHaveBeenCalled()
})

it("a throwing child loader reports saved copy without calling it a fork failure", async () => {
  const options = setup()
  options.historySelection.loadConversation.mockRejectedValue(new Error("opening failed"))
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.notification.error).not.toHaveBeenCalled()
  expect(options.notification.warning).toHaveBeenCalledWith(expect.objectContaining({message: "Copy saved; opening failed"}))
})
it.each(["unknown", "partial"])("retains native %s after deferred response without local fallback", async state => {
  const native = {kind: "native", owner_key: "local", conversation_id: "source", scope: {type: "workspace", workspaceId: "original"}}
  let resolve!: (result: any) => void
  nativeForks.prepare.mockResolvedValue({request})
  nativeForks.commit.mockImplementation(() => new Promise(done => { resolve = done }))
  const options = setup({serverChatId: "source", historyId: null, historySelection: {getCurrent: () => ({owner: native}), refreshForkOperations: vi.fn(), loadConversation: vi.fn()}})
  const pending = createBranchMessage(options)(request)
  await vi.waitFor(() => expect(nativeForks.commit).toHaveBeenCalledTimes(1))
  resolve({state, owner_key: "local", operation_id: "op", code: "response_lost", candidate_child_id: state === "partial" ? "child" : undefined})
  expect(await pending).toMatchObject({state, operation_id: "op"})
  expect(forks.prepare).not.toHaveBeenCalled()
  expect(forks.commit).not.toHaveBeenCalled()
  expect(options.historySelection.loadConversation).not.toHaveBeenCalled()
})

it("reports outcome recording failure with the known child, without opening or copying again", async () => {
  expect(i18n.exists("playground:historySelection.copyRecordFailed")).toBe(true)
  expect(
    i18n.exists("playground:historySelection.copyRecordFailedDetail")
  ).toBe(true)
  vi.mocked(finishForkOperation).mockRejectedValueOnce(
    new Error("storage unavailable")
  )
  const options = setup()
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.historySelection.loadConversation).not.toHaveBeenCalled()
  expect(forks.commit).toHaveBeenCalledTimes(1)
  expect(finishForkOperation).toHaveBeenCalledTimes(1)
  expect(options.notification.error).not.toHaveBeenCalled()
  expect(options.notification.warning).toHaveBeenCalledWith({
    message: "Copy saved; recovery record update failed",
    description:
      "Saved copy: child. Its recovery record could not be updated. Reason: storage unavailable"
  })
})

it.each(["view", "lease"])(
  "does not expose the native child when a held outcome write fails after %s invalidation",
  async (invalidation) => {
    let liveView = true
    let liveLease = true
    let rejectWrite!: (error: Error) => void
    vi.mocked(finishForkOperation).mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          rejectWrite = reject
        })
    )
    const native = {
      kind: "native",
      owner_key: "local",
      conversation_id: "source",
      scope: { type: "global" },
      validate_lease: () => liveLease
    }
    nativeForks.prepare.mockResolvedValue({ request })
    nativeForks.commit.mockResolvedValue(committed)
    const options = setup({
      serverChatId: "source",
      historyId: null,
      captureViewFence: () => () => liveView,
      historySelection: {
        getCurrent: () => ({ owner: native }),
        refreshForkOperations: vi.fn(),
        loadConversation: vi.fn()
      }
    })
    const pending = createBranchMessage(options)(request)
    await vi.waitFor(() => expect(finishForkOperation).toHaveBeenCalledOnce())
    if (invalidation === "view") liveView = false
    else liveLease = false
    rejectWrite(new Error("private old-owner write failure"))
    expect(await pending).toEqual(committed)
    expect(options.notification.warning).not.toHaveBeenCalled()
    expect(options.notification.error).not.toHaveBeenCalled()
    expect(options.historySelection.loadConversation).not.toHaveBeenCalled()
    expect(nativeForks.commit).toHaveBeenCalledOnce()
    expect(forks.commit).not.toHaveBeenCalled()
  }
)

it("keeps same-owner loader failure feedback when the intended handoff retires the source fence", async () => {
  let sourceCurrent = true
  const options = setup({ captureViewFence: () => () => sourceCurrent })
  options.historySelection.loadConversation.mockImplementation(async () => {
    sourceCurrent = false
    throw new Error("opening failed")
  })
  expect(await createBranchMessage(options)(request)).toEqual(committed)
  expect(options.notification.warning).toHaveBeenCalledWith(
    expect.objectContaining({ message: 'Copy saved; opening failed' })
  );
  expect(forks.commit).toHaveBeenCalledOnce();
});

const nativeFinderOptions = () => {
  const account = new AbortController();
  const requestScope = {
    config: { serverUrl: 'https://native-copy.test', authMode: 'single-user' as const },
    userId: null,
  };
  const native = {
    request_scope: requestScope,
    kind: 'native' as const,
    owner_key: 'local',
    conversation_id: 'source',
    scope: { type: 'global' as const },
    validate_lease: () => true,
  };
  return {
    ...setup(),
    retireAccount: () => account.abort(),
    serverChatId: 'source',
    historyId: null,
    historySelection: {
      ...testController(),
      draftAccount: {
        requestScope,
        isCurrent: () => !account.signal.aborted,
        retryVerification: vi.fn(),
      },
      getCurrent: () => ({ owner: native }),
      refreshForkOperations: vi.fn(),
      loadConversation: vi.fn(async () => true),
    },
    onServerChatMutated: vi.fn<(isCurrent: () => boolean) => Promise<void>>(async () => {}),
  };
};
const nativeFinderResult = {
  state: 'legacy_completed' as const,
  operation_id: 'op',
  owner_key: 'local',
  child_id: 'child',
};
const acknowledgeNativeChild = (result = nativeFinderResult) => {
  nativeForks.prepare.mockResolvedValue({ request });
  nativeForks.commit.mockImplementation(async (_prepared, candidate) => {
    await candidate('child');
    return result;
  });
};

describe('native Copy acknowledged finder refresh', () => {
  it('refreshes the acknowledged child before its fallible recovery write', async () => {
    const options = nativeFinderOptions();
    acknowledgeNativeChild();
    let notificationsAtRecord = -1;
    vi.mocked(recordForkCandidate).mockImplementationOnce(async () => {
      notificationsAtRecord = options.onServerChatMutated.mock.calls.length;
      throw new Error('candidate storage unavailable');
    });
    await createBranchMessage(options)(request);
    expect(notificationsAtRecord).toBe(1);
    expect(nativeForks.commit).toHaveBeenCalledOnce();
  });

  it('refreshes copied rows before final recovery storage and preserves saved status on storage failure', async () => {
    const options = nativeFinderOptions();
    acknowledgeNativeChild();
    let notificationsAtRecord = -1;
    vi.mocked(finishForkOperation).mockImplementationOnce(async () => {
      notificationsAtRecord = options.onServerChatMutated.mock.calls.length;
      throw new Error('final storage unavailable');
    });
    expect(await createBranchMessage(options)(request)).toEqual(nativeFinderResult);
    expect(notificationsAtRecord).toBe(2);
    expect(options.notification.warning).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Copy saved; recovery record update failed' })
    );
    expect(options.historySelection.loadConversation).not.toHaveBeenCalled();
  });

  it('refreshes a known partial child without opening or calling the copy complete', async () => {
    const options = nativeFinderOptions();
    const partial = {
      state: 'partial',
      operation_id: 'op',
      owner_key: 'local',
      candidate_child_id: 'child',
      code: 'reply acknowledgement lost',
    };
    nativeForks.prepare.mockResolvedValue({ request });
    nativeForks.commit.mockImplementation(async (_prepared, candidate) => {
      await candidate('child');
      return partial;
    });
    expect(await createBranchMessage(options)(request)).toEqual(partial);
    expect(options.onServerChatMutated).toHaveBeenCalledTimes(2);
    expect(options.historySelection.loadConversation).not.toHaveBeenCalled();
  });

  it.each(['unknown', 'partial', 'legacy_completed'])(
    'does not infer a mutation from an unobserved %s result',
    async (state) => {
      const options = nativeFinderOptions();
      nativeForks.prepare.mockResolvedValue({ request });
      nativeForks.commit.mockResolvedValue({
        state,
        operation_id: 'op',
        owner_key: 'local',
        child_id: 'child',
        candidate_child_id: 'child',
      });
      await createBranchMessage(options)(request);
      expect(options.onServerChatMutated).not.toHaveBeenCalled();
    }
  );

  it.each(['operation', 'owner', 'child'])(
    'does not refresh a foreign final %s binding',
    async (mismatch) => {
      const options = nativeFinderOptions();
      acknowledgeNativeChild({
        ...nativeFinderResult,
        ...(mismatch === 'operation'
          ? { operation_id: 'other' }
          : mismatch === 'owner'
            ? { owner_key: 'other' }
            : { child_id: 'other' }),
      });
      await createBranchMessage(options)(request);
      expect(options.onServerChatMutated).toHaveBeenCalledOnce();
    }
  );

  it('keeps the original account refresh after the source view is retired', async () => {
    const options = nativeFinderOptions();
    let current = true;
    options.historySelection.getCurrent().owner.validate_lease = () => current;
    nativeForks.prepare.mockResolvedValue({ request });
    nativeForks.commit.mockImplementation(async (_prepared, candidate) => {
      current = false;
      await candidate('child');
      return nativeFinderResult;
    });
    const pending = createBranchMessage({ ...options, captureViewFence: () => () => current })(
      request
    );
    expect(await pending).toEqual(nativeFinderResult);
    expect(options.onServerChatMutated).toHaveBeenCalledTimes(2);
    expect(options.onServerChatMutated.mock.calls.every(([isCurrent]) => isCurrent())).toBe(true);
    expect(options.historySelection.loadConversation).not.toHaveBeenCalled();
  });

  it('permanently retires original-account refresh on account ABA before creation acknowledgement', async () => {
    const options = nativeFinderOptions();
    nativeForks.prepare.mockResolvedValue({ request });
    nativeForks.commit.mockImplementation(async (_prepared, candidate) => {
      const original = options.historySelection.draftAccount;
      options.retireAccount();
      options.historySelection.draftAccount = { ...original, isCurrent: () => true };
      await candidate('child');
      return nativeFinderResult;
    });
    expect(await createBranchMessage(options)(request)).toEqual(nativeFinderResult);
    expect(options.onServerChatMutated).not.toHaveBeenCalled();
  });

  it('keeps the original account predicate while the actual cache helper awaits cancellation', async () => {
    const options = nativeFinderOptions();
    acknowledgeNativeChild();
    const client = createQueryClient();
    const releases: (() => void)[] = [];
    const cancel = vi.spyOn(client, 'cancelQueries').mockImplementation(() =>
      new Promise<void>(resolve => { releases.push(resolve); })
    );
    const refresh = vi.spyOn(client, 'invalidateQueries');
    options.onServerChatMutated.mockImplementation((isCurrent) =>
      invalidateServerChatHistory(client, isCurrent)
    );
    const pending = createBranchMessage(options)(request);
    expect(await pending).toEqual(nativeFinderResult);
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledTimes(2));
    options.retireAccount();
    releases.forEach(release => release());
    await vi.waitFor(() => expect(options.onServerChatMutated.mock.calls.every(([isCurrent]) => !isCurrent())).toBe(true));
    expect(refresh).not.toHaveBeenCalled();
    expect(options.onServerChatMutated).toHaveBeenCalledTimes(2);
    client.clear();
  });

  it('never converts a saved copy into failure when finder refresh rejects', async () => {
    const options = nativeFinderOptions();
    acknowledgeNativeChild();
    options.onServerChatMutated.mockRejectedValue(new Error('finder unavailable'));
    expect(await createBranchMessage(options)(request)).toEqual(nativeFinderResult);
    expect(recordForkCandidate).toHaveBeenCalledOnce();
    expect(options.notification.error).not.toHaveBeenCalled();
    expect(options.notification.warning).not.toHaveBeenCalled();
  });

  it('does not refresh server finders for a genuine local copy', async () => {
    const refresh = vi.fn();
    expect(
      await createBranchMessage({ ...setup(), onServerChatMutated: refresh })(request)
    ).toEqual(committed);
    expect(refresh).not.toHaveBeenCalled();
  });
});

it('refuses finder refresh from a different captured draft account', async () => {
  const options = nativeFinderOptions();
  options.historySelection.draftAccount.requestScope = {
    config: { serverUrl: 'https://foreign.test', authMode: 'single-user' },
    userId: null,
  };
  acknowledgeNativeChild();
  expect(await createBranchMessage(options)(request)).toEqual(nativeFinderResult);
  expect(options.onServerChatMutated).not.toHaveBeenCalled();
});


it.each(["creation", "completion"])("a held actual finder GET cannot delay Copy %s journaling or completion", async phase => {
  const options = nativeFinderOptions()
  acknowledgeNativeChild()
  const client = createQueryClient()
  const key = ["serverChatHistory", "copy-owner"]
  client.setQueryData(key, ["old source"])
  let release!: () => void
  let requested = false
  const observer = new QueryObserver(client, { queryKey: key, staleTime: Infinity,
    queryFn: async () => { requested = true; await new Promise<void>(resolve => { release = resolve }); return ["source and copied child"] } })
  const unsubscribe = observer.subscribe(() => {})
  options.onServerChatMutated.mockImplementation(isCurrent => invalidateServerChatHistory(client, isCurrent))
  if (phase === "completion") options.onServerChatMutated.mockImplementationOnce(async () => {})
  let copied: typeof nativeFinderResult | undefined
  const pending = createBranchMessage(options)(request).then(result => {
    if (result.state === "legacy_completed") copied = result
    return result
  })
  try {
    await vi.waitFor(() => expect(requested).toBe(true))
    await vi.waitFor(() => expect(copied).toEqual(nativeFinderResult))
    expect(recordForkCandidate).toHaveBeenCalledOnce()
    expect(finishForkOperation).toHaveBeenCalledOnce()
  } finally {
    unsubscribe()
    release?.()
    await pending
    client.clear()
  }
})
