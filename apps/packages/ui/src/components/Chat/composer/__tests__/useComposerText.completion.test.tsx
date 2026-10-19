// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStoreMessageOption } from "@/store/option";
import { usePlaygroundSessionStore } from "@/store/playground-session";
import { conversationDraftKey } from "@/utils/chat-draft-address";
import { useHistorySelection } from "@/hooks/chat/useHistorySelection";
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition";
import { formatSelectedHistory } from "@/db/dexie/helpers";
import { resolveHistorySelection } from "@/utils/history-selection";
import type {
  HistorySelectionSnapshotV1,
  HistoryViewSelectionV1,
} from "@/types/history-selection";
import type { HistoryOwnerV1 } from "@/services/chat-history-selection";
import {
  isChatSubmitSuccess,
  normalizeChatSubmitResult,
} from "@/hooks/chat/chat-action-utils";

const authority = vi.hoisted(() => ({ user: "alice" }));
vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => ({
    userId: authority.user,
    config: {
      serverUrl: "http://completion.test",
      authMode: "multi-user" as const,
    },
  }),
}));
import { useComposerText } from "../hooks/useComposerText";
// H1/bookmark/fork I/O is doubled; open, qualified receipt binding, followResult,
// display projection, composer restoration and account watcher are production.
vi.mock("@/db/dexie/history-selection", async (original) => ({
  ...(await original<typeof import("@/db/dexie/history-selection")>()),
  ensureLocalProfileId: async () => "profile",
  loadHistoryBookmark: async () => null,
  saveHistoryBookmark: async () => {},
  loadHistoryTurnRecoveries: async () => [],
}));
vi.mock("@/db/dexie/fork-operations", async (original) => ({
  ...(await original<typeof import("@/db/dexie/fork-operations")>()),
  loadForkOperations: async () => [],
  findForkCandidate: async () => null,
}));
const completedSnapshot: HistorySelectionSnapshotV1 = {
  version: 1,
  owner_key: "completion-owner",
  conversation_id: "first-result",
  source_digest: "source",
  storage_context_digest: "storage",
  fences: { conversation: "1", history: "1", settings: "1" },
  interpretation_status: { kind: "parent_graph_v1" },
  nodes: [
    {
      id: "u",
      revision: "u1",
      parent_id: null,
      role: "user",
      settled: true,
      preview: "Question",
    },
    {
      id: "a",
      revision: "a1",
      parent_id: "u",
      role: "assistant",
      settled: true,
      preview: "Answer",
    },
  ],
};
vi.mock("@/services/chat-history-selection", async (original) => ({
  ...(await original<typeof import("@/services/chat-history-selection")>()),
  captureHistorySnapshot: async (
    _owner: HistoryOwnerV1,
    view: HistoryViewSelectionV1,
  ) => {
    const bound = { ...view, owner_key: completedSnapshot.owner_key },
      result = resolveHistorySelection(completedSnapshot, bound, "send", "");
    if (result.status !== "ready")
      return { ...result, snapshot: completedSnapshot, view: bound };
    return {
      status: "captured",
      snapshot: completedSnapshot,
      view: bound,
      rows: result.rows,
      selected_content: result.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: row.preview,
        images: [],
      })),
      purpose: "send",
      storage_context_digest: "storage",
    };
  },
}));

// Exercise the deployed mutation callbacks with real composer/draft/owner state.
// The large UI is omitted; network, query invalidation and milestones are I/O doubles.
const deployedMutation = (
  surface: "Playground" | "Sidepanel",
  composer: ReturnType<typeof useComposerText>,
) => {
  const file =
    surface === "Playground"
      ? "../../../Option/Playground/PlaygroundForm.tsx"
      : "../../../Sidepanel/Chat/form.tsx";
  const ast = ts.createSourceFile(
    file,
    readFileSync(path.resolve(__dirname, file), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let options: ts.ObjectLiteralExpression | undefined;
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(ast).includes("sendMessage") &&
      node.initializer &&
      ts.isCallExpression(node.initializer) &&
      node.initializer.expression.getText(ast) === "useMutation" &&
      ts.isObjectLiteralExpression(node.initializer.arguments[0])
    )
      options = node.initializer.arguments[0];
    ts.forEachChild(node, visit);
  };
  visit(ast);
  if (!options) throw Error("Missing deployed send mutation");
  const expressions = options.properties.filter(
    (node): node is ts.PropertyAssignment =>
      ts.isPropertyAssignment(node) && node.name.getText(ast) !== "mutationFn",
  );
  const code = ts.transpileModule(
    `return {${expressions.map((node) => node.getText(ast)).join(",")}}`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  const focus = vi.fn(() => composer.textAreaFocus()),
    dismiss = vi.fn(),
    onboarding = vi.fn(),
    invalidate = vi.fn(),
    milestone = vi.fn();
  const callbacks = new Function(
    "captureCompletionGuard",
    "homeScope",
    "chatErrorBanner",
    "useMilestoneStore",
    "dismissChatErrorAfterSuccessfulSubmit",
    "trackOnboardingChatSubmitSuccess",
    "textAreaFocus",
    "queryClient",
    "normalizeChatSubmitResult",
    "isChatSubmitSuccess",
    code,
  )(
    composer.captureCompletionGuard,
    "alice",
    { key: "original-error" },
    { getState: () => ({ markScopedMilestone: milestone }) },
    dismiss,
    onboarding,
    focus,
    { invalidateQueries: invalidate },
    normalizeChatSubmitResult,
    isChatSubmitSuccess,
  );
  return { callbacks, focus, dismiss, onboarding, invalidate, milestone };
};

const mount = () => {
  const textarea = document.createElement("textarea")
  document.body.append(textarea)
  const hook = renderHook(() => {
    const address = useStoreMessageOption((state) => state.chatDraftAddress)
    return useComposerText({
      draftAddress: address ?? undefined,
      draftEnabled: address !== null,
      draftKey: address
        ? conversationDraftKey("completion-test", address)
        : "completion-test",
      textareaRef: { current: textarea },
      tabScopedDraft: true
    })
  })
  return { ...hook, textarea }
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  authority.user = "alice";
  useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true);
  usePlaygroundSessionStore.setState(
    usePlaygroundSessionStore.getInitialState(),
    true,
  );
  useStoreMessageOption.getState().startNewChatDraft();
});

describe.each(["Playground", "Sidepanel"] as const)(
  "%s deployed send completion",
  (surface) => {
    it.each(["submitted", "failed", "skipped"])(
      "does not focus a replacement after a held %s completion",
      async (status) => {
        const view = mount();
        await waitFor(() => expect(view.result.current.draftReady).toBe(true));
        view.textarea.focus();
        const deployed = deployedMutation(surface, view.result.current);
        const context = deployed.callbacks.onMutate?.();
        act(() => useStoreMessageOption.getState().startNewChatDraft());
        const outside = document.createElement("button");
        document.body.append(outside);
        outside.focus();
        deployed.callbacks.onSuccess({ status }, {}, context);
        expect(outside).toHaveFocus();
        expect(deployed.focus).not.toHaveBeenCalled();
        if (surface === "Playground") {
          expect(deployed.dismiss).not.toHaveBeenCalled();
          expect(deployed.onboarding).not.toHaveBeenCalled();
          expect(deployed.invalidate).not.toHaveBeenCalled();
          expect(deployed.milestone).toHaveBeenCalledTimes(
            status === "submitted" ? 1 : 0,
          );
        }
        outside.remove();
        view.textarea.remove();
      },
    );

    it("leaves deliberate outside focus alone in the same chat", async () => {
      const view = mount();
      await waitFor(() => expect(view.result.current.draftReady).toBe(true));
      view.textarea.focus();
      const deployed = deployedMutation(surface, view.result.current),
        context = deployed.callbacks.onMutate?.();
      const outside = document.createElement("button");
      document.body.append(outside);
      outside.focus();
      deployed.callbacks.onSuccess({ status: "submitted" }, {}, context);
      expect(outside).toHaveFocus();
      outside.remove();
      view.textarea.remove();
    });
  },
);

afterEach(() => {
  document.body
    .querySelectorAll("textarea,button")
    .forEach((node) => node.remove());
});

describe("completion source and input", () => {
  it("does not blur a new draft typed into the same mobile textarea", async () => {
    const view = mount();
    await waitFor(() => expect(view.result.current.draftReady).toBe(true));
    view.textarea.focus();
    const source = view.result.current.captureCompletionGuard(),
      focus = view.result.current.captureCompletionGuard(true);
    act(() => view.result.current.setMessageValue("next unsent message"));
    expect(source()).toBe(true);
    expect(focus()).toBe(false);
  });
  it("permits the same source and exactly its own first-ID bind", async () => {
    const view = mount();
    await waitFor(() => expect(view.result.current.draftReady).toBe(true));
    const current = view.result.current.captureCompletionGuard(),
      source = useStoreMessageOption.getState();
    expect(current()).toBe(true);
    act(() =>
      useStoreMessageOption.getState().setChatDraftAddress(
        {
          kind: "native",
          id: "first-result",
          ownerKey: view.result.current.draftOwnerKey!,
          scope: { type: "global" },
        },
        {
          kind: "bind",
          sourceAddress: source.chatDraftAddress!,
          expectedRevision: source.chatDraftRevision,
        },
      ),
    );
    expect(current()).toBe(true);
    act(() => useStoreMessageOption.getState().startNewChatDraft());
    expect(current()).toBe(false);
  });
  it("rejects departing and returning to the original address", async () => {
    const view = mount();
    await waitFor(() => expect(view.result.current.draftReady).toBe(true));
    const current = view.result.current.captureCompletionGuard(),
      original = useStoreMessageOption.getState().chatDraftAddress!;
    act(() => {
      useStoreMessageOption.getState().startNewChatDraft();
      useStoreMessageOption.getState().setChatDraftAddress(original);
    });
    expect(current()).toBe(false);
  });
});


describe("explicit Stop input ownership", () => {
  it("admits only its own accepted untouched optimistic restore", async () => {
    const view = mount()
    await waitFor(() => expect(view.result.current.draftReady).toBe(true))
    act(() =>
      view.result.current.setMessageValue(
        "Exact stopped input.  \nSecond line."
      )
    )
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    const current = view.result.current.captureStopGuard()
    expect(current()).toBe(true)
    act(() => view.result.current.restorePromptAssistAttempt(attempt))
    expect(view.result.current.form.values.message).toBe(
      "Exact stopped input.  \nSecond line."
    )
    expect(current()).toBe(true)
    act(() => view.result.current.setMessageValue("Newer input"))
    expect(current()).toBe(false)
  })

  it.each([
    "text Undo",
    "image Undo",
    "new attempt",
    "empty new attempt",
    "New",
    "ABA",
    "account",
    "unmount"
  ])("does not admit a restore or focus after %s", async (boundary) => {
    const view = mount()
    await waitFor(() => expect(view.result.current.draftReady).toBe(true))
    act(() => view.result.current.setMessageValue("Original stopped input"))
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    const current = view.result.current.captureStopGuard()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    const restore = view.result.current.restorePromptAssistAttempt
    act(() => {
      if (boundary === "text Undo")
        view.result.current.setMessageValue("Edited input")
      else if (boundary === "image Undo")
        view.result.current.form.setFieldValue("image", "new-image")
      else if (boundary === "new attempt") {
        view.result.current.setMessageValue("Another attempt")
        view.result.current.beginPromptAssistReset()
      } else if (boundary === "empty new attempt")
        view.result.current.beginPromptAssistReset()
      else if (boundary === "New" || boundary === "ABA")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "account") {
        authority.user = "bob"
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      } else view.unmount()
    })
    act(() => {
      if (boundary === "text Undo") view.result.current.setMessageValue("")
      else if (boundary === "image Undo")
        view.result.current.form.setFieldValue("image", "")
      else if (boundary === "ABA")
        useStoreMessageOption.getState().setChatDraftAddress(source)
      restore(attempt)
    })
    expect(current()).toBe(false)
  })
})

const compactCleanup = (imageCommand = false) => {
  const file = path.resolve(__dirname, "../../../Sidepanel/Chat/form.tsx");
  const ast = ts.createSourceFile(
    file,
    readFileSync(file, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let callback: ts.Expression | undefined;
  const visit = (node: ts.Node) => {
    if (
      ts.isPropertyAssignment(node) &&
      node.name.getText(ast) === "afterSend" &&
      node.initializer
        .getText(ast)
        .includes("clearDraft(promptAssistAttemptId)")
    )
      callback = node.initializer;
    ts.forEachChild(node, visit);
  };
  visit(ast);
  if (!callback) throw Error("Missing compact completion cleanup");
  const sentFiles = [{ id: "sent-file" }],
    nextFiles = [{ id: "next-file" }],
    sentDocs = [{ id: 1 }],
    nextDocs = [{ id: 2 }];
  let files = nextFiles,
    docs = nextDocs,
    knowledge = true;
  const knowledgeMentionRevision = { current: 2 },
    submittedKnowledgeMentionRevision = 1;
  const mark = vi.fn(),
    clearDraft = vi.fn();
  const bindings = {
    promptAssistAttemptId: 1,
    draftOwnership: { isOwnerCurrent: () => true },
    markPromptAssistAttemptSaved: mark,
    clearDraft,
    restorePromptAssistAttempt: vi.fn(),
    isChatSubmitSuccess,
    normalizeChatSubmitResult,
    clearSelectedDocuments: () => {
      docs = [];
    },
    clearSubmittedDocuments: (expected: typeof docs) => {
      if (docs === expected) docs = [];
    },
    selectedDocuments: sentDocs,
    contextFiles: sentFiles,
    setContextFiles: (
      next: typeof files | ((value: typeof files) => typeof files),
    ) => {
      files = typeof next === "function" ? next(files) : next;
    },
    knowledgeMentionRevision,
    submittedKnowledgeMentionRevision,
    setKnowledgeMentionActive: (next: boolean) => {
      knowledge = next;
    },
    intent: { isImageCommand: imageCommand },
  };
  const code = ts.transpileModule("return " + callback.getText(ast), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const afterSend = new Function(...Object.keys(bindings), code)(
    ...Object.values(bindings),
  );
  return {
    afterSend,
    mark,
    clearDraft,
    files: () => files,
    docs: () => docs,
    knowledge: () => knowledge,
    nextFiles,
    nextDocs,
    untouched: () => {
      files = sentFiles;
      docs = sentDocs;
      knowledgeMentionRevision.current = 1;
    },
  };
};
describe("compact submitted context cleanup", () => {
  it("keeps newly selected files, tabs and knowledge while recording the old acknowledgement", () => {
    const view = compactCleanup();
    view.afterSend({ status: "submitted" });
    expect(view.files()).toBe(view.nextFiles);
    expect(view.docs()).toBe(view.nextDocs);
    expect(view.knowledge()).toBe(true);
    expect(view.mark).toHaveBeenCalledWith(1, true);
    expect(view.clearDraft).toHaveBeenCalledWith(1);
  });
  it("clears unchanged context that was actually submitted", () => {
    const view = compactCleanup();
    view.untouched();
    view.afterSend({ status: "submitted" });
    expect(view.files()).toEqual([]);
    expect(view.docs()).toEqual([]);
    expect(view.knowledge()).toBe(false);
  });
  it("does not clear context excluded from an image command", () => {
    const view = compactCleanup(true);
    view.untouched();
    view.afterSend({ status: "submitted" });
    expect(view.files()).toHaveLength(1);
    expect(view.docs()).toHaveLength(1);
    expect(view.knowledge()).toBe(true);
  });
});

it("keeps current completion through a qualified first bind and actual H1 result follow", async () => {
  const view = renderHook(() => {
    const selection = useHistorySelection({
      onCapture: (c) => {
        const display = formatSelectedHistory(c);
        useStoreMessageOption.getState().setHistory(display.history);
        useStoreMessageOption.getState().setMessages(display.messages);
      },
    });
    const address = useStoreMessageOption((state) => state.chatDraftAddress)!;
    const composer = useComposerText({
      draftAddress: address,
      draftKey: conversationDraftKey("completion-test", address),
      textareaRef: { current: null },
      tabScopedDraft: true,
    });
    return { composer, selection };
  });
  const early = view.result.current.composer.captureCompletionGuard();
  expect(early()).toBe(false);
  await waitFor(() =>
    expect(view.result.current.composer.draftReady).toBe(true),
  );
  expect(early()).toBe(false);
  const current = view.result.current.composer.captureCompletionGuard(),
    deployed = deployedMutation("Playground", view.result.current.composer),
    context = deployed.callbacks.onMutate();
  const revision = useStoreMessageOption.getState().chatDraftRevision;
  await act(async () => {
    await view.result.current.selection.open(
      {
        kind: "local",
        profile_id: "profile",
        owner_key: "completion-owner",
        conversation_id: "first-result",
      },
      null,
      captureChatDraftLoad("bind"),
    );
  });
  expect(useStoreMessageOption.getState().chatDraftRevision).toBe(revision + 1);
  expect(current()).toBe(true);
  const origin = view.result.current.selection.getCurrent().view!;
  await act(async () => {
    await view.result.current.selection.followResult(origin, "a");
  });
  expect(
    view.result.current.selection.getCurrent().view?.selection_revision,
  ).toBeGreaterThan(origin.selection_revision);
  expect(current()).toBe(true);
  deployed.callbacks.onSuccess({ status: "submitted" }, {}, context);
  expect(deployed.dismiss).toHaveBeenCalledOnce();
  expect(deployed.onboarding).toHaveBeenCalledOnce();
  expect(deployed.invalidate).toHaveBeenCalledOnce();
  expect(deployed.milestone).toHaveBeenCalledWith("alice", "first_chat");
});
