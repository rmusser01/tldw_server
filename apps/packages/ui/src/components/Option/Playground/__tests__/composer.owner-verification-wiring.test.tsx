import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { MemoryRouter } from "react-router-dom";
import { Storage } from "@plasmohq/storage";
import { useStoreMessageOption } from "@/store/option";
import { COMPOSER_CONSTANTS } from "@/config/ui-constants";
import {
  chatDraftAddressKey,
  conversationDraftKey,
} from "@/utils/chat-draft-address";
import { ChatDraftRecovery } from "@/components/Chat/composer/ChatDraftRecovery";
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText";
import {
  createSidepanelChatHandoff,
  SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX,
} from "@/services/sidepanel-chat-handoff";
import { useSidepanelChatHandoffImport } from "../hooks/useSidepanelChatHandoffImport";
import {
  useComposerInput,
  type UseComposerInputDeps,
} from "../hooks/useComposerInput";

const accountIO = vi.hoisted(() => ({ resolve: vi.fn() }));
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: (...args: unknown[]) => accountIO.resolve(...args),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }),
}));

// Real adapter/shared composer/owner/form/persistence/registry and recovery UI.
// The resolver is held I/O; collapse controls and unrelated Form UI are omitted.
const readyScope = {
  userId: "alice",
  config: {
    serverUrl: "http://owner.test",
    authMode: "multi-user" as const,
    authSource: "manual" as const,
  },
};
const pendingText = "Checking your account before restoring or saving drafts…";
const errorText =
  "Your account could not be verified. Draft saving, recovery, and imports are paused. Retry or check your connection settings.";
const noop = () => undefined;
function useWebUiComposer(textareaRef: React.RefObject<HTMLTextAreaElement>) {
  return useComposerInput({
    textareaRef,
    isMessageCollapsed: false,
    setIsMessageCollapsed: noop,
    collapsedRange: null,
    setCollapsedRange: noop,
    hasExpandedLargeText: false,
    setHasExpandedLargeText: noop,
    collapseLargeMessage: noop,
    restoreCollapseState: noop,
    getCollapsedDisplayMeta: () => ({}),
    getDisplayCaretFromMessage: (value) => value,
    getMessageCaretFromDisplay: (value) => value,
    normalizeCollapsedRange: (value) => value,
    expandLargeMessage: noop,
    pendingCaretRef: React.useRef<number | null>(null),
    lastDisplaySelectionRef: React.useRef<{
      start: number;
      end: number;
    } | null>(null),
    pendingCollapsedStateRef:
      React.useRef<UseComposerInputDeps["pendingCollapsedStateRef"]["current"]>(
        null,
      ),
    pointerDownRef: React.useRef(false),
    selectionFromPointerRef: React.useRef(false),
    tabMentionsEnabled: false,
    handleTextChange: noop,
    isProMode: false,
  });
}

type Composer = ReturnType<typeof useComposerText> & {
  draftOwnerVerificationState?: "pending" | "ready" | "error";
  retryDraftOwnerVerification?: () => void;
};
type Surface = "WebUI" | "Sidepanel";
function extractSurface(relative: string, surface: Surface) {
  const source = readFileSync(path.resolve(__dirname, relative), "utf8");
  const ast = ts.createSourceFile(
    relative,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let binding: ts.ObjectBindingPattern | undefined;
  let sharedCall: ts.CallExpression | undefined;
  let recovery: ts.JsxSelfClosingElement | undefined;
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isObjectBindingPattern(node.name)
    ) {
      if (
        surface === "WebUI" &&
        node.initializer?.getText(ast) === "composerInput"
      )
        binding = node.name;
      if (
        surface === "Sidepanel" &&
        node.initializer &&
        ts.isCallExpression(node.initializer) &&
        node.initializer.expression.getText(ast) === "useComposerText"
      ) {
        binding = node.name;
        sharedCall = node.initializer;
      }
    }
    if (
      ts.isJsxSelfClosingElement(node) &&
      node.tagName.getText(ast) === "ChatDraftRecovery"
    )
      recovery = node;
    ts.forEachChild(node, visit);
  };
  visit(ast);
  if (!binding || !recovery)
    throw new Error(
      "Missing deployed " + surface + " composer/recovery boundary",
    );
  const recoveryCode = ts.transpileModule(
    `function DeployedRecovery({ composer }) {
    const ${binding.getText(ast)} = composer;
    const onFindConversations = () => {};
    const restoreRetainedDraft = () => false;
    const onRecoverDraft = () => false;
    return ${recovery.getText(ast)};
  }`,
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        jsx: ts.JsxEmit.React,
      },
    },
  ).outputText;
  const Recovery = new Function(
    "React",
    "ChatDraftRecovery",
    recoveryCode + "; return DeployedRecovery;",
  )(React, ChatDraftRecovery) as React.ComponentType<{ composer: Composer }>;
  if (surface === "WebUI") return { Recovery, useComposer: useWebUiComposer };
  if (!sharedCall)
    throw new Error("Missing actual Sidepanel shared composer call");
  const hookCode = ts.transpileModule(
    `function useDeployedComposer(textareaRef) {
    const storageKey = 'tldw:sidepanelChatDraft';
    const draftAddress = useStoreMessageOption(state => state.chatDraftAddress);
    const draftTransition = useStoreMessageOption(state => state.chatDraftTransition);
    const isProMode = false;
    return ${sharedCall.getText(ast)};
  }`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  const useComposer = new Function(
    "useComposerText",
    "useStoreMessageOption",
    "conversationDraftKey",
    "chatDraftAddressKey",
    hookCode + "; return useDeployedComposer;",
  )(
    useComposerText,
    useStoreMessageOption,
    conversationDraftKey,
    chatDraftAddressKey,
  ) as (
    textareaRef: React.RefObject<HTMLTextAreaElement>,
  ) => ReturnType<typeof useComposerText>;
  return { Recovery, useComposer };
}
const deployed = {
  WebUI: extractSurface("../PlaygroundForm.tsx", "WebUI"),
  Sidepanel: extractSurface("../../../Sidepanel/Chat/form.tsx", "Sidepanel"),
};

function CopyReview({ composer }: { composer: Composer }) {
  const review = useSidepanelChatHandoffImport({
    ready: composer.draftReady,
    destinationKey: composer.draftOwnerKey ?? "unresolved",
    captureGuard: composer.captureCompletionGuard,
    draftValue: composer.form.values.message,
    getDraftRevision: composer.form.getRevision,
    setMessageValue: composer.setMessageValue,
    notificationApi: { warning: noop },
    t: (_key, fallback) =>
      typeof fallback === "string" ? fallback : fallback.defaultValue,
  });
  return (
    <output aria-label="Owned copied draft review">
      {review.conflict?.handoff.draft.text ?? ""}
    </output>
  );
}
function SurfaceHarness({ surface }: { surface: Surface }) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const { Recovery, useComposer } = deployed[surface];
  const composer = useComposer(textareaRef) as Composer;
  return (
    <>
      <textarea
        ref={textareaRef}
        aria-label="Current draft"
        {...composer.form.getInputProps("message")}
      />
      <Recovery composer={composer} />
      {surface === "WebUI" && <CopyReview composer={composer} />}
      <output aria-label="Draft saved">{String(composer.draftSaved)}</output>
    </>
  );
}
const ownerRecords = () =>
  Array.from(
    { length: localStorage.length },
    (_, index) => localStorage.key(index)!,
  ).filter(
    (key) =>
      key.startsWith("registry:draft:") &&
      key.includes(":owner:") &&
      !key.includes(":owner:unresolved"),
  );
const settle = async () => {
  await act(async () => {
    await vi.dynamicImportSettled();
  });
  await act(async () => {
    await vi.advanceTimersByTimeAsync(
      COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1,
    );
  });
};

describe("actual composer owner verification adapters and deployed recovery boundaries", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    sessionStorage.clear();
    useStoreMessageOption.getState().startNewChatDraft(false);
    accountIO.resolve.mockReset().mockResolvedValue(readyScope);
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("exposes pending account verification from the actual WebUI adapter while draft authority is unavailable", async () => {
    accountIO.resolve.mockImplementation(() => new Promise(() => {}));
    const { result } = renderHook(() =>
      useWebUiComposer(React.useRef<HTMLTextAreaElement>(null)),
    );
    await settle();
    expect((result.current as Composer).draftOwnerVerificationState).toBe(
      "pending",
    );
    expect(result.current.draftReady).toBe(false);
    expect(ownerRecords()).toEqual([]);
  });

  it("exposes a usable same-config Retry from the actual WebUI adapter and preserves the unsaved exact draft", async () => {
    accountIO.resolve.mockRejectedValueOnce(
      new TypeError("Account request offline"),
    );
    const { result } = renderHook(() =>
      useWebUiComposer(React.useRef<HTMLTextAreaElement>(null)),
    );
    await settle();
    act(() => result.current.setMessageValue("UNSAVED\n private text  "));
    expect((result.current as Composer).draftOwnerVerificationState).toBe(
      "error",
    );
    expect((result.current as Composer).retryDraftOwnerVerification).toEqual(
      expect.any(Function),
    );
    expect(result.current.draftReady).toBe(false);
    expect(ownerRecords()).toEqual([]);
    act(() => (result.current as Composer).retryDraftOwnerVerification!());
    await settle();
    expect((result.current as Composer).draftOwnerVerificationState).toBe(
      "ready",
    );
    expect(result.current.draftReady).toBe(true);
    expect(result.current.form.values.message).toBe("UNSAVED\n private text  ");
    expect(ownerRecords()).toHaveLength(1);
    expect(localStorage.getItem(ownerRecords()[0])).toContain("private text  ");
  });

  it("keeps current verified draft input and actual scoped persistence usable", async () => {
    const { result } = renderHook(() =>
      useWebUiComposer(React.useRef<HTMLTextAreaElement>(null)),
    );
    await settle();
    expect(result.current.draftReady).toBe(true);
    act(() => result.current.setMessageValue("CURRENT READY DRAFT"));
    await settle();
    expect(result.current.draftSaved).toBe(true);
    expect(ownerRecords()).toHaveLength(1);
    expect(localStorage.getItem(ownerRecords()[0])).toContain(
      "CURRENT READY DRAFT",
    );
  });

  it.each(["WebUI", "Sidepanel"] as const)(
    "%s deployed recovery JSX exposes unknown-owner status and Retry despite disabled recovery",
    async (surface) => {
      let rejectInitial!: (error: Error) => void;
      accountIO.resolve.mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectInitial = reject;
          }),
      );
      const handoff =
        surface === "WebUI"
          ? await createSidepanelChatHandoff({
              draftText: "COPIED DRAFT",
              routeIntent: { path: "/chat" },
            })
          : null;
      const read = vi.spyOn(Storage.prototype, "get");
      render(
        <MemoryRouter
          initialEntries={[handoff ? "/chat?handoff=" + handoff.id : "/chat"]}
        >
          <SurfaceHarness surface={surface} />
        </MemoryRouter>,
      );
      expect(screen.getByText(pendingText)).toHaveAttribute("role", "status");
      expect(
        screen.getByRole("button", { name: "Recover drafts" }),
      ).toBeDisabled();
      fireEvent.change(screen.getByRole("textbox", { name: "Current draft" }), {
        target: { value: "MY EXACT\n QUESTION  " },
      });
      await act(async () =>
        rejectInitial(new TypeError("Account request offline")),
      );
      await settle();
      expect(screen.getByText(errorText)).toHaveAttribute("role", "status");
      const retry = screen.getByRole("button", {
        name: "Retry account verification",
      });
      expect(retry).toBeEnabled();
      expect(
        screen.getByRole("button", { name: "Recover drafts" }),
      ).toBeDisabled();
      expect(
        screen.getByRole("textbox", { name: "Current draft" }),
      ).toHaveValue("MY EXACT\n QUESTION  ");
      expect(ownerRecords()).toEqual([]);
      if (handoff) {
        expect(read).not.toHaveBeenCalledWith(
          SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX + handoff.id,
        );
        expect(
          screen.getByLabelText("Owned copied draft review"),
        ).toBeEmptyDOMElement();
      }
      fireEvent.click(retry);
      await settle();
      expect(accountIO.resolve).toHaveBeenCalledTimes(2);
      expect(screen.queryByText(pendingText)).not.toBeInTheDocument();
      expect(screen.queryByText(errorText)).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Recover drafts" }),
      ).toBeEnabled();
      expect(
        screen.getByRole("textbox", { name: "Current draft" }),
      ).toHaveValue("MY EXACT\n QUESTION  ");
      expect(screen.getByLabelText("Draft saved")).toHaveTextContent("true");
      expect(ownerRecords()).toHaveLength(1);
      if (handoff) {
        expect(read).toHaveBeenCalledWith(
          SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX + handoff.id,
        );
        expect(
          screen.getByLabelText("Owned copied draft review"),
        ).toHaveTextContent("COPIED DRAFT");
        expect(
          localStorage.getItem(
            SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX + handoff.id,
          ),
        ).toContain(handoff.id);
      }
    },
  );

  it.each(["WebUI", "Sidepanel"] as const)(
    "%s deployed recovery JSX avoids a false account outage during a verified transient recheck",
    async (surface) => {
      render(
        <MemoryRouter>
          <SurfaceHarness surface={surface} />
        </MemoryRouter>,
      );
      await settle();
      fireEvent.change(screen.getByRole("textbox", { name: "Current draft" }), {
        target: { value: "KNOWN OWNER DRAFT" },
      });
      accountIO.resolve.mockRejectedValueOnce(
        new TypeError("Transient network failure"),
      );
      act(() => window.dispatchEvent(new Event("focus")));
      await settle();
      expect(screen.queryByText(errorText)).not.toBeInTheDocument();
      expect(screen.queryByText(pendingText)).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Retry account verification" }),
      ).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Recover drafts" }),
      ).toBeEnabled();
      expect(
        screen.getByRole("textbox", { name: "Current draft" }),
      ).toHaveValue("KNOWN OWNER DRAFT");
      expect(ownerRecords()).toHaveLength(1);
    },
  );
});
