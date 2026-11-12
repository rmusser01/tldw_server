import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  waitFor
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { useStoreMessageOption } from "@/store/option";
import { usePlaygroundSessionStore } from "@/store/playground-session";
import { useMessageCollapse } from "@/hooks/playground/useMessageCollapse";
import { useComposerInput } from "@/components/Option/Playground/hooks/useComposerInput";
import { QuickPromptInsertionReview } from "@/components/Chat/composer/QuickPromptInsertionReview";
import { PASTED_TEXT_CHAR_LIMIT } from "@/utils/constant";

const accountIO = vi.hoisted(() => ({ resolve: vi.fn() }));
const readyScope = {
  userId: "alice",
  config: {
    serverUrl: "http://owner.test",
    authMode: "multi-user" as const,
    authSource: "manual" as const
  }
};
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: (...args: unknown[]) => accountIO.resolve(...args)
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}));

// Actual collapse, adapter, form, owner, store, review, and native textarea.
// Only account I/O and animation frames are controlled. The deployed Form JSX
// is executed below; its H1 boundary uses the actual composer's source guard,
// not the full Form/history/backend. This is a DOM contract, not browser UAT.
const formPath = path.resolve(
  __dirname,
  "../../../components/Option/Playground/PlaygroundForm.tsx"
);
const source = readFileSync(formPath, "utf8");
const ast = ts.createSourceFile(
  formPath,
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
);
let binding: ts.JsxSelfClosingElement | undefined;
const visit = (node: ts.Node) => {
  if (
    ts.isJsxSelfClosingElement(node) &&
    node.tagName.getText(ast) === "QuickPromptInsertionReview"
  )
    binding = node;
  ts.forEachChild(node, visit);
};
visit(ast);
if (!binding) throw new Error("Missing deployed quick-prompt review binding");
const code = ts.transpileModule(
  `function DeployedReview({ composer, textareaRef }) {
    const { form, draftReady, draftOwnerKey, setMessageValue, restoreMessageValue, captureCompletionGuard } = composer;
    const handoffDestinationKey = draftOwnerKey ?? 'unresolved';
    const captureHandoffGuard = captureCompletionGuard;
    return ${binding.getText(ast)};
  }`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React } }
).outputText;
type Composer = ReturnType<typeof useComposerInput>;
type Collapse = ReturnType<typeof useMessageCollapse>;
const DeployedReview = new Function(
  "React",
  "QuickPromptInsertionReview",
  code + "; return DeployedReview;"
)(React, QuickPromptInsertionReview) as React.ComponentType<{
  composer: Composer;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
}>;
const noop = () => undefined;
function Harness({
  expose
}: {
  expose: (composer: Composer, collapse: Collapse) => void;
}) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const collapse = useMessageCollapse();
  const composer = useComposerInput({
    ...collapse,
    textareaRef,
    restoreCollapseState: collapse.restoreMessageValue,
    tabMentionsEnabled: false,
    handleTextChange: noop,
    isProMode: false
  });
  React.useLayoutEffect(() => expose(composer, collapse));
  return (
    <>
      <textarea
        ref={composer.attachTextarea}
        aria-label="Current draft"
        value={composer.messageDisplayValue}
        onChange={composer.handleTextareaChange}
        onSelect={composer.handleTextareaSelect}
      />
      <DeployedReview composer={composer} textareaRef={textareaRef} />
      <output aria-label="Draft ready">{String(composer.draftReady)}</output>
      <button type="button">Other control</button>
    </>
  );
}
const longDraft = "Unfinished roleplay context. ".repeat(80);
const longPrompt =
  "Use {first} as background. " +
  "Roleplay context. ".repeat(100) +
  "Explore {topic}";
const shortPrompt = "Explain {topic}";
let nextFrame = 0;
const frames = new Map<number, FrameRequestCallback>();
const flushFrames = () => {
  act(() => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => callback(1));
  });
};
const mount = async (verification: "ready" | "error" | "pending" = "ready") => {
  let composer!: Composer;
  let collapse!: Collapse;
  const view = render(
    <Harness
      expose={(current, currentCollapse) => {
        composer = current;
        collapse = currentCollapse;
      }}
    />
  );
  await waitFor(() =>
    expect(composer?.draftOwnerVerificationState).toBe(verification)
  );
  return {
    ...view,
    textarea: view.getByLabelText("Current draft") as HTMLTextAreaElement,
    composer: () => composer,
    collapse: () => collapse
  };
};
const review = (
  view: Awaited<ReturnType<typeof mount>>,
  prompt: string,
  choice = "Overwrite message"
) => {
  act(() => useStoreMessageOption.getState().setSelectedQuickPrompt(prompt));
  const button = view.getByRole("button", { name: choice });
  button.focus();
  fireEvent.click(button);
};
const selected = (textarea: HTMLTextAreaElement) =>
  textarea.value.slice(textarea.selectionStart, textarea.selectionEnd);

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  accountIO.resolve.mockReset().mockResolvedValue(readyScope);
  useStoreMessageOption.setState({
    selectedQuickPrompt: null,
    quickPromptRevision: 0
  });
  useStoreMessageOption.getState().startNewChatDraft(false);
  frames.clear();
  nextFrame = 0;
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback);
    return nextFrame;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  frames.clear();
});

describe("accepted prompt visibility at the actual WebUI collapse binding", () => {
  it("reveals long Overwrite text and selects the existing last variable for editing", async () => {
    const view = await mount();
    review(view, longPrompt);
    expect(view.textarea.value).toBe(longPrompt);
    expect(selected(view.textarea)).toBe("{topic}");
    const { selectionStart: start, selectionEnd: end } = view.textarea;
    fireEvent.change(view.textarea, {
      target: {
        value: longPrompt.slice(0, start) + "a forest" + longPrompt.slice(end)
      }
    });
    expect(view.composer().form.getValues().message).toBe(
      longPrompt.replace("{topic}", "a forest")
    );
  });
  it("reveals a collapsed source when appending and selects at its canonical offset", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    expect(view.textarea.value).not.toBe(longDraft);
    review(view, shortPrompt, "Append");
    expect(view.textarea.value).toBe(longDraft + "\n\n" + shortPrompt);
    expect([view.textarea.selectionStart, selected(view.textarea)]).toEqual([
      longDraft.length + 2 + shortPrompt.indexOf("{topic}"),
      "{topic}"
    ]);
  });
  it("reveals long prompts without variables as editable text", async () => {
    const view = await mount();
    review(view, longDraft);
    expect([view.textarea.value, document.activeElement]).toEqual([
      longDraft,
      view.textarea
    ]);
  });
  it("preserves short variable selection", async () => {
    const view = await mount();
    review(view, shortPrompt);
    expect([view.textarea.value, selected(view.textarea)]).toEqual([
      shortPrompt,
      "{topic}"
    ]);
  });
  it("keeps ordinary large-paste collapse and Cancel intact", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    const display = view.textarea.value;
    expect(longDraft.length).toBeGreaterThan(PASTED_TEXT_CHAR_LIMIT);
    expect(display).not.toBe(longDraft);
    review(view, longPrompt, "Cancel");
    expect([
      view.composer().form.getValues().message,
      view.textarea.value,
      view.collapse().isMessageCollapsed
    ]).toEqual([longDraft, display, true]);
  });
  it("retains accepted variable selection after an older collapsed-caret frame", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    expect(frames.size).toBeGreaterThan(0);
    review(view, shortPrompt);
    expect(selected(view.textarea)).toBe("{topic}");
    flushFrames();
    expect(selected(view.textarea)).toBe("{topic}");
  });
  it("retains accepted variable selection after an older short-edit caret frame", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().commitCollapsedEdit("A short edit", 2, null));
    review(view, shortPrompt);
    expect(selected(view.textarea)).toBe("{topic}");
    flushFrames();
    expect(selected(view.textarea)).toBe("{topic}");
  });
  it("retains accepted variable selection after an older expansion frame", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    view.textarea.setSelectionRange(1, 1);
    act(() => {
      view.composer().handleTextareaMouseDown();
      view.composer().handleTextareaSelect();
    });
    review(view, shortPrompt);
    expect(selected(view.textarea)).toBe("{topic}");
    flushFrames();
    expect(selected(view.textarea)).toBe("{topic}");
  });
  it("retires an older caret even when accepting identical canonical text", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().commitCollapsedEdit(shortPrompt, 2, null));
    const held = [...frames.values()];
    const revision = view.composer().form.getRevision();
    review(view, shortPrompt);
    expect(view.composer().form.getRevision()).toBe(revision);
    expect(selected(view.textarea)).toBe("{topic}");
    act(() => held.forEach((callback) => callback(1)));
    expect(selected(view.textarea)).toBe("{topic}");
  });
  it("keeps current collapsed edits usable after their own form write", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    const next = "Before " + longDraft;
    act(() =>
      view
        .composer()
        .commitCollapsedEdit(next, 7, { start: 7, end: next.length })
    );
    flushFrames();
    expect([
      view.composer().form.getValues().message,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual([next, 7, 7]);
  });
  it("keeps a current short edit's caret usable after its own form write", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().commitCollapsedEdit("A short edit", 2, null));
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual(["A short edit", 2, 2]);
  });
  it("expands a current pointer selection at the canonical block boundary", async () => {
    const view = await mount();
    const value = "Before " + longDraft + " After";
    const end = 7 + longDraft.length;
    act(() =>
      view.composer().setMessageValue(value, {
        collapseLarge: true,
        collapsedRange: { start: 7, end }
      })
    );
    view.textarea.focus();
    view.textarea.setSelectionRange(8, 8);
    act(() => {
      view.composer().handleTextareaMouseDown();
      view.composer().handleTextareaSelect();
    });
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd,
      document.activeElement
    ]).toEqual([value, end, end, view.textarea]);
  });
  it("does not steal focus when the user moves to another control", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    const other = view.getByRole("button", { name: "Other control" });
    other.focus();
    const selection = [
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ];
    flushFrames();
    expect([
      document.activeElement,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual([other, ...selection]);
  });
  it("does not let an unrelated collapsed change focus an inactive textarea", async () => {
    const view = await mount();
    const other = view.getByRole("button", { name: "Other control" });
    other.focus();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    flushFrames();
    expect(document.activeElement).toBe(other);
  });
  it("retires a held frame after New chooses a different source", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    const held = [...frames.values()];
    act(() => useStoreMessageOption.getState().startNewChatDraft(false));
    const selection = vi.spyOn(view.textarea, "setSelectionRange");
    act(() => held.forEach((callback) => callback(1)));
    expect(selection).not.toHaveBeenCalled();
  });
  it("retires a held frame after an input edit and Undo to identical text", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    const held = [...frames.values()];
    act(() => {
      view.composer().form.setFieldValue("message", "Temporary edit");
      view.composer().form.setFieldValue("message", longDraft);
    });
    const selection = vi.spyOn(view.textarea, "setSelectionRange");
    act(() => held.forEach((callback) => callback(1)));
    expect(selection).not.toHaveBeenCalled();
  });
  it("retires held callbacks on unmount before touching the old node", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    const held = [...frames.values()];
    view.unmount();
    const selection = vi.spyOn(view.textarea, "setSelectionRange");
    act(() => held.forEach((callback) => callback(1)));
    expect(selection).not.toHaveBeenCalled();
  });
  it("does not let a retained old callback cancel a newer input caret", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    const oldSync = view.composer().syncCollapsedCaret;
    view.textarea.focus();
    const next = "Before " + longDraft;
    act(() =>
      view.composer().setMessageValue(next, {
        collapseLarge: true,
        collapsedRange: { start: 7, end: next.length }
      })
    );
    act(() => view.composer().syncCollapsedCaret({ caret: 7 }));
    act(() => oldSync({ caret: 0 }));
    flushFrames();
    expect([view.textarea.selectionStart, view.textarea.selectionEnd]).toEqual([
      7, 7
    ]);
  });
  it("does not let an old collapsed callback cancel expansion of unchanged text", async () => {
    const view = await mount();
    const value = "Before " + longDraft + " After";
    const end = 7 + longDraft.length;
    act(() =>
      view.composer().setMessageValue(value, {
        collapseLarge: true,
        collapsedRange: { start: 7, end }
      })
    );
    const oldSync = view.composer().syncCollapsedCaret;
    view.textarea.focus();
    view.textarea.setSelectionRange(8, 8);
    act(() => {
      view.composer().handleTextareaMouseDown();
      view.composer().handleTextareaSelect();
    });
    act(() => oldSync({ caret: 0 }));
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual([value, end, end]);
  });
  it.each(["collapsed", "expanded"] as const)(
    "preserves a newer highlighted range while an older %s caret frame is held",
    async (display) => {
      const view = await mount();
      act(() =>
        view.composer().setMessageValue(longDraft, { collapseLarge: true })
      );
      view.textarea.focus();
      if (display === "collapsed") {
        act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
      } else {
        view.textarea.setSelectionRange(1, 1);
        act(() => {
          view.composer().handleTextareaMouseDown();
          view.composer().handleTextareaSelect();
        });
      }
      const held = [...frames.values()];
      view.textarea.setSelectionRange(2, 6);
      act(() => view.composer().handleTextareaSelect());
      act(() => held.forEach((callback) => callback(1)));
      expect([
        view.textarea.selectionStart,
        view.textarea.selectionEnd
      ]).toEqual([2, 6]);
    }
  );
  it("keeps local short editing usable before account verification", async () => {
    accountIO.resolve.mockRejectedValue(new Error("offline"));
    const view = await mount("error");
    expect(view.composer().draftReady).toBe(false);
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().commitCollapsedEdit("A short edit", 2, null));
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd,
      view.composer().draftSaved
    ]).toEqual(["A short edit", 2, 2, false]);
  });
  it("keeps local pointer expansion usable before account verification", async () => {
    accountIO.resolve.mockRejectedValue(new Error("offline"));
    const view = await mount("error");
    const value = "Before " + longDraft + " After";
    const end = 7 + longDraft.length;
    act(() =>
      view.composer().setMessageValue(value, {
        collapseLarge: true,
        collapsedRange: { start: 7, end }
      })
    );
    view.textarea.focus();
    view.textarea.setSelectionRange(8, 8);
    act(() => {
      view.composer().handleTextareaMouseDown();
      view.composer().handleTextareaSelect();
    });
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd,
      view.composer().draftReady
    ]).toEqual([value, end, end, false]);
  });
  it("keeps a local postwrite caret usable through initial account verification", async () => {
    let verify!: (scope: typeof readyScope) => void;
    accountIO.resolve.mockReturnValue(
      new Promise((resolve) => {
        verify = resolve;
      })
    );
    const view = await mount("pending");
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().commitCollapsedEdit("A short edit", 2, null));
    await act(async () => {
      verify(readyScope);
      await Promise.resolve();
    });
    await waitFor(() => expect(view.composer().draftReady).toBe(true));
    flushFrames();
    expect([
      view.textarea.value,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual(["A short edit", 2, 2]);
  });
  it.each(["selection", "restore"] as const)(
    "retires a held caret at the existing session %s boundary before React commits",
    async (boundary) => {
      const view = await mount();
      act(() =>
        view.composer().setMessageValue(longDraft, { collapseLarge: true })
      );
      view.textarea.focus();
      act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
      const held = [...frames.values()];
      const selection = vi.spyOn(view.textarea, "setSelectionRange");
      act(() => {
        const session = usePlaygroundSessionStore.getState();
        if (boundary === "selection") session.markSourceSelectionIntent();
        else session.cancelPendingRestore();
        held.forEach((callback) => callback(1));
      });
      expect(selection).not.toHaveBeenCalled();
    }
  );
  it("retires a held caret on the actual account event even if text is restored", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: 0 }));
    const held = [...frames.values()];
    const revision = usePlaygroundSessionStore.getState().restoreRevision;
    const selection = vi.spyOn(view.textarea, "setSelectionRange");
    act(() => {
      window.dispatchEvent(new Event("tldw:auth-principal-changed"));
      view.composer().form.setFieldValue("message", longDraft);
      held.forEach((callback) => callback(1));
    });
    expect(
      usePlaygroundSessionStore.getState().restoreRevision
    ).toBeGreaterThan(revision);
    expect(selection).not.toHaveBeenCalled();
  });
  it("keeps a current collapsed caret update usable", async () => {
    const view = await mount();
    act(() =>
      view.composer().setMessageValue(longDraft, { collapseLarge: true })
    );
    view.textarea.focus();
    act(() => view.composer().syncCollapsedCaret({ caret: longDraft.length }));
    flushFrames();
    expect([
      document.activeElement,
      view.textarea.selectionStart,
      view.textarea.selectionEnd
    ]).toEqual([
      view.textarea,
      view.textarea.value.length,
      view.textarea.value.length
    ]);
  });
});
