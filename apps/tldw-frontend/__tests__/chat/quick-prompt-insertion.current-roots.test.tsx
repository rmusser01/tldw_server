// This root fixture explicitly exercises the WebUI storage hook alias.
// @vitest-environment jsdom
import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStorage } from "@web/extension/shims/plasmo-storage-hook";
import { useSimpleForm } from "@/hooks/useSimpleForm";
import { useStoreMessageOption } from "@/store/option";
import {
  QuickPromptInsertionReview,
  useQuickPromptInsertion,
} from "@/components/Chat/composer/QuickPromptInsertionReview";
import { usePromptPersistence } from "@/hooks/chat/usePromptPersistence";

// The initial red log ran the exact extracted legacy Form effects. After
// replacement, these same draft/replay assertions run the actual shared hook,
// actual form/store and WebUI storage shim. Only translation copy and the
// account/H1 guard boundary are doubled; New uses the real draft transition.
// Surface rows exercise their common hook contract, not full deployed Forms or
// native extension/browser I/O. Parent integration suites cover Form wiring.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }),
}));
type Surface = "webui" | "sidepanel";
type Form = ReturnType<typeof useSimpleForm<{ message: string }>>;
const setQuick = useStoreMessageOption.getState().setSelectedQuickPrompt;
const command = "Explain {topic}";
const plainCommand = "Offer three possibilities.";
const nodes = new Set<HTMLElement>();
const strict = ({ children }: { children: React.ReactNode }) => (
  <React.StrictMode>{children}</React.StrictMode>
);
const mountInsertion = (
  _surface: Surface,
  initialDraft = "",
  options: {
    ready?: boolean;
    isOwnerCurrent?: () => boolean;
    beforeSelection?: (form: Form) => void;
    strict?: boolean;
  } = {},
) => {
  const textarea = document.createElement("textarea");
  document.body.append(textarea);
  nodes.add(textarea);
  const textareaRef = { current: textarea };
  const view = renderHook(
    ({ ready }) => {
      const form = useSimpleForm({ initialValues: { message: initialDraft } });
      const sourceRevision = useStoreMessageOption(
        (state) => state.chatDraftRevision,
      );
      const sourceAddress = useStoreMessageOption(
        (state) => state.chatDraftAddress,
      );
      const { setFieldValue } = form;
      const setMessageValue = React.useCallback(
        (value: string) => setFieldValue("message", value),
        [setFieldValue],
      );
      const captureGuard = React.useCallback(
        () => () => {
          const current = useStoreMessageOption.getState();
          return (
            current.chatDraftRevision === sourceRevision &&
            current.chatDraftAddress === sourceAddress &&
            (options.isOwnerCurrent?.() ?? true)
          );
        },
        [sourceRevision, sourceAddress],
      );
      React.useLayoutEffect(() => {
        textarea.value = form.values.message;
        options.beforeSelection?.(form);
      });
      const insertion = useQuickPromptInsertion({
        form,
        setMessageValue,
        ready,
        destinationKey: JSON.stringify([sourceAddress, sourceRevision]),
        captureGuard,
        textareaRef,
      });
      return { form, sourceRevision, insertion };
    },
    {
      initialProps: { ready: options.ready ?? true },
      wrapper: options.strict ? strict : undefined,
    },
  );
  return { ...view, textarea };
};
const mountPersistence = () =>
  renderHook(() => {
    const selectedSystemPrompt = useStoreMessageOption(
      (state) => state.selectedSystemPrompt,
    );
    usePromptPersistence({
      selectedSystemPrompt,
      setSelectedSystemPrompt:
        useStoreMessageOption.getState().setSelectedSystemPrompt,
    });
    return {
      system: useStorage<string | null>("selectedSystemPrompt", null),
      quick: useStorage<string | null>("selectedQuickPrompt", null),
    };
  });
const ready = async (view: ReturnType<typeof mountPersistence>) => {
  await waitFor(() =>
    expect([
      view.result.current.system[2].isLoading,
      view.result.current.quick[2].isLoading,
    ]).toEqual([false, false]),
  );
};
beforeEach(() => {
  window.localStorage.removeItem("selectedQuickPrompt");
  window.localStorage.removeItem("selectedSystemPrompt");
  useStoreMessageOption.setState({
    selectedQuickPrompt: null,
    selectedSystemPrompt: null,
    quickPromptRevision: 0,
  });
  useStoreMessageOption.getState().startNewChatDraft(false);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  nodes.forEach((node) => node.remove());
  nodes.clear();
  window.localStorage.removeItem("selectedQuickPrompt");
  window.localStorage.removeItem("selectedSystemPrompt");
});

describe("quick-prompt persistence root regressions", () => {
  it("does not replay a previously saved insertion command on hydration", async () => {
    window.localStorage.setItem("selectedQuickPrompt", JSON.stringify(command));
    const view = mountPersistence();
    await ready(view);
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
  });
  it("keeps a newly selected insertion command out of durable storage", async () => {
    const view = mountPersistence();
    await ready(view);
    await act(async () => {
      setQuick(command);
      await Promise.resolve();
    });
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(command);
    expect(window.localStorage.getItem("selectedQuickPrompt")).toBeNull();
  });
  it("preserves durable system-prompt selection and subsequent changes", async () => {
    window.localStorage.setItem(
      "selectedSystemPrompt",
      JSON.stringify("saved-system-prompt-id"),
    );
    const view = mountPersistence();
    await ready(view);
    expect(useStoreMessageOption.getState().selectedSystemPrompt).toBe(
      "saved-system-prompt-id",
    );
    await act(async () => {
      useStoreMessageOption
        .getState()
        .setSelectedSystemPrompt("another-system-prompt-id");
      await Promise.resolve();
    });
    expect(window.localStorage.getItem("selectedSystemPrompt")).toBe(
      JSON.stringify("another-system-prompt-id"),
    );
  });
});

describe("owned quick-prompt insertion root regressions", () => {
  it.each<Surface>(["webui", "sidepanel"])(
    "%s preserves a nonempty draft until an explicit insertion choice",
    (surface) => {
      const view = mountInsertion(surface, "  My unfinished question  ");
      act(() => setQuick(plainCommand));
      expect(view.result.current.insertion.promptReview).toBe(plainCommand);
      expect(view.result.current.form.getValues().message).toBe(
        "  My unfinished question  ",
      );
    },
  );
  it("does not let a retained WebUI overwrite choice replace an edited draft", () => {
    const view = mountInsertion("webui", "Original question");
    act(() => setQuick(plainCommand));
    const oldChoice = view.result.current.insertion.overwrite;
    act(() =>
      view.result.current.form.setFieldValue("message", "New user intent"),
    );
    act(oldChoice);
    expect(view.result.current.form.getValues().message).toBe(
      "New user intent",
    );
  });
  it("does not let a retained WebUI append choice copy the previous source draft into New", () => {
    const view = mountInsertion("webui", "Previous source question");
    act(() => setQuick(plainCommand));
    const oldChoice = view.result.current.insertion.append;
    const previousRevision = view.result.current.sourceRevision;
    act(() => {
      useStoreMessageOption.getState().startNewChatDraft(false);
      view.result.current.form.setFieldValue(
        "message",
        "New conversation text",
      );
    });
    expect(view.result.current.sourceRevision).toBeGreaterThan(
      previousRevision,
    );
    act(oldChoice);
    expect(view.result.current.form.getValues().message).toBe(
      "New conversation text",
    );
  });
  it("retires a retained WebUI choice after an edit and Undo to identical text", () => {
    const view = mountInsertion("webui", "Original question");
    act(() => setQuick(plainCommand));
    const oldChoice = view.result.current.insertion.overwrite;
    const revision = view.result.current.form.getRevision();
    act(() => {
      view.result.current.form.setFieldValue("message", "Temporary edit");
      view.result.current.form.setFieldValue("message", "Original question");
    });
    expect(view.result.current.form.getRevision()).toBeGreaterThan(revision);
    act(oldChoice);
    expect(view.result.current.form.getValues().message).toBe(
      "Original question",
    );
  });
  it.each<Surface>(["webui", "sidepanel"])(
    "%s retires variable selection and command clearing after source/input replacement",
    (surface) => {
      vi.useFakeTimers({
        toFake: ["setTimeout", "clearTimeout", "clearInterval"],
      });
      let retireOnCommit = false;
      const view = mountInsertion(surface, "", {
        beforeSelection: (form) => {
          if (!retireOnCommit || form.values.message !== command) return;
          retireOnCommit = false;
          useStoreMessageOption.getState().startNewChatDraft(false);
          form.setFieldValue("message", "Unrelated new input");
          setQuick("A newer command");
        },
      });
      const selection = vi.spyOn(view.textarea, "setSelectionRange");
      act(() => setQuick(command));
      expect(view.result.current.insertion.promptReview).toBe(command);
      retireOnCommit = true;
      act(() => view.result.current.insertion.overwrite());
      act(() => vi.advanceTimersByTime(100));
      expect(selection).not.toHaveBeenCalled();
      // The newer command has been admitted into its own review, not cleared by
      // the old insertion completion or left replayable in the global slot.
      expect(view.result.current.insertion.promptReview).toBe(
        "A newer command",
      );
      expect(view.result.current.form.getValues().message).toBe(
        "Unrelated new input",
      );
    },
  );
});

describe("current shared quick-prompt review controls", () => {
  it("offers a true native Cancel and preserves the exact draft", () => {
    const Harness = () => {
      const form = useSimpleForm({
        initialValues: { message: "  Current draft\n" },
      });
      const textareaRef = React.useRef<HTMLTextAreaElement>(null);
      return (
        <>
          <textarea
            ref={textareaRef}
            value={form.values.message}
            readOnly
            aria-label="Draft"
          />
          <QuickPromptInsertionReview
            form={form}
            setMessageValue={(value) => form.setFieldValue("message", value)}
            ready
            destinationKey="current-source"
            captureGuard={() => () => true}
            textareaRef={textareaRef}
          />
        </>
      );
    };
    const view = render(<Harness />);
    act(() => setQuick(plainCommand));
    const cancel = view.getByRole("button", { name: "Cancel" });
    expect(cancel).toHaveAttribute("type", "button");
    expect(view.getByRole("button", { name: "Append" })).toHaveAttribute(
      "type",
      "button",
    );
    expect(
      view.getByRole("button", { name: "Overwrite message" }),
    ).toHaveAttribute("type", "button");
    fireEvent.click(cancel);
    expect(view.getByRole("textbox", { name: "Draft" })).toHaveValue(
      "  Current draft\n",
    );
    expect(view.queryByRole("button", { name: "Append" })).toBeNull();
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
  });
  it("appends exact whitespace and selects the inserted variable at its adjusted offset", () => {
    const initial = "  Existing draft \n";
    const text = "  Explain {topic}  ";
    const view = mountInsertion("webui", initial);
    act(() => setQuick(text));
    act(() => view.result.current.insertion.append());
    const result = initial + "\n\n" + text;
    expect(view.result.current.form.getValues().message).toBe(result);
    expect(view.textarea.value).toBe(result);
    expect(document.activeElement).toBe(view.textarea);
    expect([view.textarea.selectionStart, view.textarea.selectionEnd]).toEqual([
      initial.length + 2 + text.indexOf("{topic}"),
      initial.length + 2 + text.indexOf("{topic}") + "{topic}".length,
    ]);
    expect(view.result.current.insertion.promptReview).toBeNull();
  });
  it("does not steal focus moved to another field before accepted text commits", () => {
    const other = document.createElement("input");
    document.body.append(other);
    nodes.add(other);
    let moveFocus = false;
    const view = mountInsertion("sidepanel", "", {
      beforeSelection: (form) => {
        if (moveFocus && form.values.message === command) other.focus();
      },
    });
    act(() => setQuick(command));
    moveFocus = true;
    act(() => view.result.current.insertion.overwrite());
    expect(view.result.current.form.getValues().message).toBe(command);
    expect(document.activeElement).toBe(other);
  });
  it("rejects old choices after same-text command ABA and accepts the fresh command once", () => {
    const view = mountInsertion("sidepanel", "Original draft");
    act(() => setQuick(plainCommand));
    const oldChoice = view.result.current.insertion;
    act(() => {
      setQuick("Other command");
      setQuick(plainCommand);
    });
    act(oldChoice.overwrite);
    act(oldChoice.cancel);
    expect(view.result.current.insertion.promptReview).toBe(plainCommand);
    expect(view.result.current.form.getValues().message).toBe("Original draft");
    const freshChoice = view.result.current.insertion.overwrite;
    act(freshChoice);
    act(freshChoice);
    expect(view.result.current.form.getValues().message).toBe(plainCommand);
    expect(view.result.current.insertion.promptReview).toBeNull();
  });
  it("holds admission until restoration is ready and does not replay after StrictMode retirement", () => {
    const view = mountInsertion("webui", "Restored draft", {
      ready: false,
      strict: true,
    });
    act(() => setQuick(plainCommand));
    expect(view.result.current.insertion.promptReview).toBeNull();
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
      plainCommand,
    );
    view.rerender({ ready: true });
    expect(view.result.current.insertion.promptReview).toBe(plainCommand);
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
    const oldChoice = view.result.current.insertion.overwrite;
    view.rerender({ ready: false });
    view.rerender({ ready: true });
    act(oldChoice);
    expect(view.result.current.form.getValues().message).toBe("Restored draft");
    expect(view.result.current.insertion.promptReview).toBeNull();
  });
  it("admits a preexisting command once at the first ready StrictMode mount", () => {
    setQuick(command);
    const commandRevision =
      useStoreMessageOption.getState().quickPromptRevision;
    const view = mountInsertion("webui", "  Cold restored input  ", {
      strict: true,
    });
    expect(view.result.current.insertion.promptReview).toBe(command);
    expect(view.result.current.form.getValues().message).toBe(
      "  Cold restored input  ",
    );
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
    expect(useStoreMessageOption.getState().quickPromptRevision).toBe(
      commandRevision + 1,
    );
    act(() => view.result.current.insertion.overwrite());
    expect(view.result.current.form.getValues().message).toBe(command);
    expect(view.result.current.insertion.promptReview).toBeNull();
    expect(useStoreMessageOption.getState().quickPromptRevision).toBe(
      commandRevision + 1,
    );
  });
  it("does not publish a consumed review after an actual core subscriber changes input during CAS", () => {
    const view = mountInsertion("sidepanel", "Original source input");
    const sourceRevision = view.result.current.sourceRevision;
    let consumptions = 0;
    const stop = useStoreMessageOption.subscribe((current, previous) => {
      if (
        previous.selectedQuickPrompt !== plainCommand ||
        current.selectedQuickPrompt !== null
      )
        return;
      consumptions += 1;
      view.result.current.form.setFieldValue(
        "message",
        "Synchronous subscriber input",
      );
    });
    try {
      act(() => setQuick(plainCommand));
      expect(consumptions).toBe(1);
      expect(view.result.current.sourceRevision).toBe(sourceRevision);
      expect(view.result.current.form.getValues().message).toBe(
        "Synchronous subscriber input",
      );
      expect(view.result.current.insertion.promptReview).toBeNull();
      expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
    } finally {
      stop();
    }
  });
  it("refuses stale Cancel after input edit and Undo while the current review remains cancellable", () => {
    const view = mountInsertion("webui", "Original draft");
    act(() => setQuick(plainCommand));
    const staleCancel = view.result.current.insertion.cancel;
    const revision = view.result.current.form.getRevision();
    act(() => {
      view.result.current.form.setFieldValue("message", "Edited draft");
      view.result.current.form.setFieldValue("message", "Original draft");
    });
    expect(view.result.current.form.getRevision()).toBeGreaterThan(revision);
    act(staleCancel);
    expect(view.result.current.form.getValues().message).toBe("Original draft");
    expect(view.result.current.insertion.promptReview).toBe(plainCommand);
    act(() => view.result.current.insertion.cancel());
    expect(view.result.current.insertion.promptReview).toBeNull();
    expect(view.result.current.form.getValues().message).toBe("Original draft");
  });
  it("does not admit an unready source's pending command after New replaces its actual address", () => {
    const sourceAddress = useStoreMessageOption.getState().chatDraftAddress;
    expect(sourceAddress).not.toBeNull();
    const view = mountInsertion("sidepanel", "Previous source text", {
      ready: false,
    });
    act(() => setQuick(command));
    expect(view.result.current.insertion.promptReview).toBeNull();
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(command);
    act(() => {
      useStoreMessageOption.getState().startNewChatDraft(false);
      view.result.current.form.setFieldValue(
        "message",
        "Replacement source input",
      );
    });
    expect(useStoreMessageOption.getState().chatDraftAddress).not.toBe(
      sourceAddress,
    );
    view.rerender({ ready: true });
    expect(view.result.current.insertion.promptReview).toBeNull();
    expect(view.result.current.form.getValues().message).toBe(
      "Replacement source input",
    );
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull();
  });
  it("preserves an unready library command across initial null to first actual draft address", () => {
    useStoreMessageOption.setState({
      chatDraftAddress: null,
      chatDraftTransition: null,
    });
    const view = mountInsertion("webui", "Cold restored input", {
      ready: false,
    });
    act(() => setQuick(command));
    act(() => useStoreMessageOption.getState().startNewChatDraft(false));
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe(
      "fresh",
    );
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(command);
    view.rerender({ ready: true });
    expect(view.result.current.insertion.promptReview).toBe(command);
    expect(view.result.current.form.getValues().message).toBe(
      "Cold restored input",
    );
  });
  it("preserves an unready command across an explicit qualified bind of its current address", () => {
    const view = mountInsertion("webui", "Current input", { ready: false });
    act(() => setQuick(command));
    const source = useStoreMessageOption.getState();
    const address = source.chatDraftAddress;
    if (address?.kind !== "fresh")
      throw new Error("Expected actual fresh source address");
    let accepted = false;
    act(() => {
      accepted = useStoreMessageOption.getState().setChatDraftAddress(
        {
          kind: "native",
          id: "acknowledged-conversation",
          ownerKey: "declared-current-owner",
          scope: address.scope,
        },
        {
          kind: "bind",
          sourceAddress: address,
          expectedRevision: source.chatDraftRevision,
        },
      );
    });
    expect(accepted).toBe(true);
    expect(useStoreMessageOption.getState().chatDraftTransition?.kind).toBe(
      "bind",
    );
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(command);
    view.rerender({ ready: true });
    expect(view.result.current.insertion.promptReview).toBe(command);
    expect(view.result.current.form.getValues().message).toBe("Current input");
  });
  it("rejects a consumed review after account/H1 source authority retires before rerender", () => {
    let current = true;
    const view = mountInsertion("webui", "Current draft", {
      isOwnerCurrent: () => current,
    });
    act(() => setQuick(plainCommand));
    const oldChoice = view.result.current.insertion.overwrite;
    current = false;
    act(oldChoice);
    expect(view.result.current.form.getValues().message).toBe("Current draft");
  });
  it("admits a command into only one mounted composer instead of replaying it to a sibling", () => {
    const first = mountInsertion("webui", "First draft");
    const second = mountInsertion("sidepanel", "Second draft");
    act(() => setQuick(plainCommand));
    expect(
      [
        first.result.current.insertion.promptReview,
        second.result.current.insertion.promptReview,
      ].filter((review) => review !== null),
    ).toEqual([plainCommand]);
    expect(first.result.current.form.getValues().message).toBe("First draft");
    expect(second.result.current.form.getValues().message).toBe("Second draft");
  });
});
