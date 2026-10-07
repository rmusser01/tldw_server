// @vitest-environment jsdom
import React from "react";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useSimpleForm } from "@/hooks/useSimpleForm";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { useStoreMessageOption } from "@/store/option";
import {
  encodeSidepanelChatWebUiHandoff,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS,
  type SidepanelChatWebUiHandoffPayload,
} from "@/services/tldw/sidepanel-chat-webui-handoff";
import { useSidepanelChatHandoffImport } from "../useSidepanelChatHandoffImport";

// Actual import hook, packet codec, simple form, option store, selected-model
// storage composition and model helpers.
// Catalog I/O, browser model storage and surrounding Form account/H1 ownership
// are declared boundary doubles; this suite does not prove native/browser I/O.
const io = vi.hoisted(() => ({ catalog: vi.fn(), persistModel: vi.fn() }));
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: (...args: unknown[]) => io.catalog(...args),
}));
// This storage boundary starts at the same model as the real store and reflects
// writes synchronously to avoid artificial hydration races. Durability/rejection
// belongs to the declared writer Promise; native storage is not exercised.
vi.mock("@plasmohq/storage/hook", async () => {
  const React = await import("react");
  return {
    useStorage: () => {
      const [stored, setStored] = React.useState<string | null>(
        "anthropic:shared",
      );
      const write = React.useCallback((next: string | null) => {
        setStored(next);
        return io.persistModel(next);
      }, []);
      return [stored, write, { isLoading: false }];
    },
  };
});
const initialOptions = {
  selectedModel: "anthropic:shared",
  chatMode: "normal" as const,
  webSearch: false,
  toolChoice: "none" as const,
  useOCR: false,
};
const copiedOptions = {
  selectedModel: "openai:shared",
  chatMode: "rag" as const,
  webSearch: true,
  toolChoice: "required" as const,
  useOCR: true,
};
const currentOptions = () => {
  const state = useStoreMessageOption.getState();
  return {
    selectedModel: state.selectedModel,
    chatMode: state.chatMode,
    webSearch: state.webSearch,
    toolChoice: state.toolChoice,
    useOCR: state.useOCR,
  };
};
const catalog = [
  { model: "shared", provider: "openai", is_configured: true },
  { model: "shared", provider: "anthropic", is_configured: true },
];
const copiedDraft = "  Copied question\n\nKeep the exact spaces.  ";
const packet = (overrides: Partial<SidepanelChatWebUiHandoffPayload> = {}) =>
  encodeSidepanelChatWebUiHandoff({
    source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
    createdAt: Date.now(),
    draft: copiedDraft,
    ...copiedOptions,
    selectedSystemPrompt: "foreign-local-prompt-id",
    selectedQuickPrompt: "A quick prompt must not replace the reviewed draft",
    temporaryChat: true,
    historyId: "foreign-history-id",
    ragMediaIds: [54],
    ...overrides,
  });
const entryFor = (token: string) =>
  `/chat?keep=query#keep=fragment&handoff=${token}`;
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
};
const t = (_key: string, fallback?: string | { defaultValue?: string }) =>
  typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? _key);
type ImportResult = ReturnType<typeof useSidepanelChatHandoffImport> & {
  literalOptionsReview: Array<{
    key: keyof typeof copiedOptions;
    label: string;
    current: unknown;
    copied: unknown;
  }>;
  includeLiteralOptions: boolean;
  setIncludeLiteralOptions: (include: boolean) => void;
  applyHandoffOptions: () => Promise<void> | void;
};
let owner: object;
const mount = (
  token = packet(),
  initialDraft = "Destination question",
  canApply?: () => boolean,
) => {
  const initialEntry = entryFor(token);
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter initialEntries={[initialEntry]}>{children}</MemoryRouter>
  );
  const warning = vi.fn();
  const view = renderHook(
    () => {
      const state = useStoreMessageOption((value) => value);
      const form = useSimpleForm({
        initialValues: { message: initialDraft, image: "" },
      });
      const captureGuard = React.useCallback(() => {
        const capturedOwner = owner;
        return () => owner === capturedOwner;
      }, []);
      const { setFieldValue } = form;
      const setMessageValue = React.useCallback(
        (value: string) => setFieldValue("message", value),
        [setFieldValue],
      );
      const { setSelectedModel } = useSelectedModel();
      const dependencies = {
        ready: true,
        destinationKey: "owned-destination-chat",
        captureGuard,
        draftValue: form.values.message,
        getDraftRevision: form.getRevision,
        setMessageValue,
        notificationApi: { warning },
        t,
        literalOptions: {
          current: {
            selectedModel: state.selectedModel,
            chatMode: state.chatMode,
            webSearch: state.webSearch,
            toolChoice: state.toolChoice,
            useOCR: state.useOCR,
          },
          canApply,
          setSelectedModel,
          setChatMode: state.setChatMode,
          setWebSearch: state.setWebSearch,
          setToolChoice: state.setToolChoice,
          setUseOCR: state.setUseOCR,
        },
      };
      const handoff = useSidepanelChatHandoffImport(
        dependencies,
      ) as ImportResult;
      return {
        handoff,
        setSelectedModel,
        form,
        location: useLocation(),
        navigate: useNavigate(),
      };
    },
    { wrapper },
  );
  return { ...view, warning };
};
const reviewed = async (view: ReturnType<typeof mount>) => {
  await waitFor(() =>
    expect(view.result.current.handoff.conflict).not.toBeNull(),
  );
};
const consent = (view: ReturnType<typeof mount>) => {
  expect(typeof view.result.current.handoff.setIncludeLiteralOptions).toBe(
    "function",
  );
  act(() => view.result.current.handoff.setIncludeLiteralOptions(true));
};
const expectUnchanged = (view: ReturnType<typeof mount>) => {
  expect(currentOptions()).toEqual(initialOptions);
  expect(view.result.current.form.getValues().message).toBe(
    "Destination question",
  );
  expect(io.persistModel).not.toHaveBeenCalled();
};
beforeEach(() => {
  owner = {};
  useStoreMessageOption.setState({
    ...initialOptions,
    selectedModelStorageHandled: false,
    selectedModelRevision: 0,
    selectedQuickPrompt: null,
    selectedSystemPrompt: "destination-local-prompt-id",
    temporaryChat: false,
    historyId: "destination-history-id",
    serverChatId: "destination-server-chat-id",
    ragMediaIds: [19],
  });
  io.catalog.mockReset().mockResolvedValue(catalog);
  io.persistModel.mockReset().mockResolvedValue(undefined);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("explicit passive options in an encoded WebUI handoff", () => {
  it("starts unchecked with five current/copied rows, then applies exact text and a qualified model only after consent", async () => {
    const view = mount();
    await reviewed(view);
    expect(view.result.current.handoff.includeLiteralOptions).toBe(false);
    expect(
      view.result.current.handoff.literalOptionsReview
        .map((row) => row.key)
        .sort(),
    ).toEqual(Object.keys(copiedOptions).sort());
    expect(
      view.result.current.handoff.literalOptionsReview.every(
        (row) => row.label && "current" in row && "copied" in row,
      ),
    ).toBe(true);
    expectUnchanged(view);
    consent(view);
    await act(async () => {
      await view.result.current.handoff.replaceWithHandoffDraft();
    });
    expect(io.catalog).toHaveBeenCalledWith({ requireFresh: true });
    expect(currentOptions()).toEqual(copiedOptions);
    expect(view.result.current.form.getValues().message).toBe(copiedDraft);
    expect(useStoreMessageOption.getState()).toMatchObject({
      selectedQuickPrompt: null,
      selectedSystemPrompt: "destination-local-prompt-id",
      temporaryChat: false,
      historyId: "destination-history-id",
      serverChatId: "destination-server-chat-id",
      ragMediaIds: [19],
    });
    expect(view.result.current.handoff.importReceipt?.description).toMatch(
      /imported|applied/i,
    );
    expect(view.result.current.handoff.importReceipt?.description).toMatch(
      /not imported/i,
    );
    expect(view.result.current.location.search).toBe("?keep=query");
    expect(view.result.current.location.hash).toBe("#keep=fragment");
  });
  it("keeps all options for text-only import and for Cancel", async () => {
    const view = mount();
    await reviewed(view);
    await act(async () => {
      await view.result.current.handoff.insertHandoffDraft();
    });
    expect(view.result.current.form.getValues().message).toBe(
      `Destination question\n\n${copiedDraft}`,
    );
    expect(currentOptions()).toEqual(initialOptions);
    expect(io.catalog).not.toHaveBeenCalled();
    expect(io.persistModel).not.toHaveBeenCalled();
    act(() =>
      view.result.current.navigate(entryFor(packet({ draft: "Another copy" }))),
    );
    await reviewed(view);
    act(() => view.result.current.handoff.cancelHandoffImport());
    expect(currentOptions()).toEqual(initialOptions);
    expect(view.result.current.form.getValues().message).toBe(
      `Destination question\n\n${copiedDraft}`,
    );
    expect(view.result.current.location.hash).toBe("#keep=fragment");
  });
  it("applies consented settings-only copies without writing or claiming imported draft text", async () => {
    const view = mount(packet({ draft: undefined }));
    await reviewed(view);
    expect(view.result.current.handoff.conflict?.handoff.hasText).toBe(false);
    consent(view);
    await act(async () => {
      await view.result.current.handoff.applyHandoffOptions();
    });
    expect(currentOptions()).toEqual(copiedOptions);
    expect(view.result.current.form.getValues().message).toBe(
      "Destination question",
    );
    expect(view.result.current.form.getRevision()).toBe(0);
    expect(view.result.current.handoff.importReceipt?.description).toMatch(
      /options.*applied|settings.*applied/i,
    );
    expect(view.result.current.handoff.importReceipt?.description).not.toMatch(
      /draft text imported/i,
    );
    expect(view.result.current.location.hash).toBe("#keep=fragment");
  });
  it.each([
    { label: "ambiguous bare model", selectedModel: "shared", models: catalog },
    {
      label: "missing qualified model",
      selectedModel: "openai:missing",
      models: catalog,
    },
    {
      label: "unconfigured provider despite a configured namesake",
      selectedModel: "openai:shared",
      models: [
        { model: "shared", provider: "openai", is_configured: false },
        catalog[1],
      ],
    },
  ])(
    "retains draft, options and review for $label",
    async ({ selectedModel, models }) => {
      io.catalog.mockResolvedValue(models);
      const view = mount(packet({ selectedModel }));
      await reviewed(view);
      consent(view);
      await act(async () => {
        await view.result.current.handoff.replaceWithHandoffDraft();
      });
      expectUnchanged(view);
      expect(view.result.current.handoff.conflict).not.toBeNull();
      expect(view.result.current.handoff.conflictFeedback).toBeTruthy();
      expect(view.result.current.location.hash).toContain("handoff=");
    },
  );
  it.each(["options", "draft"] as const)(
    "rejects %s edit then Undo while fresh model validation is held",
    async (kind) => {
      const held = deferred<typeof catalog>();
      io.catalog.mockReturnValueOnce(held.promise);
      const view = mount();
      await reviewed(view);
      consent(view);
      let completion!: Promise<unknown>;
      act(() => {
        completion = view.result.current.handoff.replaceWithHandoffDraft();
      });
      await waitFor(() => expect(io.catalog).toHaveBeenCalledTimes(1));
      act(() => {
        if (kind === "options") {
          const store = useStoreMessageOption.getState();
          store.setWebSearch(true);
          store.setWebSearch(false);
        } else {
          view.result.current.form.setFieldValue("message", "Transient edit");
          view.result.current.form.setFieldValue(
            "message",
            "Destination question",
          );
        }
      });
      await act(async () => {
        held.resolve(catalog);
        await completion;
      });
      expectUnchanged(view);
      expect(view.result.current.handoff.conflict).not.toBeNull();
      expect(view.result.current.handoff.conflictFeedback).toBeTruthy();
      expect(view.result.current.location.hash).toContain("handoff=");
    },
  );
  it.each(["owner", "route", "Cancel"] as const)(
    "cannot apply held validation after %s retirement",
    async (retirement) => {
      const held = deferred<typeof catalog>();
      io.catalog.mockReturnValueOnce(held.promise);
      const view = mount();
      await reviewed(view);
      consent(view);
      let completion!: Promise<unknown>;
      act(() => {
        completion = view.result.current.handoff.replaceWithHandoffDraft();
      });
      await waitFor(() => expect(io.catalog).toHaveBeenCalledTimes(1));
      act(() => {
        if (retirement === "owner") owner = {};
        if (retirement === "route")
          view.result.current.navigate("/other?keep=elsewhere");
        if (retirement === "Cancel")
          view.result.current.handoff.cancelHandoffImport();
      });
      await act(async () => {
        held.resolve(catalog);
        await completion;
      });
      expectUnchanged(view);
      expect(view.result.current.handoff.importReceipt).toBeNull();
      if (retirement === "route")
        expect(view.result.current.location.pathname).toBe("/other");
      if (retirement === "Cancel")
        expect(view.result.current.location.hash).toBe("#keep=fragment");
    },
  );
  it("retains review after failed discovery and retries only by another explicit current choice", async () => {
    io.catalog.mockRejectedValueOnce(new Error("Model metadata unavailable"));
    const view = mount();
    await reviewed(view);
    consent(view);
    await act(async () => {
      await view.result.current.handoff.replaceWithHandoffDraft();
    });
    expectUnchanged(view);
    expect(view.result.current.handoff.conflictFeedback).toBeTruthy();
    expect(view.result.current.location.hash).toContain("handoff=");
    await act(async () => {
      await view.result.current.handoff.replaceWithHandoffDraft();
    });
    expect(io.catalog).toHaveBeenCalledTimes(2);
    expect(currentOptions()).toEqual(copiedOptions);
    expect(view.result.current.form.getValues().message).toBe(copiedDraft);
  });
  it("reports applied destination options plus failed model persistence without rollback or replay", async () => {
    io.persistModel.mockRejectedValueOnce(
      new Error("Browser storage write failed"),
    );
    const view = mount();
    await reviewed(view);
    consent(view);
    await act(async () => {
      await view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() =>
      expect(view.result.current.handoff.importReceipt?.description).toMatch(
        /could not.*sav|not saved|failed.*sav/i,
      ),
    );
    expect(currentOptions()).toEqual(copiedOptions);
    expect(view.result.current.form.getValues().message).toBe(copiedDraft);
    expect(view.result.current.handoff.importReceipt?.description).toMatch(
      /applied|imported/i,
    );
    expect(view.result.current.handoff.conflict).toBeNull();
    expect(view.result.current.location.hash).toBe("#keep=fragment");
    expect(io.catalog).toHaveBeenCalledTimes(1);
    expect(io.persistModel).toHaveBeenCalledTimes(1);
  });
  it("does not let an old model persistence rejection replace a newer transfer receipt", async () => {
    const oldWrite = deferred<void>();
    io.persistModel.mockReturnValueOnce(oldWrite.promise);
    const view = mount();
    await reviewed(view);
    consent(view);
    act(() => {
      void view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() =>
      expect(view.result.current.handoff.conflict).toBeNull(),
    );
    act(() =>
      view.result.current.navigate(
        entryFor(packet({ draft: "New reviewed text" })),
      ),
    );
    await reviewed(view);
    expect(view.result.current.handoff.includeLiteralOptions).toBe(false);
    consent(view);
    await act(async () => {
      await view.result.current.handoff.replaceWithHandoffDraft();
    });
    const newReceipt = view.result.current.handoff.importReceipt;
    expect(newReceipt).not.toBeNull();
    await act(async () => {
      oldWrite.reject(new Error("Old browser write failed"));
    });
    expect(view.result.current.handoff.importReceipt).toBe(newReceipt);
    expect(view.result.current.form.getValues().message).toBe(
      "New reviewed text",
    );
    expect(io.persistModel).toHaveBeenCalledTimes(2);
  });
  it("refuses a retained checked accept callback after the user unchecks consent", async () => {
    const view = mount();
    await reviewed(view);
    consent(view);
    const checkedAccept = view.result.current.handoff.replaceWithHandoffDraft;
    act(() => view.result.current.handoff.setIncludeLiteralOptions(false));
    expect(view.result.current.handoff.includeLiteralOptions).toBe(false);
    await act(async () => {
      await checkedAccept();
    });
    expect(currentOptions()).toEqual(initialOptions);
    expect(io.persistModel).not.toHaveBeenCalled();
    expect(io.catalog).not.toHaveBeenCalled();
    expect(view.result.current.handoff.conflict).not.toBeNull();
    expect(view.result.current.location.hash).toContain("handoff=");
  });
  it("refuses applying pending settings when the destination lease retires during catalog validation", async () => {
    let settingsAvailable = true;
    const canApply = () => settingsAvailable;
    const held = deferred<typeof catalog>();
    io.catalog.mockReturnValueOnce(held.promise);
    const view = mount(packet(), "Destination question", canApply);
    await reviewed(view);
    consent(view);
    let completion!: Promise<unknown>;
    act(() => {
      completion = view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() => expect(io.catalog).toHaveBeenCalledTimes(1));
    settingsAvailable = false;
    await act(async () => {
      held.resolve(catalog);
      await completion;
    });
    expectUnchanged(view);
    expect(view.result.current.handoff.conflict).not.toBeNull();
    expect(view.result.current.handoff.conflictFeedback).toBeTruthy();
    expect(view.result.current.location.hash).toContain("handoff=");
  });
  it("does not apply a packet that expires during held fresh catalog validation", async () => {
    const now = Date.now();
    const clock = vi.spyOn(Date, "now").mockReturnValue(now);
    const held = deferred<typeof catalog>();
    io.catalog.mockReturnValueOnce(held.promise);
    const view = mount(packet());
    await reviewed(view);
    consent(view);
    let completion!: Promise<unknown>;
    act(() => {
      completion = view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() => expect(io.catalog).toHaveBeenCalledTimes(1));
    clock.mockReturnValue(now + SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS + 1);
    await act(async () => {
      held.resolve(catalog);
      await completion;
    });
    expectUnchanged(view);
    expect(view.result.current.handoff.importReceipt).toBeNull();
    expect(view.warning).toHaveBeenCalledTimes(1);
    expect(view.result.current.location.hash).toBe("#keep=fragment");
  });
  it("retires a pending model storage receipt after a later option edit without rollback or replay", async () => {
    const held = deferred<void>();
    io.persistModel.mockReturnValueOnce(held.promise);
    const view = mount();
    await reviewed(view);
    consent(view);
    let completion!: Promise<unknown>;
    act(() => {
      completion = view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() =>
      expect(view.result.current.handoff.importReceipt?.description).toMatch(
        /saving/i,
      ),
    );
    act(() => useStoreMessageOption.getState().setWebSearch(false));
    await act(async () => {
      held.reject(new Error("Browser storage failed after editing"));
      await completion;
    });
    expect(
      view.result.current.handoff.importReceipt?.description ?? "",
    ).not.toMatch(/saving/i);
    expect(currentOptions()).toEqual({ ...copiedOptions, webSearch: false });
    expect(view.result.current.form.getValues().message).toBe(copiedDraft);
    expect(io.persistModel).toHaveBeenCalledTimes(1);
    expect(io.catalog).toHaveBeenCalledTimes(1);
    expect(view.result.current.location.hash).toBe("#keep=fragment");
  });
  it("retires a pending save notice when the same model is explicitly selected again", async () => {
    const held = deferred<void>();
    io.persistModel.mockReturnValueOnce(held.promise);
    const view = mount();
    await reviewed(view);
    consent(view);
    let completion!: Promise<unknown>;
    act(() => {
      completion = view.result.current.handoff.replaceWithHandoffDraft();
    });
    await waitFor(() =>
      expect(view.result.current.handoff.importReceipt?.description).toMatch(
        /saving/i,
      ),
    );
    await act(async () => {
      await view.result.current.setSelectedModel(copiedOptions.selectedModel);
    });
    expect(view.result.current.handoff.importReceipt).toBeNull();
    await act(async () => {
      held.reject(new Error("Old write failed"));
      await completion;
    });
    expect(view.result.current.handoff.importReceipt).toBeNull();
    expect(currentOptions()).toEqual(copiedOptions);
    expect(io.persistModel).toHaveBeenCalledTimes(2);
    expect(view.result.current.form.getValues().message).toBe(copiedDraft);
  });
});
