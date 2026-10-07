/**
 * CC-05 (#3112) with the real antd Dropdown: antd closes the menu after any
 * menu click, so this guards that picking a server prompt keeps the menu open
 * while the prompt loads (and when it fails), then closes once it applies.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildFakeServerPrompt,
  createFakePromptsApi,
} from "@/services/__tests__/server-prompts-api-fixture";
import { PromptSelect } from "../PromptSelect";
import { useStoreMessageOption } from "@/store/option";
import * as focusReturn from "@/utils/focus-return";
import {
  HistorySelectionContext,
  useHistorySelection,
} from "@/hooks/chat/useHistorySelection";

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getAllPrompts: vi.fn(async () => [] as unknown[]),
  getPromptById: vi.fn(async () => undefined),
  upsertServerLibraryPromptCopy: vi.fn(async () => ({
    id: "owned-system-copy",
  })),
}));

vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next");
  const i18n = createInstance();
  await i18n.init({
    lng: "en",
    resources: {},
    interpolation: { escapeValue: false },
  });
  return {
    useTranslation: () => ({
      t: (...args: Parameters<typeof i18n.t>) => i18n.t(...args),
    }),
  };
});

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue),
}));

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
}));

vi.mock("@/db/dexie/helpers", () => ({
  updatePrompt: vi.fn(),
  markPromptSyncError: vi.fn(),
  getAllPrompts: mocks.getAllPrompts,
  getPromptById: mocks.getPromptById,
  upsertServerLibraryPromptCopy: mocks.upsertServerLibraryPromptCopy,
}));

const createDeferred = () => {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
};

/** Open = rendered and neither hidden nor leaving (jsdom never ends motion). */
const isMenuOpen = () => {
  const popup = screen.queryByRole("menu")?.closest(".ant-dropdown");
  if (!popup) return false;
  return !Array.from(popup.classList).some(
    (name) => name === "ant-dropdown-hidden" || name.endsWith("-leave"),
  );
};

describe(
  "PromptSelect server prompts with the real antd Dropdown",
  { timeout: 60_000 },
  () => {
    let detailGate = createDeferred();

    beforeEach(() => {
      vi.clearAllMocks();
      useStoreMessageOption.getState().startNewChatDraft(false);
      detailGate = createDeferred();
      const api = createFakePromptsApi([
        buildFakeServerPrompt(5, {
          name: "Rewrite plainly",
          user_prompt: "Rewrite this plainly",
        }),
      ]);
      mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
        if (/\/api\/v1\/prompts\/5$/.test(String(request.path))) {
          await detailGate.promise;
        }
        return api.handle(request) ?? {};
      });
    });

    const renderPicker = (
      withHistorySelection = false,
      withActualSetters = false,
    ) => {
      let controller: ReturnType<typeof useHistorySelection> | undefined;
      const props = {
        selectedSystemPrompt: undefined,
        systemPrompt: "",
        setSystemPrompt: vi.fn(),
        setSelectedSystemPrompt: withActualSetters
          ? vi.fn((value?: string) =>
              useStoreMessageOption
                .getState()
                .setSelectedSystemPrompt(value ?? null),
            )
          : vi.fn(),
        setSelectedQuickPrompt: vi.fn((value?: string) =>
          useStoreMessageOption
            .getState()
            .setSelectedQuickPrompt(value ?? null),
        ),
      };
      const OwnedPicker = () => {
        const selection = useHistorySelection();
        React.useLayoutEffect(() => {
          controller = selection;
        }, [selection]);
        return (
          <HistorySelectionContext.Provider value={selection}>
            <PromptSelect {...props} />
          </HistorySelectionContext.Provider>
        );
      };
      render(
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          {withHistorySelection ? <OwnedPicker /> : <PromptSelect {...props} />}
        </QueryClientProvider>,
      );
      return {
        ...props,
        get controller() {
          if (!controller) throw new Error("H1 provider not requested");
          return controller;
        },
      };
    };

    it("keeps the menu open while a server prompt loads, then closes it once applied", async () => {
      const user = userEvent.setup();
      const props = renderPicker();

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" }),
      );
      await user.click(
        await screen.findByRole(
          "menuitem",
          { name: /Rewrite plainly/ },
          { timeout: 5_000 },
        ),
      );

      expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
      expect(isMenuOpen()).toBe(true);

      detailGate.resolve();
      await waitFor(() =>
        expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith(
          "Rewrite this plainly",
        ),
      );
      await waitFor(() => expect(isMenuOpen()).toBe(false));
    });

    it("keeps the menu open with a retry hint when a server prompt fails to load", async () => {
      const user = userEvent.setup();
      const props = renderPicker();

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" }),
      );
      await user.click(
        await screen.findByRole(
          "menuitem",
          { name: /Rewrite plainly/ },
          { timeout: 5_000 },
        ),
      );
      detailGate.reject(new Error("Request failed: 500"));

      expect(
        await screen.findByText(
          "Couldn't load this prompt from the server. Select it to retry.",
        ),
      ).toBeInTheDocument();
      expect(isMenuOpen()).toBe(true);
      expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
    });

    it("refuses a retained rendered picker action after same-source H1 departure and return", async () => {
      const user = userEvent.setup();
      const props = renderPicker(true);
      const trigger = await screen.findByRole("button", {
        name: "selectAPrompt",
      });
      await user.click(trigger);
      const choice = await screen.findByRole("menuitem", {
        name: /Rewrite plainly/,
      });
      const source = useStoreMessageOption.getState();
      try {
        act(() => {
          props.controller.beginLoad();
          props.controller.reset();
          // The actual antd menu still holds the previous render's callback.
          fireEvent.click(choice);
        });
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          source.chatDraftAddress,
        );
        expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
          source.chatDraftRevision,
        );
        expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
        expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled();
        if (isMenuOpen()) await user.click(trigger);
        await user.click(trigger);
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        await waitFor(() =>
          expect(props.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
            "Rewrite this plainly",
          ),
        );
        await waitFor(() => expect(isMenuOpen()).toBe(false));
      } finally {
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
      }
    });

    it("publishes the current server quick prompt through actual core setters and closes the menu", async () => {
      const user = userEvent.setup();
      const props = renderPicker(false, true);
      const focus = vi.spyOn(focusReturn, "scheduleFocusFirstVisibleElement");
      try {
        await user.click(
          await screen.findByRole("button", { name: "selectAPrompt" }),
        );
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
        await waitFor(() =>
          expect(props.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
            "Rewrite this plainly",
          ),
        );
        expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
          "Rewrite this plainly",
        );
        await waitFor(() => expect(isMenuOpen()).toBe(false));
        expect(focus).toHaveBeenCalledOnce();
      } finally {
        focus.mockRestore();
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
      }
    });

    it("refuses quick publication if its actual system setter synchronously replaces the source", async () => {
      const user = userEvent.setup();
      useStoreMessageOption
        .getState()
        .setSelectedSystemPrompt("original-system");
      const props = renderPicker(false, true);
      const focus = vi.spyOn(focusReturn, "scheduleFocusFirstVisibleElement");
      let intervened = false;
      let stop = () => {};
      try {
        await user.click(
          await screen.findByRole("button", { name: "selectAPrompt" }),
        );
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        stop = useStoreMessageOption.subscribe((state) => {
          if (!intervened && state.selectedSystemPrompt === null) {
            intervened = true;
            state.startNewChatDraft(false);
            state.setSelectedQuickPrompt("Replacement source command");
          }
        });
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
        expect(intervened).toBe(true);
        expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
        expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
          "Replacement source command",
        );
        expect(isMenuOpen()).toBe(true);
        expect(focus).not.toHaveBeenCalled();
      } finally {
        stop();
        focus.mockRestore();
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
      }
    });

    it("does not close or schedule focus after its actual quick setter publishes a newer command", async () => {
      const user = userEvent.setup();
      const props = renderPicker(false, true);
      const focus = vi.spyOn(focusReturn, "scheduleFocusFirstVisibleElement");
      let intervened = false;
      let stop = () => {};
      try {
        await user.click(
          await screen.findByRole("button", { name: "selectAPrompt" }),
        );
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        stop = useStoreMessageOption.subscribe((state) => {
          if (
            !intervened &&
            state.selectedQuickPrompt === "Rewrite this plainly"
          ) {
            intervened = true;
            state.setSelectedQuickPrompt("Newer synchronous command");
          }
        });
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
        expect(intervened).toBe(true);
        expect(props.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
          "Rewrite this plainly",
        );
        expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
          "Newer synchronous command",
        );
        expect(isMenuOpen()).toBe(true);
        expect(focus).not.toHaveBeenCalled();
      } finally {
        stop();
        focus.mockRestore();
        await act(async () => {
          detailGate.resolve();
          await detailGate.promise;
        });
      }
    });

    it.each([false, true])(
      "closes a system selection only while its original quick intent remains current (replaced=%s)",
      async (replaced) => {
        const user = userEvent.setup();
        const initial = useStoreMessageOption.getState();
        initial.setSelectedQuickPrompt("Original unrelated command");
        const originalRevision =
          useStoreMessageOption.getState().quickPromptRevision;
        const api = createFakePromptsApi([
          buildFakeServerPrompt(5, {
            name: "Rewrite plainly",
            system_prompt: "Use a concise system voice.",
            user_prompt: null,
          }),
        ]);
        mocks.bgRequest.mockImplementation(
          async (request: { path?: string }) => {
            if (/\/api\/v1\/prompts\/5$/.test(String(request.path)))
              await detailGate.promise;
            return api.handle(request) ?? {};
          },
        );
        const props = renderPicker(false, true);
        const focus = vi.spyOn(focusReturn, "scheduleFocusFirstVisibleElement");
        let intervened = false;
        let stop = () => {};
        try {
          await user.click(
            await screen.findByRole("button", { name: "selectAPrompt" }),
          );
          await user.click(
            await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
          );
          expect(
            await screen.findByText("Loading prompt…"),
          ).toBeInTheDocument();
          if (replaced) {
            stop = useStoreMessageOption.subscribe((state) => {
              if (
                !intervened &&
                state.selectedSystemPrompt === "owned-system-copy"
              ) {
                intervened = true;
                state.setSelectedQuickPrompt(
                  "Newer command during system selection",
                );
              }
            });
          }
          // The local upsert is a declared database I/O double; menu actions,
          // source core and both publication setters are actual production paths.
          await act(async () => {
            detailGate.resolve();
            await detailGate.promise;
          });
          expect(mocks.upsertServerLibraryPromptCopy).toHaveBeenCalledOnce();
          expect(props.setSelectedSystemPrompt).toHaveBeenCalledExactlyOnceWith(
            "owned-system-copy",
          );
          expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
          if (replaced) {
            expect(intervened).toBe(true);
            expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
              "Newer command during system selection",
            );
            expect(isMenuOpen()).toBe(true);
            expect(focus).not.toHaveBeenCalled();
          } else {
            expect(useStoreMessageOption.getState().quickPromptRevision).toBe(
              originalRevision,
            );
            expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
              "Original unrelated command",
            );
            await waitFor(() => expect(isMenuOpen()).toBe(false));
            expect(focus).toHaveBeenCalledOnce();
          }
        } finally {
          stop();
          focus.mockRestore();
          await act(async () => {
            detailGate.resolve();
            await detailGate.promise;
          });
        }
      },
    );

    it("allows a current same-key choice without closing after a held H1 attempt retires", async () => {
      const user = userEvent.setup();
      const first = createDeferred();
      const second = createDeferred();
      const gates = [first, second];
      const api = createFakePromptsApi([
        buildFakeServerPrompt(5, {
          name: "Rewrite plainly",
          user_prompt: "Rewrite this plainly",
        }),
      ]);
      let details = 0;
      mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
        if (/\/api\/v1\/prompts\/5$/.test(String(request.path))) {
          const gate = gates[details++];
          if (!gate) throw new Error("Unexpected detail request");
          await gate.promise;
        }
        return api.handle(request) ?? {};
      });
      const props = renderPicker(true);
      try {
        await user.click(
          await screen.findByRole("button", { name: "selectAPrompt" }),
        );
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        expect(details).toBe(1);
        const source = useStoreMessageOption.getState();
        act(() => {
          props.controller.beginLoad();
          props.controller.reset();
        });
        await act(async () => {
          first.resolve();
          await first.promise;
        });
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          source.chatDraftAddress,
        );
        expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
          source.chatDraftRevision,
        );
        expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
        expect(isMenuOpen()).toBe(true);
        // Retry is a fresh current menu action; do not close/reopen to repair it.
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        await waitFor(() => expect(details).toBe(2));
        await act(async () => {
          second.resolve();
          await second.promise;
        });
        await waitFor(() =>
          expect(props.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
            "Rewrite this plainly",
          ),
        );
        expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
          "Rewrite this plainly",
        );
        await waitFor(() => expect(isMenuOpen()).toBe(false));
      } finally {
        await act(async () => {
          first.resolve();
          second.resolve();
          await Promise.all(gates.map((gate) => gate.promise));
        });
      }
    });

    // Actual antd disclosure/menu events and actual draft core are used here.
    // The HTTP library fixture holds detail responses; setters are visible UI
    // sinks, so these cases do not claim native background/backend transport.
    it("does not adopt an older same-key response after closing and reopening the picker", async () => {
      const user = userEvent.setup();
      const first = createDeferred();
      const second = createDeferred();
      const gates = [first, second];
      const api = createFakePromptsApi([
        buildFakeServerPrompt(5, {
          name: "Rewrite plainly",
          user_prompt: "Rewrite this plainly",
        }),
      ]);
      let details = 0;
      let finished = 0;
      mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
        if (/\/api\/v1\/prompts\/5$/.test(String(request.path))) {
          const gate = gates[details++];
          if (!gate) throw new Error("Unexpected detail request");
          await gate.promise;
          finished += 1;
        }
        return api.handle(request) ?? {};
      });
      const props = renderPicker();
      try {
        const trigger = await screen.findByRole("button", {
          name: "selectAPrompt",
        });
        await user.click(trigger);
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        await user.click(trigger);
        await waitFor(() => expect(isMenuOpen()).toBe(false));
        await user.click(trigger);
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        await waitFor(() => expect(details).toBe(2));
        await act(async () => {
          first.resolve();
          await first.promise;
        });
        await waitFor(() => expect(finished).toBe(1));
        expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
        expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled();
        expect(isMenuOpen()).toBe(true);
        expect(screen.getByText("Loading prompt…")).toBeInTheDocument();
        await act(async () => {
          second.resolve();
          await second.promise;
        });
        await waitFor(() =>
          expect(props.setSelectedQuickPrompt).toHaveBeenCalledOnce(),
        );
        expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith(
          "Rewrite this plainly",
        );
        await waitFor(() => expect(isMenuOpen()).toBe(false));
      } finally {
        await act(async () => {
          first.resolve();
          second.resolve();
          await Promise.all(gates.map((gate) => gate.promise));
        });
      }
    });

    it("does not publish a held prompt into New and lets the current picker choose it again", async () => {
      const user = userEvent.setup();
      const first = createDeferred();
      const second = createDeferred();
      const gates = [first, second];
      const api = createFakePromptsApi([
        buildFakeServerPrompt(5, {
          name: "Rewrite plainly",
          user_prompt: "Rewrite this plainly",
        }),
      ]);
      let details = 0;
      mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
        if (/\/api\/v1\/prompts\/5$/.test(String(request.path))) {
          const gate = gates[details++];
          if (!gate) throw new Error("Unexpected detail request");
          await gate.promise;
        }
        return api.handle(request) ?? {};
      });
      const props = renderPicker();
      try {
        const trigger = await screen.findByRole("button", {
          name: "selectAPrompt",
        });
        await user.click(trigger);
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        expect(await screen.findByText("Loading prompt…")).toBeInTheDocument();
        const source = useStoreMessageOption.getState().chatDraftAddress;
        act(() => useStoreMessageOption.getState().startNewChatDraft(false));
        expect(useStoreMessageOption.getState().chatDraftAddress).not.toBe(
          source,
        );
        await act(async () => {
          first.resolve();
          await first.promise;
        });
        expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled();
        expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled();
        // Closing and reopening is an explicit new choice in the current owner.
        if (isMenuOpen()) await user.click(trigger);
        await user.click(trigger);
        await user.click(
          await screen.findByRole("menuitem", { name: /Rewrite plainly/ }),
        );
        await waitFor(() => expect(details).toBe(2));
        await act(async () => {
          second.resolve();
          await second.promise;
        });
        await waitFor(() =>
          expect(props.setSelectedQuickPrompt).toHaveBeenCalledOnce(),
        );
        expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith(
          "Rewrite this plainly",
        );
        await waitFor(() => expect(isMenuOpen()).toBe(false));
      } finally {
        await act(async () => {
          first.resolve();
          second.resolve();
          await Promise.all(gates.map((gate) => gate.promise));
        });
      }
    });
  },
);
