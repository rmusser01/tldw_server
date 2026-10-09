import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createDefaultActorSettings } from "@/types/actor";
import { useActorStore } from "@/store/actor";
import { RolePlaySetupDrawer, type RolePlayApplyContext, type RolePlaySetupApplyPayload } from "../RolePlaySetupDrawer";
import type { RolePlayState } from "../role-play-state";
import { createStartupTemplateBundle } from "../startup-template-bundles";

const mocks = vi.hoisted(() => ({
  setSelectedAssistant: vi.fn(async () => undefined),
  updateSettings: vi.fn(async () => undefined),
  saveScene: vi.fn(async () => true),
  loadScene: vi.fn(),
  option: {
    historyId: "history-1" as string | null,
    serverChatId: "server-1" as string | null,
    serverChatAssistantKind: "character",
    serverChatAssistantId: "char-mira",
    serverChatCharacterId: "char-mira",
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatMetaLoaded: vi.fn(),
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key,
  }),
}));
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => React.useState(fallback),
}));
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    { kind: "character", id: "char-mira", name: "Mira" },
    mocks.setSelectedAssistant,
  ],
}));
vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign((selector: (state: unknown) => unknown) =>
    selector(mocks.option), { getState: () => mocks.option }),
}));
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: mocks.updateSettings,
  }),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    listAllCharacters: async () => [{ id: "char-ada", name: "Ada" }],
    listPersonaProfiles: async () => [
      { id: "persona-guide", name: "Guide Persona" },
    ],
  },
}));
vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChatWithCharacterFallback: mocks.loadScene,
  saveActorSettingsForChat: mocks.saveScene,
}));
vi.mock("../playground-features", () => ({
  PRESETS: [],
  SystemPromptTemplatesModal: () => null,
}));

const beforeState: RolePlayState = {
  active: true,
  identity: { kind: "character", id: "char-mira", name: "Mira" },
  behavior: null,
  scene: null,
  generationStyle: null,
  context: { pinnedCount: 0, hasExternalContext: false },
};

const renderDrawer = (
  props: Partial<React.ComponentProps<typeof RolePlaySetupDrawer>> = {},
) => {
  const onApply = vi.fn(async () => undefined);
  const onClose = vi.fn();
  const Harness = () => {
    const [open, setOpen] = React.useState(true);
    const returnFocusRef = React.useRef<HTMLButtonElement>(null);
    return (
      <>
        <button ref={returnFocusRef} onClick={() => setOpen(true)}>
          Setup trigger
        </button>
        <RolePlaySetupDrawer
          open={open}
          beforeState={beforeState}
          historyId="history-1"
          serverChatId="server-1"
          characterId="char-mira"
          returnFocusRef={returnFocusRef}
          onClose={() => {
            onClose();
            setOpen(false);
          }}
          onApply={onApply}
          {...props}
        />
      </>
    );
  };
  render(<Harness />);
  return { onApply, onClose };
};

const choosePersona = async () => {
  const user = userEvent.setup();
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Apply", exact: true }),
    ).toBeEnabled(),
  );
  await user.click(screen.getByTestId("character-select"));
  await user.click(await screen.findByRole("tab", { name: "Personas" }));
  await user.click(
    await screen.findByRole("button", { name: "Guide Persona", exact: true }),
  );
  return user;
};

describe("conversation setup identity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.option.historyId = "history-1";
    mocks.option.serverChatId = "server-1";
    mocks.loadScene.mockResolvedValue({
      ...createDefaultActorSettings(),
      notes: "Original scene",
    });
    mocks.saveScene.mockResolvedValue(true);
    useActorStore.getState().reset();
  });

  it("captures the notified destination after deferred preparation before saving scene", async () => {
    let resume!: () => void;
    const ready = new Promise<void>(resolve => { resume = resolve; });
    const onApply = vi.fn(async (_payload: RolePlaySetupApplyPayload, options?: RolePlayApplyContext) => {
      await ready;
      mocks.option.historyId = null;
      mocks.option.serverChatId = null;
      options?.onDestination?.({ historyId: null, serverChatId: null, isCurrent: () => true });
      return true;
    });
    const { onClose } = renderDrawer({ onApply });
    const user = await choosePersona();
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    expect(mocks.saveScene).not.toHaveBeenCalled();
    await act(async () => { resume(); });
    await waitFor(() => expect(mocks.saveScene).toHaveBeenCalledWith(expect.objectContaining({ historyId: null, serverChatId: null })));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("previews the staged Persona and applies it once without picker side effects", async () => {
    const { onApply } = renderDrawer();
    const user = await choosePersona();

    expect(
      within(
        screen.getByRole("region", { name: "Role-play preview" }),
      ).getByText("Guide Persona"),
    ).toBeInTheDocument();
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled();
    expect(mocks.option.setHistoryId).not.toHaveBeenCalled();
    expect(mocks.saveScene).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "Apply", exact: true }),
    );
    await waitFor(() => expect(onApply).toHaveBeenCalledOnce());
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        identitySelection: expect.objectContaining({
          kind: "persona",
          id: "persona-guide",
          name: "Guide Persona",
        }),
      }),
      expect.objectContaining({ isCurrent: expect.any(Function), onDestination: expect.any(Function) }),
    );
  });

  it.each(["Cancel", "Escape", "Close"])(
    "discards identity and scene changes on %s and returns focus",
    async (action) => {
      const { onApply, onClose } = renderDrawer();
      const originalActor = useActorStore.getState().settings;
      const user = await choosePersona();
      await user.type(
        screen.getByRole("textbox", { name: "Scene notes" }),
        " draft",
      );
      if (action === "Escape") await user.keyboard("{Escape}");
      else
        await user.click(
          screen.getByRole("button", { name: action, exact: true }),
        );

      if (action !== "Cancel") await user.click(await screen.findByRole("button", { name: "Discard", exact: true }));

      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Setup trigger" }),
        ).toHaveFocus(),
      );
      expect(onClose).toHaveBeenCalledOnce();
      expect(onApply).not.toHaveBeenCalled();
      expect(mocks.setSelectedAssistant).not.toHaveBeenCalled();
      expect(mocks.option.setHistoryId).not.toHaveBeenCalled();
      expect(mocks.saveScene).not.toHaveBeenCalled();
      expect(useActorStore.getState().settings).toEqual(originalActor);
      await user.click(screen.getByRole("button", { name: "Setup trigger" }));
      expect(screen.getByTestId("character-select")).toHaveAccessibleName(
        "Mira",
      );
    },
  );

  it.each(["Escape", "Close"])("retains the staged roleplay setup when %s dismissal is cancelled", async action => {
    const { onApply, onClose } = renderDrawer();
    const user = await choosePersona();
    const notes = screen.getByRole("textbox", { name: "Scene notes" });
    await user.type(notes, " retained draft");
    if (action === "Escape") await user.keyboard("{Escape}");
    else await user.click(screen.getByRole("button", { name: "Close", exact: true }));
    await user.click(await screen.findByRole("button", { name: "Keep editing", exact: true }));
    expect(notes).toHaveValue("Original scene retained draft");
    expect(screen.getByTestId("character-select")).toHaveAccessibleName("Guide Persona");
    expect({ closed: onClose.mock.calls.length, applied: onApply.mock.calls.length, writes: mocks.saveScene.mock.calls.length }).toEqual({ closed: 0, applied: 0, writes: 0 });
  });

  it("closes the nested Persona picker first when Escape is pressed in search", async () => {
    const { onClose } = renderDrawer();
    const user = userEvent.setup();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Apply", exact: true }),
      ).toBeEnabled(),
    );
    await user.click(screen.getByTestId("character-select"));
    await user.click(
      await screen.findByRole("textbox", {
        name: "Search characters and personas",
      }),
    );
    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.getByTestId("character-select")).toHaveAttribute(
        "aria-expanded",
        "false",
      ),
    );
    expect(onClose).not.toHaveBeenCalled();
    expect(
      screen.getByRole("dialog", { name: "Role-play setup" }),
    ).toBeInTheDocument();
  });

  it("discards on Escape when a shell capture listener blocks bubble handlers", async () => {
    const shellCapture = (event: KeyboardEvent) => {
      if (event.key === "Escape") event.stopPropagation();
    };
    window.addEventListener("keydown", shellCapture, true);
    try {
      const { onClose, onApply } = renderDrawer();
      const user = await choosePersona();
      await user.click(screen.getByRole("textbox", { name: "Scene notes" }));
      await user.keyboard("{Escape}");
      await user.click(await screen.findByRole("button", { name: "Discard", exact: true }));
      await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
      expect(onApply).not.toHaveBeenCalled();
      expect(mocks.setSelectedAssistant).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("keydown", shellCapture, true);
    }
  });

  it("saves the staged identity in a reusable setup without applying it", async () => {
    const onSaveRolePlaySetup = vi.fn();
    const { onApply } = renderDrawer({
      onSaveRolePlaySetup,
      onSavedSetupDraftNameChange: vi.fn(),
      onPreviewSavedSetup: vi.fn(),
      onApplySavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn(),
    });
    const user = await choosePersona();
    await user.click(
      screen.getByRole("button", { name: "Save setup", exact: true }),
    );
    expect(onSaveRolePlaySetup).toHaveBeenCalledWith(
      expect.objectContaining({
        rolePlay: expect.objectContaining({
          identity: {
            kind: "persona",
            id: "persona-guide",
            name: "Guide Persona",
          },
        }),
      }),
    );
    expect(onApply).not.toHaveBeenCalled();
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled();
  });

  it("keeps the draft open for retry when scene storage returns false", async () => {
    mocks.saveScene.mockResolvedValueOnce(false);
    const { onApply, onClose } = renderDrawer();
    const user = await choosePersona();
    await user.click(
      screen.getByRole("button", { name: "Apply", exact: true }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /could not be saved/i,
    );
    expect(onApply).toHaveBeenCalledOnce();
    expect(onClose).not.toHaveBeenCalled();
    await user.click(
      await screen.findByRole("button", { name: "Apply", exact: true }),
    );
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(onApply).toHaveBeenCalledOnce();
    expect(mocks.saveScene).toHaveBeenCalledTimes(2);
  });

  it("retries an edited scene without applying the accepted identity again", async () => {
    mocks.saveScene.mockResolvedValueOnce(false);
    const { onApply, onClose } = renderDrawer();
    const user = await choosePersona();
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    await screen.findByRole("alert");
    fireEvent.change(screen.getByRole("textbox", { name: "Scene notes" }), {
      target: { value: "Corrected scene" },
    });
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(onApply).toHaveBeenCalledOnce();
    expect(mocks.saveScene).toHaveBeenLastCalledWith(expect.objectContaining({
      settings: expect.objectContaining({ notes: "Corrected scene" }),
    }));
  });

  it("applies a changed identity proposal after a partial scene failure", async () => {
    mocks.saveScene.mockResolvedValueOnce(false);
    const { onApply, onClose } = renderDrawer();
    const user = await choosePersona();
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    await screen.findByRole("alert");
    await user.click(screen.getByRole("button", { name: "Clear identity", exact: true }));
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(onApply).toHaveBeenCalledTimes(2);
    expect(onApply).toHaveBeenLastCalledWith(expect.objectContaining({ clearIdentity: true }), expect.anything());
  });

  it("reports a saved setup complete only after scene Retry is acknowledged", async () => {
    const saved = createStartupTemplateBundle({
      name: "Saved scene Retry",
      source: "role-play-setup",
      selectedModel: null,
      systemPrompt: "",
      rolePlay: {
        source: "role-play-setup",
        identity: null,
        behavior: null,
        scene: {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Saved exact scene"
        },
        generation: null,
        context: null
      }
    })
    const onApplySavedSetup = vi.fn(async () => true)
    const onSavedSetupApplied = vi.fn()
    const props = {
      savedRolePlaySetups: [saved],
      onApplySavedSetup,
      onSavedSetupApplied,
      onSaveRolePlaySetup: vi.fn(),
      onSavedSetupDraftNameChange: vi.fn(),
      onPreviewSavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn()
    }
    mocks.saveScene.mockResolvedValueOnce(false)
    const { onClose } = renderDrawer(props)
    const user = userEvent.setup()
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Apply {{name}}" })).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: "Apply {{name}}" }))
    await screen.findByRole("alert")
    expect(onSavedSetupApplied).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Apply {{name}}" }))
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(onSavedSetupApplied).toHaveBeenCalledOnce()
    expect(onApplySavedSetup).toHaveBeenCalledOnce()
    expect(mocks.saveScene).toHaveBeenCalledTimes(2)
    expect(useActorStore.getState().settings.notes).toBe("Saved exact scene")
  })

  it("keeps a saved application's exact ID and notified destination through model failure and delayed own-view publication", async () => {
    const saved = createStartupTemplateBundle({
      name: "Own destination Retry",
      source: "role-play-setup",
      selectedModel: null,
      systemPrompt: "",
      rolePlay: {
        source: "role-play-setup",
        identity: null,
        behavior: null,
        scene: null,
        generation: null,
        context: null
      }
    })
    let reject!: (error: Error) => void
    const held = new Promise<boolean>((_resolve, fail) => {
      reject = fail
    })
    const applicationIds: unknown[] = []
    const onApplySavedSetup = vi.fn((_setup, context?: RolePlayApplyContext) => {
      applicationIds.push(
        (context as RolePlayApplyContext & { applicationId?: object })
          ?.applicationId
      )
      mocks.option.historyId = null
      mocks.option.serverChatId = null
      context?.onDestination?.({
        historyId: null,
        serverChatId: null,
        isCurrent: () => true
      })
      return applicationIds.length === 1 ? held : Promise.resolve(true)
    })
    const onClose = vi.fn()
    let publish!: () => void
    const Harness = () => {
      const [ids, setIds] = React.useState({
        historyId: "history-1" as string | null,
        serverChatId: "server-1" as string | null
      })
      React.useLayoutEffect(() => {
        publish = () => setIds({ historyId: null, serverChatId: null })
      }, [])
      return (
        <RolePlaySetupDrawer
          open
          beforeState={beforeState}
          {...ids}
          characterId="char-mira"
          onClose={onClose}
          onApply={vi.fn()}
          savedRolePlaySetups={[saved]}
          onApplySavedSetup={onApplySavedSetup}
          onSaveRolePlaySetup={vi.fn()}
          onSavedSetupDraftNameChange={vi.fn()}
          onPreviewSavedSetup={vi.fn()}
          onRenameSavedSetup={vi.fn()}
          onDeleteSavedSetup={vi.fn()}
        />
      )
    }
    const view = render(<Harness />)
    const user = userEvent.setup()
    try {
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Apply {{name}}" })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: "Apply {{name}}" }))
      await waitFor(() => expect(onApplySavedSetup).toHaveBeenCalledOnce())
      await act(async () => {
        reject(new Error("Model device write failed"))
        await held.catch(() => undefined)
      })
      await screen.findByRole("alert")
      act(() => publish())
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Apply {{name}}" })
        ).toBeEnabled()
      )
      expect(mocks.loadScene).toHaveBeenCalledOnce()
      await user.click(screen.getByRole("button", { name: "Apply {{name}}" }))
      await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
      expect(applicationIds[0]).toEqual(expect.any(Object))
      expect(applicationIds[1]).toBe(applicationIds[0])
      expect(mocks.saveScene).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ historyId: null, serverChatId: null })
      )
    } finally {
      view.unmount()
      reject(new Error("Test cleanup"))
      await held.catch(() => undefined)
    }
  })

  it.each(["Escape", "Close"])(
    "admits dirty %s dismissal after the failed model operation is superseded",
    async (action) => {
      let modelRevision = 0
      const onApply = vi.fn(
        async (
          _payload: RolePlaySetupApplyPayload,
          context?: RolePlayApplyContext
        ) => {
          const captured = modelRevision
          context?.onDestination?.({
            historyId: "history-1",
            serverChatId: "server-1",
            isCurrent: () => modelRevision === captured
          })
          throw new Error("Model preference save failed")
        }
      )
      const { onClose } = renderDrawer({ onApply })
      const originalActor = useActorStore.getState().settings
      const user = await choosePersona()
      await user.type(
        screen.getByRole("textbox", { name: "Scene notes" }),
        " retained"
      )
      await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
      await screen.findByRole("alert")
      modelRevision++
      if (action === "Escape") await user.keyboard("{Escape}")
      else
        await user.click(
          screen.getByRole("button", { name: "Close", exact: true })
        )
      await user.click(
        await screen.findByRole("button", { name: "Discard", exact: true })
      )
      await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
      expect(onApply).toHaveBeenCalledOnce()
      expect(mocks.saveScene).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings).toEqual(originalActor)
    }
  )

  it("does not close a saved setup after its completion callback replaces the selected chat", async () => {
    const saved = createStartupTemplateBundle({
      name: "Completion replacement",
      source: "role-play-setup",
      selectedModel: null,
      systemPrompt: "",
      rolePlay: {
        source: "role-play-setup",
        identity: null,
        behavior: null,
        scene: null,
        generation: null,
        context: null
      }
    })
    const onSavedSetupApplied = vi.fn(() => {
      mocks.option.serverChatId = "replacement-chat"
    })
    const props = {
      savedRolePlaySetups: [saved],
      onApplySavedSetup: vi.fn(async () => true),
      onSavedSetupApplied,
      onSaveRolePlaySetup: vi.fn(),
      onSavedSetupDraftNameChange: vi.fn(),
      onPreviewSavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn()
    }
    const { onClose } = renderDrawer(props)
    const user = userEvent.setup()
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Apply {{name}}" })).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: "Apply {{name}}" }))
    await waitFor(() => expect(mocks.saveScene).toHaveBeenCalledOnce())
    expect(onClose).not.toHaveBeenCalled()
    expect(onSavedSetupApplied).toHaveBeenCalledOnce()
  })

  it("keeps the saved scene untouched when applying the other settings fails", async () => {
    const originalActor = useActorStore.getState().settings;
    const onApply = vi
      .fn()
      .mockRejectedValueOnce(new Error("settings unavailable"));
    const { onClose } = renderDrawer({ onApply });
    const user = await choosePersona();
    fireEvent.change(screen.getByRole("textbox", { name: "Scene notes" }), {
      target: { value: "Draft scene" },
    });
    await user.click(
      screen.getByRole("button", { name: "Apply", exact: true }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /could not be applied/i,
    );
    expect(onClose).not.toHaveBeenCalled();
    expect(mocks.saveScene).not.toHaveBeenCalled();
    expect(useActorStore.getState().settings).toEqual(originalActor);
    expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(
      "Draft scene",
    );
  });
});
