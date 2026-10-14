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
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import { useStoreMessageOption } from "@/store/option"
import type {
  ActorSceneRead,
  ActorSceneRecord
} from "@/services/actor-settings"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import {
  captureActorSettingsTarget,
  type ActorSettingsTarget
} from "@/utils/actor-settings-target"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { useActorStore } from "@/store/actor";
import { RolePlaySetupDrawer, type RolePlayApplyContext, type RolePlaySetupApplyPayload } from "../RolePlaySetupDrawer";
import type { RolePlayState } from "../role-play-state";
import { createStartupTemplateBundle } from "../startup-template-bundles";

const mocks = vi.hoisted(() => ({
  setSelectedAssistant: vi.fn(async () => undefined),
  updateSettings: vi.fn(async () => undefined),
  saveScene:
    vi.fn<
      typeof import("@/services/actor-settings").saveActorSettingsForChat
    >(),
  loadScene:
    vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>(),
  captureCurrentActorSceneTarget:
    vi.fn<
      typeof import("@/hooks/chat/actor-scene-source").captureCurrentActorSceneTarget
    >(),
  setHistoryId: vi.fn<(id: string | null) => void>(),
  historySelection: null as Record<string, unknown> | null,
  accountRevision: 0
}))
// Picker catalog/account/native-view acceptance are declared unit boundaries.
// The actual option store publishes typed addresses; source/session checks remain.
vi.mock("@/hooks/chat/useHistorySelection", async importOriginal => ({
  ...await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => mocks.historySelection
}))

vi.mock("@/hooks/chat/actor-scene-source", async (original) => ({
  ...await original<typeof import("@/hooks/chat/actor-scene-source")>(),
  captureCurrentActorSceneTarget: mocks.captureCurrentActorSceneTarget
}))

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
  getActorSceneForChat: mocks.loadScene,
  saveActorSettingsForChat: mocks.saveScene
}))
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

const declaredSceneScope: ServicePromptRequestScope = {
  config: {
    serverUrl: "https://drawer-identity-unit.example",
    authMode: "single-user"
  },
  userId: 1
}
const declaredNativeOwnerKey = "drawer-identity-unit-owner"
const originalScene = (): ActorSettings => ({
  ...createDefaultActorSettings(),
  notes: "Original scene"
})
const originalRecord = (): ActorSceneRecord => ({
  value: structuredClone(originalScene()),
  updatedAt: 41
})
const sceneRead = (target: ActorSettingsTarget): ActorSceneRead => ({
  target,
  originalRecord: originalRecord(),
  settings: originalScene(),
  origin: "scene",
  profileUnavailable: false
})
const declaredSceneTarget = (
  commandIsCurrent: () => boolean = () => true
): ActorSettingsTarget | null => {
  const source = useStoreMessageOption.getState()
  const address = source.chatDraftAddress
  const accountRevision = mocks.accountRevision
  if (
    !address ||
    (address.kind === "native" && source.serverChatId !== address.id) ||
    (address.kind === "fresh" &&
      (source.serverChatId !== null ||
        (source.historyId !== null && source.historyId !== "temp")))
  )
    return null
  const accountIsCurrent = () => mocks.accountRevision === accountRevision
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    return (
      commandIsCurrent() &&
      accountIsCurrent() &&
      current.chatDraftAddress === address &&
      current.chatDraftRevision === source.chatDraftRevision &&
      current.historyId === source.historyId &&
      current.serverChatId === source.serverChatId
    )
  }
  return captureActorSettingsTarget({
    address,
    requestScope: declaredSceneScope,
    accountIsCurrent,
    isCurrent,
    owner:
      address.kind === "native"
        ? {
            kind: "native",
            conversation_id: address.id,
            owner_key: address.ownerKey,
            request_scope: declaredSceneScope,
            scope: address.scope,
            validate_lease: accountIsCurrent
          }
        : null,
    view:
      address.kind === "native"
        ? {
            owner_key: address.ownerKey,
            conversation_id: address.id,
            view_session_id: "declared-identity-view",
            interpretation: { kind: "parent_graph_v1" },
            cursor: { kind: "empty" },
            selection_revision: 0
          }
        : null
  })
}
const originalSceneTarget = () => {
  const target = mocks.loadScene.mock.calls[0]?.[0].target
  if (!target)
    throw new Error("The fixture must reach its original qualified Scene read")
  return target
}
type DeclaredDestination = Parameters<
  NonNullable<RolePlayApplyContext["onDestination"]>
>[0] & { sceneTarget: ActorSettingsTarget }
const issueFreshDestination = (
  original: ActorSettingsTarget
): DeclaredDestination => {
  const source = useStoreMessageOption.getState()
  const account = original.account
  if (!original.isCurrent() || !account?.isCurrent())
    throw new Error("Original declared owner retired")
  const address = freshChatDraftAddress(false)
  const revision = source.chatDraftRevision + 1
  if (
    !source.setChatDraftAddress(address, {
      sourceAddress: source.chatDraftAddress,
      expectedRevision: source.chatDraftRevision
    })
  )
    throw new Error("The fixture's own fresh address was not acknowledged")
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    return (
      account.isCurrent() &&
      current.chatDraftAddress === address &&
      current.chatDraftRevision === revision &&
      current.historyId === null &&
      current.serverChatId === null
    )
  }
  const sceneTarget = captureActorSettingsTarget({
    address,
    requestScope: account.requestScope,
    accountIsCurrent: account.isCurrent,
    isCurrent
  })
  if (
    !sceneTarget ||
    sceneTarget.accountKey !== original.accountKey ||
    !sceneTarget.isCurrent()
  )
    throw new Error("Missing exact acknowledged fresh Scene destination")
  return { historyId: null, serverChatId: null, isCurrent, sceneTarget }
}
const physicalSetHistoryId = useStoreMessageOption.getState().setHistoryId

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
  const view = render(<Harness />);
  return { onApply, onClose, rerender: () => view.rerender(<Harness />) };
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
    vi.clearAllMocks()
    mocks.accountRevision = 0
    mocks.historySelection = null
    useStoreMessageOption.setState({
      historyId: "history-1",
      serverChatId: "server-1",
      chatDraftAddress: {
        kind: "native",
        id: "server-1",
        ownerKey: declaredNativeOwnerKey,
        scope: { type: "global" }
      },
      chatDraftRevision: useStoreMessageOption.getState().chatDraftRevision + 1,
      setHistoryId: (id) => {
        mocks.setHistoryId(id)
        physicalSetHistoryId(id)
      },
      serverChatAssistantKind: "character",
      serverChatAssistantId: "char-mira",
      serverChatCharacterId: "char-mira"
    })
    mocks.captureCurrentActorSceneTarget
      .mockReset()
      .mockImplementation((_controller, commandIsCurrent) =>
        declaredSceneTarget(commandIsCurrent)
      )
    mocks.loadScene
      .mockReset()
      .mockImplementation(async ({ target }) => sceneRead(target))
    mocks.saveScene.mockReset().mockResolvedValue(true)
    useActorStore.getState().reset()
  })

  it("keeps an already staged persona when protected history appears and explains unavailable Apply", async () => {
    const { onApply, rerender } = renderDrawer()
    const user = await choosePersona()
    mocks.historySelection = { getSignal: () => undefined, getCurrent: () => ({ owner: null, view: null }) }
    rerender()
    const apply = screen.getByRole("button", { name: "Apply", exact: true })
    expect(apply).toBeDisabled()
    expect(apply).toHaveAccessibleDescription(/tracked persona.*unavailable.*overlay/i)
    expect(screen.getByTestId("character-select")).toHaveAccessibleName("Guide Persona")
    await user.click(apply)
    expect(onApply).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.saveScene).not.toHaveBeenCalled()
  })

  it("captures the notified destination after deferred preparation before saving scene", async () => {
    let resume!: () => void
    const ready = new Promise<void>((resolve) => {
      resume = resolve
    })
    let destination!: DeclaredDestination
    const onApply = vi.fn(
      async (
        _payload: RolePlaySetupApplyPayload,
        options?: RolePlayApplyContext
      ) => {
        await ready
        destination = issueFreshDestination(originalSceneTarget())
        options?.onDestination?.(destination)
        return true
      }
    )
    const { onClose } = renderDrawer({ onApply })
    const user = await choosePersona()
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
    expect(mocks.saveScene).not.toHaveBeenCalled()
    await act(async () => {
      resume()
    })
    await waitFor(() =>
      expect(mocks.saveScene).toHaveBeenCalledWith(
        expect.objectContaining({
          historyId: null,
          serverChatId: null,
          target: destination.sceneTarget,
          expectedRecord: null,
          validate_lease: expect.any(Function)
        })
      )
    )
    expect(onClose).toHaveBeenCalledOnce()
    expect(useActorStore.getState().applied?.target).toBe(destination.sceneTarget)
    expect(destination.sceneTarget.isCurrent()).toBe(true)
  })

  it("previews the staged Persona and applies it once without picker side effects", async () => {
    const { onApply } = renderDrawer()
    const user = await choosePersona()

    expect(
      within(screen.getByRole("region", { name: "Role-play preview" })).getByText(
        "Guide Persona"
      )
    ).toBeInTheDocument()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.setHistoryId).not.toHaveBeenCalled()
    expect(mocks.saveScene).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
    await waitFor(() => expect(onApply).toHaveBeenCalledOnce())
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        identitySelection: expect.objectContaining({
          kind: "persona",
          id: "persona-guide",
          name: "Guide Persona"
        })
      }),
      expect.objectContaining({
        isCurrent: expect.any(Function),
        onDestination: expect.any(Function)
      })
    )
  })

  it.each(["Cancel", "Escape", "Close"])(
    "discards identity and scene changes on %s and returns focus",
    async (action) => {
      const { onApply, onClose } = renderDrawer()
      const originalActor = useActorStore.getState().settings
      const user = await choosePersona()
      await user.type(
        screen.getByRole("textbox", { name: "Scene notes" }),
        " draft"
      )
      if (action === "Escape") await user.keyboard("{Escape}")
      else
        await user.click(
          screen.getByRole("button", { name: action, exact: true })
        )

      if (action !== "Cancel")
        await user.click(
          await screen.findByRole("button", { name: "Discard", exact: true })
        )

      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Setup trigger" })
        ).toHaveFocus()
      )
      expect(onClose).toHaveBeenCalledOnce()
      expect(onApply).not.toHaveBeenCalled()
      expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
      expect(mocks.setHistoryId).not.toHaveBeenCalled()
      expect(mocks.saveScene).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings).toEqual(originalActor)
      await user.click(screen.getByRole("button", { name: "Setup trigger" }))
      expect(screen.getByTestId("character-select")).toHaveAccessibleName("Mira")
    }
  )

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
    mocks.saveScene.mockResolvedValueOnce(false)
    const { onApply, onClose } = renderDrawer()
    const user = await choosePersona()
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /could not be saved/i
    )
    expect(onApply).toHaveBeenCalledOnce()
    expect(onClose).not.toHaveBeenCalled()
    await user.click(
      await screen.findByRole("button", { name: "Apply", exact: true })
    )
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(onApply).toHaveBeenCalledOnce()
    expect(mocks.saveScene).toHaveBeenCalledTimes(2)
    for (const [save] of mocks.saveScene.mock.calls) {
      expect(save.target).toBe(originalSceneTarget())
      expect(save.expectedRecord).toEqual(originalRecord())
      expect(save.validate_lease).toEqual(expect.any(Function))
    }
    expect(useActorStore.getState().applied?.target).toBe(originalSceneTarget())
    expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
  })

  it("retries an edited scene without applying the accepted identity again", async () => {
    mocks.saveScene.mockResolvedValueOnce(false)
    const { onApply, onClose } = renderDrawer()
    const user = await choosePersona()
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
    await screen.findByRole("alert")
    fireEvent.change(screen.getByRole("textbox", { name: "Scene notes" }), {
      target: { value: "Corrected scene" }
    })
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }))
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(onApply).toHaveBeenCalledOnce()
    expect(mocks.saveScene).toHaveBeenLastCalledWith(
      expect.objectContaining({
        settings: expect.objectContaining({ notes: "Corrected scene" }),
        target: originalSceneTarget(),
        expectedRecord: originalRecord(),
        validate_lease: expect.any(Function)
      })
    )
    expect(mocks.saveScene.mock.calls[0][0].target).toBe(originalSceneTarget())
    expect(mocks.saveScene.mock.calls[0][0].expectedRecord).toEqual(
      originalRecord()
    )
  })

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
    expect(useActorStore.getState().settings?.notes).toBe("Saved exact scene")
    expect(useActorStore.getState().applied?.target).toBe(originalSceneTarget())
    expect(useActorStore.getState().applied?.settings).toBe(
      useActorStore.getState().settings
    )
    expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
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
    const applicationIds: Array<object | undefined> = []
    let destination: DeclaredDestination | null = null
    const onApplySavedSetup = vi.fn((_setup, context?: RolePlayApplyContext) => {
      applicationIds.push(context?.applicationId)
      if (!destination) destination = issueFreshDestination(originalSceneTarget())
      context?.onDestination?.(destination)
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
        expect.objectContaining({
          historyId: null,
          serverChatId: null,
          target: destination?.sceneTarget,
          expectedRecord: null,
          validate_lease: expect.any(Function)
        })
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
      useStoreMessageOption.setState({ serverChatId: "replacement-chat" })
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
