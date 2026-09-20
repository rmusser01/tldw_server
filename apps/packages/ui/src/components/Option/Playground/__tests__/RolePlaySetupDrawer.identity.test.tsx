import React from "react";
import {
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
import { RolePlaySetupDrawer } from "../RolePlaySetupDrawer";
import type { RolePlayState } from "../role-play-state";

const mocks = vi.hoisted(() => ({
  setSelectedAssistant: vi.fn(async () => undefined),
  updateSettings: vi.fn(async () => undefined),
  saveScene: vi.fn(async () => true),
  loadScene: vi.fn(),
  option: {
    historyId: "history-1",
    serverChatId: "server-1",
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
  useStoreMessageOption: (selector: (state: unknown) => unknown) =>
    selector(mocks.option),
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
    mocks.loadScene.mockResolvedValue({
      ...createDefaultActorSettings(),
      notes: "Original scene",
    });
    mocks.saveScene.mockResolvedValue(true);
    useActorStore.getState().reset();
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
    expect(onApply).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    await user.click(
      await screen.findByRole("button", { name: "Apply", exact: true }),
    );
    await waitFor(() => expect(onApply).toHaveBeenCalledOnce());
  });

  it("restores the saved scene when applying the other settings fails", async () => {
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
    expect(mocks.saveScene).toHaveBeenLastCalledWith(
      expect.objectContaining({
        settings: expect.objectContaining({ notes: "Original scene" }),
      }),
    );
    expect(useActorStore.getState().settings?.notes).not.toBe("Draft scene");
    expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(
      "Draft scene",
    );
  });
});
