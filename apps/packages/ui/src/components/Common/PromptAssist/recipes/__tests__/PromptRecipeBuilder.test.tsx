import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PromptRecipeBuilder } from "../PromptRecipeBuilder";
import { CLEAR_TASK_RECIPE } from "../built-in-recipes";

const state = vi.hoisted(() => ({ online: true, privateMode: false }));
const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(),
  markPromptSyncError: vi.fn(),
  permanentlyDeletePrompt: vi.fn(),
  restorePromptSnapshot: vi.fn(),
  savePrompt: vi.fn(),
  updatePrompt: vi.fn(),
  shouldAutoSyncWorkspacePrompts: vi.fn(),
  autoSyncPrompt: vi.fn(),
}));

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => state.online,
}));

vi.mock("@/utils/is-private-mode", () => ({
  get isFireFoxPrivateMode() {
    return state.privateMode;
  },
}));

vi.mock("@/db/dexie/helpers", () => ({
  getAllPrompts: mocks.getAllPrompts,
  markPromptSyncError: mocks.markPromptSyncError,
  permanentlyDeletePrompt: mocks.permanentlyDeletePrompt,
  restorePromptSnapshot: mocks.restorePromptSnapshot,
  savePrompt: mocks.savePrompt,
  updatePrompt: mocks.updatePrompt,
}));

vi.mock("@/services/prompt-sync", () => ({
  shouldAutoSyncWorkspacePrompts: mocks.shouldAutoSyncWorkspacePrompts,
  autoSyncPrompt: mocks.autoSyncPrompt,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));

const capabilities = (
  recipeSupported: boolean,
  createAuthorized: boolean | null = true,
  updateAuthorized: boolean | null = true,
) => ({
  availability: "available" as const,
  prompt_improvement_v1: { supported: true, limits: null },
  single_text_recipe_v2: { supported: recipeSupported },
  prompt_persistence: {
    create_authorized: createAuthorized,
    update_authorized: updateAuthorized,
  },
});

const recipeRecord = (id: string, name: string, target: "system" | "user") => ({
  id,
  title: name,
  name,
  content: "<task>Hello {{task}}</task>",
  is_system: target === "system",
  createdAt: 1,
  syncStatus: "synced",
  promptFormat: "structured",
  promptSchemaVersion: 2,
  structuredPromptDefinition: {
    ...structuredClone(CLEAR_TASK_RECIPE.definition),
    assembly_config: {
      ...structuredClone(CLEAR_TASK_RECIPE.definition.assembly_config),
      target_role: target,
    },
    blocks: CLEAR_TASK_RECIPE.definition.blocks?.map((block) => ({
      ...structuredClone(block),
      role: target,
    })),
  },
});

const renderBuilder = (
  props: Partial<React.ComponentProps<typeof PromptRecipeBuilder>> = {},
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const onApply = vi.fn();
  const onBack = vi.fn();
  render(
    <QueryClientProvider client={queryClient}>
      <PromptRecipeBuilder
        target="system"
        capabilities={capabilities(true)}
        onApply={onApply}
        onBack={onBack}
        {...props}
      />
    </QueryClientProvider>,
  );
  return { onApply, onBack, queryClient };
};

describe("PromptRecipeBuilder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state.online = true;
    state.privateMode = false;
    mocks.getAllPrompts.mockResolvedValue([
      recipeRecord("system-id", "System saved", "system"),
      recipeRecord("user-id", "User saved", "user"),
      {
        ...recipeRecord("invalid-id", "Invalid recipe", "system"),
        promptSchemaVersion: 3,
      },
    ]);
    mocks.savePrompt.mockResolvedValue({ id: "new-exact-id" });
    mocks.updatePrompt.mockImplementation(async ({ id }) => id);
    mocks.permanentlyDeletePrompt.mockResolvedValue(undefined);
    mocks.markPromptSyncError.mockResolvedValue(undefined);
    mocks.restorePromptSnapshot.mockResolvedValue(undefined);
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(false);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: true,
      localId: "new-exact-id",
      syncStatus: "synced",
    });
  });

  it("requires positive create and update authorization independently", async () => {
    const user = userEvent.setup();
    renderBuilder({ capabilities: capabilities(true, true, false) });
    await screen.findByRole("combobox", { name: "Recipe source" });

    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeEnabled();
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:system-id");
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Update recipe" }),
    ).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("not authorized");
  });

  it("allows an authorized Update while an unauthorized Save stays disabled", async () => {
    const user = userEvent.setup();
    renderBuilder({ capabilities: capabilities(true, false, true) });
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:system-id");

    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Update recipe" }),
    ).toBeEnabled();
  });

  it("treats an old response without authorization as unknown and fail-closed", async () => {
    const user = userEvent.setup();
    const oldCapabilities = capabilities(true) as Omit<
      ReturnType<typeof capabilities>,
      "prompt_persistence"
    > & { prompt_persistence?: never };
    delete (oldCapabilities as { prompt_persistence?: unknown })
      .prompt_persistence;
    renderBuilder({ capabilities: oldCapabilities });

    expect(await screen.findByRole("status")).toHaveTextContent(
      "authorization could not be confirmed",
    );
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Task (not saved)",
      }),
      "valid local task",
    );
    expect(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    ).toBeEnabled();
  });

  it("strictly quarantines invalid v2 records and filters saved sources by target", async () => {
    renderBuilder();

    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    expect(
      within(source).queryByRole("option", { name: "User saved" }),
    ).toBeNull();
    expect(
      within(source).queryByRole("option", { name: "Invalid recipe" }),
    ).toBeNull();
  });

  it("filters the user target and applies the exact saved-recipe preview", async () => {
    const user = userEvent.setup();
    const { onApply } = renderBuilder({ target: "user_message" });
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "User saved" }),
      ).toBeTruthy(),
    );
    expect(
      within(source).queryByRole("option", { name: "System saved" }),
    ).toBeNull();
    await user.selectOptions(source, "saved:user-id");
    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Task (not saved)",
      }),
      "saved target",
    );
    const preview = screen.getByRole("textbox", {
      name: "Compiled prompt preview",
    }) as HTMLTextAreaElement;
    const exactPreview = preview.value;
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" }),
    );

    expect(onApply).toHaveBeenCalledWith(exactPreview);
    expect(mocks.updatePrompt).not.toHaveBeenCalled();
  });

  it("applies the exact local preview without persistence", async () => {
    const user = userEvent.setup();
    const { onApply } = renderBuilder({ capabilities: undefined });

    await user.type(
      await screen.findByRole("textbox", {
        name: "Current value for Task (not saved)",
      }),
      "ship 🚀",
    );
    const preview = screen.getByRole("textbox", {
      name: "Compiled prompt preview",
    }) as HTMLTextAreaElement;
    await user.click(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    );

    expect(onApply).toHaveBeenCalledWith(preview.value);
    expect(mocks.savePrompt).not.toHaveBeenCalled();
    expect(mocks.updatePrompt).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
  });

  it("saves with a new exact id and invalidates both existing prompt queries", async () => {
    const user = userEvent.setup();
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");

    await screen.findByRole("combobox", { name: "Recipe source" });
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    await waitFor(() => expect(mocks.savePrompt).toHaveBeenCalledTimes(1));
    expect(mocks.shouldAutoSyncWorkspacePrompts).toHaveBeenCalledTimes(1);
    expect(mocks.autoSyncPrompt).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["getAllPromptsForSelect"],
    });
  });

  it("updates the exact selected saved id and preserves its record metadata", async () => {
    const user = userEvent.setup();
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:system-id");
    await user.click(screen.getByRole("button", { name: "Update recipe" }));

    await waitFor(() => expect(mocks.updatePrompt).toHaveBeenCalledTimes(1));
    expect(mocks.updatePrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "system-id",
        title: "System saved",
        syncStatus: "synced",
        promptFormat: "structured",
        promptSchemaVersion: 2,
      }),
    );
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
  });

  it("keeps transient-only sync failure as locally pending", async () => {
    const user = userEvent.setup();
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "new-exact-id",
      syncStatus: "pending",
      failureKind: "transient",
      error: "temporary",
    });
    renderBuilder();
    await screen.findByRole("combobox", { name: "Recipe source" });
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(
      await screen.findByText(/saved locally and will sync/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Could not save/)).not.toBeInTheDocument();
    expect(mocks.permanentlyDeletePrompt).not.toHaveBeenCalled();
    expect(mocks.restorePromptSnapshot).not.toHaveBeenCalled();
  });

  it("retains an uncertain Save locally, invalidates both queries, and locks retry", async () => {
    const user = userEvent.setup();
    const records = [recipeRecord("system-id", "System saved", "system")];
    mocks.getAllPrompts.mockImplementation(async () => structuredClone(records));
    mocks.savePrompt.mockImplementation(async (fields) => {
      const saved = {
        ...recipeRecord("uncertain-new", fields.title, "system"),
        ...fields,
        id: "uncertain-new",
        syncStatus: "local",
      };
      records.push(saved);
      return saved;
    });
    mocks.markPromptSyncError.mockImplementation(async (id) => {
      const record = records.find((item) => item.id === id);
      if (record) record.syncStatus = "error";
    });
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "uncertain-new",
      syncStatus: "pending",
      failureKind: "invalid_server_payload",
      error: "missing response identity",
    });
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });

    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(
      await screen.findByText(/saved locally.*server outcome.*not.*verified/i),
    ).toBeInTheDocument();
    expect(records.find((item) => item.id === "uncertain-new")?.syncStatus).toBe(
      "error",
    );
    expect(mocks.permanentlyDeletePrompt).not.toHaveBeenCalled();
    expect(mocks.restorePromptSnapshot).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["getAllPromptsForSelect"],
    });
    const save = screen.getByRole("button", { name: "Save as new recipe" });
    expect(save).toBeDisabled();
    await user.click(save);
    expect(mocks.savePrompt).toHaveBeenCalledTimes(1);

    await user.selectOptions(source, "built_in:blank");
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
  });

  it("never rolls back an uncertain remote result when marking local error fails", async () => {
    const user = userEvent.setup();
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "new-exact-id",
      syncStatus: "pending",
      failureKind: "invalid_server_payload",
      error: "missing response identity",
    });
    mocks.markPromptSyncError.mockRejectedValue(new Error("storage blocked"));
    renderBuilder();
    await screen.findByRole("combobox", { name: "Recipe source" });

    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(
      await screen.findByText(/saved locally.*server outcome.*not.*verified/i),
    ).toBeInTheDocument();
    expect(mocks.permanentlyDeletePrompt).not.toHaveBeenCalled();
    expect(mocks.restorePromptSnapshot).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
  });

  it("retains an uncertain Update, marks it error, and locks both writes", async () => {
    const user = userEvent.setup();
    const records = [recipeRecord("system-id", "System saved", "system")];
    mocks.getAllPrompts.mockImplementation(async () => structuredClone(records));
    mocks.updatePrompt.mockImplementation(async (fields) => {
      records[0] = { ...records[0], ...structuredClone(fields) };
      return fields.id;
    });
    mocks.markPromptSyncError.mockImplementation(async (id) => {
      const record = records.find((item) => item.id === id);
      if (record) record.syncStatus = "error";
    });
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "system-id",
      syncStatus: "pending",
      failureKind: "invalid_server_payload",
      error: "malformed response",
    });
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:system-id");
    await user.clear(screen.getByRole("textbox", { name: "Block name" }));
    await user.type(
      screen.getByRole("textbox", { name: "Block name" }),
      "Locally retained change",
    );

    await user.click(screen.getByRole("button", { name: "Update recipe" }));

    expect(
      await screen.findByText(/updated locally.*server outcome.*not.*verified/i),
    ).toBeInTheDocument();
    expect(records[0].syncStatus).toBe("error");
    expect(mocks.restorePromptSnapshot).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["getAllPromptsForSelect"],
    });
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    const update = screen.getByRole("button", { name: "Update recipe" });
    expect(update).toBeDisabled();
    await user.click(update);
    expect(mocks.updatePrompt).toHaveBeenCalledTimes(1);
  });

  it("keeps a reopened error-state recipe write-locked while Apply stays local", async () => {
    const user = userEvent.setup();
    mocks.getAllPrompts.mockResolvedValue([
      {
        ...recipeRecord("error-id", "Unverified recipe", "system"),
        syncStatus: "error",
      },
    ]);
    const { onApply } = renderBuilder();
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "Unverified recipe" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:error-id");

    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Update recipe" }),
    ).toBeDisabled();
    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Task (not saved)",
      }),
      "Apply remains available",
    );
    await user.click(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    );
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(mocks.savePrompt).not.toHaveBeenCalled();
    expect(mocks.updatePrompt).not.toHaveBeenCalled();
  });

  it("rolls back a non-transient Save, invalidates both queries, and retries without a duplicate", async () => {
    const user = userEvent.setup();
    const records = [recipeRecord("system-id", "System saved", "system")];
    let nextId = 0;
    mocks.getAllPrompts.mockImplementation(async () =>
      structuredClone(records),
    );
    mocks.savePrompt.mockImplementation(async (fields) => {
      const id = `new-${++nextId}`;
      const saved = {
        ...recipeRecord(id, fields.title, "system"),
        ...fields,
        id,
      };
      records.push(saved);
      return saved;
    });
    mocks.permanentlyDeletePrompt.mockImplementation(async (id) => {
      records.splice(
        records.findIndex((record) => record.id === id),
        1,
      );
    });
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt
      .mockResolvedValueOnce({
        success: false,
        localId: "new-1",
        syncStatus: "pending",
        failureKind: "validation",
        error: "invalid",
      })
      .mockResolvedValueOnce({
        success: true,
        localId: "new-2",
        syncStatus: "synced",
      });
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );
    expect(
      await screen.findByText(/Could not save the recipe/),
    ).toBeInTheDocument();
    expect(records.map((record) => record.id)).toEqual(["system-id"]);
    expect(mocks.permanentlyDeletePrompt).toHaveBeenCalledWith("new-1");
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["getAllPromptsForSelect"],
    });

    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );
    await waitFor(() =>
      expect(records.map((record) => record.id)).toEqual([
        "system-id",
        "new-2",
      ]),
    );
    expect(
      within(source).getAllByRole("option", { name: "Untitled recipe" }),
    ).toHaveLength(1);
  });

  it("restores the exact Update snapshot after a non-transient sync failure", async () => {
    const user = userEvent.setup();
    const original = recipeRecord("system-id", "System saved", "system");
    const records = [structuredClone(original)];
    mocks.getAllPrompts.mockImplementation(async () =>
      structuredClone(records),
    );
    mocks.updatePrompt.mockImplementation(async (fields) => {
      records[0] = { ...records[0], ...structuredClone(fields) };
      return fields.id;
    });
    mocks.restorePromptSnapshot.mockImplementation(async (snapshot) => {
      records[0] = structuredClone(snapshot);
    });
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "system-id",
      syncStatus: "local",
      failureKind: "validation",
      error: "invalid",
    });
    const { queryClient } = renderBuilder();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "System saved" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:system-id");
    const blockName = screen.getByRole("textbox", { name: "Block name" });
    await user.clear(blockName);
    await user.type(blockName, "Changed objective");
    await user.click(screen.getByRole("button", { name: "Update recipe" }));

    expect(
      await screen.findByText(/Could not update the recipe/),
    ).toBeInTheDocument();
    expect(records[0]).toEqual(original);
    expect(mocks.restorePromptSnapshot).toHaveBeenCalledWith(original);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["fetchAllPrompts"] });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["getAllPromptsForSelect"],
    });
  });

  it("surfaces rollback failure without losing the recipe working copy", async () => {
    const user = userEvent.setup();
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "new-exact-id",
      syncStatus: "local",
      failureKind: "validation",
      error: "invalid",
    });
    mocks.permanentlyDeletePrompt.mockRejectedValue(
      new Error("storage blocked"),
    );
    renderBuilder();
    const runtime = await screen.findByRole("textbox", {
      name: "Current value for Task (not saved)",
    });
    await user.type(runtime, "Keep this working value");
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(
      await screen.findByText(/local rollback also failed/i),
    ).toBeInTheDocument();
    expect(runtime).toHaveValue("Keep this working value");
  });

  it("keeps a conflicting saved recipe guarded before any local write", async () => {
    const user = userEvent.setup();
    mocks.getAllPrompts.mockResolvedValue([
      {
        ...recipeRecord("conflict-id", "Conflicting recipe", "system"),
        syncStatus: "conflict",
      },
    ]);
    renderBuilder();
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "Conflicting recipe" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:conflict-id");

    expect(screen.queryByRole("button", { name: "Update recipe" })).toBeNull();
    expect(mocks.updatePrompt).not.toHaveBeenCalled();
  });

  it("honors private-mode persistence authorization while keeping Apply local", async () => {
    state.privateMode = true;
    renderBuilder();

    expect(await screen.findByRole("status")).toHaveTextContent(
      "private browsing",
    );
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
  });

  it.each([
    [false, capabilities(true), "offline"],
    [true, undefined, "Checking"],
    [true, capabilities(false), "does not support"],
  ])(
    "disables persistence but not Apply when online=%s",
    async (online, caps, reason) => {
      state.online = online;
      renderBuilder({ capabilities: caps });

      expect(await screen.findByRole("status")).toHaveTextContent(reason);
      expect(
        screen.getByRole("button", { name: "Save as new recipe" }),
      ).toBeDisabled();
      await userEvent.setup().type(
        screen.getByRole("textbox", {
          name: "Current value for Task (not saved)",
        }),
        "valid task",
      );
      expect(
        screen.getByRole("button", { name: "Apply to system prompt" }),
      ).toBeEnabled();
    },
  );
});
