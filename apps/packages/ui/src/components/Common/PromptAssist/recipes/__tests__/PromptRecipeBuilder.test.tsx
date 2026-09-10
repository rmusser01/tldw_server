import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PromptRecipeBuilder } from "../PromptRecipeBuilder";
import { CLEAR_TASK_RECIPE } from "../built-in-recipes";

const state = vi.hoisted(() => ({ online: true, privateMode: false }));
const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(),
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

const capabilities = (recipeSupported: boolean) => ({
  availability: "available" as const,
  prompt_improvement_v1: { supported: true, limits: null },
  single_text_recipe_v2: { supported: recipeSupported },
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
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(false);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: true,
      localId: "new-exact-id",
      syncStatus: "synced",
    });
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
  });

  it("surfaces non-transient sync failure and preserves conflict Update semantics", async () => {
    const user = userEvent.setup();
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true);
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      localId: "new-exact-id",
      syncStatus: "pending",
      failureKind: "validation",
      error: "invalid",
    });
    mocks.getAllPrompts.mockResolvedValue([
      {
        ...recipeRecord("conflict-id", "Conflict recipe", "system"),
        syncStatus: "conflict",
      },
    ]);
    renderBuilder();
    const source = await screen.findByRole("combobox", {
      name: "Recipe source",
    });
    await waitFor(() =>
      expect(
        within(source).getByRole("option", { name: "Conflict recipe" }),
      ).toBeTruthy(),
    );
    await user.selectOptions(source, "saved:conflict-id");
    expect(screen.queryByRole("button", { name: "Update recipe" })).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );
    expect(
      await screen.findByText(/Could not save the recipe/),
    ).toBeInTheDocument();
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
