import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { SavedRecipeSource } from "../types";
import { SingleFieldRecipeEditor } from "../SingleFieldRecipeEditor";

const savedRecipe = (
  overrides: Partial<SavedRecipeSource> = {},
): SavedRecipeSource => ({
  source_kind: "saved",
  id: "saved-greeting",
  name: "Saved greeting",
  definition: {
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "system",
      render_format: "xml",
      block_separator: "\n\n",
    },
    variables: [
      {
        name: "audience",
        label: "Audience",
        description: "Who will read this.",
        required: true,
        default_value: "Maintainers",
        input_type: "text",
        options: null,
        max_length: null,
      },
    ],
    blocks: [
      {
        id: "greeting",
        name: "Greeting",
        section_key: "greeting",
        role: "system",
        kind: "context",
        content: "Hello {{audience}}",
        enabled: true,
        order: 10,
        is_template: true,
      },
      {
        id: "rules",
        name: "Rules",
        section_key: "rules",
        role: "system",
        kind: "constraints",
        content: "Be concise.",
        enabled: true,
        order: 20,
        is_template: false,
      },
    ],
  },
  ...overrides,
});

const renderEditor = (
  props: Partial<React.ComponentProps<typeof SingleFieldRecipeEditor>> = {},
) => {
  const callbacks = {
    onApply: vi.fn(),
    onSaveAsNew: vi.fn(),
    onUpdate: vi.fn(),
  };
  render(<SingleFieldRecipeEditor target="system" {...callbacks} {...props} />);
  return callbacks;
};

describe("SingleFieldRecipeEditor", () => {
  it("offers immutable starters and locks every block to the selected target", () => {
    renderEditor({ target: "user_message" });

    const picker = screen.getByRole("combobox", { name: "Recipe source" });
    expect(
      within(picker).getByRole("option", { name: "Clear task" }),
    ).toBeTruthy();
    expect(
      within(picker).getByRole("option", { name: "Research and analysis" }),
    ).toBeTruthy();
    expect(
      within(picker).getByRole("option", { name: "Agent workflow" }),
    ).toBeTruthy();
    expect(within(picker).getByRole("option", { name: "Blank" })).toBeTruthy();
    expect(screen.getByText("User message draft")).toBeTruthy();
    expect(screen.queryByRole("combobox", { name: "Block role" })).toBeNull();
  });

  it("opens a saved recipe as an isolated copy and exposes Update only for that source", async () => {
    const source = savedRecipe();
    const callbacks = renderEditor({ initialSource: source });
    const user = userEvent.setup();

    expect(screen.getByRole("button", { name: "Update recipe" })).toBeEnabled();
    await user.clear(screen.getByRole("textbox", { name: "Block content" }));
    await user.type(
      screen.getByRole("textbox", { name: "Block content" }),
      "Changed",
    );
    await user.click(screen.getByRole("button", { name: "Update recipe" }));

    expect(callbacks.onUpdate).toHaveBeenCalledTimes(1);
    expect(callbacks.onUpdate.mock.calls[0][0].blocks[0].content).toBe(
      "Changed",
    );
    expect(
      (source.definition as { blocks: Array<{ content: string }> }).blocks[0]
        .content,
    ).toBe("Hello {{audience}}");
  });

  it("clears runtime inputs when switching recipe sources", async () => {
    renderEditor();
    const user = userEvent.setup();

    const currentValue = screen.getByRole("textbox", {
      name: "Current value for Task (not saved)",
    });
    await user.type(currentValue, "Draft value");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recipe source" }),
      "built_in:blank",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recipe source" }),
      "built_in:clear_task",
    );

    expect(
      screen.getByRole("textbox", {
        name: "Current value for Task (not saved)",
      }),
    ).toHaveValue("");
    expect(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    ).toBeDisabled();
  });

  it("adds, edits, validates, toggles, and removes a target-locked block with useful focus", async () => {
    renderEditor({
      initialSource: {
        source_kind: "built_in",
        id: "blank",
        name: "Blank",
        definition: {
          schema_version: 2,
          format: "structured",
          definition_kind: "single_text_recipe",
          assembly_config: {
            assembly_mode: "single_text",
            target_role: "system",
            render_format: "xml",
            block_separator: "\n\n",
          },
          variables: [],
          blocks: [],
        },
      },
    });
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Add block" }));
    const name = screen.getByRole("textbox", { name: "Block name" });
    expect(name).toHaveFocus();
    await user.clear(name);
    await user.type(name, "Rules");
    await user.clear(screen.getByRole("textbox", { name: "Section key" }));
    await user.type(
      screen.getByRole("textbox", { name: "Section key" }),
      "bad key",
    );

    expect(screen.getByRole("alert")).toHaveTextContent("valid section key");
    expect(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    ).toBeDisabled();

    await user.clear(screen.getByRole("textbox", { name: "Section key" }));
    await user.type(
      screen.getByRole("textbox", { name: "Section key" }),
      "rules",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Block content" }),
      "Follow them.",
    );
    await user.click(screen.getByRole("checkbox", { name: "Block enabled" }));
    expect(screen.getByText("disabled", { exact: false })).toBeTruthy();
    await user.click(screen.getByRole("checkbox", { name: "Block enabled" }));
    await user.click(screen.getByRole("button", { name: "Remove Rules" }));

    expect(screen.getByRole("button", { name: "Add block" })).toHaveFocus();
  });

  it("supports pointer drag ordering and keyboard-operable move controls", async () => {
    renderEditor();
    const user = userEvent.setup();
    const list = screen.getByTestId("structured-block-list");

    const outputUp = within(list).getByRole("button", {
      name: "Move Output up",
    });
    outputUp.focus();
    await user.keyboard("{Enter}");
    expect(
      within(list)
        .getAllByTestId(/structured-block-item-/)
        .map((item) => item.textContent),
    ).toEqual([
      expect.stringContaining("Objective"),
      expect.stringContaining("Context / inputs"),
      expect.stringContaining("Output"),
      expect.stringContaining("Constraints"),
    ]);

    fireEvent.dragStart(
      screen.getByTestId("structured-block-item-constraints"),
    );
    fireEvent.drop(screen.getByTestId("structured-block-item-objective"));
    expect(
      within(list).getAllByTestId(/structured-block-item-/)[0],
    ).toHaveTextContent("Constraints");
  });

  it("renders an exact plain-text preview for XML, Markdown, and free-form formats", async () => {
    renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();
    const preview = screen.getByRole("textbox", {
      name: "Compiled prompt preview",
    });

    expect(preview).toHaveValue(
      "<greeting>Hello Maintainers</greeting>\n\n<rules>Be concise.</rules>",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Output format" }),
      "markdown",
    );
    expect(preview).toHaveValue(
      "## Greeting\n\nHello Maintainers\n\n## Rules\n\nBe concise.",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Output format" }),
      "freeform",
    );
    expect(preview).toHaveValue("Hello Maintainers\n\nBe concise.");
    expect(preview).toHaveAttribute("readonly");
  });

  it("keeps starter defaults separate from runtime inputs and omits runtime values from save", async () => {
    const callbacks = renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();
    const runtime = screen.getByRole("textbox", {
      name: "Current value for Audience (not saved)",
    });
    const savedDefault = screen.getByRole("textbox", {
      name: "Starter default for Audience (saved)",
    });

    expect(savedDefault).toHaveValue("Maintainers");
    await user.clear(savedDefault);
    await user.type(savedDefault, "Contributors");
    await user.clear(runtime);
    await user.type(runtime, "Readers");
    expect(
      screen.getByRole("textbox", { name: "Compiled prompt preview" }),
    ).toHaveValue(
      "<greeting>Hello Readers</greeting>\n\n<rules>Be concise.</rules>",
    );
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    const definition = callbacks.onSaveAsNew.mock.calls[0][0];
    expect(definition.variables[0].default_value).toBe("Contributors");
    expect(JSON.stringify(definition)).not.toContain("Readers");
    expect(definition).not.toHaveProperty("runtimeValues");
    expect(definition).not.toHaveProperty("runtime_values");
  });

  it("edits every persisted variable field through the declaration editor", async () => {
    const callbacks = renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();

    await user.clear(screen.getByRole("textbox", { name: "Variable label" }));
    await user.type(
      screen.getByRole("textbox", { name: "Variable label" }),
      "Reader",
    );
    await user.clear(
      screen.getByRole("textbox", { name: "Variable description" }),
    );
    await user.type(
      screen.getByRole("textbox", { name: "Variable description" }),
      "Intended reader",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Variable input type" }),
      "select",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Variable options" }),
      "\nContributors",
    );
    await user.type(
      screen.getByRole("spinbutton", { name: "Variable maximum length" }),
      "40",
    );
    await user.click(
      screen.getByRole("checkbox", { name: "Variable required" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(callbacks.onSaveAsNew.mock.calls[0][0].variables[0]).toEqual({
      name: "audience",
      label: "Reader",
      description: "Intended reader",
      required: false,
      default_value: "Maintainers",
      input_type: "select",
      options: ["Contributors"],
      max_length: 40,
    });
  });

  it("commits a valid variable rename through variable_updated and migrates its runtime input", async () => {
    const callbacks = renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();
    const variableName = screen.getByRole("textbox", { name: "Variable name" });

    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
      "Readers",
    );
    await user.clear(variableName);
    await user.type(variableName, "reader");
    await user.tab();
    expect(variableName).toHaveValue("reader");

    await user.clear(screen.getByRole("textbox", { name: "Block content" }));
    await user.type(
      screen.getByRole("textbox", { name: "Block content" }),
      "Hello {{reader}}",
    );
    expect(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
    ).toHaveValue("Readers");
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(callbacks.onSaveAsNew.mock.calls[0][0].variables[0].name).toBe(
      "reader",
    );
    expect(
      JSON.stringify(callbacks.onSaveAsNew.mock.calls[0][0]),
    ).not.toContain("Readers");
  });

  it("adds and removes declarations with an explicitly authored starter default", async () => {
    const callbacks = renderEditor();
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recipe source" }),
      "built_in:blank",
    );

    await user.click(screen.getByRole("button", { name: "Add variable" }));
    expect(screen.getByRole("textbox", { name: "Variable name" })).toHaveValue(
      "variable_1",
    );
    await user.click(
      screen.getByRole("checkbox", {
        name: "Use a saved starter default for Variable 1",
      }),
    );
    await user.type(
      screen.getByRole("textbox", {
        name: "Starter default for Variable 1 (saved)",
      }),
      "Saved value",
    );
    await user.click(
      screen.getByRole("button", { name: "Save as new recipe" }),
    );

    expect(
      callbacks.onSaveAsNew.mock.calls[0][0].variables[0].default_value,
    ).toBe("Saved value");
    await user.click(screen.getByRole("button", { name: "Remove Variable 1" }));
    expect(screen.queryByRole("textbox", { name: "Variable name" })).toBeNull();
  });

  it("reports invalid variable-name drafts inline and blocks actions until corrected", async () => {
    renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();
    const variableName = screen.getByRole("textbox", { name: "Variable name" });

    await user.clear(variableName);
    await user.type(variableName, "bad name");
    await user.tab();

    expect(screen.getByRole("alert")).toHaveTextContent("valid variable name");
    expect(variableName).toHaveAttribute("aria-invalid", "true");
    expect(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Update recipe" }),
    ).toBeDisabled();
  });

  it("allows Apply without Save only after required runtime values render", async () => {
    const source = savedRecipe();
    (
      source.definition as { variables: Array<{ default_value: null }> }
    ).variables[0].default_value = null;
    const callbacks = renderEditor({ initialSource: source });
    const user = userEvent.setup();
    const apply = screen.getByRole("button", {
      name: "Apply to system prompt",
    });

    expect(apply).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeEnabled();
    expect(screen.getByRole("alert")).toHaveTextContent("Audience");
    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
      "Operators",
    );
    await user.click(apply);

    expect(callbacks.onApply).toHaveBeenCalledWith(
      "<greeting>Hello Operators</greeting>\n\n<rules>Be concise.</rules>",
    );
    expect(callbacks.onSaveAsNew).not.toHaveBeenCalled();
    expect(callbacks.onUpdate).not.toHaveBeenCalled();
  });

  it("shows render errors inline and prevents applying an unsafe compiled XML section", async () => {
    const callbacks = renderEditor({ initialSource: savedRecipe() });
    const user = userEvent.setup();

    await user.clear(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
    );
    await user.type(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
      "{/greeting}",
    );
    fireEvent.change(
      screen.getByRole("textbox", {
        name: "Current value for Audience (not saved)",
      }),
      { target: { value: "</greeting>" } },
    );

    expect(screen.getByRole("alert")).toHaveTextContent("closing tag");
    expect(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    ).toBeDisabled();
    expect(callbacks.onApply).not.toHaveBeenCalled();
  });

  it("disables persistence while offline but keeps local preview and Apply available", async () => {
    const callbacks = renderEditor({
      initialSource: savedRecipe(),
      persistenceAvailable: false,
      persistenceUnavailableReason: "Reconnect to save recipes.",
    });
    const user = userEvent.setup();

    expect(screen.getByText("Reconnect to save recipes.")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Save as new recipe" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Update recipe" }),
    ).toBeDisabled();
    await user.click(
      screen.getByRole("button", { name: "Apply to system prompt" }),
    );
    expect(callbacks.onApply).toHaveBeenCalledTimes(1);
  });

  it("uses a responsive, overflow-safe shared layout", () => {
    renderEditor();
    const editor = screen.getByTestId("single-field-recipe-editor");

    expect(editor.className).toContain("min-w-0");
    expect(editor.className).toContain("max-w-full");
    expect(
      screen.getByTestId("single-field-recipe-workspace").className,
    ).toContain("minmax(0,1fr)");
  });
});
