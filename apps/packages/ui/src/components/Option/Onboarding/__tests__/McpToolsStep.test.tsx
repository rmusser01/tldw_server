// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  McpToolsApplyResponse,
  McpToolsCatalogResponse,
  McpToolsValidateResponse,
} from "@/types/setup-onboarding";
import { McpToolsStep, type McpToolsStepProps } from "../steps/McpToolsStep";

const catalog: McpToolsCatalogResponse = {
  catalog_version: "2026-07-04",
  confirmation_version: "v1",
  packs: [
    {
      pack_id: "research",
      label: "Research",
      purpose: "Search and summarize saved sources.",
      default_selected: true,
      available: true,
      module_targets: ["mcp"],
      tool_patterns: ["mcp.tools.list"],
      available_tools: [
        { tool_name: "mcp.tools.list", available: true },
        { tool_name: "media.search", available: true },
      ],
      unavailable_tools: [],
      add_on_ids: ["external_network_read"],
      sample_validation_candidates: ["mcp.tools.list"],
      catalog_version: "2026-07-04",
    },
    {
      pack_id: "writing",
      label: "Writing",
      purpose: "Draft and edit notes.",
      default_selected: false,
      available: true,
      module_targets: ["mcp"],
      tool_patterns: ["notes.*"],
      available_tools: [{ tool_name: "notes.create", available: true }],
      unavailable_tools: [],
      add_on_ids: ["workspace_write"],
      sample_validation_candidates: ["notes.create"],
      catalog_version: "2026-07-04",
    },
  ],
  add_ons: [
    {
      addon_id: "external_network_read",
      label: "External network read",
      default_selected: false,
      requirement: "Allow read-only network lookups.",
      strong_confirmation: false,
    },
    {
      addon_id: "local_file_read",
      label: "Local file read",
      default_selected: false,
      requirement: "Allow tools to read local files.",
      strong_confirmation: false,
    },
    {
      addon_id: "workspace_write",
      label: "Workspace write",
      default_selected: false,
      requirement: "Allow tools to write workspace files.",
      strong_confirmation: true,
    },
  ],
  validation_states: ["not_run", "built_in_passed"],
};

const applied: McpToolsApplyResponse = {
  status: "applied",
  profile_id: 7,
  assignment_id: 9,
  catalog_version: "2026-07-04",
  selected_pack_ids: ["research"],
  selected_addon_ids: [],
  effective_tool_count: 2,
  effective_tools: ["mcp.tools.list", "media.search"],
  disabled_addons: ["local_file_read"],
  validation_state: "not_run",
  conflict: null,
};

const validated: McpToolsValidateResponse = {
  status: "validated",
  validation_state: "built_in_passed",
  profile_id: 7,
  assignment_id: 9,
  catalog_version: "2026-07-04",
  selected_pack_ids: ["research"],
  selected_addon_ids: [],
  effective_tool_count: 2,
  validated_at: "2026-07-05T00:00:00Z",
  validation_message: "Sample tool passed.",
  sample_tool_name: "mcp.tools.list",
  external_status: "not_configured",
};

const createProps = (
  overrides: Partial<McpToolsStepProps> = {},
): McpToolsStepProps => ({
  catalog,
  loadCatalog: vi.fn().mockResolvedValue(catalog),
  applyMcpTools: vi.fn().mockResolvedValue(applied),
  validateMcpTools: vi.fn().mockResolvedValue(validated),
  onContinue: vi.fn(),
  onBack: vi.fn(),
  onSkip: vi.fn(),
  ...overrides,
});

const renderMcpToolsStep = (props: McpToolsStepProps) =>
  render(
    <MemoryRouter>
      <McpToolsStep {...props} />
    </MemoryRouter>,
  );

describe("McpToolsStep", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads missing catalog and renders default selected packs checked", async () => {
    const loadCatalog = vi.fn().mockResolvedValue(catalog);
    renderMcpToolsStep(createProps({ catalog: null, loadCatalog }));

    expect(loadCatalog).toHaveBeenCalledTimes(1);
    expect(await screen.findByLabelText(/research/i)).toBeChecked();
    expect(screen.getByLabelText(/writing/i)).not.toBeChecked();
  });

  it("renders risky add-ons collapsed and off by default", () => {
    renderMcpToolsStep(createProps());

    expect(screen.getByTestId("mcp-tools-addons")).not.toHaveAttribute("open");

    fireEvent.click(screen.getByText(/add-ons/i));

    expect(screen.getByLabelText(/external network read/i)).not.toBeChecked();
    expect(screen.getByLabelText(/local file read/i)).not.toBeChecked();
    expect(screen.getByLabelText(/workspace write/i)).not.toBeChecked();
  });

  it("requires inline confirmation before a strong add-on can be saved", () => {
    renderMcpToolsStep(createProps());

    fireEvent.click(screen.getByText(/add-ons/i));
    fireEvent.click(screen.getByLabelText(/workspace write/i));

    expect(screen.getByLabelText(/confirm workspace write/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /save packs/i })).toBeDisabled();
  });

  it("saves selected packs, add-ons, confirmed strong add-ons, and confirmation version", async () => {
    const applyMcpTools = vi.fn().mockResolvedValue({
      ...applied,
      selected_pack_ids: ["research", "writing"],
      selected_addon_ids: ["external_network_read", "workspace_write"],
      effective_tool_count: 3,
      effective_tools: ["mcp.tools.list", "media.search", "notes.create"],
    });
    renderMcpToolsStep(createProps({ applyMcpTools }));

    fireEvent.click(screen.getByLabelText(/writing/i));
    fireEvent.click(screen.getByText(/add-ons/i));
    fireEvent.click(screen.getByLabelText(/external network read/i));
    fireEvent.click(screen.getByLabelText(/workspace write/i));
    fireEvent.click(screen.getByLabelText(/confirm workspace write/i));
    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));

    await waitFor(() => {
      expect(applyMcpTools).toHaveBeenCalledWith({
        selected_pack_ids: ["research", "writing"],
        selected_addon_ids: ["external_network_read", "workspace_write"],
        confirmed_addon_ids: ["workspace_write"],
        confirmation_version: "v1",
      });
    });
  });

  it("restores saved selections and saved state from initial step data", async () => {
    const applyMcpTools = vi.fn().mockResolvedValue({
      ...applied,
      selected_pack_ids: ["writing"],
      selected_addon_ids: ["workspace_write"],
      effective_tool_count: 1,
      effective_tools: ["notes.create"],
    });
    renderMcpToolsStep(
      createProps({
        applyMcpTools,
        initialStepData: {
          acknowledged: true,
          validation_state: "not_run",
          profile_id: 11,
          assignment_id: 12,
          selected_pack_ids: ["writing"],
          selected_addon_ids: ["workspace_write"],
          confirmed_addon_ids: ["workspace_write"],
          effective_tool_count: 1,
          effective_tools: ["notes.create"],
          disabled_addons: [],
        },
      }),
    );

    expect(screen.getByLabelText(/research/i)).not.toBeChecked();
    expect(screen.getByLabelText(/writing/i)).toBeChecked();
    expect(screen.getByRole("button", { name: /run sample tool/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.click(screen.getByText(/add-ons/i));
    expect(
      screen.getByRole("checkbox", { name: /^workspace write/i }),
    ).toBeChecked();
    expect(screen.getByLabelText(/confirm workspace write/i)).toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));

    await waitFor(() => {
      expect(applyMcpTools).toHaveBeenCalledWith({
        selected_pack_ids: ["writing"],
        selected_addon_ids: ["workspace_write"],
        confirmed_addon_ids: ["workspace_write"],
        confirmation_version: "v1",
      });
    });
  });

  it("shows conflict actions and resolves by keeping or replacing the existing profile", async () => {
    const conflict = {
      ...applied,
      status: "conflict",
      profile_id: 7,
      effective_tool_count: 0,
      effective_tools: [],
      conflict: {
        reason: "profile_manually_changed",
        profile_id: 7,
        current_hash: "changed",
        expected_hash: "expected",
      },
    };
    const applyMcpTools = vi
      .fn()
      .mockResolvedValueOnce(conflict)
      .mockResolvedValue(applied);
    renderMcpToolsStep(createProps({ applyMcpTools }));

    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));
    expect(
      await screen.findByText(/generated profile was changed/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/profile_manually_changed/i)).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open mcp hub/i })).toHaveAttribute(
      "href",
      "/mcp-hub?source=first-run&profile_id=7",
    );

    fireEvent.click(screen.getByRole("button", { name: /keep existing/i }));
    await waitFor(() => {
      expect(applyMcpTools).toHaveBeenLastCalledWith(
        expect.objectContaining({
          conflict_resolution: "keep_existing",
          profile_id: 7,
        }),
      );
    });

    applyMcpTools.mockResolvedValueOnce(conflict);
    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));
    await screen.findByText(/generated profile was changed/i);
    fireEvent.click(
      screen.getByRole("button", { name: /replace generated profile/i }),
    );
    await waitFor(() => {
      expect(applyMcpTools).toHaveBeenLastCalledWith(
        expect.objectContaining({
          conflict_resolution: "replace_existing",
          profile_id: 7,
        }),
      );
    });
  });

  it("runs sample tool only after save", async () => {
    const validateMcpTools = vi.fn().mockResolvedValue(validated);
    renderMcpToolsStep(createProps({ validateMcpTools }));

    expect(screen.getByRole("button", { name: /run sample tool/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /run sample tool/i })).toBeEnabled();
    });
    fireEvent.click(screen.getByRole("button", { name: /run sample tool/i }));

    await waitFor(() => {
      expect(validateMcpTools).toHaveBeenCalledWith({});
    });
    expect(await screen.findByText(/sample tool passed/i)).toBeInTheDocument();
  });

  it("allows continuing after save even when validation has not run", async () => {
    const onContinue = vi.fn();
    renderMcpToolsStep(createProps({ onContinue }));

    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    });
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    expect(onContinue).toHaveBeenCalledTimes(1);
  });

  it("requires saving again after a saved selection changes", async () => {
    renderMcpToolsStep(createProps());

    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /run sample tool/i })).toBeEnabled();
      expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    });

    fireEvent.click(screen.getByLabelText(/writing/i));

    expect(screen.getByRole("button", { name: /run sample tool/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("calls the provided skip handler", () => {
    const onSkip = vi.fn();
    renderMcpToolsStep(createProps({ onSkip }));

    fireEvent.click(screen.getByRole("button", { name: /skip mcp tools/i }));

    expect(onSkip).toHaveBeenCalledTimes(1);
  });

  it("summarizes saved packs, tools, disabled add-ons, external status, and hub link", async () => {
    renderMcpToolsStep(createProps());

    fireEvent.click(screen.getByRole("button", { name: /save packs/i }));

    expect(await screen.findByText(/enabled packs/i)).toBeInTheDocument();
    expect(screen.getAllByText(/^Research$/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/2 tools/i)).toBeInTheDocument();
    expect(screen.getByText(/mcp.tools.list/i)).toBeInTheDocument();
    expect(screen.getByText(/media.search/i)).toBeInTheDocument();
    expect(screen.getAllByText(/^Local file read$/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Not run/i)).toBeInTheDocument();
    expect(screen.queryByText(/local_file_read/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/not_run/i)).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open mcp hub/i })).toHaveAttribute(
      "href",
      "/mcp-hub?source=first-run&profile_id=7",
    );

    fireEvent.click(screen.getByRole("button", { name: /run sample tool/i }));
    expect(await screen.findByText(/Not configured/i)).toBeInTheDocument();
    expect(screen.queryByText(/not_configured/i)).not.toBeInTheDocument();
  });
});
