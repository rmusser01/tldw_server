import { bgRequest } from "@/services/background-proxy";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { setupOnboardingMethods } from "../domains/setup-onboarding";

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn(),
}));

describe("setup onboarding API domain", () => {
  beforeEach(() => {
    vi.mocked(bgRequest).mockReset();
  });

  it("fetches first-run state from setup endpoint", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ status: "not_started" });

    const result = await setupOnboardingMethods.getFirstRunState.call({});

    expect(result.status).toBe("not_started");
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/state",
      method: "GET",
      noAuth: true,
    });
  });

  it("fetches setup metadata for auth and setup path decisions", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://127.0.0.1:3000",
        api_origin: "http://127.0.0.1:8000",
        browser_access: "local",
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" },
    });

    const result = await setupOnboardingMethods.getFirstRunMetadata.call({});

    expect(result.manual_auth_required).toBe(false);
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/metadata",
      method: "GET",
      noAuth: true,
    });
  });

  it("saves provider setup without leaking raw secret into return shape", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      provider_key: "openai",
      status: "saved",
      masked_api_key: "sk-...abcd",
      api_key: "sk-secret",
      provider: {
        provider_key: "openai",
        api_key: "sk-secret",
      },
      providers: [
        {
          provider_key: "openai",
          api_key: "sk-secret",
        },
      ],
    });

    const result = await setupOnboardingMethods.saveSetupProvider.call(
      {},
      {
        provider_key: "openai",
        api_key: "sk-secret",
        make_default: true,
      },
    );

    expect(result.masked_api_key).toBe("sk-...abcd");
    expect(result).not.toHaveProperty("api_key");
    expect(JSON.stringify(result)).not.toContain("sk-secret");
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/providers",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      noAuth: true,
      body: {
        provider_key: "openai",
        api_key: "sk-secret",
        make_default: true,
      },
    });
  });

  it("posts setup completion through the unauthenticated recovery surface", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ status: "completed" });

    await setupOnboardingMethods.completeFirstRun.call(
      {},
      {
        acknowledged_steps: ["first_chat"],
      },
    );

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/complete",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      noAuth: true,
      body: {
        acknowledged_steps: ["first_chat"],
      },
    });
  });

  it("fetches setup audio recommendations without requiring configured auth", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      machine_profile: {},
      catalog: [],
      recommendations: [],
      excluded: [],
    });

    await setupOnboardingMethods.getSetupAudioRecommendations.call({});

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/audio/recommendations",
      method: "GET",
      noAuth: true,
    });
  });

  it("fetches the first-run MCP tools catalog without auth", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      catalog_version: "2026-07-04",
      confirmation_version: "v1",
      packs: [],
      add_ons: [],
      validation_states: ["not_run"],
    });

    const result = await setupOnboardingMethods.getMcpToolsCatalog.call({});

    expect(result.catalog_version).toBe("2026-07-04");
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/mcp-tools/catalog",
      method: "GET",
      noAuth: true,
    });
  });

  it("applies first-run MCP tools with conflict responses expected", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      status: "applied",
      profile_id: 7,
      assignment_id: 9,
      catalog_version: "2026-07-04",
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      effective_tool_count: 3,
      effective_tools: ["mcp.tools.list"],
      disabled_addons: [],
      validation_state: "not_run",
      conflict: null,
    });

    const payload = {
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      confirmed_addon_ids: [],
      confirmation_version: "v1",
    };
    const result = await setupOnboardingMethods.applyMcpTools.call({}, payload);

    expect(result.profile_id).toBe(7);
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/mcp-tools/apply",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      noAuth: true,
      expectedStatuses: [409],
      body: payload,
    });
  });

  it("returns typed first-run MCP tools conflict bodies", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      status: "conflict",
      profile_id: 7,
      assignment_id: null,
      catalog_version: "2026-07-04",
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      effective_tool_count: 0,
      effective_tools: [],
      disabled_addons: [],
      validation_state: "not_run",
      conflict: {
        reason: "profile_manually_changed",
        profile_id: 7,
        current_hash: "changed",
        expected_hash: "expected",
      },
    });

    const result = await setupOnboardingMethods.applyMcpTools.call(
      {},
      { selected_pack_ids: ["research"] },
    );

    expect(result.status).toBe("conflict");
    expect(result.conflict?.reason).toBe("profile_manually_changed");
  });

  it("validates first-run MCP tools with an empty default payload", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      status: "validated",
      validation_state: "built_in_passed",
      profile_id: 7,
      assignment_id: 9,
      catalog_version: "2026-07-04",
      selected_pack_ids: ["research"],
      selected_addon_ids: [],
      effective_tool_count: 3,
    });

    await setupOnboardingMethods.validateMcpTools.call({});

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/mcp-tools/validate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      noAuth: true,
      body: {},
    });
  });
});
