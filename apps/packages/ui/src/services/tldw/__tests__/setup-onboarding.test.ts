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
});
