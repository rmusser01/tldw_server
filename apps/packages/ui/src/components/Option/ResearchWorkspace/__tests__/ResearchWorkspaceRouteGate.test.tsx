import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResearchWorkspaceRouteGate } from "../ResearchWorkspaceRouteGate";
import { parseSharedWorkspaceRoute } from "../shared-workspace-route-state";

const localWorkspaceFactory = vi.hoisted(() => vi.fn());
const localWorkspaceApiRequests = vi.hoisted(() => vi.fn());
const locationState = vi.hoisted(() => ({ search: "" }));
const sharedControllerState = vi.hoisted(() => ({
  status: "loading",
  bootstrap: null as null | {
    share: { share_id: number };
    workspace: { name: string };
  },
  errors: { bootstrap: null as null | { code?: string } },
}));

vi.mock("react-router-dom", () => ({
  useLocation: () => locationState,
}));

vi.mock("../index", () => {
  localWorkspaceFactory();
  return {
    ResearchWorkspace: () => {
      localWorkspaceApiRequests();
      return <div data-testid="local-research-workspace" />;
    },
  };
});

vi.mock("../SharedResearchWorkspace/useSharedResearchWorkspace", () => ({
  useSharedResearchWorkspace: () => ({ state: sharedControllerState }),
}));

const renderGate = (search: string) => {
  locationState.search = search;
  return render(<ResearchWorkspaceRouteGate />);
};

describe("parseSharedWorkspaceRoute", () => {
  it("uses the local workspace when no shared parameter is present", () => {
    expect(parseSharedWorkspaceRoute("?tab=chat")).toEqual({ kind: "local" });
  });

  it("accepts exactly one positive base-10 safe integer", () => {
    expect(parseSharedWorkspaceRoute("?shared=42")).toEqual({
      kind: "shared-valid",
      shareId: 42,
    });
  });

  it.each([
    "?shared=",
    "?shared=0",
    "?shared=-1",
    "?shared=1.5",
    "?shared=01",
    "?shared=1&shared=2",
    `?shared=${Number.MAX_SAFE_INTEGER + 1}`,
  ])("fails closed for %s", (search) => {
    expect(parseSharedWorkspaceRoute(search)).toEqual({
      kind: "shared-invalid",
    });
  });
});

describe("ResearchWorkspaceRouteGate", () => {
  afterEach(() => {
    locationState.search = "";
    sharedControllerState.status = "loading";
    sharedControllerState.bootstrap = null;
    sharedControllerState.errors.bootstrap = null;
    vi.clearAllMocks();
  });

  it("mounts the local workspace only in local mode", async () => {
    renderGate("");

    expect(await screen.findByTestId("local-research-workspace")).toBeVisible();
    expect(localWorkspaceFactory).toHaveBeenCalledTimes(1);
    expect(localWorkspaceApiRequests).toHaveBeenCalledTimes(1);
  });

  it("does not import or mount the local workspace for a valid shared route", async () => {
    renderGate("?shared=42");

    expect(
      await screen.findByRole("heading", {
        name: /loading shared workspace/i,
      }),
    ).toBeVisible();
    expect(localWorkspaceFactory).not.toHaveBeenCalled();
    expect(localWorkspaceApiRequests).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("local-research-workspace"),
    ).not.toBeInTheDocument();
  });

  it("replaces a valid shared route without mounting the local workspace", async () => {
    const view = renderGate("?shared=42");

    expect(await screen.findByText("Share 42")).toBeVisible();

    locationState.search = "?shared=43";
    view.rerender(<ResearchWorkspaceRouteGate />);

    expect(await screen.findByText("Share 43")).toBeVisible();
    expect(localWorkspaceFactory).not.toHaveBeenCalled();
  });

  it("returns to local mode when the shared parameter is removed", async () => {
    const view = renderGate("?shared=42");

    await screen.findByRole("heading", {
      name: /loading shared workspace/i,
    });
    locationState.search = "";
    view.rerender(<ResearchWorkspaceRouteGate />);

    expect(await screen.findByTestId("local-research-workspace")).toBeVisible();
    expect(localWorkspaceApiRequests).toHaveBeenCalledTimes(1);
  });

  it("focuses the invalid shared-route state without mounting local workspace", async () => {
    renderGate("?shared=01");

    const heading = await screen.findByRole("heading", {
      name: /shared workspace unavailable/i,
    });
    expect(heading).toHaveFocus();
    expect(localWorkspaceFactory).not.toHaveBeenCalled();
  });

  it.each([
    ["not-found", "Shared workspace not found"],
    ["unavailable", "Shared workspace unavailable"],
  ])("renders the stable %s placeholder", async (status, heading) => {
    sharedControllerState.status = status;
    renderGate("?shared=42");

    expect(await screen.findByRole("heading", { name: heading })).toBeVisible();
    expect(localWorkspaceFactory).not.toHaveBeenCalled();
  });

  it("renders the loaded shared identity without mounting local workspace", async () => {
    sharedControllerState.status = "loaded";
    sharedControllerState.bootstrap = {
      share: { share_id: 42 },
      workspace: { name: "Shared climate research" },
    };
    renderGate("?shared=42");

    expect(
      await screen.findByRole("heading", { name: "Shared climate research" }),
    ).toBeVisible();
    expect(localWorkspaceFactory).not.toHaveBeenCalled();
  });
});
