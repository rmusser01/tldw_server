import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  navigateFromSettingsExit,
  SettingsLayout,
} from "../SettingsOptionLayout";

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  pathname: "/settings/tldw",
}));

const settingsReturnMocks = vi.hoisted(() => ({
  returnTo: "/chat" as string | null,
}));

const IconStub = () => null;

vi.mock("react-router-dom", async () => {
  const React = await import("react");
  return {
    Link: ({
      children,
      to,
      ...props
    }: {
      children?: React.ReactNode;
      to: string;
    }) =>
      React.createElement(
        "a",
        {
          ...props,
          href: to,
        },
        children,
      ),
    useLocation: () => ({
      pathname: routerMocks.pathname,
      search: "",
      hash: "",
      state: null,
      key: "settings-test",
    }),
    useNavigate: () => routerMocks.navigate,
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (token: string, fallback?: string) => fallback ?? token,
  }),
}));

vi.mock("@/config/platform", () => ({
  isChromeTarget: false,
}));

vi.mock("../settings-nav", () => ({
  getSettingsNavGroups: () => [
    {
      key: "server",
      titleToken: "settings:navigation.serverAndAuth",
      items: [
        {
          to: "/settings/tldw",
          labelToken: "settings:tldw.serverNav",
          icon: IconStub,
        },
      ],
    },
  ],
}));

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: null,
    loading: false,
  }),
}));

vi.mock("@/utils/sidepanel", () => ({
  isSidepanelSupported: () => false,
  openSidepanel: vi.fn(),
}));

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>();
  return {
    ...actual,
    setSetting: vi.fn(),
  };
});

vi.mock("@/utils/settings-return", () => ({
  getSettingsReturnTo: () => settingsReturnMocks.returnTo,
}));

const renderSettingsLayout = () =>
  render(
    <SettingsLayout>
      <div>settings content</div>
    </SettingsLayout>,
  );

describe("settings layout exit navigation", () => {
  beforeEach(() => {
    routerMocks.navigate.mockReset();
    routerMocks.pathname = "/settings/tldw";
    settingsReturnMocks.returnTo = "/chat";
    delete (window as typeof window & { __NEXT_DATA__?: unknown })
      .__NEXT_DATA__;
  });

  it("uses flushSync when exiting to the saved return route", async () => {
    const user = userEvent.setup();
    renderSettingsLayout();

    await user.click(screen.getByRole("button", { name: "Close" }));

    expect(routerMocks.navigate).toHaveBeenCalledWith("/chat", {
      flushSync: true,
    });
  });

  it("uses document navigation in the Next web app", () => {
    const assignLocation = vi.fn();

    navigateFromSettingsExit(routerMocks.navigate, "/chat", {
      isNextWebApp: true,
      assignLocation,
    });

    expect(assignLocation).toHaveBeenCalledWith("/chat");
    expect(routerMocks.navigate).not.toHaveBeenCalled();
  });

  it("normalizes external settings exit targets to the app root", () => {
    const assignLocation = vi.fn();

    navigateFromSettingsExit(
      routerMocks.navigate,
      "https://example.test/chat",
      {
        isNextWebApp: true,
        assignLocation,
      },
    );

    expect(assignLocation).toHaveBeenCalledWith("/");
    expect(routerMocks.navigate).not.toHaveBeenCalled();
  });
});
