import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  navigateFromSettingsExit,
  SettingsLayout,
} from "../SettingsOptionLayout";
import { SETTINGS_NAVIGATION_REQUEST_EVENT } from "@/utils/settings-return";

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
        {
          to: "/settings/prompt",
          labelToken: "settings:servicePrompts.navigationLabel",
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

vi.mock("@/utils/settings-return", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/utils/settings-return")
  >();
  return {
    ...actual,
    getSettingsReturnTo: () => settingsReturnMocks.returnTo,
  };
});

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

  it("requests navigation before exiting to the saved return route", async () => {
    const user = userEvent.setup();
    const request = vi.fn();
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, request);
    renderSettingsLayout();

    try {
      await user.click(screen.getByRole("button", { name: "Close" }));

      expect(request).toHaveBeenCalledTimes(1);
      expect((request.mock.calls[0][0] as CustomEvent).detail).toEqual({
        destination: "/chat",
      });
      expect(routerMocks.navigate).toHaveBeenCalledWith("/chat");
    } finally {
      window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, request);
    }
  });

  it("cancels the extension Close when a mounted settings editor declines", async () => {
    const user = userEvent.setup();
    const preventNavigation = (event: Event) => event.preventDefault();
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      preventNavigation,
    );
    renderSettingsLayout();

    try {
      await user.click(screen.getByRole("button", { name: "Close" }));
      expect(routerMocks.navigate).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener(
        SETTINGS_NAVIGATION_REQUEST_EVENT,
        preventNavigation,
      );
    }
  });

  it("cancels or allows the actual mobile settings section selection", async () => {
    const user = userEvent.setup();
    const preventNavigation = (event: Event) => event.preventDefault();
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      preventNavigation,
    );
    renderSettingsLayout();
    const sectionSelect = screen.getByRole("combobox", {
      name: "Settings section",
    });

    try {
      await user.selectOptions(sectionSelect, "/settings/prompt");
      expect(routerMocks.navigate).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener(
        SETTINGS_NAVIGATION_REQUEST_EVENT,
        preventNavigation,
      );
    }

    await user.selectOptions(sectionSelect, "/settings/prompt");
    expect(routerMocks.navigate).toHaveBeenCalledWith("/settings/prompt");
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
