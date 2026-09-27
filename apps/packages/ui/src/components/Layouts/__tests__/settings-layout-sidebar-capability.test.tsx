import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { browser as webBrowser } from "../../../../../../tldw-frontend/extension/shims/wxt-browser";
import { setSetting } from "@/services/settings/registry";
import { isSidepanelSupported } from "@/utils/sidepanel";
import { SettingsLayout } from "../SettingsOptionLayout";

const browserEnvironment = vi.hoisted(() => ({
  current: {} as Record<string, unknown>,
}));

vi.mock("wxt/browser", () => ({
  get browser() {
    return browserEnvironment.current;
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (token: string, fallback?: string) => fallback ?? token,
  }),
}));

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: null,
    loading: false,
  }),
}));

// Observe Settings' persistence requests without writing extension storage.
vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>();
  return {
    ...actual,
    setSetting: vi.fn().mockResolvedValue(undefined),
  };
});

const renderSettingsLayout = () =>
  render(
    <MemoryRouter initialEntries={["/settings/tldw"]}>
      <SettingsLayout>
        <div>settings content</div>
      </SettingsLayout>
    </MemoryRouter>,
  );

const expectSidebarPreferences = () => {
  expect(
    vi
      .mocked(setSetting)
      .mock.calls.map(([setting, value]) => [setting.key, value]),
  ).toEqual([
    ["uiMode", "sidePanel"],
    ["actionIconClick", "sidePanel"],
    ["contextMenuClick", "sidePanel"],
  ]);
};

describe("settings sidebar capability", () => {
  beforeEach(() => {
    vi.mocked(setSetting).mockClear();
    browserEnvironment.current = {};
    vi.stubGlobal("chrome", undefined);
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("disables the WebUI action despite the real browser shim advertising support", async () => {
    const user = userEvent.setup();
    // Match runtime-bootstrap: the WebUI shim is also exposed as global chrome.
    browserEnvironment.current = webBrowser;
    vi.stubGlobal("chrome", webBrowser);
    vi.stubGlobal("__NEXT_DATA__", {});
    const open = vi.spyOn(webBrowser.sidePanel, "open");
    const setOptions = vi.spyOn(webBrowser.sidePanel, "setOptions");

    expect(isSidepanelSupported()).toBe(true);
    renderSettingsLayout();
    const switchButton = screen.getByRole("button", {
      name: "Switch to Sidebar",
    });

    expect(switchButton).toBeDisabled();
    await user.click(switchButton);

    expect(setSetting).not.toHaveBeenCalled();
    expect(open).not.toHaveBeenCalled();
    expect(setOptions).not.toHaveBeenCalled();
  });

  it("keeps the Chrome extension action enabled and opens the active tab sidebar", async () => {
    const user = userEvent.setup();
    const open = vi.fn().mockResolvedValue(undefined);
    const setOptions = vi.fn().mockResolvedValue(undefined);
    const query = vi.fn(
      (_query: unknown, callback: (tabs: { id: number }[]) => void) =>
        callback([{ id: 41 }]),
    );
    vi.stubGlobal("chrome", {
      runtime: { id: "test-extension" },
      sidePanel: { open, setOptions },
      tabs: { query },
    });
    renderSettingsLayout();
    const switchButton = screen.getByRole("button", {
      name: "Switch to Sidebar",
    });

    expect(switchButton).toBeEnabled();
    await user.click(switchButton);

    expectSidebarPreferences();
    expect(query).toHaveBeenCalledExactlyOnceWith(
      { active: true, currentWindow: true },
      expect.any(Function),
    );
    expect(setOptions).toHaveBeenCalledExactlyOnceWith({
      tabId: 41,
      path: "sidepanel.html",
      enabled: true,
    });
    expect(open).toHaveBeenCalledExactlyOnceWith({ tabId: 41 });
  });

  it("keeps the Firefox extension action enabled and opens its sidebar", async () => {
    const user = userEvent.setup();
    const open = vi.fn().mockResolvedValue(undefined);
    browserEnvironment.current = { sidebarAction: { open } };
    renderSettingsLayout();
    const switchButton = screen.getByRole("button", {
      name: "Switch to Sidebar",
    });

    expect(switchButton).toBeEnabled();
    await user.click(switchButton);

    expectSidebarPreferences();
    expect(open).toHaveBeenCalledExactlyOnceWith();
  });

  it("keeps an unsupported extension action disabled without changing preferences", async () => {
    const user = userEvent.setup();
    renderSettingsLayout();
    const switchButton = screen.getByRole("button", {
      name: "Switch to Sidebar",
    });

    expect(switchButton).toBeDisabled();
    await user.click(switchButton);

    expect(setSetting).not.toHaveBeenCalled();
  });
});
