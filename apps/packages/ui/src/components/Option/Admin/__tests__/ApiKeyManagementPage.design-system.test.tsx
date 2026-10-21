// @vitest-environment jsdom

import React from "react";
import { Form, message } from "antd";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ApiKeyManagementPage from "../ApiKeyManagementPage";

const apiMock = vi.hoisted(() => ({
  canonical: vi.fn(),
  listAdminUsers: vi.fn(),
  listUserApiKeys: vi.fn(),
  createUserApiKey: vi.fn(),
  revokeUserApiKey: vi.fn(),
  rotateUserApiKey: vi.fn(),
}));

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: apiMock.canonical,
}));
const testConfig = (user = 1, serverUrl = "https://admin-a.example.test") => ({
  serverUrl,
  authMode: "multi-user",
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`,
});

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd");

  const Select = ({
    options = [],
    value,
    onChange,
    placeholder,
    loading,
  }: {
    options?: Array<{ value: number; label: string }>;
    value?: number | null;
    onChange?: (value: number) => void;
    placeholder?: string;
    loading?: boolean;
  }) => (
    <select
      aria-label="Select User"
      disabled={loading}
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.currentTarget.value))}
    >
      <option value="">{placeholder ?? "Select"}</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );

  const Modal = ({
    children,
    confirmLoading,
    okText,
    onCancel,
    onOk,
    open,
    title,
  }: {
    children?: React.ReactNode;
    confirmLoading?: boolean;
    okText?: React.ReactNode;
    onCancel?: () => void;
    onOk?: () => void;
    open?: boolean;
    title?: string;
  }) => {
    if (!open) return null;

    return (
      <div role="dialog" aria-label={title}>
        {children}
        <button type="button" disabled={confirmLoading} onClick={onOk}>
          {okText ?? "OK"}
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    );
  };

  return {
    ...actual,
    Modal,
    Select,
  };
});

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock,
}));

const expectDesignSystemAlertForText = async (text: string) => {
  const title = await screen.findByText(text);
  const alert = title.closest('[data-ds-component="Alert"]');

  expect(alert).not.toBeNull();
  return alert as HTMLElement;
};

const renderWithUser = async () => {
  render(<ApiKeyManagementPage />);

  await screen.findByText("API Key Management");
  fireEvent.change(await screen.findByLabelText("Select User"), {
    target: { value: "11" },
  });
};

describe("ApiKeyManagementPage design-system states", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    apiMock.canonical.mockReturnValue({
      config: testConfig(),
      authorityLoading: false,
    });

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      });
    }

    if (!(window as any).ResizeObserver) {
      (window as any).ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      };
    }

    apiMock.listAdminUsers.mockResolvedValue({
      users: [{ id: 11, username: "admin", email: "admin@example.com" }],
    });
    apiMock.listUserApiKeys.mockResolvedValue([]);
    apiMock.createUserApiKey.mockResolvedValue({ key: "sk-test-secret" });
    apiMock.revokeUserApiKey.mockResolvedValue({});
    apiMock.rotateUserApiKey.mockResolvedValue({});
  });

  it("explains single-user primary-key authentication without offering unusable managed keys", () => {
    apiMock.canonical.mockReturnValue({
      config: { serverUrl: "https://admin-a.example.test", authMode: "single-user" },
      authorityLoading: false,
    });
    render(<ApiKeyManagementPage />);
    expect(screen.getByText("Single-user API key")).toBeVisible();
    expect(screen.getByText(/accepts only the primary API key configured on the server/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Create Key" })).not.toBeInTheDocument();
    expect(apiMock.listAdminUsers).not.toHaveBeenCalled();
    expect(apiMock.listUserApiKeys).not.toHaveBeenCalled();
    expect(apiMock.createUserApiKey).not.toHaveBeenCalled();
  });

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce({ status: 403 });

    render(<ApiKeyManagementPage />);

    const alert = await expectDesignSystemAlertForText("Access Denied");
    expect(alert).toHaveAttribute("role", "alert");
    expect(alert).toHaveTextContent(
      "You don't have permission to manage API keys.",
    );
  });

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce({ status: 404 });

    render(<ApiKeyManagementPage />);

    const alert = await expectDesignSystemAlertForText("Not Available");
    expect(alert).toHaveAttribute("role", "alert");
    expect(alert).toHaveTextContent(
      "API key management is not available on this server.",
    );
  });

  it("renders key-load failures through the design-system Alert primitive", async () => {
    apiMock.listUserApiKeys.mockRejectedValueOnce(new Error("key API failed"));

    await renderWithUser();

    const alert = await expectDesignSystemAlertForText("key API failed");
    expect(alert).toHaveAttribute("role", "alert");
  });

  it("offers only supported key creation settings and retains rate-limiting navigation", async () => {
    await renderWithUser();
    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));

    expect(
      screen.queryByLabelText("Rate Limit (requests/minute, optional)"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("columnheader", { name: "Rate Limit" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: "Baseline limits and endpoint coverage live in Rate Limiting.",
      }),
    ).toHaveAttribute("href", "/admin/rate-limiting");
  });

  it("renders created-key success feedback through the design-system Alert primitive", async () => {
    await renderWithUser();

    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.change(screen.getByPlaceholderText("e.g. Production Key"), {
      target: { value: "Production Key" },
    });
    // The dialog commits with the verb, not a generic OK (#2892).
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));

    await waitFor(() => {
      expect(apiMock.createUserApiKey).toHaveBeenCalledWith(
        11,
        {
          name: "Production Key",
        },
        expect.objectContaining({
          configSnapshot: testConfig(),
          abortSignal: expect.any(AbortSignal),
        }),
      );
    });

    const alert = await expectDesignSystemAlertForText("New API Key Created");
    expect(alert).toHaveAttribute("role", "status");
    expect(alert).toHaveTextContent(
      "Copy this key now -- it will not be shown again:",
    );
    expect(alert).toHaveTextContent("sk-test-secret");
  });

  it("masks the created key after copy and supports reveal (#2898 L3)", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    await renderWithUser();

    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));

    const alert = await expectDesignSystemAlertForText("New API Key Created");
    expect(alert).toHaveTextContent("sk-test-secret");

    fireEvent.click(screen.getByRole("button", { name: "Copy key" }));
    expect(writeText).toHaveBeenCalledWith("sk-test-secret");
    await waitFor(() => {
      expect(alert).not.toHaveTextContent("sk-test-secret");
    });

    fireEvent.click(screen.getByRole("button", { name: "Reveal" }));
    expect(alert).toHaveTextContent("sk-test-secret");
  });

  it("auto-selects the only account on a multi-user server", async () => {
    render(<ApiKeyManagementPage />);

    await waitFor(() => {
      expect(apiMock.listUserApiKeys).toHaveBeenCalledWith(
        11,
        expect.objectContaining({
          configSnapshot: testConfig(),
          abortSignal: expect.any(AbortSignal),
        }),
      );
    });
    expect(await screen.findByText("API Keys")).toBeTruthy();
  });

  it("surfaces user-list failures with a retry instead of a silent dead end", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce(
      new Error("Request failed: 500 (GET /api/v1/admin/users)"),
    );

    render(<ApiKeyManagementPage />);

    const alert = await expectDesignSystemAlertForText("Unable to load users");
    expect(alert).toHaveTextContent("Request failed: 500");

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => {
      expect(apiMock.listAdminUsers).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(apiMock.listUserApiKeys).toHaveBeenCalledWith(
        11,
        expect.objectContaining({
          configSnapshot: testConfig(),
          abortSignal: expect.any(AbortSignal),
        }),
      );
    });
  });

  it("clears the revealed key when the selected user changes", async () => {
    apiMock.listAdminUsers.mockResolvedValue({
      users: [
        { id: 11, username: "alice", email: "alice@example.test" },
        { id: 22, username: "bob", email: "bob@example.test" },
      ],
    });
    await renderWithUser();
    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    await screen.findByText("sk-test-secret");
    fireEvent.change(screen.getByLabelText("Select User"), {
      target: { value: "22" },
    });
    expect(screen.queryByText("sk-test-secret")).not.toBeInTheDocument();
    expect(screen.queryByText("New API Key Created")).not.toBeInTheDocument();
  });

  it("ignores a previous user key list that settles after the new user", async () => {
    apiMock.listAdminUsers.mockResolvedValue({
      users: [
        { id: 11, username: "alice", email: "alice@example.test" },
        { id: 22, username: "bob", email: "bob@example.test" },
      ],
    });
    let finish!: (value: unknown) => void;
    apiMock.listUserApiKeys.mockImplementation((id) =>
      id === 11
        ? new Promise((resolve) => {
            finish = resolve;
          })
        : Promise.resolve([
            { id: 202, name: "Bob scoped key", is_active: true },
          ]),
    );
    await renderWithUser();
    await waitFor(() =>
      expect(apiMock.listUserApiKeys).toHaveBeenCalledWith(
        11,
        expect.objectContaining({
          configSnapshot: testConfig(),
          abortSignal: expect.any(AbortSignal),
        }),
      ),
    );
    fireEvent.change(screen.getByLabelText("Select User"), {
      target: { value: "22" },
    });
    await screen.findByText("Bob scoped key");
    await act(async () =>
      finish([{ id: 101, name: "Alice stale key", is_active: true }]),
    );
    expect(screen.queryByText("Alice stale key")).not.toBeInTheDocument();
    expect(screen.getByText("Bob scoped key")).toBeVisible();
  });

  it.each(["server", "account", "unresolved", "single-user"])(
    "clears held key material when %s authority changes",
    async (change) => {
      const view = render(<ApiKeyManagementPage />);
      await waitFor(() =>
        expect(apiMock.listUserApiKeys).toHaveBeenCalledWith(
          11,
          expect.objectContaining({
            configSnapshot: testConfig(),
            abortSignal: expect.any(AbortSignal),
          }),
        ),
      );
      fireEvent.click(
        await screen.findByRole("button", { name: "Create Key" }),
      );
      fireEvent.click(screen.getByRole("button", { name: "Create key" }));
      await screen.findByText("sk-test-secret");
      apiMock.canonical.mockReturnValue({
        config:
          change === "server"
            ? testConfig(1, "https://admin-b.example.test")
            : change === "account"
              ? testConfig(2)
              : change === "single-user"
                ? { serverUrl: testConfig().serverUrl, authMode: "single-user" }
                : testConfig(),
        authorityLoading: change === "unresolved",
      });
      view.rerender(<ApiKeyManagementPage />);
      expect(screen.queryByText("sk-test-secret")).not.toBeInTheDocument();
      expect(screen.queryByText("New API Key Created")).not.toBeInTheDocument();
    },
  );

  it("does not dispatch a key creation after authority changes during form validation", async () => {
    const originalUseForm = Form.useForm;
    let form!: ReturnType<typeof Form.useForm>[0];
    vi.spyOn(Form, "useForm").mockImplementation((...args) => {
      const result = originalUseForm(...args);
      form = result[0];
      return result;
    });
    const view = render(<ApiKeyManagementPage />);
    await screen.findByRole("button", { name: "Create Key" });
    let finish!: (value: { name: string }) => void;
    vi.spyOn(form, "validateFields").mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    apiMock.canonical.mockReturnValue({
      config: testConfig(2),
      authorityLoading: false,
    });
    view.rerender(<ApiKeyManagementPage />);
    await act(async () => finish({ name: "old authority" }));
    expect(apiMock.createUserApiKey).not.toHaveBeenCalled();
    expect(screen.queryByText("sk-test-secret")).not.toBeInTheDocument();
  });

  it.each([
    ["create", "authority"],
    ["rotate", "authority"],
    ["revoke", "authority"],
    ["create", "user"],
    ["rotate", "user"],
    ["revoke", "user"],
  ])(
    "retires pending %s results when %s changes",
    async (operation, boundary) => {
      apiMock.listAdminUsers.mockResolvedValue({
        users: [
          { id: 11, username: "alice" },
          { id: 22, username: "bob" },
        ],
      });
      apiMock.listUserApiKeys.mockResolvedValue([
        { id: 101, name: "Old account key", is_active: true },
      ]);
      const method =
        operation === "create"
          ? apiMock.createUserApiKey
          : operation === "rotate"
            ? apiMock.rotateUserApiKey
            : apiMock.revokeUserApiKey;
      let finish!: (value: unknown) => void;
      method.mockReturnValueOnce(
        new Promise((resolve) => {
          finish = resolve;
        }),
      );
      const success = vi.spyOn(message, "success");
      const view = render(<ApiKeyManagementPage />);
      fireEvent.change(await screen.findByLabelText("Select User"), {
        target: { value: "11" },
      });
      await screen.findByText("Old account key");
      if (operation === "create") {
        fireEvent.click(screen.getByRole("button", { name: "Create Key" }));
        fireEvent.click(screen.getByRole("button", { name: "Create key" }));
      } else {
        fireEvent.click(
          screen.getByRole("button", {
            name: operation === "rotate" ? "Rotate" : "Revoke",
          }),
        );
        fireEvent.click(
          await screen.findByRole("button", {
            name: operation === "rotate" ? "Rotate key" : "Revoke key",
          }),
        );
      }
      await waitFor(() => expect(method).toHaveBeenCalledTimes(1));
      const options = method.mock.calls[0].at(-1);
      apiMock.listUserApiKeys.mockResolvedValue([
        { id: 202, name: "New account key", is_active: true },
      ]);
      if (boundary === "authority") {
        apiMock.canonical.mockReturnValue({
          config: testConfig(2),
          authorityLoading: false,
        });
        view.rerender(<ApiKeyManagementPage />);
      }
      fireEvent.change(await screen.findByLabelText("Select User"), {
        target: { value: "22" },
      });
      await act(async () => finish({ key: "sk-retired-secret" }));
      expect(screen.queryByText("sk-retired-secret")).not.toBeInTheDocument();
      expect(success).not.toHaveBeenCalled();
      expect(options.abortSignal.aborted).toBe(true);
      expect(await screen.findByText("New account key")).toBeVisible();
      expect(screen.queryByText("Old account key")).not.toBeInTheDocument();
    },
  );

  it("ignores stale key errors after switching users", async () => {
    apiMock.listAdminUsers.mockResolvedValue({
      users: [
        { id: 11, username: "alice" },
        { id: 22, username: "bob" },
      ],
    });
    let fail!: (reason: unknown) => void;
    apiMock.listUserApiKeys.mockImplementation((id) =>
      id === 11
        ? new Promise((_, reject) => {
            fail = reject;
          })
        : Promise.resolve([{ id: 202, name: "Bob current key" }]),
    );
    await renderWithUser();
    await waitFor(() => expect(fail).toBeTypeOf("function"));
    fireEvent.change(screen.getByLabelText("Select User"), {
      target: { value: "22" },
    });
    await screen.findByText("Bob current key");
    await act(async () =>
      fail({ status: 403, message: "Old account forbidden" }),
    );
    expect(screen.queryByText("Access Denied")).not.toBeInTheDocument();
    expect(screen.getByText("Bob current key")).toBeVisible();
  });

  it("does not revive an earlier request when the user selection returns to Alice", async () => {
    apiMock.listAdminUsers.mockResolvedValue({
      users: [
        { id: 11, username: "alice" },
        { id: 22, username: "bob" },
      ],
    });
    let finish!: (value: unknown) => void;
    apiMock.listUserApiKeys
      .mockReturnValueOnce(
        new Promise((resolve) => {
          finish = resolve;
        }),
      )
      .mockResolvedValueOnce([{ id: 202, name: "Bob key" }])
      .mockResolvedValueOnce([{ id: 102, name: "Alice current key" }]);
    await renderWithUser();
    await waitFor(() =>
      expect(apiMock.listUserApiKeys).toHaveBeenCalledTimes(1),
    );
    fireEvent.change(screen.getByLabelText("Select User"), {
      target: { value: "22" },
    });
    await screen.findByText("Bob key");
    fireEvent.change(screen.getByLabelText("Select User"), {
      target: { value: "11" },
    });
    await screen.findByText("Alice current key");
    await act(async () => finish([{ id: 101, name: "Alice obsolete key" }]));
    expect(screen.queryByText("Alice obsolete key")).not.toBeInTheDocument();
    expect(screen.getByText("Alice current key")).toBeVisible();
  });

  it("keeps the same authority workspace during a verified same-user token refresh", async () => {
    let finish!: (value: unknown) => void;
    apiMock.createUserApiKey.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const view = render(<ApiKeyManagementPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    await waitFor(() =>
      expect(apiMock.createUserApiKey).toHaveBeenCalledTimes(1),
    );
    const config = testConfig();
    apiMock.canonical.mockReturnValue({
      config: { ...config, accessToken: config.accessToken + "-refreshed" },
      authorityLoading: false,
    });
    view.rerender(<ApiKeyManagementPage />);
    await act(async () => finish({ key: "sk-refreshed-owner-secret" }));
    expect(await screen.findByText("sk-refreshed-owner-secret")).toBeVisible();
    expect(apiMock.listAdminUsers).toHaveBeenCalledTimes(1);
    expect(apiMock.createUserApiKey).toHaveBeenCalledTimes(1);
  });

  it("suppresses a retired mutation error and leaves the new authority usable", async () => {
    let fail!: (reason: unknown) => void;
    apiMock.createUserApiKey.mockReturnValueOnce(
      new Promise((_, reject) => {
        fail = reject;
      }),
    );
    const error = vi.spyOn(message, "error");
    const view = render(<ApiKeyManagementPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    await waitFor(() =>
      expect(apiMock.createUserApiKey).toHaveBeenCalledTimes(1),
    );
    apiMock.canonical.mockReturnValue({
      config: testConfig(2),
      authorityLoading: false,
    });
    view.rerender(<ApiKeyManagementPage />);
    await act(async () => fail({ status: 403, message: "retired forbidden" }));
    expect(error).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    expect(await screen.findByText("sk-test-secret")).toBeVisible();
    expect(apiMock.createUserApiKey).toHaveBeenLastCalledWith(
      11,
      { name: undefined },
      expect.objectContaining({
        configSnapshot: testConfig(2),
        abortSignal: expect.any(AbortSignal),
      }),
    );
  });

  it("keeps a current create error visible and permits one user retry", async () => {
    apiMock.createUserApiKey.mockRejectedValueOnce(
      new Error("current create failed"),
    );
    const error = vi.spyOn(message, "error");
    await renderWithUser();
    fireEvent.click(screen.getByRole("button", { name: "Create Key" }));
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    await waitFor(() =>
      expect(error).toHaveBeenCalledWith("current create failed"),
    );
    expect(
      screen.getByRole("dialog", { name: "Create API Key" }),
    ).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Create key" }));
    expect(await screen.findByText("sk-test-secret")).toBeVisible();
    expect(apiMock.createUserApiKey).toHaveBeenCalledTimes(2);
  });
});
