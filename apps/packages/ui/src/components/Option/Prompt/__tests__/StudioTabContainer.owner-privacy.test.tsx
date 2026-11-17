import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { usePromptStudioStore } from "@/store/prompt-studio";
import { StudioTabContainer } from "../Studio/StudioTabContainer";

const fixture = vi.hoisted(() => ({
  ownerId: "bob",
  ready: true,
  online: true,
}));
const api = vi.hoisted(() => ({
  projects: vi.fn(),
  defaults: vi.fn(),
  save: vi.fn(),
  capability: vi.fn(),
}));
vi.mock("@/hooks/usePromptLocalAuthority", () => ({
  usePromptLocalAuthority: () => ({ ...fixture, boundaryKey: fixture.ownerId }),
}));
vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => fixture.online,
}));
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? key,
  }),
}));
vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: api.capability,
  getPromptStudioStatus: async () => ({ data: { data: {} } }),
  listProjects: api.projects,
  archiveProject: vi.fn(),
  unarchiveProject: vi.fn(),
  deleteProject: vi.fn(),
}));
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: () => api.defaults(),
  getPromptStudioDefaultsForOwner: (owner: string) => api.defaults(owner),
  setPromptStudioDefaults: api.save,
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: async () => null },
}));
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(),
}));
vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    onClick,
  }: React.PropsWithChildren<{ onClick?: () => void }>) => (
    <button onClick={onClick}>{children}</button>
  ),
}));
vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title }: { title: string }) => <div>{title}</div>,
}));
vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  default: () => <div>Connect to server</div>,
}));
vi.mock("../Studio/QueueHealthWidget", () => ({
  QueueHealthWidget: () => null,
}));
vi.mock("../Studio/Projects/ProjectFormModal", () => ({
  ProjectFormModal: () => null,
}));
vi.mock("../Studio/Prompts/StudioPromptsTab", () => ({
  StudioPromptsTab: () => {
    const id = usePromptStudioStore((state) => state.selectedProjectId);
    return <div>Selected project {id}</div>;
  },
}));
vi.mock("antd", () => ({
  Segmented: ({
    value,
    options,
    onChange,
  }: {
    value: string;
    options: Array<{
      value: string;
      label: React.ReactNode;
      disabled?: boolean;
    }>;
    onChange: (value: string) => void;
  }) => (
    <div>
      {options.map((option) => (
        <button
          key={option.value}
          aria-pressed={value === option.value}
          disabled={option.disabled}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  ),
  Select: ({
    value,
    options,
    onChange,
    "data-testid": testId,
  }: {
    value: string | number;
    options: Array<{ value: string | number; label: string }>;
    onChange: (value: string) => void;
    "data-testid"?: string;
  }) => (
    <select
      data-testid={testId}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Popover: ({
    children,
    content,
  }: React.PropsWithChildren<{ content: React.ReactNode }>) => (
    <>
      {children}
      {content}
    </>
  ),
  Switch: () => null,
  Badge: () => null,
  Tooltip: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Tag: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Dropdown: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Skeleton: () => <div>Loading projects</div>,
  Input: ({
    value,
    onChange,
  }: {
    value: string;
    onChange: React.ChangeEventHandler<HTMLInputElement>;
  }) => <input value={value} onChange={onChange} />,
  Table: ({
    dataSource,
    onRow,
  }: {
    dataSource: Array<{ id: number; name: string }>;
    onRow: (record: { id: number; name: string }) => { onClick: () => void };
  }) => (
    <div>
      {dataSource.map((record) => (
        <button key={record.id} onClick={onRow(record).onClick}>
          {record.name}
        </button>
      ))}
    </div>
  ),
  notification: { success: vi.fn(), error: vi.fn() },
}));

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const projects = (name: string, id = 1) => ({ data: { data: [{ id, name }] } });
const defaults: {
  defaultProjectId: number | null;
  autoSyncWorkspacePrompts: boolean;
} = { defaultProjectId: null, autoSyncWorkspacePrompts: true };
let client: QueryClient;
const tree = () => (
  <QueryClientProvider client={client}>
    <MemoryRouter>
      <StudioTabContainer />
    </MemoryRouter>
  </QueryClientProvider>
);

beforeEach(() => {
  fixture.ownerId = "bob";
  fixture.ready = true;
  fixture.online = true;
  usePromptStudioStore.getState().resetStore();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  api.capability.mockReset().mockResolvedValue(true);
  api.projects
    .mockReset()
    .mockImplementation(async () =>
      fixture.ownerId === "bob"
        ? projects("Bob private project")
        : { data: { data: [] } },
    );
  api.defaults.mockReset().mockResolvedValue(defaults);
  api.save
    .mockReset()
    .mockImplementation(async (update) => ({ ...defaults, ...update }));
});
afterEach(() => {
  cleanup();
  client.clear();
});

describe("Studio private workspace ownership", () => {
  it("retires cached projects and selected project immediately on A to B", async () => {
    const view = render(tree());
    fireEvent.click(
      await screen.findByRole("button", { name: "Bob private project" }),
    );
    await screen.findByText("Selected project 1");
    fixture.ownerId = "alice";
    view.rerender(tree());
    expect(screen.queryByText("Selected project 1")).toBeNull();
    expect(screen.queryByText("Bob private project")).toBeNull();
    await screen.findByText("No project selected");
    expect(
      screen.getByRole("button", {
        name: /Prompts \(Select a project first\)/,
      }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Projects" }));
    await screen.findByText("No projects yet");
  });

  it("never reads a prior owner's shared cache while ownership is unresolved", async () => {
    client.setQueryData(
      ["prompt-studio", "settings-projects"],
      projects("Bob cached secret"),
    );
    client.setQueryData(
      ["prompt-studio", "projects", { include_deleted: false }],
      projects("Bob cached secret"),
    );
    fixture.ready = false;
    render(tree());
    expect(screen.queryByText("Bob cached secret")).toBeNull();
    await act(async () => {
      await Promise.resolve();
    });
    expect(api.capability).not.toHaveBeenCalled();
    expect(api.projects).not.toHaveBeenCalled();
    expect(api.defaults).not.toHaveBeenCalled();
  });

  it("ignores a late prior-owner list even when both owners use project id 1", async () => {
    const pending = deferred<ReturnType<typeof projects>>();
    api.projects.mockImplementation(() =>
      fixture.ownerId === "bob"
        ? pending.promise
        : Promise.resolve(projects("Alice project")),
    );
    const view = render(tree());
    await waitFor(() => expect(api.projects).toHaveBeenCalled());
    fixture.ownerId = "alice";
    view.rerender(tree());
    await act(async () => pending.resolve(projects("Bob late secret")));
    expect(screen.queryByText("Bob late secret")).toBeNull();
    await screen.findByRole("button", { name: "Alice project" });
  });

  it("keeps late defaults and mutation selection inside the retired owner", async () => {
    const pending = deferred<typeof defaults>();
    api.save.mockReturnValue(pending.promise);
    const view = render(tree());
    await screen.findByRole("button", { name: "Bob private project" });
    fireEvent.change(screen.getByTestId("studio-settings-default-project"), {
      target: { value: "1" },
    });
    await waitFor(() =>
      expect(api.save).toHaveBeenCalledWith({ defaultProjectId: 1 }, "bob"),
    );
    fixture.ownerId = "alice";
    view.rerender(tree());
    await act(async () =>
      pending.resolve({ ...defaults, defaultProjectId: 1 }),
    );
    await screen.findByText("No projects yet");
    expect(
      screen.getByRole("button", {
        name: /Prompts \(Select a project first\)/,
      }),
    ).toBeDisabled();
    expect(screen.getByTestId("studio-settings-default-project")).toHaveValue(
      "none",
    );
    expect(api.defaults).toHaveBeenCalledWith("alice");
  });

  it("preserves selection and draft search on a same-owner rerender", async () => {
    const view = render(tree());
    await screen.findByRole("button", { name: "Bob private project" });
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Bob" } });
    view.rerender(tree());
    expect(screen.getByRole("textbox")).toHaveValue("Bob");
    fireEvent.click(
      screen.getByRole("button", { name: "Bob private project" }),
    );
    await screen.findByText("Selected project 1");
    view.rerender(tree());
    expect(screen.getByText("Selected project 1")).toBeInTheDocument();
  });

  it("does not revive an earlier A lifetime after A to B to A with late defaults", async () => {
    const pending = deferred<typeof defaults>();
    api.defaults
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValue(defaults);
    const view = render(tree());
    await screen.findByRole("button", { name: "Bob private project" });
    fixture.ownerId = "alice";
    view.rerender(tree());
    await screen.findByText("No projects yet");
    fixture.ownerId = "bob";
    view.rerender(tree());
    await screen.findByRole("button", { name: "Bob private project" });
    await act(async () =>
      pending.resolve({ ...defaults, defaultProjectId: 1 }),
    );
    expect(screen.getByTestId("studio-settings-default-project")).toHaveValue(
      "none",
    );
    expect(
      screen.getByRole("button", {
        name: /Prompts \(Select a project first\)/,
      }),
    ).toBeDisabled();
  });

  it("can load a private workspace through StrictMode effect cleanup", async () => {
    render(<React.StrictMode>{tree()}</React.StrictMode>);
    fireEvent.click(
      await screen.findByRole("button", { name: "Bob private project" }),
    );
    await screen.findByText("Selected project 1");
  });

  it("keeps selected project when the same owner goes offline and returns", async () => {
    const view = render(tree());
    fireEvent.click(
      await screen.findByRole("button", { name: "Bob private project" }),
    );
    await screen.findByText("Selected project 1");
    fixture.online = false;
    view.rerender(tree());
    await screen.findByText("Connect to server");
    fixture.online = true;
    view.rerender(tree());
    await screen.findByText("Selected project 1");
  });

  it("preserves offline guidance without private queries", async () => {
    fixture.online = false;
    fixture.ready = false;
    render(tree());
    await screen.findByText("Connect to server");
    expect(api.projects).not.toHaveBeenCalled();
  });
});
