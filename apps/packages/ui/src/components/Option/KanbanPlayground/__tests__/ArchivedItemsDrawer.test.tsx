import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, it, vi } from "vitest";
import { ArchivedItemsDrawer } from "../ArchivedItemsDrawer";
import { KanbanScopeProvider } from "../KanbanScope";
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope";

const scopeConfig = { serverUrl: "https://kanban.example.test", authMode: "single-user" as const, apiKey: "fixture-key" };

it("opens the 400px archive drawer without a deprecated width warning", () => {
  const client = new QueryClient();
  client.setQueryData(["kanban", deriveConnectionAuthorityId(scopeConfig), "kanban-boards-archived"], { boards: [] });
  const errors = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    render(
      <QueryClientProvider client={client}>
        <KanbanScopeProvider config={scopeConfig}>
          <ArchivedItemsDrawer open onClose={() => {}} />
        </KanbanScopeProvider>
      </QueryClientProvider>,
    );
    expect(screen.getByText("No archived items")).toBeInTheDocument();
    expect(document.querySelector(".ant-drawer-content-wrapper")).toHaveStyle({
      width: "400px",
    });
    expect(
      errors.mock.calls.some((args) =>
        args.some((arg) => String(arg).includes("`width` is deprecated")),
      ),
    ).toBe(false);
  } finally {
    errors.mockRestore();
    client.clear();
  }
});
