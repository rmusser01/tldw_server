import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, it, vi } from "vitest";
import { ArchivedItemsDrawer } from "../ArchivedItemsDrawer";

it("opens the 400px archive drawer without a deprecated width warning", () => {
  const client = new QueryClient();
  client.setQueryData(["kanban-boards-archived"], { boards: [] });
  const errors = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    render(
      <QueryClientProvider client={client}>
        <ArchivedItemsDrawer open onClose={() => {}} />
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
