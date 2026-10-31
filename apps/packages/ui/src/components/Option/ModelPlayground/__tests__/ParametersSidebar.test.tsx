import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { ParametersSidebar } from "../ParametersSidebar";
import { useStoreChatModelSettings } from "@/store/model";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }),
}));
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ragSearchMode: "hybrid",
    setRagSearchMode: vi.fn(),
    ragTopK: 8,
    setRagTopK: vi.fn(),
    ragEnableGeneration: true,
    setRagEnableGeneration: vi.fn(),
    ragEnableCitations: true,
    setRagEnableCitations: vi.fn(),
  }),
}));
vi.mock("@/components/Option/Playground/playground-features", () => ({
  ParameterPresetsDropdown: () => null,
  JsonModeToggle: () => null,
}));

beforeEach(() => {
  useStoreChatModelSettings.getState().reset();
});

it("explains one chat answer without offering an ineffective RAG generation toggle", () => {
  render(<ParametersSidebar />);
  fireEvent.click(screen.getByText("RAG Settings"));
  expect(
    screen.queryByText("Enable Answer Generation"),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText(
      "Chat retrieves sources, then the selected model generates one answer.",
    ),
  ).toBeInTheDocument();
  expect(screen.getByText("Temperature")).toBeInTheDocument();
  expect(screen.getByText("Top P")).toBeInTheDocument();
});
