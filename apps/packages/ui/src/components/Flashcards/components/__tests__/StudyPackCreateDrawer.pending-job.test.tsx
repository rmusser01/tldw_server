import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createInstance } from "i18next";
import { I18nextProvider } from "react-i18next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import optionEnglish from "@/assets/locale/en/option.json";
import {
  createStudyPackJob,
  getStudyPackJob,
  type StudyPackJobApiStatus,
  type StudyPackJobStatusResponse,
} from "@/services/flashcards";
import { StudyPackCreateDrawer } from "../StudyPackCreateDrawer";

const navigate = vi.hoisted(() => vi.fn());
const message = vi.hoisted(() => ({ error: vi.fn() }));

vi.mock("react-router-dom", () => ({ useNavigate: () => navigate }));
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => message }));
vi.mock(
  "../../hooks",
  async () => await vi.importActual("../../hooks/useStudyPackQueries"),
);
vi.mock("../../hooks/useFlashcardQueries", () => ({
  useFlashcardsEnabled: () => ({ flashcardsEnabled: true }),
}));
vi.mock("@/services/flashcards", async () => ({
  ...(await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards",
  )),
  createStudyPackJob: vi.fn(),
  getStudyPackJob: vi.fn(),
}));

const intent = {
  title: "Citrine study pack",
  sourceItems: [
    {
      sourceType: "media" as const,
      sourceId: "42",
      sourceTitle: "Citrine source",
    },
  ],
};
const jobKey = ["flashcards:study-packs:job", 91];
const response = (
  status: StudyPackJobApiStatus,
  id = 91,
): StudyPackJobStatusResponse => ({
  job: {
    id,
    status,
    domain: "study_packs",
    queue: "default",
    job_type: "study_pack_generate",
  },
  study_pack:
    status === "completed"
      ? {
          id: 31,
          title: "Citrine study pack",
          deck_id: 8,
          source_bundle_json: {},
          status: "active",
          deleted: false,
          client_id: "2",
          version: 1,
        }
      : null,
  error: status === "failed" ? "Generation failed" : null,
});

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

const clients: QueryClient[] = [];
const newClient = () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
  clients.push(client);
  return client;
};
let i18n: ReturnType<typeof createInstance>;
const button = () => screen.getByRole("button", { name: /Create study pack/i });
const mount = (
  client = newClient(),
  props: Partial<React.ComponentProps<typeof StudyPackCreateDrawer>> = {},
) => {
  const onClose = vi.fn();
  const onCreated = vi.fn();
  const view = render(
    <QueryClientProvider client={client}>
      <I18nextProvider i18n={i18n}>
        <StudyPackCreateDrawer
          open
          initialIntent={intent}
          onClose={onClose}
          onCreated={onCreated}
          {...props}
        />
      </I18nextProvider>
    </QueryClientProvider>,
  );
  return { ...view, client, onClose, onCreated };
};
const submit = async (client: QueryClient) => {
  fireEvent.click(button());
  await waitFor(() =>
    expect(client.getQueryState(jobKey)?.status).toBe("success"),
  );
};

describe("StudyPackCreateDrawer accepted job lifecycle", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.mocked(createStudyPackJob).mockResolvedValue({
      job: response("queued").job,
    });
    vi.mocked(getStudyPackJob).mockResolvedValue(response("queued"));
    i18n = createInstance();
    await i18n.init({
      lng: "en",
      fallbackLng: "en",
      resources: { en: { option: optionEnglish } },
    });
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((media: string) => ({
        matches: false,
        media,
        addListener() {},
        removeListener() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent: () => false,
      })),
    });
  });

  afterEach(() => {
    clients.splice(0).forEach((client) => client.clear());
    vi.unstubAllGlobals();
  });

  it("keeps an accepted request pending before the first status response", async () => {
    const poll = deferred<StudyPackJobStatusResponse>();
    vi.mocked(getStudyPackJob).mockReturnValue(poll.promise);
    mount();
    fireEvent.click(button());
    await waitFor(() => expect(getStudyPackJob).toHaveBeenCalledWith(91));
    expect(screen.getByRole("status")).toHaveTextContent(/accepted/i);
    expect(button()).toBeDisabled();
    fireEvent.click(button());
    expect(createStudyPackJob).toHaveBeenCalledTimes(1);
    await act(async () => {
      poll.resolve(response("queued"));
    });
  });

  it.each(["queued", "running"] as const)(
    "blocks duplicate submission while %s between polls",
    async (status) => {
      vi.mocked(getStudyPackJob).mockResolvedValue(response(status));
      const { client } = mount();
      await submit(client);
      expect(client.getQueryState(jobKey)?.fetchStatus).toBe("idle");
      expect(button()).toBeDisabled();
      expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
      expect(screen.getByRole("status")).toHaveTextContent(
        status === "queued" ? /queued/i : /creating your study pack/i,
      );
      fireEvent.click(button());
      expect(createStudyPackJob).toHaveBeenCalledTimes(1);
      expect(
        screen.getByDisplayValue("Citrine study pack"),
      ).toBeInTheDocument();
      expect(screen.getByText("media · 42")).toBeInTheDocument();
    },
  );

  it("keeps the same job pending across queued, active fetch and running gaps", async () => {
    const { client } = mount();
    await submit(client);
    const nextPoll = deferred<StudyPackJobStatusResponse>();
    vi.mocked(getStudyPackJob).mockReturnValueOnce(nextPoll.promise);
    let refresh!: Promise<void>;
    await act(async () => {
      refresh = client.refetchQueries({ queryKey: jobKey });
    });
    expect(button()).toBeDisabled();
    await act(async () => {
      nextPoll.resolve(response("running"));
      await refresh;
    });
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(
        /creating your study pack/i,
      ),
    );
    expect(client.getQueryState(jobKey)?.fetchStatus).toBe("idle");
    expect(button()).toBeDisabled();
    expect(createStudyPackJob).toHaveBeenCalledTimes(1);
  });

  it("keeps the accepted job locked on a poll error and recovers on the same query", async () => {
    const { client, onClose } = mount();
    await submit(client);
    vi.mocked(getStudyPackJob).mockRejectedValueOnce(
      new Error("temporary status failure"),
    );
    await act(async () => {
      await client.refetchQueries({ queryKey: jobKey });
    });
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(/unable to check/i),
    );
    expect(button()).toBeDisabled();
    fireEvent.click(button());
    expect(createStudyPackJob).toHaveBeenCalledTimes(1);
    expect(message.error).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    vi.mocked(getStudyPackJob).mockResolvedValue(response("running"));
    await act(async () => {
      await client.refetchQueries({ queryKey: jobKey });
    });
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(
        /creating your study pack/i,
      ),
    );
    expect(navigate).not.toHaveBeenCalled();
  });

  it.each(["failed", "cancelled"] as const)(
    "allows deliberate retry after %s with the same inputs",
    async (status) => {
      const { client } = mount();
      await submit(client);
      vi.mocked(getStudyPackJob).mockResolvedValueOnce(response(status));
      await act(async () => {
        await client.refetchQueries({ queryKey: jobKey });
      });
      await waitFor(() => expect(message.error).toHaveBeenCalledTimes(1));
      expect(button()).toBeEnabled();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      expect(
        screen.getByDisplayValue("Citrine study pack"),
      ).toBeInTheDocument();
      expect(screen.getByText("Citrine source")).toBeInTheDocument();
      vi.mocked(createStudyPackJob).mockResolvedValueOnce({
        job: response("queued", 92).job,
      });
      vi.mocked(getStudyPackJob).mockResolvedValue(response("queued", 92));
      fireEvent.click(button());
      await waitFor(() => expect(getStudyPackJob).toHaveBeenCalledWith(92));
      expect(createStudyPackJob).toHaveBeenLastCalledWith({
        title: "Citrine study pack",
        source_items: [
          {
            source_type: "media",
            source_id: "42",
            source_title: "Citrine source",
          },
        ],
        deck_mode: "new",
      });
      expect(navigate).not.toHaveBeenCalled();
    },
  );

  it("navigates once to the returned deck on successful completion", async () => {
    const { client, onClose, onCreated } = mount();
    await submit(client);
    vi.mocked(getStudyPackJob).mockResolvedValue(response("completed"));
    await act(async () => {
      await client.refetchQueries({ queryKey: jobKey });
    });
    await waitFor(() =>
      expect(navigate).toHaveBeenCalledWith(
        "/flashcards?tab=review&deck_id=8",
        { replace: true },
      ),
    );
    expect(onCreated).toHaveBeenCalledWith(response("completed").study_pack);
    expect(onClose).toHaveBeenCalledTimes(1);
    await act(async () => {
      await client.refetchQueries({ queryKey: jobKey });
    });
    expect(navigate).toHaveBeenCalledTimes(1);
  });

  it.each(["missing-pack", "missing-deck"] as const)(
    "releases an unusable completed result (%s) for deliberate retry",
    async (missing) => {
      const { client, onClose, onCreated } = mount();
      await submit(client);
      const completed = response("completed");
      if (missing === "missing-pack") completed.study_pack = null;
      else completed.study_pack!.deck_id = null;
      vi.mocked(getStudyPackJob).mockResolvedValueOnce(completed);
      await act(async () => {
        await client.refetchQueries({ queryKey: jobKey });
      });
      await waitFor(() => expect(button()).toBeEnabled());
      expect(message.error).toHaveBeenCalledWith(
        "Study pack completed, but its review deck is unavailable.",
      );
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      expect(
        screen.getByDisplayValue("Citrine study pack"),
      ).toBeInTheDocument();
      expect(screen.getByText("Citrine source")).toBeInTheDocument();
      expect(onClose).not.toHaveBeenCalled();
      expect(onCreated).not.toHaveBeenCalled();
      expect(navigate).not.toHaveBeenCalled();
      vi.mocked(createStudyPackJob).mockResolvedValueOnce({
        job: response("queued", 92).job,
      });
      vi.mocked(getStudyPackJob).mockResolvedValue(response("queued", 92));
      fireEvent.click(button());
      await waitFor(() => expect(getStudyPackJob).toHaveBeenCalledWith(92));
      expect(createStudyPackJob).toHaveBeenCalledTimes(2);
    },
  );

  it("allows retry of a rejected creation request without losing sources", async () => {
    vi.mocked(createStudyPackJob).mockRejectedValueOnce(
      new Error("POST failed"),
    );
    mount();
    fireEvent.click(button());
    await waitFor(() => expect(message.error).toHaveBeenCalledTimes(1));
    expect(button()).toBeEnabled();
    expect(screen.getByDisplayValue("Citrine study pack")).toBeInTheDocument();
    expect(screen.getByText("media · 42")).toBeInTheDocument();
    expect(getStudyPackJob).not.toHaveBeenCalled();
    expect(navigate).not.toHaveBeenCalled();
  });

  it("does not attach a late old-account acceptance to a freshly keyed drawer", async () => {
    const accepted = deferred<Awaited<ReturnType<typeof createStudyPackJob>>>();
    vi.mocked(createStudyPackJob).mockReturnValueOnce(accepted.promise);
    const { rerender, client } = mount();
    fireEvent.click(button());
    await waitFor(() => expect(createStudyPackJob).toHaveBeenCalledTimes(1));
    rerender(
      <QueryClientProvider client={client}>
        <I18nextProvider i18n={i18n}>
          <StudyPackCreateDrawer
            key="other-account"
            open
            onClose={vi.fn()}
            initialIntent={{ ...intent, title: "Other account draft" }}
          />
        </I18nextProvider>
      </QueryClientProvider>,
    );
    await act(async () => {
      accepted.resolve({ job: response("completed").job });
    });
    expect(screen.getByDisplayValue("Other account draft")).toBeInTheDocument();
    expect(button()).toBeEnabled();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(getStudyPackJob).not.toHaveBeenCalled();
    expect(navigate).not.toHaveBeenCalled();
  });
});
