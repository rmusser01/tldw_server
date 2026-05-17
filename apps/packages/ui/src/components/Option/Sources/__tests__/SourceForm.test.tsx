import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const hookMocks = vi.hoisted(() => ({
  useCreateIngestionSourceMutation: vi.fn(),
  useUpdateIngestionSourceMutation: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => routerMocks.navigate
  }
})

vi.mock("@/hooks/use-ingestion-sources", () => ({
  useCreateIngestionSourceMutation: (...args: unknown[]) =>
    hookMocks.useCreateIngestionSourceMutation(...args),
  useUpdateIngestionSourceMutation: (...args: unknown[]) =>
    hookMocks.useUpdateIngestionSourceMutation(...args)
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

import { SourceForm } from "@/components/Option/Sources/SourceForm"

describe("SourceForm", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: "42" })),
      isPending: false
    })
    hookMocks.useUpdateIngestionSourceMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: "42" })),
      isPending: false
    })
    capabilityMocks.useServerCapabilities.mockReturnValue({
      capabilities: { canCreateLocalDirectoryIngestionSource: true },
      loading: false,
      refresh: vi.fn()
    })
  })

  it("uses notes folder sync preset defaults when creating a source", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" preset="notes-folder-sync" />)

    fireEvent.change(screen.getByLabelText("Server directory path"), {
      target: { value: "/srv/tldw/notes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith({
        source_type: "local_directory",
        sink_type: "notes",
        policy: "canonical",
        enabled: true,
        schedule_enabled: false,
        schedule: {},
        config: {
          path: "/srv/tldw/notes"
        }
      })
    })
  })

  it("renders a Scheduled rescans switch", () => {
    render(<SourceForm mode="create" />)

    expect(screen.getByText("Scheduled rescans")).toBeInTheDocument()
    expect(
      screen.getByText(/cadence is managed by the server/i)
    ).toBeInTheDocument()
  })

  it("sends schedule_enabled true when Scheduled rescans is enabled", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" />)

    fireEvent.click(screen.getByRole("switch", { name: "Scheduled rescans" }))
    fireEvent.change(screen.getByLabelText("Server directory path"), {
      target: { value: "/srv/tldw/notes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          schedule_enabled: true
        })
      )
    })
  })

  it("disables local-directory create when the server capability is unavailable", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })
    capabilityMocks.useServerCapabilities.mockReturnValue({
      capabilities: { canCreateLocalDirectoryIngestionSource: false },
      loading: false,
      refresh: vi.fn()
    })

    render(<SourceForm mode="create" />)

    const createButton = screen.getByRole("button", { name: "Create source" })
    expect(createButton).toBeDisabled()
    expect(
      screen.getByText(
        "The administrator must enable server folder sync before you can create a local directory source."
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "Archive snapshot" }))

    await waitFor(() => {
      expect(createButton).not.toBeDisabled()
    })
    fireEvent.click(createButton)

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          source_type: "archive_snapshot"
        })
      )
    })
  })

  it("switches fields between local directory and archive source modes", async () => {
    render(<SourceForm mode="create" />)

    expect(screen.getByLabelText("Server directory path")).toBeInTheDocument()
    expect(
      screen.getByText(/path on the tldw server host, not a local browser or extension folder/i)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "Archive snapshot" }))

    await waitFor(() => {
      expect(screen.queryByLabelText("Server directory path")).not.toBeInTheDocument()
    })
    expect(screen.getByText("Upload archive after creation")).toBeInTheDocument()
  })

  it("switches git repository fields between local and remote modes", async () => {
    render(<SourceForm mode="create" />)

    fireEvent.click(screen.getByRole("radio", { name: "Git repository" }))

    expect(screen.getByLabelText("Repository path")).toBeInTheDocument()
    expect(screen.getByLabelText("Branch, tag, or ref")).toBeInTheDocument()
    expect(screen.getByLabelText("Root subpath")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "Remote GitHub repository" }))

    await waitFor(() => {
      expect(screen.queryByLabelText("Repository path")).not.toBeInTheDocument()
    })
    expect(screen.getByLabelText("GitHub repository URL")).toBeInTheDocument()
    expect(screen.getByLabelText("Linked account ID")).toBeInTheDocument()
    expect(screen.queryByText("Media")).not.toBeInTheDocument()
  })

  it("renders inline validation errors from a failed create request", async () => {
    const mutateAsync = vi.fn(async () => {
      throw new Error("Path outside allowed roots")
    })
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" />)

    fireEvent.change(screen.getByLabelText("Server directory path"), {
      target: { value: "/tmp/imports" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalled()
    })
    expect(await screen.findByText("Path outside allowed roots")).toBeInTheDocument()
  })

  it("navigates to the source detail route after a successful create", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" />)

    fireEvent.change(screen.getByLabelText("Server directory path"), {
      target: { value: "/srv/tldw/notes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(routerMocks.navigate).toHaveBeenCalledWith("/sources/42")
    })
  })

  it("submits git repository config when creating a source", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" />)

    fireEvent.click(screen.getByRole("radio", { name: "Git repository" }))
    fireEvent.change(screen.getByLabelText("Repository path"), {
      target: { value: "/srv/repos/notes" }
    })
    fireEvent.change(screen.getByLabelText("Branch, tag, or ref"), {
      target: { value: "main" }
    })
    fireEvent.change(screen.getByLabelText("Root subpath"), {
      target: { value: "docs/notes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith({
        source_type: "git_repository",
        sink_type: "notes",
        policy: "canonical",
        enabled: true,
        schedule_enabled: false,
        schedule: {},
        config: {
          mode: "local_repo",
          path: "/srv/repos/notes",
          ref: "main",
          root_subpath: "docs/notes",
          respect_gitignore: true
        }
      })
    })
  })

  it("blocks invalid linked account ids for remote git repositories", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useCreateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(<SourceForm mode="create" />)

    fireEvent.click(screen.getByRole("radio", { name: "Git repository" }))
    fireEvent.click(screen.getByRole("radio", { name: "Remote GitHub repository" }))
    fireEvent.change(screen.getByLabelText("GitHub repository URL"), {
      target: { value: "https://github.com/example/repo" }
    })
    fireEvent.change(screen.getByLabelText("Linked account ID"), {
      target: { value: "abc" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    await waitFor(() => {
      expect(mutateAsync).not.toHaveBeenCalled()
    })
    expect(await screen.findByText("Linked account ID must be a positive integer")).toBeInTheDocument()
  })

  it("renders immutable source identity fields as summaries after first successful sync", () => {
    render(
      <SourceForm
        mode="edit"
        source={{
          id: "42",
          user_id: 7,
          source_type: "local_directory",
          sink_type: "notes",
          policy: "canonical",
          enabled: true,
          schedule_enabled: false,
          schedule_config: {},
          config: { path: "/srv/tldw/notes" },
          last_successful_snapshot_id: "9",
          last_successful_sync_summary: {
            changed_count: 2,
            degraded_count: 0,
            conflict_count: 0,
            sink_failure_count: 0,
            ingestion_failure_count: 0,
            created_count: 1,
            updated_count: 1,
            deleted_count: 0,
            unchanged_count: 3
          }
        }}
      />
    )

    expect(
      screen.getByText("Locked after first successful sync")
    ).toBeInTheDocument()
    expect(screen.getByText("/srv/tldw/notes")).toBeInTheDocument()
    expect(screen.queryByRole("radio", { name: "Archive snapshot" })).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Server directory path")).not.toBeInTheDocument()
    expect(screen.queryByText("Destination")).not.toBeInTheDocument()
  })

  it("omits immutable git repository config when editing a locked source", async () => {
    const mutateAsync = vi.fn(async () => ({ id: "42" }))
    hookMocks.useUpdateIngestionSourceMutation.mockReturnValue({
      mutateAsync,
      isPending: false
    })

    render(
      <SourceForm
        mode="edit"
        source={{
          id: "42",
          user_id: 7,
          source_type: "git_repository",
          sink_type: "notes",
          policy: "canonical",
          enabled: true,
          schedule_enabled: false,
          schedule_config: {},
          config: {
            mode: "remote_github_repo",
            repo_url: "https://github.com/example/repo",
            account_id: 12,
            ref: "main",
            root_subpath: "docs"
          },
          last_successful_snapshot_id: "9",
          last_successful_sync_summary: {
            changed_count: 2,
            degraded_count: 0,
            conflict_count: 0,
            sink_failure_count: 0,
            ingestion_failure_count: 0,
            created_count: 1,
            updated_count: 1,
            deleted_count: 0,
            unchanged_count: 3
          }
        }}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith({
        source_type: "git_repository",
        sink_type: "notes",
        policy: "canonical",
        enabled: true,
        schedule_enabled: false,
        schedule: {}
      })
    })
  })

  it("keeps source identity editable until a successful snapshot exists", () => {
    render(
      <SourceForm
        mode="edit"
        source={{
          id: "42",
          user_id: 7,
          source_type: "local_directory",
          sink_type: "notes",
          policy: "canonical",
          enabled: true,
          schedule_enabled: false,
          schedule_config: {},
          config: { path: "/srv/tldw/notes" },
          last_successful_snapshot_id: null,
          last_sync_completed_at: "2026-03-08T22:15:00Z",
          last_successful_sync_summary: {
            changed_count: 2,
            degraded_count: 0,
            conflict_count: 0,
            sink_failure_count: 0,
            ingestion_failure_count: 0,
            created_count: 1,
            updated_count: 1,
            deleted_count: 0,
            unchanged_count: 3
          }
        }}
      />
    )

    expect(screen.queryByText("Locked after first successful sync")).not.toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Archive snapshot" })).toBeInTheDocument()
    expect(screen.getByLabelText("Server directory path")).toBeInTheDocument()
    expect(screen.getByText("Destination")).toBeInTheDocument()
  })
})
