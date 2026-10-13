import React from "react"
import {
  act,
  fireEvent,
  render as rtlRender,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider } from "antd"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSourceSavedViewResponse } from "@/services/tldw/domains/workspace-api"
import {
  SourceViewControls,
  SourceViewOverlayHost,
  type SourceViewOverlayRequest
} from "../SourcesPane/SourceViewControls"
import { DEFAULT_SOURCE_LIST_VIEW_STATE } from "../SourcesPane/source-list-view"
import type { SourceSavedViewsController } from "../SourcesPane/use-source-saved-views"

// rc-component uses the same "test-id" for tooltip and dialog in NODE_ENV=test.
// Keep the real tooltip and give it the React ID used outside that test branch.
vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  const React = await import("react")
  return {
    ...actual,
    Tooltip: React.forwardRef<
      React.ComponentRef<typeof actual.Tooltip>,
      React.ComponentProps<typeof actual.Tooltip>
    >(function Tooltip(props, ref) {
      const id = React.useId()
      return <actual.Tooltip {...props} id={props.id ?? id} ref={ref} />
    })
  }
})

// Keep real Ant Design dialogs and menus; jsdom never completes CSS leave motion.
const render = (ui: React.ReactElement) =>
  rtlRender(ui, {
    wrapper: ({ children }) => (
      <ConfigProvider theme={{ token: { motion: false } }}>
        {children}
      </ConfigProvider>
    )
  })

const validView = (
  overrides: Partial<WorkspaceSourceSavedViewResponse> = {}
): WorkspaceSourceSavedViewResponse => ({
  id: "view-1",
  workspace_id: "workspace-a",
  name: "My PDFs",
  schema_version: 1,
  version: 2,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
  valid: true,
  invalid_reason: null,
  state: {
    type_filters: ["pdf"],
    status_filters: [],
    review_state_filters: [],
    lifecycle_state_filters: [],
    date_field: "added_at",
    date_from: null,
    date_to: null,
    require_url: false,
    require_file_size: false,
    require_duration: false,
    require_page_count: false,
    file_size_min: null,
    file_size_max: null,
    duration_min: null,
    duration_max: null,
    page_count_min: null,
    page_count_max: null,
    sort: "name_asc"
  },
  ...overrides
} as WorkspaceSourceSavedViewResponse)

const invalidView = (): WorkspaceSourceSavedViewResponse => ({
  ...validView(),
  id: "view-invalid",
  name: "Old view",
  valid: false,
  state: null,
  invalid_reason: "unsupported_schema_version"
})

const controller = (
  overrides: Partial<SourceSavedViewsController> = {}
): SourceSavedViewsController =>
  ({
    available: true,
    generation: 3,
    views: [validView(), invalidView()],
    loading: false,
    listError: null,
    activeViewId: null,
    activeSnapshot: null,
    currentSignature: "signature",
    modified: false,
    serializationIssues: [],
    duplicateConflict: null,
    limitState: null,
    versionConflict: null,
    mutation: null,
    busy: false,
    mutationError: null,
    announcement: null,
    canRetryMutation: false,
    canRetryVersion: false,
    load: vi.fn(),
    retry: vi.fn(),
    retryMutation: vi.fn(),
    retryVersionConflict: vi.fn(),
    applyView: vi.fn(),
    createView: vi.fn(),
    confirmReplace: vi.fn(),
    dismissDuplicateConflict: vi.fn(),
    dismissMutationFailure: vi.fn(),
    replaceView: vi.fn(),
    resetView: vi.fn(),
    deleteView: vi.fn(),
    ...overrides
  }) as SourceSavedViewsController

const Harness = ({
  model = controller(),
  showControls = true,
  sourceListViewState = DEFAULT_SOURCE_LIST_VIEW_STATE
}: {
  model?: SourceSavedViewsController
  showControls?: boolean
  sourceListViewState?: typeof DEFAULT_SOURCE_LIST_VIEW_STATE
}) => {
  const [request, setRequest] = React.useState<SourceViewOverlayRequest | null>(null)
  return (
    <div>
      {showControls && (
        <SourceViewControls
          controller={model}
          sourceListViewState={sourceListViewState}
          onApplySourceListViewState={vi.fn()}
          onOpenOverlay={setRequest}
        />
      )}
      <div role="complementary" aria-label="Sources" tabIndex={-1} />
      <SourceViewOverlayHost
        controller={model}
        request={request}
        onRequestHandled={() => setRequest(null)}
      />
    </div>
  )
}

const openSavedViewCommands = async (
  user: ReturnType<typeof userEvent.setup>,
  viewName: string
) => {
  const submenu = await screen.findByRole("menuitem", {
    name: new RegExp(`^${viewName}`)
  })
  await user.hover(submenu)
  await screen.findByRole("menuitem", {
    name: new RegExp(`^(Apply|Reset) saved view ${viewName}$`)
  })
  return submenu
}

const activateMenuItemByKeyboard = (item: HTMLElement) => {
  item.focus()
  expect(item).toHaveFocus()
  fireEvent.keyDown(item, {
    key: "Enter",
    code: "Enter",
    keyCode: 13,
    which: 13
  })
  fireEvent.keyUp(item, {
    key: "Enter",
    code: "Enter",
    keyCode: 13,
    which: 13
  })
}

describe("SourceViewControls", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("reserves hidden validation text and exposes it only for an invalid touched name", async () => {
    const model = controller()
    render(<Harness model={model} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    await waitFor(() => expect(dialog).toBeVisible())
    const input = within(dialog).getByRole("textbox", { name: "View name" })
    const save = within(dialog).getByRole("button", { name: "Save" })
    const feedback = within(dialog).getByText(/Name must contain between 1 and 120/)
    expect(feedback).not.toBeVisible()
    expect(within(dialog).queryByRole("alert")).toBeNull()
    expect(input).not.toHaveAttribute("aria-describedby")
    expect(input).toHaveAttribute("aria-invalid", "false")
    expect(save).toBeDisabled()

    fireEvent.blur(input)
    expect(within(dialog).getByRole("alert")).toBe(feedback)
    expect(feedback).toBeVisible()
    expect(input).toHaveAccessibleDescription(feedback.textContent ?? "")
    for (const value of ["   ", "x".repeat(121)]) {
      fireEvent.change(input, { target: { value } })
      expect(input).toHaveAttribute("aria-invalid", "true")
      expect(save).toBeDisabled()
      expect(within(dialog).getByRole("alert")).toBe(feedback)
    }
    fireEvent.change(input, { target: { value: "x".repeat(120) } })
    expect(feedback).not.toBeVisible()
    expect(within(dialog).queryByRole("alert")).toBeNull()
    expect(input).toHaveAttribute("aria-invalid", "false")
    expect(input).not.toHaveAttribute("aria-describedby")
    expect(save).toBeEnabled()
    expect(model.createView).not.toHaveBeenCalled()
  })

  it("renders fixed-order grouped built-ins and saved views and applies them by keyboard", async () => {
    const user = userEvent.setup()
    const model = controller()
    const applyState = vi.fn()
    render(
      <>
        <SourceViewControls
          controller={model}
          sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
          onApplySourceListViewState={applyState}
          onOpenOverlay={vi.fn()}
        />
        <button>Outside destination</button>
      </>
    )

    const trigger = screen.getByRole("button", { name: "Source views" })
    trigger.focus()
    await user.keyboard("{Enter}")

    const menu = await screen.findByRole("menu")
    const labels = within(menu)
      .getAllByRole("menuitem", { hidden: false })
      .map((item) => item.textContent?.replace(/\s+/g, " ").trim())
    expect(labels).toEqual([
      "Needs review",
      "Unreviewed",
      "Failed ingest",
      "Partially indexed",
      "PDFs",
      "Web captures",
      "Large files",
      "My PDFs",
      "Old viewInvalidUnsupported schema version. Apply unavailable."
    ])
    expect(within(menu).getByText("Built-in views")).toBeInTheDocument()
    expect(within(menu).getByText("Saved views")).toBeInTheDocument()

    const needsReview = within(menu).getByRole("menuitem", {
      name: "Needs review"
    })
    activateMenuItemByKeyboard(needsReview)
    expect(applyState).toHaveBeenCalledTimes(1)
    expect(trigger).toHaveFocus()

    trigger.focus()
    await user.keyboard(" ")
    expect(await screen.findByRole("menu")).toBeInTheDocument()
    const pdfs = screen.getByRole("menuitem", { name: "PDFs" })
    activateMenuItemByKeyboard(pdfs)
    expect(applyState).toHaveBeenCalledTimes(2)
    expect(trigger).toHaveFocus()
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument())

    // Escape must close an open menu, not merely observe selection's prior close.
    trigger.focus()
    await user.keyboard("{Enter}")
    const openMenu = await screen.findByRole("menu")
    const focusedItem = within(openMenu).getByRole("menuitem", { name: "PDFs" })
    focusedItem.focus()
    fireEvent.keyDown(focusedItem, {
      key: "Escape", code: "Escape", keyCode: 27, which: 27
    })
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument())
    await waitFor(() => expect(trigger).toHaveFocus())

    await user.click(trigger)
    expect(await screen.findByRole("menu")).toBeInTheDocument()
    const outside = screen.getByRole("button", { name: "Outside destination" })
    await user.click(outside)
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument())
    expect(outside).toHaveFocus()
  })

  it("keeps built-ins enabled while disabling save for a null workspace", async () => {
    const model = controller({ available: false, views: [] })
    const open = vi.fn()
    const applyState = vi.fn()
    render(
      <SourceViewControls
        controller={model}
        sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
        onApplySourceListViewState={applyState}
        onOpenOverlay={open}
      />
    )

    const save = screen.getByRole("button", { name: "Save source view" })
    expect(save).toBeDisabled()
    expect(save).toHaveAccessibleDescription("Select a workspace")
    fireEvent.click(save)
    expect(open).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Source views" }))
    fireEvent.click(await screen.findByRole("menuitem", { name: "PDFs" }))
    expect(applyState).toHaveBeenCalledWith(
      expect.objectContaining({ typeFilters: ["pdf"] })
    )
  })

  it("shows Modified, retryable errors, invalid recovery actions, and one polite status", async () => {
    const model = controller({
      activeViewId: "view-1",
      modified: true,
      listError: { message: "Offline", retryable: true },
      announcement: "Saved view reset."
    })
    render(<Harness model={model} />)

    expect(screen.getByText("Modified")).toBeInTheDocument()
    expect(screen.getByRole("alert")).toHaveTextContent("Offline")
    fireEvent.click(screen.getByRole("button", { name: "Retry saved views" }))
    expect(model.retry).toHaveBeenCalled()
    expect(screen.getAllByRole("status")).toHaveLength(1)

    fireEvent.click(screen.getByRole("button", { name: "Source views" }))
    const invalid = await screen.findByRole("menuitem", { name: /Old view/ })
    expect(invalid).not.toHaveAttribute("aria-disabled", "true")
    await openSavedViewCommands(userEvent.setup(), "Old view")
    expect(
      screen.getByRole("menuitem", { name: "Reset saved view Old view" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: "Delete saved view Old view" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("menuitem", { name: "Apply saved view Old view" })
    ).not.toBeInTheDocument()
  })

  it("disables list Retry while a saved-view mutation is in flight", () => {
    const model = controller({
      busy: true,
      mutation: "replace",
      listError: { message: "Offline", retryable: true }
    })
    render(<Harness model={model} />)

    const retry = screen.getByRole("button", { name: "Retry saved views" })
    expect(retry).toBeDisabled()
    fireEvent.click(retry)
    expect(model.retry).not.toHaveBeenCalled()
  })

  it("routes Enter and Space on row actions without applying the saved view", async () => {
    const user = userEvent.setup()
    const model = controller()
    render(<Harness model={model} />)

    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    const replace = await screen.findByRole("menuitem", {
      name: "Replace saved view My PDFs"
    })
    activateMenuItemByKeyboard(replace)
    expect(
      await screen.findByRole("dialog", { name: "Replace saved view?" })
    ).toBeInTheDocument()
    expect(screen.getAllByRole("dialog", { name: "Replace saved view?" })).toHaveLength(1)
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument()
    expect(model.applyView).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    const deleteAction = await screen.findByRole("menuitem", {
      name: "Delete saved view My PDFs"
    })
    deleteAction.focus()
    expect(deleteAction).toHaveFocus()
    await user.keyboard(" ")
    expect(
      await screen.findByRole("dialog", { name: "Delete saved view?" })
    ).toBeInTheDocument()
    expect(model.applyView).not.toHaveBeenCalled()
  })

  it("keeps invalid rows keyboard navigable for Reset/Delete and exposes the reason", async () => {
    const user = userEvent.setup()
    const model = controller()
    render(<Harness model={model} />)

    await user.click(screen.getByRole("button", { name: "Source views" }))
    const invalid = await screen.findByRole("menuitem", { name: /Old view/ })
    expect(invalid).not.toHaveAttribute("aria-disabled", "true")
    expect(within(invalid).getByText(/Unsupported schema version/i)).toBeInTheDocument()

    await openSavedViewCommands(user, "Old view")
    const reset = screen.getByRole("menuitem", { name: "Reset saved view Old view" })
    activateMenuItemByKeyboard(reset)
    expect(
      await screen.findByRole("dialog", { name: "Reset saved view?" })
    ).toBeInTheDocument()
    expect(model.applyView).not.toHaveBeenCalled()
  })

  it("validates save fields, confirms replacement, exposes busy state, and returns focus", async () => {
    const createView = vi.fn().mockResolvedValue(undefined)
    const confirmReplace = vi.fn().mockResolvedValue(undefined)
    const initial = controller({ createView })
    const { rerender } = render(<Harness model={initial} />)

    const invoker = screen.getByRole("button", { name: "Save source view" })
    fireEvent.click(invoker)
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    const save = within(dialog).getByRole("button", { name: "Save" })
    expect(save).toBeDisabled()
    fireEvent.change(within(dialog).getByRole("textbox", { name: "View name" }), {
      target: { value: "   " }
    })
    expect(within(dialog).getByText(/between 1 and 120/)).toBeInTheDocument()

    fireEvent.change(within(dialog).getByRole("textbox", { name: "View name" }), {
      target: { value: "My PDFs" }
    })
    fireEvent.click(save)
    await waitFor(() => expect(createView).toHaveBeenCalledWith("My PDFs"))
    expect(dialog).toBeInTheDocument() // A resolved mutation is not server confirmation.

    rerender(
      <Harness
        model={controller({
          createView,
          activeViewId: "view-1",
          announcement: "Saved view created."
        })}
      />
    )
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    await waitFor(() => expect(document.activeElement).toBe(invoker))

    fireEvent.click(invoker)
    expect(
      await screen.findByRole("dialog", { name: "Save source view" })
    ).toBeInTheDocument()

    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    rerender(<Harness model={controller({ duplicateConflict: duplicate, confirmReplace })} />)
    const replacement = await screen.findByRole("dialog", {
      name: "Replace saved view?"
    })
    fireEvent.click(within(replacement).getByRole("button", { name: "Replace" }))
    await waitFor(() => expect(confirmReplace).toHaveBeenCalled())

    rerender(<Harness model={controller({ busy: true, mutation: "create" })} />)
    expect(screen.getByRole("button", { name: "Save source view" })).toHaveAttribute(
      "aria-busy",
      "true"
    )
  })

  it("validates the full saved-view name by Unicode code point", async () => {
    const createView = vi.fn().mockResolvedValue(undefined)
    render(<Harness model={controller({ createView })} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    const input = within(dialog).getByRole("textbox", { name: "View name" })
    const save = within(dialog).getByRole("button", { name: "Save" })
    const astralCharacter = "\u{1F4C4}"
    const maximumName = astralCharacter.repeat(120)

    expect(input).not.toHaveAttribute("maxlength")
    fireEvent.change(input, { target: { value: maximumName } })
    expect(input).toHaveValue(maximumName)
    expect(save).toBeEnabled()
    fireEvent.click(save)
    await waitFor(() => expect(createView).toHaveBeenCalledWith(maximumName))

    fireEvent.change(input, {
      target: { value: astralCharacter.repeat(121) }
    })
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "Name must contain between 1 and 120 characters."
    )
    expect(input).toHaveAccessibleDescription(
      "Name must contain between 1 and 120 characters."
    )
    expect(save).toBeDisabled()
  })

  it("keeps direct Replace version recovery inside its confirmation dialog", async () => {
    const user = userEvent.setup()
    const replaceView = vi.fn().mockResolvedValue(undefined)
    const retryVersionConflict = vi.fn().mockResolvedValue(undefined)
    const { rerender } = render(
      <Harness model={controller({ replaceView, retryVersionConflict })} />
    )
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    activateMenuItemByKeyboard(
      screen.getByRole("menuitem", { name: "Replace saved view My PDFs" })
    )
    let dialog = await screen.findByRole("dialog", { name: "Replace saved view?" })
    const replace = within(dialog).getByRole("button", { name: "Replace" })
    replace.focus()
    await user.keyboard("{Enter}")
    expect(replaceView).toHaveBeenCalledWith(validView())

    rerender(
      <Harness
        model={controller({
          replaceView,
          retryVersionConflict,
          versionConflict: {
            viewId: "view-1",
            currentVersion: 5,
            retryable: true
          },
          canRetryVersion: true
        })}
      />
    )
    dialog = await screen.findByRole("dialog", { name: "Replace saved view?" })
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "This saved view changed on the server."
    )
    const retry = within(dialog).getByRole("button", {
      name: "Retry with latest version"
    })
    expect(screen.getAllByRole("button", { name: "Retry with latest version" })).toHaveLength(1)
    retry.focus()
    expect(dialog).toContainElement(document.activeElement)
    await user.keyboard("{Enter}")
    expect(retryVersionConflict).toHaveBeenCalledTimes(1)
    expect(replaceView).toHaveBeenCalledTimes(1)
  })

  it("keeps invalid Reset version recovery inside its confirmation dialog", async () => {
    const user = userEvent.setup()
    const resetView = vi.fn().mockResolvedValue(undefined)
    const retryVersionConflict = vi.fn().mockResolvedValue(undefined)
    const { rerender } = render(
      <Harness model={controller({ resetView, retryVersionConflict })} />
    )
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "Old view")
    activateMenuItemByKeyboard(
      screen.getByRole("menuitem", { name: "Reset saved view Old view" })
    )
    let dialog = await screen.findByRole("dialog", { name: "Reset saved view?" })
    const reset = within(dialog).getByRole("button", { name: "Reset" })
    reset.focus()
    await user.keyboard("{Enter}")
    expect(resetView).toHaveBeenCalledWith(invalidView())

    rerender(
      <Harness
        model={controller({
          resetView,
          retryVersionConflict,
          versionConflict: {
            viewId: "view-invalid",
            currentVersion: 6,
            retryable: true
          },
          canRetryVersion: true
        })}
      />
    )
    dialog = await screen.findByRole("dialog", { name: "Reset saved view?" })
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "This saved view changed on the server."
    )
    const retry = within(dialog).getByRole("button", {
      name: "Retry with latest version"
    })
    retry.focus()
    await user.keyboard("{Enter}")
    expect(retryVersionConflict).toHaveBeenCalledTimes(1)
    expect(resetView).toHaveBeenCalledTimes(1)
  })

  it("keeps duplicate replacement version recovery inside the Save dialog", async () => {
    const user = userEvent.setup()
    const createView = vi.fn().mockResolvedValue(undefined)
    const confirmReplace = vi.fn().mockResolvedValue(undefined)
    const retryVersionConflict = vi.fn().mockResolvedValue(undefined)
    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    const { rerender } = render(
      <Harness
        model={controller({ createView, confirmReplace, retryVersionConflict })}
      />
    )
    const saveTrigger = screen.getByRole("button", { name: "Save source view" })
    saveTrigger.focus()
    await user.keyboard("{Enter}")
    let dialog = await screen.findByRole("dialog", { name: "Save source view" })
    await user.type(within(dialog).getByRole("textbox", { name: "View name" }), "My PDFs")
    const save = within(dialog).getByRole("button", { name: "Save" })
    save.focus()
    await user.keyboard("{Enter}")
    expect(createView).toHaveBeenCalledWith("My PDFs")

    rerender(
      <Harness
        model={controller({
          createView,
          confirmReplace,
          retryVersionConflict,
          duplicateConflict: duplicate
        })}
      />
    )
    dialog = await screen.findByRole("dialog", { name: "Replace saved view?" })
    const replace = within(dialog).getByRole("button", { name: "Replace" })
    replace.focus()
    await user.keyboard("{Enter}")
    expect(confirmReplace).toHaveBeenCalledTimes(1)

    rerender(
      <Harness
        model={controller({
          createView,
          confirmReplace,
          retryVersionConflict,
          duplicateConflict: { ...duplicate, version: 5 },
          versionConflict: {
            viewId: "view-1",
            currentVersion: 5,
            retryable: true
          },
          canRetryVersion: true
        })}
      />
    )
    dialog = await screen.findByRole("dialog", { name: "Replace saved view?" })
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "This saved view changed on the server."
    )
    const retry = within(dialog).getByRole("button", {
      name: "Retry with latest version"
    })
    retry.focus()
    await user.keyboard("{Enter}")
    expect(retryVersionConflict).toHaveBeenCalledTimes(1)
    expect(confirmReplace).toHaveBeenCalledTimes(1)
  })

  it.each([
    {
      label: "Replace",
      viewName: "My PDFs",
      actionName: "Replace saved view My PDFs",
      dialogName: "Replace saved view?",
      method: "replaceView" as const
    },
    {
      label: "Reset",
      viewName: "Old view",
      actionName: "Reset saved view Old view",
      dialogName: "Reset saved view?",
      method: "resetView" as const
    }
  ])("closes a $label dialog after PATCH reports the row missing", async (testCase) => {
    const user = userEvent.setup()
    const mutation = vi.fn().mockResolvedValue(undefined)
    const initial = controller({ [testCase.method]: mutation })
    const { rerender } = render(<Harness model={initial} />)
    const focusFallback = screen.getByRole("button", { name: "Source views" })
    await user.click(focusFallback)
    await openSavedViewCommands(user, testCase.viewName)
    await user.click(screen.getByRole("menuitem", { name: testCase.actionName }))
    const dialog = await screen.findByRole("dialog", {
      name: testCase.dialogName
    })
    await user.click(within(dialog).getByRole("button", { name: testCase.label }))
    expect(mutation).toHaveBeenCalledTimes(1)

    rerender(
      <Harness
        model={controller({
          [testCase.method]: mutation,
          announcement: "Saved view no longer exists."
        })}
      />
    )

    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: testCase.dialogName })).not.toBeInTheDocument()
    )
    await waitFor(() => expect(focusFallback).toHaveFocus())
  })

  it.each([
    {
      label: "Replace",
      viewName: "My PDFs",
      actionName: "Replace saved view My PDFs",
      announcement: "Saved view replaced.",
      method: "replaceView" as const
    },
    {
      label: "Replace missing row",
      viewName: "My PDFs",
      actionName: "Replace saved view My PDFs",
      announcement: "Saved view no longer exists.",
      method: "replaceView" as const
    },
    {
      label: "Reset",
      viewName: "Old view",
      actionName: "Reset saved view Old view",
      announcement: "Saved view reset.",
      method: "resetView" as const
    }
  ])(
    "restores $label fallback only after native closing motion completes",
    async (testCase) => {
      const user = userEvent.setup()
      const mutation = vi.fn().mockResolvedValue(undefined)
      // Unlike compatibility tests above, keep the real Modal/CSSMotion enabled.
      const nativeHarness = (
        model: SourceSavedViewsController,
        showControls = true
      ) => (
        <>
          <Harness model={model} showControls={showControls} />
          {testCase.label === "Replace missing row" && (
            <SourceViewControls
              controller={model}
              sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
              onApplySourceListViewState={vi.fn()}
              onOpenOverlay={vi.fn()}
            />
          )}
        </>
      )
      const { rerender } = rtlRender(
        nativeHarness(controller({ [testCase.method]: mutation }))
      )
      await user.click(
        screen.getAllByRole("button", { name: "Source views" })[0]!
      )
      await openSavedViewCommands(user, testCase.viewName)
      const invoker = screen.getByRole("menuitem", { name: testCase.actionName })
      await user.click(invoker)
      const dialog = await screen.findByRole("dialog")
      await waitFor(() =>
        expect(dialog.className).toContain("ant-zoom-appear-active")
      )
      fireEvent(dialog, new Event("webkitAnimationEnd", { bubbles: true }))
      fireEvent.animationEnd(dialog)
      await waitFor(() =>
        expect(dialog.className).not.toContain("ant-zoom-appear")
      )
      const submit = within(dialog).getByRole("button", {
        name: testCase.method === "replaceView" ? "Replace" : "Reset"
      })
      await user.click(submit)
      expect(mutation).toHaveBeenCalledTimes(1)
      expect(dialog).toBeInTheDocument() // Promise resolution is not server acknowledgment.
      rerender(
        nativeHarness(
          controller({
            busy: true,
            mutation: "replace",
            [testCase.method]: mutation
          })
        )
      )
      fireEvent.keyDown(dialog, {
        key: "Escape",
        code: "Escape",
        keyCode: 27,
        which: 27
      })
      expect(dialog).toBeInTheDocument()
      // Remove the actual invoking pane/menu, retaining a current visible fallback.
      rerender(
        nativeHarness(
          controller({
            [testCase.method]: mutation,
            announcement: testCase.announcement
          }),
          false
        )
      )
      expect(invoker.isConnected).toBe(false)
      const fallback =
        testCase.label === "Replace missing row"
          ? screen.getByRole("button", { name: "Source views" })
          : screen.getByRole("complementary", { name: "Sources" })
      await waitFor(() =>
        expect(dialog.className).toContain("ant-zoom-leave-active")
      )
      expect(fallback).not.toHaveFocus()
      fireEvent(dialog, new Event("webkitAnimationEnd", { bubbles: true }))
      fireEvent.animationEnd(dialog)
      await waitFor(() => expect(dialog).not.toBeInTheDocument())
      await waitFor(() => expect(fallback).toHaveFocus())
    }
  )

  it.each(["outside destination", "later dialog", "later generation"])(
    "respects %s focus authority during native closing motion",
    async (authority) => {
      const initial = controller()
      const { rerender } = rtlRender(
        <>
          <Harness model={initial} />
          <button>Outside destination</button>
        </>
      )
      const invoker = screen.getByRole("button", { name: "Save source view" })
      fireEvent.click(invoker)
      const dialog = await screen.findByRole("dialog")
      await waitFor(() =>
        expect(dialog.className).toContain("ant-zoom-appear-active")
      )
      fireEvent(dialog, new Event("webkitAnimationEnd", { bubbles: true }))
      fireEvent.animationEnd(dialog)
      await waitFor(() =>
        expect(dialog.className).not.toContain("ant-zoom-appear")
      )
      fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
      await waitFor(() =>
        expect(dialog.className).toContain("ant-zoom-leave-active")
      )
      const outside = screen.getByRole("button", { name: "Outside destination" })
      if (authority === "later generation") {
        rerender(
          <>
            <Harness model={controller({ generation: 4 })} />
            <button>Outside destination</button>
          </>
        )
      }
      if (authority === "later dialog") {
        fireEvent.click(invoker)
        const input = await screen.findByRole("textbox", { name: "View name" })
        input.focus()
        await waitFor(() =>
          expect(dialog.className).toContain("ant-zoom-enter-active")
        )
        fireEvent(dialog, new Event("webkitAnimationEnd", { bubbles: true }))
        fireEvent.animationEnd(dialog)
        await waitFor(() => expect(dialog.className).not.toContain("ant-zoom"))
        expect(input).toHaveFocus()
        expect(dialog).toBeInTheDocument()
      } else {
        if (authority === "outside destination") outside.focus()
        fireEvent(dialog, new Event("webkitAnimationEnd", { bubbles: true }))
        fireEvent.animationEnd(dialog)
        await waitFor(() => expect(dialog).not.toBeInTheDocument())
        expect(invoker).not.toHaveFocus()
        if (authority === "outside destination") expect(outside).toHaveFocus()
      }
    }
  )

  it("closes a duplicate replacement dialog when PATCH reports the row missing", async () => {
    const user = userEvent.setup()
    const createView = vi.fn().mockResolvedValue(undefined)
    const confirmReplace = vi.fn().mockResolvedValue(undefined)
    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    const { rerender } = render(
      <Harness model={controller({ createView, confirmReplace })} />
    )
    const focusTarget = screen.getByRole("button", { name: "Save source view" })
    await user.click(focusTarget)
    let dialog = await screen.findByRole("dialog", { name: "Save source view" })
    await user.type(within(dialog).getByRole("textbox", { name: "View name" }), "My PDFs")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    rerender(
      <Harness
        model={controller({
          createView,
          confirmReplace,
          duplicateConflict: duplicate
        })}
      />
    )
    dialog = await screen.findByRole("dialog", { name: "Replace saved view?" })
    await user.click(within(dialog).getByRole("button", { name: "Replace" }))
    expect(confirmReplace).toHaveBeenCalledTimes(1)

    rerender(
      <Harness
        model={controller({
          createView,
          confirmReplace,
          announcement: "Saved view no longer exists."
        })}
      />
    )

    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Replace saved view?" })).not.toBeInTheDocument()
    )
    await waitFor(() => expect(focusTarget).toHaveFocus())
  })

  it("renders field-specific local validation and nonretryable limit guidance", async () => {
    const model = controller({
      serializationIssues: [
        { field: "fileSizeMax", message: "Must be greater than fileSizeMin." }
      ],
      limitState: {
        limit: 100,
        retryable: false,
        guidance: "Delete an existing saved view before creating another."
      }
    })
    render(<Harness model={model} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))

    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    expect(within(dialog).getByText(/File size max/i)).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeDisabled()
    const limitAlert = screen.getByText(/Saved view limit of 100/).closest("[role='alert']")
    expect(limitAlert).toHaveTextContent("100")
    expect(limitAlert).toHaveTextContent(/Delete an existing saved view/)
    expect(screen.queryByRole("button", { name: /Retry/ })).not.toBeInTheDocument()
  })

  it("visibly blocks replacement when the current local state is invalid", async () => {
    const user = userEvent.setup()
    const model = controller()
    render(
      <Harness
        model={model}
        sourceListViewState={{
          ...DEFAULT_SOURCE_LIST_VIEW_STATE,
          fileSizeMin: 20,
          fileSizeMax: 10
        }}
      />
    )

    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    await user.click(
      screen.getByRole("menuitem", { name: "Replace saved view My PDFs" })
    )

    const dialog = await screen.findByRole("dialog", {
      name: "Replace saved view?"
    })
    expect(within(dialog).getByRole("alert")).toHaveTextContent(/File size max/i)
    expect(within(dialog).getByRole("button", { name: "Replace" })).toBeDisabled()
    expect(model.replaceView).not.toHaveBeenCalled()
  })

  it("unblocks a corrected same-workspace state without discarding a valid name", async () => {
    const user = userEvent.setup()
    const issues = [
      { field: "fileSizeMax", message: "Must be greater than fileSizeMin." }
    ]
    const invalidModel = controller({ serializationIssues: issues })
    const invalidState = {
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      fileSizeMin: 20,
      fileSizeMax: 10
    }
    const { rerender } = render(
      <Harness model={invalidModel} sourceListViewState={invalidState} />
    )

    await user.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    const name = within(dialog).getByRole("textbox", { name: "View name" })
    await user.type(name, "Corrected view")
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeDisabled()

    const correctedModel = controller({ serializationIssues: [] })
    rerender(
      <Harness
        model={correctedModel}
        sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
      />
    )

    expect(name).toHaveValue("Corrected view")
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeEnabled()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))

    await user.click(screen.getByRole("button", { name: "Save source view" }))
    const reopened = await screen.findByRole("dialog", { name: "Save source view" })
    const reopenedName = within(reopened).getByRole("textbox", {
      name: "View name"
    })
    expect(reopenedName).toHaveValue("")
    expect(within(reopened).getByRole("button", { name: "Save" })).toBeDisabled()
    await user.type(reopenedName, "Reopened view")
    expect(within(reopened).getByRole("button", { name: "Save" })).toBeEnabled()
    await user.click(within(reopened).getByRole("button", { name: "Cancel" }))

    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    await user.click(
      screen.getByRole("menuitem", { name: "Replace saved view My PDFs" })
    )
    expect(
      within(
        await screen.findByRole("dialog", { name: "Replace saved view?" })
      ).getByRole("button", { name: "Replace" })
    ).toBeEnabled()
  })

  it("dismisses duplicate replacement on Cancel so the next Save starts fresh", async () => {
    const user = userEvent.setup()
    const dismissDuplicateConflict = vi.fn()
    const dismissMutationFailure = vi.fn()
    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    const { rerender } = render(
      <Harness
        model={controller({ dismissDuplicateConflict, dismissMutationFailure })}
      />
    )
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    rerender(
      <Harness
        model={controller({
          duplicateConflict: duplicate,
          dismissDuplicateConflict,
          dismissMutationFailure
        })}
      />
    )
    expect(
      await screen.findByRole("dialog", { name: "Replace saved view?" })
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(dismissDuplicateConflict).toHaveBeenCalledTimes(1)
    expect(dismissMutationFailure).not.toHaveBeenCalled()

    rerender(<Harness model={controller({ dismissDuplicateConflict })} />)
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    expect(
      await screen.findByRole("dialog", { name: "Save source view" })
    ).toBeInTheDocument()
    expect(screen.queryByRole("dialog", { name: "Replace saved view?" })).not.toBeInTheDocument()
  })

  it("blocks every duplicate replacement dismissal path while busy", async () => {
    const user = userEvent.setup()
    const dismissDuplicateConflict = vi.fn()
    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    const { rerender } = render(
      <Harness model={controller({ dismissDuplicateConflict })} />
    )
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    rerender(
      <Harness
        model={controller({
          busy: true,
          mutation: "replace",
          duplicateConflict: duplicate,
          dismissDuplicateConflict
        })}
      />
    )

    const dialog = await screen.findByRole("dialog", {
      name: "Replace saved view?"
    })
    const cancel = within(dialog).getByRole("button", { name: "Cancel" })
    expect(cancel).toBeDisabled()
    fireEvent.click(cancel)
    expect(screen.getByRole("dialog", { name: "Replace saved view?" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Close" })).not.toBeInTheDocument()

    fireEvent.keyDown(dialog, {
      key: "Escape", code: "Escape", keyCode: 27, which: 27
    })
    expect(screen.getByRole("dialog", { name: "Replace saved view?" })).toBeInTheDocument()

    const wrapper = dialog.closest(".ant-modal-wrap")
    expect(wrapper).not.toBeNull()
    fireEvent.click(wrapper!)

    expect(screen.getByRole("dialog", { name: "Replace saved view?" })).toBeInTheDocument()
    expect(dismissDuplicateConflict).not.toHaveBeenCalled()
  })

  it("dismisses a failed saved-view mutation when its overlay is canceled", async () => {
    const user = userEvent.setup()
    const dismissMutationFailure = vi.fn()
    const initial = controller({ dismissMutationFailure })
    const { rerender } = render(<Harness model={initial} />)
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    expect(
      await screen.findByRole("dialog", { name: "Save source view" })
    ).toBeInTheDocument()

    rerender(
      <Harness
        model={controller({
          dismissMutationFailure,
          mutationError: { message: "Offline", retryable: true },
          canRetryMutation: true
        })}
      />
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    expect(dismissMutationFailure).toHaveBeenCalledTimes(1)
  })

  it.each([
    { label: "another workspace", available: true },
    { label: "no workspace", available: false }
  ])("invalidates overlays for $label and refuses stale submit", async ({ available }) => {
    const createA = vi.fn()
    const { rerender } = render(<Harness model={controller({ generation: 1, createView: createA })} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    fireEvent.change(within(dialog).getByRole("textbox", { name: "View name" }), {
      target: { value: "Workspace A" }
    })

    rerender(<Harness model={controller({ generation: 2, available })} />)
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(createA).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("Workspace A")).not.toBeInTheDocument()
  })

  it("restores fallback focus when rejecting a stale unopened overlay", async () => {
    const handled = vi.fn()
    const request: SourceViewOverlayRequest = {
      id: 42,
      kind: "save",
      generation: 1,
      invoker: document.createElement("button")
    }
    render(
      <React.StrictMode>
        <div role="complementary" aria-label="Sources" tabIndex={-1} />
        <SourceViewOverlayHost
          controller={controller({ generation: 2 })}
          request={request}
          onRequestHandled={handled}
        />
      </React.StrictMode>
    )
    const sourcesLandmark = screen.getByRole("complementary", {
      name: "Sources"
    })
    const focusSpy = vi.spyOn(sourcesLandmark, "focus")

    await waitFor(() =>
      expect(document.activeElement).toBe(sourcesLandmark)
    )
    expect(handled).toHaveBeenCalledTimes(1)
    expect(focusSpy).toHaveBeenCalledTimes(1)
  })

  it("refuses an attached Save handler after its controller generation becomes stale", async () => {
    const user = userEvent.setup()
    const model = controller({ generation: 1 })
    render(<Harness model={model} />)
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    await user.type(
      within(dialog).getByRole("textbox", { name: "View name" }),
      "Stale save"
    )

    model.generation = 2
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(model.createView).not.toHaveBeenCalled()
  })

  it("refuses an attached Replace handler after its controller generation becomes stale", async () => {
    const user = userEvent.setup()
    const model = controller({ generation: 1 })
    render(<Harness model={model} />)
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "My PDFs")
    await user.click(
      screen.getByRole("menuitem", { name: "Replace saved view My PDFs" })
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Replace saved view?"
    })

    model.generation = 2
    await user.click(within(dialog).getByRole("button", { name: "Replace" }))

    expect(model.replaceView).not.toHaveBeenCalled()
  })

  it("falls back to the Sources landmark when the invoking pane unmounts", async () => {
    const model = controller()
    const { rerender } = render(<Harness model={model} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    expect(await screen.findByRole("dialog", { name: "Save source view" })).toBeInTheDocument()

    rerender(<Harness model={model} showControls={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("complementary", { name: "Sources" })
      )
    )
  })

  it("restores fallback focus exactly once in StrictMode", async () => {
    const user = userEvent.setup()
    const model = controller()
    const { rerender } = render(
      <React.StrictMode>
        <Harness model={model} />
      </React.StrictMode>
    )
    await user.click(screen.getByRole("button", { name: "Save source view" }))
    expect(
      await screen.findByRole("dialog", { name: "Save source view" })
    ).toBeInTheDocument()
    const landmark = screen.getByRole("complementary", { name: "Sources" })
    const focus = vi.spyOn(landmark, "focus")

    rerender(
      <React.StrictMode>
        <Harness model={model} showControls={false} />
      </React.StrictMode>
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => expect(focus).toHaveBeenCalledTimes(1))
  })

  it("closes after a repeated success announcement completes a new mutation cycle", async () => {
    const createView = vi.fn().mockResolvedValue(undefined)
    const initial = controller({
      announcement: "Saved view created.",
      createView
    })
    const { rerender } = render(<Harness model={initial} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    const dialog = await screen.findByRole("dialog", { name: "Save source view" })
    fireEvent.change(within(dialog).getByRole("textbox", { name: "View name" }), {
      target: { value: "Second save" }
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))

    rerender(
      <Harness
        model={controller({
          announcement: null,
          busy: true,
          mutation: "create",
          createView
        })}
      />
    )
    rerender(
      <Harness
        model={controller({
          announcement: "Saved view created.",
          createView
        })}
      />
    )

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })

  it("falls back to another visible saved-view trigger when the invoking pane unmounts", async () => {
    const model = controller()
    const DualHarness = ({ showFirst }: { showFirst: boolean }) => {
      const [request, setRequest] = React.useState<SourceViewOverlayRequest | null>(null)
      return (
        <div>
          {showFirst && (
            <SourceViewControls
              controller={model}
              sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
              onApplySourceListViewState={vi.fn()}
              onOpenOverlay={setRequest}
            />
          )}
          <SourceViewControls
            controller={model}
            sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
            onApplySourceListViewState={vi.fn()}
            onOpenOverlay={setRequest}
          />
          <SourceViewOverlayHost
            controller={model}
            request={request}
            onRequestHandled={() => setRequest(null)}
          />
        </div>
      )
    }
    const { rerender } = render(<DualHarness showFirst />)
    fireEvent.click(screen.getAllByRole("button", { name: "Save source view" })[0]!)
    expect(await screen.findByRole("dialog", { name: "Save source view" })).toBeInTheDocument()

    rerender(<DualHarness showFirst={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Source views" })
      )
    )
  })

  it("skips a connected invoker inside a hidden pane during focus restoration", async () => {
    const model = controller()
    const HiddenPaneHarness = ({ hideFirst }: { hideFirst: boolean }) => {
      const [request, setRequest] = React.useState<SourceViewOverlayRequest | null>(null)
      return (
        <div>
          <div style={{ display: hideFirst ? "none" : "block" }}>
            <SourceViewControls
              controller={model}
              sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
              onApplySourceListViewState={vi.fn()}
              onOpenOverlay={setRequest}
            />
          </div>
          <SourceViewControls
            controller={model}
            sourceListViewState={DEFAULT_SOURCE_LIST_VIEW_STATE}
            onApplySourceListViewState={vi.fn()}
            onOpenOverlay={setRequest}
          />
          <SourceViewOverlayHost
            controller={model}
            request={request}
            onRequestHandled={() => setRequest(null)}
          />
        </div>
      )
    }
    const { rerender } = render(<HiddenPaneHarness hideFirst={false} />)
    fireEvent.click(screen.getAllByRole("button", { name: "Save source view" })[0]!)
    expect(await screen.findByRole("dialog", { name: "Save source view" })).toBeInTheDocument()

    rerender(<HiddenPaneHarness hideFirst />)
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Source views" })
      )
    )
  })

  it.each([
    { label: "workspace B", available: true },
    { label: "a null workspace", available: false }
  ])("discards replacement confirmation on transition to $label", async ({ available }) => {
    const confirmReplace = vi.fn()
    const initial = controller({ generation: 8, confirmReplace })
    const { rerender } = render(<Harness model={initial} />)
    fireEvent.click(screen.getByRole("button", { name: "Save source view" }))
    expect(await screen.findByRole("dialog", { name: "Save source view" })).toBeInTheDocument()

    const duplicate = {
      viewId: "view-1",
      version: 2,
      name: "My PDFs",
      state: validView().state
    }
    rerender(
      <Harness
        model={controller({ generation: 8, duplicateConflict: duplicate, confirmReplace })}
      />
    )
    expect(
      await screen.findByRole("dialog", { name: "Replace saved view?" })
    ).toBeInTheDocument()

    rerender(<Harness model={controller({ generation: 9, available })} />)
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Replace saved view?" })
      ).not.toBeInTheDocument()
    )
    expect(confirmReplace).not.toHaveBeenCalled()
  })

  it("routes invalid Reset and Delete commands through the one overlay host", async () => {
    const user = userEvent.setup()
    const model = controller()
    render(<Harness model={model} />)
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "Old view")
    await user.click(
      screen.getByRole("menuitem", { name: "Reset saved view Old view" })
    )
    const resetDialog = await screen.findByRole("dialog", {
      name: "Reset saved view?"
    })
    fireEvent.click(within(resetDialog).getByRole("button", { name: "Reset" }))
    await waitFor(() => expect(model.resetView).toHaveBeenCalledWith(model.views[1]))

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "Source views" }))
    await openSavedViewCommands(user, "Old view")
    await user.click(
      screen.getByRole("menuitem", { name: "Delete saved view Old view" })
    )
    expect(
      await screen.findByRole("dialog", { name: "Delete saved view?" })
    ).toBeInTheDocument()
  })

  it.each(["Cancel", "Close", "Escape"])(
      "dismisses a rapid reopened native Save via %s",
      async (action) => {
        const user = userEvent.setup()
        rtlRender(
          <Harness
            model={controller({ announcement: "Saved view no longer exists." })}
          />
        )
        const invoker = screen.getByRole("button", { name: "Save source view" })
        await user.click(invoker)
        const first = await screen.findByRole("dialog")
        await waitFor(() =>
          expect(first.className).toContain("ant-zoom-appear-active")
        )
        fireEvent(first, new Event("webkitAnimationEnd", { bubbles: true }))
        fireEvent.animationEnd(first)
        await waitFor(() => expect(first.className).not.toContain("ant-zoom"))
        await user.type(
          screen.getByRole("textbox", { name: "View name" }),
          "Old draft"
        )
        await user.click(within(first).getByRole("button", { name: "Cancel" }))
        invoker.focus()
        await user.keyboard("{Enter}")
        const input = await screen.findByRole("textbox", { name: "View name" })
        expect(input).toHaveValue("")
        const current = input.closest("[role=dialog]")!
        await waitFor(() =>
          expect(current.className).toContain("ant-zoom-enter-active")
        )
        fireEvent(current, new Event("webkitAnimationEnd", { bubbles: true }))
        fireEvent.animationEnd(current)
        await waitFor(() => expect(current.className).not.toContain("ant-zoom"))
        if (action === "Escape") {
          fireEvent.keyDown(current, {
            key: "Escape",
            code: "Escape",
            keyCode: 27,
            which: 27
          })
        } else
          await user.click(within(current).getByRole("button", { name: action }))
        await waitFor(() =>
          expect(current.className).toContain("ant-zoom-leave-active")
        )
        fireEvent(current, new Event("webkitAnimationEnd", { bubbles: true }))
        fireEvent.animationEnd(current)
        await waitFor(() => expect(current).not.toBeInTheDocument())
      }
    )

  it("does not restore a rejected unopened request after host teardown", async () => {
      vi.useFakeTimers()
      try {
        rtlRender(<button data-source-view-trigger>Current source views</button>)
        const fallback = screen.getByRole("button", {
          name: "Current source views"
        })
        const { unmount } = rtlRender(
          <SourceViewOverlayHost
            controller={controller({ generation: 2 })}
            request={{
              id: 987654,
              kind: "save",
              generation: 1,
              invoker: document.createElement("button")
            }}
            onRequestHandled={vi.fn()}
          />
        )
        unmount()
        await act(async () => vi.runOnlyPendingTimers())
        expect(fallback).not.toHaveFocus()
      } finally {
        vi.useRealTimers()
      }
    })
})
