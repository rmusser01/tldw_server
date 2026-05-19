import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { PromptActionsMenu } from "../PromptActionsMenu"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: Record<string, any>) =>
      options?.defaultValue ?? _key
  })
}))

vi.mock("antd", () => ({
  Dropdown: ({ menu, children }: any) => (
    <div>
      {children}
      <div data-testid="mock-dropdown-menu">
        {(menu?.items || [])
          .filter((item: any) => item && item.type !== "divider")
          .map((item: any) => (
            <button
              key={item.key}
              data-testid={
                React.isValidElement(item.label) &&
                item.label.props &&
                typeof item.label.props["data-testid"] === "string"
                  ? undefined
                  : `menu-item-${item.key}`
              }
              onClick={item.onClick}
            >
              {item.label}
            </button>
          ))}
      </div>
    </div>
  ),
  Tooltip: ({ children }: any) => <>{children}</>
}))

describe("PromptActionsMenu", () => {
  it("shows resolve conflict action when prompt is in conflict state", async () => {
    const user = userEvent.setup()
    const onResolveConflict = vi.fn()

    render(
      <PromptActionsMenu
        promptId="p1"
        syncStatus="conflict"
        serverId={11}
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
        onPullFromServer={vi.fn()}
        onUnlink={vi.fn()}
        onResolveConflict={onResolveConflict}
      />
    )

    const resolveItem = screen.getByTestId("menu-item-resolveConflict")
    expect(resolveItem).toBeInTheDocument()

    await user.click(resolveItem)
    expect(onResolveConflict).toHaveBeenCalledTimes(1)
  })

  it("does not show resolve conflict action for non-conflict prompts", () => {
    render(
      <PromptActionsMenu
        promptId="p2"
        syncStatus="synced"
        serverId={12}
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
        onPullFromServer={vi.fn()}
        onUnlink={vi.fn()}
      />
    )

    expect(screen.queryByTestId("menu-item-resolveConflict")).not.toBeInTheDocument()
  })

  it("shows share link action for synced prompts when handler is provided", async () => {
    const user = userEvent.setup()
    const onShareLink = vi.fn()

    render(
      <PromptActionsMenu
        promptId="p3"
        syncStatus="synced"
        serverId={33}
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
        onShareLink={onShareLink}
      />
    )

    const shareItem = screen.getByTestId("menu-item-shareLink")
    expect(shareItem).toBeInTheDocument()

    await user.click(shareItem)
    expect(onShareLink).toHaveBeenCalledTimes(1)
  })

  it("shows quick test action when handler is provided", async () => {
    const user = userEvent.setup()
    const onQuickTest = vi.fn()

    render(
      <PromptActionsMenu
        promptId="p4"
        syncStatus="local"
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onQuickTest={onQuickTest}
        onDelete={vi.fn()}
      />
    )

    const quickTestItem = screen.getByTestId("menu-item-quickTest")
    expect(quickTestItem).toBeInTheDocument()

    await user.click(quickTestItem)
    expect(onQuickTest).toHaveBeenCalledTimes(1)
  })

  it("hides inline use button when inlineUseInChat is disabled and keeps overflow use action", async () => {
    const user = userEvent.setup()
    const onUseInChat = vi.fn()

    render(
      <PromptActionsMenu
        promptId="p5"
        syncStatus="local"
        inlineUseInChat={false}
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={onUseInChat}
        onDelete={vi.fn()}
      />
    )

    expect(screen.queryByTestId("prompt-use-p5")).not.toBeInTheDocument()

    const useInChatItem = screen.getByTestId("menu-item-useInChat")
    expect(useInChatItem).toBeInTheDocument()
    await user.click(useInChatItem)
    expect(onUseInChat).toHaveBeenCalledTimes(1)
  })

  it("shows retry sync action when onRetrySync handler is provided", async () => {
    const user = userEvent.setup()
    const onRetrySync = vi.fn()

    render(
      <PromptActionsMenu
        promptId="p-retry"
        syncStatus="pending"
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
        onRetrySync={onRetrySync}
        onPushToServer={vi.fn()}
      />
    )

    const retrySyncItem = screen.getByTestId("menu-item-retrySync")
    expect(retrySyncItem).toBeInTheDocument()

    await user.click(retrySyncItem)
    expect(onRetrySync).toHaveBeenCalledTimes(1)
  })

  it("does not show retry sync action when onRetrySync is not provided", () => {
    render(
      <PromptActionsMenu
        promptId="p-no-retry"
        syncStatus="pending"
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
        onPushToServer={vi.fn()}
      />
    )

    expect(screen.queryByTestId("menu-item-retrySync")).not.toBeInTheDocument()
  })

  it("uses characters-like action button spacing classes", () => {
    render(
      <PromptActionsMenu
        promptId="p6"
        syncStatus="local"
        onEdit={vi.fn()}
        onDuplicate={vi.fn()}
        onUseInChat={vi.fn()}
        onDelete={vi.fn()}
      />
    )

    const editButton = screen.getByTestId("prompt-edit-p6")
    const useButton = screen.getByTestId("prompt-use-p6")
    const moreButton = screen.getByTestId("prompt-more-p6")
    const actionRow = editButton.parentElement

    expect(actionRow).toHaveClass("whitespace-nowrap")
    expect(editButton.className).toContain("p-1.5")
    expect(useButton.className).toContain("p-1.5")
    expect(moreButton.className).toContain("p-1.5")
    expect(editButton.className).not.toContain("min-h-11")
    expect(useButton.className).not.toContain("min-h-11")
    expect(moreButton.className).not.toContain("min-h-11")
  })
})
