import React from "react"
import { Dropdown } from "antd"
import type { MenuProps } from "antd"
import { MoreHorizontal } from "lucide-react"

/**
 * The tab menu a ConversationContextMenu offers, for a visible button inside
 * its tab (CS-09). Right-click alone was the only way to reach Export, Rename
 * or Delete, which keyboard and touch users can't do.
 */
export const ConversationMenuContext = React.createContext<{
  items: MenuProps["items"]
  label: string
} | null>(null)

/**
 * A "⋯" button that opens the same menu as right-clicking the tab. Renders
 * nothing outside a ConversationContextMenu.
 */
export const ConversationMenuButton = () => {
  const menu = React.useContext(ConversationMenuContext)
  if (!menu) return null
  return (
    <Dropdown menu={{ items: menu.items }} trigger={["click"]} destroyPopupOnHide>
      <button
        type="button"
        onClick={(event) => event.stopPropagation()}
        className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text"
        aria-label={menu.label}
        title={menu.label}
        aria-haspopup="menu"
      >
        <MoreHorizontal className="size-3" aria-hidden="true" />
      </button>
    </Dropdown>
  )
}
