import React from "react"
import { Tooltip } from "antd"

interface TabIconLabelProps {
  label: string
  icon: React.ReactNode
}

export const TabIconLabel: React.FC<TabIconLabelProps> = ({ label, icon }) => (
  <Tooltip title={label}>
    <span
      className="flex min-w-[44px] flex-col items-center gap-1"
      aria-label={label}
    >
      <span aria-hidden="true">{icon}</span>
      <span className="text-xs font-medium leading-none text-text-muted">
        {label}
      </span>
    </span>
  </Tooltip>
)

export default TabIconLabel
