import React from "react"
import { Switch } from "antd"

type SidepanelPersistenceSwitchProps = Omit<
  React.LabelHTMLAttributes<HTMLLabelElement>,
  "onChange" | "children"
> & {
  checked: boolean
  disabled?: boolean
  onChange: (checked: boolean) => void
  /** Accessible name of the switch, e.g. "Save chat to history". */
  ariaLabel: string
  /** Visible text beside the switch, e.g. where the chat is saved. */
  label: React.ReactNode
}

/**
 * The side panel "Save chat to history" switch with its visible label.
 *
 * It uses antd's default switch size rather than the 28x16 px small one, and
 * the whole label is a click target, so it meets the WCAG 2.5.8 target size
 * (AX-16). Extra props (a wrapping Tooltip's handlers) go to the label.
 */
export const SidepanelPersistenceSwitch = React.forwardRef<
  HTMLLabelElement,
  SidepanelPersistenceSwitchProps
>(({ checked, disabled, onChange, ariaLabel, label, className, ...rest }, ref) => (
  <label
    {...rest}
    ref={ref}
    className={[
      "flex min-h-6 items-center gap-2",
      disabled ? "cursor-not-allowed" : "cursor-pointer",
      className
    ]
      .filter(Boolean)
      .join(" ")}
  >
    <Switch
      checked={checked}
      disabled={disabled}
      onChange={(next) => onChange(next)}
      aria-label={ariaLabel}
    />
    <span className="text-xs text-text whitespace-nowrap">{label}</span>
  </label>
))

SidepanelPersistenceSwitch.displayName = "SidepanelPersistenceSwitch"
