import React from "react"
import { cn } from "@/libs/utils"
import { Button } from "@/components/Common/Button"

type ModalFooterActionVariant =
  | "primary"
  | "secondary"
  | "danger"
  | "ghost"
  | "text"
  | "outline"

export interface ModalFooterAction {
  /** Stable React key for ordered action groups */
  key?: React.Key
  /** Button label */
  label: React.ReactNode
  /** Click handler */
  onClick?: React.MouseEventHandler<HTMLButtonElement>
  /** Show loading spinner */
  loading?: boolean
  /** Disable the button */
  disabled?: boolean
  /** Use danger styling */
  danger?: boolean
  /** Explicit design-system button variant */
  variant?: ModalFooterActionVariant
  /** Optional leading icon */
  icon?: React.ReactNode
  /** Button type for forms */
  type?: "button" | "submit" | "reset"
  /** Accessible label for icon-heavy actions */
  "aria-label"?: string
  /** Tooltip/title text */
  title?: string
  /** Additional CSS classes */
  className?: string
  /** Test ID */
  "data-testid"?: string
}

export interface ModalFooterProps {
  /** Ordered left-side actions */
  leftActions?: ModalFooterAction[]
  /** Ordered right-side actions rendered before cancel/secondary/primary */
  actions?: ModalFooterAction[]
  /** Primary action (right side, emphasized) */
  primaryAction?: ModalFooterAction
  /** Secondary action (left of primary) */
  secondaryAction?: ModalFooterAction
  /** Cancel/close action */
  onCancel?: () => void
  /** Cancel button label (default: "Cancel") */
  cancelLabel?: React.ReactNode
  /** Hide the cancel button */
  hideCancel?: boolean
  /** Alignment of actions */
  align?: "left" | "center" | "right" | "between"
  /** Extra content on the left side */
  leftContent?: React.ReactNode
  /** Custom action content rendered in the right action group */
  children?: React.ReactNode
  /** Additional CSS classes */
  className?: string
  /** Test ID */
  "data-testid"?: string
}

/**
 * Standardized modal footer with consistent action button layout.
 *
 * Replaces custom footer implementations across 23+ modal files.
 *
 * @example
 * ```tsx
 * // Basic usage
 * <Modal footer={null}>
 *   <ModalFooter
 *     primaryAction={{ label: "Save", onClick: handleSave, loading: isSaving }}
 *     onCancel={handleClose}
 *   />
 * </Modal>
 *
 * // With secondary action
 * <ModalFooter
 *   primaryAction={{ label: "Submit", type: "submit" }}
 *   secondaryAction={{ label: "Save Draft", onClick: handleDraft }}
 *   onCancel={handleClose}
 * />
 *
 * // Danger action
 * <ModalFooter
 *   primaryAction={{ label: "Delete", onClick: handleDelete, danger: true }}
 *   onCancel={handleClose}
 *   cancelLabel="Keep"
 * />
 * ```
 */
export const ModalFooter = React.forwardRef<HTMLDivElement, ModalFooterProps>(
  (
    {
      leftActions,
      actions,
      primaryAction,
      secondaryAction,
      onCancel,
      cancelLabel = "Cancel",
      hideCancel = false,
      align = "right",
      leftContent,
      children,
      className,
      "data-testid": dataTestId,
    },
    ref
  ) => {
    const alignmentClasses = {
      left: "justify-start",
      center: "justify-center",
      right: "justify-end",
      between: "justify-between",
    }
    const hasLeftRegion = Boolean(leftContent) || Boolean(leftActions?.length)

    const renderAction = (
      action: ModalFooterAction,
      defaultVariant: ModalFooterActionVariant,
      index: number,
      prefix: string
    ) => {
      const variant =
        action.variant ?? (action.danger ? "danger" : defaultVariant)

      return (
        <Button
          key={action.key ?? `${prefix}-${index}`}
          variant={variant}
          type={action.type || "button"}
          onClick={action.onClick}
          loading={action.loading}
          disabled={action.disabled}
          icon={action.icon}
          ariaLabel={action["aria-label"]}
          title={action.title}
          className={action.className}
          data-testid={action["data-testid"]}
        >
          {action.label}
        </Button>
      )
    }

    return (
      <div
        ref={ref}
        className={cn(
          "flex flex-wrap items-center gap-2 border-t border-border bg-surface px-4 py-3",
          alignmentClasses[align],
          className
        )}
        data-ds-component="ModalFooter"
        data-testid={dataTestId}
      >
        {hasLeftRegion && (
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
            {leftActions?.map((action, index) =>
              renderAction(action, "secondary", index, "left")
            )}
            {leftContent}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {actions?.map((action, index) =>
            renderAction(action, "secondary", index, "action")
          )}
          {children}

          {!hideCancel && onCancel && (
            <Button variant="ghost" onClick={onCancel}>
              {cancelLabel}
            </Button>
          )}

          {secondaryAction &&
            renderAction(secondaryAction, "secondary", 0, "secondary")}

          {primaryAction &&
            renderAction(primaryAction, "primary", 0, "primary")}
        </div>
      </div>
    )
  }
)

ModalFooter.displayName = "ModalFooter"

export default ModalFooter
