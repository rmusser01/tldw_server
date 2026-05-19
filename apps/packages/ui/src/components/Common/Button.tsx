import React from "react"
import { Loader2 } from "lucide-react"

type ButtonVariant =
  | "primary"
  | "secondary"
  | "danger"
  | "ghost"
  | "text"
  | "outline"
type ButtonSize = "sm" | "md" | "lg"
type ButtonShape = "rounded" | "pill"
type ButtonHtmlType = "button" | "submit" | "reset"

type ButtonProps = {
  variant?: ButtonVariant
  size?: ButtonSize | "small"
  shape?: ButtonShape
  iconOnly?: boolean
  icon?: React.ReactNode
  children: React.ReactNode
  onClick?: React.MouseEventHandler<HTMLButtonElement>
  disabled?: boolean
  loading?: boolean
  className?: string
  type?: ButtonHtmlType | ButtonVariant
  htmlType?: ButtonHtmlType
  ariaLabel?: string
  title?: string
  "data-testid"?: string
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-white hover:bg-primaryStrong active:bg-primaryStrong",
  secondary:
    "bg-surface2 text-text hover:bg-surface active:bg-surface",
  danger:
    "bg-danger text-white hover:bg-danger active:bg-danger",
  ghost:
    "bg-transparent text-text-muted hover:bg-surface2 hover:text-text active:bg-surface2",
  text: "bg-transparent text-primary hover:text-primaryStrong hover:underline",
  outline:
    "border border-border bg-transparent text-text hover:bg-surface2 active:bg-surface2"
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: "px-2.5 py-1 text-xs min-h-[28px]",
  md: "px-3.5 py-1.5 text-sm min-h-[36px]",
  lg: "px-5 py-2 text-base min-h-[44px]"
}

const iconOnlyStyles: Record<ButtonSize, string> = {
  sm: "p-1.5 min-h-[28px] min-w-[28px]",
  md: "p-2 min-h-[36px] min-w-[36px]",
  lg: "p-2.5 min-h-[44px] min-w-[44px]"
}

const shapeStyles: Record<ButtonShape, string> = {
  rounded: "rounded-md",
  pill: "rounded-full"
}

const baseStyles =
  "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors duration-150 ease-out motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:opacity-50 disabled:cursor-not-allowed"

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant,
      size,
      shape,
      iconOnly,
      icon,
      children,
      onClick,
      disabled,
      loading,
      className,
      type,
      htmlType,
      ariaLabel,
      title,
      "data-testid": dataTestId
    },
    ref
  ) => {
    const resolvedVariant =
      variant ??
      (type && Object.prototype.hasOwnProperty.call(variantStyles, type)
        ? (type as ButtonVariant)
        : "secondary")
    const resolvedSize = size === "small" ? "sm" : size ?? "md"
    const resolvedType =
      htmlType ?? (type === "button" || type === "submit" || type === "reset"
        ? type
        : "button")
    const isDisabled = disabled || loading

    return (
      <button
        ref={ref}
        type={resolvedType}
        className={[
          "tldw-btn",
          baseStyles,
          variantStyles[resolvedVariant],
          iconOnly ? iconOnlyStyles[resolvedSize] : sizeStyles[resolvedSize],
          shape ? shapeStyles[shape] : null,
          className
        ]
          .filter(Boolean)
          .join(" ")}
        disabled={isDisabled}
        onClick={onClick}
        aria-label={ariaLabel}
        aria-busy={loading}
        title={title ?? ariaLabel}
        data-testid={dataTestId}
      >
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        {icon}
        {children}
      </button>
    )
  }
)

Button.displayName = "Button"

export default Button
