import React from "react"
import { cn } from "@/libs/utils"

type MetaTextProps = React.HTMLAttributes<HTMLSpanElement>

/**
 * Small, muted secondary text: hints, field help and toolbar group labels.
 *
 * A plain span rather than antd `Typography.Text type="secondary"`: antd's
 * hashed font reset (`.css-<hash>.ant-typography`) outranks Tailwind size
 * utilities, so `text-[11px]` on Typography rendered at 14px and hints looked
 * louder than headings (XP-15). Callers add layout classes such as `block`.
 */
export const MetaText = React.forwardRef<HTMLSpanElement, MetaTextProps>(
  ({ className, ...rest }, ref) => (
    <span
      {...rest}
      ref={ref}
      className={cn("text-[11px] text-text-muted", className)}
    />
  )
)

MetaText.displayName = "MetaText"
