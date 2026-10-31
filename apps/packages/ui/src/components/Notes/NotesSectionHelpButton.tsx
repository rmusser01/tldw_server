import React from 'react'
import { QuestionCircleOutlined } from '@ant-design/icons'

type NotesSectionHelpButtonProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-label' | 'children' | 'type'
> & {
  /** Accessible name, for example "Tags help". */
  label: string
}

/**
 * The "?" button beside a Notes section title, usually wrapped in a Tooltip.
 *
 * The icon stays small, but the button is at least 24x24 px so it meets the
 * WCAG 2.5.8 target size (AX-16), and keyboard focus shows a ring in the theme
 * focus colour (AX-08). Extra props (Tooltip's handlers) pass through.
 */
const NotesSectionHelpButton = React.forwardRef<
  HTMLButtonElement,
  NotesSectionHelpButtonProps
>(({ label, className, ...rest }, ref) => (
  <button
    {...rest}
    ref={ref}
    type="button"
    aria-label={label}
    className={[
      'inline-flex min-h-6 min-w-6 shrink-0 items-center justify-center rounded-sm text-text-muted transition-colors hover:text-text',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus',
      className
    ]
      .filter(Boolean)
      .join(' ')}
  >
    <QuestionCircleOutlined aria-hidden="true" className="text-xs" />
  </button>
))

NotesSectionHelpButton.displayName = 'NotesSectionHelpButton'

export default NotesSectionHelpButton
