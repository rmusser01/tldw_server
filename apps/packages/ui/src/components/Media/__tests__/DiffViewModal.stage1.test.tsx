import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
// Scripted scrolls honour the OS reduced-motion setting (AX-19).
import { smoothScrollBehavior } from '@/utils/reduced-motion'
import { DiffViewModal } from '../DiffViewModal'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

vi.mock('antd', async (importOriginal) => {
  const React = await import('react')
  const actual = await importOriginal<typeof import('antd')>()

  const Modal = ({ open, title, onCancel, afterOpenChange, children }: any) => {
    React.useEffect(() => {
      afterOpenChange?.(open)
    }, [open, afterOpenChange])

    if (!open) return null
    return (
      <div data-testid="diff-modal">
        <h2>{title}</h2>
        <button type="button" onClick={onCancel}>
          Close
        </button>
        {children}
      </div>
    )
  }

  const RadioButton = ({ value, children, __groupValue, __groupOnChange }: any) => (
    <button
      type="button"
      aria-pressed={__groupValue === value}
      onClick={() => __groupOnChange?.({ target: { value } })}
    >
      {children}
    </button>
  )

  const RadioGroup = ({ value, onChange, children }: any) => (
    <div data-testid="diff-view-mode-group">
      {React.Children.map(children, (child: any) =>
        React.cloneElement(child, {
          __groupValue: value,
          __groupOnChange: onChange
        })
      )}
    </div>
  )

  return {
    ...actual,
    Modal,
    Radio: {
      Group: RadioGroup,
      Button: RadioButton
    }
  }
})

describe('DiffViewModal stage 1 keyboard regressions', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('supports keyboard scrolling shortcuts in unified and side-by-side modes', () => {
    const onClose = vi.fn()

    render(
      <DiffViewModal
        open
        onClose={onClose}
        leftText={'alpha\nbeta\ngamma'}
        rightText={'alpha\nchanged\ngamma'}
        leftLabel="Version 3"
        rightLabel="Version 2"
      />
    )

    const region = screen.getByRole('region', {
      name: 'Diff content - use arrow keys to scroll'
    }) as HTMLDivElement

    const scrollBy = vi.fn()
    const scrollTo = vi.fn()
    ;(region as any).scrollBy = scrollBy
    ;(region as any).scrollTo = scrollTo
    Object.defineProperty(region, 'scrollHeight', {
      configurable: true,
      value: 1200
    })

    fireEvent.keyDown(region, { key: 'j' })
    fireEvent.keyDown(region, { key: 'k' })
    fireEvent.keyDown(region, { key: 'PageDown' })
    fireEvent.keyDown(region, { key: 'PageUp' })
    fireEvent.keyDown(region, { key: 'Home' })
    fireEvent.keyDown(region, { key: 'End' })

    expect(scrollBy).toHaveBeenCalledWith({ top: 100, behavior: smoothScrollBehavior() })
    expect(scrollBy).toHaveBeenCalledWith({ top: -100, behavior: smoothScrollBehavior() })
    expect(scrollBy).toHaveBeenCalledWith({ top: 400, behavior: smoothScrollBehavior() })
    expect(scrollBy).toHaveBeenCalledWith({ top: -400, behavior: smoothScrollBehavior() })
    expect(scrollTo).toHaveBeenCalledWith({ top: 0, behavior: smoothScrollBehavior() })
    expect(scrollTo).toHaveBeenCalledWith({ top: 1200, behavior: smoothScrollBehavior() })

    fireEvent.click(screen.getByRole('button', { name: 'Side by Side' }))
    expect(screen.getAllByText('Version 3').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Version 2').length).toBeGreaterThan(0)

    fireEvent.keyDown(region, { key: 'ArrowDown' })
    expect(scrollBy).toHaveBeenCalledWith({ top: 100, behavior: smoothScrollBehavior() })

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('keeps keyboard hint text behind mobile-hidden utility classes', () => {
    render(
      <DiffViewModal
        open
        onClose={vi.fn()}
        leftText={'alpha'}
        rightText={'beta'}
      />
    )

    const hint = screen.getByText('↑↓ or j/k to scroll, PgUp/PgDn for pages')
    expect(hint).toHaveClass('hidden')
    expect(hint).toHaveClass('sm:block')
  })
})
