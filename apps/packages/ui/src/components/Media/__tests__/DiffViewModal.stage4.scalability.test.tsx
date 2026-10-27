import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
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

describe('DiffViewModal stage 4 scalability guardrails', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('terminates pending work when the modal closes', async () => {
    const terminate = vi.fn()
    vi.stubGlobal('Worker', class {
      onmessage = null
      onerror = null
      terminate = terminate
      postMessage() {}
    })
    const text = 'line\n'.repeat(2100)
    const view = render(<DiffViewModal open onClose={vi.fn()} leftText={text} rightText={text} />)
    view.rerender(<DiffViewModal open={false} onClose={vi.fn()} leftText={text} rightText={text} />)
    await waitFor(() => expect(terminate).toHaveBeenCalledOnce())
  })

  it('shows large-document warning before computing sampled diff over hard threshold', () => {
    const hugeLeft = 'A'.repeat(170_000)
    const hugeRight = 'B'.repeat(170_000)

    render(
      <DiffViewModal
        open
        onClose={vi.fn()}
        leftText={hugeLeft}
        rightText={hugeRight}
      />
    )

    expect(screen.getByText('Large comparison detected')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate sampled diff' })).toBeInTheDocument()
  })

  it('uses worker-backed diff pipeline when line length exceeds sync threshold', async () => {
    const workerPostMessage = vi.fn()
    const workerTerminate = vi.fn()

    class MockWorker {
      onmessage: ((event: MessageEvent) => void) | null = null
      onerror: ((event: Event) => void) | null = null

      constructor() {
        // no-op
      }

      postMessage(payload: unknown) {
        workerPostMessage(payload)
        Promise.resolve().then(() => {
          this.onmessage?.({
            data: {
              type: 'result',
              lines: [{ type: 'same', text: 'line' }]
            }
          } as MessageEvent)
        })
      }

      terminate() {
        workerTerminate()
      }
    }

    vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker)

    const longLeft = Array.from({ length: 2600 }, (_, idx) => `left-${idx}`).join('\n')
    const longRight = Array.from({ length: 2600 }, (_, idx) => `right-${idx}`).join('\n')

    render(
      <DiffViewModal
        open
        onClose={vi.fn()}
        leftText={longLeft}
        rightText={longRight}
      />
    )

    await waitFor(() => {
      expect(workerPostMessage).toHaveBeenCalled()
      expect(workerTerminate).toHaveBeenCalled()
    })
  })
})
