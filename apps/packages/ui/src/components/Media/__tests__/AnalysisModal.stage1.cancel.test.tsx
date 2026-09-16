import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisModal } from '../AnalysisModal'

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn(),
  getChatModels: vi.fn(),
  resolveApiProviderForModel: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  messageWarning: vi.fn(),
  messageInfo: vi.fn(),
  setSelectedModel: vi.fn()
}))

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

  const Modal = ({ open, title, onCancel, footer, children }: any) => {
    if (!open) return null
    return (
      <div data-testid="analysis-modal">
        <h2>{title}</h2>
        <button type="button" onClick={onCancel}>
          Close
        </button>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    )
  }

  const Button = ({ children, onClick, disabled, loading, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      onClick={onClick}
      disabled={Boolean(disabled || loading)}
      data-loading={loading ? 'true' : 'false'}
      {...rest}
    >
      {children}
    </button>
  )

  const SelectComponent = ({ value, onChange, children, ...rest }: any) => (
    <select
      aria-label={rest['aria-label'] || 'Model'}
      value={value || ''}
      onChange={(event) => onChange?.(event.target.value)}
    >
      {children}
    </select>
  )
  ;(SelectComponent as any).Option = ({ value, children }: any) => (
    <option value={value}>{children}</option>
  )

  const TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea
      aria-label={rest['aria-label']}
      value={value}
      onChange={(event) => onChange?.(event)}
      placeholder={rest.placeholder}
      readOnly={rest.readOnly}
    />
  )

  return {
    ...actual,
    Modal,
    Button,
    Select: SelectComponent,
    Input: { TextArea },
    Spin: () => <div>spinner</div>
  }
})

vi.mock('@plasmohq/storage', () => ({
  Storage: class {
    async get() {
      return null
    }
    async set() {
      return
    }
  }
}))

vi.mock('@/hooks/chat/useSelectedModel', () => ({
  useSelectedModel: () => ({ selectedModel: undefined, setSelectedModel: mocks.setSelectedModel })
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mocks.bgRequest,
  bgStream: mocks.bgStream
}))

vi.mock('@/services/tldw', () => ({
  tldwModels: {
    getChatModels: mocks.getChatModels
  }
}))

vi.mock('@/utils/resolve-api-provider', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/utils/resolve-api-provider')>(),
  resolveApiProviderForModel: mocks.resolveApiProviderForModel
}))

vi.mock('@/hooks/useAntdMessage', () => ({
  useAntdMessage: () => ({
    success: mocks.messageSuccess,
    error: mocks.messageError,
    warning: mocks.messageWarning,
    info: mocks.messageInfo
  })
}))

const streamChunk = (text: string) =>
  `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}`

const createAbortableStream = (finalText: string) => {
  return ({ abortSignal }: { abortSignal?: AbortSignal }) =>
    (async function* () {
      const mid = Math.max(1, Math.floor(finalText.length / 2))
      yield streamChunk(finalText.slice(0, mid))
      await new Promise<void>((resolve, reject) => {
        if (abortSignal?.aborted) {
          reject(new Error('Aborted'))
          return
        }
        const onAbort = () => reject(new Error('Aborted'))
        abortSignal?.addEventListener('abort', onAbort, { once: true })
      })
      yield streamChunk(finalText.slice(mid))
      yield 'data: [DONE]'
    })()
}

describe('AnalysisModal stage 1 cancel plumbing', () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mocks.bgRequest.mockReset()
    mocks.bgStream.mockReset()
    mocks.getChatModels.mockReset()
    mocks.resolveApiProviderForModel.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageInfo.mockReset()
    mocks.setSelectedModel.mockReset()

    mocks.getChatModels.mockResolvedValue([{ id: 'test-model', name: 'Test model' }])
    mocks.resolveApiProviderForModel.mockResolvedValue(undefined)
  })

  afterEach(() => {
    consoleErrorSpy.mockRestore()
  })

  it('aborts active streaming generation via Cancel generation button', async () => {
    mocks.bgStream.mockImplementation(createAbortableStream('Partial analysis'))

    render(
      <AnalysisModal
        open
        onClose={vi.fn()}
        mediaId={42}
        mediaContent="input media content"
      />
    )

    const generateButton = await screen.findByRole('button', { name: 'Generate Analysis' })
    await waitFor(() => {
      expect(generateButton).not.toBeDisabled()
    })

    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Cancel generation' })).toBeInTheDocument()
    })

    const streamCall = mocks.bgStream.mock.calls[0]?.[0] as { abortSignal?: AbortSignal }
    expect(streamCall?.abortSignal).toBeDefined()
    expect(streamCall.abortSignal?.aborted).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }))

    await waitFor(() => {
      expect(mocks.messageInfo).toHaveBeenCalledWith('Analysis generation cancelled')
    })
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Cancel generation' })).not.toBeInTheDocument()
    })
    expect(screen.queryByDisplayValue('Partial')).not.toBeInTheDocument()
    expect(streamCall.abortSignal?.aborted).toBe(true)
    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/chat/completions',
        method: 'POST'
      })
    )
  })

  it('treats cancellation differently from real generation failures', async () => {
    mocks.bgStream.mockImplementation(() =>
      (async function* () {
        throw new Error('stream failed')
      })()
    )
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === '/api/v1/chat/completions' && request.method === 'POST') {
        throw new Error('fallback failed')
      }
      return {}
    })

    render(
      <AnalysisModal
        open
        onClose={vi.fn()}
        mediaId={42}
        mediaContent="input media content"
      />
    )

    const generateButton = await screen.findByRole('button', { name: 'Generate Analysis' })
    await waitFor(() => {
      expect(generateButton).not.toBeDisabled()
    })

    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(mocks.messageError).toHaveBeenCalledWith('Failed to generate analysis')
    })
    expect(mocks.messageInfo).not.toHaveBeenCalled()
  })

  it('supports cancel then restart without stale cancellation state', async () => {
    let phase: 'cancel' | 'success' = 'cancel'

    mocks.bgStream.mockImplementation((init: { abortSignal?: AbortSignal }) => {
      if (phase === 'cancel') {
        return createAbortableStream('Draft analysis')(init)
      }
      return (async function* () {
        yield streamChunk('Final analysis')
        yield 'data: [DONE]'
      })()
    })

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === '/api/v1/media/42' && request.method === 'GET') {
        return { analysis: 'Final analysis' }
      }
      return {}
    })

    const onAnalysisGenerated = vi.fn()
    const onClose = vi.fn()

    render(
      <AnalysisModal
        open
        onClose={onClose}
        mediaId={42}
        mediaContent="input media content"
        onAnalysisGenerated={onAnalysisGenerated}
      />
    )

    const generateButton = await screen.findByRole('button', { name: 'Generate Analysis' })
    await waitFor(() => {
      expect(generateButton).not.toBeDisabled()
    })

    fireEvent.click(generateButton)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Cancel generation' })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }))
    await waitFor(() => {
      expect(mocks.messageInfo).toHaveBeenCalledWith('Analysis generation cancelled')
    })

    phase = 'success'
    fireEvent.click(screen.getByRole('button', { name: 'Generate Analysis' }))

    await waitFor(() => {
      expect(onAnalysisGenerated).toHaveBeenCalledWith('Final analysis', expect.any(String))
    })
    await waitFor(() => {
      expect(onClose).toHaveBeenCalledTimes(1)
    })
    expect(mocks.bgStream).toHaveBeenCalledTimes(2)
    expect(screen.queryByText('Analysis generation cancelled')).not.toBeInTheDocument()
  })
})
