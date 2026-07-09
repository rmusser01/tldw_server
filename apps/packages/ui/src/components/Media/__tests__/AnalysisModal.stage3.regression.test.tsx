import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
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

vi.mock('@plasmohq/storage/hook', () => ({
  useStorage: () => [undefined, mocks.setSelectedModel]
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

vi.mock('@/utils/resolve-api-provider', () => ({
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

describe('AnalysisModal stage 3 regression coverage', () => {
  beforeEach(() => {
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

  it('preserves preset/custom prompt behavior and sends expected request body', async () => {
    mocks.bgStream.mockImplementation(() =>
      (async function* () {
        yield streamChunk('Generated ')
        yield streamChunk('analysis output')
        yield 'data: [DONE]'
      })()
    )

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === '/api/v1/media/42' && request.method === 'GET') {
        return { analysis: 'Generated analysis output' }
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
        mediaContent="media body"
        onAnalysisGenerated={onAnalysisGenerated}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Generate Analysis' })).not.toBeDisabled()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Show Presets' }))
    fireEvent.click(screen.getByRole('button', { name: 'Critical Review' }))

    const systemPrompt = screen.getByLabelText('System Prompt') as HTMLTextAreaElement
    expect(systemPrompt.value).toContain('Act as a critical reviewer.')

    const userPrefix = screen.getByLabelText('User Prompt Prefix') as HTMLTextAreaElement
    fireEvent.change(userPrefix, { target: { value: 'CUSTOM PREFIX' } })

    fireEvent.click(screen.getByRole('button', { name: 'Generate Analysis' }))

    await waitFor(() => {
      expect(onAnalysisGenerated).toHaveBeenCalledWith(
        'Generated analysis output',
        expect.stringContaining('Act as a critical reviewer.')
      )
    })

    expect(mocks.bgStream).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/chat/completions',
        method: 'POST',
        body: expect.objectContaining({
          stream: true,
          messages: [
            expect.objectContaining({
              role: 'system',
              content: expect.stringContaining('Act as a critical reviewer.')
            }),
            expect.objectContaining({
              role: 'user',
              content: 'CUSTOM PREFIX\n\nmedia body'
            })
          ]
        })
      })
    )

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/42',
          method: 'PUT',
          body: expect.objectContaining({
            analysis: 'Generated analysis output',
            prompt: expect.stringContaining('Act as a critical reviewer.')
          })
        })
      )
    })

    expect(mocks.messageSuccess).toHaveBeenCalledWith('Analysis generated and saved')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows provider recovery copy when generation fails without a provider', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    mocks.bgStream.mockImplementation(() =>
      (async function* () {
        throw new Error('Error: Analysis API provider is required.')
      })()
    )
    mocks.bgRequest.mockRejectedValueOnce(new Error('Error: Analysis API provider is required.'))

    render(
      <AnalysisModal
        open
        onClose={vi.fn()}
        mediaId={42}
        mediaContent="media body"
        onAnalysisGenerated={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Generate Analysis' })).not.toBeDisabled()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Generate Analysis' }))

    await waitFor(() => {
      expect(mocks.messageError).toHaveBeenCalledWith(
        'Choose an analysis provider, then retry analysis.'
      )
    })
    consoleError.mockRestore()
  })
})
