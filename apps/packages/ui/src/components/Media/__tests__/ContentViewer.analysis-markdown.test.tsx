import React from 'react'
import { App } from 'antd'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === 'string' ? fallback : fallback?.defaultValue || key
  })
}))
vi.mock('@/hooks/useSetting', async () => {
  const React = await import('react')
  return {
    useSetting: (setting: { defaultValue: unknown }) => {
      const [value, setValue] = React.useState(setting.defaultValue)
      return [value, setValue, { isLoading: false }] as const
    }
  }
})
vi.mock('@/hooks/useMediaReadingProgress', () => ({
  useMediaReadingProgress: () => ({ progressPercent: 0 })
}))
vi.mock('../AnalysisModal', () => ({ AnalysisModal: () => null }))
vi.mock('../VersionHistoryPanel', () => ({ VersionHistoryPanel: () => null }))
vi.mock('../DeveloperToolsSection', () => ({ DeveloperToolsSection: () => null }))
vi.mock('../DiffViewModal', () => ({ DiffViewModal: () => null }))

const media = {
  kind: 'media' as const,
  id: 1,
  title: 'Cedar source',
  raw: {},
  meta: { type: 'document' }
}
const renderAnalysis = (analysis: string) => render(
  <App>
    <ContentViewer selectedMedia={media} content="Original source"
      mediaDetail={{ processing: { analysis } }} contentDisplayMode="plain" />
  </App>
)

describe('Media analysis Markdown presentation', () => {
  beforeEach(() => {
    localStorage.clear()
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) }
    })
  })

  it('renders headings, emphasis and lists while copy and edit keep the original Markdown', async () => {
    const analysis = '## Cedar analysis\n\nThe **public** room serves *pear tea*.\n\n- Opens Tuesday\n- Has five chairs'
    renderAnalysis(analysis)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Cedar analysis', level: 2 })).toBeInTheDocument())
    expect(screen.getByText('public').tagName).toBe('STRONG')
    expect(screen.getByText('pear tea').tagName).toBe('EM')
    expect(screen.getByText('Opens Tuesday').tagName).toBe('LI')

    fireEvent.click(screen.getByRole('button', { name: 'Copy analysis to clipboard' }))
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith(analysis))
    fireEvent.click(screen.getByRole('button', { name: 'Edit analysis' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByRole('textbox')).toHaveValue(analysis)
  })

  it('keeps plain analysis readable and does not activate hostile Markdown URLs or HTML', async () => {
    const { container } = renderAnalysis('## Safe analysis\n\nPlain explanation.\n\n[Unsafe](javascript:alert(1)) [Safe](https://example.com/)\n\n<script>alert(1)</script><img src="x" onerror="alert(1)">')
    await screen.findByRole('heading', { name: 'Safe analysis' })
    expect(screen.getByText('Plain explanation.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Safe' })).toHaveAttribute('href', 'https://example.com/')
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(container.querySelector('script, img[onerror]')).toBeNull()
  })
})
