import React from 'react'
import { App, ConfigProvider } from 'antd'
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useContentEditState } from '../hooks/useContentEditState'

vi.mock('@/components/Common/confirm-danger', () => ({
  useConfirmDanger: () => async () => true
}))

function DeleteHarness({ onDelete }: { onDelete: () => Promise<void> }) {
  const { handleDeleteItem } = useContentEditState({
    selectedMedia: { id: 1, kind: 'media', title: 'Cedar', raw: {} },
    selectedMediaId: '1', content: 'Source text', mediaDetail: {}, isNote: false,
    onDeleteItem: onDelete,
    t: (key, options) => options?.defaultValue || key
  })
  return <button onClick={() => void handleDeleteItem()}>Delete media</button>
}

describe('Media deletion feedback context', () => {
  afterEach(() => vi.restoreAllMocks())

  it.each([false, true])('reports deletion failure=%s through the current application context', async (fails) => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    const onDelete = vi.fn(async () => {
      if (fails) throw new Error('Media deletion failed')
    })
    render(<ConfigProvider prefixCls="media-feedback" theme={{ token: { colorPrimary: '#123456' } }}>
      <App><DeleteHarness onDelete={onDelete} /></App>
    </ConfigProvider>)
    fireEvent.click(screen.getByRole('button', { name: 'Delete media' }))
    const feedback = await screen.findByText(fails ? 'Media deletion failed' : 'Deleted')
    expect(feedback.closest('.media-feedback-message')).not.toBeNull()
    expect(onDelete).toHaveBeenCalledTimes(1)
    expect(errors.mock.calls.flat().map(String).join('\n')).not.toMatch(/cannot consume context/i)
  })
})
