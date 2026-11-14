import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { TTSPanel } from '../TTSPanel'

const stop = vi.hoisted(() => vi.fn())
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (_: string, fallback: string) => fallback })
}))
vi.mock('@/hooks/document-workspace/useDocumentTTS', () => ({
  useDocumentTTS: () => ({
    state: { isLoading: true, isPlaying: false, isPaused: false, currentText: 'pending speech' },
    voices: [],
    voice: '',
    speed: 1,
    volume: 1,
    progress: 0,
    stop
  })
}))

it('lets Stop cancel synthesis while speech is still loading', async () => {
  render(<TTSPanel />)
  fireEvent.click(screen.getByRole('button', { name: 'Read aloud' }))
  const button = await screen.findByRole('button', { name: 'Stop' })
  expect(button).toBeEnabled()
  fireEvent.click(button)
  expect(stop).toHaveBeenCalledOnce()
})
