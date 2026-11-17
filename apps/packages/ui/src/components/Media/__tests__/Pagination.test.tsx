import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { Pagination } from '../Pagination'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      values?: Record<string, unknown>
    ) => {
      const text = typeof fallbackOrOptions === 'string'
        ? fallbackOrOptions
        : fallbackOrOptions?.defaultValue || key
      return text.replace(/{{(\w+)}}/g, (match, name) => String(values?.[name] ?? match))
    }
  })
}))

describe('Pagination', () => {
  it('supports changing page size via selector', () => {
    const onItemsPerPageChange = vi.fn()

    render(
      <Pagination
        currentPage={1}
        totalPages={5}
        onPageChange={vi.fn()}
        totalItems={100}
        itemsPerPage={20}
        currentItemsCount={20}
        pageSizeOptions={[20, 50, 100]}
        onItemsPerPageChange={onItemsPerPageChange}
      />
    )

    const select = screen.getByLabelText('Per page:') as HTMLSelectElement
    fireEvent.change(select, { target: { value: '50' } })

    expect(onItemsPerPageChange).toHaveBeenCalledWith(50)
  })

  it.each([
    [1, 'Next page', 2, 'Previous page'],
    [2, 'Previous page', 1, 'Next page']
  ] as const)(
    'keeps navigation from zero visible results on page %i',
    (page, action, target, disabledAction) => {
      const onPageChange = vi.fn()
      render(
        <Pagination
          currentPage={page}
          totalPages={2}
          onPageChange={onPageChange}
          totalItems={40}
          itemsPerPage={20}
          currentItemsCount={0}
        />
      )

      expect(screen.getByText('0 of 40 results')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: disabledAction })).toBeDisabled()
      fireEvent.click(screen.getByRole('button', { name: action }))
      expect(onPageChange).toHaveBeenCalledExactlyOnceWith(target)
    }
  )

  it('keeps genuinely empty single-page results compact', () => {
    render(
      <Pagination
        currentPage={1}
        totalPages={0}
        onPageChange={vi.fn()}
        totalItems={0}
        itemsPerPage={20}
        currentItemsCount={0}
      />
    )

    expect(screen.getByText('0 of 0 results')).toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Next page' })
    ).not.toBeInTheDocument()
  })

})
