import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

import { MediaSectionNavigator } from '../MediaSectionNavigator'
import type { MediaNavigationNode } from '@/hooks/useMediaNavigation'

const makeNode = (overrides: Partial<MediaNavigationNode>): MediaNavigationNode => ({
  id: overrides.id || 'node',
  parent_id: overrides.parent_id ?? null,
  level: overrides.level ?? 0,
  title: overrides.title || 'Untitled',
  order: overrides.order ?? 0,
  path_label: overrides.path_label ?? null,
  target_type: overrides.target_type || 'char_range',
  target_start: overrides.target_start ?? 0,
  target_end: overrides.target_end ?? 10,
  target_href: overrides.target_href ?? null,
  source: overrides.source || 'test',
  confidence: overrides.confidence ?? null
})

const sampleNodes: MediaNavigationNode[] = [
  makeNode({ id: 'chapter-12', title: 'Chapter 12', path_label: '12' }),
  makeNode({
    id: 'section-12-5',
    parent_id: 'chapter-12',
    level: 1,
    title: 'Section 5',
    path_label: '12.5',
    order: 1
  }),
  makeNode({
    id: 'section-12-6',
    parent_id: 'chapter-12',
    level: 1,
    title: 'Section 6',
    path_label: '12.6',
    order: 2
  })
]

const deepBranchNodes: MediaNavigationNode[] = [
  makeNode({ id: 'root', title: 'Root', path_label: '1' }),
  makeNode({
    id: 'child',
    parent_id: 'root',
    level: 1,
    title: 'Long Child',
    path_label: '1.1'
  }),
  ...Array.from({ length: 12 }).map((_, idx) =>
    makeNode({
      id: `leaf-${idx + 1}`,
      parent_id: 'child',
      level: 2,
      title: `Leaf ${idx + 1}`,
      path_label: `1.1.${idx + 1}`,
      order: idx
    })
  )
]

describe('MediaSectionNavigator', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders loading state', () => {
    render(
      <MediaSectionNavigator
        nodes={[]}
        selectedNodeId={null}
        loading={true}
        onSelectNode={vi.fn()}
      />
    )
    expect(screen.getByText('Loading sections...')).toBeInTheDocument()
  })

  it('renders empty state when no nodes exist', () => {
    render(
      <MediaSectionNavigator
        nodes={[]}
        selectedNodeId={null}
        loading={false}
        onSelectNode={vi.fn()}
      />
    )
    expect(
      screen.getByText('No section structure available for this item.')
    ).toBeInTheDocument()
  })

  it('renders tree nodes and selects on click', async () => {
    const onSelectNode = vi.fn()
    render(
      <MediaSectionNavigator
        nodes={sampleNodes}
        selectedNodeId={'section-12-5'}
        onSelectNode={onSelectNode}
      />
    )

    expect(screen.getByText('Chapter 12')).toBeInTheDocument()
    const target = await screen.findByText('Section 5')
    fireEvent.click(target)

    expect(onSelectNode).toHaveBeenCalledTimes(1)
    expect(onSelectNode.mock.calls[0][0].id).toBe('section-12-5')
    expect(screen.getByText('Chapter 12 > Section 5')).toBeInTheDocument()
  })

  it('keeps responsive container classes for mobile/desktop navigator layouts', () => {
    render(
      <MediaSectionNavigator
        nodes={sampleNodes}
        selectedNodeId={null}
        onSelectNode={vi.fn()}
      />
    )

    const navigator = screen.getByLabelText('Chapters and sections')
    expect(navigator).toHaveClass('w-full')
    expect(navigator).toHaveClass('md:w-72')
    expect(navigator).toHaveClass('md:min-w-72')
  })

  it('retains landmark and tree semantics for assistive technology', () => {
    render(
      <MediaSectionNavigator
        nodes={sampleNodes}
        selectedNodeId={null}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByLabelText('Chapters and sections')).toBeInTheDocument()
    expect(screen.getByRole('tree')).toBeInTheDocument()
  })

  it('quick-jumps to top match on Enter', async () => {
    const onSelectNode = vi.fn()
    render(
      <MediaSectionNavigator
        nodes={sampleNodes}
        selectedNodeId={null}
        onSelectNode={onSelectNode}
      />
    )

    const input = screen.getByPlaceholderText('Jump to 12.5 or title')
    fireEvent.change(input, { target: { value: '12.5' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => {
      expect(onSelectNode).toHaveBeenCalledTimes(1)
      expect(onSelectNode.mock.calls[0][0].id).toBe('section-12-5')
    })
  })

  it('shows error state with retry action', () => {
    const onRetry = vi.fn()
    render(
      <MediaSectionNavigator
        nodes={[]}
        selectedNodeId={null}
        error={new Error('Network failed')}
        onRetry={onRetry}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByText('Network failed')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('limits deep child groups with show more/fewer controls', () => {
    render(
      <MediaSectionNavigator
        nodes={deepBranchNodes}
        selectedNodeId={null}
        onSelectNode={vi.fn()}
      />
    )

    const expandButtons = screen.getAllByRole('button', { name: 'Expand section' })
    fireEvent.click(expandButtons[0])

    expect(screen.getByText('Leaf 1')).toBeInTheDocument()
    expect(screen.queryByText('Leaf 12')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Show 4 more/i }))
    expect(screen.getByText('Leaf 12')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Show fewer/i })).toBeInTheDocument()
  })

  it('reveals selected deep node even when beyond default child limit', () => {
    render(
      <MediaSectionNavigator
        nodes={deepBranchNodes}
        selectedNodeId={'leaf-12'}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByText('Leaf 12')).toBeInTheDocument()
    expect(screen.getByText('Root > Long Child > Leaf 12')).toBeInTheDocument()
  })

  it('restores deep selection when selected node changes after initial render', () => {
    const { rerender } = render(
      <MediaSectionNavigator
        nodes={deepBranchNodes}
        selectedNodeId={null}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.queryByText('Leaf 12')).not.toBeInTheDocument()

    rerender(
      <MediaSectionNavigator
        nodes={deepBranchNodes}
        selectedNodeId={'leaf-12'}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByText('Leaf 12')).toBeInTheDocument()
    expect(screen.getByText('Root > Long Child > Leaf 12')).toBeInTheDocument()
  })
})

it('allows independent section selection without changing the active reading node', () => {
  const toggle = vi.fn();
  const read = vi.fn();
  render(<MediaSectionNavigator nodes={sampleNodes} selectedNodeId={null} onSelectNode={read}
    selection={{ nodeIds: ['section-12-5'], onToggle: toggle }} />);
  const chosen = screen.getByRole('checkbox', { name: 'Include Section 5' });
  expect(chosen).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Include Section 6' }));
  expect(toggle).toHaveBeenCalledWith(expect.objectContaining({ id: 'section-12-6' }));
  expect(read).not.toHaveBeenCalled();
});
