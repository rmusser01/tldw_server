import React from "react"
import { useVirtualizer } from "@tanstack/react-virtual"

export type ChatTimelineNavigation = {
  reveal: (messageIndex: number) => Promise<boolean>
}

/** Window rendering only. Selection, search and provider input retain every row. */
export function VirtualChatTimeline<T>({
  blocks, getKey, messageBlocks, scrollParentRef, navigationRef, renderBlock
}: {
  blocks: T[]
  getKey: (index: number) => string
  messageBlocks: Map<number, number>
  scrollParentRef?: React.RefObject<HTMLDivElement>
  navigationRef?: React.MutableRefObject<ChatTimelineNavigation | null>
  renderBlock: (block: T, index: number) => React.ReactNode
}) {
  const container = React.useRef<HTMLDivElement>(null)
  const pending = React.useRef<{ index: number; resolve: (mounted: boolean) => void } | null>(null)
  const [scrollMargin, setScrollMargin] = React.useState(0)
  const enabled = Boolean(scrollParentRef) && blocks.length > 100
  const virtualizer = useVirtualizer({
    count: blocks.length,
    enabled,
    getScrollElement: () => scrollParentRef?.current ?? null,
    getItemKey: getKey,
    estimateSize: () => 160,
    overscan: 4,
    scrollMargin
  })
  const rows = virtualizer.getVirtualItems()
  React.useLayoutEffect(() => {
    const parent = scrollParentRef?.current
    const element = container.current
    if (!parent || !element || !enabled) return
    const measure = () => setScrollMargin(element.getBoundingClientRect().top - parent.getBoundingClientRect().top + parent.scrollTop)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(parent)
    if (element.parentElement) observer.observe(element.parentElement)
    return () => observer.disconnect()
  }, [enabled, scrollParentRef])
  React.useEffect(() => {
    if (pending.current && container.current?.querySelector(`[data-timeline-block="${pending.current.index}"]`)) {
      pending.current.resolve(true)
      pending.current = null
    }
  })
  React.useLayoutEffect(() => {
    if (!navigationRef) return
    navigationRef.current = {
      reveal: async messageIndex => {
        const index = messageBlocks.get(messageIndex)
        if (index === undefined) return false
        pending.current?.resolve(false)
        pending.current = null
        if (!enabled || container.current?.querySelector(`[data-timeline-block="${index}"]`)) return true
        return new Promise<boolean>(resolve => {
          pending.current = { index, resolve }
          virtualizer.scrollToIndex(index, { align: "center" })
        })
      }
    }
    return () => {
      navigationRef.current = null
      pending.current?.resolve(false)
      pending.current = null
    }
  }, [enabled, messageBlocks, navigationRef, virtualizer])
  return (
    <div ref={container} className="relative w-full" style={enabled ? { height: virtualizer.getTotalSize() } : undefined}>
      {(enabled ? rows.map(row => ({ index: row.index, row })) : blocks.map((_, index) => ({ index, row: null }))).map(({ index, row }) => (
        <div key={getKey(index)} data-index={index} data-timeline-block={index}
          ref={enabled ? virtualizer.measureElement : undefined}
          className="flex w-full flex-col items-center"
          style={row ? { position: "absolute", top: 0, left: 0, transform: `translateY(${row.start - scrollMargin}px)` } : undefined}>
          {renderBlock(blocks[index], index)}
        </div>
      ))}
    </div>
  )
}
