/**
 * TimelineModal Component
 *
 * Main modal container for the conversation timeline/graph view.
 * Displays the graph canvas, toolbar, and info panel.
 */

import React, { useEffect, useCallback } from 'react'
import { Modal, Spin } from 'antd'
import { useTimelineStore } from '@/store/timeline'
import { Alert } from '@/components/ui/primitives'
import { GraphCanvas } from './GraphCanvas'
import { TimelineToolbar } from './TimelineToolbar'
import { NodeInfoPanel } from './NodeInfoPanel'

// ============================================================================
// Component
// ============================================================================

export const TimelineModal: React.FC = () => {
  const {
    isOpen,
    isLoading,
    error,
    graph,
    selectedNodeId,
    closeTimeline,
    selectNode,
    settings
  } = useTimelineStore()

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return

      // Escape to close
      if (e.key === 'Escape') {
        if (selectedNodeId) {
          // First escape clears selection
          selectNode(null)
        } else {
          // Second escape closes modal
          closeTimeline()
        }
        e.preventDefault()
      }

      // Ctrl+F to focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        const searchInput = document.getElementById(
          'timeline-search-input'
        ) as HTMLInputElement | null
        searchInput?.focus()
        e.preventDefault()
      }
    },
    [isOpen, selectedNodeId, selectNode, closeTimeline]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Don't render if not open
  if (!isOpen) return null

  return (
    <Modal
      open={isOpen}
      onCancel={closeTimeline}
      title={null}
      footer={null}
      width="100vw"
      style={{
        top: 0,
        padding: 0,
        maxWidth: '100vw'
      }}
      styles={{
        body: {
          height: 'calc(100vh - 55px)',
          padding: 0,
          overflow: 'hidden'
        }
      }}
      closable={true}
      maskClosable={false}
      destroyOnHidden
      className="timeline-modal"
    >
      <div className="timeline-container">
        {/* Toolbar */}
        <TimelineToolbar />

        {/* Main content area */}
        <div className="timeline-content">
          {/* Loading state */}
          {isLoading && (
            <div className="timeline-loading">
              <Spin size="large" tip="Building conversation tree..." />
            </div>
          )}

          {/* Error state */}
          {error && !isLoading && (
            <div className="timeline-error">
              <Alert
                variant="error"
                title="Failed to load timeline"
              >
                {error}
              </Alert>
            </div>
          )}

          {/* Graph canvas */}
          {graph && !isLoading && !error && (
            <GraphCanvas />
          )}

          {/* Empty state */}
          {!graph && !isLoading && !error && (
            <div className="timeline-empty">
              <Alert
                variant="info"
                title="No conversation data"
              >
                This conversation doesn't have any messages yet.
              </Alert>
            </div>
          )}
        </div>

        {/* Info panel (shown when node selected) */}
        {selectedNodeId && (
          <NodeInfoPanel />
        )}

        {/* Legend */}
        {settings.showLegend && graph && !isLoading && (
          <div className="timeline-legend">
            <div className="legend-item">
              <span
                className="legend-dot"
                style={{ backgroundColor: settings.userNodeColor }}
              />
              <span>User</span>
            </div>
            <div className="legend-item">
              <span
                className="legend-dot"
                style={{
                  backgroundColor: settings.assistantNodeColor,
                  border: '1px solid #ccc'
                }}
              />
              <span>Assistant</span>
            </div>
            <div className="legend-item">
              <span
                className="legend-dot"
                style={{ backgroundColor: settings.systemNodeColor }}
              />
              <span>System</span>
            </div>
          </div>
        )}
      </div>

    </Modal>
  )
}

export default TimelineModal
