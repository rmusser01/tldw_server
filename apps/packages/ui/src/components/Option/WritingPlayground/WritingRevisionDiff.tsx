import React from "react"
import { Typography } from "antd"

const { Text } = Typography

export type WritingRevisionDiffProps = {
  beforeText: string
  afterText: string
}

const splitWords = (text: string): string[] => text.match(/\S+|\s+/g) ?? []

const findSharedPrefix = (before: string[], after: string[]): number => {
  let index = 0
  while (index < before.length && index < after.length && before[index] === after[index]) {
    index += 1
  }
  return index
}

const findSharedSuffix = (
  before: string[],
  after: string[],
  prefixLength: number
): number => {
  let count = 0
  while (
    count + prefixLength < before.length &&
    count + prefixLength < after.length &&
    before[before.length - 1 - count] === after[after.length - 1 - count]
  ) {
    count += 1
  }
  return count
}

export function WritingRevisionDiff({
  beforeText,
  afterText
}: WritingRevisionDiffProps) {
  const beforeChunks = splitWords(beforeText)
  const afterChunks = splitWords(afterText)
  const prefixLength = findSharedPrefix(beforeChunks, afterChunks)
  const suffixLength = findSharedSuffix(beforeChunks, afterChunks, prefixLength)
  const beforeChanged = beforeChunks.slice(
    prefixLength,
    beforeChunks.length - suffixLength
  )
  const afterChanged = afterChunks.slice(
    prefixLength,
    afterChunks.length - suffixLength
  )

  return (
    <div className="grid gap-2 md:grid-cols-2" data-testid="writing-revision-diff">
      <div className="min-h-20 rounded border border-red-100 bg-red-50 p-2">
        <Text strong className="block text-xs">
          Before
        </Text>
        <pre className="m-0 whitespace-pre-wrap text-xs">{beforeText}</pre>
        {beforeChanged.length > 0 ? (
          <Text type="danger" className="text-xs">
            Changed: {beforeChanged.join("")}
          </Text>
        ) : null}
      </div>
      <div className="min-h-20 rounded border border-green-100 bg-green-50 p-2">
        <Text strong className="block text-xs">
          After
        </Text>
        <pre className="m-0 whitespace-pre-wrap text-xs">{afterText}</pre>
        {afterChanged.length > 0 ? (
          <Text type="success" className="text-xs">
            Changed: {afterChanged.join("")}
          </Text>
        ) : null}
      </div>
    </div>
  )
}
