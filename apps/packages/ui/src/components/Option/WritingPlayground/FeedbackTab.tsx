import { Divider, Empty, List, Spin, Switch, Tag, Typography } from "antd"
import type { UseWritingFeedbackReturn } from "./hooks/useWritingFeedback"
import { MOOD_COLORS } from "./feedback-constants"

export function FeedbackTab(props: UseWritingFeedbackReturn) {
  const {
    moodEnabled, setMoodEnabled, currentMood, moodAnalyzing,
    echoEnabled, setEchoEnabled, echoReactions, echoAnalyzing, charsSinceLastEcho,
  } = props

  return (
    <div className="flex flex-col gap-4">
      {/* Mood Detection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <Typography.Text strong className="text-sm">Mood Detection</Typography.Text>
          <Switch size="small" checked={moodEnabled} onChange={setMoodEnabled} aria-label="Toggle mood detection" />
        </div>
        {moodEnabled && (
          <div className="flex items-center gap-2">
            {moodAnalyzing ? (
              <Spin size="small" />
            ) : currentMood ? (
              <Tag color={MOOD_COLORS[currentMood] || "default"} className="!text-sm">
                {currentMood}
              </Tag>
            ) : (
              <Typography.Text type="secondary" className="text-xs">
                Start writing to detect mood...
              </Typography.Text>
            )}
          </div>
        )}
      </div>

      <Divider className="!my-0" />

      {/* Echo Chamber */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <Typography.Text strong className="text-sm">Echo Chamber</Typography.Text>
          <Switch size="small" checked={echoEnabled} onChange={setEchoEnabled} aria-label="Toggle Echo Chamber" />
        </div>
        {echoEnabled && (
          <div className="flex flex-col gap-2">
            {echoAnalyzing && <Spin size="small" />}
            {echoReactions.length === 0 ? (
              <Typography.Text type="secondary" className="text-xs">
                {charsSinceLastEcho < 500
                  ? `Write ${500 - charsSinceLastEcho} more characters for reader reactions...`
                  : "Generating reactions..."}
              </Typography.Text>
            ) : (
              <List
                size="small"
                dataSource={echoReactions}
                renderItem={(reaction) => (
                  <List.Item className="!px-0 !py-2">
                    <div className="flex gap-2 w-full">
                      <span className="text-lg flex-shrink-0">{reaction.emoji}</span>
                      <div className="flex-1 min-w-0">
                        <Typography.Text strong className="text-xs">{reaction.persona}</Typography.Text>
                        <Typography.Paragraph type="secondary" className="!text-xs !mb-0 !mt-0.5">
                          {reaction.message}
                        </Typography.Paragraph>
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default FeedbackTab
