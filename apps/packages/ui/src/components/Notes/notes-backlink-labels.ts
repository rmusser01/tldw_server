type Translate = (key: string, options: { defaultValue: string }) => string

/**
 * The ids behind a note's chat backlink, for its tooltip only. The visible
 * text names the chat ("Linked chat" until its title resolves); raw ids are
 * kept out of it (XP-14).
 */
export const formatBacklinkIdsTooltip = (
  t: Translate,
  conversationId: string,
  messageId?: string | null
): string =>
  [
    `${t('option:notesSearch.linkedConversationIdLabel', { defaultValue: 'Conversation ID' })}: ${conversationId}`,
    messageId
      ? `${t('option:notesSearch.linkedMessageIdLabel', { defaultValue: 'Message ID' })}: ${messageId}`
      : null
  ]
    .filter(Boolean)
    .join(' · ')
