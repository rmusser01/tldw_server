export type NoteListItem = {
  id: string | number
  title?: string
  content_language?: string | null;
  created_at?: string;
  content?: string
  content_preview?: string | null
  updated_at?: string
  deleted?: boolean
  conversation_id?: string | null
  message_id?: string | null
  keywords?: string[]
  cover_image_url?: string | null
  membership_source?: 'manual' | 'smart' | 'both'
  version?: number
}
