export interface Character {
  id: string
  name: string
  avatar_url?: string | null
  image_base64?: string | null
  image_mime?: string | null
  system_prompt?: string | null
  greeting?: string | null
  slug?: string | null
  title?: string | null
  tags?: string[]
  extensions?: Record<string, unknown> | null
  metadata?: Record<string, unknown> | null
  version?: number
}

export type CharacterApiResponse = Omit<Character, "id"> & {
  id: string | number
  description?: string | null
  first_message?: string | null
  firstMessage?: string | null
  greet?: string | null
  alternate_greetings?: string[] | string | null
  alternateGreetings?: string[] | string | null
  image_mime?: string | null
}
