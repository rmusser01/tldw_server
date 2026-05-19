import { ChatHistory } from "@/store"

export * from "./archetype"

export type BotResponse = {
    bot: {
        text: string
        sourceDocuments: any[]
    }
    history: ChatHistory
    history_id: string
}
