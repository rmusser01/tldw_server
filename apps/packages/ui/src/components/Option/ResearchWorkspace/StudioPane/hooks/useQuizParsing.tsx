import React from "react"
import type { GeneratedArtifact } from "@/types/workspace"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type FlashcardDraft = {
  front: string
  back: string
}

export type QuizQuestionDraft = {
  question: string
  options: string[]
  answer: string
  explanation?: string
}

export type ParsedQuizQuestion = {
  question: string
  options: string[]
  answer: string
  explanation: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure parsing / formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

export function parseFlashcards(content: string): FlashcardDraft[] {
  const cards: FlashcardDraft[] = []
  const lines = content.split("\n")
  let currentFront = ""
  let currentBack = ""

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.toLowerCase().startsWith("front:")) {
      if (currentFront && currentBack) {
        cards.push({ front: currentFront, back: currentBack })
      }
      currentFront = trimmed.substring(6).trim()
      currentBack = ""
    } else if (trimmed.toLowerCase().startsWith("back:")) {
      currentBack = trimmed.substring(5).trim()
    }
  }

  if (currentFront && currentBack) {
    cards.push({ front: currentFront, back: currentBack })
  }

  return cards
}

export function formatFlashcardsContent(cards: FlashcardDraft[]): string {
  return cards
    .map((card) => `Front: ${card.front}\nBack: ${card.back}`)
    .join("\n\n")
}

export function parseQuizQuestions(content: string): QuizQuestionDraft[] {
  const questions: QuizQuestionDraft[] = []
  const lines = content.split("\n")
  let current: QuizQuestionDraft | null = null

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) continue

    const questionMatch = line.match(/^Q\d+:\s*(.+)$/i)
    if (questionMatch) {
      if (current && current.question) {
        questions.push(current)
      }
      current = {
        question: questionMatch[1].trim(),
        options: [],
        answer: "",
        explanation: ""
      }
      continue
    }

    if (!current) continue

    const optionMatch = line.match(/^(?:[-*]\s*)?[A-Z]\.\s*(.+)$/)
    if (optionMatch) {
      current.options.push(optionMatch[1].trim())
      continue
    }

    if (line.toLowerCase().startsWith("answer:")) {
      current.answer = line.substring("answer:".length).trim()
      continue
    }

    if (line.toLowerCase().startsWith("explanation:")) {
      current.explanation = line.substring("explanation:".length).trim()
    }
  }

  if (current && current.question) {
    questions.push(current)
  }
  return questions
}

export function formatQuizQuestionsContent(
  questions: QuizQuestionDraft[],
  title: string
): string {
  let content = `Quiz: ${title}\n`
  content += `Total Questions: ${questions.length}\n\n`
  questions.forEach((question, index) => {
    content += `Q${index + 1}: ${question.question}\n`
    question.options.forEach((option, optionIndex) => {
      content += `  ${String.fromCharCode(65 + optionIndex)}. ${option}\n`
    })
    content += `Answer: ${question.answer}\n`
    if (question.explanation && question.explanation.trim().length > 0) {
      content += `Explanation: ${question.explanation}\n`
    }
    content += "\n"
  })
  return content
}

export function getArtifactFlashcards(artifact: GeneratedArtifact): FlashcardDraft[] {
  const flashcardsFromData = isRecord(artifact.data) &&
    Array.isArray(artifact.data.flashcards)
      ? artifact.data.flashcards
            .map((entry) => {
              if (!isRecord(entry)) return null
              const front = String(entry.front || "").trim()
              const back = String(entry.back || "").trim()
              if (!front || !back) return null
              return { front, back }
            })
            .filter((entry): entry is FlashcardDraft => entry !== null)
      : []

  if (flashcardsFromData.length > 0) {
    return flashcardsFromData
  }

  const parsed = parseFlashcards(artifact.content || "")
  if (parsed.length > 0) return parsed
  return [{ front: "", back: "" }]
}

export function getArtifactQuizQuestions(artifact: GeneratedArtifact): QuizQuestionDraft[] {
  const questionsFromData = isRecord(artifact.data) &&
    Array.isArray(artifact.data.questions)
      ? artifact.data.questions
            .map((entry): ParsedQuizQuestion | null => {
              if (!isRecord(entry)) return null
              const question = String(
                entry.question || entry.question_text || ""
              ).trim()
              const options = Array.isArray(entry.options)
                ? entry.options.map((option) => String(option).trim()).filter(Boolean)
                : []
              const answer = String(
                entry.answer || entry.correct_answer || ""
              ).trim()
              const explanation = entry.explanation
                ? String(entry.explanation)
                : ""
              if (!question) return null
              return { question, options, answer, explanation }
            })
            .filter((entry): entry is ParsedQuizQuestion => entry !== null)
      : []

  if (questionsFromData.length > 0) {
    return questionsFromData
  }

  const parsed = parseQuizQuestions(artifact.content || "")
  if (parsed.length > 0) return parsed
  return [{ question: "", options: [], answer: "", explanation: "" }]
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export interface UseQuizParsingDeps {
  // intentionally empty - this hook wraps pure helpers for consistency
}

export function useQuizParsing(_deps?: UseQuizParsingDeps) {
  const stableParseFlashcards = React.useCallback(
    (content: string) => parseFlashcards(content),
    []
  )
  const stableFormatFlashcardsContent = React.useCallback(
    (cards: FlashcardDraft[]) => formatFlashcardsContent(cards),
    []
  )
  const stableParseQuizQuestions = React.useCallback(
    (content: string) => parseQuizQuestions(content),
    []
  )
  const stableFormatQuizQuestionsContent = React.useCallback(
    (questions: QuizQuestionDraft[], title: string) =>
      formatQuizQuestionsContent(questions, title),
    []
  )
  const stableGetArtifactFlashcards = React.useCallback(
    (artifact: GeneratedArtifact) => getArtifactFlashcards(artifact),
    []
  )
  const stableGetArtifactQuizQuestions = React.useCallback(
    (artifact: GeneratedArtifact) => getArtifactQuizQuestions(artifact),
    []
  )

  return {
    parseFlashcards: stableParseFlashcards,
    formatFlashcardsContent: stableFormatFlashcardsContent,
    parseQuizQuestions: stableParseQuizQuestions,
    formatQuizQuestionsContent: stableFormatQuizQuestionsContent,
    getArtifactFlashcards: stableGetArtifactFlashcards,
    getArtifactQuizQuestions: stableGetArtifactQuizQuestions,
  }
}
