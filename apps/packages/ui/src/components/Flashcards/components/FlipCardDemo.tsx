/**
 * FlipCardDemo — A standalone flip-card component for demo mode.
 *
 * Shows sample flashcards without requiring a server connection or SRS.
 * Users can flip cards and navigate between them to experience the
 * flashcard study flow before setting up a server.
 */

import React, { useState, useCallback } from "react"
import { Button, Space, Tag } from "antd"
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react"
import { useTranslation } from "react-i18next"

type DemoCard = {
  front: string
  back: string
  deck: string
}

export const FlipCardDemo: React.FC = () => {
  const { t } = useTranslation()
  const [cardIndex, setCardIndex] = useState(0)
  const [flipped, setFlipped] = useState(false)

  const demoCards: DemoCard[] = [
    {
      front: t("option:flashcards.demoCard1Front", {
        defaultValue: "What is spaced repetition?"
      }),
      back: t("option:flashcards.demoCard1Back", {
        defaultValue:
          "A learning technique that reviews material at increasing intervals based on how well you remember it. Cards you know well are shown less often; cards you struggle with are shown more frequently."
      }),
      deck: t("option:flashcards.demoCard1Deck", {
        defaultValue: "Study Techniques"
      })
    },
    {
      front: t("option:flashcards.demoCard2Front", {
        defaultValue: "What does RAG stand for in AI?"
      }),
      back: t("option:flashcards.demoCard2Back", {
        defaultValue:
          "Retrieval-Augmented Generation — a technique that enhances LLM responses by retrieving relevant documents from a knowledge base before generating an answer."
      }),
      deck: t("option:flashcards.demoCard2Deck", {
        defaultValue: "AI Concepts"
      })
    },
    {
      front: t("option:flashcards.demoCard3Front", {
        defaultValue: "What is the difference between a podcast and an audiobook?"
      }),
      back: t("option:flashcards.demoCard3Back", {
        defaultValue:
          "Podcasts are episodic audio series, often conversational, released on a schedule. Audiobooks are narrated versions of written books, typically consumed as a complete work."
      }),
      deck: t("option:flashcards.demoCard3Deck", {
        defaultValue: "Media Literacy"
      })
    },
    {
      front: t("option:flashcards.demoCard4Front", {
        defaultValue: "What is an API key?"
      }),
      back: t("option:flashcards.demoCard4Back", {
        defaultValue:
          "A unique identifier used to authenticate requests to an API. Like a password for software — it proves your application is authorized to access the service."
      }),
      deck: t("option:flashcards.demoCard4Deck", {
        defaultValue: "Developer Basics"
      })
    },
    {
      front: t("option:flashcards.demoCard5Front", {
        defaultValue: "What is the Feynman Technique?"
      }),
      back: t("option:flashcards.demoCard5Back", {
        defaultValue:
          "A learning method: (1) Choose a concept, (2) Explain it simply as if teaching a child, (3) Identify gaps in your explanation, (4) Review and simplify further."
      }),
      deck: t("option:flashcards.demoCard5Deck", {
        defaultValue: "Study Techniques"
      })
    }
  ]

  const card = demoCards[cardIndex]

  const handleFlip = useCallback(() => setFlipped((f) => !f), [])
  const handleNext = useCallback(() => {
    setCardIndex((i) => (i + 1) % demoCards.length)
    setFlipped(false)
  }, [demoCards.length])
  const handlePrev = useCallback(() => {
    setCardIndex((i) => (i - 1 + demoCards.length) % demoCards.length)
    setFlipped(false)
  }, [demoCards.length])

  return (
    <div className="mx-auto max-w-md">
      <div className="mb-3 flex items-center justify-between text-xs text-text-muted">
        <Tag>{card.deck}</Tag>
        <span>
          {cardIndex + 1} / {demoCards.length}
        </span>
      </div>

      <button
        type="button"
        onClick={handleFlip}
        className="w-full cursor-pointer rounded-xl border border-border bg-surface p-6 text-left shadow-sm transition-all hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary/40"
        aria-label={
          flipped
            ? t("option:flashcards.demoShowQuestion", {
                defaultValue: "Show question"
              })
            : t("option:flashcards.demoShowAnswer", {
                defaultValue: "Show answer"
              })
        }
      >
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-text-subtle">
          {flipped
            ? t("option:flashcards.demoAnswerLabel", {
                defaultValue: "Answer"
              })
            : t("option:flashcards.demoQuestionLabel", {
                defaultValue: "Question"
              })}
        </div>
        <div className="min-h-[80px] text-sm leading-relaxed text-text">
          {flipped ? card.back : card.front}
        </div>
        <div className="mt-4 text-center text-[10px] text-text-subtle">
          {flipped
            ? t("option:flashcards.demoClickToSeeQuestion", {
                defaultValue: "Click to see question"
              })
            : t("option:flashcards.demoClickToRevealAnswer", {
                defaultValue: "Click to reveal answer"
              })}
        </div>
      </button>

      <div className="mt-4 flex items-center justify-center">
        <Space>
          <Button
            size="small"
            icon={<ChevronLeft className="h-3.5 w-3.5" />}
            onClick={handlePrev}
            aria-label={t("option:flashcards.demoPreviousCard", {
              defaultValue: "Previous card"
            })}
          />
          <Button
            size="small"
            icon={<RotateCcw className="h-3.5 w-3.5" />}
            onClick={handleFlip}
            aria-label="Flip card"
          >
            {t("option:flashcards.demoFlip", {
              defaultValue: "Flip"
            })}
          </Button>
          <Button
            size="small"
            icon={<ChevronRight className="h-3.5 w-3.5" />}
            onClick={handleNext}
            aria-label={t("option:flashcards.demoNextCard", {
              defaultValue: "Next card"
            })}
          />
        </Space>
      </div>

      <p className="mt-4 text-center text-[11px] text-text-muted">
        {t("option:flashcards.demoPreviewDescription", {
          defaultValue:
            "This is a preview. Connect a server to create your own decks with spaced repetition."
        })}
      </p>
    </div>
  )
}

export default FlipCardDemo
