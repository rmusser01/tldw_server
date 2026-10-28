import { z } from "zod";

const card = z
  .object({
    title: z.string().trim().min(1).max(500),
    description: z.string().max(5000).nullable().optional(),
  })
  .strict();
const schema = z
  .object({
    board: z
      .object({
        name: z.string().trim().min(1).max(255),
        description: z.string().max(5000).nullable().optional(),
      })
      .strict(),
    lists: z
      .array(
        z
          .object({
            name: z.string().trim().min(1).max(255),
            cards: z.array(card).max(200),
          })
          .strict(),
      )
      .min(1)
      .max(20),
  })
  .strict()
  .refine((value) => {
    const count = value.lists.reduce(
      (total, list) => total + list.cards.length,
      0,
    );
    return count > 0 && count <= 500;
  });
export type GeneratedBoardDraft = z.infer<typeof schema>;

/** Admit only a complete bounded model draft, never model-supplied database identity. */
export function parseGeneratedBoardDraft(
  response: unknown,
): GeneratedBoardDraft {
  try {
    const completion = z
      .object({
        choices: z
          .array(
            z.object({
              finish_reason: z.enum(["stop"]).nullable().optional(),
              message: z.object({ content: z.string() }),
            }),
          )
          .min(1),
      })
      .parse(response);
    const text = completion.choices[0].message.content.trim();
    if (
      text.length > 128_000 ||
      new TextEncoder().encode(text).length > 128_000
    )
      throw new Error("Draft too large");
    const fenced = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
    return schema.parse(JSON.parse(fenced ? fenced[1] : text));
  } catch {
    throw new Error(
      "Model returned an invalid board draft. Adjust the instructions and generate again.",
    );
  }
}
