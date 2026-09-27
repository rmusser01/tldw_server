import React from "react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PromptStarterCards } from "../PromptStarterCards"
import { BlockEditorPanel } from "../Structured/BlockEditorPanel"
import { BlockListPanel } from "../Structured/BlockListPanel"
import { VariableEditorPanel } from "../Structured/VariableEditorPanel"
import settings from "@/assets/locale/en/settings.json"
import extensionSettings from "@/public/_locales/en/settings.json"

const renderTranslated = async (component: React.ReactElement) => {
  const i18n = createInstance()
  await i18n.init({
    lng: "fr",
    fallbackLng: "en",
    interpolation: { escapeValue: false },
    resources: {
      en: { settings },
      fr: {
        settings: {
          managePrompts: {
            recipe: {
              starterTitle: "Recette structurée",
              starterDescription: "Commencez avec des blocs ordonnés.",
              starterAction: "Créer une recette"
            },
            structured: {
              blockEditor: {
                nameLabel: "Nom du bloc",
                roleLabel: "Rôle du bloc",
                sectionKey: "Clé de section",
                contentLabel: "Contenu du bloc",
                enabledLabel: "Bloc activé",
                templateLabel: "Variables du bloc"
              },
              blockList: {
                edit: "Modifier le bloc {{name}}",
                moveUp: "Monter {{name}}",
                moveDown: "Descendre {{name}}",
                remove: "Supprimer {{name}}",
                number: "Bloc {{index}}",
                disabled: " • désactivé"
              },
              variables: {
                number: "Variable numéro {{index}}",
                unnamed: "variable numéro {{index}}",
                remove: "Supprimer {{name}}",
                nameLabel: "Nom de variable",
                inputTypeLabel: "Type de variable",
                labelLabel: "Libellé de variable",
                maxLengthLabel: "Longueur maximale",
                descriptionLabel: "Description de variable",
                optionsLabel: "Options de variable",
                savedDefaultToggleLabel: "Enregistrer une valeur pour {{name}}",
                savedDefaultLabel: "Valeur enregistrée pour {{name}}",
                requiredLabel: "Variable obligatoire",
                runtimeDescription: "Ces valeurs ne sont pas enregistrées.",
                currentValue: "{{name}} : valeur temporaire",
                currentValueLabel: "Valeur temporaire pour {{name}}"
              }
            }
          }
        }
      }
    }
  })
  return render(<I18nextProvider i18n={i18n}>{component}</I18nextProvider>)
}

const block = {
  id: "task",
  name: "Tâche",
  role: "system" as const,
  content: "Instructions",
  enabled: false,
  order: 0,
  is_template: true
}

describe("prompt recipe localization", () => {
  it("translates the recipe starter card", async () => {
    await renderTranslated(<PromptStarterCards onUse={vi.fn()} />)
    expect(
      screen.getByRole("heading", { name: "Recette structurée" })
    ).toBeVisible()
    expect(screen.getByText("Commencez avec des blocs ordonnés.")).toBeVisible()
    expect(
      screen.getByRole("button", { name: "Créer une recette" })
    ).toBeVisible()
  })

  it("translates block editor accessible names", async () => {
    await renderTranslated(
      <BlockEditorPanel
        block={block}
        onChange={vi.fn()}
        onSectionKeyChange={vi.fn()}
      />
    )
    for (const label of [
      "Nom du bloc",
      "Rôle du bloc",
      "Clé de section",
      "Contenu du bloc",
      "Bloc activé",
      "Variables du bloc"
    ]) {
      expect(screen.getByLabelText(label)).toBeVisible()
    }
  })

  it("interpolates translated block actions, indexes, and disabled state", async () => {
    await renderTranslated(
      <BlockListPanel
        blocks={[block]}
        selectedBlockId="task"
        onSelect={vi.fn()}
        onAddBlock={vi.fn()}
        onMoveBlock={vi.fn()}
        onRemoveBlock={vi.fn()}
        showRole={false}
      />
    )
    for (const name of [
      "Modifier le bloc Tâche",
      "Monter Tâche",
      "Descendre Tâche",
      "Supprimer Tâche"
    ]) {
      expect(screen.getByRole("button", { name })).toBeVisible()
    }
    expect(screen.getByText("Bloc 1 • désactivé")).toBeVisible()
  })

  it.each(["text", "textarea"])(
    "translates declarations and %s preview labels",
    async (input_type) => {
      await renderTranslated(
        <VariableEditorPanel
          variables={[
            {
              name: "audience",
              label: "Public",
              input_type,
              default_value: "Lecteurs"
            }
          ]}
          previewValues={{}}
          runtimeValues={{ audience: "Équipe" }}
          showDeclarationFields
        />
      )
      for (const label of [
        "Nom de variable",
        "Type de variable",
        "Libellé de variable",
        "Longueur maximale",
        "Description de variable",
        "Options de variable",
        "Enregistrer une valeur pour Public",
        "Valeur enregistrée pour Public",
        "Variable obligatoire",
        "Valeur temporaire pour Public"
      ]) {
        expect(screen.getByLabelText(label)).toBeVisible()
      }
      expect(
        screen.getByRole("button", { name: "Supprimer Public" })
      ).toBeVisible()
      expect(screen.getByText("Variable numéro 1")).toBeVisible()
      expect(screen.getByText("Public : valeur temporaire")).toBeVisible()
      expect(
        screen.getByText("Ces valeurs ne sont pas enregistrées.")
      ).toBeVisible()
    }
  )

  it("translates unnamed variable removal with its index", async () => {
    await renderTranslated(
      <VariableEditorPanel variables={[{ name: "" }]} previewValues={{}} />
    )
    expect(
      screen.getByRole("button", { name: "Supprimer variable numéro 1" })
    ).toBeVisible()
  })

  it.each([null, undefined, "", false, 0])(
    "preserves saved-default presence for %s",
    async (default_value) => {
      await renderTranslated(
        <VariableEditorPanel
          variables={[{ name: "audience", default_value }]}
          previewValues={{}}
          showDeclarationFields
        />
      )
      const toggle = screen.getByRole("checkbox", {
        name: "Enregistrer une valeur pour audience"
      })
      if (default_value === null || default_value === undefined) {
        expect(toggle).not.toBeChecked()
        expect(
          screen.queryByLabelText("Valeur enregistrée pour audience")
        ).not.toBeInTheDocument()
      } else {
        expect(toggle).toBeChecked()
        expect(
          screen.getByLabelText("Valeur enregistrée pour audience")
        ).toBeVisible()
      }
    }
  )

  it("ships matching web and extension English editor resources", () => {
    expect(settings.managePrompts.structured).toBeDefined()
    const checkMessages = (value: Record<string, unknown>, prefix: string) => {
      for (const [key, message] of Object.entries(value)) {
        const path = `${prefix}_${key}`
        if (typeof message === "string") {
          expect(
            extensionSettings[path as keyof typeof extensionSettings]?.message,
            path
          ).toBe(message)
        } else {
          checkMessages(message as Record<string, unknown>, path)
        }
      }
    }
    checkMessages(settings.managePrompts.structured, "managePrompts_structured")
    checkMessages(settings.managePrompts.recipe, "managePrompts_recipe")
  })
})
