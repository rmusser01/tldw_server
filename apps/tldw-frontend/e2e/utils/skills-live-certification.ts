import type { Expect, Page } from "@playwright/test"

export const SKILLS_CERT_DESCRIPTION = "Skills live certification fixture"
export const SKILLS_CERT_INSTRUCTIONS = "Organize $ARGUMENTS into verified notes."
export const SKILLS_CERT_ARGUMENTS = "bounded certification input"
export const SKILLS_CERT_RENDERED = "Organize bounded certification input into verified notes."

type InitialExpectation = "empty-library-and-trash" | "target-absent"

type SkillsCertificationLifecycleOptions = {
  page: Page
  expect: Expect
  initialExpectation: InitialExpectation
  name: string
  arguments: string
  expectedRenderedPrompt: string
  step: (title: string, body: () => Promise<void>) => Promise<void>
}

const skillsPath = "/api/v1/skills"

const isSkillsListRequest = (url: string, method: string, name: string): boolean => {
  const parsed = new URL(url)
  return (
    method === "GET"
    && parsed.pathname.replace(/\/$/, "") === skillsPath
    && parsed.searchParams.get("q") === name
  )
}

async function submitExactSearch(page: Page, name: string): Promise<void> {
  const search = page.getByPlaceholder("Search skills...")
  const listResponse = page.waitForResponse((response) =>
    isSkillsListRequest(response.url(), response.request().method(), name)
  )

  await search.fill(name)
  await listResponse
}

async function searchForSkill(page: Page, expect: Expect, name: string): Promise<void> {
  await submitExactSearch(page, name)
  await expect(page.getByText(name, { exact: true })).toBeVisible()
}

async function moveSkillToTrash(page: Page, expect: Expect, name: string): Promise<void> {
  await page.getByRole("button", { name: `More actions for ${name}`, exact: true }).click()
  await page.getByRole("menuitem", { name: "Delete", exact: true }).click()

  const dialog = page.getByRole("dialog", { name: `Delete ${name}?`, exact: true })
  await expect(dialog).toBeVisible()
  await dialog.getByRole("button", { name: "Move to Trash", exact: true }).click()
  await expect(dialog).toBeHidden()
}

/** Exercise the shared Skills lifecycle without owning backend or evidence setup. */
export async function runSkillsLiveCertification({
  page,
  expect,
  initialExpectation,
  name,
  arguments: args,
  expectedRenderedPrompt,
  step,
}: SkillsCertificationLifecycleOptions): Promise<void> {
  const skillsView = page.getByRole("radiogroup", { name: "Skills view" })

  await step("1. Verify initial Skills state", async () => {
    await expect(page.getByRole("radio", { name: "Library", exact: true })).toBeChecked()

    if (initialExpectation === "empty-library-and-trash") {
      await expect(page.getByRole("heading", { name: "Start with a reusable skill" })).toBeVisible()
      await expect(page.getByTestId("skills-empty-state")).toBeVisible()
    } else {
      await submitExactSearch(page, name)
      await expect(page.getByText(name, { exact: true })).toHaveCount(0)
    }

    await skillsView.getByText("Trash", { exact: true }).click()
    await expect(page.getByRole("radio", { name: "Trash", exact: true })).toBeChecked()

    if (initialExpectation === "empty-library-and-trash") {
      await expect(page.getByRole("heading", { name: "Trash is empty" })).toBeVisible()
    } else {
      await expect(page.getByText(name, { exact: true })).toHaveCount(0)
    }

    await skillsView.getByText("Library", { exact: true }).click()
    await expect(page.getByRole("radio", { name: "Library", exact: true })).toBeChecked()
    if (initialExpectation === "target-absent") {
      await page.getByPlaceholder("Search skills...").clear()
    }
  })

  await step("2. Create the certification skill", async () => {
    await page.getByRole("button", { name: "New Skill", exact: true }).click()
    const drawer = page.getByRole("dialog", { name: /^New Skill:/ })
    await expect(drawer).toBeVisible()
    await drawer.getByLabel("Name", { exact: true }).fill(name)
    await drawer.getByLabel("Description", { exact: true }).fill(SKILLS_CERT_DESCRIPTION)
    await drawer.getByLabel("Instructions", { exact: true }).fill(SKILLS_CERT_INSTRUCTIONS)
    await drawer.getByRole("button", { name: "Save", exact: true }).click()
  })

  await step("3. Confirm skill creation", async () => {
    await expect(page.getByText("Skill created", { exact: true })).toBeVisible()
    await expect(page.getByText(name, { exact: true })).toBeVisible()
  })

  await step("4. Search for the exact skill", async () => {
    await searchForSkill(page, expect, name)
  })

  await step("5. Render the exact prompt", async () => {
    await page.getByRole("button", { name: `Test run ${name}`, exact: true }).click()
    const dialog = page.getByRole("dialog", { name: `Test run: ${name}`, exact: true })
    await expect(dialog).toBeVisible()
    await dialog.getByPlaceholder("Enter test arguments...").fill(args)

    const executeResponse = page.waitForResponse((response) => {
      const parsed = new URL(response.url())
      return (
        response.request().method() === "POST"
        && parsed.pathname.replace(/\/$/, "") === `${skillsPath}/${encodeURIComponent(name)}/execute`
      )
    })
    await dialog.getByRole("button", { name: "Render prompt only", exact: true }).click()

    const response = await executeResponse
    await expect(response.status()).toBe(200)
    expect(response.request().postDataJSON()).toMatchObject({ args, dry_run: true })
    expect(await response.json()).toMatchObject({
      dry_run: true,
      rendered_prompt: expectedRenderedPrompt,
    })
    await expect(dialog.getByLabel("Rendered Prompt", { exact: true })).toHaveValue(
      expectedRenderedPrompt
    )
  })

  await step("6. Reload and confirm persistence", async () => {
    await page.keyboard.press("Escape")
    await expect(page.getByRole("dialog", { name: `Test run: ${name}`, exact: true })).toBeHidden()
    await page.reload({ waitUntil: "domcontentloaded" })
    await searchForSkill(page, expect, name)
  })

  await step("7. Move the skill to Trash", async () => {
    await moveSkillToTrash(page, expect, name)
  })

  await step("8. Restore the skill from Trash", async () => {
    await skillsView.getByText("Trash", { exact: true }).click()
    await expect(page.getByRole("radio", { name: "Trash", exact: true })).toBeChecked()
    await expect(page.getByText(name, { exact: true })).toBeVisible()
    await page.getByRole("button", { name: `Restore ${name}`, exact: true }).click()

    await skillsView.getByText("Library", { exact: true }).click()
    await expect(page.getByRole("radio", { name: "Library", exact: true })).toBeChecked()
    await expect(page.getByText(name, { exact: true })).toBeVisible()
  })

  await step("9. Move the restored skill to Trash", async () => {
    await moveSkillToTrash(page, expect, name)
  })

  await step("10. Permanently delete the certification skill", async () => {
    await skillsView.getByText("Trash", { exact: true }).click()
    await expect(page.getByRole("radio", { name: "Trash", exact: true })).toBeChecked()
    await page.getByRole("button", { name: `Permanently delete ${name}`, exact: true }).click()

    const dialog = page.getByRole("dialog", {
      name: `Permanently delete ${name}?`,
      exact: true,
    })
    await expect(dialog).toBeVisible()
    await dialog.getByRole("button", { name: "Delete permanently", exact: true }).click()
    await expect(page.getByRole("heading", { name: "Trash is empty" })).toBeVisible()
  })
}
