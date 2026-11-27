import re
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_quick_ingest_document_and_verify_media(configured_page, sample_text_path):
    page = configured_page
    page.set_default_timeout(120_000)

    page.goto("/media")
    trigger = page.get_by_role("button", name=re.compile(r"^(open )?quick ingest$", re.I))
    first_ingest = page.get_by_test_id("first-ingest-tutorial").get_by_role(
        "button", name=re.compile(r"^ingest$", re.I)
    )
    trigger.or_(first_ingest).first.click()

    dialog = page.get_by_role("dialog", name="Quick Ingest")
    dialog.get_by_role("button", name="Files", exact=True).click()
    dialog.get_by_test_id("qi-file-input").set_input_files(sample_text_path)
    dialog.get_by_role("button", name="Change settings").click()
    dialog.get_by_role("combobox", name="Processing preset").select_option("quick")
    dialog.get_by_role("button", name="Save 1 source", exact=True).click()
    dialog.get_by_role("button", name="Open in Media", exact=True).click()

    page.wait_for_url(re.compile(r"/media\?id=\d+"))
    search = page.get_by_placeholder("Search media (title/content)")
    search.fill("Hello world from E2E document processing.")
    result_button = page.get_by_role("button", name=f"Select media: {Path(sample_text_path).stem}")
    result_button.wait_for()
    result_button.click()
    page.get_by_text("Hello world from E2E document processing.").wait_for()

    page.get_by_role("button", name="Delete item").click()
    delete_dialog = page.get_by_role("dialog")
    delete_dialog.get_by_role("button", name="Delete").wait_for()
    delete_dialog.get_by_role("button", name="Delete").click()
