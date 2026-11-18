import pytest

from tldw_Server_API.app.core.StudyPacks.media_sections import plan_media_sections, section_passages

pytestmark = pytest.mark.unit


def test_chapter_plan_keeps_preamble_and_coalesces_repeated_epub_headings():
    book = (
        "This is the foreword.\r\n\r\n"
        "Chapter 1\r\n## Chapter 1\r\n\r\n" + "First lesson. " * 12
        + "\r\nChapter 2\r\n\r\n" + "Last lesson. " * 12
    )
    _, sections = plan_media_sections(book, 80)

    assert [item.title for item in sections] == ["Section 1", "Chapter 1", "Chapter 2"]  # nosec B101
    passages = [text for section in sections for _, _, text in section_passages(book, section, 80)]
    assert max(map(len, passages)) <= 80  # nosec B101
    assert "".join("".join(text.split()) for text in passages) == "".join(book.split())  # nosec B101


def test_media_without_headings_uses_paragraph_boundaries():
    first = "This paragraph introduces the subject clearly."
    second = "This paragraph applies the same topic more deeply."
    _, sections = plan_media_sections(f"{first}\n\n{second}", 63)

    assert [item.title for item in sections] == ["Section 1", "Section 2"]  # nosec B101
    assert sections[0].end == len(first) + 2  # nosec B101
