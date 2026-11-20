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


def test_chapter_plan_names_a_heading_only_title_chain_after_the_chapter():
    book = (
        "# Book title\n\n## Chapter 1: First lesson\n\n" + "First lesson. " * 10
        + "\n\n## Chapter 2: Last lesson\n\n" + "Last lesson. " * 10
    )
    _, sections = plan_media_sections(book, 80)

    assert [section.title for section in sections] == [  # nosec B101
        "Chapter 1: First lesson", "Chapter 2: Last lesson",
    ]
    assert sections[0].start == 0  # nosec B101
    assert book[sections[1].start:].startswith("## Chapter 2:")  # nosec B101
