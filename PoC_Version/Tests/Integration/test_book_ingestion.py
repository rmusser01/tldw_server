# test_book_ingestion.py
# Integration test for book ingestion functionality
# Usage: pytest tests/integration/test_book_ingestion.py
# This test assumes that your ingest_text_file function handles the parsing of the book's metadata (title, author, keywords) from the file content. If your function doesn't do this, you might need to adjust the test or the function to handle this.
# The test doesn't check for things like chapter structure or more complex book formatting. Depending on your requirements, you might want to add more detailed checks.
#
import pytest
import os
from PoC_Version.App_Function_Libraries.Books.Book_Ingestion_Lib import ingest_text_file
from PoC_Version.App_Function_Libraries.DB.DB_Manager import db, fetch_item_details


@pytest.fixture
def sample_book_content():
    return """Title: Test Book
Author: Integration Tester
Keywords: test, integration, book

Chapter 1: Introduction

This is a test book for integration testing.
It contains multiple lines and simulates a simple book structure.

Chapter 2: Content

The content of this book is not particularly meaningful.
It's just for testing the book ingestion process.

Chapter 3: Conclusion

This concludes our test book. Thank you for reading!
"""


@pytest.fixture
def sample_book_file(tmp_path, sample_book_content):
    book_file = tmp_path / "test_book.txt"
    with open(book_file, "w") as f:
        f.write(sample_book_content)
    return str(book_file)


def test_book_ingestion_integration(sample_book_file):
    try:
        # Step 1: Ingest the book
        result = ingest_text_file(
            file_path=sample_book_file,
            title="Test Book",
            author="Integration Tester",
            keywords="test,integration,book"
        )
        assert "ingested successfully" in result, f"Book ingestion failed: {result}"

        # Step 2: Verify the book was added to the database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, author FROM Media WHERE title = ?", ("Test Book",))
            book_record = cursor.fetchone()

        assert book_record is not None, "Book not found in database"
        book_id, book_title, book_author = book_record
        assert book_title == "Test Book"
        assert book_author == "Integration Tester"

        # Step 3: Fetch and verify book details
        content, prompt, summary = fetch_item_details(book_id)

        assert "This is a test book for integration testing." in content
        assert "Chapter 1: Introduction" in content
        assert "Chapter 2: Content" in content
        assert "Chapter 3: Conclusion" in content

        # Verify that the content doesn't contain the metadata
        assert "Title: Test Book" not in content
        assert "Author: Integration Tester" not in content
        assert "Keywords: test, integration, book" not in content

        # Step 4: Verify keywords
        cursor.execute("""
            SELECT k.keyword 
            FROM Keywords k
            JOIN MediaKeywords mk ON k.id = mk.keyword_id
            WHERE mk.media_id = ?
        """, (book_id,))
        keywords = [row[0] for row in cursor.fetchall()]
        assert set(keywords) == {"test", "integration", "book"}

        # Additional checks can be added here, such as verifying the summary or prompt if applicable

    except Exception as e:
        pytest.fail(f"Integration test failed with error: {str(e)}")

    finally:
        # Clean up: remove the test book file
        if os.path.exists(sample_book_file):
            os.remove(sample_book_file)

        # Optionally, remove the book from the database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Media WHERE title = ?", ("Test Book",))
            conn.commit()


if __name__ == "__main__":
    pytest.main([__file__])