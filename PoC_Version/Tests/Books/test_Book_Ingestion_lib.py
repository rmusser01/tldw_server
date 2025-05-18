# tests/test_Book_Ingestion_Lib.py
import sys
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import zipfile
# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tldw')))

from App_Function_Libraries.Books.Book_Ingestion_Lib import (
    import_file_handler,
    import_epub,
    process_zip_file
)

class TestBookIngestionTab(unittest.TestCase):

    @patch('Book_Ingestion_tab.import_epub')
    def test_import_epub_file(self, mock_import_epub):
        # Mock import_epub to return a success message
        mock_import_epub.return_value = "Ebook 'Test Title' by Test Author imported successfully."

        # Create a temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix='.epub') as tmp_epub:
            tmp_epub.write(b"Dummy EPUB content")
            tmp_epub.seek(0)

            # Call the handler function
            result = import_file_handler(
                file=tmp_epub,
                title="Test Title",
                author="Test Author",
                keywords="test,epub",
                system_prompt="System prompt",
                custom_prompt="Custom prompt",
                auto_summarize=False,
                api_name=None,
                api_key=None,
                max_chunk_size=500,
                chunk_overlap=200,
                custom_chapter_pattern=None
            )

            # Assertions
            self.assertIn("EPUB Imported Successfully", result)
            mock_import_epub.assert_called_once()

    @patch('Book_Ingestion_Lib.epub_to_markdown')
    @patch('Book_Ingestion_Lib.extract_epub_metadata')
    @patch('Book_Ingestion_Lib.add_media_to_database')
    @patch('Book_Ingestion_Lib.chunk_ebook_by_chapters')
    def test_import_epub_missing_metadata(
        self, mock_chunk_ebook_by_chapters, mock_add_media_to_database,
        mock_extract_epub_metadata, mock_epub_to_markdown
    ):
        # Mock dependencies
        mock_epub_to_markdown.return_value = "# Sample Title\n\nSample content."
        mock_extract_epub_metadata.return_value = ("Extracted Title", "Extracted Author")
        mock_chunk_ebook_by_chapters.return_value = [
            {'text': 'Chapter content', 'metadata': {}},
        ]
        mock_add_media_to_database.return_value = "Success"

        # Create a temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix='.epub') as tmp_epub:
            tmp_epub.write(b"Dummy EPUB content")
            tmp_epub.seek(0)

            # Call the function without title and author
            result = import_epub(
                file_path=tmp_epub.name,
                title=None,
                author=None,
                keywords=None,
                custom_prompt=None,
                system_prompt=None,
                summary=None,
                auto_summarize=False,
                api_name=None,
                api_key=None,
                chunk_options=None,
                custom_chapter_pattern=None
            )

            # Assertions
            self.assertIn("imported successfully", result.lower())
            mock_extract_epub_metadata.assert_called_once()
            mock_add_media_to_database.assert_called_once()

    def test_import_epub_invalid_file(self):
        # Call the function with an invalid file path
        result = import_epub(
            file_path="non_existent_file.epub",
            title=None,
            author=None,
            keywords=None,
            custom_prompt=None,
            system_prompt=None,
            summary=None,
            auto_summarize=False,
            api_name=None,
            api_key=None,
            chunk_options=None,
            custom_chapter_pattern=None
        )

        # Assertions
        self.assertIn("error importing ebook", result.lower())

    @patch('Book_Ingestion_Lib.epub_to_markdown')
    @patch('Book_Ingestion_Lib.perform_summarization')
    @patch('Book_Ingestion_Lib.chunk_ebook_by_chapters')
    @patch('Book_Ingestion_Lib.add_media_to_database')
    def test_import_epub_with_auto_summarize(
        self, mock_add_media_to_database, mock_chunk_ebook_by_chapters,
        mock_perform_summarization, mock_epub_to_markdown
    ):
        # Mock dependencies
        mock_epub_to_markdown.return_value = "# Sample Title\n\nSample content."
        mock_chunk_ebook_by_chapters.return_value = [
            {'text': 'Chapter 1 content', 'metadata': {}},
        ]
        mock_perform_summarization.return_value = "Summarized content"
        mock_add_media_to_database.return_value = "Success"

        # Create a temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix='.epub') as tmp_epub:
            tmp_epub.write(b"Dummy EPUB content")
            tmp_epub.seek(0)

            # Call the function with auto_summarize=True
            result = import_epub(
                file_path=tmp_epub.name,
                title="Test Title",
                author="Test Author",
                keywords="test,epub",
                custom_prompt="Custom prompt",
                system_prompt="System prompt",
                summary=None,
                auto_summarize=True,
                api_name="OpenAI",
                api_key="fake_api_key",
                chunk_options=None,
                custom_chapter_pattern=None
            )

            # Assertions
            self.assertIn("imported successfully", result.lower())
            mock_perform_summarization.assert_called()
            mock_add_media_to_database.assert_called_once()

    @patch('Book_Ingestion_Lib.import_epub')
    def test_process_zip_file(self, mock_import_epub):
        # Mock import_epub to return success message
        mock_import_epub.return_value = "Ebook imported successfully."

        # Create a temporary ZIP file containing dummy EPUB files
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_file_path = os.path.join(temp_dir, 'test.epub')
            with open(epub_file_path, 'wb') as epub_file:
                epub_file.write(b"Dummy EPUB content")

            zip_file_path = os.path.join(temp_dir, 'test.zip')
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                zipf.write(epub_file_path, arcname='test.epub')

            # Mock the zip_file object
            class MockZipFile:
                def __init__(self, name):
                    self.name = name

            mock_zip_file = MockZipFile(zip_file_path)

            # Call the function under test
            result = process_zip_file(
                zip_file=mock_zip_file,
                title="Zip Title",
                author="Zip Author",
                keywords="zip,epub",
                custom_prompt=None,
                system_prompt=None,
                summary=None,
                auto_summarize=False,
                api_name=None,
                api_key=None,
                chunk_options=None
            )

            # Assertions
            self.assertIn("Ebook imported successfully.", result)
            mock_import_epub.assert_called_once()

    # Additional tests for other functions...



if __name__ == '__main__':
    unittest.main()
