# Tests/Books/test_Book_Ingestion_tab.py
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tldw')))
#
from App_Function_Libraries.Gradio_UI.Book_Ingestion_tab import import_file_handler

class TestBookIngestionTab(unittest.TestCase):

    @patch('App_Function_Libraries.Gradio_UI.Book_Ingestion_tab.import_epub')
    def test_import_epub_file(self, mock_import_epub):
        # Mock import_epub to return a success message
        mock_import_epub.return_value = "📚 EPUB Imported Successfully:\nEbook 'Test Title' by Test Author imported successfully."

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
            self.assertIn("📚 EPUB Imported Successfully", result)
            mock_import_epub.assert_called_once()

    @patch('App_Function_Libraries.Gradio_UI.Book_Ingestion_tab.process_zip_file')
    def test_import_zip_file(self, mock_process_zip_file):
        # Mock process_zip_file to return a success message
        mock_process_zip_file.return_value = "📦 ZIP Processed Successfully:\nProcessed ZIP file successfully."

        # Create a temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix='.zip') as tmp_zip:
            tmp_zip.write(b"Dummy ZIP content")
            tmp_zip.seek(0)

            # Call the handler function
            result = import_file_handler(
                file=tmp_zip,
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
            self.assertIn("📦 ZIP Processed Successfully", result)
            mock_process_zip_file.assert_called_once()

    @patch('App_Function_Libraries.Gradio_UI.Book_Ingestion_tab.import_epub')
    def test_import_unsupported_file(self, mock_import_epub):
        # No import_epub should be called for unsupported files
        # Create a temporary unsupported file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(b"Dummy PDF content")
            tmp_pdf.seek(0)

            # Call the handler function
            result = import_file_handler(
                file=tmp_pdf,
                title="Test Title",
                author="Test Author",
                keywords="test,pdf",
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
            self.assertIn("file import is not yet supported", result)
            mock_import_epub.assert_not_called()

    def test_import_no_file(self):
        # Call the handler function with no file
        result = import_file_handler(
            file=None,
            title="Test Title",
            author="Test Author",
            keywords="test",
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
        self.assertEqual("No file uploaded.", result)

if __name__ == '__main__':
    unittest.main()