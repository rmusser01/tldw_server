# tests/test_Chunk_Lib.py
import os
import sys
import unittest

# Add the project root (parent directory of App_Function_Libraries) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
#
print(f"Project root added to sys.path: {project_root}")


from PoC_Version.App_Function_Libraries.Chunk_Lib import (
    improved_chunking_process,
    chunk_text_by_words,
    chunk_text_by_sentences,
    chunk_text_by_paragraphs,
    chunk_ebook_by_chapters,
    semantic_chunking,
    get_chunk_metadata,
)

class TestChunkLib(unittest.TestCase):

    def test_improved_chunking_process(self):
        text = "This is a sample text. It has multiple sentences. Here is another one."
        chunk_options = {'method': 'sentences', 'max_size': 2, 'overlap': 1}
        chunks = improved_chunking_process(text, chunk_options)
        self.assertEqual(len(chunks), 2)
        self.assertIn('text', chunks[0])
        self.assertIn('metadata', chunks[0])

    def test_chunk_text_by_words(self):
        text = "Word " * 50  # 50 words
        chunks = chunk_text_by_words(text, max_words=10, overlap=2)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(len(chunks[0].split()), 10)

    def test_chunk_text_by_sentences(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunk_text_by_sentences(text, max_sentences=2, overlap=1)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Sentence one."))

    def test_chunk_text_by_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text_by_paragraphs(text, max_paragraphs=1, overlap=0)
        self.assertEqual(len(chunks), 3)

    def test_chunk_ebook_by_chapters(self):
        text = "# Chapter 1\nContent of chapter 1.\n# Chapter 2\nContent of chapter 2."
        chunk_options = {'max_size': 1000, 'overlap': 0}
        chunks = chunk_ebook_by_chapters(text, chunk_options)
        self.assertEqual(len(chunks), 2)
        self.assertIn('Chapter 1', chunks[0]['text'])

    def test_semantic_chunking(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = semantic_chunking(text, max_chunk_size=5)
        self.assertTrue(len(chunks) >= 1)

    def test_get_chunk_metadata(self):
        chunk = "Sample chunk text."
        full_text = "This is the full text. Sample chunk text. End of full text."
        metadata = get_chunk_metadata(chunk, full_text)
        self.assertIn('start_index', metadata)
        self.assertIn('relative_position', metadata)

if __name__ == '__main__':
    unittest.main()
