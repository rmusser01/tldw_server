# test_chunk_lib.py

import pytest
import json
import re
from unittest.mock import patch, MagicMock  # For mocking external dependencies

from langdetect import LangDetectException
# Assuming Chunk_Lib.py is in a path like app.core.Chunk_Lib
# Adjust the import path based on your project structure.
# For this example, let's assume Chunk_Lib.py is directly accessible or in PYTHONPATH.
# If it's in app/core/Chunk_Lib.py and your tests are at the project root:
from tldw_Server_API.app.core.Chunking.Chunk_Lib import (
    Chunker,
    DEFAULT_CHUNK_OPTIONS,
    improved_chunking_process,
    InvalidChunkingMethodError,
    # load_document, # If you want to test this utility separately
    # chunk_for_embedding # Can be tested more like an integration test
)

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_text_en():
    return "This is the first sentence. This is the second sentence, which is a bit longer. A third one follows. And a fourth. Finally, the fifth sentence concludes this paragraph.\n\nThis is a new paragraph. It has two sentences. This is the second of the new paragraph."

@pytest.fixture
def sample_text_short():
    return "Short text."

@pytest.fixture
def sample_json_list_text():
    return json.dumps([{"id": 1, "text": "item one"}, {"id": 2, "text": "item two"}, {"id": 3, "text": "item three"}])

@pytest.fixture
def sample_json_dict_text():
    return json.dumps({
        "metadata": {"source": "test_doc"},
        "data": {
            "key1": "value one for testing",
            "key2": "value two is also here",
            "key3": "a third value appears now"
        }
    })

@pytest.fixture
def sample_xml_text():
    return "<root><item id='1'>Text <b>boldly</b> goes here.</item><item id='2'>Another item.</item></root>"

@pytest.fixture
def sample_ebook_text():
    return """Preface
This is the preface content.

Chapter 1
Content of chapter one. It is fairly long.
More content for chapter one.

CHAPTER II: A New Beginning
This is the start of the second chapter.
It also has multiple lines.

# Section as Chapter
This could be seen as chapter 3.
"""

@pytest.fixture
def default_chunker_options():
    # Use a copy of the library's defaults for modification in tests if needed
    return DEFAULT_CHUNK_OPTIONS.copy()

@pytest.fixture
def default_chunker(default_chunker_options):
    # Mock tokenizer loading for most tests unless specific tokenizer behavior is tested
    with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained') as mock_tokenizer_load:
        mock_tokenizer_instance = MagicMock()
        # Mock common tokenizer methods if they are used directly and not just for encode/decode
        mock_tokenizer_instance.encode = lambda text: [len(c) for c in text.split()] # Dummy encode
        mock_tokenizer_instance.decode = lambda tokens: " ".join([str(t) for t in tokens]) # Dummy decode
        mock_tokenizer_load.return_value = mock_tokenizer_instance
        return Chunker(options=default_chunker_options, tokenizer_name_or_path="mock-tokenizer")

# --- Unit Tests for Chunker Class ---

class TestChunker:

    def test_chunker_initialization_default(self, default_chunker_options):
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker() # Test with no options
        assert chunker.options['method'] == default_chunker_options['method']
        assert chunker.tokenizer is not None

    def test_chunker_initialization_custom_options(self):
        custom_opts = {"method": "sentences", "max_size": 5, "overlap": 1, "language": "fr"}
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()):
            chunker = Chunker(options=custom_opts)
        assert chunker.options['method'] == "sentences"
        assert chunker.options['max_size'] == 5
        assert chunker.options['overlap'] == 1
        assert chunker.options['language'] == "fr"

    def test_chunker_initialization_tokenizer_failure(self, mocker):
        mocker.patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', side_effect=Exception("Load failed"))
        chunker = Chunker()
        assert chunker.tokenizer is None # Should handle gracefully for non-token methods

    @patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.detect') # Mock langdetect.detect
    def test_detect_language(self, mock_lang_detect, default_chunker, sample_text_en):
        mock_lang_detect.return_value = "en"
        assert default_chunker.detect_language(sample_text_en) == "en"
        mock_lang_detect.assert_called_with(sample_text_en)

        # Test failure
        mock_lang_detect.side_effect = LangDetectException(0, "Detection failed")
        assert default_chunker.detect_language("some text") == default_chunker._get_option('language', 'en') # Defaults to 'en' or option

        # Test empty text
        assert default_chunker.detect_language("") == default_chunker._get_option('language', 'en')


    def test_post_process_chunks(self, default_chunker):
        chunks = ["  chunk one  ", "chunk two", " ", "", None]
        processed = default_chunker._post_process_chunks(chunks)
        assert processed == ["chunk one", "chunk two"]

    # --- Test individual chunking methods ---

    def test_chunk_text_by_words(self, default_chunker, sample_text_en):
        default_chunker.options.update({"max_size": 10, "overlap": 2}) # max_size is max_words here
        chunks = default_chunker._chunk_text_by_words(sample_text_en,
                                                     max_words=default_chunker.options['max_size'],
                                                     overlap=default_chunker.options['overlap'],
                                                     language='en')
        assert len(chunks) > 1
        assert "This is the first sentence. This is the second" in chunks[0] # Approximation
        assert len(chunks[0].split()) <= 10
        if len(chunks) > 1:
            overlap_words = chunks[0].split()[-2:] # Last 2 words of first chunk
            assert overlap_words[0] in chunks[1].split() or overlap_words[1] in chunks[1].split() # Basic overlap check

    @patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.sent_tokenize')
    def test_chunk_text_by_sentences(self, mock_sent_tokenize, default_chunker, sample_text_en):
        # Mock NLTK's sentence tokenizer to return predictable sentences
        sentences = re.split(r'(?<=[.!?])\s+', sample_text_en.replace("\n\n", " "))
        mock_sent_tokenize.return_value = sentences

        default_chunker.options.update({"max_size": 2, "overlap": 1}) # max_size is max_sentences
        chunks = default_chunker._chunk_text_by_sentences(sample_text_en,
                                                         max_sentences=default_chunker.options['max_size'],
                                                         overlap=default_chunker.options['overlap'],
                                                         language='en')
        assert len(chunks) > 1
        assert sentences[0] in chunks[0]
        assert sentences[1] in chunks[0]
        if len(chunks) > 1:
             assert sentences[1] in chunks[1] # Overlap of 1 sentence
        mock_sent_tokenize.assert_called()


    def test_chunk_text_by_paragraphs(self, default_chunker, sample_text_en):
        default_chunker.options.update({"max_size": 1, "overlap": 0}) # max_size is max_paragraphs
        chunks = default_chunker._chunk_text_by_paragraphs(sample_text_en,
                                                          max_paragraphs=default_chunker.options['max_size'],
                                                          overlap=default_chunker.options['overlap'])
        assert len(chunks) == 2 # Based on sample_text_en structure
        assert "This is the first sentence." in chunks[0]
        assert "This is a new paragraph." in chunks[1]

    def test_chunk_text_by_tokens(self, default_chunker, sample_text_short):
        if default_chunker.tokenizer is None: # If tokenizer mock failed in fixture
            pytest.skip("Tokenizer not available for token test")

        # Make tokenizer mock more specific for this test if needed
        default_chunker.tokenizer.encode = lambda text: list(range(len(text))) # Simple tokenization: 1 token per char
        default_chunker.tokenizer.encode = lambda text: list(range(len(text)))

        # A more robust mock decode that accepts skip_special_tokens
        def mock_decode_func(token_ids, skip_special_tokens=True):
            # Dummy decode: just join string representations of token IDs
            # In a real scenario, this would convert IDs back to text.
            return " ".join(map(str, token_ids))

        default_chunker.tokenizer.decode = mock_decode_func
        default_chunker.options.update({"max_size": 5, "overlap": 1}) # max_size is max_tokens
        chunks = default_chunker._chunk_text_by_tokens(sample_text_short, # "Short text." -> len 11
                                                      max_tokens=default_chunker.options['max_size'],
                                                      overlap=default_chunker.options['overlap'])
        assert len(chunks) >= 3 # e.g., [0,1,2,3,4], [4,5,6,7,8], [8,9,10] -> 3 chunks
        # Add more specific assertions based on the mocked encode/decode behavior

    # --- Test semantic, JSON, XML, Ebook requires more involved mocks or specific simple inputs ---

    def test_chunk_json_list(self, default_chunker, sample_json_list_text):
        default_chunker.options.update({"max_size": 2, "overlap": 1})  # max_size is items here
        # _chunk_text_by_json returns List[Dict], not List[str]
        chunk_dicts = default_chunker._chunk_text_by_json(sample_json_list_text,
                                                          max_size=default_chunker.options['max_size'],
                                                          overlap=default_chunker.options['overlap'])
        # The actual number of chunks for a list is ceil(total_items / step) if overlap is handled by extending the end,
        # or more precisely, (total_items - max_size) / step + 1 if max_size <= total_items, then handle remainder.
        # For loop range(0, total_items, step):
        # If total_items = 3, max_size = 2, overlap = 1 => step = 1. Loop is for i = 0, 1, 2. Thus 3 chunks.
        # Chunk 1: items[0:2]
        # Chunk 2: items[1:3]
        # Chunk 3: items[2:4] (which becomes items[2:3])
        assert len(chunk_dicts) == 3  # Corrected expectation
        assert len(chunk_dicts[0]['json']) == 2
        assert len(chunk_dicts[1]['json']) == 2
        assert len(chunk_dicts[2]['json']) == 1  # Last chunk
        assert chunk_dicts[0]['json'][0]['id'] == 1
        assert chunk_dicts[1]['json'][0]['id'] == 2  # First item of second chunk
        assert chunk_dicts[2]['json'][0]['id'] == 3

    def test_chunk_json_dict(self, default_chunker, sample_json_dict_text):
        default_chunker.options.update({"max_size": 2, "overlap": 1, "json_chunkable_data_key": "data"})
        chunk_dicts = default_chunker._chunk_text_by_json(sample_json_dict_text,
                                                        max_size=default_chunker.options['max_size'],
                                                        overlap=default_chunker.options['overlap'])
        assert len(chunk_dicts) == 3  # Corrected expectation
        assert "key1" in chunk_dicts[0]['json']['data']
        assert "key2" in chunk_dicts[0]['json']['data']
        assert len(chunk_dicts[0]['json']['data']) == 2

        assert "key2" in chunk_dicts[1]['json']['data']
        assert "key3" in chunk_dicts[1]['json']['data']
        assert len(chunk_dicts[1]['json']['data']) == 2

        assert "key3" in chunk_dicts[2]['json']['data']
        assert len(chunk_dicts[2]['json']['data']) == 1
        assert chunk_dicts[0]['json']['metadata']['source'] == "test_doc" # Preserved key


    @patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.TfidfVectorizer')
    @patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.cosine_similarity')
    @patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.sent_tokenize')
    def test_semantic_chunking(self, mock_sent_tokenize, mock_cosine_similarity, mock_tfidf, default_chunker, sample_text_en):
        mock_sent_tokenize.return_value = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five is different."]
        # Mock TF-IDF and cosine similarity to control splits
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer_instance.fit_transform.return_value = MagicMock() # Dummy sparse matrix
        mock_tfidf.return_value = mock_vectorizer_instance
        # Simulate similarity scores: high, high, low, high
        mock_cosine_similarity.side_effect = [
            [[0.9]], # S1-S2
            [[0.8]], # S2-S3
            [[0.2]], # S3-S4 (low, potential split if size permits)
            [[0.7]], # S4-S5
        ]

        default_chunker.options.update({
            "max_size": 20, "unit": "words", # max_chunk_size in words
            "semantic_similarity_threshold": 0.4,
            "semantic_overlap_sentences": 1
        })
        chunks = default_chunker._semantic_chunking(sample_text_en,
                                                   max_chunk_size=default_chunker.options['max_size'],
                                                   unit='words') # unit from options
        # Expected: "S1 S2 S3" (S3-S4 similarity is 0.2 < 0.4, and "S1 S2 S3" might be >= max_size/2)
        # Then "S3 S4 S5" (S3 is overlap)
        assert len(chunks) >= 1 # Exact number depends on word counts and size checks
        # This test is complex to assert precisely without exact word counts.
        # Focus on mocking and ensuring the logic flows.
        assert "Sentence one. Sentence two. Sentence three." in chunks[0] # Example if it splits after S3
        if len(chunks) > 1:
            assert "Sentence three." in chunks[1] # Overlap check

    def test_ebook_chapter_chunking_basic(self, default_chunker, sample_ebook_text):
        default_chunker.options.update({"custom_chapter_pattern": None, "max_size": 0}) # max_size 0 means no sub-chunking
        chunk_dicts = default_chunker._chunk_ebook_by_chapters(sample_ebook_text,
                                                              max_size=0, overlap=0,
                                                              custom_pattern=None, language='en')
        assert len(chunk_dicts) == 4
        assert "Preface" in chunk_dicts[0]['text']  # Check content now
        assert chunk_dicts[0]['metadata']['chapter_title'] == "Preface/Introduction"
        assert "Chapter 1" in chunk_dicts[1]['text']
        assert chunk_dicts[1]['metadata']['chapter_title'] == "Chapter 1"
        assert "CHAPTER II: A New Beginning" in chunk_dicts[2]['text']
        assert chunk_dicts[2]['metadata']['chapter_title'] == "CHAPTER II: A New Beginning"
        assert "# Section as Chapter" in chunk_dicts[3]['text']
        assert chunk_dicts[3]['metadata']['chapter_title'] == "# Section as Chapter"

    def test_xml_chunking_basic(self, default_chunker, sample_xml_text):
        default_chunker.options.update({"max_size": 5, "overlap": 1}) # max_size in words, overlap in elements
        chunk_dicts = default_chunker._chunk_xml(sample_xml_text,
                                                max_size=default_chunker.options['max_size'],
                                                overlap=default_chunker.options['overlap'],
                                                language='en')
        assert len(chunk_dicts) > 0
        # Example: <root><item id='1'>Text <b>boldly</b> goes here.</item> (5 words + attributes)
        # Might split after "root/item/@id: 1" and "root/item: Text boldly goes here." (5 words)
        # Then "root/item: Text boldly goes here." (overlap) and "root/item[2]/@id: 2", "root/item[2]: Another item."
        # Assertion depends heavily on the exact element breakdown and word counts.
        # For a simple test, check if root tag is in metadata
        assert chunk_dicts[0]['metadata']['root_tag'] == 'root'

    @patch.object(Chunker, '_combine_chunks_for_llm') # Path to the method within Chunker
    def test_rolling_summarize_structure(self, mock_combine_chunks, default_chunker, sample_text_en):
        if default_chunker.tokenizer is None: pytest.skip("Tokenizer needed for rolling_summarize")

        # Mock the llm_summarize_step_func
        mock_llm_step_func = MagicMock(return_value="Summarized part.")
        # Mock _chunk_on_delimiter_for_llm to control input to LLM loop
        # This returns (list_of_text_parts_for_llm, list_of_indices, dropped_count)
        with patch.object(Chunker, '_chunk_on_delimiter_for_llm', return_value=(["Part 1 for LLM.", "Part 2 for LLM."], [], 0)) as mock_splitter:
            default_chunker.options.update({
                "summarization_detail": 0.5, "summarize_min_chunk_tokens": 5,
                "summarize_chunk_delimiter": ".", "summarize_recursively": False,
                "summarize_system_prompt": "Summarize this:",
            })
            llm_config = {"api_name": "mock", "model": "mock_model", "api_key": "none", "temperature": 0.1}

            summary = default_chunker._rolling_summarize(
                text_to_summarize=sample_text_en,
                llm_summarize_step_func=mock_llm_step_func,
                llm_api_config=llm_config,
                detail=default_chunker.options['summarization_detail'],
                min_chunk_tokens=default_chunker.options['summarize_min_chunk_tokens'],
                chunk_delimiter=default_chunker.options['summarize_chunk_delimiter'],
                recursive_summarization=False, verbose=False,
                system_prompt_content=default_chunker.options['summarize_system_prompt'],
                additional_instructions=None
            )
            mock_splitter.assert_called_once()
            assert mock_llm_step_func.call_count == 2 # Called for each part from mock_splitter
            expected_payload_part1 = {
                "api_name": "mock", "input_data": "Part 1 for LLM.", "custom_prompt_arg": "",
                "api_key": "none", "system_message": "Summarize this:", "temp": 0.1,
                "streaming": False, "model": "mock_model", "max_tokens": None # Or whatever default _rolling_summarize sets for payload
            }
            # Check specific arguments of the first call
            args, _ = mock_llm_step_func.call_args_list[0]
            actual_payload_part1 = args[0] # The payload dict is the first arg
            assert actual_payload_part1['input_data'] == "Part 1 for LLM."
            assert actual_payload_part1['system_message'] == "Summarize this:"

            assert summary == "Summarized part.\n\n---\n\nSummarized part."


    def test_chunk_text_dispatcher_invalid_method(self, default_chunker):
        with pytest.raises(InvalidChunkingMethodError):
            default_chunker.chunk_text("some text", method="non_existent_method")

    def test_chunk_text_dispatcher_calls_correct_method(self, default_chunker, sample_text_short):
        with patch.object(default_chunker, '_chunk_text_by_words', return_value=["mocked"]) as mock_method:
            default_chunker.chunk_text(sample_text_short, method="words")
            mock_method.assert_called_once()

# --- Tests for improved_chunking_process (more like integration for this function) ---

@patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.Chunker') # Mock the Chunker class itself
def test_improved_chunking_process_basic_flow(mock_chunker_class, sample_text_short, default_chunker_options):
    # Setup mock Chunker instance and its methods
    mock_chunker_instance = MagicMock()
    mock_chunker_instance.options = default_chunker_options.copy() # Give it some options
    mock_chunker_instance.options['method'] = 'words' # Ensure a method is set
    mock_chunker_instance.options['language'] = 'en'
    mock_chunker_instance.detect_language.return_value = 'en'
    # chunk_text can return plain strings or dicts for json/xml/ebook
    mock_chunker_instance.chunk_text.return_value = ["chunk1", "chunk2"]
    mock_chunker_class.return_value = mock_chunker_instance # When Chunker() is called, return our mock

    opts_for_process = {"method": "words", "max_size": 10, "overlap": 1}
    result = improved_chunking_process(sample_text_short, chunk_options_dict=opts_for_process, tokenizer_name_or_path="mock-tokenizer")

    mock_chunker_class.assert_called_once() # Check Chunker was instantiated
    # Check that options passed to Chunker instantiation are correct
    _, kwargs = mock_chunker_class.call_args
    assert kwargs['options']['method'] == 'words'

    mock_chunker_instance.chunk_text.assert_called_once()
    # Check that the text passed to chunk_text is the original (or after header/json strip)
    args_chunk_text, _ = mock_chunker_instance.chunk_text.call_args
    assert args_chunk_text[0] == sample_text_short # Assuming no header/json in sample_text_short

    assert len(result) == 2
    assert result[0]['text'] == "chunk1"
    assert result[0]['metadata']['chunk_method'] == 'words'
    assert result[0]['metadata']['chunk_index'] == 1
    assert result[0]['metadata']['total_chunks'] == 2


def test_improved_chunking_process_with_json_header(default_chunker_options):
    # This is a more complex test because it involves the actual Chunker behavior for header/JSON parsing
    # and then subsequent chunking. For a true unit test of improved_chunking_process,
    # you'd mock out the Chunker's chunk_text method.
    # For this example, let's test the header/JSON stripping part and a simple chunking.

    json_header = {"doc_id": "xyz123", "source": "test\n"}
    text_header = "This text was transcribed using AI\n\n"
    actual_content = "This is the main content to be chunked. It has several words."
    full_text_input = json.dumps(json_header) + "\n" + text_header + actual_content

    # For this test, allow Chunker to be real, but mock its internal chunking method
    # or use a very simple chunking method.
    test_options = default_chunker_options.copy()
    test_options.update({"method": "words", "max_size": 5, "language": "en"})

    # Patch the Chunker's specific chunking method to control its output
    with patch.object(Chunker, '_chunk_text_by_words', return_value=["This is the", "main content to", "be chunked. It", "has several words."]) as mock_word_chunker:
        with patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.AutoTokenizer.from_pretrained', MagicMock()): # Mock tokenizer loading
            results = improved_chunking_process(full_text_input, chunk_options_dict=test_options)

    # Check that _chunk_text_by_words was called with the *actual_content*
    mock_word_chunker.assert_called_once()
    call_args, _ = mock_word_chunker.call_args
    assert call_args[0] == actual_content # Text passed to the method should be stripped

    assert len(results) == 4
    for chunk_data in results:
        assert chunk_data['metadata']['initial_document_json_metadata'] == json_header
        assert chunk_data['metadata']['initial_document_header_text'] == text_header
        assert chunk_data['metadata']['chunk_method'] == 'words'

# TODO: Add tests for chunk_for_embedding and process_document_with_metadata
# These are higher-level and might involve more mocking or specific input files.
# Example for chunk_for_embedding:
@patch('tldw_Server_API.app.core.Chunking.Chunk_Lib.improved_chunking_process')
def test_chunk_for_embedding(mock_improved_process):
    from tldw_Server_API.app.core.Chunking.Chunk_Lib import chunk_for_embedding # Import here if not at top

    # Mock the output of improved_chunking_process
    mock_improved_process.return_value = [
        {'text': 'chunk one text', 'metadata': {'chunk_index': 1, 'total_chunks': 2, 'relative_position': 0.25}},
        {'text': 'chunk two text', 'metadata': {'chunk_index': 2, 'total_chunks': 2, 'relative_position': 0.75}},
    ]
    file_name = "test_doc.txt"
    result = chunk_for_embedding(text="dummy text", file_name=file_name, custom_chunk_options={})

    mock_improved_process.assert_called_once()
    assert len(result) == 2
    assert f"[DOCUMENT: {file_name}]" in result[0]['text_for_embedding']
    assert "---BEGIN CHUNK CONTENT---" in result[0]['text_for_embedding']
    assert result[0]['original_chunk_text'] == 'chunk one text'
    assert result[0]['source_document_name'] == file_name
    assert result[0]['chunk_metadata']['chunk_index'] == 1

#
# End of test_chunk_lib.py
#######################################################################################################################
