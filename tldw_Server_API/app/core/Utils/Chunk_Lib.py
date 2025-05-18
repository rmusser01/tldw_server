# Chunk_Lib.py
#########################################
# Chunking Library
# This library is used to perform chunking of input files.
# Currently, uses naive approaches. Nothing fancy.
#
####
# Import necessary libraries
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
import xml.etree.ElementTree as ET
#
# Import 3rd party
from tqdm import tqdm
from langdetect import detect, LangDetectException # Import specific exception
from transformers import AutoTokenizer, PreTrainedTokenizerBase # Using AutoTokenizer for flexibility
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#
# Import Local
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.config import load_and_log_configs
#
#######################################################################################################################
# Custom Exceptions
class ChunkingError(Exception):
    """Base exception for chunking errors."""
    pass

class InvalidChunkingMethodError(ChunkingError):
    """Raised when an invalid chunking method is specified."""
    pass

class InvalidInputError(ChunkingError):
    """Raised for invalid input data, e.g., bad JSON."""
    pass

class LanguageDetectionError(ChunkingError):
    """Raised when language detection fails critically."""
    pass

#######################################################################################################################
# Config Settings & NLTK
#

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
        try:
            nltk.download('punkt')
            logging.info("'punkt' downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download 'punkt': {e}")
            # Depending on how critical this is, you might raise an error or just warn
            # For now, we'll let it proceed, and sent_tokenize will fail later if needed.
ensure_nltk_data()

# Load configuration (used for default options)
# We keep this here for now to get default values, but the Chunker class will manage its own options.
_global_config = load_and_log_configs()
_default_chunk_options_from_config = {
    'method': _global_config['chunking_config'].get('chunking_method', 'words'),
    'max_size': int(_global_config['chunking_config'].get('chunk_max_size', 400)),
    'overlap': int(_global_config['chunking_config'].get('chunk_overlap', 200)),
    'adaptive': _global_config['chunking_config'].get('adaptive_chunking', False),
    'multi_level': _global_config['chunking_config'].get('multi_level', False),
    'language': _global_config['chunking_config'].get('chunk_language', None), # Can be None
    'custom_chapter_pattern': _global_config['chunking_config'].get('custom_chapter_pattern', None),
    'semantic_similarity_threshold': float(_global_config['chunking_config'].get('semantic_similarity_threshold', 0.5)),
    'semantic_overlap_sentences': int(_global_config['chunking_config'].get('semantic_overlap_sentences', 3)),
    'base_adaptive_chunk_size': int(_global_config['chunking_config'].get('base_adaptive_chunk_size', 1000)),
    'min_adaptive_chunk_size': int(_global_config['chunking_config'].get('min_adaptive_chunk_size', 500)),
    'max_adaptive_chunk_size': int(_global_config['chunking_config'].get('max_adaptive_chunk_size', 2000)),
    'tokenizer_name_or_path': _global_config['chunking_config'].get('tokenizer_name_or_path', "gpt2"),  # Add this
    'summarization_detail': float(_global_config['chunking_config'].get('summarization_detail', 0.5)),
    'summarize_min_chunk_tokens': int(_global_config['chunking_config'].get('summarize_min_chunk_tokens', 500)),
    'summarize_chunk_delimiter': _global_config['chunking_config'].get('summarize_chunk_delimiter', "."),
    'summarize_recursively': _global_config['chunking_config'].get('summarize_recursively', False),
    'summarize_verbose': _global_config['chunking_config'].get('summarize_verbose', False),
    'summarize_system_prompt': _global_config['chunking_config'].get('summarize_system_prompt', "Rewrite this text in summarized form."),
    'summarize_additional_instructions': _global_config['chunking_config'].get('summarize_additional_instructions', None),
    'summarize_temperature': float(_global_config['chunking_config'].get('summarize_temperature', 0.1)),
    'summarization_llm_provider': _global_config['chunking_config'].get('summarization_llm_provider', 'openai'),
    'summarization_llm_model': _global_config['chunking_config'].get('summarization_llm_model', 'gpt-4o')
}
# Expose the library's default options for the endpoint to use
DEFAULT_CHUNK_OPTIONS = _default_chunk_options_from_config.copy()

# openai_api_key = _global_config.get('API', 'openai_api_key') # Will handle this later with OpenAI client

#
# End of settings
#######################################################################################################################
#
# Functions:

class Chunker:
    def __init__(self,
                 options: Optional[Dict[str, Any]] = None,
                 tokenizer_name_or_path: str = "gpt2",
                 # Specific methods needing LLMs will take them as args or use a callback.
                 ):
        """
        Initializes the Chunker.

        Args:
            options (Optional[Dict[str, Any]]): Custom chunking options to override defaults.
            tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer to use.
                                           Defaults to "gpt2".
        """
        # Initialize options: start with defaults, then update with provided options
        self.options = DEFAULT_CHUNK_OPTIONS.copy()
        if options:
            # Ensure type consistency for options that need it
            for key in ['max_size', 'overlap',
                        'semantic_overlap_sentences', 'base_adaptive_chunk_size',
                        'min_adaptive_chunk_size', 'max_adaptive_chunk_size']:
                if key in options and options[key] is not None:
                    try:
                        options[key] = int(options[key])
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid type for option '{key}': {options[key]}. Using default or ignoring.")
                        options[key] = self.options.get(key) # Revert to default from self.options

            for key in ['semantic_similarity_threshold']:
                 if key in options and options[key] is not None:
                    try:
                        options[key] = float(options[key])
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid type for option '{key}': {options[key]}. Using default or ignoring.")
                        options[key] = self.options.get(key) # Revert to default

            self.options.update(options)

        logging.debug(f"Chunker initialized with options: {self.options}")

        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            logging.info(f"Tokenizer '{tokenizer_name_or_path}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load tokenizer '{tokenizer_name_or_path}': {e}. Some token-based methods may fail.")
            # Fallback or raise error? For now, set to None and let methods handle it.
            self.tokenizer = None
            # raise ChunkingError(f"Failed to load tokenizer '{tokenizer_name_or_path}': {e}") from e

    def _get_option(self, key: str, default_override: Optional[Any] = None) -> Any:
        """Helper to get an option, allowing for a dynamic default."""
        # Try to get from self.options first
        value = self.options.get(key)
        if value is not None:
            return value
        # If not found in self.options or is None, use default_override
        return default_override

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.

        Args:
            text (str): The text to detect language from.

        Returns:
            str: The detected language code (e.g., 'en', 'zh-cn').
                 Defaults to 'en' if detection fails.
        """
        if not text or not text.strip():
            logging.warning("Attempted to detect language from empty or whitespace-only text. Defaulting to 'en'.")
            return self._get_option('language', 'en') # Use option if available, else 'en'
        try:
            # langdetect can be sensitive to very short texts.
            # Add a minimum length check if it becomes an issue.
            # For example: if len(text) < 20: return 'en' (or self.options.get('language', 'en'))
            lang = detect(text)
            logging.debug(f"Detected language: {lang}")
            return lang
        except LangDetectException as e:
            logging.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
            return self._get_option('language', 'en')
        except Exception as e_gen:
            logging.error(f"Unexpected error during language detection: {e_gen}. Defaulting to 'en'.")
            return self._get_option('language', 'en')


    def _ensure_language(self, text: str, language_option: Optional[str] = None) -> str:
        """
        Ensures a language is determined, using option, detection, or default.
        """
        # Priority: 1. Explicit language_option, 2. self.options['language'], 3. detect_language
        if language_option:
            return language_option
        instance_lang_opt = self._get_option('language') # Get from self.options
        if instance_lang_opt:
            return instance_lang_opt
        return self.detect_language(text)


    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Strips whitespace from each chunk and removes empty chunks.
        """
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def chunk_text(self,
                   text: str,
                   method: Optional[str] = None,
                   llm_call_function: Optional[Callable[[Dict[str, Any]], Union[str, Generator[str, None, None]]]] = None,
                   llm_api_config: Optional[Dict[str, Any]] = None,
                   ) -> List[Union[str, Dict[str, Any]]]:
        """
        Main method to chunk text based on the specified method in options or argument.

        Args:
            text (str): The text to chunk.
            method (Optional[str]): Override the chunking method defined in options.

        Returns:
            List[Union[str, Dict[str, Any]]]: A list of chunks.
                                              Strings for most methods, Dicts for JSON-based chunking.

        Raises:
            InvalidChunkingMethodError: If the method is not supported.
            ChunkingError: For errors during the chunking process.
        """
        chunk_method = method if method else self._get_option('method', 'words')
        max_size = self._get_option('max_size') # Already int from __init__
        overlap = self._get_option('overlap')   # Already int from __init__
        language = self._ensure_language(text, self._get_option('language')) # Ensure language is determined

        logging.debug(f"Chunking text with method='{chunk_method}', max_size={max_size}, overlap={overlap}, language='{language}'")

        # Adaptive chunking can modify max_size before the main method is called
        if self._get_option('adaptive', False) and chunk_method not in ['semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize']:
            # Note: Adaptive sizing might not make sense for all methods.
            # Here, we apply it to general text methods.
            base_adaptive_size = self._get_option('base_adaptive_chunk_size')
            min_adaptive_size = self._get_option('min_adaptive_chunk_size')
            max_adaptive_size = self._get_option('max_adaptive_chunk_size')
            if self.tokenizer: # NLTK based adaptive_chunk_size needs punkt
                 max_size = self._adaptive_chunk_size_nltk(text, base_adaptive_size, min_adaptive_size, max_adaptive_size, language)
            else: # Fallback if no tokenizer for NLTK based one.
                 max_size = self._adaptive_chunk_size_non_punkt(text, base_adaptive_size, min_adaptive_size, max_adaptive_size)
            logging.info(f"Adaptive chunking adjusted max_size to: {max_size}")


        # Multi-level chunking is a wrapper around other methods
        if self._get_option('multi_level', False) and chunk_method in ['words', 'sentences']:
             logging.info(f"Applying multi-level chunking with base method: {chunk_method}")
             return self._multi_level_chunking(text, chunk_method, max_size, overlap, language)


        if chunk_method == 'words':
            return self._chunk_text_by_words(text, max_words=max_size, overlap=overlap, language=language)
        elif chunk_method == 'sentences':
            return self._chunk_text_by_sentences(text, max_sentences=max_size, overlap=overlap, language=language)
        elif chunk_method == 'paragraphs':
            return self._chunk_text_by_paragraphs(text, max_paragraphs=max_size, overlap=overlap)
        elif chunk_method == 'tokens':
            if not self.tokenizer:
                raise ChunkingError("Tokenizer not loaded, cannot use 'tokens' chunking method.")
            return self._chunk_text_by_tokens(text, max_tokens=max_size, overlap=overlap)
        elif chunk_method == 'semantic':
            # semantic_chunking needs to be a method of the class too
            return self._semantic_chunking(text, max_chunk_size=max_size, unit='words') # unit can be an option
        elif chunk_method == 'json':
            # chunk_text_by_json and its helpers need to be methods
            return self._chunk_text_by_json(text, max_size=max_size, overlap=overlap)
        elif chunk_method == 'ebook_chapters':
            # Needs to be a method
            return self._chunk_ebook_by_chapters(text,
                                                 max_size=max_size, # max_size here might mean something different, e.g. sub-chunking chapters
                                                 overlap=overlap,
                                                 custom_pattern=self._get_option('custom_chapter_pattern'),
                                                 language=language)
        elif chunk_method == 'xml':
            # Needs to be a method
            return self._chunk_xml(text, max_size=max_size, overlap=overlap, language=language)
        elif chunk_method == 'rolling_summarize':
            if not llm_call_function:
                raise ChunkingError("Missing 'llm_call_function' for 'rolling_summarize' method.")
            if not self.tokenizer:  # Still need tokenizer for token counting in helper
                raise ChunkingError("Tokenizer required for 'rolling_summarize' to estimate chunk sizes for LLM.")

            summary = self._rolling_summarize(
                text_to_summarize=text,
                llm_summarize_step_func=llm_call_function,  # Pass the generic call function
                llm_api_config=llm_api_config or {},  # Pass relevant API name, model, key for the call_func
                # Other summarization-specific options from self.options
                detail=self._get_option('summarization_detail', 0.5),
                min_chunk_tokens=self._get_option('summarize_min_chunk_tokens', 500),
                chunk_delimiter=self._get_option('summarize_chunk_delimiter', "."),
                recursive_summarization=self._get_option('summarize_recursively', False),
                verbose=self._get_option('summarize_verbose', False),
                system_prompt_content=self._get_option('summarize_system_prompt',
                                                       "Rewrite this text in summarized form."),
                additional_instructions=self._get_option('summarize_additional_instructions', None)
            )
            return [summary]  # Wrap in list
        # Add 'hybrid' if you still need it. It was similar to token based.
        # def chunk_text_hybrid(text: str, max_tokens: int = 1000, overlap: int = 0) -> List[str]:
        else:
            logging.warning(f"Unknown chunking method '{chunk_method}'. Returning full text as a single chunk.")
            # return [text] # Previous behavior
            raise InvalidChunkingMethodError(f"Unsupported chunking method: '{chunk_method}'")


    def _chunk_text_by_words(self, text: str, max_words: int, overlap: int, language: str) -> List[str]:
        logging.debug(f"Chunking by words: max_words={max_words}, overlap={overlap}, language='{language}'")
        # Language-specific word tokenization
        words: List[str]
        if language.startswith('zh'):  # Chinese
            try:
                import jieba
                words = list(jieba.cut(text))
            except ImportError:
                logging.warning("jieba library not found for Chinese word tokenization. Falling back to space splitting.")
                words = text.split()
            except Exception as e:
                logging.warning(f"Error using jieba for Chinese tokenization: {e}. Falling back to space splitting.")
                words = text.split()
        elif language == 'ja':  # Japanese
            try:
                import fugashi
                tagger = fugashi.Tagger('-Owakati') # Output wakachi-gaki (space-separated words)
                words = tagger.parse(text).split()
            except ImportError:
                logging.warning("fugashi library not found for Japanese word tokenization. Falling back to space splitting.")
                words = text.split()
            except Exception as e: # fugashi can raise various errors
                logging.warning(f"Error using fugashi for Japanese tokenization: {e}. Falling back to space splitting.")
                words = text.split()
        else:  # Default to simple splitting for other languages
            words = text.split()

        logging.debug(f"Total words: {len(words)}")
        if max_words <= 0 :
            logging.warning(f"max_words is {max_words}, must be positive. Defaulting to 1 if text exists, or empty list.")
            return [text] if text else [] # Or raise error
        if overlap >= max_words :
            logging.warning(f"Overlap {overlap} is >= max_words {max_words}. Setting overlap to 0.")
            overlap = 0


        chunks = []
        # Ensure step is at least 1 to prevent infinite loops if max_words equals overlap (though handled above)
        step = max_words - overlap
        if step <= 0: step = max_words # Should not happen if overlap < max_words

        for i in range(0, len(words), step):
            chunk_words = words[i : i + max_words]
            chunks.append(' '.join(chunk_words))
            logging.debug(f"Created word chunk {len(chunks)} with {len(chunk_words)} words")

        return self._post_process_chunks(chunks)


    def _chunk_text_by_sentences(self, text: str, max_sentences: int, overlap: int, language: str) -> List[str]:
        logging.debug(f"Chunking by sentences: max_sentences={max_sentences}, overlap={overlap}, lang='{language}'")
        sentences: List[str]

        if language.startswith('zh'):
            # Basic punctuation-based sentence splitting for Chinese
            sentences = [s.strip() for s in re.split(r'([。！？；])', text) if s.strip()]
            # Join sentence with its delimiter if present
            processed_sentences = []
            temp_sentence = ""
            for i, part in enumerate(sentences):
                if part in ['。', '！', '？', '；']:
                    if temp_sentence: # Add delimiter to previous sentence part
                        processed_sentences.append(temp_sentence + part)
                        temp_sentence = ""
                    # else: # Delimiter at start, could be an issue or just keep it.
                    #    processed_sentences.append(part)
                else:
                    if temp_sentence : # If previous part was also text (should not happen with this regex)
                        processed_sentences.append(temp_sentence)
                    temp_sentence = part
            if temp_sentence: # last sentence part
                processed_sentences.append(temp_sentence)
            sentences = [s for s in processed_sentences if s]

        elif language == 'ja':
            # Basic punctuation-based sentence splitting for Japanese
            # Consider using a library like "JaSP" for more robust Japanese sentence splitting if needed.
            sentences = [s.strip() for s in re.split(r'([。！？])', text) if s.strip()]
            processed_sentences = []
            temp_sentence = ""
            for i, part in enumerate(sentences):
                if part in ['。', '！', '？']:
                    if temp_sentence:
                        processed_sentences.append(temp_sentence + part)
                        temp_sentence = ""
                else:
                    if temp_sentence:
                         processed_sentences.append(temp_sentence)
                    temp_sentence = part
            if temp_sentence:
                processed_sentences.append(temp_sentence)
            sentences = [s for s in processed_sentences if s]
        else:
            try:
                # NLTK expects language names like 'english', 'spanish', etc.
                # Map 'en' to 'english' if necessary for NLTK compatibility.
                nltk_lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'} # extend as needed
                nltk_language = nltk_lang_map.get(language.lower(), language.lower()) # Default to language if not in map
                sentences = sent_tokenize(text, language=nltk_language)
            except LookupError:
                logging.warning(f"NLTK Punkt tokenizer not found for language '{language}' (mapped to '{nltk_language}'). Using default 'english'.")
                sentences = sent_tokenize(text, language='english')
            except Exception as e_sent_tokenize:
                logging.error(f"Error during NLTK sentence tokenization for language '{language}': {e_sent_tokenize}. Falling back to newline splitting.")
                sentences = text.splitlines() # Basic fallback

        if max_sentences <= 0:
            logging.warning(f"max_sentences is {max_sentences}, must be positive. Defaulting to 1 sentence if text exists.")
            return [text] if text else []
        if overlap >= max_sentences:
            logging.warning(f"Overlap {overlap} >= max_sentences {max_sentences}. Setting overlap to 0.")
            overlap = 0

        chunks = []
        step = max_sentences - overlap
        if step <= 0: step = max_sentences

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i : i + max_sentences]
            chunks.append(' '.join(chunk_sentences))
        return self._post_process_chunks(chunks)


    def _chunk_text_by_paragraphs(self, text: str, max_paragraphs: int, overlap: int) -> List[str]:
        logging.debug(f"Chunking by paragraphs: max_paragraphs={max_paragraphs}, overlap={overlap}")
        # Split by one or more empty lines (common paragraph delimiter)
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs

        if not paragraphs:
            return []
        if max_paragraphs <= 0:
            logging.warning("max_paragraphs must be positive. Returning single chunk or empty.")
            return [text] if text.strip() else []
        if overlap >= max_paragraphs:
            logging.warning(f"Overlap {overlap} >= max_paragraphs {max_paragraphs}. Setting overlap to 0.")
            overlap = 0

        chunks = []
        step = max_paragraphs - overlap
        if step <= 0: step = max_paragraphs

        for i in range(0, len(paragraphs), step):
            chunk_paragraphs = paragraphs[i : i + max_paragraphs]
            chunks.append('\n\n'.join(chunk_paragraphs)) # Join with double newline to preserve paragraph structure
        return self._post_process_chunks(chunks) # post_process_chunks strips leading/trailing, which is fine


    def _chunk_text_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        # This uses the accurate tokenizer version
        if not self.tokenizer:
            logging.error("Tokenizer not available for token-based chunking.")
            raise ChunkingError("Tokenizer not loaded, cannot use 'tokens' chunking method.")

        logging.debug(f"Chunking by tokens: max_tokens={max_tokens}, overlap_tokens={overlap} (token overlap)")
        if max_tokens <= 0:
            logging.warning("max_tokens must be positive. Returning single chunk or empty.")
            return [text] if text.strip() else []

        tokens = self.tokenizer.encode(text)
        logging.debug(f"Total tokens: {len(tokens)}")

        # Overlap here is in number of tokens
        if overlap >= max_tokens :
            logging.warning(f"Token overlap {overlap} >= max_tokens {max_tokens}. Setting overlap to 0.")
            overlap = 0

        step = max_tokens - overlap
        if step <= 0: step = max_tokens


        chunks = []
        for i in range(0, len(tokens), step):
            chunk_token_ids = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True) # skip_special_tokens might be an option
            chunks.append(chunk_text)
        return self._post_process_chunks(chunks)


    # --- Adaptive Chunking Methods ---
    def _adaptive_chunk_size_nltk(self, text: str, base_size: int, min_size: int, max_size: int, language: str) -> int:
        """Adjusts chunk size based on NLTK sentence tokenization."""
        logging.debug(f"Calculating adaptive chunk size (NLTK) for lang '{language}'. Base: {base_size}, Min: {min_size}, Max: {max_size}")
        try:
            nltk_lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
            nltk_language = nltk_lang_map.get(language.lower(), language.lower())
            sentences = sent_tokenize(text, language=nltk_language)
        except LookupError:
            logging.warning(f"NLTK Punkt for '{language}' not found for adaptive sizing. Using non-NLTK fallback.")
            return self._adaptive_chunk_size_non_punkt(text, base_size, min_size, max_size)
        except Exception as e:
            logging.warning(f"Error tokenizing sentences for adaptive sizing with NLTK: {e}. Using non-NLTK fallback.")
            return self._adaptive_chunk_size_non_punkt(text, base_size, min_size, max_size)


        if not sentences:
            return base_size

        avg_sentence_length_words = sum(len(s.split()) for s in sentences) / len(sentences)
        logging.debug(f"Avg sentence length (words): {avg_sentence_length_words}")

        size_factor = 1.0
        if avg_sentence_length_words < 10: # Short sentences
            size_factor = 1.2
        elif avg_sentence_length_words > 25: # Long sentences
            size_factor = 0.8

        adaptive_size = int(base_size * size_factor)
        final_size = max(min_size, min(adaptive_size, max_size))
        logging.debug(f"Adaptive size calculated (NLTK): {final_size} (factor: {size_factor})")
        return final_size

    def _adaptive_chunk_size_non_punkt(self, text: str, base_size: int, min_size: int, max_size: int) -> int:
        """Adjusts chunk size based on average word length if NLTK is not available."""
        logging.debug(f"Calculating adaptive chunk size (non-NLTK). Base: {base_size}, Min: {min_size}, Max: {max_size}")
        words = text.split()
        if not words:
            return base_size

        # Using character length of words as a proxy for complexity
        avg_word_char_length = sum(len(word) for word in words) / len(words) if words else 0
        logging.debug(f"Avg word char length: {avg_word_char_length}")

        size_factor = 1.0
        if avg_word_char_length > 7:  # Longer average words -> potentially more complex
            size_factor = 0.85
        elif avg_word_char_length < 4: # Shorter average words -> potentially simpler
            size_factor = 1.15

        adaptive_size = int(base_size * size_factor)
        final_size = max(min_size, min(adaptive_size, max_size))
        logging.debug(f"Adaptive size calculated (non-NLTK): {final_size} (factor: {size_factor})")
        return final_size

    # Multi-level chunking - can be a wrapper method
    def _multi_level_chunking(self, text: str, base_method_name: str, max_size: int, overlap: int, language: str) -> List[str]:
        logging.debug(f"Multi-level chunking: base_method='{base_method_name}', max_size={max_size}, overlap={overlap}, lang='{language}'")

        # First level: chunk by paragraphs (configurable, but paragraphs is common)
        # The max_paragraphs for this initial split could be larger or an independent option.
        # Using a doubled max_size as a heuristic for paragraph character length.
        initial_paragraph_chunks = self._chunk_text_by_paragraphs(text, max_paragraphs=5, overlap=0) # Small overlap for first pass

        final_chunks = []
        for para_chunk in initial_paragraph_chunks:
            if base_method_name == 'words':
                final_chunks.extend(self._chunk_text_by_words(para_chunk, max_words=max_size, overlap=overlap, language=language))
            elif base_method_name == 'sentences':
                final_chunks.extend(self._chunk_text_by_sentences(para_chunk, max_sentences=max_size, overlap=overlap, language=language))
            else:
                # Should not happen if multi_level is only enabled for words/sentences
                final_chunks.append(para_chunk)
        return final_chunks # Already post-processed by the inner calls

    # --- Stubs for other methods to be moved ---
    def _semantic_chunking(self, text: str, max_chunk_size: int, unit: str) -> List[str]:
        logging.info("Semantic chunking called...")
        # Lazy import for sklearn if not already at top
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ChunkingError("Scikit-learn not installed. Cannot use 'semantic' chunking. Install with 'pip install scikit-learn'")

        language = self._ensure_language(text, self._get_option('language'))
        nltk_lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
        nltk_language = nltk_lang_map.get(language.lower(), language.lower())

        try:
            sentences = sent_tokenize(text, language=nltk_language)
        except LookupError:
            logging.warning(f"NLTK Punkt for '{language}' (for semantic chunking) not found. Defaulting to 'english'.")
            sentences = sent_tokenize(text, language='english')
        except Exception as e:
            logging.error(f"Error sentence tokenizing for semantic chunking: {e}. Using newline split.")
            sentences = text.splitlines()

        if not sentences:
            return []

        try:
            vectorizer = TfidfVectorizer()
            # Filter out empty strings from sentences before fitting, if any from regex or bad tokenization
            valid_sentences = [s for s in sentences if s.strip()]
            if not valid_sentences: return [] # No valid sentences to process
            sentence_vectors = vectorizer.fit_transform(valid_sentences)
        except ValueError as ve: # TFidfVectorizer can raise ValueError if vocabulary is empty (e.g. all stop words)
            logging.warning(f"TF-IDF Vectorizer error during semantic chunking (perhaps all stop words or very short text): {ve}. Returning single chunk.")
            return [text] if text.strip() else []


        chunks = []
        current_chunk_sentences = []
        current_size_units = 0
        # Semantic options
        similarity_threshold = self._get_option('semantic_similarity_threshold', 0.3) # Default if not in options
        overlap_sentences_count = self._get_option('semantic_overlap_sentences', 1) # Default if not in options

        # Helper to count units (words, tokens, characters)
        def _count_units(txt: str, unit_type: str) -> int:
            if unit_type == 'words':
                return len(txt.split())
            elif unit_type == 'tokens' and self.tokenizer:
                return len(self.tokenizer.encode(txt))
            elif unit_type == 'characters':
                return len(txt)
            logging.warning(f"Unknown unit type '{unit_type}' or tokenizer missing for tokens. Defaulting to word count.")
            return len(txt.split())


        for i, sentence_text in enumerate(valid_sentences):
            sentence_unit_count = _count_units(sentence_text, unit)

            # Break condition 1: Max chunk size exceeded
            if current_size_units + sentence_unit_count > max_chunk_size and current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                # Apply overlap: take last N sentences for the new chunk
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences_count:] if overlap_sentences_count > 0 and len(current_chunk_sentences) > overlap_sentences_count else []
                current_size_units = _count_units(' '.join(current_chunk_sentences), unit)

            current_chunk_sentences.append(sentence_text)
            current_size_units += sentence_unit_count

            # Break condition 2: Semantic similarity drop (only if we have a next sentence)
            if i + 1 < len(valid_sentences):
                # Ensure vectors are 2D for cosine_similarity
                current_sentence_vector = sentence_vectors[i:i+1]
                next_sentence_vector = sentence_vectors[i+1:i+2]

                similarity = 0.0
                try:
                    similarity = cosine_similarity(current_sentence_vector, next_sentence_vector)[0, 0]
                except IndexError: # Can happen if vectors are not as expected
                    logging.warning(f"Could not compute similarity for sentence index {i}. Assuming low similarity.")

                # Break if similarity drops AND current chunk has substantial size (e.g., half of max_chunk_size)
                if similarity < similarity_threshold and current_size_units >= (max_chunk_size // 2) and current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences_count:] if overlap_sentences_count > 0 and len(current_chunk_sentences) > overlap_sentences_count else []
                    current_size_units = _count_units(' '.join(current_chunk_sentences), unit)

        # Add any remaining sentences in current_chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        return self._post_process_chunks(chunks)


    def _chunk_text_by_json(self, text: str, max_size: int, overlap: int) -> List[Dict[str, Any]]:
        logging.debug("chunk_text_by_json (method) started...")
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON data provided to _chunk_text_by_json: {e}")
            raise InvalidInputError(f"Invalid JSON data: {e}") from e

        if isinstance(json_data, list):
            return self._chunk_json_list(json_data, max_size, overlap)
        elif isinstance(json_data, dict):
            # `chunk_json_dict` assumes a specific structure ('data', 'metadata').
            # This might need to be more generic or have options to specify keys.
            return self._chunk_json_dict(json_data, max_size, overlap)
        else:
            msg = "Unsupported JSON structure. _chunk_text_by_json only supports top-level JSON objects or arrays."
            logging.error(msg)
            raise InvalidInputError(msg)

    def _chunk_json_list(self, json_list: List[Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
        logging.debug(f"Chunking JSON list: max_items_per_chunk={max_size}, overlap_items={overlap}")
        if max_size <= 0:
            raise ValueError("max_size for JSON list chunking must be positive.")
        if overlap >= max_size:
            logging.warning(f"JSON list overlap {overlap} >= max_size {max_size}. Setting overlap to 0.")
            overlap = 0

        chunks_output = []
        total_items = len(json_list)
        step = max_size - overlap
        if step <= 0: step = max_size

        for i in range(0, total_items, step):
            chunk_data = json_list[i : i + max_size]
            # Metadata specific to this chunking method
            metadata = {
                'original_list_size': total_items,
                'item_start_index': i,
                'item_end_index': i + len(chunk_data) -1,
                # 'chunk_index', 'total_chunks' etc. will be added by improved_chunking_process wrapper
            }
            chunks_output.append({
                'json': chunk_data, # The actual JSON content of the chunk
                'metadata': metadata  # Metadata specific to this JSON chunk
            })
        return chunks_output


    def _chunk_json_dict(self, json_dict: Dict[str, Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
        # This method is quite specific to a schema with a 'data' key.
        # Consider making 'chunkable_key' an option.
        logging.debug(f"Chunking JSON dict: max_keys_in_data_per_chunk={max_size}, overlap_keys={overlap}")
        chunkable_key = self._get_option('json_chunkable_data_key', 'data') # e.g. {'json_chunkable_data_key': 'entries'}

        if chunkable_key not in json_dict or not isinstance(json_dict[chunkable_key], dict):
            msg = f"Chunkable key '{chunkable_key}' not found in JSON dictionary or is not a dictionary itself."
            logging.error(msg)
            raise InvalidInputError(msg)

        if max_size <= 0:
            raise ValueError("max_size for JSON dict chunking must be positive.")
        if overlap >= max_size:
            logging.warning(f"JSON dict overlap {overlap} >= max_size {max_size}. Setting overlap to 0.")
            overlap = 0

        data_to_chunk = json_dict[chunkable_key]
        all_keys = list(data_to_chunk.keys())
        total_keys = len(all_keys)

        chunks_output = []
        step = max_size - overlap
        if step <= 0: step = max_size

        # Preserve other parts of the original dictionary
        preserved_data_shell = {k: v for k, v in json_dict.items() if k != chunkable_key}

        for i in range(0, total_keys, step):
            current_chunk_keys = all_keys[i : i + max_size]
            # Note: Overlap for dict keys might be complex if order isn't guaranteed or meaningful.
            # This simple slicing assumes order is somewhat stable or user understands the implication.

            chunk_data_content = {key: data_to_chunk[key] for key in current_chunk_keys}

            # Create the new chunked JSON structure
            new_json_chunk = preserved_data_shell.copy()
            new_json_chunk[chunkable_key] = chunk_data_content

            metadata = {
                'original_dict_total_keys_in_data': total_keys,
                'key_start_index_in_data': i, # Based on original list of keys
                'keys_in_this_chunk_data': len(current_chunk_keys),
            }
            chunks_output.append({
                'json': new_json_chunk,
                'metadata': metadata
            })
        return chunks_output

        # In tldw_Server_API/app/core/Utils/Chunk_Lib.py

    def _chunk_ebook_by_chapters(self, text: str, max_size: int, overlap: int, custom_pattern: Optional[str],
                                 language: str) -> List[Dict[str, Any]]:
        logging.debug(f"Chunking Ebook by Chapters. Custom pattern: {custom_pattern}, Lang: {language}")

        chapter_patterns = [
            custom_pattern,
            r'^\s*chapter\s+\d+([:.\-\s].*)?$',
            r'^\s*chapter\s+[ivxlcdm]+([:.\-\s].*)?$',
            r'^\s*(Part|Book|Volume)\s+[A-Za-z0-9]+([:.\-\s].*)?$',
            r'^\s*\d+\s*([:.\-\s][^\r\n]{1,150}|[^\r\n]{1,150})?$',
            r'^\s*#{1,4}\s+[^\r\n]+',
            r'^\s*(PREFACE|INTRODUCTION|CONTENTS|APPENDIX|EPILOGUE|PROLOGUE|ACKNOWLEDGMENTS?|SECTION\s*\d*|UNIT\s*\d*)\s*$',
            # r'^\s*(?=[A-Za-z])[A-Z0-9:.,!?\'\s]{5,100}\s*$' # This was too greedy and commented out
        ]
        active_patterns = [p for p in chapter_patterns if p is not None]
        if not active_patterns:  # pragma: no cover
            logging.warning("No chapter patterns available for ebook chunking.")
            if text.strip():
                return [{'text': text,
                         'metadata': {'chunk_type': 'single_document_no_chapters', 'chapter_title': 'Full Document'}}]
            return []

        lines = text.splitlines()
        chapter_splits: List[Dict[str, Any]] = []
        chapter_number = 0

        first_heading_index = -1
        first_heading_title_text = "Preface or Introduction"

        current_scan_patterns = list(active_patterns)
        for line_idx, line_content in enumerate(lines):
            for pattern_str in list(current_scan_patterns):
                try:
                    if re.match(pattern_str, line_content, re.IGNORECASE):
                        first_heading_index = line_idx
                        first_heading_title_text = line_content.strip()
                        break
                except re.error as re_e:  # pragma: no cover
                    logging.warning(
                        f"Regex error in chapter pattern '{pattern_str}' during initial scan: {re_e}. Disabling this pattern.")
                    if pattern_str in current_scan_patterns: current_scan_patterns.remove(pattern_str)
                    if pattern_str in active_patterns: active_patterns.remove(pattern_str)
            if first_heading_index != -1:
                break

        if first_heading_index > 0:
            preface_content_lines = lines[:first_heading_index]
            preface_text = "\n".join(preface_content_lines).strip()
            if preface_text:
                chapter_number += 1
                chapter_splits.append({
                    'text': preface_text,
                    'metadata': {
                        'chunk_type': 'preface',
                        'chapter_number': chapter_number,
                        'chapter_title': "Preface/Introduction",  # Standardized title for this type of preface
                        'detected_chapter_pattern': 'preface_heuristic',
                    }
                })
        elif first_heading_index == -1:  # pragma: no cover
            if text.strip():
                logging.warning(
                    "No chapter headings found using patterns. Returning document as a single chapter chunk.")
                return [{'text': text,
                         'metadata': {'chunk_type': 'single_document_no_chapters', 'chapter_title': 'Full Document'}}]
            return []

        start_line_of_current_chapter_content = first_heading_index
        current_chapter_title = first_heading_title_text

        current_chapter_pattern_str = "unknown"
        if first_heading_index != -1 and first_heading_index < len(lines):
            for p_str in active_patterns:
                if re.match(p_str, lines[first_heading_index], re.IGNORECASE):
                    current_chapter_pattern_str = p_str
                    break

        for line_idx in range(first_heading_index + 1, len(lines)):
            line_content = lines[line_idx]
            is_new_chapter_heading = False
            new_heading_pattern_str = None

            for pattern_str in active_patterns:
                try:
                    if re.match(pattern_str, line_content, re.IGNORECASE):
                        is_new_chapter_heading = True
                        new_heading_pattern_str = pattern_str
                        break
                except re.error as re_e_inner:  # pragma: no cover
                    logging.warning(f"Regex error (inner loop) for pattern '{pattern_str}': {re_e_inner}.")

            if is_new_chapter_heading:
                chapter_content_lines = lines[start_line_of_current_chapter_content: line_idx]
                chapter_text = "\n".join(chapter_content_lines).strip()

                if chapter_text:
                    chapter_number += 1
                    chunk_type = 'chapter'
                    # Determine if this block should be considered a preface based on its content/title
                    # This logic is specifically for cases where the preface *is* the first detected heading block
                    if chapter_number == 1 and first_heading_index == 0 and \
                            (current_chapter_title.lower() == "preface" or \
                             current_chapter_title.lower() == "introduction" or \
                             current_chapter_title == "Preface or Introduction"):
                        chunk_type = 'preface'

                    # Set the final title for metadata
                    final_metadata_title = current_chapter_title
                    if chunk_type == 'preface':
                        final_metadata_title = "Preface/Introduction"

                    chapter_splits.append({
                        'text': chapter_text,
                        'metadata': {
                            'chunk_type': chunk_type,
                            'chapter_number': chapter_number,
                            'chapter_title': final_metadata_title,  # MODIFIED LINE
                            'detected_chapter_pattern': current_chapter_pattern_str,
                        }
                    })

                start_line_of_current_chapter_content = line_idx
                current_chapter_title = line_content.strip()
                current_chapter_pattern_str = new_heading_pattern_str or "unknown"

        # Add the last chapter
        if start_line_of_current_chapter_content < len(lines):
            last_chapter_content_lines = lines[start_line_of_current_chapter_content:]
            last_chapter_text = "\n".join(last_chapter_content_lines).strip()
            if last_chapter_text:
                chapter_number += 1
                chunk_type = 'chapter'
                is_first_processed_block = not chapter_splits

                # Determine if this last block (if it's the only/first block) should be a preface
                condition_is_first_block_and_preface_like_title = \
                    (is_first_processed_block or (chapter_number == 1 and first_heading_index == 0)) and \
                    (current_chapter_title.lower() == "preface" or \
                     current_chapter_title.lower() == "introduction" or \
                     current_chapter_title == "Preface or Introduction")

                if condition_is_first_block_and_preface_like_title:
                    chunk_type = 'preface'

                # Set the final title for metadata
                final_metadata_title = current_chapter_title
                if chunk_type == 'preface':
                    final_metadata_title = "Preface/Introduction"

                chapter_splits.append({
                    'text': last_chapter_text,
                    'metadata': {
                        'chunk_type': chunk_type,
                        'chapter_number': chapter_number,
                        'chapter_title': final_metadata_title,  # MODIFIED LINE
                        'detected_chapter_pattern': current_chapter_pattern_str,
                    }
                })

        final_chapter_chunks: List[Dict[str, Any]] = []
        for i, chap_data in enumerate(chapter_splits):
            chap_data['metadata']['chunk_index_in_book'] = i + 1
            chap_data['metadata']['total_chapters_detected'] = len(chapter_splits)

            tokenizer_available = hasattr(self, 'tokenizer') and self.tokenizer and hasattr(self.tokenizer,
                                                                                            'encode') and callable(
                self.tokenizer.encode)
            if max_size > 0 and tokenizer_available and len(
                    self.tokenizer.encode(chap_data['text'])) > max_size:  # pragma: no cover
                logging.info(
                    f"Chapter '{chap_data['metadata']['chapter_title']}' (length {len(self.tokenizer.encode(chap_data['text']))} tokens) exceeds max_size {max_size}. Sub-chunking.")
                sub_chunks = self._chunk_text_by_tokens(chap_data['text'], max_tokens=max_size,
                                                        overlap=overlap if overlap < max_size else max_size // 5)
                for sub_idx, sub_chunk_text in enumerate(sub_chunks):
                    sub_chunk_metadata = chap_data['metadata'].copy()
                    sub_chunk_metadata['sub_chunk_index_in_chapter'] = sub_idx + 1
                    sub_chunk_metadata['total_sub_chunks_in_chapter'] = len(sub_chunks)
                    sub_chunk_metadata['chunk_type'] = 'chapter_sub_chunk'
                    final_chapter_chunks.append({'text': sub_chunk_text, 'metadata': sub_chunk_metadata})
            else:
                final_chapter_chunks.append(chap_data)

        return final_chapter_chunks


    def _extract_xml_structure_recursive(self, element, path="") -> List[Tuple[str, str]]:
        """ Helper for _chunk_xml: Recursively extract XML structure and content. """
        results = []
        current_path = f"{path}/{element.tag}" if path else element.tag

        if element.text and element.text.strip():
            results.append((current_path, element.text.strip()))
        if element.attrib:
            for key, value in element.attrib.items():
                results.append((f"{current_path}/@{key}", value))
        for child in element:
            results.extend(self._extract_xml_structure_recursive(child, current_path))
        # Include tail text if present (text after a child element, within its parent)
        if element.tail and element.tail.strip():
            results.append((f"{path}/{element.tag}/#tail", element.tail.strip())) # Path indicates it's tail of current_path
        return results

    def _chunk_xml(self, xml_text: str, max_size: int, overlap: int, language: str) -> List[Dict[str, Any]]:
        # max_size is in words for the content of combined XML elements
        # overlap is in number of XML elements (path-content pairs)
        logging.debug(f"Chunking XML: max_words_per_chunk_content={max_size}, overlap_elements={overlap}, Lang: {language}")
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logging.error(f"XML parsing error: {e}")
            raise InvalidInputError(f"Invalid XML content: {e}") from e

        xml_elements_with_paths = self._extract_xml_structure_recursive(root)
        if not xml_elements_with_paths:
            return []

        if max_size <=0:
            raise ValueError("max_size for XML chunking must be positive.")
        if overlap >= len(xml_elements_with_paths) and len(xml_elements_with_paths) > 0 : # Check if overlap makes sense
            logging.warning(f"XML overlap elements {overlap} >= total elements {len(xml_elements_with_paths)}. Setting overlap to 0.")
            overlap = 0
        elif overlap < 0:
            overlap = 0


        chunks_output = []
        current_chunk_elements = [] # List of (path, content) tuples
        current_word_count = 0

        # Step is by element, but decision to cut is by word count of accumulated content
        for i in range(len(xml_elements_with_paths)):
            path, content = xml_elements_with_paths[i]
            content_word_count = len(content.split())

            if current_word_count + content_word_count > max_size and current_chunk_elements:
                # Finalize current chunk
                chunk_text_parts = [f"{p}: {c}" for p, c in current_chunk_elements]
                chunk_metadata = {
                    'xml_paths': [p for p, _ in current_chunk_elements],
                    'root_tag': root.tag,
                    'original_xml_attributes': dict(root.attrib),
                    'num_xml_elements_in_chunk': len(current_chunk_elements)
                }
                chunks_output.append({'text': '\n'.join(chunk_text_parts), 'metadata': chunk_metadata})

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk_elements) > overlap:
                    current_chunk_elements = current_chunk_elements[-overlap:]
                else: # Not enough elements for full overlap, or overlap is 0
                    current_chunk_elements = []
                current_word_count = sum(len(c.split()) for _, c in current_chunk_elements)

            current_chunk_elements.append((path, content))
            current_word_count += content_word_count

        # Add the last remaining chunk
        if current_chunk_elements:
            chunk_text_parts = [f"{p}: {c}" for p, c in current_chunk_elements]
            chunk_metadata = {
                'xml_paths': [p for p, _ in current_chunk_elements],
                'root_tag': root.tag,
                'original_xml_attributes': dict(root.attrib),
                'num_xml_elements_in_chunk': len(current_chunk_elements)
            }
            chunks_output.append({'text': '\n'.join(chunk_text_parts), 'metadata': chunk_metadata})

        return chunks_output


    def _rolling_summarize(self,
                           text_to_summarize: str,
                           llm_summarize_step_func: Callable, # The function that will call the actual LLM API
                           llm_api_config: Dict[str, Any],    # Contains {'api_name', 'model', 'api_key', 'temp', etc.}
                           detail: float,
                           min_chunk_tokens: int,
                           chunk_delimiter: str,
                           recursive_summarization: bool,
                           verbose: bool,
                           system_prompt_content: str,
                           additional_instructions: Optional[str]
                           ) -> str:
        if not self.tokenizer: # Should have been checked by caller (chunk_text)
            raise ChunkingError("Tokenizer required for rolling summarization.")

        logging.info(f"Rolling summarization called. Detail: {detail}")
        text_token_length = len(self.tokenizer.encode(text_to_summarize))
        max_summarization_chunks = max(1, text_token_length // min_chunk_tokens)
        min_summarization_chunks = 1
        num_summarization_chunks = int(min_summarization_chunks + detail * (max_summarization_chunks - min_summarization_chunks))
        num_summarization_chunks = max(1, num_summarization_chunks)
        llm_input_chunk_size_tokens = max(min_chunk_tokens, text_token_length // num_summarization_chunks)

        text_chunks_for_llm, _, dropped_count = self._chunk_on_delimiter_for_llm(
            text_to_summarize,
            llm_input_chunk_size_tokens,
            delimiter=chunk_delimiter
        )
        if dropped_count > 0 and verbose:
            logging.warning(f"{dropped_count} parts were dropped during text splitting for summarization.")
        if verbose:
            logging.info(f"Splitting text for summarization into {len(text_chunks_for_llm)} parts.")

        final_system_prompt = system_prompt_content
        if additional_instructions:
            final_system_prompt += f"\n\n{additional_instructions}"

        accumulated_summaries = []
        for i, chunk_for_llm in enumerate(tqdm(text_chunks_for_llm, desc="Summarizing parts", disable=not verbose)):
            user_message_content = chunk_for_llm
            if recursive_summarization and accumulated_summaries:
                user_message_content = f"Previous summary context:\n{accumulated_summaries[-1]}\n\nNew content to summarize and integrate:\n{chunk_for_llm}"

            # Prepare payload for the llm_summarize_step_func
            # This payload structure should match what your `Summarization_General_Lib.analyze` or `_dispatch_to_api` expects
            payload_for_llm_call = {
                "api_name": llm_api_config.get("api_name", "openai"), # Default or from config
                "input_data": user_message_content, # This is the text for the LLM
                "custom_prompt_arg": "", # Rolling summarize manages the full prompt content
                "api_key": llm_api_config.get("api_key"),
                "system_message": final_system_prompt,
                "temp": llm_api_config.get("temperature", self._get_option('summarize_temperature', 0.1)),
                "streaming": False, # Internal steps of rolling summary should not stream to Chunker
                "model": llm_api_config.get("model"),
                "max_tokens": llm_api_config.get("max_tokens")
            }
            try:
                # `llm_summarize_step_func` should be blocking and return a string
                summary_content = llm_summarize_step_func(payload_for_llm_call)

                if isinstance(summary_content, str) and summary_content.startswith("Error:"):
                    logging.error(f"LLM call for summarization part {i+1} failed: {summary_content}")
                    accumulated_summaries.append(f"[Summarization failed for this part: {chunk_for_llm[:100]}...]")
                elif isinstance(summary_content, str):
                    accumulated_summaries.append(summary_content)
                else: # Should not happen if llm_summarize_step_func is well-behaved
                    logging.error(f"LLM call for summarization part {i+1} returned non-string: {type(summary_content)}")
                    accumulated_summaries.append(f"[Summarization error for this part (unexpected type): {chunk_for_llm[:100]}...]")

            except Exception as e_llm:
                logging.error(f"Exception calling llm_summarize_step_func for part {i+1}: {e_llm}", exc_info=True)
                accumulated_summaries.append(f"[Exception during summarization for this part: {chunk_for_llm[:100]}...]")

        final_summary = '\n\n---\n\n'.join(accumulated_summaries) # Join with a clear separator
        return final_summary.strip()

    # Helper for rolling_summarize (was combine_chunks_with_no_minimum)
    def _combine_chunks_for_llm(self,
                                 chunks: List[str],
                                 max_tokens: int, # Max tokens for the combined output for the LLM
                                 chunk_delimiter: str = "\n\n",
                                 header: Optional[str] = None,
                                 add_ellipsis_for_overflow: bool = True,
                                 ) -> Tuple[List[str], List[List[int]], int]:
        if not self.tokenizer:
            raise ChunkingError("Tokenizer required for _combine_chunks_for_llm.")

        dropped_chunk_count = 0
        output_combined_texts = []
        output_original_indices = [] # To track which original chunks went into which combined text

        current_candidate_text_parts = [header] if header else []
        current_candidate_indices = []

        for chunk_idx, chunk_content in enumerate(chunks):
            # Tentatively add the new chunk (with header if it's the first part of a new candidate)
            parts_to_test = current_candidate_text_parts + ([chunk_content] if current_candidate_text_parts or not header else [header, chunk_content])

            # If current_candidate_text_parts is empty and header exists, it means we are starting a new combination.
            # The header should only be added once at the beginning of such a combination.
            if not current_candidate_text_parts and header:
                # This logic seems slightly off if header is meant per combined chunk rather than per original chunk.
                # Assuming header is for the combined block.
                # Let's simplify: if current_candidate_text_parts is empty, it might start with header.
                # The passed `chunks` list are the primary content.
                pass # Header is already in current_candidate_text_parts if it's the very start.

            test_text = chunk_delimiter.join(parts_to_test)
            token_count = len(self.tokenizer.encode(test_text))

            if token_count > max_tokens:
                # Current candidate (before adding new chunk) was likely the max fit
                if current_candidate_text_parts and (not header or len(current_candidate_text_parts) > 1 or current_candidate_text_parts[0] != header): # Check if it's more than just a header
                    if add_ellipsis_for_overflow:
                        # Check if adding ellipsis to the *previous* candidate (current_candidate_text_parts) is valid
                        # This part is tricky. Ellipsis usually indicates the *new* chunk couldn't fit.
                        # The original code added ellipsis if the *new* chunk made it overflow.
                        # Let's stick to: if adding the current 'chunk_content' overflows, then the 'current_candidate_text_parts' is finalized.
                        # If 'add_ellipsis_for_overflow' is true, it means the *dropped* chunk_content is represented by ellipsis.
                        # This seems more aligned with the original 'dropped_chunk_count'.
                        pass # Ellipsis logic might be better applied if a single chunk is too big.

                    output_combined_texts.append(chunk_delimiter.join(current_candidate_text_parts))
                    output_original_indices.append(current_candidate_indices)

                    # Start new candidate with the current chunk_content that caused overflow
                    current_candidate_text_parts = [header, chunk_content] if header else [chunk_content]
                    current_candidate_indices = [chunk_idx]

                    # If this new chunk *itself* is too large (even with header)
                    current_candidate_only_text = chunk_delimiter.join(current_candidate_text_parts)
                    if len(self.tokenizer.encode(current_candidate_only_text)) > max_tokens:
                        logging.warning(f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}) even after starting new. It will be dropped.")
                        dropped_chunk_count +=1
                        current_candidate_text_parts = [header] if header else [] # Reset for next
                        current_candidate_indices = []
                        # continue # Skip to next chunk_content
                else: # current_candidate_text_parts was empty or just header, and new chunk overflows
                    logging.warning(f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}). It will be dropped.")
                    dropped_chunk_count +=1
                    # current_candidate_text_parts remains [header] or []
                    # current_candidate_indices remains []
            else:
                # It fits, so add current chunk_content to candidate
                # If current_candidate_text_parts is empty and there's a header, it's already there.
                # If it's not empty, just append.
                # If it's empty and no header, it becomes the first part.
                if not current_candidate_text_parts: # Starting fresh
                    if header: current_candidate_text_parts = [header, chunk_content]
                    else: current_candidate_text_parts = [chunk_content]
                else: # Appending to existing candidate
                    current_candidate_text_parts.append(chunk_content)
                current_candidate_indices.append(chunk_idx)

        # Add the last candidate if it has content (more than just a header)
        if current_candidate_text_parts and (not header or len(current_candidate_text_parts) > 1 or (header and current_candidate_text_parts[0] != header) or not header):
            output_combined_texts.append(chunk_delimiter.join(current_candidate_text_parts))
            output_original_indices.append(current_candidate_indices)

        return output_combined_texts, output_original_indices, dropped_chunk_count

    # Helper for rolling_summarize (was chunk_on_delimiter)
    def _chunk_on_delimiter_for_llm(self, input_string: str,
                                     max_tokens_for_llm_input: int, # Max tokens for each final combined chunk for LLM
                                     delimiter: str) -> Tuple[List[str], List[List[int]], int]:
        # This function first splits the input_string by delimiter,
        # then combines these smaller parts into larger blocks suitable for an LLM,
        # ensuring no block exceeds max_tokens_for_llm_input.

        initial_parts = input_string.split(delimiter)

        # We need to re-add the delimiter for context, but only *between* parts, not at the very end of the LLM input block.
        # The _combine_chunks_for_llm will handle joining with its own delimiter.
        # So, we will pass parts and let _combine_chunks_for_llm join them.
        # If the original delimiter is important *within* the LLM's view of a combined chunk, it should be part of 'initial_parts'.

        # Let's adjust how parts are formed: append delimiter to all but the last part from the split.
        reconstructed_parts = []
        for i, part_text in enumerate(initial_parts):
            if i < len(initial_parts) - 1: # Not the last part
                reconstructed_parts.append(part_text + delimiter)
            else: # Last part, don't append delimiter
                if part_text: # Add if not empty
                     reconstructed_parts.append(part_text)

        # Filter out any empty strings that might result if there were multiple delimiters together
        reconstructed_parts = [p for p in reconstructed_parts if p]

        if not reconstructed_parts:
            return [], [], 0

        # Now, combine these 'reconstructed_parts' into blocks that are under 'max_tokens_for_llm_input'.
        # The delimiter for _combine_chunks_for_llm should be something that makes sense for joining these parts,
        # often an empty string if the delimiter is already part of reconstructed_parts, or a space.
        # Let's use "" as the delimiter for _combine_chunks_for_llm, as reconstructed_parts already contain their trailing delimiters.

        combined_texts_for_llm, original_indices, dropped_count = self._combine_chunks_for_llm(
            chunks=reconstructed_parts, # These are the parts including their original delimiters
            max_tokens=max_tokens_for_llm_input,
            chunk_delimiter="", # Join these parts directly
            add_ellipsis_for_overflow=True # Or make this an option
        )

        # The original `chunk_on_delimiter` added the delimiter to the end of each *combined_chunk*.
        # This might not be what we want for direct LLM input if the LLM is processing it as a whole.
        # If the goal is that each item in `combined_texts_for_llm` *looks like* it was split by `delimiter`
        # only where appropriate, the current `reconstructed_parts` logic is better.
        # The example `combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]` is removed.

        return combined_texts_for_llm, original_indices, dropped_count


# ... (end of Chunker class) ...

# The global `improved_chunking_process` function (defined in Part 1)
# already instantiates and uses `Chunker`.

# Remove old global functions that are now methods in Chunker:
# detect_language (done)
# chunk_text (done, is main dispatcher)
# chunk_text_by_words, _sentences, _paragraphs, _tokens (done)
# post_process_chunks (done)
# adaptive_chunk_size, adaptive_chunking (partially done, integrated into chunk_text and as _adaptive_... methods)
# multi_level_chunking (done)
# semantic_chunking (done)
# chunk_text_by_json, chunk_json_list, chunk_json_dict (done)
# chunk_ebook_by_chapters (done)
# chunk_xml, extract_xml_structure (done)
# rolling_summarize, combine_chunks_with_no_minimum, chunk_on_delimiter (done, integrated as _rolling_summarize and helpers)

# The following functions might still be useful as standalone utilities or need review if they should be class methods:
# - load_document (utility, can stay global or become static method if Chunker needs to load files)
# - determine_chunk_position (utility for metadata, improved_chunking_process handles relative_position differently)
# - get_chunk_metadata (largely superseded or its logic integrated elsewhere)
# - chunk_for_embedding (uses improved_chunking_process, so it's fine. Might add a Chunker arg)
# - process_document_with_metadata (uses improved_chunking_process)

# Let's refine `chunk_for_embedding` and `process_document_with_metadata`
# to potentially accept a Chunker instance or Chunker options.

def chunk_for_embedding(text: str,
                        file_name: str,
                        custom_chunk_options: Optional[Dict[str, Any]] = None,
                        tokenizer_name_or_path: str = "gpt2",
                        llm_call_function: Optional[Callable] = None, # Added
                        llm_api_config: Optional[Dict[str, Any]] = None # Added
                        ) -> List[Dict[str, Any]]:
    """
    Prepares chunks specifically for embedding, adding headers with context.
    Uses improved_chunking_process internally.
    """
    # `improved_chunking_process` will create a Chunker instance with these options
    logging.info(f"Chunking for embedding. File: {file_name}. Custom options: {custom_chunk_options}")
    chunks_from_improved_process = improved_chunking_process(
        text,
        chunk_options_dict=custom_chunk_options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function, # Pass through
        llm_api_config_for_chunker=llm_api_config      # Pass through
    )
    # The options used are now part of the Chunker instance within improved_chunking_process

    chunked_text_with_headers_list = []
    total_chunks_count = len(chunks_from_improved_process) # Get from the result

    for i, chunk_data in enumerate(chunks_from_improved_process):
        # chunk_data is {'text': ..., 'metadata': ...}
        chunk_text_content = chunk_data['text']
        chunk_metadata = chunk_data['metadata'] # This metadata is already rich

        # Determine position string (optional, could make this a helper)
        relative_pos = chunk_metadata.get('relative_position', 0.0)
        position_description = "middle"
        if relative_pos < 0.33: position_description = "beginning"
        elif relative_pos > 0.66: position_description = "end"

        # Construct header for embedding context
        # The metadata already contains chunk_index and total_chunks from improved_chunking_process
        chunk_header = f"""[DOCUMENT: {file_name}]
[CHUNK: {chunk_metadata.get('chunk_index', i+1)} OF {chunk_metadata.get('total_chunks', total_chunks_count)}]
[POSITION: This chunk is from the {position_description} of the document.]
---BEGIN CHUNK CONTENT---
"""
        # Add more from metadata if useful, e.g., chunk_metadata.get('initial_document_json_metadata')
        # or specific things from ebook/xml if the method was that.

        full_chunk_text_for_embedding = chunk_header + chunk_text_content + "\n---END CHUNK CONTENT---"

        # Create a new structure for the embedding-specific output
        embedding_chunk_data = {
            'text_for_embedding': full_chunk_text_for_embedding,
            'original_chunk_text': chunk_text_content,
            'source_document_name': file_name,
            'chunk_metadata': chunk_metadata # Carry over all the detailed metadata
        }
        chunked_text_with_headers_list.append(embedding_chunk_data)

    return chunked_text_with_headers_list


def process_document_with_metadata(text: str,
                                   chunk_options_dict: Dict[str, Any],
                                   document_metadata: Dict[str, Any],
                                   tokenizer_name_or_path: str = "gpt2",
                                   llm_call_function: Optional[Callable] = None, # Added
                                   llm_api_config: Optional[Dict[str, Any]] = None # Added
                                   ) -> Dict[str, Any]:
    """
    Processes a document, chunks it, and associates document-level metadata with the chunked output.
    """
    logging.info(f"Processing document with metadata. Options: {chunk_options_dict}")
    chunks_result = improved_chunking_process(
        text,
        chunk_options_dict=chunk_options_dict,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function, # Pass through
        llm_api_config_for_chunker=llm_api_config      # Pass through
    )
    for chunk_item in chunks_result:
        if 'document_level_metadata' not in chunk_item['metadata']:
            chunk_item['metadata']['document_level_metadata'] = {}
        chunk_item['metadata']['document_level_metadata'].update(document_metadata)
    return {
        'original_document_metadata': document_metadata,
        'chunks': chunks_result
    }


# Keep improved_chunking_process and process_document_with_metadata outside the class for now,
# or make them static methods / helper functions that instantiate and use the Chunker.
# For now, let's adapt improved_chunking_process to use the Chunker class.

def improved_chunking_process(text: str,
                              chunk_options_dict: Optional[Dict[str, Any]] = None,
                              tokenizer_name_or_path: str = "gpt2",
                              # Parameters for LLM calls if needed by a chunking method
                              llm_call_function_for_chunker: Optional[Callable] = None,
                              llm_api_config_for_chunker: Optional[Dict[str, Any]] = None
                              ) -> List[Dict[str, Any]]:
    logging.debug("Improved chunking process started...")
    logging.debug(f"Received chunk_options_dict: {chunk_options_dict}")

    chunker_instance = Chunker(options=chunk_options_dict,
                               tokenizer_name_or_path=tokenizer_name_or_path,
                                )

    # Get effective options from the chunker instance (these are now resolved)
    effective_options = chunker_instance.options.copy() # Use a copy
    # (JSON and header extraction logic using `text` and storing in `json_content_metadata`, `header_text_content`, update `processed_text`)
    # This was the previous flow:
    json_content_metadata = {}
    processed_text = text
    try:
        if processed_text.strip().startswith("{"):
            json_end_match = re.search(r"\}\s*\n", processed_text)
            if json_end_match:
                json_end_index = json_end_match.end()
                potential_json_str = processed_text[:json_end_index].strip()
                try:
                    json_content_metadata = json.loads(potential_json_str)
                    processed_text = processed_text[json_end_index:].strip()
                    logging.debug(f"Extracted JSON metadata: {json_content_metadata}")
                except json.JSONDecodeError:
                    logging.debug("Potential JSON at start, but failed to parse. Treating as normal text.")
                    json_content_metadata = {}
            else:
                logging.debug("Text starts with '{' but no clear '}\\n' end for JSON metadata.")
        else:
            logging.debug("No JSON metadata found at the beginning of the text.")
    except Exception as e_json:
        logging.warning(f"Error during JSON metadata extraction: {e_json}. Proceeding without it.")
        json_content_metadata = {}

    header_re = re.compile(r"""^ (This[ ]text[ ]was[ ]transcribed[ ]using (?:[^\n]*\n)*?\n) """, re.MULTILINE | re.VERBOSE)
    header_text_content = ""
    header_match = header_re.match(processed_text)
    if header_match:
        header_text_content = header_match.group(1)
        processed_text = processed_text[len(header_text_content):].strip()
        logging.debug(f"Extracted header text: {header_text_content}")

    if not effective_options.get('language'):
        detected_lang = chunker_instance.detect_language(processed_text)
        effective_options['language'] = str(detected_lang)
        logging.debug(f"Language for overall process set to: {effective_options['language']}")

    try:
        raw_chunks = chunker_instance.chunk_text(
            processed_text,
            method=effective_options['method'],
            llm_call_function=llm_call_function_for_chunker, # Pass it down
            llm_api_config=llm_api_config_for_chunker      # Pass it down
        )
        logging.debug(f"Created {len(raw_chunks)} raw_chunks using method {effective_options['method']}")
    except ChunkingError as ce:
        logging.error(f"ChunkingError in chunking process: {ce}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in chunking process: {e}", exc_info=True)
        raise ChunkingError(f"Unexpected error in chunking process: {e}") from e

    chunks_with_metadata_list = []
    total_chunks_count = len(raw_chunks)
    try:
        for i, chunk_item in enumerate(raw_chunks):
            actual_text_content: str
            is_json_chunk = False
            chunk_specific_metadata = {} # Initialize

            if isinstance(chunk_item, dict) and 'json' in chunk_item and 'metadata' in chunk_item : # From JSON methods
                actual_text_content = json.dumps(chunk_item['json'], ensure_ascii=False)
                chunk_specific_metadata = chunk_item['metadata']
                is_json_chunk = True
            elif isinstance(chunk_item, dict) and 'text' in chunk_item and 'metadata' in chunk_item: # From Ebook/XML methods
                actual_text_content = chunk_item['text']
                chunk_specific_metadata = chunk_item['metadata']
            elif isinstance(chunk_item, str):
                actual_text_content = chunk_item
            else:
                logging.warning(f"Unexpected chunk item type: {type(chunk_item)}. Skipping.")
                continue

            current_chunk_metadata = {
                'chunk_index': i + 1,
                'total_chunks': total_chunks_count,
                'chunk_method': effective_options['method'],
                'max_size_setting': effective_options['max_size'],
                'overlap_setting': effective_options['overlap'],
                'language': effective_options.get('language', 'unknown'),
                'relative_position': float((i + 1) / total_chunks_count) if total_chunks_count > 0 else 0.0,
                'adaptive_chunking_used': effective_options.get('adaptive', False),
            }
            current_chunk_metadata.update(chunk_specific_metadata) # Merge method-specific metadata

            if json_content_metadata:
                current_chunk_metadata['initial_document_json_metadata'] = json_content_metadata
            if header_text_content:
                current_chunk_metadata['initial_document_header_text'] = header_text_content
            current_chunk_metadata['chunk_content_hash'] = hashlib.md5(actual_text_content.encode('utf-8')).hexdigest()

            chunks_with_metadata_list.append({
                'text': actual_text_content,
                'metadata': current_chunk_metadata
            })

        logging.debug(f"Successfully created metadata for all {len(chunks_with_metadata_list)} chunks")
        return chunks_with_metadata_list
    except Exception as e:
        logging.error(f"Error creating chunk metadata: {e}", exc_info=True)
        raise ChunkingError(f"Error creating chunk metadata: {e}") from e


# Example of how other functions might change or be integrated:
# load_document can remain a utility function if needed outside, or become a static method.
def load_document(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return re.sub(r'\s+', ' ', text).strip()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading document {file_path}: {e}")
        raise