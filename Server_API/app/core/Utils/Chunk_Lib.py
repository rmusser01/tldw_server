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
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
#
# Import 3rd party
from openai import OpenAI
from tqdm import tqdm
from langdetect import detect
from transformers import GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#
# Import Local
from App_Function_Libraries.Utils.Utils import load_and_log_configs, logging


#
#######################################################################################################################
# Config Settings
#
#
# FIXME - Make sure it only downloads if it already exists, and does a check first.
# Ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
ensure_nltk_data()

#
# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# Load configuration
config = load_and_log_configs()
# Embedding Chunking options
chunk_options = {
    'method': config['chunking_config']['chunking_method'] or 'words',
    'max_size': config['chunking_config']['chunk_max_size'] or '400',
    'overlap': config['chunking_config']['chunk_overlap'] or '200',
    'adaptive': config['chunking_config']['adaptive_chunking'] or False,
    'multi_level': config['chunking_config']['multi_level'] or False,
    'language': config['chunking_config']['chunk_language'] or None
}

openai_api_key = config.get('API', 'openai_api_key')
#
# End of settings
#######################################################################################################################
#
# Functions:

# Create a chunking class for refactoring FIXME
# class Chunker:
#     def __init__(self, tokenizer: GPT2Tokenizer):
#         self.tokenizer = tokenizer
#
#     def detect_language(self, text: str) -> str:
#         try:
#             return detect(text)
#         except:
#             return 'en'
#
#     def chunk_text(self, text: str, method: str, max_size: int, overlap: int, language: str = None) -> List[str]:
#         if language is None:
#             language = self.detect_language(text)
#
#         if method == 'words':
#             return self.chunk_text_by_words(text, max_size, overlap, language)
#         elif method == 'sentences':
#             return self.chunk_text_by_sentences(text, max_size, overlap, language)
#         elif method == 'paragraphs':
#             return self.chunk_text_by_paragraphs(text, max_size, overlap)
#         elif method == 'tokens':
#             return self.chunk_text_by_tokens(text, max_size, overlap, language)
#         elif method == 'semantic':
#             return self.semantic_chunking(text, max_size)
#         else:
#             return [text]

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        # Default to English if detection fails
        return 'en'


def load_document(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return re.sub(r'\s+', ' ', text).strip()


def improved_chunking_process(text: str, chunk_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    logging.debug("Improved chunking process started...")
    logging.debug(f"Received chunk_options: {chunk_options}")

    # Extract JSON metadata if present
    json_content = {}
    try:
        json_end = text.index("}\n") + 1
        json_content = json.loads(text[:json_end])
        text = text[json_end:].strip()
        logging.debug(f"Extracted JSON metadata: {json_content}")
    except (ValueError, json.JSONDecodeError):
        logging.debug("No JSON metadata found at the beginning of the text")

    # Extract any additional header text
    header_match = re.match(r"(This text was transcribed using.*?)\n\n", text, re.DOTALL)
    header_text = ""
    if header_match:
        header_text = header_match.group(1)
        text = text[len(header_text):].strip()
        logging.debug(f"Extracted header text: {header_text}")

    # Make a copy of chunk_options and ensure values are correct types
    options = {}
    if chunk_options:
        try:
            options['method'] = str(chunk_options.get('method', 'words'))
            options['max_size'] = int(chunk_options.get('max_size', 2000))
            options['overlap'] = int(chunk_options.get('overlap', 0))
            # Handle language specially - it can be None
            lang = chunk_options.get('language')
            options['language'] = str(lang) if lang is not None else None
            logging.debug(f"Processed options: {options}")
        except Exception as e:
            logging.error(f"Error processing chunk options: {e}")
            raise
    else:
        options = {'method': 'words', 'max_size': 2000, 'overlap': 0, 'language': None}
        logging.debug("Using default options")

    if options.get('language') is None:
        detected_lang = detect_language(text)
        options['language'] = str(detected_lang)
        logging.debug(f"Detected language: {options['language']}")

    try:
        if options['method'] == 'json':
            chunks = chunk_text_by_json(text, max_size=options['max_size'], overlap=options['overlap'])
        else:
            chunks = chunk_text(text, options['method'], options['max_size'], options['overlap'], options['language'])
        logging.debug(f"Created {len(chunks)} chunks using method {options['method']}")
    except Exception as e:
        logging.error(f"Error in chunking process: {e}")
        raise

    chunks_with_metadata = []
    total_chunks = len(chunks)
    try:
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_index': i + 1,
                'total_chunks': total_chunks,
                'chunk_method': options['method'],
                'max_size': options['max_size'],
                'overlap': options['overlap'],
                'language': options['language'],
                'relative_position': float((i + 1) / total_chunks)
            }
            metadata.update(json_content)
            metadata['header_text'] = header_text

            if options['method'] == 'json':
                chunk_text_content = json.dumps(chunk['json'], ensure_ascii=False)
            else:
                chunk_text_content = chunk

            chunks_with_metadata.append({
                'text': chunk_text_content,
                'metadata': metadata
            })

        logging.debug(f"Successfully created metadata for all chunks")
        return chunks_with_metadata
    except Exception as e:
        logging.error(f"Error creating chunk metadata: {e}")
        raise


def multi_level_chunking(text: str, method: str, max_size: int, overlap: int, language: str) -> List[str]:
    logging.debug("Multi-level chunking process started...")
    # First level: chunk by paragraphs
    paragraphs = chunk_text_by_paragraphs(text, max_size * 2, overlap)

    # Second level: chunk each paragraph further
    chunks = []
    for para in paragraphs:
        if method == 'words':
            chunks.extend(chunk_text_by_words(para, max_words=max_size, overlap=overlap, language=language))
        elif method == 'sentences':
            chunks.extend(chunk_text_by_sentences(para, max_sentences=max_size, overlap=overlap, language=language))
        else:
            chunks.append(para)

    return chunks


# FIXME - ensure language detection occurs in each chunk function
def chunk_text(text: str, method: str, max_size: int, overlap: int, language: str = None) -> List[str]:
    if method == 'words':
        logging.debug("Chunking by words...")
        return chunk_text_by_words(text, max_words=max_size, overlap=overlap, language=language)
    elif method == 'sentences':
        logging.debug("Chunking by sentences...")
        return chunk_text_by_sentences(text, max_sentences=max_size, overlap=overlap, language=language)
    elif method == 'paragraphs':
        logging.debug("Chunking by paragraphs...")
        return chunk_text_by_paragraphs(text, max_paragraphs=max_size, overlap=overlap)
    elif method == 'tokens':
        logging.debug("Chunking by tokens...")
        return chunk_text_by_tokens(text, max_tokens=max_size, overlap=overlap)
    elif method == 'semantic':
        logging.debug("Chunking by semantic similarity...")
        return semantic_chunking(text, max_chunk_size=max_size)
    else:
        logging.warning(f"Unknown chunking method '{method}'. Returning full text as a single chunk.")
        return [text]

def determine_chunk_position(relative_position: float) -> str:
    if relative_position < 0.33:
        return "This chunk is from the beginning of the document"
    elif relative_position < 0.66:
        return "This chunk is from the middle of the document"
    else:
        return "This chunk is from the end of the document"


def chunk_text_by_words(text: str, max_words: int = 300, overlap: int = 0, language: str = None) -> List[str]:
    logging.debug("chunk_text_by_words...")
    logging.debug(f"Parameters: max_words={max_words}, overlap={overlap}, language={language}")

    try:
        # Ensure that max_words and overlap are integers.
        max_words = int(max_words)
        overlap = int(overlap)

        if language is None:
            language = detect_language(text)
            logging.debug(f"Detected language: {language}")

        if language.startswith('zh'):  # Chinese
            import jieba
            words = list(jieba.cut(text))
        elif language == 'ja':  # Japanese
            import fugashi
            tagger = fugashi.Tagger()
            words = [word.surface for word in tagger(text)]
        else:  # Default to simple splitting for other languages
            words = text.split()

        logging.debug(f"Total words: {len(words)}")

        chunks = []
        for i in range(0, len(words), max_words - overlap):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
            logging.debug(f"Created chunk {len(chunks)} with {len(chunk.split())} words")

        return post_process_chunks(chunks)
    except Exception as e:
        logging.error(f"Error in chunk_text_by_words: {e}")
        raise


def chunk_text_by_sentences(text: str, max_sentences: int = 10, overlap: int = 0, language: str = None) -> List[str]:
    logging.debug("chunk_text_by_sentences...")
    if language is None:
        language = detect_language(text)

    if language.startswith('zh'):  # Chinese
        import jieba
        # Use jieba to perform sentence segmentation
        # jieba does not support sentence segmentation out of the box
        # Use punctuation as delimiters
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    elif language == 'ja':  # Japanese
        import fugashi
        tagger = fugashi.Tagger()
        # Simple sentence segmentation based on punctuation
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:  # Default to NLTK for other languages
        try:
            sentences = sent_tokenize(text, language=language)
        except LookupError:
            logging.warning(f"Punkt tokenizer not found for language '{language}'. Using default 'english'.")
            sentences = sent_tokenize(text, language='english')

    chunks = []
    previous_overlap = []

    for i in range(0, len(sentences), max_sentences - overlap):
        current_sentences = sentences[i:i + max_sentences]
        if overlap > 0 and previous_overlap:
            current_sentences = previous_overlap + current_sentences
        chunk = ' '.join(current_sentences)
        chunks.append(chunk)
        previous_overlap = sentences[i + max_sentences - overlap:i + max_sentences] if overlap > 0 else []

    return post_process_chunks(chunks)


def chunk_text_by_paragraphs(text: str, max_paragraphs: int = 5, overlap: int = 0) -> List[str]:
    logging.debug("chunk_text_by_paragraphs...")
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs - overlap):
        chunk = '\n\n'.join(paragraphs[i:i + max_paragraphs])
        chunks.append(chunk)
    return post_process_chunks(chunks)


def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 0) -> List[str]:
    logging.debug("chunk_text_by_tokens...")
    # This is a simplified token-based chunking. For more accurate tokenization,
    # consider using a proper tokenizer like GPT-2 TokenizerFast
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_token_count = len(word) // 4 + 1  # Rough estimate of token count
        if current_token_count + word_token_count > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_token_count = sum(len(w) // 4 + 1 for w in current_chunk)

        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return post_process_chunks(chunks)
# def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 0) -> List[str]:
#     logging.debug("chunk_text_by_tokens...")
#     # Use GPT2 tokenizer for tokenization
#     tokens = tokenizer.encode(text)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens - overlap):
#         chunk_tokens = tokens[i:i + max_tokens]
#         chunk = tokenizer.decode(chunk_tokens)
#         chunks.append(chunk)
#     return post_process_chunks(chunks)


def post_process_chunks(chunks: List[str]) -> List[str]:
    return [chunk.strip() for chunk in chunks if chunk.strip()]


# FIXME - F
def get_chunk_metadata(chunk: str, full_text: str, chunk_type: str = "generic",
                      chapter_number: Optional[int] = None,
                      chapter_pattern: Optional[str] = None,
                      language: str = None) -> Dict[str, Any]:
    """
    Generate metadata for a chunk based on its position in the full text.
    """
    chunk_length = len(chunk)
    start_index = full_text.find(chunk)
    end_index = start_index + chunk_length if start_index != -1 else -1

    # Calculate a hash for the chunk
    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

    metadata = {
        'start_index': int(start_index),
        'end_index': int(end_index),
        'word_count': int(len(chunk.split())),
        'char_count': int(chunk_length),
        'chunk_type': chunk_type,
        'language': language,
        'chunk_hash': chunk_hash,
        'relative_position': float(start_index / len(full_text) if len(full_text) > 0 and start_index != -1 else 0)
    }

    if chunk_type == "chapter":
        metadata['chapter_number'] = int(chapter_number) if chapter_number is not None else None
        metadata['chapter_pattern'] = chapter_pattern

    return metadata


def process_document_with_metadata(text: str, chunk_options: Dict[str, Any],
                                   document_metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = improved_chunking_process(text, chunk_options)

    return {
        'document_metadata': document_metadata,
        'chunks': chunks
    }


# Hybrid approach, chunk each sentence while ensuring total token size does not exceed a maximum number
def chunk_text_hybrid(text: str, max_tokens: int = 1000, overlap: int = 0) -> List[str]:
    logging.debug("chunk_text_hybrid...")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        if current_length + len(tokens) > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Handle overlap
            if overlap > 0:
                overlap_tokens = tokenizer.encode(' '.join(current_chunk[-overlap:]))
                current_chunk = current_chunk[-overlap:]
                current_length = len(overlap_tokens)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += len(tokens)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return post_process_chunks(chunks)


# Thanks openai
def chunk_on_delimiter(input_string: str,
                       max_tokens: int,
                       delimiter: str) -> List[str]:
    logging.debug("chunk_on_delimiter...")
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True)
    if dropped_chunk_count > 0:
        logging.warning(f"Warning: {dropped_chunk_count} chunks were dropped due to exceeding the token limit.")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks




# FIXME
def recursive_summarize_chunks(chunks: List[str], summarize_func, custom_prompt: Optional[str] = None,
                               temp: Optional[float] = None, system_prompt: Optional[str] = None) -> List[str]:
    logging.debug("recursive_summarize_chunks...")
    summarized_chunks = []
    current_summary = ""

    logging.debug(f"Summarizing {len(chunks)} chunks recursively...")
    logging.debug(f"Temperature is set to {temp}")
    for i, chunk in enumerate(chunks):
        if i == 0:
            current_summary = summarize_func(chunk, custom_prompt, temp, system_prompt)
        else:
            combined_text = current_summary + "\n\n" + chunk
            current_summary = summarize_func(combined_text, custom_prompt, temp, system_prompt)

        summarized_chunks.append(current_summary)

    return summarized_chunks


# Sample text for testing
sample_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
concerned with the interactions between computers and human language, in particular how to program computers 
to process and analyze large amounts of natural language data. The result is a computer capable of "understanding" 
the contents of documents, including the contextual nuances of the language within them. The technology can then 
accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
and natural language generation.

Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled 
"Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.
"""

# Example usage of different chunking methods
# print("Chunking by words:")
# print(chunk_text_by_words(sample_text, max_words=50))
#
# print("\nChunking by sentences:")
# print(chunk_text_by_sentences(sample_text, max_sentences=2))
#
# print("\nChunking by paragraphs:")
# print(chunk_text_by_paragraphs(sample_text, max_paragraphs=1))
#
# print("\nChunking by tokens:")
# print(chunk_text_by_tokens(sample_text, max_tokens=50))
#
# print("\nHybrid chunking:")
# print(chunk_text_hybrid(sample_text, max_tokens=50))



#######################################################################################################################
#
# Experimental Semantic Chunking
#

# Chunk text into segments based on semantic similarity
def count_units(text: str, unit: str = 'words') -> int:
    if unit == 'words':
        return len(text.split())
    elif unit == 'tokens':
        return len(tokenizer.encode(text))
    elif unit == 'characters':
        return len(text)
    else:
        raise ValueError("Invalid unit. Choose 'words', 'tokens', or 'characters'.")



def semantic_chunking(text: str, max_chunk_size: int = 2000, unit: str = 'words') -> List[str]:
    logging.debug("semantic_chunking...")
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    chunks = []
    current_chunk = []
    current_size = 0

    for i, sentence in enumerate(sentences):
        sentence_size = count_units(sentence, unit)
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Use last 3 sentences for overlap
            current_chunk = current_chunk[-3:]
            current_size = count_units(' '.join(current_chunk), unit)

        current_chunk.append(sentence)
        current_size += sentence_size

        if i + 1 < len(sentences):
            current_vector = sentence_vectors[i]
            next_vector = sentence_vectors[i + 1]
            similarity = cosine_similarity(current_vector, next_vector)[0][0]
            if similarity < 0.5 and current_size >= max_chunk_size // 2:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-3:]
                current_size = count_units(' '.join(current_chunk), unit)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def semantic_chunk_long_file(file_path: str, max_chunk_size: int = 1000, overlap: int = 100, unit: str = 'words') -> Optional[List[str]]:
    logging.debug("semantic_chunk_long_file...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = semantic_chunking(content, max_chunk_size, unit)
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text file: {str(e)}")
        return None

#
#
#######################################################################################################################


#######################################################################################################################
#
#  Embedding Chunking

def chunk_for_embedding(text: str, file_name: str, custom_chunk_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    options = chunk_options.copy()
    if custom_chunk_options:
        options.update(custom_chunk_options)

    logging.info(f"Chunking options: {options}")
    chunks = improved_chunking_process(text, options)
    total_chunks = len(chunks)
    logging.info(f"Total chunks created: {total_chunks}")

    chunked_text_with_headers = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk['text']
        chunk_position = determine_chunk_position(chunk['metadata']['relative_position'])
        chunk_header = f"""
        Original Document: {file_name}
        Chunk: {i} of {total_chunks}
        Position: {chunk_position}

        --- Chunk Content ---
        """

        full_chunk_text = chunk_header + chunk_text
        chunk['text'] = full_chunk_text
        chunk['metadata']['file_name'] = file_name
        chunked_text_with_headers.append(chunk)

    return chunked_text_with_headers

#
# End of Embedding Chunking
#######################################################################################################################


#######################################################################################################################
#
# JSON Chunking

# FIXME
def chunk_text_by_json(text: str, max_size: int = 1000, overlap: int = 0) -> List[Dict[str, Any]]:
    """
    Chunk JSON-formatted text into smaller JSON chunks while preserving structure.

    Parameters:
        - text (str): The JSON-formatted text to be chunked.
        - max_size (int): Maximum number of items or characters per chunk.
        - overlap (int): Number of items or characters to overlap between chunks.

    Returns:
        - List[Dict[str, Any]]: A list of chunks with their metadata.
    """
    logging.debug("chunk_text_by_json started...")
    try:
        json_data = json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON data: {e}")
        raise ValueError(f"Invalid JSON data: {e}")

    # Determine if JSON data is a list or a dict
    if isinstance(json_data, list):
        return chunk_json_list(json_data, max_size, overlap)
    elif isinstance(json_data, dict):
        return chunk_json_dict(json_data, max_size, overlap)
    else:
        logging.error("Unsupported JSON structure. Only JSON objects and arrays are supported.")
        raise ValueError("Unsupported JSON structure. Only JSON objects and arrays are supported.")


def chunk_json_list(json_list: List[Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Chunk a JSON array into smaller chunks.

    Parameters:
        - json_list (List[Any]): The JSON array to be chunked.
        - max_size (int): Maximum number of items per chunk.
        - overlap (int): Number of items to overlap between chunks.

    Returns:
        - List[Dict[str, Any]]: A list of JSON chunks with metadata.
    """
    logging.debug("chunk_json_list started...")
    chunks = []
    total_items = len(json_list)
    step = max_size - overlap
    if step <= 0:
        raise ValueError("max_size must be greater than overlap.")

    for i in range(0, total_items, step):
        chunk = json_list[i:i + max_size]
        metadata = {
            'chunk_index': i // step + 1,
            'total_chunks': (total_items + step - 1) // step,
            'chunk_method': 'json_list',
            'max_size': max_size,
            'overlap': overlap,
            'relative_position': i / total_items
        }
        chunks.append({
            'json': chunk,
            'metadata': metadata
        })

    logging.debug(f"chunk_json_list created {len(chunks)} chunks.")
    return chunks



def chunk_json_dict(json_dict: Dict[str, Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Chunk a JSON object into smaller chunks based on its 'data' key while preserving other keys like 'metadata'.

    Parameters:
        - json_dict (Dict[str, Any]): The JSON object to be chunked.
        - max_size (int): Maximum number of key-value pairs per chunk in the 'data' section.
        - overlap (int): Number of key-value pairs to overlap between chunks.

    Returns:
        - List[Dict[str, Any]]: A list of JSON chunks with metadata.
    """
    logging.debug("chunk_json_dict started...")

    # Preserve non-chunked sections
    preserved_keys = ['metadata']
    preserved_data = {key: value for key, value in json_dict.items() if key in preserved_keys}

    # Identify the chunkable section
    chunkable_key = 'data'
    if chunkable_key not in json_dict or not isinstance(json_dict[chunkable_key], dict):
        logging.error("No chunkable 'data' section found in JSON dictionary.")
        raise ValueError("No chunkable 'data' section found in JSON dictionary.")

    chunkable_data = json_dict[chunkable_key]
    data_keys = list(chunkable_data.keys())
    total_keys = len(data_keys)
    chunks = []
    step = max_size - overlap
    if step <= 0:
        raise ValueError("max_size must be greater than overlap.")

    # Adjust the loop to prevent creating an extra chunk
    for i in range(0, total_keys, step):
        chunk_keys = data_keys[i:i + max_size]

        # Handle overlap
        if i != 0 and overlap > 0:
            overlap_keys = data_keys[i - overlap:i]
            chunk_keys = overlap_keys + chunk_keys

        # Remove duplicate keys caused by overlap
        unique_chunk_keys = []
        seen_keys = set()
        for key in chunk_keys:
            if key not in seen_keys:
                unique_chunk_keys.append(key)
                seen_keys.add(key)

        chunk_data = {key: chunkable_data[key] for key in unique_chunk_keys}

        metadata = {
            'chunk_index': (i // step) + 1,
            'total_chunks': (total_keys + step - 1) // step,
            'chunk_method': 'json_dict',
            'max_size': max_size,
            'overlap': overlap,
            'language': 'english',  # Assuming English; modify as needed
            'relative_position': (i // step + 1) / ((total_keys + step - 1) // step)
        }

        # Merge preserved data into metadata
        metadata.update(preserved_data.get('metadata', {}))

        # Create the chunk with preserved data
        chunk = {
            'metadata': preserved_data,
            'data': chunk_data
        }

        chunks.append({
            'json': chunk,
            'metadata': metadata
        })

    logging.debug(f"chunk_json_dict created {len(chunks)} chunks.")
    return chunks


#
# End of JSON Chunking
#######################################################################################################################


#######################################################################################################################
#
# OpenAI Rolling Summarization
#

client = OpenAI(api_key=openai_api_key)
def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


# This function combines text chunks into larger blocks without exceeding a specified token count.
#   It returns the combined chunks, their original indices, and the number of dropped chunks due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter: str = "\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow: bool = False,
) -> Tuple[List[str], List[List[int]], int]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = [header] if header else []  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if not header else [header, chunk]
        combined_text = chunk_delimiter.join(candidate + chunk_with_header)
        token_count = len(tokenizer.encode(combined_text))
        if token_count > max_tokens:
            if add_ellipsis_for_overflow and len(candidate) > 0:
                ellipsis_text = chunk_delimiter.join(candidate + ["..."])
                if len(tokenizer.encode(ellipsis_text)) <= max_tokens:
                    candidate = candidate + ["..."]
                    dropped_chunk_count += 1
            if len(candidate) > 0:
                output.append(chunk_delimiter.join(candidate))
                output_indices.append(candidate_indices)
                candidate = chunk_with_header
                candidate_indices = [chunk_i]
            else:
                logging.warning(f"Single chunk at index {chunk_i} exceeds max_tokens and will be dropped.")
                dropped_chunk_count += 1
        else:
            candidate.extend(chunk_with_header)
            candidate_indices.append(chunk_i)

    if candidate:
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def rolling_summarize(text: str,
                      detail: float = 0,
                      model: str = 'gpt-4o',
                      additional_instructions: Optional[str] = None,
                      minimum_chunk_size: Optional[int] = 500,
                      chunk_delimiter: str = ".",
                      summarize_recursively: bool = False,
                      verbose: bool = False) -> str:
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
        - text (str): The text to be summarized.
        - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
        - additional_instructions (Optional[str], optional): Additional instructions for the model.
        - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks.
        - chunk_delimiter (str, optional): The delimiter used to split the text into chunks.
        - summarize_recursively (bool, optional): If True, summaries are generated recursively.
        - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count
    based on the `detail` parameter. It then splits the text into chunks and summarizes each chunk. If
    `summarize_recursively` is True, each summary is based on the previous summaries, adding more context to the
    summarization process. The function returns a compiled summary of all chunks.
    """

    # Check detail is set correctly
    assert 0 <= detail <= 1, "Detail must be between 0 and 1."

    # Interpolate the number of chunks based on the detail parameter
    text_length = len(tokenizer.encode(text))
    max_chunks = text_length // minimum_chunk_size if minimum_chunk_size else 10
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # Adjust chunk_size based on interpolated number of chunks
    chunk_size = max(minimum_chunk_size, text_length // num_chunks) if num_chunks else text_length
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenizer.encode(x)) for x in text_chunks]} tokens.")

    # Set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Summarizing chunks")):
        if summarize_recursively and accumulated_summaries:
            # Combine previous summary with current chunk for recursive summarization
            combined_text = accumulated_summaries[-1] + "\n\n" + chunk
            user_message_content = f"Previous summary and new content to summarize:\n\n{combined_text}"
        else:
            user_message_content = chunk

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    final_summary = '\n\n'.join(accumulated_summaries)
    return final_summary

#
#
#######################################################################################################################
#
# Ebook Chapter Chunking


def chunk_ebook_by_chapters(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    logging.debug("chunk_ebook_by_chapters")
    max_chunk_size = int(chunk_options.get('max_size', 300))
    overlap = int(chunk_options.get('overlap', 0))
    custom_pattern = chunk_options.get('custom_chapter_pattern', None)

    # List of chapter heading patterns to try, in order
    chapter_patterns = [
        custom_pattern,
        r'^#{1,2}\s+',  # Markdown style: '# ' or '## '
        r'^Chapter\s+\d+',  # 'Chapter ' followed by numbers
        r'^\d+\.\s+',  # Numbered chapters: '1. ', '2. ', etc.
        r'^[A-Z\s]+$'  # All caps headings
    ]

    chapter_positions = []
    used_pattern = None

    for pattern in chapter_patterns:
        if pattern is None:
            continue
        chapter_regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        chapter_positions = [match.start() for match in chapter_regex.finditer(text)]
        if chapter_positions:
            used_pattern = pattern
            break

    # If no chapters found, return the entire content as one chunk
    if not chapter_positions:
        metadata = get_chunk_metadata(
            chunk=text,
            full_text=text,
            chunk_type="whole_document",
            language=chunk_options.get('language', 'english')
        )
        return [{'text': text, 'metadata': metadata}]

    # Split content into chapters
    chunks = []
    for i in range(len(chapter_positions)):
        start = chapter_positions[i]
        end = chapter_positions[i + 1] if i + 1 < len(chapter_positions) else None
        chapter = text[start:end]

        # Apply overlap if specified
        if overlap > 0 and i > 0:
            overlap_start = max(0, chapter_positions[i] - overlap)
            chapter = text[overlap_start:end]

        chunks.append(chapter)

    # Post-process chunks
    processed_chunks = post_process_chunks(chunks)

    # Add metadata to chunks
    chunks_with_metadata = []
    for i, chunk in enumerate(processed_chunks):
        metadata = get_chunk_metadata(
            chunk=chunk,
            full_text=text,
            chunk_type="chapter",
            chapter_number=i + 1,
            chapter_pattern=used_pattern,
            language=chunk_options.get('language', 'english')
        )
        chunks_with_metadata.append({'text': chunk, 'metadata': metadata})

    return chunks_with_metadata

#
# End of ebook chapter chunking
#######################################################################################################################
#
# XML Chunking

def extract_xml_structure(element, path=""):
    """
    Recursively extract XML structure and content.
    Returns a list of (path, text) tuples.
    """
    results = []
    current_path = f"{path}/{element.tag}" if path else element.tag

    # Get direct text content
    if element.text and element.text.strip():
        results.append((current_path, element.text.strip()))

    # Process attributes if any
    if element.attrib:
        for key, value in element.attrib.items():
            results.append((f"{current_path}/@{key}", value))

    # Process child elements
    for child in element:
        results.extend(extract_xml_structure(child, current_path))

    return results


def chunk_xml(xml_text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enhanced XML chunking that preserves structure and hierarchy.
    Processes XML content into chunks while maintaining structural context.

    Args:
        xml_text (str): The XML content as a string
        chunk_options (Dict[str, Any]): Configuration options including:
            - max_size (int): Maximum chunk size (default: 1000)
            - overlap (int): Number of overlapping elements (default: 0)
            - method (str): Chunking method (default: 'xml')
            - language (str): Content language (default: 'english')

    Returns:
        List[Dict[str, Any]]: List of chunks, each containing:
            - text: The chunk content
            - metadata: Chunk metadata including XML paths and chunking info
    """
    logging.debug("Starting XML chunking process...")

    try:
        # Parse XML content
        root = ET.fromstring(xml_text)
        chunks = []

        # Get chunking parameters with defaults
        max_size = chunk_options.get('max_size', 1000)
        overlap = chunk_options.get('overlap', 0)
        language = chunk_options.get('language', 'english')

        logging.debug(f"Chunking parameters - max_size: {max_size}, overlap: {overlap}, language: {language}")

        # Extract full structure with hierarchy
        xml_content = extract_xml_structure(root)
        logging.debug(f"Extracted {len(xml_content)} XML elements")

        # Initialize chunking variables
        current_chunk = []
        current_size = 0
        chunk_count = 0

        # Process XML content into chunks
        for path, content in xml_content:
            # Calculate content size (by words)
            content_size = len(content.split())

            # Check if adding this content would exceed max_size
            if current_size + content_size > max_size and current_chunk:
                # Create chunk from current content
                chunk_text = '\n'.join(f"{p}: {c}" for p, c in current_chunk)
                chunk_count += 1

                # Create chunk with metadata
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'paths': [p for p, _ in current_chunk],
                        'chunk_method': 'xml',
                        'chunk_index': chunk_count,
                        'max_size': max_size,
                        'overlap': overlap,
                        'language': language,
                        'root_tag': root.tag,
                        'xml_attributes': dict(root.attrib)
                    }
                })

                # Handle overlap if specified
                if overlap > 0:
                    # Keep last few items for overlap
                    overlap_items = current_chunk[-overlap:]
                    current_chunk = overlap_items
                    current_size = sum(len(c.split()) for _, c in overlap_items)
                    logging.debug(f"Created overlap chunk with {len(overlap_items)} items")
                else:
                    current_chunk = []
                    current_size = 0

            # Add current content to chunk
            current_chunk.append((path, content))
            current_size += content_size

        # Process final chunk if content remains
        if current_chunk:
            chunk_text = '\n'.join(f"{p}: {c}" for p, c in current_chunk)
            chunk_count += 1

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'paths': [p for p, _ in current_chunk],
                    'chunk_method': 'xml',
                    'chunk_index': chunk_count,
                    'max_size': max_size,
                    'overlap': overlap,
                    'language': language,
                    'root_tag': root.tag,
                    'xml_attributes': dict(root.attrib)
                }
            })

        # Update total chunks count in metadata
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = chunk_count

        logging.debug(f"XML chunking complete. Created {len(chunks)} chunks")
        return chunks

    except ET.ParseError as e:
        logging.error(f"XML parsing error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during XML chunking: {str(e)}")
        raise

#
# End of XML Chunking
#######################################################################################################################

#######################################################################################################################
#
# Functions for adapative chunking:

# FIXME - punkt

def adaptive_chunk_size(text: str, base_size: int = 1000, min_size: int = 500, max_size: int = 2000) -> int:
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    if not sentences:
        return base_size

    # Calculate average sentence length
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

    # Adjust chunk size based on average sentence length
    if avg_sentence_length < 10:
        size_factor = 1.2  # Increase chunk size for short sentences
    elif avg_sentence_length > 20:
        size_factor = 0.8  # Decrease chunk size for long sentences
    else:
        size_factor = 1.0

    # Calculate adaptive chunk size
    adaptive_size = int(base_size * size_factor)

    # Ensure chunk size is within bounds
    return max(min_size, min(adaptive_size, max_size))


def adaptive_chunk_size_non_punkt(text: str, base_size: int, min_size: int = 100, max_size: int = 2000) -> int:
    # Adaptive logic: adjust chunk size based on text complexity
    words = text.split()
    if not words:
        return base_size  # Return base_size if text is empty

    avg_word_length = sum(len(word) for word in words) / len(words)

    if avg_word_length > 6:  # Threshold for "complex" text
        adjusted_size = int(base_size * 0.8)  # Reduce chunk size for complex text
    elif avg_word_length < 4:  # Threshold for "simple" text
        adjusted_size = int(base_size * 1.2)  # Increase chunk size for simple text
    else:
        adjusted_size = base_size

    # Ensure the chunk size is within the specified range
    return max(min_size, min(adjusted_size, max_size))


def adaptive_chunking(text: str, base_size: int = 1000, min_size: int = 500, max_size: int = 2000) -> List[str]:
    logging.debug("adaptive_chunking...")
    chunk_size = adaptive_chunk_size(text, base_size, min_size, max_size)
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# FIXME - usage example
# chunk_options = {
#     'method': 'words',  # or any other method
#     'base_size': 1000,
#     'min_size': 100,
#     'max_size': 2000,
#     'adaptive': True,
#     'language': 'en'
# }
#chunks = improved_chunking_process(your_text, chunk_options)

# Example of chunking a document with metadata
# document_metadata = {
#     'title': 'Example Document',
#     'author': 'John Doe',
#     'creation_date': '2023-06-14',
#     'source': 'https://example.com/document',
#     'document_type': 'article'
# }
#
# chunk_options = {
#     'method': 'sentences',
#     'base_size': 1000,
#     'adaptive': True,
#     'language': 'en'
# }
#
# processed_document = process_document_with_metadata(your_text, chunk_options, document_metadata)


#
# End of Chunking Library
#######################################################################################################################