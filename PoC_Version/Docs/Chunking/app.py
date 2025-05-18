from flask import Flask, request, jsonify, render_template
import re
import logging
import nltk
from typing import List
from langdetect import detect

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Default text (you can add more default texts as needed)
default_prose = """One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]

You can't understand the world without understanding the concept of superlinear returns. And if you're ambitious you definitely should, because this will be the wave you surf on.

It may seem as if there are a lot of different situations with superlinear returns, but as far as I can tell they reduce to two fundamental causes: exponential growth and thresholds.

The most obvious case of superlinear returns is when you're working on something that grows exponentially. For example, growing bacterial cultures. When they grow at all, they grow exponentially. But they're tricky to grow. Which means the difference in outcome between someone who's adept at it and someone who's not is very great.

Startups can also grow exponentially, and we see the same pattern there. Some manage to achieve high growth rates. Most don't. And as a result you get qualitatively different outcomes: the companies with high growth rates tend to become immensely valuable, while the ones with lower growth rates may not even survive.

Y Combinator encourages founders to focus on growth rate rather than absolute numbers. It prevents them from being discouraged early on, when the absolute numbers are still low. It also helps them decide what to focus on: you can use growth rate as a compass to tell you how to evolve the company. But the main advantage is that by focusing on growth rate you tend to get something that grows exponentially.

YC doesn't explicitly tell founders that with growth rate "you get out what you put in," but it's not far from the truth. And if growth rate were proportional to performance, then the reward for performance p over time t would be proportional to pt.

Even after decades of thinking about this, I find that sentence startling."""

def detect_language(text: str) -> str:
    return detect(text)

def post_process_chunks(chunks: List[str]) -> List[str]:
    # Implement any post-processing logic here if needed
    return chunks

def chunk_text_by_words(text: str, max_words: int = 300, overlap: int = 0, language: str = None) -> List[str]:
    logging.debug("chunk_text_by_words...")
    if language is None:
        language = detect_language(text)

    if language.startswith('zh'):  # Chinese
        import jieba
        words = list(jieba.cut(text))
    elif language == 'ja':  # Japanese
        import fugashi
        tagger = fugashi.Tagger()
        words = [word.surface for word in tagger(text)]
    else:  # Default to simple splitting for other languages
        words = text.split()

    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)
    return post_process_chunks(chunks)

def chunk_text_by_sentences(text: str, max_sentences: int = 10, overlap: int = 0, language: str = None) -> List[str]:
    logging.debug("chunk_text_by_sentences...")
    if language is None:
        language = detect_language(text)

    if language.startswith('zh'):  # Chinese
        import jieba
        sentences = list(jieba.cut(text, cut_all=False))
    elif language == 'ja':  # Japanese
        import fugashi
        tagger = fugashi.Tagger()
        sentences = [word.surface for word in tagger(text) if word.feature.pos1 in ['記号', '補助記号'] and word.surface.strip()]
    else:  # Default to NLTK for other languages
        sentences = nltk.sent_tokenize(text)

    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chunk', methods=['POST'])
def chunk_text():
    data = request.json
    logger.debug(f"Received data: {data}")
    text = data.get('text', default_prose)
    chunk_size = data.get('chunkSize', 300)
    overlap = data.get('overlap', 0)
    splitter_type = data.get('splitter', 'words')
    
    logger.debug(f"Chunking with: splitter={splitter_type}, chunk_size={chunk_size}, overlap={overlap}")
    
    if splitter_type == 'words':
        chunks = chunk_text_by_words(text, chunk_size, overlap)
    elif splitter_type == 'sentences':
        chunks = chunk_text_by_sentences(text, chunk_size, overlap)
    elif splitter_type == 'paragraphs':
        chunks = chunk_text_by_paragraphs(text, chunk_size, overlap)
    elif splitter_type == 'tokens':
        chunks = chunk_text_by_tokens(text, chunk_size, overlap)
    else:
        return jsonify({'error': 'Invalid splitter type'}), 400
    
    logger.debug(f"Number of chunks created: {len(chunks)}")
    
    # Process chunks to include start and end indices
    processed_chunks = []
    current_index = 0
    for chunk in chunks:
        chunk_length = len(chunk)
        end_index = current_index + chunk_length
        processed_chunks.append({
            'text': chunk,
            'startIndex': current_index,
            'endIndex': end_index,
            'overlapWithNext': overlap if end_index < len(text) else 0
        })
        current_index = end_index - overlap
    
    logger.debug(f"Processed chunks: {processed_chunks}")

    response = {
        'chunks': processed_chunks,
        'totalCharacters': len(text),
        'numberOfChunks': len(chunks),
        'averageChunkSize': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    }
    logger.debug(f"Sending response: {response}")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)