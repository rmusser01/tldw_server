# eval_Chroma_Embeddings.py
# Description: This script is used to evaluate the embeddings and chunking process for the ChromaDB model.
#
# Imports
import io
from typing import List
#
# External Imports
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from chunking_evaluation import BaseChunker, rigorous_document_search
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.evaluation_framework.base_evaluation import BaseEvaluation

#
# Local Imports
from App_Function_Libraries.Chunk_Lib import improved_chunking_process
from App_Function_Libraries.RAG.ChromaDB_Library import embedding_model, embedding_api_url
from App_Function_Libraries.RAG.Embeddings_Create import create_embeddings_batch, embedding_provider
from App_Function_Libraries.Utils.Utils import load_comprehensive_config
#
########################################################################################################################
#
# Functions:
import chardet
# FIXME


def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        print(chardet.detect(raw_data)['encoding'])
    return chardet.detect(raw_data)['encoding']


class CustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Load config here
        config = load_comprehensive_config()
        embedding_provider = config.get('Embeddings', 'embedding_provider', fallback='openai')
        embedding_model = config.get('Embeddings', 'embedding_model', fallback='text-embedding-3-small')
        embedding_api_url = config.get('Embeddings', 'api_url', fallback='')

        # Use your existing create_embeddings_batch function
        embeddings = create_embeddings_batch(input, embedding_provider, embedding_model, embedding_api_url)
        return embeddings


class CustomChunker(BaseChunker):
    def __init__(self, chunk_options):
        self.chunk_options = chunk_options

    def split_text(self, text: str) -> List[str]:
        # Use your existing improved_chunking_process function
        chunks = improved_chunking_process(text, self.chunk_options)
        return [chunk['text'] for chunk in chunks]

    def read_file(self, file_path: str) -> str:
        encoding = detect_file_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()

def utf8_file_reader(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


class CustomEvaluation(BaseEvaluation):
    def _get_chunks_and_metadata(self, splitter):
        documents = []
        metadatas = []
        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            corpus = splitter.read_file(corpus_path)

            current_documents = splitter.split_text(corpus)
            current_metadatas = []
            for document in current_documents:
                try:
                    _, start_index, end_index = rigorous_document_search(corpus, document)
                except:
                    print(f"Error in finding {document} in {corpus_id}")
                    raise Exception(f"Error in finding {document} in {corpus_id}")
                current_metadatas.append({"start_index": start_index, "end_index": end_index, "corpus_id": corpus_id})
            documents.extend(current_documents)
            metadatas.extend(current_metadatas)
        return documents, metadatas


# Instantiate your custom chunker
chunk_options = {
    'method': 'words',
    'max_size': 400,
    'overlap': 200,
    'adaptive': False,
    'multi_level': False,
    'language': 'english'
}
custom_chunker = CustomChunker(chunk_options)

# Instantiate your custom embedding function
custom_ef = CustomEmbeddingFunction()


# Evaluate the embedding function

# Evaluate the chunker
evaluation = GeneralEvaluation()
import chardet

def smart_file_reader(file_path):
    encoding = detect_file_encoding(file_path)
    with io.open(file_path, 'r', encoding=encoding) as file:
        return file.read()

# Set the custom file reader
#evaluation._file_reader = smart_file_reader


# Generate Embedding results
embedding_results = evaluation.run(custom_chunker, custom_ef)
print(f"Embedding Results:\n\t{embedding_results}")

# Generate Chunking results
chunk_results = evaluation.run(custom_chunker, custom_ef)
print(f"Chunking Results:\n\t{chunk_results}")

#
# End of File
########################################################################################################################
