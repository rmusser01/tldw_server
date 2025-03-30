# Requirements
# scikit-learn umap-learn
from itertools import chain
from typing import List, Dict

from PoC_Version.App_Function_Libraries.RAG.ChromaDB_Library import store_in_chroma, create_embedding, vector_search, chroma_client
from PoC_Version.App_Function_Libraries.Chunk_Lib import improved_chunking_process, recursive_summarize_chunks
import logging
from sklearn.mixture import GaussianMixture
import umap
from nltk.corpus import wordnet


# Logging setup
logging.basicConfig(filename='raptor.log', level=logging.DEBUG)

# FIXME
MAX_LEVELS = 3


def log_and_summarize(text, prompt):
    logging.debug(f"Summarizing text: {text[:100]} with prompt: {prompt}")
    return dummy_summarize(text, prompt)

# 1. Data Preparation
def prepare_data(content: str, media_id: int, chunk_options: dict):
    chunks = improved_chunking_process(content, chunk_options)
    embeddings = [create_embedding(chunk['text']) for chunk in chunks]
    return chunks, embeddings

# 2. Recursive Summarization
def recursive_summarization(chunks, summarize_func, custom_prompt):
    summarized_chunks = recursive_summarize_chunks(
        [chunk['text'] for chunk in chunks],
        summarize_func=summarize_func,
        custom_prompt=custom_prompt
    )
    return summarized_chunks

# Initial gen
# 3. Tree Organization
#def build_tree_structure(chunks, embeddings, collection_name, level=0):
#    if len(chunks) <= 1:
#        return chunks  # Base case: if chunks are small enough, return as is

    # Recursive case: cluster and summarize
#    summarized_chunks = recursive_summarization(chunks, summarize_func=dummy_summarize, custom_prompt="Summarize:")
#    new_chunks, new_embeddings = prepare_data(' '.join(summarized_chunks), media_id, chunk_options)

    # Store in ChromaDB
#    ids = [f"{media_id}_L{level}_chunk_{i}" for i in range(len(new_chunks))]
#    store_in_chroma(collection_name, [chunk['text'] for chunk in new_chunks], new_embeddings, ids)

    # Recursively build tree
#    return build_tree_structure(new_chunks, new_embeddings, collection_name, level+1)

# Second iteration
def build_tree_structure(chunks, collection_name, level=0):
    # Dynamic clustering
    clustered_texts = dynamic_clustering([chunk['text'] for chunk in chunks])

    # Summarize each cluster
    summarized_clusters = {}
    for cluster_id, cluster_texts in clustered_texts.items():
        summary = dummy_summarize(' '.join(cluster_texts), custom_prompt="Summarize:")
        summarized_clusters[cluster_id] = summary

    # Store summaries at current level
    ids = []
    embeddings = []
    summaries = []
    for cluster_id, summary in summarized_clusters.items():
        ids.append(f"{collection_name}_L{level}_C{cluster_id}")
        embeddings.append(create_embedding(summary))
        summaries.append(summary)

    store_in_chroma(collection_name, summaries, embeddings, ids)

    # Recursively build tree structure if necessary
    if level < MAX_LEVELS:
        for cluster_id, cluster_texts in clustered_texts.items():
            build_tree_structure(cluster_texts, collection_name, level + 1)




# Dummy summarize function (replace with actual summarization)
def dummy_summarize(text, custom_prompt, temp=None, system_prompt=None):
    return text  # Replace this with actual call to summarization model (like GPT-3.5-turbo)

# 4. Retrieval
def raptor_retrieve(query, collection_name, level=0):
    results = vector_search(collection_name, query, k=5)
    return results

# Main function integrating RAPTOR
def raptor_pipeline(media_id, content, chunk_options):
    collection_name = f"media_{media_id}_raptor"
    
    # Step 1: Prepare Data
    chunks, embeddings = prepare_data(content, media_id, chunk_options)
    
    # Step 2: Build Tree
    build_tree_structure(chunks, embeddings, collection_name)
    
    # Step 3: Retrieve Information
    query = "Your query here"
    result = raptor_retrieve(query, collection_name)
    print(result)
    
# Example usage
content = "Your long document content here"
chunk_options = {
    'method': 'sentences',
    'max_size': 300,
    'overlap': 50
}
media_id = 1
raptor_pipeline(media_id, content, chunk_options)


#
#
###################################################################################################################
#
# Additions:


def dynamic_clustering(texts, n_components=2):
    # Step 1: Convert text to embeddings
    embeddings = [create_embedding(text) for text in texts]
    
    # Step 2: Dimensionality reduction (UMAP)
    reducer = umap.UMAP(n_components=n_components)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Step 3: Find optimal number of clusters using BIC
    best_gmm = None
    best_bic = float('inf')
    n_clusters = range(2, 10)
    for n in n_clusters:
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(reduced_embeddings)
        bic = gmm.bic(reduced_embeddings)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    # Step 4: Cluster the reduced embeddings
    cluster_labels = best_gmm.predict(reduced_embeddings)
    clustered_texts = {i: [] for i in range(best_gmm.n_components)}
    for label, text in zip(cluster_labels, texts):
        clustered_texts[label].append(text)

    return clustered_texts


def tree_traversal_retrieve(query, collection_name, max_depth=3):
    logging.info(f"Starting tree traversal for query: {query}")
    results = []
    current_level = 0
    current_nodes = [collection_name + '_L0']

    while current_level <= max_depth and current_nodes:
        next_level_nodes = []
        for node_id in current_nodes:
            documents = vector_search(node_id, query, k=5)
            results.extend(documents)
            next_level_nodes.extend([doc['id'] for doc in documents])  # Assuming your doc structure includes an 'id' field
        current_nodes = next_level_nodes
        current_level += 1

    logging.info(f"Tree traversal completed with {len(results)} results")
    return results


def collapsed_tree_retrieve(query, collection_name):
    all_layers = [f"{collection_name}_L{level}" for level in range(MAX_LEVELS)]
    all_results = []

    for layer in all_layers:
        all_results.extend(vector_search(layer, query, k=5))

    # Sort and rank results by relevance
    sorted_results = sorted(all_results, key=lambda x: x['relevance'], reverse=True)  # Assuming 'relevance' is a key
    return sorted_results[:5]  # Return top 5 results

# Test collaped tree retrieval
query = "Your broad query here"
results = collapsed_tree_retrieve(query, collection_name=f"media_{media_id}_raptor")
print(results)


# Parallel processing
# pip install joblib
from joblib import Parallel, delayed

def parallel_process_chunks(chunks):
    return Parallel(n_jobs=-1)(delayed(create_embedding)(chunk['text']) for chunk in chunks)

def build_tree_structure(chunks, collection_name, level=0):
    clustered_texts = dynamic_clustering([chunk['text'] for chunk in chunks])

    summarized_clusters = {}
    for cluster_id, cluster_texts in clustered_texts.items():
        summary = dummy_summarize(' '.join(cluster_texts), custom_prompt="Summarize:")
        summarized_clusters[cluster_id] = summary

    # Parallel processing of embeddings
    embeddings = parallel_process_chunks([{'text': summary} for summary in summarized_clusters.values()])

    ids = [f"{collection_name}_L{level}_C{cluster_id}" for cluster_id in summarized_clusters.keys()]
    store_in_chroma(collection_name, list(summarized_clusters.values()), embeddings, ids)

    if len(summarized_clusters) > 1 and level < MAX_LEVELS:
        build_tree_structure(summarized_clusters.values(), collection_name, level + 1)

# Asynchronous processing
import asyncio

async def async_create_embedding(text):
    return create_embedding(text)  # Assuming create_embedding is now async

async def build_tree_structure_async(chunks, collection_name, level=0):
    clustered_texts = dynamic_clustering([chunk['text'] for chunk in chunks])

    summarized_clusters = {}
    for cluster_id, cluster_texts in clustered_texts.items():
        summary = await async_create_embedding(' '.join(cluster_texts))
        summarized_clusters[cluster_id] = summary

    embeddings = await asyncio.gather(*[async_create_embedding(summary) for summary in summarized_clusters.values()])

    ids = [f"{collection_name}_L{level}_C{cluster_id}" for cluster_id in summarized_clusters.keys()]
    store_in_chroma(collection_name, list(summarized_clusters.values()), embeddings, ids)

    if len(summarized_clusters) > 1 and level < MAX_LEVELS:
        await build_tree_structure_async(summarized_clusters.values(), collection_name, level + 1)


# User feedback Loop
def get_user_feedback(results):
    print("Please review the following results:")
    for i, result in enumerate(results):
        print(f"{i + 1}: {result['text'][:100]}...")

    feedback = input("Enter the numbers of the results that were relevant (comma-separated): ")
    relevant_indices = [int(i.strip()) - 1 for i in feedback.split(",")]
    return relevant_indices


def raptor_pipeline_with_feedback(media_id, content, chunk_options):
    # ... Existing pipeline steps ...

    query = "Your query here"
    initial_results = tree_traversal_retrieve(query, collection_name=f"media_{media_id}_raptor")
    relevant_indices = get_user_feedback(initial_results)

    if relevant_indices:
        relevant_results = [initial_results[i] for i in relevant_indices]
        refined_query = " ".join([res['text'] for res in relevant_results])
        try:
            final_results = tree_traversal_retrieve(refined_query, collection_name=f"media_{media_id}_raptor")
        except Exception as e:
            logging.error(f"Error during retrieval: {str(e)}")
            raise
        print("Refined Results:", final_results)
    else:
        print("No relevant results were found in the initial search.")


def identify_uncertain_results(results):
    threshold = 0.5  # Define a confidence threshold
    uncertain_results = [res for res in results if res['confidence'] < threshold]
    return uncertain_results


def raptor_pipeline_with_active_learning(media_id, content, chunk_options):
    # ... Existing pipeline steps ...

    query = "Your query here"
    initial_results = tree_traversal_retrieve(query, collection_name=f"media_{media_id}_raptor")
    uncertain_results = identify_uncertain_results(initial_results)

    if uncertain_results:
        print("The following results are uncertain. Please provide feedback:")
        feedback_indices = get_user_feedback(uncertain_results)
        # Use feedback to adjust retrieval or refine the query
        refined_query = " ".join([uncertain_results[i]['text'] for i in feedback_indices])
        final_results = tree_traversal_retrieve(refined_query, collection_name=f"media_{media_id}_raptor")
        print("Refined Results:", final_results)
    else:
        print("No uncertain results were found.")


# Query Expansion
def expand_query_with_synonyms(query):
    words = query.split()
    expanded_query = []
    for word in words:
        synonyms = wordnet.synsets(word)
        lemmas = set(chain.from_iterable([syn.lemma_names() for syn in synonyms]))
        expanded_query.append(" ".join(lemmas))
    return " ".join(expanded_query)


def contextual_query_expansion(query, context):
    # FIXME: Replace with actual contextual model
    expanded_terms = some_contextual_model.get_expansions(query, context)
    return query + " " + " ".join(expanded_terms)


def raptor_pipeline_with_query_expansion(media_id, content, chunk_options):
    # ... Existing pipeline steps ...

    query = "Your initial query"
    expanded_query = expand_query_with_synonyms(query)
    initial_results = tree_traversal_retrieve(expanded_query, collection_name=f"media_{media_id}_raptor")
    # ... Continue with feedback loop ...


def generate_summary_with_citations(query: str, collection_name: str):
    results = vector_search_with_citation(collection_name, query)
    # FIXME
    summary = summarize([res['text'] for res in results])
    # Deduplicate sources
    sources = list(set(res['source'] for res in results))
    return f"{summary}\n\nCitations:\n" + "\n".join(sources)


def vector_search_with_citation(collection_name: str, query: str, k: int = 10) -> List[Dict[str, str]]:
    query_embedding = create_embedding(query)
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return [{'text': doc, 'source': meta['source']} for doc, meta in zip(results['documents'], results['metadatas'])]


def generate_summary_with_footnotes(query: str, collection_name: str):
    results = vector_search_with_citation(collection_name, query)
    summary_parts = []
    citations = []
    for i, res in enumerate(results):
        summary_parts.append(f"{res['text']} [{i + 1}]")
        citations.append(f"[{i + 1}] {res['source']}")
    return " ".join(summary_parts) + "\n\nFootnotes:\n" + "\n".join(citations)


def generate_summary_with_hyperlinks(query: str, collection_name: str):
    results = vector_search_with_citation(collection_name, query)
    summary_parts = []
    for res in results:
        summary_parts.append(f'<a href="{res["source"]}">{res["text"][:100]}...</a>')
    return " ".join(summary_parts)


#
# End of Additions
############################################3############################################3##############################