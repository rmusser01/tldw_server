# First gen

# Install the necessary libraries
# !pip install transformers
# !pip install sentence-transformers
# !pip install torch
# !pip install requests
# !pip install bs4

import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch

# Step 1: Load Models for Summarization and Similarity
model_name = "facebook/bart-large-cnn"  # Summarization model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Sentence similarity model
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Step 2: Define Retrieval Evaluator
def evaluate_retrieval(query, retrieved_docs):
    """
    Evaluate the relevance of retrieved documents using cosine similarity
    with sentence embeddings.
    """
    query_embedding = similarity_model.encode(query, convert_to_tensor=True)
    doc_embeddings = similarity_model.encode(retrieved_docs, convert_to_tensor=True)

    # Calculate cosine similarity between the query and each document
    similarities = [util.pytorch_cos_sim(query_embedding, doc_embedding).item() for doc_embedding in doc_embeddings]

    # Set a threshold for relevance (adjustable)
    relevance_threshold = 0.5
    relevance_scores = ['Correct' if sim > relevance_threshold else 'Incorrect' for sim in similarities]

    return relevance_scores


# Step 3: Knowledge Refinement (Decompose-then-Recompose)
def decompose_then_recompose(retrieved_docs):
    """
    Refine the retrieved documents by summarizing their key information.
    """
    refined_knowledge = []
    for doc in retrieved_docs:
        summary = summarizer(doc, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
        refined_knowledge.append(summary)
    return refined_knowledge


# Step 4: Web Search for External Knowledge
def web_search(query):
    """
    Perform a web search to retrieve additional external knowledge if the
    retrieved documents are not relevant.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract URLs from search results (simplified)
    links = []
    for item in soup.find_all('a'):
        link = item.get('href')
        if link and "http" in link:
            links.append(link)
    return links[:5]  # Return the first 5 URLs


# Step 5: Generate Final Output
def generate_final_output(query, refined_knowledge):
    """
    Generate the final output summary using the refined knowledge.
    """
    combined_knowledge = " ".join(refined_knowledge)
    final_summary = summarizer(combined_knowledge, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
    return final_summary


# Step 6: CRAG Workflow Integration
def crag_workflow(query, retrieved_docs):
    """
    Full CRAG workflow integrating evaluation, knowledge refinement,
    and web search to generate a robust output summary.
    """
    # Step 1: Evaluate retrieval
    relevance_scores = evaluate_retrieval(query, retrieved_docs)

    if 'Correct' in relevance_scores:
        # Step 2: Decompose-then-Recompose for correct documents
        refined_knowledge = decompose_then_recompose(
            [doc for doc, score in zip(retrieved_docs, relevance_scores) if score == 'Correct'])
    else:
        # Step 3: Web search if retrieval is incorrect
        web_results = web_search(query)
        refined_knowledge = decompose_then_recompose(web_results)

    # Step 4: Generate final output
    final_summary = generate_final_output(query, refined_knowledge)

    return final_summary


# Example Usage
if __name__ == "__main__":
    # Example query and retrieved documents
    query = "What are the latest advancements in renewable energy?"
    retrieved_docs = [
        "Renewable energy is becoming increasingly important in today's world...",
        "Solar energy has seen significant advancements in the past decade...",
        "Wind energy technology is rapidly evolving, with new innovations expected soon..."
    ]

    # Perform the CRAG workflow
    final_summary = crag_workflow(query, retrieved_docs)
    print("Final Summary:", final_summary)
