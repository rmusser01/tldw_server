# RAG_Persona_Chat.py
# Description: Functions for RAG Persona Chat
#
# Imports
from typing import List, Dict, Any, Tuple
#
# External Imports
#
# Local Imports
from App_Function_Libraries.RAG.Embeddings_Create import create_embedding, embedding_provider, embedding_model, \
    embedding_api_url
from App_Function_Libraries.RAG.ChromaDB_Library import chroma_client, store_in_chroma
from App_Function_Libraries.Utils.Utils import logging


#
#######################################################################################################################
#
# RAG Chat Embeddings

def perform_vector_search_chat(query: str, relevant_chat_ids: List[int], k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a vector search within the specified chat IDs.

    Args:
        query (str): The user's query.
        relevant_chat_ids (List[int]): List of chat IDs to search within.
        k (int): Number of top results to retrieve.

    Returns:
        List[Dict[str, Any]]: List of search results with content and metadata.
    """
    try:
        # Convert chat IDs to unique identifiers used in ChromaDB
        chat_ids = [f"chat_{chat_id}" for chat_id in relevant_chat_ids]

        # Define the collection name for chat embeddings
        collection_name = "all_chat_embeddings"  # Ensure this collection exists and contains chat embeddings

        # Generate the query embedding
        query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)

        # Get the collection
        collection = chroma_client.get_collection(name=collection_name)

        # Perform the vector search
        results = collection.query(
            query_embeddings=[query_embedding],
            where={"id": {"$in": chat_ids}},  # Assuming 'id' is stored as document IDs
            n_results=k,
            include=["documents", "metadatas"]
        )

        # Process results
        search_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            search_results.append({
                "content": doc,
                "metadata": meta
            })

        return search_results
    except Exception as e:
        logging.error(f"Error in perform_vector_search_chat: {e}")
        return []


def embed_and_store_chat(chat_id: int, chat_history: List[Tuple[str, str]], conversation_name: str):
    """
    Embed and store chat messages in ChromaDB.

    Args:
        chat_id (int): The ID of the chat.
        chat_history (List[Tuple[str, str]]): List of (user_message, bot_response) tuples.
        conversation_name (str): The name of the conversation.
    """
    try:
        for idx, (user_msg, bot_msg) in enumerate(chat_history, 1):
            # Combine user and bot messages for context
            combined_content = f"User: {user_msg}\nBot: {bot_msg}"

            # Create embedding
            embedding = create_embedding(combined_content, embedding_provider, embedding_model, embedding_api_url)

            # Unique identifier for ChromaDB
            document_id = f"chat_{chat_id}_msg_{idx}"

            # Metadata with chat_id
            metadata = {"chat_id": chat_id, "message_index": idx, "conversation_name": conversation_name}

            # Store in ChromaDB
            store_in_chroma(
                collection_name="all_chat_embeddings",
                texts=[combined_content],
                embeddings=[embedding],
                ids=[document_id],
                metadatas=[metadata]
            )
            logging.debug(f"Stored chat message {idx} of chat ID {chat_id} in ChromaDB.")
    except Exception as e:
        logging.error(f"Error embedding and storing chat ID {chat_id}: {e}")

#
# End of RAG_Persona_Chat.py
#######################################################################################################################
