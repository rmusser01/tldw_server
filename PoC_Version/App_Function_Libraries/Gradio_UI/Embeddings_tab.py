# Embeddings_tabc.py
# Description: This file contains the code for the RAG Chat tab in the Gradio UI
#
# Imports
import json
import os
#
# External Imports
import gradio as gr
import numpy as np
from tqdm import tqdm
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import get_all_content_from_database, get_all_conversations, \
    get_conversation_text, get_note_by_id
from App_Function_Libraries.DB.RAG_QA_Chat_DB import get_all_notes
from App_Function_Libraries.RAG.ChromaDB_Library import chroma_client, \
    store_in_chroma, situate_context
from App_Function_Libraries.RAG.Embeddings_Create import create_embedding, create_embeddings_batch
from App_Function_Libraries.Chunk_Lib import improved_chunking_process, chunk_for_embedding
from App_Function_Libraries.Utils.Utils import load_and_log_configs, logging


#
########################################################################################################################
#
# Functions:

def create_embeddings_tab():
    # Load configuration first
    config = load_and_log_configs()
    if not config:
        raise ValueError("Could not load configuration")

    # Get database paths from config
    db_config = config['db_config']
    media_db_path = db_config['sqlite_path']
    rag_qa_db_path = os.path.join(os.path.dirname(media_db_path), "rag_qa.db")
    character_chat_db_path = os.path.join(os.path.dirname(media_db_path), "chatDB.db")
    chroma_db_path = db_config['chroma_db_path']

    with gr.TabItem("Create Embeddings", visible=True):
        gr.Markdown("# Create Embeddings for All Content")

        with gr.Row():
            with gr.Column():
                # Database selection at the top
                database_selection = gr.Radio(
                    choices=["Media DB", "RAG Chat", "Character Chat"],
                    label="Select Content Source",
                    value="Media DB",
                    info="Choose which database to create embeddings from"
                )

                # Add database path display
                current_db_path = gr.Textbox(
                    label="Current Database Path",
                    value=media_db_path,
                    interactive=False
                )

                embedding_provider = gr.Radio(
                    choices=["huggingface", "local", "openai"],
                    label="Select Embedding Provider",
                    value=config['embedding_config']['embedding_provider'] or "huggingface"
                )
                gr.Markdown("Note: Local provider requires a running Llama.cpp/llamafile server.")
                gr.Markdown("OpenAI provider requires a valid API key.")

                huggingface_model = gr.Dropdown(
                    choices=[
                        "jinaai/jina-embeddings-v3",
                        "Alibaba-NLP/gte-large-en-v1.5",
                        "dunzhang/setll_en_400M_v5",
                        "custom"
                    ],
                    label="Hugging Face Model",
                    value="jinaai/jina-embeddings-v3",
                    visible=True
                )

                openai_model = gr.Dropdown(
                    choices=[
                        "text-embedding-3-small",
                        "text-embedding-3-large"
                    ],
                    label="OpenAI Embedding Model",
                    value="text-embedding-3-small",
                    visible=False
                )

                custom_embedding_model = gr.Textbox(
                    label="Custom Embedding Model",
                    placeholder="Enter your custom embedding model name here",
                    visible=False
                )

                embedding_api_url = gr.Textbox(
                    label="API URL (for local provider)",
                    value=config['embedding_config']['embedding_api_url'],
                    visible=False
                )

                # Add chunking options with config defaults
                chunking_method = gr.Dropdown(
                    choices=["words", "sentences", "paragraphs", "tokens", "semantic"],
                    label="Chunking Method",
                    value="words"
                )
                max_chunk_size = gr.Slider(
                    minimum=1, maximum=8000, step=1,
                    value=config['embedding_config']['chunk_size'],
                    label="Max Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0, maximum=4000, step=1,
                    value=config['embedding_config']['chunk_overlap'],
                    label="Chunk Overlap"
                )
                adaptive_chunking = gr.Checkbox(
                    label="Use Adaptive Chunking",
                    value=False
                )

                create_button = gr.Button("Create Embeddings")

            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=10)
                progress = gr.Progress()

        def update_provider_options(provider):
            if provider == "huggingface":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif provider == "local":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            else:  # OpenAI
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        def update_huggingface_options(model):
            if model == "custom":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        def update_database_path(database_type):
            if database_type == "Media DB":
                return media_db_path
            elif database_type == "RAG Chat":
                return rag_qa_db_path
            else:  # Character Chat
                return character_chat_db_path

        def create_all_embeddings(provider, hf_model, openai_model, custom_model, api_url, method,
                                max_size, overlap, adaptive, database_type, progress=gr.Progress()):
            try:
                # Initialize content based on database selection
                if database_type == "Media DB":
                    all_content = get_all_content_from_database()
                    content_type = "media"
                elif database_type == "RAG Chat":
                    all_content = []
                    page = 1
                    while True:
                        conversations, total_pages, _ = get_all_conversations(page=page)
                        if not conversations:
                            break
                        all_content.extend([{
                            'id': conv['conversation_id'],
                            'content': get_conversation_text(conv['conversation_id']),
                            'title': conv['title'],
                            'type': 'conversation'
                        } for conv in conversations])
                        progress(page / total_pages, desc=f"Loading conversations... Page {page}/{total_pages}")
                        page += 1
                else:  # Character Chat
                    all_content = []
                    page = 1
                    while True:
                        notes, total_pages, _ = get_all_notes(page=page)
                        if not notes:
                            break
                        all_content.extend([{
                            'id': note['id'],
                            'content': f"{note['title']}\n\n{note['content']}",
                            'conversation_id': note['conversation_id'],
                            'type': 'note'
                        } for note in notes])
                        progress(page / total_pages, desc=f"Loading notes... Page {page}/{total_pages}")
                        page += 1

                if not all_content:
                    return "No content found in the selected database."

                chunk_options = {
                    'method': method,
                    'max_size': max_size,
                    'overlap': overlap,
                    'adaptive': adaptive
                }

                collection_name = f"{database_type.lower().replace(' ', '_')}_embeddings"
                collection = chroma_client.get_or_create_collection(name=collection_name)

                # Determine the model to use
                if provider == "huggingface":
                    model = custom_model if hf_model == "custom" else hf_model
                elif provider == "openai":
                    model = openai_model
                else:
                    model = api_url

                total_items = len(all_content)
                for idx, item in enumerate(all_content):
                    progress((idx + 1) / total_items, desc=f"Processing item {idx + 1} of {total_items}")

                    content_id = item['id']
                    text = item['content']

                    chunks = improved_chunking_process(text, chunk_options)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_text = chunk['text']
                        chunk_id = f"{database_type.lower()}_{content_id}_chunk_{chunk_idx}"

                        try:
                            embedding = create_embedding(chunk_text, provider, model, api_url)
                            metadata = {
                                'content_id': str(content_id),
                                'chunk_index': int(chunk_idx),
                                'total_chunks': int(len(chunks)),
                                'chunking_method': method,
                                'max_chunk_size': int(max_size),
                                'chunk_overlap': int(overlap),
                                'adaptive_chunking': bool(adaptive),
                                'embedding_model': model,
                                'embedding_provider': provider,
                                'content_type': item.get('type', 'media'),
                                'conversation_id': item.get('conversation_id'),
                                **{k: (int(v) if isinstance(v, str) and v.isdigit() else v)
                                   for k, v in chunk['metadata'].items()}
                            }
                            store_in_chroma(collection_name, [chunk_text], [embedding], [chunk_id], [metadata])

                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_id}: {str(e)}")
                            continue

                return f"Embeddings created and stored successfully for all {database_type} content."
            except Exception as e:
                logging.error(f"Error during embedding creation: {str(e)}")
                return f"Error: {str(e)}"

        # Event handlers
        embedding_provider.change(
            fn=update_provider_options,
            inputs=[embedding_provider],
            outputs=[huggingface_model, openai_model, custom_embedding_model, embedding_api_url]
        )

        huggingface_model.change(
            fn=update_huggingface_options,
            inputs=[huggingface_model],
            outputs=[custom_embedding_model]
        )

        database_selection.change(
            fn=update_database_path,
            inputs=[database_selection],
            outputs=[current_db_path]
        )

        create_button.click(
            fn=create_all_embeddings,
            inputs=[
                embedding_provider, huggingface_model, openai_model, custom_embedding_model,
                embedding_api_url, chunking_method, max_chunk_size, chunk_overlap,
                adaptive_chunking, database_selection
            ],
            outputs=status_output
        )


def create_view_embeddings_tab():
    # Load configuration first
    config = load_and_log_configs()
    if not config:
        raise ValueError("Could not load configuration")

    # Get database paths from config
    db_config = config['db_config']
    media_db_path = db_config['sqlite_path']
    rag_qa_db_path = os.path.join(os.path.dirname(media_db_path), "rag_chat.db")
    character_chat_db_path = os.path.join(os.path.dirname(media_db_path), "character_chat.db")
    chroma_db_path = db_config['chroma_db_path']

    with gr.TabItem("View/Update Embeddings", visible=True):
        gr.Markdown("# View and Update Embeddings")
        # Initialize item_mapping as a Gradio State


        with gr.Row():
            with gr.Column():
                # Add database selection
                database_selection = gr.Radio(
                    choices=["Media DB", "RAG Chat", "Character Chat"],
                    label="Select Content Source",
                    value="Media DB",
                    info="Choose which database to view embeddings from"
                )

                # Add database path display
                current_db_path = gr.Textbox(
                    label="Current Database Path",
                    value=media_db_path,
                    interactive=False
                )

                item_dropdown = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                refresh_button = gr.Button("Refresh Item List")
                embedding_status = gr.Textbox(label="Embedding Status", interactive=False)
                embedding_preview = gr.Textbox(label="Embedding Preview", interactive=False, lines=5)
                embedding_metadata = gr.Textbox(label="Embedding Metadata", interactive=False, lines=10)

            with gr.Column():
                create_new_embedding_button = gr.Button("Create New Embedding")
                embedding_provider = gr.Radio(
                    choices=["huggingface", "local", "openai"],
                    label="Select Embedding Provider",
                    value="huggingface"
                )
                gr.Markdown("Note: Local provider requires a running Llama.cpp/llamafile server.")
                gr.Markdown("OpenAI provider requires a valid API key.")

                huggingface_model = gr.Dropdown(
                    choices=[
                        "jinaai/jina-embeddings-v3",
                        "Alibaba-NLP/gte-large-en-v1.5",
                        "dunzhang/stella_en_400M_v5",
                        "custom"
                    ],
                    label="Hugging Face Model",
                    value="jinaai/jina-embeddings-v3",
                    visible=True
                )

                openai_model = gr.Dropdown(
                    choices=[
                        "text-embedding-3-small",
                        "text-embedding-3-large"
                    ],
                    label="OpenAI Embedding Model",
                    value="text-embedding-3-small",
                    visible=False
                )

                custom_embedding_model = gr.Textbox(
                    label="Custom Embedding Model",
                    placeholder="Enter your custom embedding model name here",
                    visible=False
                )

                embedding_api_url = gr.Textbox(
                    label="API URL (for local provider)",
                    value=config['embedding_config']['embedding_api_url'],
                    visible=False
                )

                chunking_method = gr.Dropdown(
                    choices=["words", "sentences", "paragraphs", "tokens", "semantic"],
                    label="Chunking Method",
                    value="words"
                )
                max_chunk_size = gr.Slider(
                    minimum=1, maximum=8000, step=5, value=500,
                    label="Max Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0, maximum=5000, step=5, value=200,
                    label="Chunk Overlap"
                )
                adaptive_chunking = gr.Checkbox(
                    label="Use Adaptive Chunking",
                    value=False
                )
                contextual_api_choice = gr.Dropdown(
                    choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace"],
                    label="Select API for Contextualized Embeddings",
                    value="OpenAI"
                )
                use_contextual_embeddings = gr.Checkbox(
                    label="Use Contextual Embeddings",
                    value=True
                )
                contextual_api_key = gr.Textbox(label="API Key", lines=1)

        item_mapping = gr.State(value={})

        def update_database_path(database_type):
            if database_type == "Media DB":
                return media_db_path
            elif database_type == "RAG Chat":
                return rag_qa_db_path
            else:  # Character Chat
                return character_chat_db_path

        def get_items_with_embedding_status(database_type):
            try:
                # Get items based on database selection
                if database_type == "Media DB":
                    items = get_all_content_from_database()
                elif database_type == "RAG Chat":
                    conversations, _, _ = get_all_conversations(page=1)
                    items = [{
                        'id': conv['conversation_id'],
                        'title': conv['title'],
                        'type': 'conversation'
                    } for conv in conversations]
                else:  # Character Chat
                    notes, _, _ = get_all_notes(page=1)
                    items = [{
                        'id': note['id'],
                        'title': note['title'],
                        'type': 'note'
                    } for note in notes]

                collection_name = f"{database_type.lower().replace(' ', '_')}_embeddings"
                collection = chroma_client.get_or_create_collection(name=collection_name)

                choices = []
                new_item_mapping = {}
                for item in items:
                    try:
                        chunk_id = f"{database_type.lower()}_{item['id']}_chunk_0"
                        result = collection.get(ids=[chunk_id])
                        embedding_exists = result is not None and result.get('ids') and len(result['ids']) > 0
                        status = "Embedding exists" if embedding_exists else "No embedding"
                    except Exception as e:
                        print(f"Error checking embedding for item {item['id']}: {str(e)}")
                        status = "Error checking"
                    choice = f"{item['title']} ({status})"
                    choices.append(choice)
                    new_item_mapping[choice] = item['id']
                return gr.update(choices=choices), new_item_mapping
            except Exception as e:
                print(f"Error in get_items_with_embedding_status: {str(e)}")
                return gr.update(choices=["Error: Unable to fetch items"]), {}

        def update_provider_options(provider):
            if provider == "huggingface":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif provider == "local":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            else:  # OpenAI
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        def update_huggingface_options(model):
            if model == "custom":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        def check_embedding_status(selected_item, database_type, item_mapping):
            if not selected_item:
                return "Please select an item", "", ""

            if item_mapping is None:
                # If mapping is None, try to refresh it
                try:
                    _, item_mapping = get_items_with_embedding_status(database_type)
                except Exception as e:
                    return f"Error initializing item mapping: {str(e)}", "", ""

            try:
                item_id = item_mapping.get(selected_item)
                if item_id is None:
                    return f"Invalid item selected: {selected_item}", "", ""

                item_title = selected_item.rsplit(' (', 1)[0]
                collection_name = f"{database_type.lower().replace(' ', '_')}_embeddings"
                collection = chroma_client.get_or_create_collection(name=collection_name)
                chunk_id = f"{database_type.lower()}_{item_id}_chunk_0"

                try:
                    result = collection.get(ids=[chunk_id], include=["embeddings", "metadatas"])
                except Exception as e:
                    logging.error(f"ChromaDB get error: {str(e)}")
                    return f"Error retrieving embedding for '{item_title}': {str(e)}", "", ""

                # Check if result exists and has the expected structure
                if not result or not isinstance(result, dict):
                    return f"No embedding found for item '{item_title}' (ID: {item_id})", "", ""

                # Check if we have any results
                if not result.get('ids') or len(result['ids']) == 0:
                    return f"No embedding found for item '{item_title}' (ID: {item_id})", "", ""

                # Check if embeddings exist
                if not result.get('embeddings') or not result['embeddings'][0]:
                    return f"Embedding data missing for item '{item_title}' (ID: {item_id})", "", ""

                embedding = result['embeddings'][0]
                metadata = result.get('metadatas', [{}])[0] if result.get('metadatas') else {}
                embedding_preview = str(embedding[:50])
                status = f"Embedding exists for item '{item_title}' (ID: {item_id})"
                return status, f"First 50 elements of embedding:\n{embedding_preview}", json.dumps(metadata, indent=2)

            except Exception as e:
                logging.error(f"Error in check_embedding_status: {str(e)}", exc_info=True)
                return f"Error processing item: {selected_item}. Details: {str(e)}", "", ""

        def refresh_and_update(database_type):
            choices_update, new_mapping = get_items_with_embedding_status(database_type)
            return choices_update, new_mapping

        def create_new_embedding_for_item(selected_item, database_type, provider, hf_model, openai_model,
                                        custom_model, api_url, method, max_size, overlap, adaptive,
                                        item_mapping, use_contextual, contextual_api_choice=None):
            if not selected_item:
                return "Please select an item", "", ""

            try:
                item_id = item_mapping.get(selected_item)
                if item_id is None:
                    return f"Invalid item selected: {selected_item}", "", ""

                # Get item content based on database type
                if database_type == "Media DB":
                    items = get_all_content_from_database()
                    item = next((item for item in items if item['id'] == item_id), None)
                elif database_type == "RAG Chat":
                    item = {
                        'id': item_id,
                        'content': get_conversation_text(item_id),
                        'title': selected_item.rsplit(' (', 1)[0],
                        'type': 'conversation'
                    }
                else:  # Character Chat
                    note = get_note_by_id(item_id)
                    item = {
                        'id': item_id,
                        'content': f"{note['title']}\n\n{note['content']}",
                        'title': note['title'],
                        'type': 'note'
                    }

                if not item:
                    return f"Item not found: {item_id}", "", ""

                chunk_options = {
                    'method': method,
                    'max_size': max_size,
                    'overlap': overlap,
                    'adaptive': adaptive
                }

                logging.info(f"Chunking content for item: {item['title']} (ID: {item_id})")
                chunks = chunk_for_embedding(item['content'], item['title'], chunk_options)
                collection_name = f"{database_type.lower().replace(' ', '_')}_embeddings"
                collection = chroma_client.get_or_create_collection(name=collection_name)

                # Delete existing embeddings for this item
                existing_ids = [f"{database_type.lower()}_{item_id}_chunk_{i}" for i in range(len(chunks))]
                collection.delete(ids=existing_ids)
                logging.info(f"Deleted {len(existing_ids)} existing embeddings for item {item_id}")

                texts, ids, metadatas = [], [], []
                chunk_count = 0
                logging.info("Generating contextual summaries and preparing chunks for embedding")
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk['text']
                    chunk_metadata = chunk['metadata']
                    if use_contextual:
                        logging.debug(f"Generating contextual summary for chunk {chunk_count}")
                        context = situate_context(contextual_api_choice, item['content'], chunk_text)
                        contextualized_text = f"{chunk_text}\n\nContextual Summary: {context}"
                    else:
                        contextualized_text = chunk_text
                        context = None

                    chunk_id = f"{database_type.lower()}_{item_id}_chunk_{i}"

                    # Determine the model to use
                    if provider == "huggingface":
                        model = custom_model if hf_model == "custom" else hf_model
                    elif provider == "openai":
                        model = openai_model
                    else:
                        model = custom_model

                    metadata = {
                        "content_id": str(item_id),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_method": method,
                        "max_chunk_size": max_size,
                        "chunk_overlap": overlap,
                        "adaptive_chunking": adaptive,
                        "embedding_model": model,
                        "embedding_provider": provider,
                        "original_text": chunk_text,
                        "use_contextual_embeddings": use_contextual,
                        "contextual_summary": context,
                        **chunk_metadata
                    }

                    texts.append(contextualized_text)
                    ids.append(chunk_id)
                    metadatas.append(metadata)
                    chunk_count += 1

                # Create embeddings in batch
                logging.info(f"Creating embeddings for {len(texts)} chunks")
                embeddings = create_embeddings_batch(texts, provider, model, api_url)

                # Store in Chroma
                store_in_chroma(collection_name, texts, embeddings, ids, metadatas)

                # Create a preview of the first embedding
                if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                    embedding_preview = str(embeddings[0][:50])
                elif isinstance(embeddings, list) and len(embeddings) > 0:
                    embedding_preview = str(embeddings[0][:50])
                else:
                    embedding_preview = "No embeddings created"

                # Return status message
                status = f"New embeddings created and stored for item: {item['title']} (ID: {item_id})"

                # Add contextual summaries to status message if enabled
                if use_contextual:
                    status += " (with contextual summaries)"

                # Return status message, embedding preview, and metadata
                return status, f"First 50 elements of new embedding:\n{embedding_preview}", json.dumps(metadatas[0],
                                                                                                       indent=2)
            except Exception as e:
                logging.error(f"Error in create_new_embedding_for_item: {str(e)}", exc_info=True)
                return f"Error creating embedding: {str(e)}", "", ""

        # Wire up all the event handlers
        database_selection.change(
            update_database_path,
            inputs=[database_selection],
            outputs=[current_db_path]
        )

        refresh_button.click(
            get_items_with_embedding_status,
            inputs=[database_selection],
            outputs=[item_dropdown, item_mapping]
        )

        item_dropdown.change(
            check_embedding_status,
            inputs=[item_dropdown, database_selection, item_mapping],
            outputs=[embedding_status, embedding_preview, embedding_metadata]
        )

        create_new_embedding_button.click(
            create_new_embedding_for_item,
            inputs=[item_dropdown, embedding_provider, huggingface_model, openai_model, custom_embedding_model, embedding_api_url,
                    chunking_method, max_chunk_size, chunk_overlap, adaptive_chunking, item_mapping,
                    use_contextual_embeddings, contextual_api_choice],
            outputs=[embedding_status, embedding_preview, embedding_metadata]
        )
        embedding_provider.change(
            update_provider_options,
            inputs=[embedding_provider],
            outputs=[huggingface_model, openai_model, custom_embedding_model, embedding_api_url]
        )
        huggingface_model.change(
            update_huggingface_options,
            inputs=[huggingface_model],
            outputs=[custom_embedding_model]
        )

    return (item_dropdown, refresh_button, embedding_status, embedding_preview, embedding_metadata,
            create_new_embedding_button, embedding_provider, huggingface_model, openai_model,
            custom_embedding_model, embedding_api_url, chunking_method, max_chunk_size,
            chunk_overlap, adaptive_chunking, use_contextual_embeddings,
            contextual_api_choice, contextual_api_key)


def create_purge_embeddings_tab():
    with gr.TabItem("Purge Embeddings", visible=True):
        gr.Markdown("# Purge Embeddings")

        with gr.Row():
            with gr.Column():
                purge_button = gr.Button("Purge All Embeddings")
            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=10)

    def purge_all_embeddings():
        try:
            # It came to me in a dream....I literally don't remember how the fuck this works, cant find documentation...
            collection_name = "all_content_embeddings"
            chroma_client.delete_collection(collection_name)
            chroma_client.create_collection(collection_name)
            logging.info(f"All embeddings have been purged successfully.")
            return "All embeddings have been purged successfully."
        except Exception as e:
            logging.error(f"Error during embedding purge: {str(e)}")
            return f"Error: {str(e)}"

    purge_button.click(
        fn=purge_all_embeddings,
        outputs=status_output
    )



#
# End of file
########################################################################################################################
