# Gradio_Related.py
#########################################
# Gradio UI Functions Library
# I fucking hate Gradio.
#
#########################################
#
# Built-In Imports
import os
import webbrowser
#
# Import 3rd-Party Libraries
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import get_db_config, backup_dir
from App_Function_Libraries.DB.RAG_QA_Chat_DB import create_tables
from App_Function_Libraries.Gradio_UI.Anki_tab import create_anki_validation_tab, create_anki_generator_tab
from App_Function_Libraries.Gradio_UI.Arxiv_tab import create_arxiv_tab
from App_Function_Libraries.Gradio_UI.Audio_ingestion_tab import create_audio_processing_tab
from App_Function_Libraries.Gradio_UI.Backup_RAG_Notes_Character_Chat_tab import create_database_management_interface
from App_Function_Libraries.Gradio_UI.Book_Ingestion_tab import create_import_book_tab
from App_Function_Libraries.Gradio_UI.Character_Chat_tab import create_character_card_interaction_tab, create_character_chat_mgmt_tab, create_custom_character_card_tab, \
    create_character_card_validation_tab, create_export_characters_tab
from App_Function_Libraries.Gradio_UI.Character_interaction_tab import create_narrator_controlled_conversation_tab, \
    create_multiple_character_chat_tab
from App_Function_Libraries.Gradio_UI.Chat_ui import create_chat_interface_four, create_chat_interface_multi_api, \
    create_chat_interface_stacked, create_chat_interface
from App_Function_Libraries.Gradio_UI.Config_tab import create_config_editor_tab
from App_Function_Libraries.Gradio_UI.Explain_summarize_tab import create_summarize_explain_tab
from App_Function_Libraries.Gradio_UI.Export_Functionality import create_export_tabs
from App_Function_Libraries.Gradio_UI.Import_Obsidian import create_import_obsidian_vault_tab
from App_Function_Libraries.Gradio_UI.Import_Prompts_tab import create_import_single_prompt_tab, \
    create_import_multiple_prompts_tab
from App_Function_Libraries.Gradio_UI.Import_RAG_Chat import create_conversation_import_tab
from App_Function_Libraries.Gradio_UI.Import_Text_MD import create_import_item_tab
#from App_Function_Libraries.Gradio_UI.Backup_Functionality import create_backup_tab, create_view_backups_tab, \
#    create_restore_backup_tab
from App_Function_Libraries.Gradio_UI.Introduction_tab import create_introduction_tab
from App_Function_Libraries.Gradio_UI.Keywords import create_view_keywords_tab, create_add_keyword_tab, \
    create_delete_keyword_tab, create_export_keywords_tab, create_rag_qa_keywords_tab, create_character_keywords_tab, \
    create_meta_keywords_tab, create_prompt_keywords_tab
from App_Function_Libraries.Gradio_UI.Live_Recording import create_live_recording_tab
from App_Function_Libraries.Gradio_UI.Llamafile_tab import create_chat_with_llamafile_tab
#from App_Function_Libraries.Gradio_UI.MMLU_Pro_tab import create_mmlu_pro_tab
from App_Function_Libraries.Gradio_UI.Media_edit import create_prompt_clone_tab, create_prompt_edit_tab, \
    create_media_edit_and_clone_tab, create_media_edit_tab
from App_Function_Libraries.Gradio_UI.Media_wiki_tab import create_mediawiki_import_tab, create_mediawiki_config_tab
from App_Function_Libraries.Gradio_UI.Mind_Map_tab import create_mindmap_tab
from App_Function_Libraries.Gradio_UI.PDF_ingestion_tab import create_pdf_ingestion_tab, create_pdf_ingestion_test_tab
from App_Function_Libraries.Gradio_UI.Plaintext_tab_import import create_plain_text_import_tab
from App_Function_Libraries.Gradio_UI.Podcast_tab import create_podcast_tab
from App_Function_Libraries.Gradio_UI.Prompt_Suggestion_tab import create_prompt_suggestion_tab
from App_Function_Libraries.Gradio_UI.RAG_QA_Chat_tab import create_rag_qa_chat_tab, create_rag_qa_notes_management_tab, \
    create_rag_qa_chat_management_tab
from App_Function_Libraries.Gradio_UI.Re_summarize_tab import create_resummary_tab
from App_Function_Libraries.Gradio_UI.Search_Tab import create_prompt_search_tab, \
    create_search_summaries_tab, create_search_tab
from App_Function_Libraries.Gradio_UI.RAG_Chat_tab import create_rag_tab
from App_Function_Libraries.Gradio_UI.Embeddings_tab import create_embeddings_tab, create_view_embeddings_tab, \
    create_purge_embeddings_tab
from App_Function_Libraries.Gradio_UI.Semantic_Scholar_tab import create_semantic_scholar_tab
from App_Function_Libraries.Gradio_UI.TTS_Playground import create_audio_generation_tab
from App_Function_Libraries.Gradio_UI.Trash import create_view_trash_tab, create_empty_trash_tab, \
    create_delete_trash_tab, create_search_and_mark_trash_tab
from App_Function_Libraries.Gradio_UI.Utilities import create_utilities_yt_timestamp_tab, create_utilities_yt_audio_tab, \
    create_utilities_yt_video_tab
from App_Function_Libraries.Gradio_UI.Video_transcription_tab import create_video_transcription_tab
from App_Function_Libraries.Gradio_UI.View_tab import create_manage_items_tab
from App_Function_Libraries.Gradio_UI.WebSearch_tab import create_websearch_tab
from App_Function_Libraries.Gradio_UI.Website_scraping_tab import create_website_scraping_tab
from App_Function_Libraries.Gradio_UI.Workflows_tab import chat_workflows_tab
from App_Function_Libraries.Gradio_UI.View_DB_Items_tab import create_view_all_mediadb_with_versions_tab, \
    create_viewing_mediadb_tab, create_view_all_rag_notes_tab, create_viewing_ragdb_tab, \
    create_mediadb_keyword_search_tab, create_ragdb_keyword_items_tab
from App_Function_Libraries.Gradio_UI.Prompts_tab import create_prompt_view_tab, create_prompts_export_tab
#
# Gradio UI Imports
from App_Function_Libraries.Gradio_UI.Evaluations_Benchmarks_tab import create_geval_tab, create_infinite_bench_tab
from App_Function_Libraries.Gradio_UI.XML_Ingestion_Tab import create_xml_import_tab
#from App_Function_Libraries.Local_LLM.Local_LLM_huggingface import create_huggingface_tab
from App_Function_Libraries.Local_LLM.Local_LLM_ollama import create_ollama_tab
from App_Function_Libraries.Utils.Utils import load_and_log_configs, logging

#
#######################################################################################################################
# Function Definitions
#

# Disable Gradio Analytics
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
custom_prompt_input = None
server_mode = False
share_public = False

#
# End of globals
#######################################################################################################################
#
# Start of Video/Audio Transcription and Summarization Functions
#
# Functions:
# FIXME
#
#
################################################################################################################
# Functions for Re-Summarization
#
# Functions:
# FIXME
# End of Re-Summarization Functions
#
############################################################################################################################################################################################################################
#
# Explain/Summarize This Tab
#
# Functions:
# FIXME
#
#
############################################################################################################################################################################################################################
#
# Transcript Comparison Tab
#
# Functions:
# FIXME
#
#
###########################################################################################################################################################################################################################
#
# Search Tab
#
# Functions:
# FIXME
#
# End of Search Tab Functions
#
##############################################################################################################################################################################################################################
#
# Llamafile Tab
#
# Functions:
# FIXME
#
# End of Llamafile Tab Functions
##############################################################################################################################################################################################################################
#
# Chat Interface Tab Functions
#
# Functions:
# FIXME
#
#
# End of Chat Interface Tab Functions
################################################################################################################################################################################################################################
#
# Media Edit Tab Functions
# Functions:
# Fixme
# create_media_edit_tab():
##### Trash Tab
# FIXME
# Functions:
#
# End of Media Edit Tab Functions
################################################################################################################
#
# Import Items Tab Functions
#
# Functions:
#FIXME
# End of Import Items Tab Functions
################################################################################################################
#
# Export Items Tab Functions
#
# Functions:
# FIXME
#
#
# End of Export Items Tab Functions
################################################################################################################
#
# Keyword Management Tab Functions
#
# Functions:
#  create_view_keywords_tab():
# FIXME
#
# End of Keyword Management Tab Functions
################################################################################################################
#
# Document Editing Tab Functions
#
# Functions:
#   #FIXME
#
#
################################################################################################################
#
# Utilities Tab Functions
# Functions:
#   create_utilities_yt_video_tab():
# #FIXME

#
# End of Utilities Tab Functions
################################################################################################################

# FIXME - Prompt sample box
#
# # Sample data
# prompts_category_1 = [
#     "What are the key points discussed in the video?",
#     "Summarize the main arguments made by the speaker.",
#     "Describe the conclusions of the study presented."
# ]
#
# prompts_category_2 = [
#     "How does the proposed solution address the problem?",
#     "What are the implications of the findings?",
#     "Can you explain the theory behind the observed phenomenon?"
# ]
#
# all_prompts2 = prompts_category_1 + prompts_category_2



#######################################################################################################################
#
# Migration Script
import os
from datetime import datetime
import shutil

# def migrate_media_db_to_rag_chat_db(media_db_path, rag_chat_db_path):
#     # Check if migration is needed
#     if not os.path.exists(media_db_path):
#         logging.info("Media DB does not exist. No migration needed.")
#         return
#
#     # Optional: Check if migration has already been completed
#     migration_flag = os.path.join(os.path.dirname(rag_chat_db_path), 'migration_completed.flag')
#     if os.path.exists(migration_flag):
#         logging.info("Migration already completed. Skipping migration.")
#         return
#
#     # Backup databases
#     backup_database(media_db_path)
#     backup_database(rag_chat_db_path)
#
#     # Connect to both databases
#     try:
#         media_conn = sqlite3.connect(media_db_path)
#         rag_conn = sqlite3.connect(rag_chat_db_path)
#
#         # Enable foreign key support
#         media_conn.execute('PRAGMA foreign_keys = ON;')
#         rag_conn.execute('PRAGMA foreign_keys = ON;')
#
#         media_cursor = media_conn.cursor()
#         rag_cursor = rag_conn.cursor()
#
#         # Begin transaction
#         rag_conn.execute('BEGIN TRANSACTION;')
#
#         # Extract conversations from media DB
#         media_cursor.execute('''
#             SELECT id, media_id, media_name, conversation_name, created_at, updated_at
#             FROM ChatConversations
#         ''')
#         conversations = media_cursor.fetchall()
#
#         for conv in conversations:
#             old_conv_id, media_id, media_name, conversation_name, created_at, updated_at = conv
#
#             # Convert timestamps if necessary
#             created_at = parse_timestamp(created_at)
#             updated_at = parse_timestamp(updated_at)
#
#             # Generate a new conversation_id
#             conversation_id = str(uuid.uuid4())
#             title = conversation_name or (f"{media_name}-chat" if media_name else "Untitled Conversation")
#
#             # Insert into conversation_metadata
#             rag_cursor.execute('''
#                 INSERT INTO conversation_metadata (conversation_id, created_at, last_updated, title, media_id)
#                 VALUES (?, ?, ?, ?, ?)
#             ''', (conversation_id, created_at, updated_at, title, media_id))
#
#             # Extract messages from media DB
#             media_cursor.execute('''
#                 SELECT sender, message, timestamp
#                 FROM ChatMessages
#                 WHERE conversation_id = ?
#                 ORDER BY timestamp ASC
#             ''', (old_conv_id,))
#             messages = media_cursor.fetchall()
#
#             for msg in messages:
#                 sender, content, timestamp = msg
#
#                 # Convert timestamp if necessary
#                 timestamp = parse_timestamp(timestamp)
#
#                 role = sender  # Assuming 'sender' is 'user' or 'ai'
#
#                 # Insert message into rag_qa_chats
#                 rag_cursor.execute('''
#                     INSERT INTO rag_qa_chats (conversation_id, timestamp, role, content)
#                     VALUES (?, ?, ?, ?)
#                 ''', (conversation_id, timestamp, role, content))
#
#         # Commit transaction
#         rag_conn.commit()
#         logging.info("Migration completed successfully.")
#
#         # Mark migration as complete
#         with open(migration_flag, 'w') as f:
#             f.write('Migration completed on ' + datetime.now().isoformat())
#
#     except Exception as e:
#         # Rollback transaction in case of error
#         rag_conn.rollback()
#         logging.error(f"Error during migration: {e}")
#         raise
#     finally:
#         media_conn.close()
#         rag_conn.close()

def backup_database(db_path):
    backup_path = db_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copyfile(db_path, backup_path)
        logging.info(f"Database backed up to {backup_path}")
    else:
        logging.info(f"Backup already exists at {backup_path}")

def parse_timestamp(timestamp_value):
    """
    Parses the timestamp from the old database and converts it to a standard format.
    Adjust this function based on the actual format of your timestamps.
    """
    try:
        # Attempt to parse ISO format
        return datetime.fromisoformat(timestamp_value).isoformat()
    except ValueError:
        # Handle other timestamp formats if necessary
        # For example, if timestamps are in Unix epoch format
        try:
            timestamp_float = float(timestamp_value)
            return datetime.fromtimestamp(timestamp_float).isoformat()
        except ValueError:
            # Default to current time if parsing fails
            logging.warning(f"Unable to parse timestamp '{timestamp_value}', using current time.")
            return datetime.now().isoformat()

#
# End of Migration Script
#######################################################################################################################


#######################################################################################################################
#
# Launch UI Function
def launch_ui(share_public=None, server_mode=False):
    webbrowser.open_new_tab('http://127.0.0.1:7860/?__theme=dark')
    share=share_public
    css = """
    .result-box {
        margin-bottom: 20px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .result-box.error {
        border-color: #ff0000;
        background-color: #ffeeee;
    }
    .transcription, .summary {
        max-height: 800px;
        overflow-y: auto;
        border: 1px solid #eee;
        padding: 10px;
        margin-top: 10px;
    }
    /* ID-based selector for the textbox */
    #scrollable-textbox textarea {
        max-height: 500px !important; 
        overflow-y: auto !important;
    }
    """

    config = load_and_log_configs()
    # Get database paths from config
    db_config = config['db_config']
    media_db_path = db_config['sqlite_path']
    character_chat_db_path = os.path.join(os.path.dirname(media_db_path), "chatDB.db")
    rag_chat_db_path = os.path.join(os.path.dirname(media_db_path), "rag_qa.db")
    # Initialize the RAG Chat DB (create tables and update schema)
    create_tables()

    # Migrate data from the media DB to the RAG Chat DB
    #migrate_media_db_to_rag_chat_db(media_db_path, rag_chat_db_path)


    with gr.Blocks(theme='bethecloud/storj_theme',css=css) as iface:
        gr.HTML(
            """
            <script>
            document.addEventListener('DOMContentLoaded', (event) => {
                document.body.classList.add('dark');
                document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)';
            });
            </script>
            """
        )
        db_config = get_db_config()
        db_type = db_config['type']
        gr.Markdown(f"# tl/dw: Your LLM-powered Research Multi-tool")
        gr.Markdown(f"(Using {db_type.capitalize()} Database)")
        with gr.Tabs():
            with gr.TabItem("Transcribe / Analyze / Ingestion", id="ingestion-grouping", visible=True):
                with gr.Tabs():
                    create_video_transcription_tab()
                    create_audio_processing_tab()
                    create_podcast_tab()
                    create_import_book_tab()
                    create_plain_text_import_tab()
                    create_xml_import_tab()
                    create_website_scraping_tab()
                    create_pdf_ingestion_tab()
                    create_pdf_ingestion_test_tab()
                    create_resummary_tab()
                    create_summarize_explain_tab()
                    create_live_recording_tab()
                    create_audio_generation_tab()
                    create_arxiv_tab()
                    create_semantic_scholar_tab()

            with gr.TabItem("RAG Chat/Search", id="RAG Chat Notes group", visible=True):
                create_rag_tab()
                create_rag_qa_chat_tab()
                create_rag_qa_notes_management_tab()
                create_rag_qa_chat_management_tab()

            with gr.TabItem("Chat with an LLM", id="LLM Chat group", visible=True):
                create_chat_interface()
                create_chat_interface_stacked()
                create_chat_interface_multi_api()
                create_chat_interface_four()
                chat_workflows_tab()

            with gr.TabItem("Web Search & Review", id="websearch group", visible=True):
                create_websearch_tab()
            with gr.TabItem("Character Chat", id="character chat group", visible=True):
                create_character_card_interaction_tab()
                create_character_chat_mgmt_tab()
                create_custom_character_card_tab()
                create_character_card_validation_tab()
                create_multiple_character_chat_tab()
                create_narrator_controlled_conversation_tab()
                create_export_characters_tab()

            with gr.TabItem("Writing Tools", id="writing_tools group", visible=True):
                from App_Function_Libraries.Gradio_UI.Writing_tab import create_document_feedback_tab
                create_document_feedback_tab()
                from App_Function_Libraries.Gradio_UI.Writing_tab import create_grammar_style_check_tab
                create_grammar_style_check_tab()
                from App_Function_Libraries.Gradio_UI.Writing_tab import create_tone_adjustment_tab
                create_tone_adjustment_tab()
                from App_Function_Libraries.Gradio_UI.Writing_tab import create_creative_writing_tab
                create_creative_writing_tab()
                from App_Function_Libraries.Gradio_UI.Writing_tab import create_mikupad_tab
                create_mikupad_tab()

            with gr.TabItem("Search/View DB Items", id="view db items group", visible=True):
                create_search_tab()
                create_search_summaries_tab()
                create_view_all_mediadb_with_versions_tab()
                create_viewing_mediadb_tab()
                create_mediadb_keyword_search_tab()
                create_view_all_rag_notes_tab()
                create_viewing_ragdb_tab()
                create_ragdb_keyword_items_tab()

            with gr.TabItem("Prompts", id='view prompts group', visible=True):
                with gr.Tabs():
                    create_prompt_view_tab()
                    create_prompt_search_tab()
                    create_prompt_edit_tab()
                    create_prompt_clone_tab()
                    create_prompt_suggestion_tab()
                    create_prompts_export_tab()

            with gr.TabItem("Manage Media DB Items", id="manage group", visible=True):
                create_media_edit_tab()
                create_manage_items_tab()
                create_media_edit_and_clone_tab()

            with gr.TabItem("Embeddings Management", id="embeddings group", visible=True):
                create_embeddings_tab()
                create_view_embeddings_tab()
                create_purge_embeddings_tab()

            with gr.TabItem("Keywords", id="keywords group", visible=True):
                create_view_keywords_tab()
                create_add_keyword_tab()
                create_delete_keyword_tab()
                create_export_keywords_tab()
                create_character_keywords_tab()
                create_rag_qa_keywords_tab()
                create_meta_keywords_tab()
                create_prompt_keywords_tab()

            with gr.TabItem("Import", id="import group", visible=True):
                create_import_item_tab()
                create_import_obsidian_vault_tab()
                create_import_single_prompt_tab()
                create_import_multiple_prompts_tab()
                create_mediawiki_import_tab()
                create_mediawiki_config_tab()
                create_conversation_import_tab()

            with gr.TabItem("Export", id="export group", visible=True):
                create_export_tabs()


            with gr.TabItem("Database Management", id="database_management_group", visible=True):
                create_database_management_interface(
                    media_db_config={
                        'db_path': media_db_path,
                        'backup_dir': backup_dir
                    },
                    rag_db_config={
                        'db_path': rag_chat_db_path,
                        'backup_dir': backup_dir
                    },
                    char_db_config={
                        'db_path': character_chat_db_path,
                        'backup_dir': backup_dir
                    }
                )

            with gr.TabItem("Utilities", id="util group", visible=True):
                create_mindmap_tab()
                create_utilities_yt_video_tab()
                create_utilities_yt_audio_tab()
                create_utilities_yt_timestamp_tab()

            with gr.TabItem("Anki Deck Creation/Validation", id="anki group", visible=True):
                create_anki_generator_tab()
                create_anki_validation_tab()

            with gr.TabItem("Local LLM", id="local llm group", visible=True):
                create_chat_with_llamafile_tab()
                create_ollama_tab()
                #create_huggingface_tab()

            with gr.TabItem("Trashcan", id="trashcan group", visible=True):
                create_search_and_mark_trash_tab()
                create_view_trash_tab()
                create_delete_trash_tab()
                create_empty_trash_tab()

            with gr.TabItem("Evaluations", id="eval", visible=True):
                create_geval_tab()
                create_infinite_bench_tab()
                # FIXME
                #create_mmlu_pro_tab()

            with gr.TabItem("Introduction/Help", id="introduction group", visible=True):
                create_introduction_tab()

            with gr.TabItem("Config Editor", id="config group"):
                create_config_editor_tab()

    # Launch the interface
    server_port_variable = 7860
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    if share==True:
        iface.launch(share=True)
    elif server_mode and not share_public:
        iface.launch(share=False, server_name="0.0.0.0", server_port=server_port_variable, )
    else:
        try:
            iface.launch(share=False, server_name="0.0.0.0", server_port=server_port_variable, )
        except Exception as e:
            logging.error(f"Error launching interface: {str(e)}")
