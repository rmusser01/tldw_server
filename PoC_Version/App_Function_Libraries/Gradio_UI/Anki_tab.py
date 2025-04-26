# Anki_Validation_tab.py
# Description: Gradio functions for the Anki Validation tab
#
# Imports
import json
import os
import tempfile
from typing import Optional, Tuple, List, Dict
#
# External Imports
import genanki
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Chat.Chat_Functions import approximate_token_count, update_chat_content, save_chat_history, \
    save_chat_history_to_db_wrapper
from PoC_Version.App_Function_Libraries.DB.DB_Manager import list_prompts
from PoC_Version.App_Function_Libraries.Gradio_UI.Chat_ui import update_dropdown_multiple, chat_wrapper, update_selected_parts, \
    search_conversations, regenerate_last_message, load_conversation, debug_output
from PoC_Version.App_Function_Libraries.Third_Party.Anki import sanitize_html, generate_card_choices, \
    export_cards, load_card_for_editing, handle_file_upload, \
    validate_for_ui, update_card_with_validation, update_card_choices, enhanced_file_upload, \
    handle_validation
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging


#
############################################################################################################
#
# Functions:

def create_anki_validation_tab():
    with gr.TabItem("Anki Flashcard Validation", visible=True):
        gr.Markdown("# Anki Flashcard Validation and Editor")

        # State variables for internal tracking
        current_card_data = gr.State({})
        preview_update_flag = gr.State(False)

        with gr.Row():
            # Left Column: Input and Validation
            with gr.Column(scale=1):
                gr.Markdown("## Import or Create Flashcards")

                input_type = gr.Radio(
                    choices=["JSON", "APKG"],
                    label="Input Type",
                    value="JSON"
                )

                with gr.Group() as json_input_group:
                    flashcard_input = gr.TextArea(
                        label="Enter Flashcards (JSON format)",
                        placeholder='''{
    "cards": [
        {
            "id": "CARD_001",
            "type": "basic",
            "front": "What is the capital of France?",
            "back": "Paris",
            "tags": ["geography", "europe"],
            "note": "Remember: City of Light"
        }
    ]
}''',
                        lines=10
                    )

                    import_json = gr.File(
                        label="Or Import JSON File",
                        file_types=[".json"]
                    )

                with gr.Group(visible=False) as apkg_input_group:
                    import_apkg = gr.File(
                        label="Import APKG File",
                        file_types=[".apkg"]
                    )
                    deck_info = gr.JSON(
                        label="Deck Information",
                        visible=False
                    )

                validate_button = gr.Button("Validate Flashcards")

            # Right Column: Validation Results and Editor
            with gr.Column(scale=1):
                gr.Markdown("## Validation Results")
                validation_status = gr.Markdown("")

                with gr.Accordion("Validation Rules", open=False):
                    gr.Markdown("""
                    ### Required Fields:
                    - Unique ID
                    - Card Type (basic, cloze, reverse)
                    - Front content
                    - Back content
                    - At least one tag

                    ### Content Rules:
                    - No empty fields
                    - Front side should be a clear question/prompt
                    - Back side should contain complete answer
                    - Cloze deletions must have valid syntax
                    - No duplicate IDs

                    ### Image Rules:
                    - Valid image tags
                    - Supported formats (JPG, PNG, GIF)
                    - Base64 encoded or valid URL

                    ### APKG-specific Rules:
                    - Valid SQLite database structure
                    - Media files properly referenced
                    - Note types match Anki standards
                    - Card templates are well-formed
                    """)

        with gr.Row():
            # Card Editor
            gr.Markdown("## Card Editor")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Edit Individual Cards", open=True):
                    card_selector = gr.Dropdown(
                        label="Select Card to Edit",
                        choices=[],
                        interactive=True
                    )

                    card_type = gr.Radio(
                        choices=["basic", "cloze", "reverse"],
                        label="Card Type",
                        value="basic"
                    )

                    # Front content with preview
                    with gr.Group():
                        gr.Markdown("### Front Content")
                        front_content = gr.TextArea(
                            label="Content (HTML supported)",
                            lines=3
                        )
                        front_preview = gr.HTML(
                            label="Preview"
                        )

                    # Back content with preview
                    with gr.Group():
                        gr.Markdown("### Back Content")
                        back_content = gr.TextArea(
                            label="Content (HTML supported)",
                            lines=3
                        )
                        back_preview = gr.HTML(
                            label="Preview"
                        )

                    tags_input = gr.TextArea(
                        label="Tags (comma-separated)",
                        lines=1
                    )

                    notes_input = gr.TextArea(
                        label="Additional Notes",
                        lines=2
                    )

                    with gr.Row():
                        update_card_button = gr.Button("Update Card")
                        delete_card_button = gr.Button("Delete Card", variant="stop")

        with gr.Row():
            with gr.Column(scale=1):
                # Export Options
                gr.Markdown("## Export Options")
                export_format = gr.Radio(
                    choices=["Anki CSV", "JSON", "Plain Text"],
                    label="Export Format",
                    value="Anki CSV"
                )
                export_button = gr.Button("Export Valid Cards")
                export_file = gr.File(label="Download Validated Cards")
                export_status = gr.Markdown("")
            with gr.Column(scale=1):
                gr.Markdown("## Export Instructions")
                gr.Markdown("""
                ### Anki CSV Format:
                - Front, Back, Tags, Type, Note
                - Use for importing into Anki
                - Images preserved as HTML

                ### JSON Format:
                - JSON array of cards
                - Images as base64 or URLs
                - Use for custom processing

                ### Plain Text Format:
                - Question and Answer pairs
                - Images represented as [IMG] placeholder
                - Use for manual review
                """)

        def update_preview(content):
            """Update preview with sanitized content."""
            if not content:
                return ""
            return sanitize_html(content)

        # Event handlers
        def validation_chain(content: str) -> Tuple[str, List[str]]:
            """Combined validation and card choice update."""
            validation_message = validate_for_ui(content)
            card_choices = update_card_choices(content)
            return validation_message, card_choices

        def delete_card(card_selection, current_content):
            """Delete selected card and return updated content."""
            if not card_selection or not current_content:
                return current_content, "No card selected", []

            try:
                data = json.loads(current_content)
                selected_id = card_selection.split(" - ")[0]

                data['cards'] = [card for card in data['cards'] if card['id'] != selected_id]
                new_content = json.dumps(data, indent=2)

                return (
                    new_content,
                    "Card deleted successfully!",
                    generate_card_choices(new_content)
                )

            except Exception as e:
                return current_content, f"Error deleting card: {str(e)}", []

        def process_validation_result(is_valid, message):
            """Process validation result into a formatted markdown string."""
            if is_valid:
                return f"✅ {message}"
            else:
                return f"❌ {message}"

        # Register event handlers
        input_type.change(
            fn=lambda t: (
                gr.update(visible=t == "JSON"),
                gr.update(visible=t == "APKG"),
                gr.update(visible=t == "APKG")
            ),
            inputs=[input_type],
            outputs=[json_input_group, apkg_input_group, deck_info]
        )

        # File upload handlers
        import_json.upload(
            fn=handle_file_upload,
            inputs=[import_json, input_type],
            outputs=[
                flashcard_input,
                deck_info,
                validation_status,
                card_selector
            ]
        )

        import_apkg.upload(
            fn=enhanced_file_upload,
            inputs=[import_apkg, input_type],
            outputs=[
                flashcard_input,
                deck_info,
                validation_status,
                card_selector
            ]
        )

        # Validation handler
        validate_button.click(
            fn=lambda content, input_format: (
                handle_validation(content, input_format),
                generate_card_choices(content) if content else []
            ),
            inputs=[flashcard_input, input_type],
            outputs=[validation_status, card_selector]
        )

        # Card editing handlers
        # Card selector change event
        card_selector.change(
            fn=load_card_for_editing,
            inputs=[card_selector, flashcard_input],
            outputs=[
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input,
                front_preview,
                back_preview
            ]
        )

        # Live preview updates
        front_content.change(
            fn=update_preview,
            inputs=[front_content],
            outputs=[front_preview]
        )

        back_content.change(
            fn=update_preview,
            inputs=[back_content],
            outputs=[back_preview]
        )

        # Card update handler
        update_card_button.click(
            fn=update_card_with_validation,
            inputs=[
                flashcard_input,
                card_selector,
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input
            ],
            outputs=[
                flashcard_input,
                validation_status,
                card_selector
            ]
        )

        # Delete card handler
        delete_card_button.click(
            fn=delete_card,
            inputs=[card_selector, flashcard_input],
            outputs=[flashcard_input, validation_status, card_selector]
        )

        # Export handler
        export_button.click(
            fn=export_cards,
            inputs=[flashcard_input, export_format],
            outputs=[export_status, export_file]
        )

        return (
            flashcard_input,
            import_json,
            import_apkg,
            validate_button,
            validation_status,
            card_selector,
            card_type,
            front_content,
            back_content,
            front_preview,
            back_preview,
            tags_input,
            notes_input,
            update_card_button,
            delete_card_button,
            export_format,
            export_button,
            export_file,
            export_status,
            deck_info
        )


def create_anki_generator_tab():
    with gr.TabItem("Anki Deck Generator", visible=True):
        try:
            default_value = None
            if default_api_endpoint:
                if default_api_endpoint in global_api_endpoints:
                    default_value = format_api_name(default_api_endpoint)
                else:
                    logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
        except Exception as e:
            logging.error(f"Error setting default API endpoint: {str(e)}")
            default_value = None
        custom_css = """
        .chatbot-container .message-wrap .message {
            font-size: 14px !important;
        }
        """
        with gr.TabItem("LLM Chat & Anki Deck Creation", visible=True):
            gr.Markdown("# Chat with an LLM to help you come up with Questions/Answers for an Anki Deck")
            chat_history = gr.State([])
            media_content = gr.State({})
            selected_parts = gr.State([])
            conversation_id = gr.State(None)
            initial_prompts, total_pages, current_page = list_prompts(page=1, per_page=10)

            with gr.Row():
                with gr.Column(scale=1):
                    search_query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query here..."
                    )
                    search_type_input = gr.Radio(
                        choices=["Title", "Content", "Author", "Keyword"],
                        value="Keyword",
                        label="Search By"
                    )
                    keyword_filter_input = gr.Textbox(
                        label="Filter by Keywords (comma-separated)",
                        placeholder="ml, ai, python, etc..."
                    )
                    search_button = gr.Button("Search")
                    items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                    item_mapping = gr.State({})
                    with gr.Row():
                        use_content = gr.Checkbox(label="Use Content")
                        use_summary = gr.Checkbox(label="Use Summary")
                        use_prompt = gr.Checkbox(label="Use Prompt")
                        save_conversation = gr.Checkbox(label="Save Conversation", value=False, visible=True)
                    with gr.Row():
                        temperature = gr.Slider(label="Temperature", minimum=0.00, maximum=1.0, step=0.05, value=0.7)
                    with gr.Row():
                        conversation_search = gr.Textbox(label="Search Conversations")
                    with gr.Row():
                        search_conversations_btn = gr.Button("Search Conversations")
                    with gr.Row():
                        previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                    with gr.Row():
                        load_conversations_btn = gr.Button("Load Selected Conversation")

                    # Refactored API selection dropdown
                    api_endpoint = gr.Dropdown(
                        choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                        value=default_value,
                        label="API for Chat Interaction (Optional)"
                    )
                    api_key = gr.Textbox(label="API Key (if required)", type="password")
                    custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                         value=False,
                                                         visible=True)
                    preset_prompt_checkbox = gr.Checkbox(label="Use a Pre-set Prompt",
                                                         value=False,
                                                         visible=True)
                    with gr.Row(visible=False) as preset_prompt_controls:
                        prev_prompt_page = gr.Button("Previous")
                        next_prompt_page = gr.Button("Next")
                        current_prompt_page_text = gr.Text(f"Page {current_page} of {total_pages}")
                        current_prompt_page_state = gr.State(value=1)

                    preset_prompt = gr.Dropdown(
                        label="Select Preset Prompt",
                        choices=initial_prompts
                    )
                    user_prompt = gr.Textbox(label="Custom Prompt",
                                             placeholder="Enter custom prompt here",
                                             lines=3,
                                             visible=False)
                    system_prompt_input = gr.Textbox(label="System Prompt",
                                                     value="You are a helpful AI assitant",
                                                     lines=3,
                                                     visible=False)
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=800, elem_classes="chatbot-container")
                    msg = gr.Textbox(label="Enter your message")
                    submit = gr.Button("Submit")
                    regenerate_button = gr.Button("Regenerate Last Message")
                    token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                    clear_chat_button = gr.Button("Clear Chat")

                    chat_media_name = gr.Textbox(label="Custom Chat Name(optional)")
                    save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                    save_chat_history_as_file = gr.Button("Save Chat History as File")
                    download_file = gr.File(label="Download Chat History")

            search_button.click(
                fn=update_dropdown_multiple,
                inputs=[search_query_input, search_type_input, keyword_filter_input],
                outputs=[items_output, item_mapping]
            )

            def update_prompt_visibility(custom_prompt_checked, preset_prompt_checked):
                user_prompt_visible = custom_prompt_checked
                system_prompt_visible = custom_prompt_checked
                preset_prompt_visible = preset_prompt_checked
                preset_prompt_controls_visible = preset_prompt_checked
                return (
                    gr.update(visible=user_prompt_visible, interactive=user_prompt_visible),
                    gr.update(visible=system_prompt_visible, interactive=system_prompt_visible),
                    gr.update(visible=preset_prompt_visible, interactive=preset_prompt_visible),
                    gr.update(visible=preset_prompt_controls_visible)
                )

            def update_prompt_page(direction, current_page_val):
                new_page = current_page_val + direction
                if new_page < 1:
                    new_page = 1
                prompts, total_pages, _ = list_prompts(page=new_page, per_page=20)
                if new_page > total_pages:
                    new_page = total_pages
                    prompts, total_pages, _ = list_prompts(page=new_page, per_page=20)
                return (
                    gr.update(choices=prompts),
                    gr.update(value=f"Page {new_page} of {total_pages}"),
                    new_page
                )

            def clear_chat():
                return [], None  # Return empty list for chatbot and None for conversation_id

            custom_prompt_checkbox.change(
                update_prompt_visibility,
                inputs=[custom_prompt_checkbox, preset_prompt_checkbox],
                outputs=[user_prompt, system_prompt_input, preset_prompt, preset_prompt_controls]
            )

            preset_prompt_checkbox.change(
                update_prompt_visibility,
                inputs=[custom_prompt_checkbox, preset_prompt_checkbox],
                outputs=[user_prompt, system_prompt_input, preset_prompt, preset_prompt_controls]
            )

            prev_prompt_page.click(
                lambda x: update_prompt_page(-1, x),
                inputs=[current_prompt_page_state],
                outputs=[preset_prompt, current_prompt_page_text, current_prompt_page_state]
            )

            next_prompt_page.click(
                lambda x: update_prompt_page(1, x),
                inputs=[current_prompt_page_state],
                outputs=[preset_prompt, current_prompt_page_text, current_prompt_page_state]
            )

            submit.click(
                chat_wrapper,
                inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt,
                        conversation_id,
                        save_conversation, temperature, system_prompt_input],
                outputs=[msg, chatbot, conversation_id]
            ).then(  # Clear the message box after submission
                lambda x: gr.update(value=""),
                inputs=[chatbot],
                outputs=[msg]
            ).then(  # Clear the user prompt after the first message
                lambda: (gr.update(value=""), gr.update(value="")),
                outputs=[user_prompt, system_prompt_input]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chatbot],
                outputs=[token_count_display]
            )


            clear_chat_button.click(
                clear_chat,
                outputs=[chatbot, conversation_id]
            )

            items_output.change(
                update_chat_content,
                inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
                outputs=[media_content, selected_parts]
            )

            use_content.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                               outputs=[selected_parts])
            use_summary.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                               outputs=[selected_parts])
            use_prompt.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                              outputs=[selected_parts])
            items_output.change(debug_output, inputs=[media_content, selected_parts], outputs=[])

            search_conversations_btn.click(
                search_conversations,
                inputs=[conversation_search],
                outputs=[previous_conversations]
            )

            load_conversations_btn.click(
                clear_chat,
                outputs=[chatbot, chat_history]
            ).then(
                load_conversation,
                inputs=[previous_conversations],
                outputs=[chatbot, conversation_id]
            )

            previous_conversations.change(
                load_conversation,
                inputs=[previous_conversations],
                outputs=[chat_history]
            )

            save_chat_history_as_file.click(
                save_chat_history,
                inputs=[chatbot, conversation_id],
                outputs=[download_file]
            )

            save_chat_history_to_db.click(
                save_chat_history_to_db_wrapper,
                inputs=[chatbot, conversation_id, media_content, chat_media_name],
                outputs=[conversation_id, gr.Textbox(label="Save Status")]
            )

            regenerate_button.click(
                regenerate_last_message,
                inputs=[chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, temperature,
                        system_prompt_input],
                outputs=[chatbot, save_status]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chatbot],
                outputs=[token_count_display]
            )
        gr.Markdown("# Create Anki Deck")

        with gr.Row():
            # Left Column: Deck Settings
            with gr.Column(scale=1):
                gr.Markdown("## Deck Settings")
                deck_name = gr.Textbox(
                    label="Deck Name",
                    placeholder="My Study Deck",
                    value="My Study Deck"
                )

                deck_description = gr.Textbox(
                    label="Deck Description",
                    placeholder="Description of your deck",
                    lines=2
                )

                note_type = gr.Radio(
                    choices=["Basic", "Basic (and reversed)", "Cloze"],
                    label="Note Type",
                    value="Basic"
                )

                # Card Fields based on note type
                with gr.Group() as basic_fields:
                    front_template = gr.Textbox(
                        label="Front Template (HTML)",
                        value="{{Front}}",
                        lines=3
                    )
                    back_template = gr.Textbox(
                        label="Back Template (HTML)",
                        value="{{FrontSide}}<hr id='answer'>{{Back}}",
                        lines=3
                    )

                with gr.Group() as cloze_fields:
                    cloze_template = gr.Textbox(
                        label="Cloze Template (HTML)",
                        value="{{cloze:Text}}",
                        lines=3,
                        visible=False
                    )

                css_styling = gr.Textbox(
                    label="Card Styling (CSS)",
                    value=".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n\n.cloze {\n font-weight: bold;\n color: blue;\n}",
                    lines=5
                )

            # Right Column: Card Creation
            with gr.Column(scale=1):
                gr.Markdown("## Add Cards")

                with gr.Group() as basic_input:
                    front_content = gr.TextArea(
                        label="Front Content",
                        placeholder="Question or prompt",
                        lines=3
                    )
                    back_content = gr.TextArea(
                        label="Back Content",
                        placeholder="Answer",
                        lines=3
                    )

                with gr.Group() as cloze_input:
                    cloze_content = gr.TextArea(
                        label="Cloze Content",
                        placeholder="Text with {{c1::cloze}} deletions",
                        lines=3,
                        visible=False
                    )

                tags_input = gr.TextArea(
                    label="Tags (comma-separated)",
                    placeholder="tag1, tag2, tag3",
                    lines=1
                )

                add_card_btn = gr.Button("Add Card")

                cards_list = gr.JSON(
                    label="Cards in Deck",
                    value={"cards": []}
                )

                clear_cards_btn = gr.Button("Clear All Cards", variant="stop")

        with gr.Row():
            generate_deck_btn = gr.Button("Generate Deck", variant="primary")
            download_deck = gr.File(label="Download Deck")
            generation_status = gr.Markdown("")

        def update_note_type_fields(note_type: str):
            if note_type == "Cloze":
                return {
                    basic_input: gr.update(visible=False),
                    cloze_input: gr.update(visible=True),
                    basic_fields: gr.update(visible=False),
                    cloze_fields: gr.update(visible=True)
                }
            else:
                return {
                    basic_input: gr.update(visible=True),
                    cloze_input: gr.update(visible=False),
                    basic_fields: gr.update(visible=True),
                    cloze_fields: gr.update(visible=False)
                }

        def add_card(note_type: str, front: str, back: str, cloze: str, tags: str, current_cards: Dict[str, List]):
            if not current_cards:
                current_cards = {"cards": []}

            cards_data = current_cards["cards"]

            # Process tags
            card_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]

            new_card = {
                "id": f"CARD_{len(cards_data) + 1}",
                "tags": card_tags
            }

            if note_type == "Cloze":
                if not cloze or "{{c" not in cloze:
                    return current_cards, "❌ Invalid cloze format. Use {{c1::text}} syntax."
                new_card.update({
                    "type": "cloze",
                    "content": cloze
                })
            else:
                if not front or not back:
                    return current_cards, "❌ Both front and back content are required."
                new_card.update({
                    "type": "basic",
                    "front": front,
                    "back": back,
                    "is_reverse": note_type == "Basic (and reversed)"
                })

            cards_data.append(new_card)
            return {"cards": cards_data}, "✅ Card added successfully!"

        def clear_cards() -> Tuple[Dict[str, List], str]:
            return {"cards": []}, "✅ All cards cleared!"

        def generate_anki_deck(
            deck_name: str,
            deck_description: str,
            note_type: str,
            front_template: str,
            back_template: str,
            cloze_template: str,
            css: str,
            cards_data: Dict[str, List]
        ) -> Tuple[Optional[str], str]:
            try:
                if not cards_data or not cards_data.get("cards"):
                    return None, "❌ No cards to generate deck from!"

                # Create model based on note type
                if note_type == "Cloze":
                    model = genanki.Model(
                        1483883320,  # Random model ID
                        'Cloze Model',
                        fields=[
                            {'name': 'Text'},
                            {'name': 'Back Extra'}
                        ],
                        templates=[{
                            'name': 'Cloze Card',
                            'qfmt': cloze_template,
                            'afmt': cloze_template + '<br><hr id="extra">{{Back Extra}}'
                        }],
                        css=css,
                        # FIXME CLOZE DOESNT EXIST
                        model_type=1
                    )
                else:
                    templates = [{
                        'name': 'Card 1',
                        'qfmt': front_template,
                        'afmt': back_template
                    }]

                    if note_type == "Basic (and reversed)":
                        templates.append({
                            'name': 'Card 2',
                            'qfmt': '{{Back}}',
                            'afmt': '{{FrontSide}}<hr id="answer">{{Front}}'
                        })

                    model = genanki.Model(
                        1607392319,  # Random model ID
                        'Basic Model',
                        fields=[
                            {'name': 'Front'},
                            {'name': 'Back'}
                        ],
                        templates=templates,
                        css=css
                    )

                # Create deck
                deck = genanki.Deck(
                    2059400110,  # Random deck ID
                    deck_name,
                    description=deck_description
                )

                # Add cards to deck
                for card in cards_data["cards"]:
                    if card["type"] == "cloze":
                        note = genanki.Note(
                            model=model,
                            fields=[card["content"], ""],
                            tags=card["tags"]
                        )
                    else:
                        note = genanki.Note(
                            model=model,
                            fields=[card["front"], card["back"]],
                            tags=card["tags"]
                        )
                    deck.add_note(note)

                # Save deck to temporary file
                temp_dir = tempfile.mkdtemp()
                deck_path = os.path.join(temp_dir, f"{deck_name}.apkg")
                genanki.Package(deck).write_to_file(deck_path)

                return deck_path, "✅ Deck generated successfully!"

            except Exception as e:
                return None, f"❌ Error generating deck: {str(e)}"

        # Register event handlers
        note_type.change(
            fn=update_note_type_fields,
            inputs=[note_type],
            outputs=[basic_input, cloze_input, basic_fields, cloze_fields]
        )

        add_card_btn.click(
            fn=add_card,
            inputs=[
                note_type,
                front_content,
                back_content,
                cloze_content,
                tags_input,
                cards_list
            ],
            outputs=[cards_list, generation_status]
        )

        clear_cards_btn.click(
            fn=clear_cards,
            inputs=[],
            outputs=[cards_list, generation_status]
        )

        generate_deck_btn.click(
            fn=generate_anki_deck,
            inputs=[
                deck_name,
                deck_description,
                note_type,
                front_template,
                back_template,
                cloze_template,
                css_styling,
                cards_list
            ],
            outputs=[download_deck, generation_status]
        )


        return (
            deck_name,
            deck_description,
            note_type,
            front_template,
            back_template,
            cloze_template,
            css_styling,
            front_content,
            back_content,
            cloze_content,
            tags_input,
            cards_list,
            add_card_btn,
            clear_cards_btn,
            generate_deck_btn,
            download_deck,
            generation_status
        )

#
# End of Anki_Validation_tab.py
############################################################################################################
