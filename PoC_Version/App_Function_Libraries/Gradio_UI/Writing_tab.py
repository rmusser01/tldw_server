# Writing_tab.py
# Description: This file contains the functions that are used for writing in the Gradio UI.
#
# Imports
#
# 3rd-Party Imports
import gradio as gr
import textstat
#
# Local Imports
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging
#
########################################################################################################################
#
# Functions:

def adjust_tone(text, concise, casual, api_name, api_key):
    tones = [
        {"tone": "concise", "weight": concise},
        {"tone": "casual", "weight": casual},
        {"tone": "professional", "weight": 1 - casual},
        {"tone": "expanded", "weight": 1 - concise}
    ]
    tones = sorted(tones, key=lambda x: x['weight'], reverse=True)[:2]

    tone_prompt = " and ".join([f"{t['tone']} (weight: {t['weight']:.2f})" for t in tones])

    prompt = f"Rewrite the following text to match these tones: {tone_prompt}. Text: {text}"
    # Performing tone adjustment request...
    adjusted_text = perform_summarization(api_name, text, prompt, api_key)

    return adjusted_text


def grammar_style_check(input_text, custom_prompt, api_name, api_key, system_prompt):
    default_prompt = "Please analyze the following text for grammar and style. Offer suggestions for improvement and point out any misused words or incorrect spellings:\n\n"
    full_prompt = custom_prompt if custom_prompt else default_prompt
    full_text = full_prompt + input_text

    return perform_summarization(api_name, full_text, custom_prompt, api_key, system_prompt)


def create_grammar_style_check_tab():
    with gr.TabItem("Grammar and Style Check", visible=True):
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
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Grammar and Style Check")
                gr.Markdown("This utility checks the grammar and style of the provided text by feeding it to an LLM and returning suggestions for improvement.")
                input_text = gr.Textbox(label="Input Text", lines=10)
                custom_prompt_checkbox = gr.Checkbox(label="Use Custom Prompt", value=False, visible=True)
                system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Please analyze the provided text for grammar and style. Offer any suggestions or points to improve you can identify. Additionally please point out any misuses of any words or incorrect spellings.", lines=5, visible=False)
                custom_prompt_input = gr.Textbox(label="user Prompt",
                                                     value="""<s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>{{ .Prompt }}
""",
                                                     lines=3,
                                                     visible=False)
                custom_prompt_checkbox.change(
                    fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                    inputs=[custom_prompt_checkbox],
                    outputs=[custom_prompt_input, system_prompt_input]
                )
                # Refactored API selection dropdown
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in Config_Files/config.txt)", placeholder="Enter your API key here",
                                               type="password")
                check_grammar_button = gr.Button("Check Grammar and Style")

            with gr.Column():
                gr.Markdown("# Resulting Suggestions")
                gr.Markdown("(Keep in mind the API used can affect the quality of the suggestions)")

                output_text = gr.Textbox(label="Grammar and Style Suggestions", lines=15)

            check_grammar_button.click(
                fn=grammar_style_check,
                inputs=[input_text, custom_prompt_input, api_name_input, api_key_input, system_prompt_input],
                outputs=output_text
            )


def create_tone_adjustment_tab():
    with gr.TabItem("Tone Analyzer & Editor", visible=True):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", lines=10)
                concise_slider = gr.Slider(minimum=0, maximum=1, value=0.5, label="Concise vs Expanded")
                casual_slider = gr.Slider(minimum=0, maximum=1, value=0.5, label="Casual vs Professional")
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    label="API for Grammar Check"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in Config_Files/config.txt)", placeholder="Enter your API key here",
                                               type="password")
                adjust_btn = gr.Button("Adjust Tone")

            with gr.Column():
                output_text = gr.Textbox(label="Adjusted Text", lines=15)

                adjust_btn.click(
                    adjust_tone,
                    inputs=[input_text, concise_slider, casual_slider],
                    outputs=output_text
                )


persona_prompts = {
    "Hemingway": "As Ernest Hemingway, known for concise and straightforward prose, provide feedback on the following text:",
    "Shakespeare": "Channel William Shakespeare's poetic style and provide feedback on the following text:",
    "Jane Austen": "Embodying Jane Austen's wit and social commentary, critique the following text:",
    "Stephen King": "With Stephen King's flair for suspense and horror, analyze the following text:",
    "J.K. Rowling": "As J.K. Rowling, creator of the magical world of Harry Potter, review the following text:"
}

def generate_writing_feedback(text, persona, aspect, api_name, api_key):
    if isinstance(persona, dict):  # If it's a character card
        base_prompt = f"You are {persona['name']}. {persona['personality']}\n\nScenario: {persona['scenario']}\n\nRespond to the following message in character:"
    else:  # If it's a regular persona
        base_prompt = persona_prompts.get(persona, f"As {persona}, provide feedback on the following text:")

    if aspect != "Overall":
        prompt = f"{base_prompt}\n\nFocus specifically on the {aspect.lower()} in the following text:\n\n{text}"
    else:
        prompt = f"{base_prompt}\n\n{text}"

    return perform_summarization(api_name, text, prompt, api_key, system_message="You are a helpful AI assistant. You will respond to the user as if you were the persona declared in the user prompt.")

def generate_writing_prompt(persona, api_name, api_key):
    prompt = f"Generate a writing prompt in the style of {persona}. The prompt should inspire a short story or scene that reflects {persona}'s typical themes and writing style."
    #FIXME
    return perform_summarization(api_name, prompt, "", api_key, system_message="You are a helpful AI assistant. You will respond to the user as if you were the persona declared in the user prompt." )

def calculate_readability(text):
    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    return f"Readability: Flesch Reading Ease: {ease:.2f}, Flesch-Kincaid Grade Level: {grade:.2f}"


def generate_feedback_history_html(history):
    html = "<h3>Recent Feedback History</h3>"
    for entry in reversed(history):
        html += f"<details><summary>{entry['persona']} Feedback</summary>"
        html += f"<p><strong>Original Text:</strong> {entry['text'][:100]}...</p>"

        feedback = entry.get('feedback')
        if feedback:
            html += f"<p><strong>Feedback:</strong> {feedback[:200]}...</p>"
        else:
            html += "<p><strong>Feedback:</strong> No feedback provided.</p>"

        html += "</details>"
    return html


# FIXME
def create_document_feedback_tab():
    with gr.TabItem("Writing Feedback", visible=True):
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Your Writing", lines=10)
                persona_dropdown = gr.Dropdown(
                    label="Select Persona",
                    choices=[
                        "Agatha Christie",
                        "Arthur Conan Doyle",
                        "Charles Bukowski",
                        "Charles Dickens",
                        "Chinua Achebe",
                        "Cormac McCarthy",
                        "David Foster Wallace",
                        "Edgar Allan Poe",
                        "F. Scott Fitzgerald",
                        "Flannery O'Connor",
                        "Franz Kafka",
                        "Fyodor Dostoevsky",
                        "Gabriel Garcia Marquez",
                        "George R.R. Martin",
                        "George Orwell",
                        "Haruki Murakami",
                        "Hemingway",
                        "Herman Melville",
                        "Isabel Allende",
                        "James Joyce",
                        "Jane Austen",
                        "J.K. Rowling",
                        "J.R.R. Tolkien",
                        "Jorge Luis Borges",
                        "Kurt Vonnegut",
                        "Leo Tolstoy",
                        "Margaret Atwood",
                        "Mark Twain",
                        "Mary Shelley",
                        "Milan Kundera",
                        "Naguib Mahfouz",
                        "Neil Gaiman",
                        "Octavia Butler",
                        "Philip K Dick",
                        "Ray Bradbury",
                        "Salman Rushdie",
                        "Shakespeare",
                        "Stephen King",
                        "Toni Morrison",
                        "T.S. Eliot",
                        "Ursula K. Le Guin",
                        "Virginia Woolf",
                        "Virginia Woolf",
                        "Zadie Smith"],
                    value="Hemingway"
                )
                custom_persona_name = gr.Textbox(label="Custom Persona Name")
                custom_persona_description = gr.Textbox(label="Custom Persona Description", lines=3)
                add_custom_persona_button = gr.Button("Add Custom Persona")
                aspect_dropdown = gr.Dropdown(
                    label="Focus Feedback On",
                    choices=["Overall", "Grammar", "Word choice", "Structure of delivery", "Character Development", "Character Dialogue", "Descriptive Language", "Plot Structure"],
                    value="Overall"
                )
                api_name_input = gr.Dropdown(
                    choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                             "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace", "Custom-OpenAI-API"],
                    value=None,
                    label="API for Feedback"
                )
                api_key_input = gr.Textbox(label="API Key (if not set in Config_Files/config.txt)", type="password")
                get_feedback_button = gr.Button("Get Feedback")
                generate_prompt_button = gr.Button("Generate Writing Prompt")

            with gr.Column(scale=2):
                feedback_output = gr.Textbox(label="Feedback", lines=15)
                readability_output = gr.Textbox(label="Readability Metrics")
                feedback_history_display = gr.HTML(label="Feedback History")

        with gr.Row():
            compare_personas = gr.CheckboxGroup(
                choices=[
                    "Agatha Christie",
                    "Arthur Conan Doyle",
                    "Charles Bukowski",
                    "Charles Dickens",
                    "Chinua Achebe",
                    "Cormac McCarthy",
                    "David Foster Wallace",
                    "Edgar Allan Poe",
                    "F. Scott Fitzgerald",
                    "Flannery O'Connor",
                    "Franz Kafka",
                    "Fyodor Dostoevsky",
                    "Gabriel Garcia Marquez",
                    "George R.R. Martin",
                    "George Orwell",
                    "Haruki Murakami",
                    "Hemingway",
                    "Herman Melville",
                    "Isabel Allende",
                    "James Joyce",
                    "Jane Austen",
                    "J.K. Rowling",
                    "J.R.R. Tolkien",
                    "Jorge Luis Borges",
                    "Kurt Vonnegut",
                    "Leo Tolstoy",
                    "Margaret Atwood",
                    "Mark Twain",
                    "Mary Shelley",
                    "Milan Kundera",
                    "Naguib Mahfouz",
                    "Neil Gaiman",
                    "Octavia Butler",
                    "Philip K Dick",
                    "Ray Bradbury",
                    "Salman Rushdie",
                    "Shakespeare",
                    "Stephen King",
                    "Toni Morrison",
                    "T.S. Eliot",
                    "Ursula K. Le Guin",
                    "Virginia Woolf",
                    "Virginia Woolf",
                    "Zadie Smith"],
                label="Compare Multiple Persona's Feedback at Once(Compares existing feedback, doesn't create new ones)"
            )
        with gr.Row():
            compare_button = gr.Button("Compare Feedback")

        feedback_history = gr.State([])

        def add_custom_persona(name, description):
            updated_choices = persona_dropdown.choices + [name]
            persona_prompts[name] = f"As {name}, {description}, provide feedback on the following text:"
            return gr.update(choices=updated_choices)

        def update_feedback_history(current_text, persona, feedback):
            # Ensure feedback_history.value is initialized and is a list
            if feedback_history.value is None:
                feedback_history.value = []

            history = feedback_history.value

            # Append the new entry to the history
            history.append({"text": current_text, "persona": persona, "feedback": feedback})

            # Keep only the last 5 entries in the history
            feedback_history.value = history[-10:]

            # Generate and return the updated HTML
            return generate_feedback_history_html(feedback_history.value)

        def compare_feedback(text, selected_personas, api_name, api_key):
            results = []
            for persona in selected_personas:
                feedback = generate_writing_feedback(text, persona, "Overall", api_name, api_key)
                results.append(f"### {persona}'s Feedback:\n{feedback}\n\n")
            return "\n".join(results)

        add_custom_persona_button.click(
            fn=add_custom_persona,
            inputs=[custom_persona_name, custom_persona_description],
            outputs=persona_dropdown
        )

        get_feedback_button.click(
            fn=lambda text, persona, aspect, api_name, api_key: (
                generate_writing_feedback(text, persona, aspect, api_name, api_key),
                calculate_readability(text),
                update_feedback_history(text, persona, generate_writing_feedback(text, persona, aspect, api_name, api_key))
            ),
            inputs=[input_text, persona_dropdown, aspect_dropdown, api_name_input, api_key_input],
            outputs=[feedback_output, readability_output, feedback_history_display]
        )

        compare_button.click(
            fn=compare_feedback,
            inputs=[input_text, compare_personas, api_name_input, api_key_input],
            outputs=feedback_output
        )

        generate_prompt_button.click(
            fn=generate_writing_prompt,
            inputs=[persona_dropdown, api_name_input, api_key_input],
            outputs=input_text
        )

    return input_text, feedback_output, readability_output, feedback_history_display


def create_creative_writing_tab():
    with gr.TabItem("Creative Writing Assistant", visible=True):
        gr.Markdown("# Utility to be added...")



def create_mikupad_tab():
    with gr.TabItem("Mikupad", visible=True):
        gr.Markdown("I Wish. Gradio won't embed it successfully...")

#
# End of Writing_tab.py
########################################################################################################################
