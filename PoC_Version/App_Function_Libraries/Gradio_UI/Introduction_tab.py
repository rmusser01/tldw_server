# Introduction_tab.py
# Gradio UI functions for the Introduction tab
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import get_db_config
#
####################################################################################################
#
# Functions:



def create_introduction_tab():
    with gr.TabItem("Introduction", visible=True):
        db_config = get_db_config()
        db_type = db_config['type']
        gr.Markdown(f"# tldw: Your LLM-powered Research Multi-tool (Using {db_type.capitalize()} Database)")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""### What can it do?
                - Transcribe and summarize videos from URLs/Local files
                - Transcribe and Summarize Audio files/Podcasts (URL/local file)
                - Summarize articles from URLs/Local notes
                - Ingest and summarize books(epub/PDF)
                - Ingest and summarize research papers (PDFs - WIP)
                - Search and display ingested content + summaries
                - Create and manage custom prompts
                - Chat with an LLM of your choice to generate content using the selected item + Prompts
                - Keyword support for content search and display
                - Export keywords/items to markdown/CSV(csv is wip)
                - Import existing notes from Obsidian to the database (Markdown/txt files or a zip containing a collection of files)
                - View and manage chat history
                - Writing Tools: Grammar & Style check, Tone Analyzer & Editor, more planned...
                - RAG (Retrieval-Augmented Generation) support for content generation(think about asking questions about your entire library of items)
                - More features planned...
                - All powered by your choice of LLM.
                    - Currently supports: Local-LLM(llamafile-server), OpenAI, Anthropic, Cohere, Groq, DeepSeek, OpenRouter, Llama.cpp, Kobold, Ooba, Tabbyapi, VLLM and more to come...
                - All data is stored locally in a SQLite database for easy access and management.
                - No trackers (Gradio has some analytics but it's disabled here...)
                - No ads, no tracking, no BS. Just you and your content.
                - Open-source and free to use. Contributions welcome!
                - If you have any thoughts or feedback, please let me know on github or via email.
                """)
                gr.Markdown(
                    """Follow this project at [tl/dw: Too Long, Didn't Watch - Your Personal Research Multi-Tool - GitHub](https://github.com/rmusser01/tldw)""")
            with gr.Column():
                gr.Markdown("""### How to use:
                ##### Quick Start: Just click on the appropriate tab for what you're trying to do and fill in the required fields. Click "Process <video/audio/article/etc>" and wait for the results.
                #### Simple Instructions
                - Basic Usage:
                    - If you don't have an API key/don't know what an LLM is/don't know what an API key is, please look further down the page for information on getting started.
                    - If you want summaries/chat with an LLM, you'll need:
                        1. An API key for the LLM API service you want to use, or,
                        2. A local inference server running an LLM (like llamafile-server/llama.cpp - for instructions on how to do so see the projects README or below), or,
                        3. A "local" inference server you have access to running an LLM.
                    - If you just want transcriptions you can ignore the above.
                    - Select the tab for the task you want to perform
                    - Fill in the required fields
                    - Click the "Process" button
                    - Wait for the results to appear
                    - Download the results if needed
                    - Repeat as needed
                    - As of writing this, the UI is still a work in progress.
                    - That being said, I plan to replace it all eventually. In the meantime, please have patience.
                    - The UI is divided into tabs for different tasks.
                    - Each tab has a set of fields that you can fill in to perform the task.
                    - Some fields are mandatory, some are optional.
                    - The fields are mostly self-explanatory, but I will try to add more detailed instructions as I go.
                #### Detailed Usage:
                - There are 8 Top-level tabs in the UI. Each tab has a specific set of tasks that you can perform by selecting one of the 'sub-tabs' made available by clicking on the top tab.
                - The tabs are as follows:
                    1. Transcription / Summarization / Ingestion - This tab is for processing videos, audio files, articles, books, and PDFs/office docs.
                    2. Search / Detailed View - This tab is for searching and displaying content from the database. You can also view detailed information about the selected item.
                    3. Chat with an LLM - This tab is for chatting with an LLM to generate content based on the selected item and prompts.
                    4. Edit Existing Items - This tab is for editing existing items in the database (Prompts + ingested items).
                    5. Writing Tools - This tab is for using various writing tools like Grammar & Style check, Tone Analyzer & Editor, etc.
                    6. Keywords - This tab is for managing keywords for content search and display.
                    7. Import/Export - This tab is for importing notes from Obsidian and exporting keywords/items to markdown/CSV.
                    8. Utilities - This tab contains some random utilities that I thought might be useful.
                - Each sub-tab is responsible for that set of functionality. This is reflected in the codebase as well, where I have split the functionality into separate files for each tab/larger goal.
                """)
        with gr.Row():
            gr.Markdown("""### HELP! I don't know what any of this this shit is!
            ### DON'T PANIC
            #### Its ok, you're not alone, most people have no clue what any of this stuff is.
            - So let's try and fix that.

            #### Introduction to LLMs:
            - Non-Technical introduction to Generative AI and LLMs: https://paruir.medium.com/understanding-generative-ai-and-llms-a-non-technical-overview-part-1-788c0eb0dd64
            - Google's Intro to LLMs: https://developers.google.com/machine-learning/resources/intro-llms#llm_considerations
            - LLMs 101(coming from a tech background): https://vinija.ai/models/LLM/
            - LLM Fundamentals / LLM Scientist / LLM Engineer courses(Free): https://github.com/mlabonne/llm-course

            #### Various Phrases & Terms to know
            - **LLM** - Large Language Model - A type of neural network that can generate human-like text.
            - **API** - Application Programming Interface - A set of rules and protocols that allows one software application to communicate with another.
                * Think of it like a post address for a piece of software. You can send messages to and from it.
            - **API Key** - A unique identifier that is used to authenticate a user, developer, or calling program to an API.
                * Like the key to a post office box. You need it to access the contents.
            - **GUI** - Graphical User Interface - the thing facilitating your interact with this application.
            - **DB** - Database
            - **Prompt Engineering** - The process of designing prompts that are used to guide the output of a language model. Is a meme but also very much not.
            - **Quantization** - The process of converting a continuous range of values into a finite range of discrete values.
                * https://github.com/ggerganov/llama.cpp/blob/cddae4884c853b1a7ab420458236d666e2e34423/examples/quantize/README.md#L27
            - **GGUF Files** - GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML. https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
            - **Inference Engine** - A software system that is designed to execute a model that has been trained by a machine learning algorithm. Llama.cpp and Kobold.cpp are examples of inference engines.
            - **Abliteration** - https://huggingface.co/blog/mlabonne/abliteration
            """)
        with gr.Row():
            gr.Markdown("""### Ok cool, but how do I get started? I don't have an API key or a local server running...
                #### Great, glad you asked! Getting Started:
                - **Getting an API key for a commercial services provider:
                    - **OpenAI:**
                        * https://platform.openai.com/docs/quickstart
                    - **Anthropic:**
                        * https://docs.anthropic.com/en/api/getting-started
                    - **Cohere:**
                        * https://docs.cohere.com/
                        * They offer 1k free requests a month(up to 1million tokens total I think?), so you can try it out without paying.
                    - **Groq:**
                        * https://console.groq.com/keys
                        * Offer an account with free credits to try out their service. No idea how much you get.
                    - **DeepSeek:**
                        * https://platform.deepseek.com/ (Chinese-hosted/is in english)
                    - **OpenRouter:**
                        * https://openrouter.ai/
                    - **Mistral:**
                        * https://console.mistral.ai/
                - **Choosing a Model to download**
                    - You'll first need to select a model you want to use with the server.
                        - Keep in mind that the model you select will determine the quality of the output you get, and that models run fastest when offloaded fully to your GPU.
                        * So this means that you can run a large model (Command-R) on CPU+System RAM, but you're gonna see a massive performance hit. Not saying its unusable, but it's not ideal.
                        * With that in mind, I would recommend an abliterated version of Meta's Llama3.1 model for most tasks. (Abliterated since it won't refuse requests)
                        * I say this because of the general quality of the model + it's context size.
                        * You can find the model here: https://huggingface.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF
                        * And the Q8 quant(total size 8.6GB): https://huggingface.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF/resolve/main/meta-llama-3.1-8b-instruct-abliterated.Q8_0.gguf?download=true
                - **Local Inference Server:**
                    - **Llamafile-Server (wrapper for llama.cpp):**
                        * Run this script with the `--local_llm` argument next time, and you'll be walked through setting up a local instance of llamafile-server.
                    - **Llama.cpp Inference Engine:**
                        * Download the latest release for your platform here: https://github.com/ggerganov/llama.cpp/releases
                        * Windows: `llama-<release_number>-bin-win-cuda-cu<11.7.1 or 12.2.0 - version depends on installed cuda>-x64.zip`
                            * Run it: `llama-server.exe --model <path_to_model> -ctx 8192 -ngl 999`
                                - `-ctx 8192` sets the context size to 8192 tokens, `-ngl 999` sets the number of layers to offload to the GPU to 999. (essentially ensuring we only use our GPU and not CPU for processing)
                        * Macos: `llama-<release_number>-bin-macos-arm64.zip - for Apple Silicon / `llama-<release_number>-bin-macos-x64.zip` - for Intel Macs
                            * Run it: `llama-server --model <path_to_model> -ctx 8192 -ngl 999`
                                - `-ctx 8192` sets the context size to 8192 tokens, `-ngl 999` sets the number of layers to offload to the GPU to 999. (essentially ensuring we only use our GPU and not CPU for processing)
                        * Linux: You can probably figure it out.
                    - **Kobold.cpp Server:**
                        1. Download from here: https://github.com/LostRuins/koboldcpp/releases/latest
                        2. `Double click KoboldCPP.exe and select model OR run "KoboldCPP.exe --help" in CMD prompt to get command line arguments for more control.`
                        3. `Generally you don't have to change much besides the Presets and GPU Layers. Run with CuBLAS or CLBlast for GPU acceleration.`
                        4. `Select your GGUF or GGML model you downloaded earlier, and connect to the displayed URL once it finishes loading.`
                    - **Linux**
                        1. `On Linux, we provide a koboldcpp-linux-x64 PyInstaller prebuilt binary on the releases page for modern systems. Simply download and run the binary.`
                            * Alternatively, you can also install koboldcpp to the current directory by running the following terminal command: `curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64 && chmod +x koboldcpp`
                        2. When you can't use the precompiled binary directly, we provide an automated build script which uses conda to obtain all dependencies, and generates (from source) a ready-to-use a pyinstaller binary for linux users. Simply execute the build script with `./koboldcpp.sh dist` and run the generated binary.
            """)

#
# End of Introduction_tab.py
####################################################################################################
