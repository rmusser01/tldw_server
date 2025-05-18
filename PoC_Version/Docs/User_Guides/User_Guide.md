# tldw User Guide (WIP)

## TABLE OF CONTENTS
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Simple Instructions](#simple-instructions)
- [Detailed Usage](#detailed-usage)
  - [Transcription / Summarization / Ingestion](#transcription--summarization--ingestion)
  - [Search / Detailed View](#search--detailed-view)
  - [Chat with an LLM](#chat-with-an-llm)
  - [Edit Existing Items](#edit-existing-items)
  - [Writing Tools](#writing-tools)
  - [Keywords](#keywords)
  - [Import/Export](#importexport)
  - [Utilities](#utilities)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Feedback](#feedback)
- [Contributing](#contributing)


https://openai.com/chatgpt/use-cases/student-writing-guide/
https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters

Look at:
    https://upwarddynamism.com/2024/11/24/free-pocket-professor-7-ways-lifelong-learners-crush-it-with-notebooklm/


### <a name="quick-start"></a>Quick Start
Quick Start: Just click on the appropriate tab for what you're trying to do and fill in the required fields. Click "Process <video/audio/article/etc>" and wait for the results.
Simple Instructions

    Basic Usage:
        If you don't have an API key/don't know what an LLM is/don't know what an API key is, please look further down the page for information on getting started.
        If you want summaries/chat with an LLM, you'll need:
            An API key for the LLM API service you want to use, or,
            A local inference server running an LLM (like llamafile-server/llama.cpp - for instructions on how to do so see the projects README or below), or,
            A "local" inference server you have access to running an LLM.
        If you just want transcriptions you can ignore the above.
        Select the tab for the task you want to perform
        Fill in the required fields
        Click the "Process" button
        Wait for the results to appear
        Download the results if needed
        Repeat as needed
        As of writing this, the UI is still a work in progress.
        That being said, I plan to replace it all eventually. In the meantime, please have patience.
        The UI is divided into tabs for different tasks.
        Each tab has a set of fields that you can fill in to perform the task.
        Some fields are mandatory, some are optional.
        The fields are mostly self-explanatory, but I will try to add more detailed instructions as I go.


### <a name="introduction"></a>Introduction
This is a user guide for the tldw project. The tldw project is a web-based tool that allows users to interact with various LLMs (Large Language Models) to perform tasks like transcription, summarization, chat, and more. The project is still in the early stages of development, so the UI is a work in progress. That being said, I plan to replace it all(UI) eventually. In the meantime, please have patience for its shittiness.


### <a name="detailed-usage"></a>Detailed Usage
- **Detailed Usage:**
    - Currently, there are 15 Top-level tabs in the UI. Each tab has a specific set of tasks that you can perform by selecting one of the 'sub-tabs' made available by clicking on the top tab.
    - **The tabs are as follows:**
        - `Transcribe / Analyze / Ingestion` - This tab is for processing videos, audio files, articles, books, and PDFs/office docs.
        - `RAG Chat & Search` - This tab is for chatting with an LLM and searching the RAG database.
        - `Chat with an LLM` - This tab is for chatting with an LLM to generate content based on the selected item and prompts.
        - `Search / Detailed View` - This tab is for searching and displaying content from the database. You can also view detailed information about the selected item.
        - `Character Chat` - This tab is for .
        - `Writing Tools` - This tab is for using various writing tools like Grammar & Style check, Tone Analyzer & Editor, etc.
        - `Search / View DB Items` - 
        - `Prompts` - 
        - `Manage Media DB Items` - 
        - `Embeddings Management` - 
        - `Keywords` - 
        - `Import` - 
        - `Export` - 
        - `Database Management` - 
        - `Utilities` -
        - `Anki Deck Creation/Validation` - 
        - `Local LLM` - 
        - `Trashcan` - 
        - `Evaluations` - 
        - `Config Editor` - 
        - Edit Existing Items - This tab is for editing existing items in the database (Prompts + ingested items).
              Writing Tools - 
              Keywords - This tab is for managing keywords for content search and display.
              Import/Export - This tab is for importing notes from Obsidian and exporting keywords/items to markdown/CSV.
              Utilities - This tab contains some random utilities that I thought might be useful.
          Each sub-tab is responsible for that set of functionality. This is reflected in the codebase as well, where I have split the functionality into separate files for each tab/larger goal.


#### <a name="transcription--summarization--ingestion"></a>Transcription / Summarization / Ingestion

#### <a name="search--detailed-view"></a> Search / Detailed View

#### <a name="chat-with-an-llm"></a> Chat with an LLM

#### <a name="edit-existing-items"></a> Edit Existing Items

#### <a name="writing-tools"></a> Writing Tools

#### <a name="keywords"></a> Keywords

#### <a name="importexport"></a> Import/Export

#### <a name="utilities"></a> Utilities


### <a name="faq"></a>FAQ
- [Troubleshooting](#troubleshooting)
- [Feedback](#feedback)
- [Contributing](#contributing)