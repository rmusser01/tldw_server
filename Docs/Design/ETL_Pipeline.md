# Extract Transform Load (ETL) Pipelines


## Introduction
This page serves as documentation regarding the ETL pipelines within tldw and provides context/justification for the details of each.
https://towardsdatascience.com/etl-pipelines-in-python-best-practices-and-techniques-0c148452cc68

https://python.langchain.com/docs/integrations/document_loaders/source_code/
https://python.langchain.com/docs/integrations/document_loaders/
https://github.com/whyhow-ai/knowledge-table
https://olmocr.allenai.org/papers/olmocr.pdf
https://github.com/breezedeus/Pix2Text
https://huggingface.co/papers/2409.01704
https://huggingface.co/abhinand/GOT-OCR-2.0
https://huggingface.co/papers/2412.07626
https://towardsdatascience.com/ai-powered-information-extraction-and-matchmaking-0408c93ec1b9/
https://github.com/databridge-org/databridge-core
https://github.com/cocoindex-io/cocoindex
https://github.com/AIDC-AI/Ovis
https://github.com/matthsena/AlcheMark
  https://github.com/run-llama/llama_cloud_services
https://github.com/kellyjonbrazil/jc
https://github.com/john-friedman/doc2dict/tree/main
https://github.com/QwenLM/Qwen-Agent
  https://qwenlm.github.io/blog/qwen-agent-2405/
  https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_rag.py
  https://github.com/QwenLM/Qwen-Agent/blob/main/examples/parallel_doc_qa.py




## ETL Pipelines

### Data Sources
- **Audio**
    - faster_whisper
    - pyaudio
- **Ebooks (epub)**
    - ebooklib
- **PDFs**
    - Docling
    - pymupdf4llm
- **Plain Text(`.md`, `.txt`)**
    - stdlib
- **PowerPoint Presentations** - need to add
    - docling
    - https://github.com/ssine/pptx2md
- **Rich Text(`.rtf`, `.docx`)**
    - doc2txt
    - pypandoc
- **RSS Feeds**: 
    - f
- **Videos**
    - f
- **Websites**: 
    - playwright
    - bs4
    - requests
- **XML Files**
    - xml.etree.ElementTree
- **3rd-Party Services**
    - Sharepoint
        * https://llamahub.ai/l/readers/llama-index-readers-microsoft-sharepoint
        * 

### Tools
https://github.com/ucbepic/docetl
https://ucbepic.github.io/docetl/concepts/optimization/


### Links
https://arxiv.org/html/2410.21169



### Link Dump:
https://github.com/shoryasethia/FinChat
https://github.com/dgunning/edgartools
Confluence
  https://openwebui.com/t/romainneup/confluence_search


 	
llm_trw 26 minutes ago | unvote | prev | next [–]

This is using exactly the wrong tools at every stage of the OCR pipeline, and the cost is astronomical as a result.

You don't use multimodal models to extract a wall of text from an image. They hallucinate constantly the second you get past perfect 100% high-fidelity images.

You use an object detection model trained on documents to find the bounding boxes of each document section as _images_; each bounding box comes with a confidence score for free.

You then feed each box of text to a regular OCR model, also gives you a confidence score along with each prediction it makes.

You feed each image box into a multimodal model to describe what the image is about.

For tables, use a specialist model that does nothing but extract tables—models like GridFormer that aren't hyped to hell and back.

You then stitch everything together in an XML file because Markdown is for human consumption.

You now have everything extracted with flat XML markup for each category the object detection model knows about, along with multiple types of probability metadata for each bounding box, each letter, and each table cell.

You can now start feeding this data programmatically into an LLM to do _text_ processing, where you use the XML to control what parts of the document you send to the LLM.

You then get chunking with location data and confidence scores of every part of the document to put as meta data into the RAG store.

I've build a system that read 500k pages _per day_ using the above completely locally on a machine that cost $20k.