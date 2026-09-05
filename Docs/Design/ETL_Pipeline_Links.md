# Extract Transform Load (ETL) Pipelines


## Introduction
This page serves as documentation regarding the ETL pipelines within tldw and provides context/justification for the details of each.
https://towardsdatascience.com/etl-pipelines-in-python-best-practices-and-techniques-0c148452cc68
https://idp-leaderboard.org/#leaderboard
https://github.com/mickael-kerjean/filestash
https://github.com/opendatalab/OmniDocBench
https://python.langchain.com/docs/integrations/document_loaders/source_code/
https://python.langchain.com/docs/integrations/document_loaders/
https://github.com/kreuzberg-dev/kreuzberg
https://github.com/whyhow-ai/knowledge-table
https://olmocr.allenai.org/papers/olmocr.pdf
https://github.com/NanoNets/docstrange
https://arxiv.org/pdf/2604.12047v1
https://arxiv.org/abs/2604.12352v1
https://arxiv.org/html/2604.10745v1
https://arxiv.org/abs/2601.19827
https://arxiv.org/abs/2602.03442
https://github.com/Ontos-AI/knowledge-graph-traversal-semantic-rag-research
https://github.com/aiptimizer/TurboOCR
https://github.com/OxRML/MADQA

https://www.marktechpost.com/2026/07/24/datalab-marker-v2-vs-mineru-docling-and-liteparse-benchmark-breakdown/
https://github.com/steipete/discrawl
https://github.com/breezedeus/Pix2Text
https://github.com/uxiew/epub2MD
https://huggingface.co/papers/2409.01704
https://huggingface.co/abhinand/GOT-OCR-2.0
https://huggingface.co/papers/2412.07626
https://towardsdatascience.com/ai-powered-information-extraction-and-matchmaking-0408c93ec1b9/
https://github.com/databridge-org/databridge-core
https://github.com/cocoindex-io/cocoindex
https://github.com/AIDC-AI/Ovis
https://github.com/hashangit/Extract2MD
https://github.com/matthsena/AlcheMark
  https://github.com/run-llama/llama_cloud_services
https://github.com/kellyjonbrazil/jc
https://github.com/john-friedman/doc2dict/tree/main
https://github.com/QwenLM/Qwen-Agent
  https://qwenlm.github.io/blog/qwen-agent-2405/
  https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_rag.py
  https://github.com/QwenLM/Qwen-Agent/blob/main/examples/parallel_doc_qa.py
https://github.com/bytedance/Dolphin
https://blog.det.life/i-spent-5-hours-understanding-how-uber-built-their-etl-pipelines-9079735c9103
https://github.com/microsoft/markitdown/tree/main/packages/markitdown/src/markitdown/converters
https://blog.det.life/the-end-of-etl-the-radical-shift-in-data-processing-thats-coming-next-88af7106f7a1
https://github.com/allenai/olmocr
https://github.com/GiftMungmeeprued/document-parsers-list
https://python.langchain.com/docs/how_to/recursive_json_splitter/
https://pypi.org/project/pdfsplit/
https://github.com/run-llama/workflows-py/blob/main/examples/document_processing.ipynb
https://docs.paperless-ngx.com/
https://github.com/GiftMungmeeprued/document-parsers-list
https://github.com/deepanwadhwa/zink
https://github.com/cocoindex-io/cocoindex/tree/main/examples/manuals_llm_extraction
https://cocoindex.io/blogs/academic-papers-indexing/
https://github.com/upstash/context7
https://github.com/dbamman/litbank
https://github.com/google/langextract
https://pypi.org/project/python-pptx/
https://github.com/landing-ai/agentic-doc
https://sarahconstantin.substack.com/p/the-great-data-integration-schlep
https://github.com/AndyTheFactory/newspaper4k
https://github.com/shutootaki/bookwith
https://manual.calibre-ebook.com/generated/en/calibre-server.html
https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/code/#llama_index.core.node_parser.CodeSplitter

https://github.com/landing-ai/agentic-doc


https://github.com/huggingface/aisheets?tab=readme-ov-file#running-ai-sheets-with-custom-and-local-llms
  https://huggingface.co/spaces/aisheets/sheets
  https://huggingface.co/blog/aisheets














 Proper citations and giant-pdf support are the real blockers here. A quick win would be to pull ISBN/DOI metadata on upload, stash it as JSON inside the notebook, then stamp APA/Vancouver refs straight into answers; Crossref’s API is free and usually nails the fields. For page-level anchors, chunk the pdf into 2-page slices at ingest and keep the slice IDs so the model can point to exact spots.

On the size cap, a rolling index that streams in/out sections on demand is lighter than the hard 450-page cutoff and means you don’t have to butcher textbooks. The devs could lift the limit today by letting the user pick a “slow but deep” pass that gives the model more context tokens and time. I’ve been batching uploads through Readwise (for highlight sync) and Alfresco (for versioned docs); APIWrapper.ai slots in as the glue when I need an endpoint to pump new filings straight into the notebook without touching the UI.



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
- **Videos**:
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



llm_trw 26 minutes ago | unvote | prev | next [-]

This is using exactly the wrong tools at every stage of the OCR pipeline, and the cost is astronomical as a result.

You don't use multimodal models to extract a wall of text from an image. They hallucinate constantly the second you get past perfect 100% high-fidelity images.

You use an object detection model trained on documents to find the bounding boxes of each document section as _images_; each bounding box comes with a confidence score for free.

You then feed each box of text to a regular OCR model, also gives you a confidence score along with each prediction it makes.

You feed each image box into a multimodal model to describe what the image is about.

For tables, use a specialist model that does nothing but extract tables-models like GridFormer that aren't hyped to hell and back.

You then stitch everything together in an XML file because Markdown is for human consumption.

You now have everything extracted with flat XML markup for each category the object detection model knows about, along with multiple types of probability metadata for each bounding box, each letter, and each table cell.

You can now start feeding this data programmatically into an LLM to do _text_ processing, where you use the XML to control what parts of the document you send to the LLM.

You then get chunking with location data and confidence scores of every part of the document to put as meta data into the RAG store.

I've build a system that read 500k pages _per day_ using the above completely locally on a machine that cost $20k.
