# Plan for RAG implementation and notes relating to it.

## Table of Contents
- [Goal](#goal)
- [Plan](#plan)
- [Links](#links)




https://huggingface.co/PleIAs/Pleias-RAG-1B-gguf
https://docs.anthropic.com/en/api/files-create
https://github.com/nishchaljs/MobiRAG
https://arxiv.org/abs/2504.13587
https://github.com/lmarena/search-arena
https://gerred.github.io/building-an-agentic-system/
https://github.com/deepsense-ai/ragbits
https://arxiv.org/pdf/2504.08748v1

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Goal<a name="goal"></a>
- **Main Goals**
	1. Generalized RAG setup for whatever.
	2. Easy to use/decompose
	3. Modular. Possible to use/not use various parts as defined in the UI.
	4. Knobs/Dials exposed to the user through the UI
	5. Provides Citations

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Plan<a name="plan"></a>
1. Pre-Query Steps
	1. Data is first parsed. 
	2. Data is then chunked
		- Chunking depends on data type
			1. Websites
			2. Ebooks
			3. PDFs
			4. Unstructured text
		- Chunking strategy for all items involves a tree ToC as part of each chunk
			- Each chunk contains: 
				* a summary of the parent document
				* Summary of the chunk (if larger than X size)
				* a #/# indicating where the chunk exists in relation to the larger document
				* 
	3. Data is then converted to vector Embeddings and stored using metadata from main DB entries for keyword searches
2. Performing a Query:
3. Performing Ranking and Fusion
4. Performing Selection and Prompt Generation
5. Performing Prompt Evaluation and Submittal
6. Reviewing output + Re-Prompting if Necessary
7. Providing Citations for generated result(s) + Delivery & Presentation to the user
8. Metrics for every one of the above steps
	- Time taken
	- 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Improvements<a name="improvements"></a>

https://arxiv.org/abs/2407.21712

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Links<a name="links"></a>
- Search
  	- https://github.com/SeekStorm/SeekStorm

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
