# Plan for RAG implementation and notes relating to it.

## Table of Contents
- [Goal](#goal)
- [Plan](#plan)
- [Links](#links)



Agentic RAG
	https://arxiv.org/pdf/2501.09136
	https://github.com/asinghcsu/AgenticRAG-Survey

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


```
Taken from 
Systematically Improving RAG Applications prep:

**Week 1: Cold Start Problem**

- Overview of the Playbook and RAG System Inference Flywheel
- Understanding and Justifying the Playbook
- LLM Synthetic Data Generation for Evaluation
- Leading vs Lagging Metrics
- Evaluation and Improvement Strategies

**Week 2: Identifying Failure Modes**

- Segmenting input space of questions
- Developing specialized systems based on topics and  capabilities 
- Improving precision and recall before considering generation
- Working with domain experts to decompose problem space 

**Week 3: Extending Capabilities**

- Understanding limited inventory and capabilities issues 
- Metadata filtering
- Rerankers w/ COhere 
- Fine-Tuning Embedding Models
- Multimodal data handling (PDFs, PPTs, Tables, Images)

**Week 4: Representations**

- Understanding Embeddings and Search
- Routing queries using classifiers
- Evaluating routing effectiveness
- Handling multiple routes
- Parallel Routing with instructor 
- Identifying what tools to provide agents 

**Week 5: Product**

- Generation techniques
- Streaming responses
- Managing citations
- Feedback UI and collecting labels
- Prompt engineering and rejecting wor
```
