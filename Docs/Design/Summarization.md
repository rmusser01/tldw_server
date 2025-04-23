# Summarization

## Introduction

This page is to document the 'summarization' workflow for use within the project. The goal is to create a system that can automatically generate summaries of text, documents, and other content.



### Summarization Goals


### Relevant Research



### Link Dump
https://neptune.ai/blog/llm-evaluation-text-summarization
https://phoenix.arize.com/llm-summarization-getting-to-production/
https://blog.metrostar.com/iteratively-summarize-long-documents-llm
https://arxiv.org/html/2412.15487v1
https://arxiv.org/pdf/2204.01849
https://arxiv.org/pdf/2501.08167
https://aizip.substack.com/p/evaluating-small-language-models
https://github.com/aizip/RED-flow
Unread/Unreviewed
	Articles
		https://github.com/aws-samples/llm-based-advanced-summarization
		https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb
		https://github.com/cognitivetech/llm-long-text-summarization
			https://github.com/cognitivetech/llm-long-text-summarization/tree/main/walkthrough
			https://github.com/cognitivetech/llm-long-text-summarization/tree/main/walkthrough/privateGPT
		https://devblogs.microsoft.com/ise/gpt-summary-prompt-engineering/
		https://medium.com/@singhrajni2210/large-language-models-and-text-summarization-a-powerful-combination-6400e7643b70
	Code
		https://github.com/ilyagusev/talestudio/blob/main/tale_studio/summarize_book.py
	Models + Papers
		https://huggingface.co/blog/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes
		https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes_GGUF
	Papers
		https://openreview.net/forum?id=7Ttk3RzDeu
		https://paperswithcode.com/paper/on-learning-to-summarize-with-large-language
		https://huggingface.co/papers/2402.14848
		https://ijisae.org/index.php/IJISAE/article/view/4500
		https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
		https://arxiv.org/pdf/2301.13848
	Videos
		https://www.youtube.com/watch?v=whGC1Cj_Ka8
	Prompts
		https://github.com/ilyagusev/talestudio/blob/main/tale_studio/prompts/existing_book/l1_summarize.jinja



```
 I would use a model like moondream2 and apply for each page a prompt "describe this page" and then another prompt with the current page and previous N page descriptions to ask if a new document started 
```

```
 But the gist of it is that I take a digital copy of a book and break it into separate, numbered, sections to keep them below the context limit of the LLM. That's passed on to a LLM (for me, I either go local using llamacpp's api. Mixtral has given me the most consistent results there for local models. Or cloud with gemini) with something like "give a chronological, numbered, sequence of events for the text." as part of the prompt. Basically it just iterates over every chunk in the json file I created from the original ebook and writes the output to a new json file. From there it can be hand edited as needed to repair any malformed elements. Assuming valid json formatting, it's easy to then script out something to just pick the list items, join them together, and fix the numbering so that the list from section 3 continues from 2, 2 from 1, etc. For your exact case you'd just need to insert some kind of a check for the chapter strings at some point.

With fiction I've also found that supplying the LLM with the book's name can often really help. Pushing it in the right direction to "understand" the setting, characters, etc.

I basically just scripted everything out in python. It sounds like a pain, but it's really just some loops, strings, and a post/response with the LLM's api. 
```
