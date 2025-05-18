# ISSUES

### Documented Issues:
1. FFmpeg is missing
2. cudnn8dlops.dll or whatever is missing/not in PATH
   * https://stackoverflow.com/questions/66083545/could-not-load-dynamic-library-cudnn64-8-dll-dlerror-cudnn64-8-dll-not-found
3. `Numpy is not available` (Fix: `pip install numpy<2`) https://stackoverflow.com/questions/71689095/how-to-solve-the-pytorch-runtimeerror-numpy-is-not-available-without-upgrading
4. chromadb fails to install - https://github.com/chroma-core/chroma/issues/189



Inspo
    https://github.com/hoarder-app/hoarder
    kokoro voice mixing - https://github.com/NeuralFalconYT/Kokoro-82M-WebUI

Testing
    https://tox.wiki/en/4.23.2/
    https://flake8.pycqa.org/en/latest/


Create a blog post
    tldwproject.com - Done
    Make a nicer homepage - https://vitepress.dev/guide/getting-started
    https://www.open-notebook.ai/features/podcast.html

HF Embedding API - https://huggingface.co/BAAI/bge-large-en-v1.5
    Customize to use that for HF Demo

Help Llama.cpp
    -The most important tests are those in tests/ggml-backend-ops.cpp but those are also already in a comparatively good state; if you can find cases that are not covered adding them would be useful though.
    -Any tests of components that deal with memory management
    -The llama.cpp HTTP server currently has Python tests that call the API - can be improved
    -llama-bench - add an option for determining the performance for each token position instead of the average of all tokens in the range.
    -better error messages for failure modes of scripts/compare-llama-bench.py.
    -Scripts for submitting common language model benchmarks to the server also allow for a comparison with other projects.
        When academics introduce new techniques or quantization formats they usually measure the impact on quality using benchmarks like MMLU.
        llama.cpp currently has no standardized procedure for evaluating models on these benchmarks so a comparison of llama.cpp quality with academia is difficult.
        A standardized way for running benchmarks would also make it possible to compare the impact of things like different prompting techniques and quantization, especially between models where perplexity comparisons don't make sense.
        So you could for example better determine whether a heavily quantized 123b model or a less quantized 70b model performs better.
    -a simple Python script that does a performance benchmark for batched inference on the server
    -Better methods for evaluating samplers
        -A script that can be used to evaluate the performance of the sampler on a given dataset
        -Blind testing ala ChatArena where users are given two completions for a task and asked to choose the better one, created from two different random presets.
        -Use a model to generate multiple completions for a task with different seeds. Then rate/judge teh completions in terms of quality & diversity
        -Establish which sampling methods incur the smallest quality loss for a given target diversity.


LLMs.txt
    https://directory.llmstxt.cloud/


Logits
    https://github.com/NVIDIA/logits-processor-zoo



Import HTML Files
    
Manga
https://github.com/ragavsachdeva/magi
Emails:
https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/email.py
ODT, Docx, csv
    pypandoc
Evernote:
    https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML
Excel:
    https://ajac-zero.com/posts/how-to-create-accurate-fast-rag-with-excel-files/
    https://github.com/ajac-zero/excel-rag-example
    https://github.com/sivadhulipala1999/SheetSimplify_with_RAG
    https://openpyxl.readthedocs.io/en/stable/index.html
CSV
    CSV -> HTML https://blog.finxter.com/5-best-ways-to-convert-csv-to-html-in-python/
    https://github.com/jazzband/tablib
CHM:
    https://github.com/dottedmag/archmage


SQLite studio
    https://github.com/pawelsalawa/sqlitestudio


URL Shortener
    https://realpython.com/fastapi-python-web-apis/
    https://realpython.com/build-a-python-url-shortener-with-fastapi/



Flashcard prompt
Fantastic-End9783@reddit
```
I want you to create comprehensive and easy to consume Anki flashcards. They must thoroughly cover REPLACEWITHLECTURE
Do not do this until you I ask you after you  output the following request .
Thoroughly go through the provided sources and divide them into parts. This is so you can compressively cover the lecture and not skip information. The parts do not have to have the exact same split of questions made for them.
Please list the overall topic of the part and then their sub-topics.
Here is an example:
Part 1: (Insert title here)
- (Topic 1)
- (Topic 2)
```
```
Create comprehensive and easy to consume Anki flashcards for Part 1 of REPLACEWITHLECTURE. If there is embedded questions or topics per question, split them up into their own individual questions. Start each question on a new line. 
Use the following as a scaffold for the expected output:
What noise does a cow make? ; Moo
What is the capital of Australia? ; Canberra
```


#1: `Do not start writing yet, First explain everything I wanted you to do in this Prompt in Detail?`
#2: `I need this written in human tone. Humans have fun when they write — robots don’t. Chat GPT, engagement is the highest priority. Be conversational, empathetic, and occasionally humorous. Use idioms, metaphors, anecdotes, and natural dialogue.`
#3: `Before you answer, I want you to ask me all the missing information that I didn’t provide but it will help you better understand my needs and the specific output I want.`
#4: `Criticize yourself/your answer.`
#5: `Why did you write what you wrote? Give me all the reasons, Plus I want a full detailed analysis and breakdown of everything in a tabular format. Also add How could this be made better. Use my prompt as reference to further clarify the ‘Why’.`
#6: `Before you answer this, Highlight 20 potential risks or blind spots I might not have considered based on my request.`
#7: `Identify areas in this article where examples, analogies, or case studies would improve understanding.`


Jailbreaks
    https://arxiv.org/abs/2311.16119
    https://www.lakera.ai/blog/visual-prompt-injections
    https://github.com/0xk1h0/ChatGPT_DAN
    https://arxiv.org/pdf/2409.17458
    https://github.com/verazuo/jailbreak_llms
    https://arxiv.org/pdf/2308.03825
    https://arxiv.org/pdf/2409.00137
    https://arxiv.org/pdf/2409.11445







### Model Routing
https://github.com/pepijndevos/llama_multiserver/tree/main


I trained one router LoRA capable of routing into a list of LoRAs I had previously trained on my domain data. One of the LoRAs I used was created by another user on Hugging Face but performed exceptionally well for my use case.

Given a question, the classifier LoRA first classifies the question into one of my domains. Then another LoRA, based on the domain classification and the list of available LoRA models (fetched from my database), selects the most appropriate one for the answer. The message, along with any relevant context, is then sent to the selected LoRA. The final answers are aggregated and presented to the user.

I have implemented multiple approaches. In one, the domain selector and router LoRA were trained and integrated into a single unitary adapter instead of relying on a two-step process.

Additionally, in another experiment, I used a COT LoRA. For each step, the classifier identified the domain and selected the adapter to solve each of the given steps independetly. The process involved generating content from each of the "agents," aggregating the content and feeding it to the next step, aggregating the final answer, and presenting it to the user.

I usually trained these adapters with 500–5000 samples. Most samples were generated using GPT-4o with few-shot prompts and domain-specific information to build the synthetic dataset. In my case this approach was effective in producing a tailored synthetic dataset.

Weak-Abbreviations15
You definitely can.
My approach is this:

    High Context length 2048-4096

    I get my team to produce some (50 max) high quality Chains on the domains we work on.

    Use GPT to expand and optimize these chains. The output is the OptimalChain.

    Use the OptimalChains and FewShot to generate the remaining chains in the domains we need, based on the given structure.

    Structurizing chains in json Like format does seem to help but i dont have any proof of this claim.

    Experiment with Lora + Unsloth tuning with different parameters. You can use LLamafactory to do this.

    This seemed to be enough for most use cases. Im experimenting now with adding a RLHF/OPRO post finetuning, but i've yet to try it. (Ive built the dataset by synthetically messing up the answers to create the prefered and non prefered answer for the process. )

EDIT: 8. Anecdotal advice: unquantized Lora finetuning seems to train faster, and loss also converges faster. Thus is my preferred approach.

Ive had my fair share of experience in finetuning. My MSc Thesis was based on adding new copied layers to a LLM, and then finetune the layers specifically with a new language dataset which the LLM knew very little about, and essentially aggressively train those new layers without changing any of the other LLM structures, the result was that the LLM gained new linguistical abilities without downgrading its base skills in english. So i believe that for most usecases finetuning is good enough.


In my paper i used instruct models. Later i discovered that non instruction tuned models seem to perform much better for finetuning. They converge easier and in my experience seem less prone to forgetting. I used QLora but only bcs the models i used were bigger than my VRAM. I also wasnt aware of the faster training in Lora vs Qlora. Sort of my insights were developed once i was deep in the paper. I started the paper with llama 7b and ended up using the llama 3.1 8b until i finalized the project. Using LlamaPro i added a number of layers whoch increased its size to smthing between 9.5-10.2 billion parameters i cant recall exactly, which made it larger than what fit my GPU. I used fairly large r and alpha for lora on the fiest pass, and then lowered them as the tuning progressed. I wanted to use a very large dataset, but ended up using a much smaller one due to time and compute constraints. Everything was done locally hence i had to fidget around and over optimize. Probably with more compute itd be better managed. The result was pretty good considering the base model produced only jibberish. Some issues when speaking some very peculiar features of my native language. I used Bleu4 and a couple of the rouge metrics to rest the quality of the outputs. Differences in performance was tested for signficance.
