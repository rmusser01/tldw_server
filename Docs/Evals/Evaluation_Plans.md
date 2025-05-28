# Evaluation Plans

----------------------------------------------------------------------------------------------------------------
## Table of Contents
- [Introduction](#introduction)
- [Evaluation Plan](#evaluation-plan)
- [101/General](#general)
- [Model Evaluation](#model-evaluation)
- [Search Evaluations](#search-eval)
- [Abstractive Summarization Evaluations](#abstract_summ_evals)
- [RAG Evaluation](#rag-eval)
- [Embeddings Retrieval Evaluation](#embeddings-retrieval-eval)
- [VLM Evaluations](#vlm-evals)
----------------------------------------------------------------------------------------------------------------




https://www.leahtharin.com/p/107-andres-glusman-why-your-ab-tests
https://github.com/collect-intel/llm-judge-bias-suite
https://www.cip.org/blog/llm-judges-are-unreliable
https://github.com/JackHopkins/factorio-learning-environment
https://github.com/lechmazur
https://arxiv.org/abs/2503.05244
https://liveideabench.com/
https://b-score.github.io/
https://arxiv.org/abs/2504.12312
https://github.com/JohannesGaessler/elo_hellm
https://arxiv.org/abs/2407.09141
https://www.lennysnewsletter.com/p/beyond-vibe-checks-a-pms-complete
https://arxiv.org/abs/2504.03352
https://arxiv.org/html/2411.12990v1
https://github.com/knoveleng/rainbowplus
https://github.com/jd-3d/SOLOBench
https://github.com/salman-lui/x-teaming
https://arxiv.org/abs/2504.15909
https://huggingface.co/spaces/lmgame/game_arena_bench
https://www.vgbench.com/
https://huggingface.co/NousResearch/Minos-v1
https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/classifier_factory/intent_classification.ipynb
https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/classifier_factory/moderation_classifier.ipynb
https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/classifier_factory/product_classification.ipynb
https://docs.mistral.ai/capabilities/finetuning/classifier_factory/




System Eval
    https://github.com/plasma-umass/scalene


    Answer Correctness: Checks the accuracy of the generated llm response compared to the ground truth.

    Context Sufficiency: Checks if the context contains enough information to answer the user's query

    Context Precision: Evaluates whether all relevant items present in the contexts are ranked higher or not.

    Context Recall: Measures the extent to which the retrieved context aligns with the expected response.

    Answer/Response Relevancy: Measures how pertinent the generated response is to the given prompt.

LightEval + Argilla + distilabel
- Open source, (will) support litellm and can use distilabel for synth data gen
Agents
    https://arxiv.org/html/2502.15840v1

ETL
    https://github.com/opendatalab/OmniDocBench
    https://huggingface.co/MMDocIR

OCR/VLM
    https://github.com/opendatalab/OHR-Bench
    https://zerobench.github.io/

web walking
    https://arxiv.org/pdf/2501.07572v2


TICK
    https://arxiv.org/abs/2410.03608

RAG eval dataset creation
    https://medium.com/towards-data-science/how-to-create-a-rag-evaluation-dataset-from-documents-140daa3cbe71
    https://arxiv.org/pdf/2409.02098v1



RAG EVal
    https://www.byhand.ai/p/beginners-guide-to-rag-evaluation
    https://pub.towardsai.net/around-the-rag-in-80-questions-part-ii-4df03c6dba86
    https://arxiv.org/pdf/2409.12941
    https://arxiv.org/pdf/2410.23000
    https://arxiv.org/pdf/2502.17163
    https://arxiv.org/pdf/2502.18817
        https://github.com/OpenBMB/ConsJudge
    https://arxiv.org/pdf/2502.11444
    https://arxiv.org/pdf/2502.01534
    https://github.com/David-Li0406/Preference-Leakage
    https://arxiv.org/pdf/2502.07445
    https://arxiv.org/pdf/2502.12501


https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html
https://github.com/cpldcpu/MisguidedAttention
https://github.com/lechmazur/generalization
https://docs.ragas.io/en/stable/getstarted/rag_eval/
https://github.com/plurch/ir_evaluation
https://github.com/OpenBMB/UltraEval-Audio
https://huggingface.co/papers/2502.06329
https://arxiv.org/abs/2502.05167
https://alibaba-nlp.github.io/WebWalker/
https://arxiv.org/abs/2501.03491
https://machinelearningmastery.com/rag-hallucination-detection-techniques
https://arxiv.org/abs/2501.09775/
https://github.com/CONE-MT/BenchMAX
https://physico-benchmark.github.io/
https://huggingface.co/papers/2412.06745
https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF
https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO
https://huggingface.co/papers/2502.00698
https://huggingface.co/papers/2502.01534
https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data
https://huggingface.co/mradermacher/Llama-3.1-Tulu-3-8B-SFT-no-safety-data-GGUF
https://arxiv.org/html/2412.02611v1
https://arxiv.org/abs/2412.05579
https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d
https://huggingface.co/blog/synthetic-data-generator
https://towardsdatascience.com/stop-guessing-and-measure-your-rag-system-to-drive-real-improvements-bfc03f29ede3
https://huggingface.co/SultanR/SmolTulu-1.7b-Instruct
https://huggingface.co/DevQuasar/allenai.Llama-3.1-Tulu-3-8B-SFT-no-safety-data-GGUF
https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024
https://arxiv.org/html/2412.09569v1
https://huggingface.co/tiiuae
https://github.com/Imbernoulli/SURGE
https://dxzxy12138.github.io/PhysReason
https://paperswithcode.com/paper/nolima-long-context-evaluation-beyond-literal
https://arxiv.org/abs/2502.09083
https://ai.gopubby.com/rag-evaluation-a-visual-approach-c9af26006ef5
https://huggingface.co/datasets/bytedance-research/ToolHop
https://github.com/naver/bergen?tab=readme-ov-file
https://arxiv.org/abs/2412.13147
https://arxiv.org/abs/2412.13018
https://huggingface.co/blog/big-bench-audio-release
https://github.com/chigkim/openai-api-gpqa
https://github.com/chigkim/Ollama-MMLU-Pro
https://huggingface.co/ymcki/Llama-3_1-Nemotron-51B-Instruct-GGUF
https://pub.towardsai.net/streamline-your-llm-evaluation-a-step-by-step-guide-to-rag-metrics-with-streamlit-38ed9efbdc9a
https://huggingface.co/QuantFactory/granite-3.1-8b-instruct-GGUF
https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024
https://arxiv.org/abs/2412.17758
https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard
https://www.atla-ai.com/post/evaluating-the-evaluator
https://hamel.dev/blog/posts/llm-judge/
https://github.com/scrapinghub/article-extraction-benchmark
https://github.com/Zhe-Young/SelfCorrectDecompose
https://eugeneyan.com/writing/evals/

Have LLMs play Social deception games
Different results depending on batch size during evaluations - https://x.com/bnjmn_marie/status/1846834917608407199

Benchmarking with distilabel
    https://distilabel.argilla.io/latest/sections/pipeline_samples/examples/benchmarking_with_distilabel/

General Research
    Greedy Sampling 
        https://arxiv.org/abs/2407.10457
            * `Our study addresses this issue by exploring key questions about the performance differences between greedy decoding and sampling, identifying benchmarks' consistency regarding non-determinism, and examining unique model behaviors. Through extensive experiments, we observe that greedy decoding generally outperforms sampling methods for most evaluated tasks. We also observe consistent performance across different LLM sizes and alignment methods, noting that alignment can reduce sampling variance. Moreover, our best-of-N sampling approach demonstrates that smaller LLMs can match or surpass larger models such as GPT-4-Turbo, highlighting the untapped potential of smaller LLMs. `
    Stats
        https://arxiv.org/pdf/2410.01392
        https://arxiv.org/abs/2411.00640
    
Chat arena
    Building one
        https://github.com/lm-sys/FastChat
        https://github.com/Teachings/llm_tools_benchmark
        https://github.com/Nutlope/codearena
    Potential issues with creating a chatarena system
        https://arxiv.org/abs/2412.04363
LLM-as-judge
        https://arxiv.org/html/2412.14140v1
    Basics
        https://huggingface.co/learn/cookbook/en/llm_judge
    Evaluating LLMs as Judges
        https://huggingface.co/papers/2306.05685
        https://llm-as-a-judge.github.io/
        https://arxiv.org/abs/2411.16646
        Google SAFE
            https://arxiv.org/abs/2403.18802
            https://github.com/google-deepmind/long-form-factuality
    Ranking of
        https://huggingface.co/spaces/AtlaAI/judge-arena
    Tools
        https://github.com/open-compass/CompassJudger

Quant Eval
    https://arxiv.org/abs/2411.02355

Summarization
    ClinicSum (Finetuning for Summarization)
        https://arxiv.org/abs/2412.04254

Creating Datasets
    
    
    https://github.com/argilla-io/argilla
        https://www.youtube.com/watch?v=ZsCqrAhzkFU
        https://www.youtube.com/watch?v=jWrtgf2w4VU
        https://www.youtube.com/watch?v=ZsCqrAhzkFU
        https://github.com/argilla-io/argilla-cookbook/blob/main/domain-eval/README.md

Finetuning
    https://huggingface.co/learn/cookbook/enterprise_cookbook_argilla
    https://aclanthology.org/2024.cl-3.1/
    https://scale.com/guides/data-labeling-annotation-guide
    https://aclanthology.org/2024.naacl-long.126/
    https://distilabel.argilla.io/latest/
    https://distilabel.argilla.io/latest/sections/pipeline_samples/papers/ultrafeedback/


----------------------------------------------------------------------------------------------------------------
### <a name="introduction"></a> Introduction


- **101**
    https://hamel.dev/blog/posts/evals/

Links:
    https://www.juriopitz.com/2024/10/17/evaluation-pitfalls-metric-overview-tips.html

    
----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="evaluation-plan"></a> Evaluation Plan
- Need to review this: https://github.com/tianyi-lab/BenTo (Why isn't it being used???)
- Model Evaluation
- Search Evaluation (Assuming test data set)
  - Retrieval Evaluation First
    - Identify keyword search success rate
    - Identify Semantic search success rate
    - Identify Hybrid search success rate
- RAG evaluation
  - Generation Eval Second
    - Create synthetic eval data by taking a document and generating questions about the document using an LLM.
  - User queries + ranking = Eval data
  - Integrate 'real' eval data into synthetic eval data
- Embeddings Retrieval Evaluation
- LLM Evaluations
- Evaluation Metrics
- Things to look out for
  - https://arxiv.org/abs/2411.03923
- **Reproducability**
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/troubleshooting/troubleshooting-reproducibility.md
- **Designing Evaluations**
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/model-as-a-judge/designing-your-evaluation-prompt.md
  - https://eugeneyan.com/assets/llm-eval-tree.jpg
- Human Evaluation
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/human-evaluation/using-human-annotators.md
----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="model-evaluation"></a> Model Evaluation
- https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
- 
- **101**
    - https://github.com/huggingface/evaluation-guidebook
- **Metrics**
    1. Answer Relevancy
         * Does the LLM give a relevant answer to the question?
         * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/answer_relevancy
    2. Correctness
         - Is the LLM output correct regarding a 'ground truth'
    3. Confabulation-Rate
         - How often does the LLM make up information?
   4. Contextual Relevancy
         - How relevant is the returned content to the context of the question?
         * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/contextual_relevancy
   5. Bias
         - DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/bias
   6. Task-Specific
  7. Conversation Intent
     * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/conversation_relevancy

- **Metrics should be:**
    1. Quantifiable
    2. Reproducible
    3. Sensitive
    4. Specific
    5. Interpretable

- **Evaluation Methodologies**
    - G-Eval
      - https://ai.plainenglish.io/how-to-evaluate-fluency-in-llms-and-why-g-eval-doesnt-work-3635298e9a43?gi=463011e4689d
    - QAG - QA Generation
          - https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task

- **Frameworks**
    - OpenCompass
        - https://github.com/open-compass/opencompass
    - DeepEval
    - lm-eval-harness
        - https://github.com/EleutherAI/lm-evaluation-harness
    - lighteval
        - https://github.com/huggingface/lighteval
    - NexusBench
        - https://github.com/nexusflowai/NexusBench
    - Athina.ai
        - https://docs.athina.ai/evals/preset-evals/overview
    - SimpleEval
        - https://github.com/openai/simple-evals/tree/main
    - EvalAI
    - FlashRAG
        - https://github.com/RUC-NLPIR/FlashRAG
    - Olmes
        - https://github.com/allenai/olmes
- **Books**
    - https://novelqa.github.io/
- **CheeseBench**
    - https://gist.github.com/av/db14a1f040f46dfb75e48451f4f14847
- **Citations**
    - L-CiteEval
        - https://huggingface.co/papers/2410.02115
        - https://github.com/ZetangForward/L-CITEEVAL
- **Coding Ability**
    - Aider
        - https://github.com/Aider-AI/aider/tree/main/benchmark
    - CodeMMLU
        - https://arxiv.org/abs/2410.01999
        - https://github.com/FSoft-AI4Code/CodeMMLU
    - LiveBench
        - https://github.com/LiveBench/LiveBench
    - StackUnseen
        - https://prollm.toqan.ai/leaderboard/stack-unseen
    - https://huggingface.co/papers/2412.05210
- **Cognitive Biases**
    - CBEval: https://arxiv.org/abs/2412.03605
- **Confabulation-Rate**
    - https://arxiv.org/abs/2409.11353
    - https://github.com/sylinrl/TruthfulQA
- **Context**
    - RULER
    - InfiniteBench
        - https://arxiv.org/abs/2402.13718
        - https://github.com/OpenBMB/InfiniteBench
        - `Welcome to InfiniteBench, a cutting-edge benchmark tailored for evaluating the capabilities of language models to process, understand, and reason over super long contexts (100k+ tokens). Long contexts are crucial for enhancing applications with LLMs and achieving high-level interaction. InfiniteBench is designed to push the boundaries of language models by testing them against a context length of 100k+, which is 10 times longer than traditional datasets.`
    - Babilong
        * `BABILong is a novel generative benchmark for evaluating the performance of NLP models in processing arbitrarily long documents with distributed facts.`
        * `BABILong consists of 20 tasks designed for evaluation of basic aspects of reasoning. The bAbI tasks are generated by simulating a set of characters and objects engaged in various movements and interactions with each other in multiple locations. Each interaction is represented by a fact, e.g. ”Mary travelled to the office”, and the task is to answer a question using the facts from the current simulation, for instance, ”Where is Mary?”. The bAbI tasks vary based on the number of facts, question complexity and the aspects of reasoning.`
        * https://huggingface.co/datasets/RMT-team/babilong
        * https://github.com/booydar/babilong
    - LongICLBench
        * `We created LongICLBench to conduct comprehensive evaluations of Large Language Models (LLMs) on extreme-label classification challenges with in-context learning. We compiled six datasets that encompass a broad spectrum of labels, ranging from 28 to 174 categories, and varied the lengths of input (from few-shot demonstrations) between 2K and 50K tokens to ensure thorough testing`
        * https://github.com/TIGER-AI-Lab/LongICLBench
        * https://arxiv.org/abs/2404.02060 - `Large Language Models (LLMs) have made significant strides in handling long sequences. Some models like Gemini could even to be capable of dealing with millions of tokens. However, their performance evaluation has largely been confined to metrics like perplexity and synthetic tasks, which may not fully capture their true abilities in more challenging, real-world scenarios. We introduce a benchmark (LongICLBench) for long in-context learning in extreme-label classification using six datasets with 28 to 174 classes and input lengths from 2K to 50K tokens. Our benchmark requires LLMs to comprehend the entire input to recognize the massive label spaces to make correct predictions. We evaluate on 15 long-context LLMs and find that they perform well on less challenging classification tasks with smaller label space and shorter demonstrations. However, they struggle with more challenging task like Discovery with 174 labels, suggesting a gap in their ability to process long, context-rich sequences. Further analysis reveals a bias towards labels presented later in the sequence and a need for improved reasoning over multiple pieces of information. Our study reveals that long context understanding and reasoning is still a challenging task for the existing LLMs. We believe LongICLBench could serve as a more realistic evaluation for the future long-context LLMs.`
    - Snorkel Working Memory Test
        * https://github.com/snorkel-ai/long-context-eval
        * https://arxiv.org/pdf/2407.03651
        * `This repository provides a Snorkel Working Memory Test (SWiM) to evaluate the long context capabilities of large language models (LLMs) on your own data and tasks. This is an improvement to the "needle in a haystack" (NIAH) test, where the haystack is your own set of documents, and the needles are one or more answer (complete) documents based on which the question is posed.`
    - HelloBench
        - https://github.com/Quehry/HelloBench
        - https://arxiv.org/abs/2409.16191
    - https://longbench2.github.io/
    - https://github.com/jonathan-roberts1/needle-threading/
    - Michelangelo
        - https://arxiv.org/abs/2409.12640
- **Creative Writing**
    - EQ Bench
        - https://eqbench.com/creative_writing.html
- **Culture**
    - https://arxiv.org/html/2305.14328v2
    - https://arxiv.org/abs/2412.03304
    - https://huggingface.co/datasets/CohereForAI/Global-MMLU
    - https://arxiv.org/abs/2411.06032
    - https://arxiv.org/abs/2410.02677
    - https://mbzuai-oryx.github.io/ALM-Bench/
    - https://arxiv.org/abs/2411.19799
    - User-Centric Evaluation of LLMs
        - https://github.com/Alice1998/URS
    - https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks
- **Deceptiveness/Gullibility**
    - https://github.com/lechmazur/deception
- **Game Playing**
    - https://github.com/balrog-ai/BALROG
- **Math Eval**
    - https://arxiv.org/abs/2411.04872
    - GSM8K
- **Prompt Formatting**
    - How do LLMs handle different formats:
        - https://arxiv.org/abs/2411.10541
            - `Our study reveals that the way prompts are formatted significantly impacts GPT-based models’ performance, with no single format excelling universally. This finding questions current evaluation methods that often ignore prompt structure, potentially misjudging a model’s true abilities. We advocate for diverse prompt formats in future LLM testing to accurately gauge and enhance their performance.`
- **Positional Bias**
    - https://arxiv.org/abs/2410.14641
    - https://github.com/Rachum-thu/LongPiBench
- **'Reasoning'**
    - AGI Eval
        - https://arxiv.org/abs/2304.06364
    - BoolQ
        - https://arxiv.org/abs/1905.10044
    - Counterfactual Reasoning Assessment (CRASS)
        - https://arxiv.org/abs/2112.11941
    - Discrete Reasoning Over Paragraphs (DROP)
        - https://arxiv.org/abs/1903.00161
    - MMLU-Pro
        - https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        - https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main
    - Large-scale ReAding Comprehension Dataset From Examinations (RACE)
        - https://arxiv.org/abs/1704.04683
    - Physical Interaction: Question Answering (PIQA)
        - https://github.com/ybisk/ybisk.github.io/tree/master/piqa/data
    - Question Answering in Context (QuAC)
        - https://quac.ai/
- **Role Play**
    - **Conversation Relevancy**
        - DeepEval
            * `assesses whether your LLM chatbot is able to generate relevant responses throughout a conversation. It is calculated by looping through each turn individually and adopts a sliding window approach to take the last min(0, current turn number — window size) turns into account to determine whether it is relevant or not. The final conversation relevancy metric score is simply the number of turn responses that is relevant divided by the total number of turns in a conversational test case.` 
            * https://docs.confident-ai.com/docs/metrics-conversation-relevancy
    - **Debating**
        - https://huggingface.co/blog/debate
    - Discussion from different PoV Facilitation
          - https://github.com/Neph0s/awesome-llm-role-playing-with-persona?tab=readme-ov-file
          - https://github.com/lawraa/LLM-Discussion
          - https://github.com/InteractiveNLP-Team/RoleLLM-public
    - Role Adherence
        - StickToYourRole
              - https://huggingface.co/datasets/flowers-team/StickToYourRole
              - https://huggingface.co/datasets/flowers-team/StickToYourRole
              - https://arxiv.org/abs/2402.14846
              - https://flowers-team-sticktoyourroleleaderboard.hf.space/about
        - PingPong Bench
            - https://github.com/IlyaGusev/ping_pong_bench
            - https://ilyagusev.github.io/ping_pong_bench/
            - https://arxiv.org/pdf/2409.06820
        - DeepEval
            - https://docs.confident-ai.com/docs/metrics-role-adherence
            - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/role_adherence
        - General Research / Unsorted
            - https://arxiv.org/html/2406.00627v1
            - https://mp.weixin.qq.com/s/H2KNDGRNHktHiQc3sayFsA
            - https://mp.weixin.qq.com/s/2lbCMo64-nU5yRz1cLQxYA
            - https://mp.weixin.qq.com/s/E5qp5YPYPVaLM07OumDTRw 
            - https://mp.weixin.qq.com/s/yoM-srJYGGfyd1VXirg_Hg
        - RP-Bench
            - https://boson.ai/rpbench-blog/
        - PersonaGym
            - https://arxiv.org/abs/2407.18416
        - Collections of research
            - https://github.com/MiuLab/PersonaLLM-Survey
        - Notes
            - https://ianbicking.org/blog/2024/04/roleplaying-by-llm
    - **Knowledge Retention**
         - `chatbot is able to retain information presented to it throughout a conversation. It is calculated by first extracting a list of knowledges presented to it up to the certain turn in a conversation, and determining whether the LLM is asking for information that is already present in the turn response. The knowledge retention score is simply the number of turns without knowledge attritions divided by the total number of turns.`
         - https://docs.confident-ai.com/docs/metrics-knowledge-retention
    - **Conversation Completeness**
        - `chatbot is able to fulfill user requests throughout a conversation. It is useful because conversation completeness can be used as a proxy to measure user satisfaction and chatbot effectiveness. It is calculated by first using an LLM to extract a list of high level user intentions found in the conversation turns, before using the same LLM to determine whether each intention was met and/or satisfied throughout the conversation.`
        - https://docs.confident-ai.com/docs/metrics-conversation-completeness
- **Specific-Tasks**
    - CRMArena
        - https://arxiv.org/abs/2411.02305
- **Summarization**
    - Why use LLMs for summarization vs other approaches
        - https://www.mdpi.com/2673-4591/59/1/194
        - https://www.sciencedirect.com/science/article/pii/S2949719124000189
        - https://arxiv.org/pdf/2403.02901
    - Measuring LLM Summarization
        - https://stackoverflow.com/questions/9879276/how-do-i-evaluate-a-text-summarization-tool
    - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/summarization
    - https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task
    - https://arxiv.org/abs/2009.01325
    - https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization
    - https://towardsdatascience.com/explaining-llms-for-rag-and-summarization-067e486020b4
    - https://arxiv.org/abs/2407.01370v1
    - https://arxiv.org/html/2403.19889v1
    - https://github.com/salesforce/summary-of-a-haystack
    - https://docs.cohere.com/page/summarization-evals
    - https://docs.cohere.com/page/grounded-summarization
    - Books
        - https://arxiv.org/abs/2404.01261
        - https://arxiv.org/abs/2310.00785
        - https://arxiv.org/abs/2407.06501
        - https://arxiv.org/abs/2205.09641
        - https://github.com/DISL-Lab/FineSurE-ACL24
        - https://arxiv.org/abs/2404.01261
        - https://openreview.net/forum?id=7Ttk3RzDeu
      - News Sources
          - https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News
    - ACI-Bench
        - https://arxiv.org/abs/2306.02022
    - DeepEval
        - https://docs.confident-ai.com/docs/metrics-summarization
    - MS Marco
        - https://arxiv.org/abs/1611.09268
    - Query-based Multi-domain Meeting Summarization (QMSum)
        - https://arxiv.org/abs/2104.05938
        - https://github.com/Yale-LILY/QMSum
    - RAGAS
        - https://arxiv.org/abs/2309.15217
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/summarization_score/
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/agents/#topic_adherence
- **Temporal Bias**
    - https://arxiv.org/abs/2412.13377
- **Text Comprehension**
- **QA (General)**
    * https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question
    * Multi-Genre Natural Language Inference (MultiNLI)
        * https://arxiv.org/abs/1704.05426
    * TriviaQA
        * https://arxiv.org/abs/1705.03551
        * https://github.com/mandarjoshi90/triviaqa
- **Tool Calling**
    - DeepEval
        - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/tool_correctness
- **Toxicity Testing**
    - DeepEval
        - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/toxicity
    - HHH
        - https://arxiv.org/abs/2112.00861
        - https://github.com/anthropics/hh-rlhf
    - ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection
        - https://github.com/microsoft/TOXIGEN/tree/main
    - TruthfulQA
        - https://arxiv.org/abs/2109.07958v2
        - https://github.com/sylinrl/TruthfulQA
    - Other
        - https://arxiv.org/abs/2410.01524
- **Vibes**
    - AidanBench
        - https://github.com/aidanmclaughlin/AidanBench
- **Links**
    - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/tips-and-tricks.md


----------------------------------------------------------------------------------------------------------------
 


----------------------------------------------------------------------------------------------------------------
### <a name="general"></a>101/General
- **Links**
    - https://eugeneyan.com/writing/evals/
- **Guidelines for Human Annotators when creating a dataset**
    - **Areas of Measure**:
        * **Accuracy:** Is the generated text factually correct and aligned with known information? This is closely tied to factual consistency.
        * **Relevance:** Is the output appropriate and directly applicable to the task and input?
        * **Fluency:** Is the text grammatically correct and readable? With modern LLMs, this is less of an issue than it used to be.
        * **Transparency:** Does the model communicate its thought process and reasoning? Techniques like chain-of-thought help with this.
        * **Safety:** Are there potential harms or unintended consequences from the generated text? This includes toxicity, bias, and misinformation.
        * **Human alignment:** To what extent does the model’s output align with human values, preferences, and expectations?
- **You have a dataset, now what?**
    * **Increase precision:** Select instances that the model predicts as positive with high probability and annotate them to identify false positives
    * **Increase recall:** Select instances that the model predicts have low probability and check for false negatives
    * **Increase confidence:** Select instances where the model is unsure (e.g., probability between 0.4 to 0.6) and collect human labels for finetuning


----------------------------------------------------------------------------------------------------------------
### <a name="abstract-summ-evals"></a> Abstractive Summarization Evaluations
- **101**
    - Basic Measures:
    - `Fluency`: Are sentences in the summary well-formed and easy to read? We want to avoid grammatical errors, random capitalization, etc. 
    - `Coherence`: Does the summary as a whole make sense? It should be well-structured and logically organized, and not just a jumble of information. 
    - `Consistency`: Does the summary accurately reflect the content of the source document? We want to ensure there’s no new or contradictory information added. 
    - `Relevance`: Does the summary focus on the most important aspects of the source document? It should include key points and exclude less relevant details
- **Benchmarks**
  - https://github.com/r-three/fib
  - https://huggingface.co/datasets/r-three/fib
  - https://github.com/kukrishna/usb




----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="search-eval"></a> Search Evaluations

https://arxiv.org/abs/2304.01982


- **101**
  - F
- **Basic Search Eval via swyx**
  1. Take your existing text chunks
  2. Generate questions that could be answered by each chunk
  3. Store {question, chunk_id} pairs
  4. Test: Does your retrieval system find the source chunk?

Retrieval Granularity
    https://chentong0.github.io/factoid-wiki/
    https://github.com/chentong0/factoid-wiki
----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="rag-eval"></a> RAG Evaluation
https://blog.streamlit.io/ai21_grounded_multi_doc_q-a/
https://archive.is/OtPVh
https://towardsdatascience.com/how-to-create-a-rag-evaluation-dataset-from-documents-140daa3cbe71
https://github.com/jonathan-roberts1/needle-threading/
https://huggingface.co/datasets/jonathan-roberts1/needle-threading
https://arxiv.org/abs/2411.03538
https://arxiv.org/abs/2411.19710
https://aws.amazon.com/blogs/aws/new-rag-evaluation-and-llm-as-a-judge-capabilities-in-amazon-bedrock/
https://archive.is/MZsB9
https://arxiv.org/abs/2411.00136
https://github.com/opendatalab/OHR-Bench
https://towardsdatascience.com/from-retrieval-to-intelligence-exploring-rag-agent-rag-and-evaluation-with-trulens-3c518af836ce
https://arxiv.org/pdf/2411.09213
https://huggingface.co/learn/cookbook/en/rag_evaluation




- **101**
- **RAG Eval Plan:**
    - The generic idea however: you take a (full, unchunked) document and ask an LLM to generate a question with that document as well as give the factual answer to it. Enforce via prompts to make it use the document only and make it as hard as you want (eg. maybe sometimes you want it to consider 2 documents and make a question that uses bits of both). This gives you a ground truth dataset.
        * You then kick off your RAG pipeline on your documents. They will be chunked, indexed and stored. Then you fire all the questions of your ground truth set at your RAG pipeline and check 1. If it found chunks of the correct document and 2. Ask an LLM various evaluation questions about the generated answer vs. the ground truth answer ( like: how related are they, is there content in the answer that is not in the doc chunks, etc).
        * This gives you a good idea how well your retrieval (and with that, indexing) works, and how well your full pipeline works. As a bonus you could also keep track of which chunk(s) the ground truth answer was based on and use that for retrieval evaluation too. 
    - **When to move on from Primarily Synthetic Data to Real User Data**
        - 80% recall on synthetic tests
        - Good understanding of failure cases
        - Clear chunking strategy
        - Growing set of real user examples
- **Metrics**
    - 3 General Categories
        1. Retrieval Metrics
            - Simple:
                * How good is the retrieval of the context from the Vector Database?
                * Is it relevant to the query?
                * How much noise (irrelevant information) is present?
            - Accuracy
                * `the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.`
            - Precision
                * `measures the proportion of retrieved documents that are relevant to the user query. It answers the question, “Of all the documents that were retrieved, how many were actually relevant?”` 
            - Precision@k
                * `Precision@k is a variation of precision that measures the proportion of relevant documents amongst the top ‘k’ retrieved results. It is particularly important because it focusses on the top results rather than all the retrieved documents. For RAG it is important because only the top results are most likely to be used for augmentation. For example, if our RAG system considers top 5 documents for augmentation, then Precision@5 becomes important.`
            - Recall
                - `measures the proportion of the relevant documents retrieved from all the relevant documents in the corpus. It answers the question, “Of all the relevant documents, how many were actually retrieved?”`
            - F1-Score
              - `F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both the quality and coverage of the retriever.`
              - `F1-score = 2 x (Precision x Recall) / (Precision + Recall)`
              - `The equation is such that F1-score penalizes either variable having a low score; a High F1 score is only possible when both recall and precision values are high. This means that the score cannot be positively skewed by a single variable`
              - `F1-score provides and single, balanced measure that can be used to easily compare different systems. However, it does not take ranking into account and gives equal weightage to precision and recall which might not always be ideal.`
            - 
            - 
            - Ranking Metrics(lol copilot)
                - MRR Mean Reciprocal Rank
                    - `It measures the reciprocal of the ranks of the first relevant document in the list of results. MRR is calculated over a set of queries.`
                    - `MRR = 1/N x [Summation i=1 to N (1/rank(i))]`
                      - `where N is the total number of queries and rank(i) is the rank of the first relevant document of the i-th query`
                - MAP Mean Average Precision
                    - `Mean Average Precision or MAP is a metric that combines precision and recall at different cut-off levels of ‘k’ i.e. the cut-off number for the top results. It calculates a measure called Average Precision and then averages it across all queries.`
                    - `Average Precision for a single query (i) = 1/R(i) x [Summation k=1 to n (Precision@k x relevance of k-th document)]`
                        - where R(i) is the number of relevant documents for query(i)
                        - Precision@k is the precision at cut-off ‘k’
                        - rel@k is a binary flag indicating the relevance of the document at rank ‘k’
                    - Mean Average Precision is the mean of the average precision (shown above) over all the ’N’ queries
                    - `MAP = 1/N x [Summation i=1 to N (Average Precision (i)]`
                - Normalized Discounted Cumulative Gain (nDCG)
                    - `nDCG evaluates the ranking quality by considering the position of relevant documents in the result list and assigning higher scores to relevant documents appearing earlier.`
                    - `It is particularly effective for scenarios where documents have varying degrees of relevance.`
                    - `To calculate discounted cumulative gain (DCG), each document in the retrieved list is assigned a relevance score, rel and a discount factor reduces the weight of documents as their rank position increases.`
                    - `DCG =[Summation i=1 to n ((2 ^ rel(i) — 1)/log(i+1))`
                      - Here rel(i) is the graded relevance of document at position i. IDCG is the ideal DCG which is the DCG for perfect ranking. nDCG is calculated as the ratio between actual discounted cumulative gain (DCG) and the ideal discounted cumulative gain (IDCG)
                    - `nDCG = DCG/IDCG`
                      - `nDCG is quite a complex metric to calculate. It requires documents to have a relevance score which may lead to subjectivity and the choice of the discount factor affects the values significantly, but it accounts for varying degrees of relevance in documents and gives more weightage to higher ranked items.`
                - MAP
                    - `Mean Average Precision (MAP) is a ranking metric that computes the mean precision across all relevant documents retrieved. It considers the rank of each relevant document in the ranked list.`
                    - `MAP = Σ(Precision_i * Rel_i) / N`
                    - `where Precision_i is the precision at the i-th relevant document, Rel_i is the relevance of the i-th document, and N is the total number of relevant documents.`
        2. Generation-Specific Metric
            - Simple:
               * How good is the generated response?
               * Is the response grounded in the provided context?
               * Is the response relevant to the query?
        3. RAG-specific Metric
    1. Answer Consistency
       * Whether there is information in the LLM answer that does not come from the context.
    2. Answer relevancy
    3. Answer Similarity Score
       * How well the reference answer matches the LLM answer.
    4. Retrieval Precision
       * Whether the context retrieved is relevant to answer the given question.
    5. Augmentation precision
       * Whether the relevant context is in the LLM answer.
    6. Augmentation Accuracy
       * Whether all the context is in the LLM answer.
    7. Contextual Recall
    8. Latency
       * How long it takes for the LLM to complete a request.
    9. Contains Text
         * Whether the LLM answer contains the specific text.
- **DataSets**
    - https://huggingface.co/datasets/enelpol/rag-mini-bioasq
    - https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia
    - RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems
        - https://arxiv.org/abs/2407.11005
        - https://huggingface.co/datasets/rungalileo/ragbench
- **Generating Synthetic Data**
    - https://www.turingpost.com/p/sytheticdata
    - https://arxiv.org/html/2404.07503v1
    - https://arxiv.org/pdf/2210.14348
    - https://arxiv.org/pdf/2401.02524
    - https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
    - https://arxiv.org/pdf/2402.10379
    - https://arxiv.org/pdf/2403.04190
    - https://arxiv.org/pdf/2406.20094
    - https://arxiv.org/pdf/2407.01490
    - https://www.turingpost.com/p/synthetic

- **RAG-Specific Tuning/Modfications**
    - **Pre-Training/Finetuning**
        - RankRAG
            - https://arxiv.org/abs/2407.02485v1
        - RAG-FiT
            - https://github.com/IntelLabs/RAG-FiT
        - RAG Foundry
            - https://arxiv.org/pdf/2408.02545
    - **Inference Time**
        * https://arxiv.org/pdf/2408.14906

- **RAG-Specific Models**
    - https://huggingface.co/lightblue/kurage-multilingual
    - https://arxiv.org/pdf/2407.14482

```
The fundamental idea of evaluating a retriever is to check how well the retrieved content matches the expected contents. For evaluating a RAG pipeline end to end, we need query and ground truth answer pairs. The ground truth answer must be grounded on some "ground" truth chunks. This is a search problem, it's easiest to start with tradiaional Information retrieval metrics.

You might already have access to such evaluation dataset depending on the nature of your application or you can synthetically build one. To build one you can retrieve random documents/chunks and ask an LLM to generate query-answer pairs - the underlying documents/chunks will act as your ground truth chunk.

Retriever Evaluation
We can evaluate a retriever using traditional ML metrics. We can also evaluate by using a powerful LLM (next section).

Below we are importing both traditional metrics and LLM as a judge metric from the scripts/retrieval_metrics.py file.

    Hit Rate: Measures the proportion of queries where the retriever successfully returns at least one relevant document.
    MRR (Mean Reciprocal Rank): Evaluates how quickly the retriever returns the first relevant document, based on the reciprocal of its rank.
    NDCG (Normalized Discounted Cumulative Gain): Assesses the quality of the ranked retrieval results, giving more importance to relevant documents appearing earlier.
    MAP (Mean Average Precision): Computes the mean precision across all relevant documents retrieved, considering the rank of each relevant document.
    Precision: Measures the ratio of relevant documents retrieved to the total documents retrieved by the retriever.
    Recall: Evaluates the ratio of relevant documents retrieved to the total relevant documents available for the query.
    F1 Score: The harmonic mean of precision and recall, providing a balance between both metrics to gauge retriever performance.
```

- **Evaluation Benchmarks**
    - KITE
        - https://github.com/D-Star-AI/KITE
    - https://arxiv.org/abs/2411.09213
    

Evaluating RAG Cohere
    https://docs.cohere.com/page/rag-evaluation-deep-dive#generation-evaluation

- **Frameworks**
    - RAGAS
        - https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
        - https://docs.ragas.io/en/latest/concepts/testset_generation.html
    - Tonic
        - https://github.com/TonicAI/tonic_validate
    - RAGEval
        - https://arxiv.org/pdf/2408.01262
        - https://github.com/OpenBMB/RAGEval
    - AutoRAG
        - https://github.com/Marker-Inc-Korea/AutoRAG
    - Promptflow
        - https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-qna-rag-metrics
    - HELMET: How to Evaluate Long-context Language Models Effectively and Thoroughly
        - https://github.com/princeton-nlp/helmet


- Papers
  - https://arxiv.org/abs/2309.01431
  - https://arxiv.org/abs/2411.03538
https://arxiv.org/abs/2411.11767



----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="embeddings-retrieval-eval"></a> Embeddings Retrieval Evaluation


Benchmarking
    https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark
    https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
    https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649


Databases
    https://www.timescale.com/blog/pgvector-vs-pinecone/
    https://www.timescale.com/blog/how-we-made-postgresql-as-fast-as-pinecone-for-vector-data/
    https://nextword.substack.com/p/vector-database-is-not-a-separate
    SQLite
        https://github.com/asg017/sqlite-lembed
        https://github.com/asg017/sqlite-vec
        https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite
        https://stephencollins.tech/posts/how-to-use-sqLite-to-store-and-query-vector-embeddings
        https://turso.tech/blog/sqlite-retrieval-augmented-generation-and-vector-search


Embedding Models
    https://emschwartz.me/binary-vector-embeddings-are-so-cool/
    https://arxiv.org/pdf/2409.10173
    https://huggingface.co/dunzhang/stella_en_1.5B_v5
    https://huggingface.co/dunzhang/stella_en_400M_v5


Finetuning embedding model
    https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/
    https://modal.com/blog/fine-tuning-embeddings
    https://www.reddit.com/r/LocalLLaMA/comments/1686ul6/some_lessons_learned_from_building_a_fine_tuned/
    https://huggingface.co/blog/train-sentence-transformers
    https://www.philschmid.de/fine-tune-embedding-model-for-rag
    https://www.philschmid.de/fine-tune-embedding-model-for-rag
    https://blog.gopenai.com/fine-tuning-embeddings-for-specific-domains-a-comprehensive-guide-5e4298b42185
    https://generativeai.pub/a-beginners-guide-to-fine-tuning-an-embedding-model-38bb4b4ae664
    https://newsletter.kaitchup.com/p/llama-32-embeddings-training


Generating Embeddings
    https://future.mozilla.org/builders/news_insights/llamafiles-for-embeddings-in-local-rag-applications/

Research
    https://research.trychroma.com/embedding-adapters
    https://arxiv.org/pdf/2409.15700
    https://arxiv.org/pdf/2410.02525
    Contextual document embeddings
        https://huggingface.co/jxm/cde-small-v1
    Vector graph
        https://towardsdatascience.com/vector-embeddings-are-lossy-heres-what-to-do-about-it-4f9a8ee58bb7
    MoE embeddings
        https://github.com/tianyi-lab/MoE-Embedding
    Run-time-lookup
        https://arxiv.org/abs/2406.15241
    Compression
        https://arxiv.org/abs/2407.09252
    MRL
        https://towardsdatascience.com/how-to-reduce-embedding-size-and-increase-rag-retrieval-speed-7f903d3cecf7
    Multi-Vector Retrieval
        https://huggingface.co/google/xtr-base-multilingual
    Hyperbolic Embeddings
        https://github.com/amazon-science/hyperbolic-embeddings


Quantization
    https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/
    https://jkatz05.com/post/postgres/pgvector-quantization/

RAG
    https://medium.com/intel-tech/optimize-vector-databases-enhance-rag-driven-generative-ai-90c10416cb9c


`The basic gist is that we first use the LLM to generate better, more precise keywords that the RAG’s embedding model will be able to use to create an embedding vector closer to relevant matches. The LLM is run again with the more relevant info that the RAG found to hopefully generate a more accurate response.`

Evaluate swapping from Chroma to https://github.com/neuml/txtai
Also eval swapping to vec-sql

https://www.reddit.com/r/LocalLLaMA/comments/15oome9/our_workflow_for_a_custom_questionanswering_app/
```
Last year my team worked on a fine tuned open source model, trained on US military doctrine and pubs ([workflow](https://www.reddit.com/r/LocalLLaMA/comments/15oome9/our_workflow_for_a_custom_questionanswering_app/) and [follow-up](https://www.reddit.com/r/LocalLLaMA/comments/1686ul6/some_lessons_learned_from_building_a_fine_tuned/) posts). Bottom line is that the fine tuned 7b model worked really well, especially on conceptual questions (like how maneuver and mission command interact): better than GPT-3.5 and about even with GPT-4 based on human ratings from military members.

Been itching to try fine tuning embeddings, and my team finally got a chance. We ran a series of experiments, but the big picture takeaway was that our first approach collapsed the embeddings space and made retrieval accuracy plummet, but a second approach using train+eval worked well and substantially improved retrieval.

We started with our model training data: a context+question column and answer column. We took the context chunk (500 tokens from a military publication) and the question generated from it, reversed their order and used them as the training data for the embeddings fine-tuning. So basically "When you see "What are the principles of air defense in urban areas?" then retrieve <some chunk about urban defense that has some sentences on air defense principles>.

We used Sentence Transformers and FSDP, because we had to shard the embedding model and data across multiple GPUs. To our distress however, each epoch of training made the model perform worse and worse, until at 5 epochs it was just random retrieval. Our intuition was that the model was overfitting and collapsing the embedding space until all documents were crammed next to each other. We used [WizMap](https://github.com/poloclub/wizmap/blob/main/LICENSE) to visualize embedded docs, and sure enough the base model showed clear clusters of docs, 2 epochs showed them kind of crammed closer, and at 5 epochs a giant blob with two camel humps.

We then switched to DDP from FSDP, which allows you to use an evaluator parameter during fine tuning, so we could use the eval data during training, not just post-hoc, something like:

    num_train_epochs=2,

    per_device_train_batch_size=32,

    per_device_eval_batch_size=32,

    During training, would train on a batch from the “TRAIN” dataset, and then evaluate on a batch from the “EVAL” dataet

    Use that train/eval comparison to inform the loss function

    Train for 2 or 5 epochs

    Post-training, ran our eval pipeline.

Success! Using BGE Small w. 384 dimensions, we went from:

    Base model top 20 accuracy of 54.4%.

    2 epochs fine-tuned model: Top 20 retrieval accuracy 70.8%.

    5 epochs fine-tuned model: Top 20 retrieval accuracy 73%.

We then tried Stella-400M 1024 dimensions:

    Base model top 20 accuracy of 62.9%.

    2 epochs fine-tuned model (train batch-size 4, gradient accumulation

    steps 20): Top 20 retrieval accuracy was 73.3%.

    3 epochs fine-tuned model (train batch-size 3, gradient accumulation

    steps 40): Top 20 retrieval accuracy was 72.4%

    Increased batch size (train batch size 8, grad accumulation steps 25) with 2

    epochs fine-tuning on 8 GPU clusters: Top 20 retrieval accuracy was 74.4%
```

```
This is a really tricky area of the field right now, because the current performance metrics we look for in embedding models are based on a set of ad-hoc metrics and random datasets that just so happened to be in vogue when the LLM sub-field started dominating the conversation a few years ago.

I’ve spent more hours the last two years than I can even describe on this, both personally and professionally, and here is how I currently think about this:

    The three axes to consider are concept obscurity, term volume, and top N precision.

    A model that performs well generally, aka on the MTEB leaderboard, is good at differentiating common concepts, when you have fewer terms to compare to one another, and when you’re comfortable with a “match” being in the top few results, not explicitly the first or second result.

    A more specialized model is the exact inverse, better on a set of highly specific, more obscure concepts, when you have a lot of them all at once, and when you need the top 1 or 2 matches to be “correct”.

Now, this gets even more fascinating, because there actually are real limits to how “good” a model can be on more common domains. And so, from my perspective, one simply considers the average term frequency of one’s domain relative to the dataset the model was trained on and can infer fitness from there.

Thus, models now are getting “better” at some more specialized domains because the datasets are larger and more inclusive of those sub-distributions. However, this scaling in “quality” does, from my testing, fall apart when the other two constraints come in.

So, long story short, use general models when you either have a “small” number of items to compare, OR are operating in a common domain, OR top N precision needs are loose. For most people, this is fine. For those of us in highly specialized domains where scale and precision are make or break factors, use a specialized model, up to and including creating your own.
```
- **101**
    https://www.youtube.com/watch?v=viZrOnJclY0
    https://aclanthology.org/W13-2322.pdf
- **Leaderboards**
    - https://huggingface.co/spaces/mteb/leaderboard




----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="vlm-evals"></a> VLM Evaluations
https://arxiv.org/abs/2411.04075
https://arxiv.org/abs/2411.02571
https://github.com/TRI-ML/vlm-evaluation
https://arxiv.org/abs/2411.13211
https://videoautoarena.github.io/
https://arxiv.org/abs/2411.17451
https://huggingface.co/spaces/MMInstruction/VL-RewardBench
https://github.com/illuin-tech/vidore-benchmark
- xkcd bench: https://github.com/arnokha/explain-xkcd-with-llms

- Document Understanding
    - https://arxiv.org/html/2411.01106v1

----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="mllm-evals"></a> MLLM Evaluations

https://arxiv.org/abs/2411.15296
https://arxiv.org/abs/2411.06284
----------------------------------------------------------------------------------------------------------------


