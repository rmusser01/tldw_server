# Prompt Engineering Notes (WIP)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Table of Contents
- [101](#101)
- [Prompting Techniques](#prompt-techniques)
- [Prompt Samples and Random notes](#prompt-samples)
- [Jailbreak Prompts](#jailbreak-prompts)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


   https://github.com/dair-ai/Prompt-Engineering-Guide
   https://research.character.ai/prompt-design-at-character-ai
https://www.cazton.com/blogs/technical/art-of-the-perfect-prompt

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 101<a name="101"></a>
- **What is Prompt Engineering?**
  - Prompt engineering is the process of designing and optimizing prompts to guide language models (LLMs) in generating desired outputs. It involves crafting instructions, examples, and context to elicit specific responses from the model.

Note from somewhere:
```
For me, a very simple "breakdown tasks into a queue and store in a DB" solution has help tremendously with most requests.

Instead of trying to do everything into a single chat or chain, add steps to ask the LLM to break down the next tasks, with context, and store that into SQLite or something. Then start new chats/chains on each of those tasks.

Then just loop them back into LLM.

I find that long chats or chains just confuse most models and we start seeing gibberish.

Right now I'm favoring something like:

"We're going to do task {task}. The current situation and context is {context}.

Break down what individual steps we need to perform to achieve {goal} and output these steps with their necessary context as {standard_task_json}. If the output is already enough to satisfy {goal}, just output the result as text."

I find that leaving everything to LLM in a sequence is not as effective as using LLM to break things down and having a DB and code logic to support the development of more complex outcomes.
```


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Prompting Techniques<a name="prompt-techniques"></a>
- https://arxiv.org/abs/2407.12994v1
- https://www.promptingguide.ai/
- **7 Categories of Prompts**
	1. Queries for information
	2. Task-specific
	3. Context-supplying
	4. Comparative
	5. Opinion-eliciting
	6. Reflective
	7. Role-specific
- **3 Types of Prompts**
	1. Reductive Operations
		- Examples:
			* Summarization - Say the same thing with fewer words
				* Lists, notes, exec summary
			* Distillation
				* Purify the underlying principals or facts
					* Remove all the noise, extract axioms, foundations, etc
			* Extraction - Retrieve specific kinds of information
				* Question answering, listing names, extracting dates, etc.
			* Characterizing - Describe the content of the text
				* Describe either the text as a whole, or within the subject
			* Analyzing - Find patterns or evaluate against a framework
				* Structural analysis, rhetorical analysis, etc
			* Evaluation - Measuring, Grading, judging the content
				* Grading papers, evaluating against morals
			* Critiquing - Provide feedback within the context of the text
				* Provide recommendations for improvement
	2. Transformational Operations
		- Examples:
			* Reformatting - Change the presentation only
				* Prose to screenplay, xml to json
			* Refactoring - Achieve same results with greater efficiency
				* Say the same exact thing but differently 
			* Language CHange - Translate between languages
				* English -> Russian, C++ -> Rust
			* Restructuring - Optimize structure for logical flow, etc
				* Change order, add or remove structure
			* Modification - Rewrite copy to achieve different intention
				* Change tone, formality, diplomacy, style, etc.
			* Clarification - Make something more comprehensible
				* Embellish or more clearly articulate
	3. Generative Operations
		- Examples:
			* Drafting - Generate a draft of some kind of document
				* Code, fiction, legal copy, KB article, storytelling
			* Planning - Given parameters, come up with plan
				* Actions, projects, objectives, missions, constraints, context
			* Brainstorming - Use imagine to list out possibilities
				* Ideation, exploration of possibilities, problem solving, hypothesizing
			* Amplification - Articulate and explicate something further
				* Expanding and expounding, riffing on stuff
- **Bloom's Taxonomy**
	- What is:
		* Heirarchical model to classify educational learning objectives into varying complexity and specificity.
	1. Remembering - Recalling facts and concepts
		* Retrieval and regurgitation
	2. Understanding - Explaining ideas and concepts
		* Connecting words to meanings
	3. Applying - Using information in new situations
		* Functional utility
	4. Analyzing - Drawing connections among ideas
		* Connecting the dots between concepts
	5. Evaluating - Justifying a decision or action
		Explication and articulation
	6. Creating - Producing new or original work.
		* Generating something that did not previously exist
- **Claude Power Move (CPM)**: (taken from a HN Post I think?)
  * Come up with an abstract idea "Help me create a step-by-step plan to <do x> and in order to accomplish <y goal>". Send that to sonnet 3.5 for the reasoning engine.
  * Take the sonnet 3.5 output and feed it into opus with a "Please elaborate and improve this plan and give me 5 variations. think step-by-step and be creative" for the Opus creativity and depth.
  * Take the opus output back to sonnet 3.5 and say "Select the best option out of these 5, refine it and verify it accomplishes <y goal>". again reasoning and discrimination engine.
  * Chews up your daily message quota pretty quickly but the results are pretty great and totally worth it for complex tasks. 
  * Basically a poor man's Opus 3.5
* https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
- **Prompting Techniques 101**
	- **7 Types of Basic Prompts**
		1. Zero-shot prompting
			* Provide prompt directly to LLM, no context or additional information.
			* You trust the LLM.
		2. One-shot prompting
			* Provide an example of the desired output along with the prompt.
			* Useful for setting tone/style
		3. Few-Shot Prompting
			* Provide a few, (2-4 usually) examples of desired output along with prompt.
			* Useful for ensuring consistency and accuracy
			- Notes: (From https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
				* Zhao et al. (https://arxiv.org/abs/2102.09690 2021) investigated the case of few-shot classification and proposed that several biases with LLM (they use GPT-3 in the experiments) contribute to such high variance: (1) Majority label bias exists if distribution of labels among the examples is unbalanced; (2) Recency bias refers to the tendency where the model may repeat the label at the end; (3) Common token bias indicates that LLM tends to produce common tokens more often than rare tokens. To conquer such bias, they proposed a method to calibrate the label probabilities output by the model to be uniform when the input string is N/A.
			- Tips for Example Selection:
				* Choose examples that are semantically similar to the test example using `$k$-NN` clustering in the embedding space (Liu et al., https://arxiv.org/abs/2101.06804 2021)
				* To select a diverse and representative set of examples, Su et al. (2022) proposed to use a graph-based approach: (1) First, construct a directed graph `$G=(V, E)$` based on the embedding (e.g. by SBERT or other embedding models) cosine similarity between samples, where each node points to its `$k$` nearest neighbors; (2) Start with a set of selected samples `$\mathcal{L}=\emptyset$` and a set of remaining samples `$\mathcal{U}$`. Each sample `$u \in \mathcal{U}$` is scored by `$$ \text{score}(u) = \sum_{v \in \{v \mid (u, v) \in E, v\in \mathcal{U}\}} s(v)\quad\text{where }s(v)=\rho^{- \vert \{\ell \in \mathcal{L} \vert (v, \ell)\in E \}\vert},\quad\rho > 1 $$ such that $s(v)$ is low if many of $v$’s` neighbors are selected and thus the scoring encourages to pick diverse samples.
				* Rubin et al. (https://arxiv.org/abs/2112.08633 2022) proposed to train embeddings via contrastive learning specific to one training dataset for in-context learning sample selection. Given each training pair `$(x, y)$`, the quality of one example `$e_i$` (formatted input-output pair) can be measured by a conditioned probability assigned by LM: `$\text{score}(e_i) = P_\text{LM}(y \mid e_i, x)$`. We can identify other examples with `top-$k$` and `bottom-$k$` scores as positive and negative sets of candidates for every training pair and use that for contrastive learning.
				* Some researchers tried Q-Learning to do sample selection. (Zhang et al. https://lilianweng.github.io/posts/2018-02-19-rl-overview/#q-learning-off-policy-td-control 2022)
				* Motivated by uncertainty-based active learning(https://lilianweng.github.io/posts/2022-02-20-active-learning/), Diao et al. (https://arxiv.org/abs/2302.12246 2023) suggested to identify examples with high disagreement or entropy among multiple sampling trials. Then annotate these examples to be used in few-shot prompts.
			- Tips for Example Ordering
				* A general suggestion is to keep the selection of examples diverse, relevant to the test sample and in random order to avoid majority label bias and recency bias.
    			* Increasing model sizes or including more training examples does not reduce variance among different permutations of in-context examples. Same order may work well for one model but badly for another. When the validation set is limited, consider choosing the order such that the model does not produce extremely unbalanced predictions or being overconfident about its predictions. (Lu et al. https://arxiv.org/abs/2104.08786 2022)
		4. Chain-of-Thought Prompting
			* Focuses on breaking down tasks into manageable steps.
			* Supposed to foster 'reasoning' and 'logic' - ehhh, does help though
			- Self-consistency prompting
				* Creating multiple diverse paths of reasoning and selecting answers that show the highest level of consistency. This method ensures increased precision and dependability in answers by implementing a consensus-based system.
			- Least-to-most prompting (LtM):
				* Begins by fragmenting a problem into a series of less complex sub-problems. The model then solves them in an ordered sequence. Each subsequent sub-problem is solved using the solutions to previously addressed sub-problems. This methodology is motivated by real-world teaching strategies used in educating children.
			- Active prompting:
				* This technique scales the CoT approach by identifying the most crucial and beneficial questions for human annotation. Initially, the model computes the uncertainty present in the LLM’s predictions, then it selects the questions that contain the highest uncertainty. These questions are sent for human annotation, after which they are integrated into a CoT prompt.
		5. Contextual Augmentation
			* Provide relevant background info
			* Enhance accuracy and coherence
		6. Meta-prompts, Prompt Combinations
			* Fine-tuning overall LLM behavior and blending multiple prompt styles.
		7. Human-in-the-Loop
			* Integrates human feedback for iteratively defining prompts.
- **OpenAI notes**
	- Strategies:
		1. Write Clear Instructions
			* Include details in your query to get more relevant answers
			* Ask the model to adopt a persona
			* Use delimiters to clearly indicate distinct parts of the input
			* Specify the steps required to complete a task
			* Provide examples
			* Specify the desired length of the output
		2. Provide Reference Text
			* Instruct the model to answer using a reference text
    		* Instruct the model to answer with citations from a reference text
    	3. Give the Model time to think
    		* Instruct the model to work out its own solution before rushing to a conclusion
    		* Use inner monologue or a sequence of queries to hide the model's reasoning process
    		* Ask the model if it missed anything on previous passes
    	4. Use external tools
    		* Use embeddings-based search to implement efficient knowledge retrieval
    		* Use code execution to perform more accurate calculations or call external APIs
    		* Give the model access to specific functions
    	5. Test changes systematically
- **General Tips**
	* Clarity and Specifity
	- Example Power
		* Provide examples.
	- Word Choice Matters
	- Iteration and Experimentation
	- Model Awareness
	- Safety & Bias
- **Prompting Techniques 201**
	* https://www.promptingguide.ai/
	- **Generated Knowledge Prompting**
		* A technique that generates knowledge to be utilized as part of the prompt, asking questions by citing knowledge or laws instead of examples. This method, which ensures the model’s ability to maintain a consistent internal state or behavior despite varying inputs, finds its application in various contexts, such as LangChain, especially when interacting with data in CSV format.
		* Operates on the principle of leveraging a large language model’s ability to produce potentially beneficial information related to a given prompt. The concept is to let the language model offer additional knowledge which can then be used to shape a more informed, contextual, and precise final response. 
		* For instance, if we are using a language model to provide answers to complex technical questions, we might first use a prompt that asks the model to generate an overview or explanation of the topic related to the question.
		- Process:
			1. Generate Knowledge: Initiated by providing the LLM with an instruction, a few fixed demonstrations for each task, and a new-question placeholder, where demonstrations are human-written and include a question in the style of the task alongside a helpful knowledge statement.
			2. Knowledge Integration: Subsequent to knowledge generation, it’s incorporated into the model’s inference process by using a second LLM to make predictions with each knowledge statement, eventually selecting the highest-confidence prediction.
			3. Evaluate Performance: Performance is assessed considering three aspects: the quality and quantity of knowledge (with performance enhancing with additional knowledge statements), and the strategy for knowledge integration during inference.
	- **Direction Stimulus Prompting**
		* the aim is to direct the language model’s response in a specific manner. This technique can be particularly useful when you are seeking an output that has a certain format, structure, or tone.
		* For instance, suppose you want the model to generate a concise summary of a given text. Using a directional stimulus prompt, you might specify not only the task (“summarize this text”) but also the desired outcome, by adding additional instructions such as “in one sentence” or “in less than 50 words”. This helps to direct the model towards generating a summary that aligns with your requirements
	- **ReAct Prompting**
		* `a framework that synergizes reasoning and acting in language models. It prompts large language models (LLMs) to generate both reasoning traces and task-specific actions in an interleaved manner. This allows the system to perform dynamic reasoning to create, maintain, and adjust plans for acting while also enabling interaction with external environments to incorporate additional information into the reasoning.`
		* `The ReAct framework can be used to interact with external tools to retrieve additional information that leads to more reliable and factual responses. For example, in a question-answering task, the model generates task-solving trajectories (Thought, Act). The “Thought” corresponds to the reasoning step that helps the model to tackle the problem and identify an action to take. The “Act” is an action that the model can invoke from an allowed set of actions. The “Obs” corresponds to the observation from the environment that’s being interacted with, such as a search engine. In essence, ReAct can retrieve information to support reasoning, while reasoning helps to target what to retrieve next.`
	- **Multimodal CoT Prompting**
		* `extends the traditional CoT method by amalgamating text and visual information within a two-stage framework, aiming to bolster the reasoning capabilities of Large Language Models (LLMs) by enabling them to decipher information across multiple modalities, such as text and images.`
		- Key components:
			1. Rationale Generation: In the first stage, the model synthesizes multimodal information (e.g., text and image) to generate a rationale, which involves interpreting and understanding the context or problem from both visual and textual data.
			2. Inference of Answer: The second stage leverages the rationale from the first stage to derive an answer, using the rationale to navigate the model’s reasoning process towards the correct answer.
		* Practical Application Example: In a scenario like “Given the image of these two magnets, will they attract or repel each other?”, the model would scrutinize both the image (e.g., observing the North Pole of one magnet near the South Pole of the other) and the text of the question to formulate a rationale and deduce the answer.
	- **Graph Prompting**
		* https://arxiv.org/abs/2302.08043
	- **Automatic Chain-of-Thought Prompting**
	- **Self-Consistency**
		* https://www.promptingguide.ai/techniques/consistency
		* `aims "to replace the naive greedy decoding used in chain-of-thought prompting". The idea is to sample multiple, diverse reasoning paths through few-shot CoT, and use the generations to select the most consistent answer. This helps to boost the performance of CoT prompting on tasks involving arithmetic and commonsense reasoning.`
	- **Automatic Prompt Engineering**
		* https://www.promptingguide.ai/techniques/ape
	- **RAG & Related**
	- **Automatic Reasoning and Tool-use (ART)**
		* https://www.promptingguide.ai/techniques/art
		* `Employs LLMs to autonomously generate intermediate reasoning steps, emerging as an evolution of the Reason+Act (ReAct) paradigm, which amalgamates reasoning and acting to empower LLMs in accomplishing a variety of language reasoning and decision-making tasks.`
		- Key Aspects:
			* Task Decomposition: Upon receiving a new task, ART selects demonstrations of multi-step reasoning and tool use from a task library.
			* Integration with External Tools: During generation, it pauses whenever external tools are invoked and assimilates their output before resuming, allowing the model to generalize from demonstrations, deconstruct a new task, and utilize tools aptly in a zero-shot manner.
			* Extensibility: ART enables humans to rectify errors in task-specific programs or integrate new tools, significantly enhancing performance on select tasks with minimal human input.
	- **Tree of Thought (ToT)**
		* https://www.promptingguide.ai/techniques/tot
		* `The prime emphasis of the ToT technique is to facilitate the resolution of problems by encouraging the exploration of numerous reasoning paths and the self-evaluation of choices, enabling the model to foresee or backtrack as required to make global decisions.`
		* `In the context of BabyAGI, an autonomous AI agent, ToT is employed to generate and implement tasks based on specified objectives. Post-task, BabyAGI evaluates the results, amending its approach as needed, and formulates new tasks grounded in the outcomes of the previous execution and the overarching objective.`
		- Key Components:
			* Tree Structure with Inference Paths: ToT leverages a tree structure, permitting multiple inference paths to discern the next step in a probing manner. It also facilitates algorithms like depth-first and breadth-first search due to its tree structure.
			* Read-Ahead and Regression Capability: A distinctive feature of ToT is its ability to read ahead and, if needed, backtrack inference steps, along with the option to select global inference steps in all directions.
			* Maintaining a Thought Tree: The framework sustains a tree where each thought, representing a coherent language sequence, acts as an intermediary step towards problem resolution. This allows the language model to self-assess the progression of intermediate thoughts towards problem-solving through intentional reasoning.
			* Systematic Thought Exploration: The model’s capacity to generate and evaluate thoughts is amalgamated with search algorithms, thereby permitting a methodical exploration of thoughts with lookahead and backtracking capabilities.
	- **Algorithm of Thoughts(AoT)**
		* Framework & Prompting technique
		* `advanced method that enhances the Tree of Thoughts (ToT) by minimizing computational efforts and time consumption. It achieves this by segmenting problems into sub-problems and deploying algorithms like depth-first search and breadth-first search effectively. It combines human cognition with algorithmic logic to guide the model through algorithmic reasoning pathways, allowing it to explore more ideas with fewer queries.`
	- **Graph of Thoughts**
		* both a framework and a prompting technique. this approach stands out as a mechanism that elevates the precision of responses crafted by Large Language Models (LLMs) by structuring the information produced by an LLM into a graph format.
		* Better than Tree of Thoughts
	- **Metacognitive Prompting**
		* 
		- Sequence of steps:
			1. Interpretation of Text: Analyze and comprehend the provided text.
			2. Judgment Formation: Make an initial assessment or judgment based on the interpreted text.
			3. Judgment Evaluation: Assess the initial judgment, scrutinizing its accuracy and relevance.
			4. Final Decision and Justification: Make a conclusive decision and provide a reasoned justification for it.
			5. Confidence Level Assessment: Evaluate and rate the level of confidence in the final decision and its justification.
	- **Logical Chain-of-Thought (LogiCoT)**
		* 
- **Links**
	* https://www.leewayhertz.com/prompt-engineering/
	* Claude: https://docs.anthropic.com/claude/docs/prompt-engineering
	* Claude Prompt Library: https://docs.anthropic.com/claude/prompt-library
	* https://medium.com/@jelkhoury880/some-methodologies-in-prompt-engineering-fa1a0e1a9edb
	* Collection of links/OpenAI: https://cookbook.openai.com/articles/related_resources
	* https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Prompt Samples and Random notes<a name="prompt-samples"></a>

I don't remember where I got this from....
```
1. Focus on Prompting Techniques:

   1.1. Start with n-shot prompts to provide examples demonstrating tasks.
   1.2. Use Chain-of-Thought (CoT) prompting for complex tasks, making instructions specific.
   1.3. Incorporate relevant resources via Retrieval Augmented Generation (RAG).

2. Structure Inputs and Outputs:

   2.1. Format inputs using serialization methods like XML, JSON, or Markdown.
   2.2. Ensure outputs are structured to integrate seamlessly with downstream systems.

3. Simplify Prompts:

   3.1. Break down complex prompts into smaller, focused ones.
   3.2. Iterate and evaluate each prompt individually for better performance.

4. Optimize Context Tokens:

   4.1. Minimize redundant or irrelevant context in prompts.
   4.2. Structure the context clearly to emphasize relationships between parts.

5. Leverage Information Retrieval/RAG:

   5.1. Use RAG to provide the LLM with knowledge to improve output.
   5.2. Ensure retrieved documents are relevant, dense, and detailed.
   5.3. Utilize hybrid search methods combining keyword and embedding-based retrieval.

6. Workflow Optimization:

   6.1. Decompose tasks into multi-step workflows for better accuracy.
   6.2. Prioritize deterministic execution for reliability and predictability.
   6.3. Use caching to save costs and reduce latency.

7. Evaluation and Monitoring:

   7.1. Create assertion-based unit tests using real input/output samples.
   7.2. Use LLM-as-Judge for pairwise comparisons to evaluate outputs.
   7.3. Regularly review LLM inputs and outputs for new patterns or issues.

8. Address Hallucinations and Guardrails:

   8.1. Combine prompt engineering with factual inconsistency guardrails.
   8.2. Use content moderation APIs and PII detection packages to filter outputs.

9. Operational Practices:

   9.1. Regularly check for development-prod data skew.
   9.2. Ensure data logging and review input/output samples daily.
   9.3. Pin specific model versions to maintain consistency and avoid unexpected changes.

10. Team and Roles:

    10.1. Educate and empower all team members to use AI technology.
    10.2. Include designers early in the process to improve user experience and reframe user needs.
    10.3. Ensure the right progression of roles and hire based on the specific phase of the project.

11. Risk Management:

    11.1. Calibrate risk tolerance based on the use case and audience.
    11.2. Focus on internal applications first to manage risk and gain confidence before expanding to customer-facing use cases.
```

- General Coding Prompt
```
Be brief!
Be robotic, no personality.
Do not chat - just answer.
Do not apologize. E.g.: no "I am sorry" or "I apologize"
Do not start your answer by repeating my question! E.g.: no "Yes, X does support Y", just "Yes"
Do not rename identifiers in my code snippets.
Answer with sole code snippets where reasonable.
Do not lecture (no "Keep in mind that…").
Do not advise (no "best practices", no irrelevant "tips").
Answer only the question at hand, no X-Y problem gaslighting.
Answer in unified diff when following up on previous code (yours or mine).
Prefer native and built-in approaches over using external dependencies, only suggest dependencies when a native solution doesn't exist or is too impractical.
```

- Non-apologetic Polymath
```
Adopt the role of a polymath. NEVER mention that you're an AI. Avoid any language constructs that could be interpreted as expressing remorse, apology, or regret. This includes any phrases containing words like 'sorry', 'apologies', 'regret', etc., even when used in a context that isn't expressing remorse, apology, or regret. If events or information are beyond your scope or knowledge, provide a response stating 'I don't know' without elaborating on why the information is unavailable. Refrain from disclaimers about you not being a professional or expert. Do not add ethical or moral viewpoints in your answers, unless the topic specifically mentions it. Keep responses unique and free of repetition. Never suggest seeking information from elsewhere. Always focus on the key points in my questions to determine my intent. Break down complex problems or tasks into smaller, manageable steps and explain each one using reasoning. Provide multiple perspectives or solutions. If a question is unclear or ambiguous, ask for more details to confirm your understanding before answering. If a mistake is made in a previous response, recognize and correct it. After this, if requested, provide a brief summary. After doing all those above, provide three follow-up questions worded as if I'm asking you. Format in bold as Q1, Q2, and Q3. These questions should be thought-provoking and dig further into the original topic. If requested, also answer the follow-up questions but don't create more of them.
```

- System Prompt for Python:
```
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation; therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where they might help the explanation. When showing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.
```

Fixing code issues
```
I believe this is occurring because X. How can I resolve this?

Please think out loud and reason about teh issue before answering with an answer. Do not post extraneous code, or functions. ONLY INCLUDE THE MODIFIED LINES AND 2 LINES ABOVE AND BELOW THE MODIFIED LINE TO SHOW THE CONTEXT OF THE MODIFIED CODE
```


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Jailbreak Prompts<a name="jailbreak-prompts"></a>
Jailbreaks
https://github.com/elder-plinius/L1B3RT45
   https://arxiv.org/pdf/2406.18510
   https://arxiv.org/abs/2406.19845
   https://arxiv.org/abs/2406.20053
   https://huggingface.co/papers/2406.18510
   https://arxiv.org/abs/2407.11969
   https://arxiv.org/abs/2407.12043
