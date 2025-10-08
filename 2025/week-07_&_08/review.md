# From Generation to Reasoning: The Evolution of Language Models from Generative Pre-trained Transformers to Reasoning Systems

## Abstract

In the context of the rapid advancement of large language models (LLMs), particular attention is devoted to enhancing their capacity for logical reasoning. One significant achievement in this domain is the **DeepSeek-R1** model, developed to stimulate LLM reasoning capabilities through reinforcement learning (Reinforcement Learning, RL) methods. DeepSeek-R1 represents an innovative approach aimed at improving the quality of response generation in tasks requiring multi-step logical inference.

#### Key Characteristics of DeepSeek-R1

DeepSeek-R1 belongs to the class of reasoning models such as **OpenAI o1/o3**, **Google Gemini 2.0 Flash Thinking**, and **Alibaba Cloud Qwen QwQ**. Unlike traditional LLMs, which aim to generate a final answer immediately, DeepSeek-R1 employs the **Chain-of-Thought (CoT)** method, which involves generating a sequence of intermediate reasoning steps before delivering the final result. This approach enables the model not only to improve answer accuracy but also to enhance the transparency and interpretability of the decision-making process.

#### Technical Details and Contribution to LLM Development

DeepSeek-R1 is based on a reinforcement learning paradigm, enabling the model to adapt to complex tasks requiring deep analysis and logical deduction. Unlike standard fine-tuning methods, the RL approach provides more flexible, goal-oriented training. This is especially important for tasks requiring not merely text generation but sequential reasoning—for instance, in mathematical problems, commonsense questions, and symbolic reasoning.

#### Comparison with Other Reasoning Models

DeepSeek-R1 distinguishes itself among analogous models through its effective integration of CoT with RL techniques. While OpenAI o1/o3 and Google Gemini 2.0 Flash Thinking also utilize CoT, DeepSeek-R1 emphasizes optimizing the reasoning process via reinforcement learning, allowing the model to better adapt to tasks with high uncertainty.

Thus, DeepSeek-R1 represents an important step in the evolution of reasoning models, offering a novel approach to enhancing LLM logical reasoning capabilities through the integration of CoT and RL methods.

# 1. Chain-of-Thought (CoT) Technique

#### Core Concept and Origins

The **Chain-of-Thought (CoT)** technique, introduced in the paper **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022) [[1](https://arxiv.org/abs/2201.11903  )], has become a vital tool in prompt engineering for improving LLM answer quality, particularly in reasoning tasks. CoT emerged from the observation that large language models, unlike smaller models, exhibit an **emergent property**—the ability to significantly improve their responses when prompted to generate intermediate reasoning steps. This property becomes evident in models with 100 billion parameters or more.

#### Implementation and Advantages of CoT Prompting

**CoT prompting** involves explicitly instructing the model not to provide the final answer immediately, but first to generate a sequence of intermediate reasoning steps explaining its thought process, and only then to deliver the final result. This approach is often combined with **few-shot learning**, where the model is provided with several examples of queries demonstrating the desired reasoning chains and corresponding answers.

Applying CoT prompting leads to a noticeable **improvement in answer quality**, especially in areas such as:

*   **Mathematical problems (arithmetic reasoning):** Solving tasks involving addition, subtraction, multiplication, division, and more complex mathematical operations.
*   **Commonsense reasoning:** Answering questions requiring application of world knowledge and common sense.
*   **Symbolic reasoning:** Tasks involving manipulation of symbols and logical operations.

Beyond improved accuracy, CoT offers additional advantages:

*   **Decision transparency:** The reasoning chain allows understanding how the model arrived at a particular answer.
*   **Interpretability:** Intermediate reasoning steps make the inference process more comprehensible and analyzable.
*   **Efficient resource utilization:** CoT encourages models to allocate more computational resources (via generation of intermediate tokens) to complex tasks.

<details> 
    <summary><em><strong>Brief Overview of Key Papers on Chain-of-Thought and Tree-of-Thought: The Evolution of Reasoning Methods in LLMs Leading to R1📚</strong></em></summary>

### **Paper "Chain-of-thought prompting elicits reasoning in large language models" (2022) [[1](https://arxiv.org/abs/2201.11903  )] comprehensively investigates Chain-of-Thought Prompting and its impact on LLM reasoning capabilities.**

Key aspects and findings of the paper include:

1.  **Advantages of CoT Prompting:** CoT improves accuracy across diverse reasoning domains, including arithmetic, commonsense, and symbolic tasks. The method involves generating a sequence of intermediate reasoning steps leading to the answer and is easily implemented using a few demonstration examples. Notably, the PaLM 540B model using CoT achieved a new state-of-the-art result on the GSM8K benchmark for math word problems.

2.  **Applicability to Various Reasoning Types:** CoT is effectively applied to:
    *   **Arithmetic reasoning:** Tasks from datasets GSM8K, SVAMP, ASDiv, AQuA, MAWPS.
    *   **Commonsense reasoning:** Tasks from datasets CSQA, StrategyQA, date and sports event understanding, and robot instruction tasks (SayCan).
    *   **Symbolic reasoning:** Letter Concatenation, Coin Flip tasks.

3.  **Model Scale Requirement:** CoT is an **emergent ability** that manifests with increasing model size. Effectiveness significantly increases when using very large models such as PaLM (540B parameters) and GPT-3 (175B parameters), compared to smaller models.

4.  **Examples of CoT Prompting:** The paper provides CoT examples for various reasoning types, demonstrating how breaking a task into simpler steps and explaining each step in natural language leads to the final answer.

5.  **Ablation Studies and Robustness Testing:** Analyses of various CoT prompting variants show that expressing intermediate steps in natural language is crucial to the method's success. Robustness analysis confirms that CoT is sufficiently resilient to changes in annotation style and differences between annotators.

6.  **Error Analysis:** Analysis of incorrect reasoning chains enables classification of errors (calculator error, skipped step, meaning misunderstanding, inconsistent chain) and identifies directions for model improvement. Importantly, the paper emphasizes that there are no guarantees of complete correctness and coherence in LLM-generated reasoning.

7.  **Comparison with Existing Methods:** CoT prompting differs from methods requiring neural network training or fine-tuning to generate intermediate steps. CoT enables reasoning without requiring extensive annotations and is suitable for a wide range of text-to-text NLP tasks.

**Conclusion:**

The Chain-of-Thought prompting study underscores the importance of prompting as a key method for improving reasoning quality. Key conclusions include:

- **Prompting:** Instead of additional training stages, Chain-of-Thought prompting uses specially formulated prompts to stimulate the model toward sequential logical reasoning.
- **Model Scale:** Method effectiveness increases with model size, especially with large models with billions of parameters.
- **Few-shot Examples:** Adding a few examples further enhances the model’s ability to scale and reason logically.

This approach demonstrates a direct correlation between model scaling quality and its parameters, opening new horizons in artificial intelligence.

### **Self-Consistency for Enhancing CoT**

In the pursuit of further improving the reliability and accuracy of reasoning, the Chain-of-Thought technique has evolved into the **Self-Consistency (CoT-SC)** method, introduced in the significant work "Self-Consistency Improves Chain of Thought Reasoning in Language Models" [[2](https://arxiv.org/abs/2203.11171  )]. While standard CoT prompting typically relies on greedy decoding, selecting the most probable reasoning chain, CoT-SC introduces the principle of **self-consistency**, grounded in the intuitive understanding that complex reasoning tasks may have multiple equally valid solution paths.

The core idea of CoT-SC is to generate an **ensemble of diverse reasoning chains** for the same input query via stochastic sampling from the language model. Instead of relying on a single, potentially error-prone output, CoT-SC aggregates results by selecting the final answer that demonstrates the **highest consistency** among generated chains—the principle known as **majority voting**. This approach significantly reduces dependence on random fluctuations during generation and enhances the overall robustness of the final answer.

**Advantages of Self-Consistency (CoT-SC):**

*   **Increased Reliability and Accuracy:** By accounting for multiple possible reasoning paths, CoT-SC delivers more stable, reliable, and accurate results, particularly for complex tasks requiring deep logical deduction.
*   **Simple Implementation and Computational Efficiency:** The method is straightforward to integrate, requiring no additional training or labor-intensive data labeling, while demonstrating significant performance improvements.
*   **Robustness to Prompt Variability and Sampling Strategies:** CoT-SC exhibits remarkable resilience to minor changes in prompt phrasing and different sampling strategies, highlighting its practical value.

Experimental research presented in [[2](https://arxiv.org/abs/2203.11171  )] convincingly demonstrates the empirical superiority of CoT-SC over standard CoT prompting and several alternative decoding methods. Across a broad spectrum of autoregressive models—including UL2-20B, GPT-3-175B, LaMDA-137B, and PaLM-540B—CoT-SC showed statistically significant accuracy improvements on both arithmetic and commonsense tasks. Notably, the method demonstrated impressive gains on authoritative benchmarks GSM8K, SVAMP, AQuA, StrategyQA, and ARC-challenge, confirming its effectiveness and universality.

Thus, Self-Consistency (CoT-SC) represents an important advancement in the evolution of reasoning techniques for large language models, offering an elegant and effective way to enhance answer reliability and accuracy through an ensemble approach to reasoning and majority voting.

> Stochastic sampling in a language model enables the creation of an ensemble of diverse reasoning chains, introducing variability through different hyperparameters. The best chains are selected by majority voting, where the most consistent answer is considered the best.

### Evolution of CoT: Tree-of-Thought (ToT)

Despite the recognized effectiveness of Chain-of-Thought (CoT) in tasks requiring logical reasoning, the linear structure of CoT's thought sequence can become a limiting factor when solving particularly complex and multi-faceted problems. In such scenarios—where deep exploration of hypotheses, evaluation of alternative solution paths, and the ability to backtrack to previous reasoning stages are required—the linear trajectory of CoT proves insufficient. In response to these limitations, innovative approaches, **Tree-of-Thoughts (ToT)**, were proposed in landmark works **"Large Language Model Guided Tree-of-Thought"** (Yao et al., 2023) [[3](https://arxiv.org/abs/2305.08291  )] and **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** (Long, 2023) [[4](https://arxiv.org/abs/2305.10601  )]. The ToT framework conceptually extends the CoT paradigm by introducing a tree-like organization of the reasoning process, enabling models to perform more flexible and strategic search for solutions.

#### Non-linear Reasoning Process and Cognitive Analogy to "System 2"

In stark contrast to the linear unfolding of thought chains in CoT, Tree-of-Thoughts (ToT) architecturally represents the cognitive inference process as an **hierarchical tree**. Each discrete "thought" within ToT is defined as a semantically coherent sequence of verbal units, representing a conceptually meaningful intermediate step toward solving the target task. The pivotal innovation of ToT is the implementation of a **backtracking** mechanism, enabling recursive return to previous nodes in the reasoning tree and selection of alternative exploration branches if the current trajectory proves semantically dead-ended or heuristically suboptimal. This functional feature of ToT correlates with the more complex and reflective mode of human thinking often conceptualized in cognitive psychology as "System 2". While CoT demonstrates an analogy to intuitive, fast "System 1" thinking, ToT aims to emulate the more deliberate, strategic, and resource-intensive "System 2" thinking within the context of large language models.

#### Decomposition and Key Components of the Tree-of-Thoughts Framework

Unlike CoT's predominant focus on prompt engineering techniques, Tree-of-Thoughts (ToT) constitutes a comprehensive framework requiring **programmatic orchestration** to manage the tree-based search process. Effective implementation of ToT relies on the integration of several interrelated key components, synergistically interacting to navigate the space of tree-based reasoning:

1.  **Thought Decomposition:** The initial stage involves decomposing the original task into discrete, semantically distinct "thought units" or reasoning steps. A critical aspect of decomposition is achieving an optimal balance between detail and semantic richness of each "thought". Excessive decomposition may lead to combinatorial explosion and loss of contextual integrity, while overly coarse "thoughts" may hinder generation of diverse and relevant alternatives.

2.  **Thought Generator:** This component is responsible for automated generation of a spectrum of potential "thoughts" at each tree node. Literature identifies two dominant approaches to generation:
    *   **Independent and Identical Distribution (sampling):** This method generates an ensemble of statistically independent "thoughts" based on a given CoT prompt initiating the reasoning process. This approach proves particularly productive in scenarios with a vast space of possible "thoughts," where maximizing diversity of generated alternatives is essential.
    *   **Sequential Proposing (propose prompting):** An alternative method involves iterative generation of "thoughts" using specialized prompts deliberately designed to stimulate the generation of new and conceptually distinct ideas. This approach demonstrates effectiveness in situations with a limited "thought" space, where priority is avoiding semantic duplication and redundancy.

3.  **State Evaluator:** To ensure directed and heuristically justified search within the tree of reasoning, a mechanism for evaluating intermediate progress at each step is required. The state evaluator's functionality is implemented through the following methodological solutions:
    *   **Independent Value Prompting:** This method involves autonomous evaluation of the heuristic "value" or promise of each individual reasoning state based on specialized prompts emphasizing relevant progress criteria.
    *   **Collaborative State Voting (vote prompting):** An alternative approach entails comparative evaluation of multiple competing states and heuristic selection of the most promising option through a voting or ranking procedure based on defined criteria.

4.  **Search Algorithm:** The final, yet critically important, component of the ToT framework is the algorithm defining the global strategy for navigating and exploring the "thought" tree. In pioneering ToT works, two fundamental search algorithms were proposed:
    *   **Breadth-First Search (BFS):** The BFS algorithm maintains a dynamic pool of *b* most heuristically promising states at each tree level and concurrently explores all possible "thoughts" emanating from each state in the pool.
    *   **Depth-First Search (DFS):** Conversely, the DFS algorithm prioritizes deep exploration of the most promising tree branch until reaching a terminal state (solution) or until heuristic recognition of the current path as unpromising, after which it backtracks to the nearest alternative branch and continues the search.

#### Key Advantages of the Tree-of-Thoughts Paradigm

The Tree-of-Thoughts framework is characterized by a set of significant advantages that define its potential as a promising direction for advancing LLM reasoning capabilities:

*   **Generality:** ToT possesses conceptual universality, allowing prior methods such as Input-Output (IO), Chain-of-Thought (CoT), Self-Consistency CoT (CoT-SC), and self-improvement approaches to be viewed as specialized, reduced cases of ToT, characterized by limited search tree depth and width.
*   **Architectural Modularity:** ToT architecture exhibits pronounced modularity, enabling independent modification and optimization of individual components—the base LLM, thought decomposition, generation, and evaluation mechanisms, and the search algorithm. This modularity fosters flexibility in tuning and opens prospects for targeted enhancement of specific functional blocks.
*   **Contextual Adaptability:** ToT demonstrates high adaptability to the specific characteristics of solved tasks, cognitive capabilities of the employed LLM, and computational resource constraints. Different classes of tasks may require varied ToT configurations, including optimal search algorithm selection, decomposition strategy, and state evaluation methods.
*   **Practical Applicability and Integration Convenience:** The ToT framework is practically oriented, requiring no resource-intensive additional training or fine-tuning of LLMs. ToT can be efficiently implemented on top of existing pre-trained language models via programmatic orchestration, significantly simplifying its practical application and scaling.

#### Empirical Validation and Experimental Results

Empirical validation of Tree-of-Thoughts effectiveness was conducted on several cognitively complex tasks where traditional linear approaches demonstrate limited efficacy. Specifically, ToT demonstrated statistically significant superiority in the following tasks:

*   **Mathematical Game "24" (Game of 24):** A classic puzzle requiring manipulation of four given numbers via arithmetic operations to achieve the target value of 24. Application of ToT achieved a 74% success rate, whereas CoT achieved only 4%.
*   **Creative Writing with a Given Ending (Creative Writing):** The task of generating a coherent, four-paragraph text ending with four predetermined final sentences. Expert evaluations, conducted both with GPT-4 and human raters, consistently indicated ToT's superiority in generating higher-quality and semantically cohesive texts compared to IO and CoT.
*   **Solving Mini Crosswords (Mini Crosswords):** A task requiring integration of lexical knowledge, logical reasoning, and spatial thinking to fill a 5x5 crossword grid based on verbal clues. ToT demonstrated substantial improvement in performance compared to IO and CoT in solving this complex task integrating reasoning and knowledge retrieval.

#### Potential Limitations and Future Development Directions

Despite encouraging results, the Tree-of-Thoughts framework is not without certain limitations and opens several promising directions for further development. One key limitation is **increasing computational complexity**, driven by the need for multiple LLM inferences and exponential growth of the search space with increasing tree depth and width. Additionally, **ToT effectiveness critically depends on the quality and adequacy of implementation of individual framework components**, including thought decomposition strategy, generator, and state evaluator. Future research may focus on developing more efficient and scalable tree-search algorithms, optimizing heuristic state evaluation methods, and adapting ToT to specific requirements of diverse task classes and resource constraints. A highly promising direction also involves exploring the possibility of **integrating ToT principles into the pre-training process of LLMs**, potentially leading to models inherently possessing more developed capabilities for strategic and multi-step resolution of complex problems.

### Conclusion

Technologies **Chain-of-Thought (CoT)** and **Tree-of-Thought (ToT)** mark fundamental milestones in the progressive development of methodologies for enhancing **reasoning competencies** of large language models. CoT, as an emergent property of large neural network architectures, opened new horizons in improving answer generation quality for tasks requiring logical deduction and semantic knowledge utilization. ToT, in turn, conceptually and functionally advances CoT ideas by offering a more flexible, non-linear, and strategically oriented approach to reasoning, approximating human cognitive mechanisms of problem-solving. The future research vector in this domain appears directed toward developing even more efficient, resource-efficient, and scalable algorithms for managing tree-like reasoning, as well as integrating the ToT paradigm into a wide spectrum of application domains requiring LLMs not merely to generate linguistically coherent text, but to perform advanced intellectual analysis, strategic planning, and reliable solution of complex real-world problems.

</details>

---

> These approaches are no longer pure prompt engineering—you cannot solve them with a single text. You need to write programs that manage the process. In this sense, they are already within the LLM Programs paradigm.

# 2. Large Language Model Programs

### Abstract

This paper examines modern methods for enhancing the reasoning capabilities of large language models (LLMs). Beyond well-known techniques such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), emphasis is placed on the new paradigm of "**LLM Programs**," which involves integrating LLMs into traditional algorithmic structures. This approach enables efficient decomposition of complex tasks, minimizes interference between solution steps, and expands available context without significant model fine-tuning. The presented review includes analysis of advantages and limitations of existing LLM customization methods, along with a detailed description of the LLM Programs concept based on the work of Schlag et al. [[5](https://arxiv.org/abs/2305.05364  )] and related research.

### Introduction

Over the past years, significant progress has been made in developing large language models capable of performing multi-step reasoning through methods such as Chain-of-Thought. Traditional approaches to customizing LLMs can be broadly divided into two directions:

1.  **Fine-tuning** a pre-trained model, requiring substantial computational resources, large volumes of data, and appropriate infrastructure.
2.  **In-context learning**—a method focused on prompt engineering, where desired functionality is achieved through specially constructed queries and demonstration examples (including those employing CoT). However, this approach is limited by available context volume and may encounter problems of interference between different reasoning stages.

Consequently, there is a need to develop new methodologies capable of combining the advantages of in-context learning while avoiding its limitations.

### Transition to the LLM Programs Paradigm

#### Motivation and Conceptual Foundations

Modern methods based solely on prompt engineering often fail to effectively manage multi-step reasoning processes. To address this challenge, it is proposed to integrate LLMs into classical algorithmic programs. Within the **Large Language Model Programs** paradigm, the LLM is used to solve individual subtasks, while external code (e.g., written in Python) manages state and step sequence. This approach enables:

- **Task Decomposition:** Breaking the task into a series of logically independent steps, each addressed with a specialized query.
- **Increased Context Availability:** Separating information across steps prevents overload of a single query with excessive data.
- **Enhanced Interpretability:** Each solution stage has clearly defined inputs and outputs.
- **Reduced Fine-tuning Requirements:** The model performs local subtasks and does not bear responsibility for maintaining global state.

#### Technical Implementation

Unlike methods where the LLM itself maintains state (e.g., systems with external tools like Toolformer or LaMDA), in LLM Programs, primary control is shifted to the programmatic level. Key elements of this approach include:

- **Solution Decomposition:** The task is divided into a sequence of logically independent steps, each solved separately.
- **Parsing and State Assembly:** Results from each step are analyzed, and relevant data is saved to form the next query.
- **Specialized Prompts per Step:** Each query is formulated using only information relevant to the specific stage, minimizing interference between steps.

#### Advantages of the Approach

The LLM Programs approach offers several significant advantages over traditional methods:

- **Minimized Fine-tuning Requirements:** The model requires no significant additional training, as external program manages context.
- **Ability to Describe Complex Algorithmic Tasks:** Task decomposition allows precise specification of input and output data for each step.
- **Enhanced Interpretability and Debugging:** Explicit separation of solution stages simplifies testing, debugging, and quality assessment.
- **Expanded Context Availability:** Distributing information across steps avoids query overload, positively impacting generation quality.

### Example Application: Evidence-Based Question Answering Systems

In the work by Schlag et al. [[5](https://arxiv.org/abs/2305.05364  )], an example of a question-answering system designed for complex multi-step reasoning is presented. The system is divided into two main components:

1.  **Filtering Relevant Facts:** From multiple sources, paragraphs most likely containing the answer to the given question are selected, using likelihood evaluation.
2.  **Tree-Based Search of Reasoning Chains:** For each step, alternative reasoning variants are generated using different paragraphs as context. The most consistent chain is then selected via majority voting.

Results demonstrate improved accuracy compared to baseline models using standard Chain-of-Thought.

### Brief Overview of the Paper "Large Language Model Programs"

The paper "Large Language Model Programs" (Schlag et al., 2023) [[5](https://arxiv.org/abs/2305.05364  )] proposes a methodology for integrating LLMs into algorithmic programs to expand system capabilities without significant fine-tuning. The paper's key propositions can be summarized as follows:

- **Limitations of Traditional LLMs:** Difficulties in demonstrating algorithmic abilities (e.g., sorting, searching) and generalization problems caused by the finite context size of Transformer architectures.
- **Alternative Approach: LLM Programs:** Instead of the LLM maintaining global state, on each step it is provided with a narrowly specialized prompt containing context relevant only to that specific stage.
- **Advantages of LLM Programs:**  
  - Expansion of theoretical and practical system capabilities with minimal or no fine-tuning.
  - Incorporation of algorithmic information through decomposition of complex tasks into simple subtasks.
  - Improved interpretability, testability, and controllability of the system.
- **Application Examples:**  
  - Evidence-based question-answering systems, where the system first filters relevant facts and then performs a tree-based search of reasoning chains.
  - Tasks of extracting rules from natural language, recursive text summarization, robot action planning, and integration with external tools (e.g., calculators or search engines).

The authors cite the following statements:

> *"As an alternative, we propose embedding LLMs into a program or algorithm."*  
> *"Embedding an LLM in a program can significantly expand the theoretical and practical capabilities of the system with no or little finetuning and can help the system generalise more systematically."*  
> *"In this work, we present the advantages and disadvantages of programming with LLMs and present a general approach which we call a Large Language Model Program."*

Thus, the LLM Programs methodology represents a promising direction for overcoming the limitations of large language models and expanding their functional capabilities.

### Conclusion

The review of modern approaches to enhancing LLM reasoning capabilities demonstrates that integrating language models into classical programming systems (LLM Programs) is an effective means of overcoming the limitations of both fine-tuning and in-context learning. This approach ensures more flexible state management, enables decomposition of complex tasks into simple steps, and substantially expands LLM functional capabilities without significant additional training.

<details> 
    <summary><em><strong>A few interesting examples of practical Tree-of-Thought implementations</strong></em></summary>

Beyond the conceptual developments outlined in prior research, it is worthwhile to examine concrete examples demonstrating how the Tree-of-Thought (ToT) approach can be utilized and refined in real-world tasks.

### 1. Tree-of-Thought Puzzle Solver System (Theta Labs)

In the first work, developed by a team led by Jieyi Long (Theta Labs), an architecture is proposed where a **LLM** (Large Language Model) receives input tasks as prompts and generates intermediate responses. The key component of the system is a specialized **prompter agent**—a module that receives the user’s initial query. The prompter agent’s task is to formulate prompts to the LLM that **do not require an immediate final solution**, but instead facilitate the collection of intermediate reasoning results.

The intermediate responses generated by the LLM are validated using a **checker module**. If an intermediate solution is deemed correct, it is **parsed** and stored in an **internal memory module**. In the case of invalid or contradictory generation, a backtracking process is triggered: the **ToT controller** instructs the prompter agent to modify the prompt and request a more acceptable solution from the LLM. When necessary, the system can backtrack not only to the parent node of the reasoning tree but also to earlier states if the current search branch proves unsuccessful.

In this setup, the **LLM** handles “short-range reasoning”—local logical inference steps—while the ability to return to prior intermediate states enhances the system’s capacity for “long-range reasoning” and expands the space of potential solutions. Moreover, multi-step interaction increases the number of computational steps available to the system, thereby deepening the search.

- The **checker module** can be based on explicitly coded rules (e.g., for logical tasks, 3SAT, or equation solving) or on additional neural networks when tasks require more flexible correctness evaluation.
- The **memory module** stores the entire history of dialogue between the LLM and the prompter agent, enhancing transparency and facilitating analysis.
- The **ToT controller** monitors the entire tree-structured search. It can be implemented as a set of hard-coded rules (e.g., backtracking to the parent if a branch yields no result for too long) or as a trainable **policy network**.
- The **prompter agent** generates adaptive “hints” for the LLM, adjusting dynamically to the progress and validation status of the solution.

Within this system, the authors also applied the **REINFORCE algorithm** to train the policy network, suggesting that in the future, more advanced methods (e.g., multi-agent reinforcement learning—MARL) may be employed. Analogous to AlphaGo, the model can refine its search strategy through iterative interactions and self-learning.

The system was tested on simplified variants of **Sudoku** (sizes from 3×3 to 5×5), where the ToT approach with a trainable controller demonstrated higher efficiency compared to zero-shot, one-shot, and few-shot generations based on classical Chain-of-Thought. Code and examples are available in the open repository [GitHub: tree-of-thought-puzzle-solver](https://github.com/jieyilong/tree-of-thought-puzzle-solver  ).

### 2. Research from Princeton and Google DeepMind

In a second work, conducted by a team of authors from Princeton and Google DeepMind, a similar perspective on implementing Tree-of-Thought is presented. Similar to prior research, the **LLM** here also serves as a heuristic for solution search, with each tree node corresponding to one “thought”—an intermediate step in the reasoning process.

The authors emphasize that to create an effective ToT implementation, four key questions must be addressed:

1. **Decomposing the solution process into thoughts:** Finding the optimal “size” of a thought so the model generates useful ideas while preserving diversity and semantic meaningfulness of generated hypotheses.
2. **Generating candidates for the next step:** Either perform independent sampling (i.i.d. sampling) of multiple thoughts using a CoT prompt, or iteratively request sequential alternatives via “propose prompt.”
3. **Heuristic evaluation of intermediate states:** Two mechanisms are proposed—individually evaluating each state with a specialized prompt, or generating multiple states simultaneously and applying a voting procedure to select the most promising candidate.
4. **Search algorithm:** Classical methods are considered: Depth-First Search (DFS) and Breadth-First Search (BFS), with the choice depending on the specific task and available computational resources.

For empirical validation of the ToT methodology, the following tasks were selected:

- **Game of 24** (arithmetic puzzle),  
- **Creative Writing**,  
- **Solving Mini Crosswords (5×5)**.

Experiments used the GPT-4 model, and across all tasks, the authors noted significant superiority of ToT over classical Input-Output approaches, as well as over Chain-of-Thought (CoT) and even Self-Consistency CoT (CoT-SC). The implementation repository is available at [GitHub: tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm  ).

Despite certain differences in formal framing and details, both works demonstrate the fundamental idea: Tree-of-Thought can be regarded as an **extension** of standard CoT, integrating mechanisms for nonlinear search, backtracking, and validation of intermediate hypotheses. Such systems effectively approach what is sometimes termed **LLM Programs**, where external logic (controller, validation modules, managed memory) assumes coordination of reasoning, while the language model itself solves local subtasks and generates candidate solution paths.

A distinct development direction for ToT involves projects exploring expansion of search into more complex structures (e.g., **Graph of Thoughts** [[arXiv:2308.09687](https://arxiv.org/abs/2308.09687  )]). This reflects the research community’s continuous movement toward more flexible schemes for managing large numbers of intermediate reasoning steps.  

---

Thus, contemporary research clearly confirms the high effectiveness of Tree-of-Thought and related approaches in solving non-standard and complex tasks requiring branching reasoning processes. The advancement of this direction offers hope that in the foreseeable future, even more sophisticated systems capable of deeply structured multi-step reasoning and autonomous search planning will be developed.

</details>

---

# 3. Test-time Compute: A New Dimension of Scaling Language Models

In the context of the LLM Programs paradigm, which unlocks new possibilities for managing reasoning processes, another important direction emerges—**Test-time compute**, representing a revolutionary approach to scaling language models.

### Evolution of LLM Scaling: From Training to Inference

Traditionally, scaling large language models (LLMs) focused on the training stage. Increasing model size, training data volume, and computational resources for training was the primary means of improving performance. However, with the emergence of models like OpenAI o1, a new era has opened—the era of “Test-time compute,” proposing scaling during inference.

### Essence of Test-time Compute

“**Test-time compute**” (computation during testing/inference) is a scaling paradigm for LLMs that emphasizes increasing computational resources available to the model directly at the moment of processing a user query (inference time). Unlike the traditional approach, “Test-time compute” allows enhancing the performance of a pre-trained model by providing it with more time and computational power to “think” about each specific query.

### Difference from Traditional Scaling

Traditional LLM scaling focused on the following aspects **during training**:

* **Model size:** Increasing the number of parameters and architectural complexity.
* **Data volume:** Expanding and diversifying training data.
* **Training computational resources:** Utilizing more powerful GPUs and increasing training time.

“Test-time compute” introduces an **additional scaling dimension**, applied **after model training**. This allows improving model efficiency without altering its architecture or parameters, by optimizing computational resources at inference time.

### Mechanism and Advantages of Test-time Compute

Providing the model with greater computational resources during inference enables:

* **Deeper query processing:** The model can conduct more detailed analysis of input text and context.
* **Improved reasoning:** Additional computation facilitates more effective planning, search for optimal solutions, and generation of logically grounded answers.
* **Use of complex inference algorithms:** Enables application of resource-intensive but higher-quality decoding and generation methods.

### In Summary

“Test-time compute” marks a significant shift in approaches to scaling LLMs. It complements traditional methods by focusing on optimizing computational resources at the moment of model usage. This opens prospects for creating more intelligent and reasoning-oriented language models, particularly in tasks requiring deep analysis and logical deduction.

### DeepSeek-R1: Utilizing Test-time Compute and Reinforcement Learning for Reasoning

Within the context of the Test-time compute era, the DeepSeek-R1 model exemplifies the application of this approach to enhance LLM reasoning capabilities. Moreover, DeepSeek-R1 demonstrates that reasoning capabilities can be trained not only via Supervised Fine-Tuning (SFT) on large datasets but also effectively achieved through large-scale Reinforcement Learning (RL).

The primary achievement of DeepSeek-R1, analogous to AlphaZero, is demonstrating that extensive datasets for SFT are not required to train reasoning capabilities. These capabilities can be efficiently acquired through large-scale Reinforcement Learning (RL), significantly reducing dependence on “human demonstrations” in the form of SFT. Nonetheless, using a small volume of high-quality SFT examples can aid in a more efficient “cold start” of training.

As the base model for DeepSeek-R1, DeepSeek-V3-Base was selected—a model after pre-training but before post-training, i.e., without SFT and RL. The RL algorithm applied was Group Relative Policy Optimization (GRPO), previously used in DeepSeek-V3 and DeepSeekMath, which avoids the need for a separate critic model.

# 4. Technical Details of DeepSeek-V3 and Multi-Head Latent Attention (MLA)

To gain a deeper understanding of the architectural features of DeepSeek-R1, it is essential to examine the technical details of its foundational model, DeepSeek-V3. DeepSeek-V3 represents a significant advancement in LLM development, combining the classical decoder-transformer architecture with elements of Mixture-of-Experts (MoE) and innovative attention mechanisms such as Multi-Head Latent Attention (MLA).

### Overview of DeepSeek-V3 Architecture

According to the DeepSeek-V3 technical report [[6](https://arxiv.org/abs/2412.19437  )] and the [GitHub repository](https://github.com/deepseek-ai/DeepSeek-V3  ), the model is a decoder-transformer with a Mixture-of-Experts (MoE) architecture. DeepSeek-V3 contains 671 billion parameters, of which 37 billion are active per token. The model consists of 61 transformer layers with a hidden dimension $d_h=7168$.

![Table_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_1.jpg  )

Several interesting technical solutions were applied in the development of DeepSeek-V3, historically significant in the context of LLM evolution. Some of these solutions, including MLA, were tested and refined in the previous model version, DeepSeek-V2 [[7](https://arxiv.org/abs/2405.04434  )].

### Multi-Head Latent Attention (MLA)

One of the key innovations in DeepSeek-V3 is **Multi-Head Latent Attention (MLA)**. This mechanism aims to enhance model efficiency and scalability, particularly in tasks requiring processing of long sequences and complex reasoning. To understand MLA, consider first the classical Multi-Head Attention (MHA) mechanism.

<details> 
    <summary><em><strong>Classical Multi-Head Attention (MHA) in the Transformer Decoder🤖</strong></em></summary>

#### 1. Key Role of the Decoder in Transformer

In a classic **autoregressive** sequence generation task (e.g., machine translation), the decoder performs the function of **step-by-step (conditional) formation** of the output sequence. It “looks at” the encoder’s output to incorporate context from the input sentence (in translation) while simultaneously computing probabilities for the next token based on the **partially generated** sequence.

Schematically, in the original "Attention Is All You Need" paper, the decoder is located on the **right**, receiving:
1. **Its own inputs** (for language tasks—“shifted right” tokens of previous words).
2. **The encoder’s output** (context obtained from processing the input sequence).

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Figure_1.png  )

#### 2. Decoder Architecture

Each decoder layer (Decoder Layer) consists of the following sub-blocks:

1. **Masked Multi-Head Attention**  
   - Nearly identical to standard Multi-Head Attention but with **masking of future positions** (to prevent the model from “peeking” at tokens it cannot yet see during autoregression).

2. **Add & Norm**  
   - Residual connection and Layer Normalization, analogous to the encoder.

3. **Multi-Head Attention (Cross-Attention)**  
   - Attention mechanism with queries (Q) from the **current decoder state** and keys (K) and values (V) from the **encoder’s output** (i.e., the decoder learns to extract relevant information from contextual embeddings produced by the encoder).

4. **Add & Norm**  
   - Residual connection and Layer Normalization.

5. **Feed Forward (FFN)**  
   - A two-layer fully connected network with an activation function, analogous to the encoder module.

6. **Add & Norm**  
   - Residual connection and Layer Normalization.

> As in the encoder, these six stages repeat multiple times (e.g., 6 decoder layers), forming a **deep** model.

#### 3. Decoder Input (shifted right)

In **text generation tasks** (e.g., machine translation), the decoder, at each step, seeks to predict the **next token**, using previously generated tokens. To prevent the model from **seeing future tokens**, the decoder’s input sequence is typically shifted one token to the right (shifted right).  

- For example, in a translation task, the target sentence is used as the “correct output”:
  ```
  [BOS] I love cats . [EOS]
  ```
- The input to the decoder (X_dec_inp) is the “shifted right” version:
  ```
  [BOS] I love cats .
  ```
  The final token `[EOS]` is not fed, as it is unnecessary for prediction.

- This enables the autoregressive scheme:  
  *At step i, the model does not see tokens (i+1, i+2, ...); it learns to predict the i-th token using only previous ones.*  

#### 4. Masked Multi-Head Self-Attention (Masked Multi-Head Attention)

##### 4.1 Motivation  
Unlike the encoder, where Self-Attention can observe **all** positions in the sequence, the decoder **masks** future tokens to prevent the model from “cheating” and “looking ahead.”

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_2.png  )

##### 4.2 Mathematical Formula  
Practically, this is the same Multi-Head Attention as in the encoder, but when computing  

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V
$$

*   **Q** (Query) – **Query matrix**. In the context of **Masked Multi-Head Attention** in the Transformer decoder, queries (Q) come from the **decoder’s input embeddings** (after passing through a linear layer). For each token in the decoder’s input sequence, a query vector is formed. In essence, queries represent what the current decoder position “looks at” when computing attention.

*   **K** (Key) – **Key matrix**. In **Masked Multi-Head Attention** in the decoder, keys (K) also come from the **decoder’s input embeddings** (after passing through another linear layer). Keys represent the information against which queries are compared to determine relevance. They correspond to positions in the decoder’s input sequence that the model attends to.

*   **V** (Value) – **Value matrix**. In **Masked Multi-Head Attention** in the decoder, values (V) also come from the **decoder’s input embeddings** (after passing through another linear layer). Values represent the information that will be aggregated in a weighted manner based on the attention weights calculated from query-key comparisons. It is these values that are “summed” with the weights obtained from query-key comparisons to form the attention output representation.

*   $d_k$ – **Key vector dimension** (and query dimension, as in Self-Attention dimensions are typically equal).  $\sqrt{d_k}$ is used in the denominator for **scaling** to prevent softmax saturation, especially with large $d_k$. This helps stabilize training.

**In brief, in Masked Multi-Head Attention in the decoder:**

*   **Q, K, V** all originate from the **same decoder input sequence** (embeddings with future position masking applied).
*   The attention mechanism allows each position in the decoder to weighingly consider other positions in the **preceding part** of the decoded sequence (due to masking).

we **zero out (or set to -∞) those positions** that should not yet be visible to the current token. This results in a **triangular mask** for language, where position i cannot see positions i+1, i+2, …  

```python
# Example of generating a triangular mask in PyTorch (L - sequence length)
import torch
from typing import Tuple

def subsequent_mask(size: int) -> torch.Tensor:
    """
    Description:
        Creates a mask that prohibits connections to future positions.
        Output has shape [size, size] with True/False values:
          True  - positions where attention is allowed
          False - positions where attention is forbidden (future tokens)

    Args:
        size (int): Size of the mask (sequence length).

    Returns:
        torch.Tensor: Triangular mask of shape [size, size] with True/False values.

    Examples:
        >>> mask = subsequent_mask(5)
        >>> print(mask)
        tensor([[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True]])
    """
    # Triangular matrix of ones on the lower triangle (including diagonal)
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask

# Example usage
L = 5
mask = subsequent_mask(L)
print("Generated triangular mask:\n", mask)

# Example of applying mask to softmax
def apply_softmax_with_mask(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Applies softmax with mask to the result of Q and K scalar product.

    Args:
        Q (torch.Tensor): Query tensor.
        K (torch.Tensor): Key tensor.
        V (torch.Tensor): Value tensor.
        mask (torch.Tensor): Mask to apply.

    Returns:
        torch.Tensor: Result of applying softmax with mask.

    Examples:
        >>> Q = torch.randn(5, 5)
        >>> K = torch.randn(5, 5)
        >>> V = torch.randn(5, 5)
        >>> mask = subsequent_mask(5)
        >>> result = apply_softmax_with_mask(Q, K, V, mask)
        >>> print(result)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    # mask == False -> zero out/set to -inf
    scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    Z = attn_weights @ V
    return Z
```

#### 5. Cross-Attention Mechanism at the Decoder Output

After the Masked Multi-Head Attention layer and subsequent Add & Norm, the decoder has a more "fresh" representation of the currently partially generated sequence. Next comes the **Multi-Head Attention**, where:

1. **Q** (queries) are taken from the **current decoder hidden states** (after Masked MHA).  
2. **K** (keys) and **V** (values) are taken from the **encoder output**.  

Thus, the decoder "queries" the necessary information from the encoder's output embeddings, which already encode the entire context of the input sequence.  

The formulas remain the same:  

$$
\text{CrossAttention}(Q_\text{dec}, K_\text{enc}, V_\text{enc}) = \text{softmax}\left(\frac{Q_\text{dec}K_\text{enc}^T}{\sqrt{d_k}}\right) V_\text{enc}
$$

#### 6. Residual Connections (Add) and Normalization (Norm)

Similar to the encoder, **after each** sublayer comes the Add & Norm step:

1. **Add (Residual Connection):**  
   
   $$
   \text{Add}_\text{dec} = \text{Input}_\text{dec-sublayer} + \text{Output}_\text{dec-sublayer}
   $$

2. **LayerNorm:**  
   $$
   \text{Norm}_\text{dec} = \text{LayerNorm}(\text{Add}_\text{dec})
   $$

This helps stabilize training, improves gradient flow, and enables learning deeper networks.

#### 7. Feed Forward Network (FFN) and Final Add & Norm

Similarly to the encoder, the **FFN** in the decoder consists of two linear layers with an activation function (ReLU/GELU). The formula is:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2
$$

This is followed by a **residual connection** + **Layer Normalization**.

> Let’s reiterate: the decoder layer structure is as follows:  
> 1) Masked Multi-Head Attention → Add & Norm  
> 2) (Cross) Multi-Head Attention (on encoder output) → Add & Norm  
> 3) Feed Forward → Add & Norm  

#### 8. Final Linear Layer and Softmax (for Next Token Prediction)

In the task of **text generation**, after passing through all decoder layers, we obtain hidden states. Then, a **Linear layer (Projection)** maps these to the vocabulary size, followed by **Softmax** to convert them into token probabilities. Typically, this is done via a separate "Output Projection" layer:

$$
\hat{y}_t = \text{softmax}(\text{DecoderOutput}_t \cdot W_\text{out} + b_\text{out})
$$

where $(\hat{y}_t)$ is the probability distribution over the vocabulary at time step $(t)$.

### Let’s examine the process of selecting the next token in more detail:

**Decoder output as a contextual representation:**

At each generation step, the decoder (e.g., in a Transformer architecture) processes the input sequence and previously generated tokens. The result of the decoder’s operation at time step $t$ is a **hidden state vector** $DecoderOutput_t$. This vector is a **compressed representation of all context** the model has accounted for up to this point. It "knows" about the beginning of the sentence, previously generated words, and—in seq2seq models—about the input sequence (e.g., during translation). The dimensionality of this vector is determined by the model architecture and is a hyperparameter.

**I. Hidden state vector $DecoderOutput_t$ as input for prediction:**

As we established, the vector $DecoderOutput_t$ is the output of the **final decoder block** at time step $t$. Assume each decoder block has the same internal dimensionality, denoted as $D_{model}$ (e.g., 512 or 768 in the original Transformer). Thus, $DecoderOutput_t$ is a vector of dimension $D_{model}$:

$$
DecoderOutput_t \in \mathbb{R}^{D_{model}}
$$

This vector $DecoderOutput_t$ contains compressed information about the context accumulated by the decoder up to time $t$.

**II. Linear Layer (Projection):**

The purpose of the linear layer is to transform the contextual representation vector $DecoderOutput_t$ into a vector of **logits**, whose dimension equals the vocabulary size $V$. Let the vocabulary size be $|V|$.

The linear layer is implemented using a weight matrix $W_{out}$ and a bias vector $b_{out}$.

* **Weight matrix $W_{out}$:** This matrix performs a linear transformation and has dimensions $(|V| \times D_{model})$:

$$
W_{out} \in \mathbb{R}^{|V| \times D_{model}}
$$

* **Bias vector $b_{out}$:** This vector is added after matrix multiplication and has dimensions $(|V|)$:

$$
b_{out} \in \mathbb{R}^{|V|}
$$

**Linear transformation operation:** The logits vector, denoted as $Logits_t$, is computed as follows:

$$
Logits_t = DecoderOutput_t \cdot W_{out}^T + b_{out}
$$

Here:

* $DecoderOutput_t$ — hidden state vector of dimension $(1 \times D_{model})$ (represented as a row vector for convenient matrix multiplication).
* $W_{out}^T$ — transposed weight matrix $W_{out}$ of dimension $(D_{model} \times |V|)$.
* $Logits_t$ — logits vector of dimension $(1 \times |V|)$. Each element $Logits_{t, i}$ corresponds to the $i$-th token in the vocabulary.

**Explanation:**

* Multiplication $DecoderOutput_t \cdot W_{out}^T$ effectively computes a weighted sum of elements in the vector $DecoderOutput_t$. Each row of matrix $W_{out}^T$ (or column of $W_{out}$) corresponds to one token in the vocabulary. Thus, a score is computed for each vocabulary token based on the contextual vector $DecoderOutput_t$.
* Adding the bias vector $b_{out}$ allows the model to independently shift scores for specific tokens regardless of the input vector.

**III. Softmax Function — Converting Logits to Probabilities:**

After obtaining the logits vector $Logits_t$, we apply the Softmax function to convert them into a probability distribution over all vocabulary tokens.

For each element $Logits_{t, i}$ in the logits vector, Softmax computes the probability $P(\text{token}_i | \text{context})$ as follows:

$$
P(\text{token}_i | \text{context}) = \frac{\exp(Logits_{t, i})}{\sum_{j=1}^{|V|} \exp(Logits_{t, j})}
$$

where:

* $Logits_{t, i}$ — the $i$-th element of the logits vector $Logits_t$, corresponding to the $i$-th token in the vocabulary.
* $|V|$ — vocabulary size.
* $\exp(x)$ — exponential function.
* $\sum_{j=1}^{|V|} \exp(Logits_{t, j})$ — sum of exponentials of all logits, used for normalization.

**Softmax Result:** A probability vector $\hat{y}_t$ of dimension $(1 \times |V|)$, where each element $\hat{y}_{t, i} = P(\text{token}_i | \text{context})$ represents the probability that the $i$-th token from the vocabulary is the next token in the sequence, given the context represented by $DecoderOutput_t$.

$$
\hat{y}_t = \text{softmax}(Logits_t) \in \mathbb{R}^{|V|}
$$

**IV. Selecting the Next Token (as before):**

Based on the resulting probability distribution $\hat{y}_t$, we select the next token using one of the strategies, such as argmax, sampling, top-k sampling, or nucleus sampling.

**Connection to Decoder Architecture:**

It is crucial to understand that the vector $DecoderOutput_t$, fed into the linear layer, is the result of complex processing of inputs through **all sublayers of the decoder block**:

1. **Masked Multi-Head Attention:** Allows the decoder to attend to previously generated tokens, forming a contextual representation via self-attention.
2. **(Cross) Multi-Head Attention:** Enables the decoder to attend to the **input sequence** (if an encoder exists), focusing on relevant information from the encoder output.
3. **Feed Forward Network:** Adds **non-linearity** and allows the model to process information from attention layers in a more complex manner.
4. **Add & Norm:** **Residual connections and Layer Normalization** stabilize training and improve gradient flow, enabling the construction of deeper and more efficient decoders.

Thus, $DecoderOutput_t$ is not merely a random vector but a **high-level, context-dependent representation** derived from the complex architecture of the decoder. The linear layer and Softmax are the final steps that transform this abstract representation into a concrete prediction of the next token as a probability distribution over the vocabulary.

Understanding these steps allows for a better grasp of how text generation models work and how various parameters and strategies influence the quality and diversity of generated text.

</details> 

#### Innovation: Multi-Head Latent Attention (MLA): Low-Rank Compression for Keys and Values

MLA introduces **low-rank compression** for keys and values. In MLA, the input token embedding ($h_t$) is first projected into a **low-rank latent vector ($c_t$)**. This vector is then expanded back into key ($k_t$) and value ($v_t$) vectors via separate matrices ($W_{uk}, W_{uv}$). Crucially, the dimension of the latent vector ($d_c$) is significantly smaller than the dimension of the split key and value vectors after head separation ($d_h \times n_h$).

During inference, MLA enables **reduction of the KV-cache size**, since only the low-dimensional latent vectors ($c_t$) are cached, rather than full-dimensional key vectors ($k_t$) as in standard MHA. This reduction in computational cost is particularly important for efficient test-time compute, allowing the model to "think" longer under limited resources.

**Let’s sequentially unpack each aspect of MLA, providing mathematical formalization, explanations, and code.**

**1. Low-Rank Compression of Keys and Values**

**1.1. Engineering and Mathematical View of Low-Rank KV Compression**

**Engineering Aspect:**

In standard Multi-Head Attention (MHA), for each attention head, the input token embedding $h_t \in \mathbb{R}^{d_{model}}$ is projected into three vectors: query $q_t$, key $k_t$, and value $v_t$, each of dimension $d_h = d_{model} / n_h$, where $n_h$ is the number of heads. During inference, keys and values for all previous tokens are cached in the KV-cache. The size of this cache grows linearly with sequence length, which can become a bottleneck for long sequences.

**KV-cache (Key-Value Cache)** — a mechanism used in transformer-based models to accelerate inference by storing computed keys (Keys) and values (Values) for all previous tokens in the sequence. This avoids recomputing keys and values for already processed tokens during generation of new tokens.

MLA addresses this issue by introducing a **low-rank representation**. Instead of directly projecting into $k_t$ and $v_t$, $h_t$ is first projected into a **low-dimensional latent vector** $c_t \in \mathbb{R}^{d_c}$, where $d_c \ll d_h$. Then, $c_t$ is projected back into $k_t$ and $v_t$. Since the KV-cache stores $c_t$ instead of $k_t$, the cache size is substantially reduced.

**Mathematical Formalization:**

1. **Projection into latent space:**
    $$
    c_t = h_t W_{uc}
    $$
    where $W_{uc} \in \mathbb{R}^{d_{model} \times d_c}$ is the projection matrix into the latent space.

2. **Expansion from latent space to keys and values:**
    $$
    k_t = c_t W_{uk} \\
    v_t = c_t W_{uv}
    $$
    where $W_{uk} \in \mathbb{R}^{d_c \times d_h}$ and $W_{uv} \in \mathbb{R}^{d_c \times d_h}$ are projection matrices from the latent space to the key and value spaces, respectively.

**Dimensions:**

* $h_t \in \mathbb{R}^{d_{model}}$ — input token embedding
* $c_t \in \mathbb{R}^{d_c}$ — low-dimensional latent vector ($d_c \ll d_h$)
* $k_t, v_t \in \mathbb{R}^{d_h}$ — key and value vectors for one attention head
* $W_{uc} \in \mathbb{R}^{d_{model} \times d_c}$ — projection matrix into latent space
* $W_{uk} \in \mathbb{R}^{d_c \times d_h}$ — projection matrix from latent space to keys
* $W_{uv} \in \mathbb{R}^{d_c \times d_h}$ — projection matrix from latent space to values

**1.2. Python Code Example (PyTorch):**

```python
from typing import Tuple
import torch
import torch.nn as nn

class MLALinearProjection(nn.Module):
    """
    Description:
        Class for linear projection in MLA model.

    Args:
        d_model: Dimension of input embedding.
        d_latent: Dimension of latent vector.
        d_head: Dimension of key and value.
    """

    def __init__(self, d_model: int, d_latent: int, d_head: int) -> None:
        super().__init__()
        self.W_uc = nn.Linear(d_model, d_latent)
        self.W_uk = nn.Linear(d_latent, d_head)
        self.W_uv = nn.Linear(d_latent, d_head)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Description:
            Performs linear projection of input token embedding.

        Args:
            h_t: torch.Tensor of shape [..., d_model] — input token embedding.

        Returns:
            k_t: torch.Tensor of shape [..., d_head] — key.
            v_t: torch.Tensor of shape [..., d_head] — value.
            c_t: torch.Tensor of shape [..., d_latent] — latent vector (for KV-cache).

        Examples:
            >>> d_model = 512
            >>> d_latent = 64
            >>> d_head = 64
            >>> batch_size = 2
            >>> seq_len = 10
            >>> projection_layer = MLALinearProjection(d_model, d_latent, d_head)
            >>> input_embeddings = torch.randn(batch_size, seq_len, d_model)
            >>> h_t = input_embeddings[:, 0, :]
            >>> k_t, v_t, c_t = projection_layer(h_t)
            >>> k_t.shape
            torch.Size([2, 64])
        """
        c_t = self.W_uc(h_t)
        k_t = self.W_uk(c_t)
        v_t = self.W_uv(c_t)
        return k_t, v_t, c_t

# Example usage
def main() -> None:
    """
    Description:
        Example usage of MLALinearProjection class.

    Examples:
        >>> main()
        Latent KV-cache dimension: torch.Size([2, 10, 64])
        Keys KV-cache dimension (for comparison): torch.Size([2, 10, 64])
    """
    d_model    = 512
    d_latent   = 64  # Low dimension
    d_head     = 64
    batch_size = 2
    seq_len    = 10

    projection_layer = MLALinearProjection(d_model, d_latent, d_head)
    input_embeddings = torch.randn(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]

    keys_list = []
    values_list = []
    latent_vectors_list = []

    for t in range(seq_len):
        h_t = input_embeddings[:, t, :]                                        # [batch_size, d_model]
        k_t, v_t, c_t = projection_layer(h_t)                                  # k_t, v_t, c_t: [batch_size, d_head]
        keys_list.append(k_t)
        values_list.append(v_t)
        latent_vectors_list.append(c_t)

    # MLA KV-cache stores latent_vectors_list (dimension d_latent)
    # instead of keys_list (dimension d_head) in standard MHA if keys were cached.
    latent_kv_cache = torch.stack(latent_vectors_list, dim=1)                 # [batch_size, seq_len, d_latent]
    print("Latent KV-cache dimension:", latent_kv_cache.shape)              # -> torch.Size([2, 10, 64])

    # For comparison, if keys were cached (as in standard MHA)
    keys_kv_cache = torch.stack(keys_list, dim=1)                             # [batch_size, seq_len, d_head]
    print("Keys KV-cache dimension (for comparison):", keys_kv_cache.shape)  # -> torch.Size([2, 10, 64])

    # In this example, d_latent and d_head are equal (64), but in MLA d_latent << d_head,
    # leading to substantial reduction in KV-cache size.

main()
```

**2. Optimization of Projection Matrices in MLA**

**2.1. Engineering and Mathematical View of Projection Matrix Optimization**

**Engineering Aspect:**

MLA employs **low-rank projection** for **queries (Q), keys (K), and values (V)**. This means the original projection matrices in standard Multi-Head Attention (MHA) are replaced by **a pair of matrices for each projection (Q, K, V): a dimensionality-reducing matrix and a dimensionality-restoring matrix**. The goal of this optimization is to **reduce computational cost and decrease KV-cache size (for K and V) and activation memory (for Q)**, while preserving model expressiveness.

**Mathematical Formalization:**

In standard MHA, queries, keys, and values are computed as:
$Q = XW^Q$, $K = XW^K$, $V = XW^V$, where $X$ is the input embeddings.

In MLA, these projections are replaced with low-rank equivalents:

1. **Low-Rank Projection for Keys and Values (KV):**

    * **Down-projection:** Input embeddings $X$ are projected into a low-dimensional latent space of rank $r$:

        $$
        C^{KV} = XW^{DKV}
        $$

        where:
         
        - $W^{DKV} \in \mathbb{R}^{d_{model} \times r}$ — down-projection matrix for KV 
        - $r \ll d_k$ (where $d_k$ is key/query dimension in standard MHA)

    * **Up-projection:** From the latent representation $C^{KV}$, keys $K$ and values $V$ are reconstructed:

        $$
        K = C^{KV}W^{UK} \\
        V = C^{KV}W^{UV}
        $$

        where:
        
        - $W^{UK} \in \mathbb{R}^{r \times d_k}$ and $W^{UV} \in \mathbb{R}^{r \times d_k}$ — up-projection matrices for keys and values, respectively.

2. **Low-Rank Projection for Queries (Q):**

    * **Down-projection:** Input embeddings $X$ are also projected into a low-dimensional latent space of rank $r$ (the compression rank may be the same or different for Q and KV):

        $$
        C^{Q} = XW^{DQ}
        $$

        where:
        
        - $W^{DQ} \in \mathbb{R}^{d_{model} \times r}$ — down-projection matrix for queries.

    * **Up-projection:** From the latent representation $C^{Q}$, queries $Q$ are reconstructed:

        $$
        Q = C^{Q}W^{UQ}
        $$

        where:
         
        - $W^{UQ} \in \mathbb{R}^{r \times d_k}$ — up-projection matrix for queries.

**Dimensions:**

* $X \in \mathbb{R}^{n \times d_{model}}$ — input embeddings (batch of $n$ tokens, each of dimension $d_{model}$)
* $C^{KV} \in \mathbb{R}^{n \times r}$ — latent representation for keys and values (rank $r$)
* $C^{Q} \in \mathbb{R}^{n \times r}$ — latent representation for queries (rank $r$, may differ)
* $K, V, Q \in \mathbb{R}^{n \times d_k}$ — reconstructed keys, values, and queries (dimension $d_k$)
* $W^{DKV} \in \mathbb{R}^{d_{model} \times r}$, $W^{UK} \in \mathbb{R}^{r \times d_k}$, $W^{UV} \in \mathbb{R}^{r \times d_k}$, $W^{DQ} \in \mathbb{R}^{d_{model} \times r}$, $W^{UQ} \in \mathbb{R}^{r \times d_k}$ — projection matrices.

**2.2. Python Code Example (PyTorch) (Corrected):**

```python
# Import standard libraries
import math

# Import third-party libraries
import torch
import torch.nn as nn

class MLALowRankProjection(nn.Module):
    """
    Description:
        Class for low-rank projection in MLA model.

    Args:
        d_model: Dimension of input vector.
        d_latent: Dimension of latent space.
        d_head: Dimension of output vector.

    Attributes:
        W_dq: Linear layer for down-projection of queries.
        W_uq: Linear layer for up-projection of queries.
        W_dkv: Linear layer for down-projection of keys and values.
        W_uk: Linear layer for up-projection of keys.
        W_uv: Linear layer for up-projection of values.
    """

    def __init__(self, d_model: int, d_latent: int, d_head: int) -> None:
        super().__init__()
        self.W_dq = nn.Linear(d_model, d_latent)  # Down-projection for queries
        self.W_uq = nn.Linear(d_latent, d_head)   # Up-projection for queries
        self.W_dkv = nn.Linear(d_model, d_latent) # Down-projection for keys and values
        self.W_uk = nn.Linear(d_latent, d_head)   # Up-projection for keys
        self.W_uv = nn.Linear(d_latent, d_head)   # Up-projection for values

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Description:
            Performs low-rank projection on input data.

        Args:
            x: Input tensor of shape [..., seq_len, d_model].

        Returns:
            Q: Query tensor of shape [..., seq_len, d_head].
            K: Key tensor of shape [..., seq_len, d_head].
            V: Value tensor of shape [..., seq_len, d_head].
            C_kv: Latent KV representation of shape [..., seq_len, d_latent].
            C_q: Latent Q representation of shape [..., seq_len, d_latent].

        Examples:
            >>> d_model = 512
            >>> d_latent = 64
            >>> d_head = 64
            >>> batch_size = 2
            >>> seq_len = 10
            >>> projection_layer = MLALowRankProjection(d_model, d_latent, d_head)
            >>> input_embeddings = torch.randn(batch_size, seq_len, d_model)
            >>> Q, K, V, C_kv, C_q = projection_layer(input_embeddings)
            >>> print("Q dimension:", Q.shape)  # -> torch.Size([2, 10, 64])
            >>> print("K dimension:", K.shape)  # -> torch.Size([2, 10, 64])
            >>> print("V dimension:", V.shape)  # -> torch.Size([2, 10, 64])
            >>> print("C_kv dimension (KV-cache):", C_kv.shape)  # -> torch.Size([2, 10, 64])
            >>> print("C_q dimension (Q activations):", C_q.shape)  # -> torch.Size([2, 10, 64])
        """
        C_kv = self.W_dkv(x)    # [..., seq_len, d_latent]
        C_q = self.W_dq(x)      # [..., seq_len, d_latent]
        K = self.W_uk(C_kv)     # [..., seq_len, d_head]
        V = self.W_uv(C_kv)     # [..., seq_len, d_head]
        Q = self.W_uq(C_q)      # [..., seq_len, d_head]
        return Q, K, V, C_kv, C_q

# Example usage
if __name__ == "__main__":
    d_model    = 512
    d_latent   = 64  # Low dimension
    d_head     = 64
    batch_size = 2
    seq_len    = 10

    projection_layer = MLALowRankProjection(d_model, d_latent, d_head)
    input_embeddings = torch.randn(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]

    Q, K, V, C_kv, C_q = projection_layer(input_embeddings)

    print("Q dimension:", Q.shape)       # -> torch.Size([2, 10, 64])
    print("K dimension:", K.shape)       # -> torch.Size([2, 10, 64])
    print("V dimension:", V.shape)       # -> torch.Size([2, 10, 64])
    print("C_kv dimension (KV-cache):", C_kv.shape)     # -> torch.Size([2, 10, 64])
    print("C_q dimension (Q activations):", C_q.shape)  # -> torch.Size([2, 10, 64])

    # In MLA, KV-cache stores C_kv (dimension d_latent), which is smaller than storing K or V (dimension d_head)
    # if d_latent < d_head. Similarly, Q activations can be reduced by using C_q during training, if feasible.
```

**3. Low-Rank Compression of Queries**

**3.1. Engineering and Mathematical View of Low-Rank Compression of Queries**

**Engineering Aspect:**

In DeepSeek-V3, in addition to low-rank compression of keys and values, **low-rank compression of queries** $q_t$ is also applied. Unlike KV compression, query compression **does not affect the KV-cache size**, since queries are not cached. The primary goal of query compression is to **reduce memory requirements for activations during training**. Reducing the size of intermediate activations enables training models with larger batch sizes or larger models under limited GPU resources.

**4. Decoupled RoPE Strategy for Positional Embeddings**

**4.1. Engineering and Mathematical View of Decoupled RoPE Strategy**

**Engineering Aspect and RoPE Incompatibility Problem:**

Rotary Positional Embeddings (RoPE) is a method for adding positional information to Transformers that uses rotation matrices to encode relative token positions. RoPE is applied directly to query and key vectors.

<details> 
    <summary><em><strong>Mechanism of Rotary Positional Embeddings (RoPE)</strong></em></summary>

#### Mechanism of Rotary Positional Embeddings (RoPE)

**Why are Rotary Positional Embeddings (RoPE) needed?** While traditional positional encodings (PE) add static positional information to token embeddings, Rotary Positional Embeddings (RoPE) represent an **alternative and more flexible method** for embedding positional information into the Transformer architecture. RoPE were developed to address certain limitations of standard PE and to improve the model's ability to handle **relative token positions** in sequences, which is crucial for tasks where order and token relationships play a key role.

Like standard PE, RoPE are necessary because the Self-Attention mechanism in Transformers processes all tokens in parallel and **lacks inherent understanding** of token order in a sequence. RoPE introduce positional information in a way that **naturally integrates** into the attention mechanism, influencing interactions between queries and keys and encoding relative positions directly into attention vectors.

**How do Rotary Positional Embeddings (RoPE) work?**

RoPE apply a **rotational transformation** to query ($q$) and key ($k$) vectors in the attention mechanism, depending on their absolute position in the sequence. The core idea is to encode positional information via **rotation of vectors in subspaces**, enabling efficient modeling of relative positions.

Mathematically, RoPE are implemented as follows:

1. **Splitting dimensions into pairs:** The query vector $q$ and key vector $k$ (of dimension $d_k$) are split into pairs of dimensions. For each pair of dimensions $2i$ and $2i+1$ (where $i = 0, 1, 2, ..., d_k/2 - 1$), a rotation is applied.

2. **Rotation matrix:** For each position $pos$ in the sequence and for each dimension pair $(2i, 2i+1)$, a rotation angle $\theta_{pos} = pos \cdot \theta_0$ is defined, where $\theta_0$ is the base frequency (typically chosen as $10000^{-2i/d_k}$, similar to PE). The 2D rotation matrix $R_{\theta_{pos}}$ in the subspace $(2i, 2i+1)$ has the form:

   $$
   R_{\theta_{pos}} = \begin{pmatrix}
   \cos \theta_{pos} & -\sin \theta_{pos} \\
   \sin \theta_{pos} & \cos \theta_{pos}
   \end{pmatrix}
   $$

3. **Applying rotation:** For query vector $q = [q_0, q_1, ..., q_{d_k-1}]$ and key vector $k = [k_0, k_1, ..., k_{d_k-1}]$, RoPE is applied pairwise to dimensions:

   For even dimensions $2i$:
   $$
   q'_{2i} = q_{2i} \cos \theta_{pos} - q_{2i+1} \sin \theta_{pos} \\
   k'_{2i} = k_{2i} \cos \theta_{pos} - k_{2i+1} \sin \theta_{pos}
   $$
   For odd dimensions $2i+1$:
   $$
   q'_{2i+1} = q_{2i} \sin \theta_{pos} + q_{2i+1} \cos \theta_{pos} \\
   k'_{2i+1} = k_{2i} \sin \theta_{pos} + k_{2i+1} \cos \theta_{pos}
   $$

   In matrix form, for each dimension pair $(2i, 2i+1)$, this can be represented as multiplying a 2x1 sub-vector by the rotation matrix $R_{\theta_{pos}}$. This is applied to all dimension pairs in $q$ and $k$ for position $pos$.

4. **Combining rotated vectors:** After applying rotation to each dimension pair, the rotated components $q' = [q'_0, q'_1, ..., q'_{d_k-1}]$ and $k' = [k'_0, k'_1, ..., k'_{d_k-1}]$ form the query and key vectors with positional encoding.

**Why use rotations? Advantages of RoPE:**

*   **Efficient encoding of relative positions:** RoPE are inherently well-suited for encoding relative positions. The dot product between two RoPE-encoded vectors depends only on the *relative distance* between their positions. This property is a key advantage of RoPE, enabling the model to effectively capture dependencies based on token distances.

*   **Improved extrapolation to long sequences:** Due to the rotational mechanism and relative position encoding, RoPE demonstrate better extrapolation capabilities to sequences longer than those used during training compared to standard PE.

*   **Flexibility and integration into attention mechanism:** RoPE are directly integrated into the attention mechanism, modifying interactions between queries and keys. This allows positional information to influence attention weights and thus the formation of contextualized representations.

*   **Efficient implementation potential:** RoPE computations can be implemented efficiently, especially on hardware, due to the use of trigonometric functions and matrix operations.

**Generating positional information through rotation:**

RoPE generate positional information by **rotating query and key vectors in 2D subspaces**. The rotation angle depends on the token's position, ensuring a unique transformation for each position. Crucially, rotation is applied pairwise to dimensions, preserving the original vector dimensions and efficiently encoding positional information.

**Integration into Transformer architecture:**

RoPE are **not added** to input embeddings like standard PE. Instead, RoPE are **applied directly to query and key vectors** in each Multi-Head Attention layer. This means positional information is introduced at the attention mechanism level, affecting how the model interacts with different sequence positions. The dimensionality of query and key vectors remains unchanged after applying RoPE.

**In summary, RoPE:**

*   **Are not added to embeddings, but applied to Q and K.**
*   **Encode position via rotation of vectors in subspaces.**
*   **Effectively model relative positions.**
*   **Improve extrapolation to long sequences.**
*   **Are directly integrated into the attention mechanism.**

</details> 

---

The problem arises when **combining RoPE with low-rank KV compression**. If RoPE is applied after low-rank compression and expansion, positional information may be "blurred" or insufficiently integrated due to the low-rank representation. To solve this issue, MLA introduces a **decoupled RoPE strategy**.

**Decoupled RoPE Strategy:**

The decoupled RoPE strategy introduces **additional multi-head queries ($q_R$) and shared keys ($k_R$)**, which are **specialized for processing RoPE positional information**. These vectors $q_R$ and $k_R$ have their own dimension $d^R_h$. RoPE is applied **only to $q_R$ and $k_R$**.

The final query ($Q$) and key ($K$) vectors for the attention mechanism are formed by **concatenating** vectors derived from low-rank representations ($c_t$) and RoPE vectors ($q_R, k_R$).

**Mathematical Formalization:**

1.  **Computing low-rank vectors:**
    $$
    c_t = h_t W_{uc}
    $$

2.  **Projection for RoPE vectors:**
    $$
    q_R = h_t W_{qR} \\
    k_R = h_t W_{kR}
    $$
    where $W_{qR} \in \mathbb{R}^{d_{model} \times d^R_h}$ and $W_{kR} \in \mathbb{R}^{d_{model} \times d^R_h}$ are projection matrices for RoPE queries and keys.

3.  **Applying RoPE to $q_R$ and $k_R$:**
    $$
    \tilde{q}_R = \text{RoPE}(q_R, \text{position}) \\
    \tilde{k}_R = \text{RoPE}(k_R, \text{position})
    $$
    where $\text{RoPE}(\cdot, \text{position})$ is the function applying Rotary Positional Embeddings, dependent on token position.

4.  **Expanding low-rank vector for main query and key parts:**
    $$
    q_L = c_t W_{uq} \\
    k_L = c_t W_{uk}
    $$
    where $W_{uq} \in \mathbb{R}^{d_c \times d^L_h}$ and $W_{uk} \in \mathbb{R}^{d_c \times d^L_h}$. Here $d^L_h$ is the dimension of the "low-rank" part of queries and keys. Crucially, the total head dimension $d_h = d^L_h + d^R_h$.

5.  **Concatenation to form final queries and keys:**
    $$
    Q = \text{Concat}(q_L, \tilde{q}_R) \\
    K = \text{Concat}(k_L, \tilde{k}_R)
    $$
    where $\text{Concat}(\cdot, \cdot)$ is the vector concatenation operation. Final $Q, K \in \mathbb{R}^{d_h}$, where $d_h = d^L_h + d^R_h$.

**Dimensions:**

*   $h_t \in \mathbb{R}^{d_{model}}$ — input token embedding
*   $c_t \in \mathbb{R}^{d_c}$ — low-rank vector
*   $q_R, k_R \in \mathbb{R}^{d^R_h}$ — RoPE queries and keys
*   $\tilde{q}_R, \tilde{k}_R \in \mathbb{R}^{d^R_h}$ — RoPE queries and keys after applying RoPE
*   $q_L, k_L \in \mathbb{R}^{d^L_h}$ — low-rank queries and keys
*   $Q, K \in \mathbb{R}^{d_h}$ — final queries and keys, $d_h = d^L_h + d^R_h$
*   $W_{uc} \in \mathbb{R}^{d_{model} \times d_c}$, $W_{qR} \in \mathbb{R}^{d_{model} \times d^R_h}$, $W_{kR} \in \mathbb{R}^{d_{model} \times d^R_h}$, $W_{uq} \in \mathbb{R}^{d_c \times d^L_h}$, $W_{uk} \in \mathbb{R}^{d_c \times d^L_h}$ — projection matrices

#### Detailed MLA Formulas

In MLA DeepSeek-V3, there are 128 attention heads, each with a dimension of 128. The dimension $d_c$ is 512.
For a more detailed understanding of the MLA mechanism, refer to section 2.1.2 of the DeepSeek-V3 technical report [[6](https://arxiv.org/abs/2412.19437)].

In conclusion, Multi-Head Latent Attention (MLA) is a key technical innovation in DeepSeek-V3 aimed at optimizing computational efficiency and model scalability. Reducing KV-cache size and lowering activation memory requirements contribute to more efficient use of computational resources, enabling strategies for Test-time compute and building powerful reasoning systems like DeepSeek-R1.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_3.jpg)

Remember, this is not the only way to optimize attention for faster generation. Many have transitioned from classical MHA to Noam Shazeer's Multi-Query Attention (MQA) [[7](https://arxiv.org/abs/1911.02150)], where K and V are shared across all attention heads (significantly accelerating inference with minor quality degradation), and Grouped-Query Attention (GQA) from Google [[8](https://arxiv.org/abs/2305.13245)], which served as an intermediate step between MHA and MQA. In GQA, the number of key-value heads was greater than one but less than the full set of query heads—here, one key-value head serves a group of query heads—and quality could approach that of original MHA.

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_4.jpg)

MLA effectively saves cache space comparable to GQA with 2.25 groups, while performance even exceeds MHA.

# 5. DeepSeekMoE

Next, we examine DeepSeekMoE [[10](https://arxiv.org/abs/2401.06066)], which is also used in DeepSeek-V2.

DeepSeekMoE, introduced in the work by Baidu (2024) [https://arxiv.org/abs/2401.06066] and serving as a key component of DeepSeek-V2, is a Mixture-of-Experts (MoE) architecture designed to enhance efficiency and expert specialization. Unlike traditional MoE, where experts may reside in different layers, in DeepSeekMoE, expert blocks are integrated into Feed-Forward Network (FFN) layers, replacing standard FFN layers.

In the DeepSeekMoE architecture, the FFN layer is modified by introducing a mechanism to select and activate a specified number of experts from a total pool. Each expert represents an independent FFN layer activated by a routing algorithm. In the context of MoE architectures, note that GShard (Shazeer et al., 2020) [https://arxiv.org/abs/2006.16668] activated two experts per layer, while Switch Transformer (Fedus et al., 2021) [https://arxiv.org/abs/2101.03961] used one. In DeepSeekMoE, input tokens are routed to selected experts, and when multiple experts are activated, their outputs are aggregated, for example, via weighted averaging.

The primary goal of DeepSeekMoE is to achieve more pronounced expert specialization. To realize this goal, a fine-grained expert segmentation method is applied. According to this method, each expert is subdivided into $m$ fragments, and the number of activated experts is proportionally increased by a factor of $m$. This approach maintains computational resources at the same level while enabling activation of $mK$ experts from $mN$ instead of $K$ from $N$. Fine-grained segmentation expands the combinatorial space, potentially promoting deeper and more differentiated expert specialization within the model.

To ensure efficient acquisition of common knowledge, DeepSeekMoE incorporates dedicated shared experts, to which input data is consistently routed. This approach concentrates the learning of common knowledge within specialized shared experts, rather than distributing it among routed experts. Consequently, DeepSeekMoE includes $N_s$ shared and $N_r$ routed experts. In the DeepSeek-V3 configuration, one shared expert and 256 routed experts are used, with 8 activated per layer.

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_5.jpg)

Routed experts are selected based on the top-k principle, using a similarity score computed as the dot product between the input token's representation vector and the expert's centroid. Although the technical documentation does not detail the centroid calculation method, it is assumed that the centroid represents the mean activation (or input vector) of tokens processed by that expert. In DeepSeek-V3, a sigmoid function and normalization procedure are applied to the similarity scores before routing.

![Figure_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_6.jpg)

To prevent routing collapse, DeepSeek-V2 employed auxiliary loss mechanisms, including expert-level and device-level components. In DeepSeek-V3, additional losses were abandoned in favor of a loss-free load balancing strategy (Baidu, 2024) [https://arxiv.org/abs/2408.15664]. This strategy introduces a bias to the similarity score during routing, followed by selecting top-k experts based on the adjusted scores. Crucially, the bias is used exclusively for routing and does not affect the computation of expert mixing weights. Bias control is achieved by monitoring expert activity within the data batch. When an expert is detected as overloaded, its bias is decreased; conversely, it is increased when activity is low. This approach demonstrates greater efficiency compared to loss-based methods.

![Figure_7](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_7.jpg)

The figure above compares various load balancing methods. Loss-Free Balancing eliminates the trade-off between load balancing and model quality observed in other methods. Unlike alternative approaches, it ensures balanced expert load, eliminates gradient interference, and prevents future token leakage, which is critical for language models.

![Figure_8](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_8.jpg)

The figure above illustrates the expert routing process in DeepSeekMoE. First, gating scores are computed, to which expert bias is added. Then, top-k experts are selected, determining the load distribution. Subsequently, bias updating is performed based on feedback, helping dynamically balance load among experts. This mechanism reduces the likelihood of individual expert overload and improves computational resource utilization.

![Figure_9](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_9.jpg)

The graph above illustrates the trade-off between load balancing and model quality when using auxiliary loss functions. Increasing the coefficient $𝛼$ improves load balancing but worsens perplexity, reducing model effectiveness. The Loss-Free method avoids this trade-off, achieving optimal balance and performance without requiring additional loss functions.

The DeepSeekMoE architecture also integrates a Complementary Sequence-Wise Auxiliary Loss with a small weight coefficient, aimed at minimizing imbalance during sequence processing. Additionally, an algorithmic technique called Node-Limited Routing limits the maximum number of computational nodes involved in processing each token. This mechanism is conceptually analogous to the balancing losses used in DeepSeek-V2 and restricts each token to be routed to no more than four computational nodes.

<details> 
    <summary><em><strong>Mathematical Formalization of DeepSeekMoE</strong></em></summary>

The DeepSeekMoE architecture employs several key mathematical concepts to implement the Mixture-of-Experts mechanism, emphasizing expert specialization and load balancing. Let us examine each in detail:

#### 1. **Expert Routing**

**Goal:** Determine which experts should process each input token.

**Affinity Score Formula:**

$$
a_i = \sigma( h^T c_i + b_i )
$$

**Explanation:**

*   $a_i$ – affinity score of the input token to the $i$-th expert. Higher $a_i$ indicates greater suitability of the token for this expert.
*   $h = \text{LayerNorm}(x)$ – representation of the input token $x$ after applying Layer Normalization. Layer Normalization helps stabilize training and improves model generalization.
*   $c_i \in \mathbb{R}^d$ – centroid of the $i$-th expert. It is assumed that $c_i$ represents the "center" of the representation space of tokens this expert should process. As noted in the text, the centroid is likely computed as the mean of representations of tokens processed by this expert.
*   $b_i$ – dynamic bias for the $i$-th expert. This bias is used to balance load across experts. It is dynamically adjusted to prevent overloading some experts and underutilizing others.
*   $\sigma(\cdot)$ – sigmoid function. Applied to normalize the affinity score into the range [0, 1]. The sigmoid transforms the dot product and bias into a probability or confidence that the token should be routed to this expert.

**Expert Selection Formula (Top-K Selection):**

$$
\text{Top-K} = \arg\max_{i \in \{1,...,N_r\}} ( a_i )
$$

**Explanation:**

*   $\text{Top-K}$ – set of indices of $K$ experts with the highest affinity scores $a_i$.
*   $\arg\max_{i \in \{1,...,N_r\}} ( a_i )$ – operation selecting the $K$ expert indices from the total number of routed experts $N_r$ with the largest affinity scores $a_i$.
*   In DeepSeek-V3, as stated, $K=8$ and $N_r=256$. This means that for each input token, 8 out of 256 routed experts are selected for processing.

#### 2. **Fine-Grained Expert Segmentation**

**Goal:** Increase expert specialization without increasing computational cost.

**Fine-Grained Aggregation Formula:**

$$
y = \sum_{j=1}^{mK} g_j \cdot E_j^{(m)}(h), \quad \text{where } \sum g_j = 1
$$

**Explanation:**

*   $m$ – fine-grained segmentation factor (e.g., $m=4$). Each original expert is split into $m$ sub-experts.
*   $E_j^{(m)}(h)$ – the $j$-th sub-expert (among $mN$ total) processes the input representation $h$. Importantly, each sub-expert is smaller than the original expert (approximately $m$ times fewer FLOPs).
*   $mK$ – number of activated sub-experts. If originally $K$ experts were activated, after segmentation, $mK$ sub-experts are activated.
*   $g_j$ – gate weight for the $j$-th sub-expert. These weights determine how strongly each sub-expert contributes to the final output. The sum of all weights $g_j$ equals 1, ensuring output normalization.
*   $y$ – MoE layer output after aggregating outputs of activated sub-experts with their respective weights.

**Computational Cost Preservation:**

Fine-grained segmentation allows increasing the number of "specialized" computational blocks (sub-experts) without increasing total computation, since each sub-expert is smaller than the original expert. As shown in the example, activating $mK$ sub-experts, each requiring $\text{FLOPs}/m$, results in the same total FLOPs as activating $K$ original experts.

#### 3. **Shared Experts**

**Goal:** Ensure learning of general knowledge accessible to all input tokens.

**Output Layer Formula with Shared Expert:**

$$
y = E_{\text{shared}}(h) + \sum_{j \in \text{Top-K}} g_j \cdot E_j(h)
$$

**Explanation:**

*   $E_{\text{shared}}(h)$ – output of the shared expert, which processes the representation $h$ of every input token. The shared expert is always active and not involved in routing.
*   $E_j(h)$ – output of the $j$-th routed expert (here, referring to original experts, not sub-experts, if fine-grained segmentation is applied to routed experts).
*   $\sum_{j \in \text{Top-K}} g_j \cdot E_j(h)$ – aggregated output of selected routed experts, as described in sections 1 and 2.
*   $y$ – final MoE layer output, which is the sum of the shared expert output and the aggregated output of routed experts.

The shared expert enables the model to learn general patterns and knowledge applicable to all types of input data, while routed experts specialize in narrower, more specific domains.

#### 4. **Dynamic Load Balancing**

**Goal:** Distribute load evenly among routed experts to avoid situations where some experts are overloaded and others underutilized.

**Bias Update Formula:**

$$
b_i^{(t+1)} = b_i^{(t)} - \eta \cdot \left( \text{load}_i - \frac{\text{Total load}}{N_r} \right)
$$

**Explanation:**

*   $b_i^{(t+1)}$ – new bias value for the $i$-th expert at the next update step.
*   $b_i^{(t)}$ – current bias value for the $i$-th expert.
*   $\eta$ – learning rate for bias update. Determines how quickly the bias adjusts in response to load imbalance.
*   $\text{load}_i$ – number of tokens processed by the $i$-th expert in the current batch.
*   $\text{Total load}$ – total number of tokens in the batch processed by all routed experts.
*   $N_r$ – total number of routed experts.
*   $\frac{\text{Total load}}{N_r}$ – average load per expert under ideal uniform distribution.
*   $\left( \text{load}_i - \frac{\text{Total load}}{N_r} \right)$ – difference between actual load on expert $i$ and average load. A positive value indicates overload; a negative value indicates underload.

**Balancing Mechanism:**

The formula updates bias $b_i$ to reduce load on overloaded experts and increase load on underloaded ones. If an expert is overloaded ($\text{load}_i > \frac{\text{Total load}}{N_r}$), its bias $b_i$ decreases, lowering its affinity score $a_i$ in future steps and thus reducing its selection probability. Conversely, if an expert is underloaded, its bias increases, raising its selection probability.

#### 5. **Expert Output Aggregation – Gate Weights**

**Goal:** Determine the contribution of each selected expert to the final output.

**Gate Weight Formula:**

$$
g_j = \frac{\exp(a_j / \tau)}{\sum_{k \in \text{Top-K}} \exp(a_k / \tau)}
$$

**Explanation:**

*   $g_j$ – weight coefficient for the $j$-th selected expert.
*   $a_j$ – affinity score for the $j$-th expert, computed previously.
*   $\tau$ – temperature. Parameter controlling the "softness" of the weight distribution.
    *   High $\tau$ makes the weight distribution more uniform; contributions from all selected experts become more similar.
    *   Low $\tau$ makes the distribution sharper; the expert with the highest affinity score receives significantly greater weight than others.
*   $\exp(a_j / \tau)$ – exponential of the normalized affinity score. Exponentiation amplifies differences between affinity scores.
*   $\sum_{k \in \text{Top-K}} \exp(a_k / \tau)$ – sum of exponentials of normalized affinity scores for all selected experts. Used to normalize weights $g_j$ so their sum equals 1.

**Mixing Mechanism:**

The formula employs a softmax-like mechanism to compute gate weights. Experts with higher affinity scores $a_j$ receive higher weights $g_j$, meaning their output contributes more to the final result. Temperature $\tau$ allows tuning the degree of "focus" on the most suitable experts.

#### 6. **Additional Mechanisms**

**a) Complementary Sequence-Wise Auxiliary Loss (Complementary Sequence-Level Auxiliary Loss)**

**Goal:** Ensure load balance at the sequence level to avoid imbalance in processing long sequences.

**Auxiliary Loss Formula:**

$$
\mathcal{L}_{\text{aux}} = \lambda \cdot \sum_{s=1}^S \left( \frac{1}{L} \sum_{t=1}^L \mathbb{I}(E_j \text{ processed } x_t^s) - \mu \right)^2
$$

**Explanation:**

*   $\mathcal{L}_{\text{aux}}$ – value of the auxiliary loss.
*   $\lambda$ – scaling factor for auxiliary loss ($\lambda \ll 1$). The auxiliary loss has low weight to avoid dominating the main loss function.
*   $S$ – number of sequences in the batch.
*   $L$ – sequence length (assumed equal for all sequences in the batch for simplicity; generally, average or maximum length may be used).
*   $x_t^s$ – $t$-th token in the $s$-th sequence.
*   $\mathbb{I}(E_j \text{ processed } x_t^s)$ – indicator function equal to 1 if expert $E_j$ processed token $x_t^s$, and 0 otherwise.
*   $\frac{1}{L} \sum_{t=1}^L \mathbb{I}(E_j \text{ processed } x_t^s)$ – fraction of tokens in sequence $s$ processed by expert $E_j$. This measures the load on expert $E_j$ within one sequence.
*   $\mu$ – target average load. Desired average fraction of tokens each expert should process per sequence.
*   $\left( \frac{1}{L} \sum_{t=1}^L \mathbb{I}(E_j \text{ processed } x_t^s) - \mu \right)^2$ – squared deviation of actual expert $E_j$ load in the sequence from target average load. Squaring penalizes both overload and underload.
*   $\sum_{s=1}^S ( \ldots )^2$ – sum of squared deviations across all sequences in the batch.
*   $\sum_{j}$ (implicit summation over all experts; though not explicitly written, it is logically assumed the loss is computed for each expert and summed).

**Sequence-Level Balancing Mechanism:**

The auxiliary loss penalizes the model if load distribution across experts within individual sequences deviates significantly from the target average load. This promotes more uniform load distribution not only overall across the batch but also within each sequence, which may be critical for processing long texts.

**b) Node-Limited Routing**

**Goal:** Limit the number of computational nodes each token is routed to, to improve efficiency and reduce latency.

**Constraint Formula:**

$$
\sum_{n=1}^4  \mathbb{I}(\text{Token } x \text{ routed to node } n) \leq 4
$$

**Explanation:**

*   $\mathbb{I}(\text{Token } x \text{ routed to node } n)$ – indicator function equal to 1 if token $x$ is routed to computational node $n$, and 0 otherwise.
*   $n$ – index of computational node (assumed up to 4 nodes, as stated in the text).
*   $\sum_{n=1}^4 \mathbb{I}(\text{Token } x \text{ routed to node } n)$ – total number of computational nodes to which token $x$ is routed.

**Constraint Mechanism:**

The constraint $\sum_{n=1}^4 \mathbb{I}(\text{Token } x \text{ routed to node } n) \leq 4$ ensures each token is routed to no more than four computational nodes. This may be implemented at the infrastructure level or algorithmically during expert selection. Node limitation reduces communication overhead and enhances computational parallelism, especially in distributed computing environments.

---

**Key Takeaways on DeepSeekMoE:**

*   **Expert Routing:** Based on affinity scores, sigmoid, and dynamic bias for load balancing. Top-K experts selected.
*   **Fine-Grained Segmentation:** Increases expert specialization without increasing FLOPs by splitting experts into sub-experts and activating more of them.
*   **Shared Expert:** Ensures learning of general knowledge by processing every token.
*   **Dynamic Load Balancing:** Adjusts expert biases based on current load to distribute work evenly.
*   **Gate Weights:** Uses a softmax-like mechanism to aggregate expert outputs, determining each expert’s contribution based on affinity score and temperature.
*   **Additional Mechanisms:**
    *   Auxiliary loss for sequence-level load balancing.
    *   Node routing limit to enhance efficiency in distributed systems.

</details>

# 6. Multi-Token Prediction (MTP)

We now turn to examining the innovative feature known as "Multi-Token Prediction" (MTP). The essence of MTP lies in a conceptual expansion of the prediction paradigm, assuming forecasting not a single token but an entire set of tokens for each position in the sequence. In the current model architecture, specifically, prediction of two tokens is implemented—the current token and the one immediately following it. Theoretically, this approach aims to strengthen the training signal, which in turn potentially enhances data efficiency. Moreover, it is hypothesized that MTP facilitates more thorough preparation of the model for predicting future tokens by enabling deeper understanding of contextual dependencies.

![Figure_10](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_10.jpg    )

Token prediction in MTP is implemented sequentially. For predicting $D$ additional tokens, $D$ specialized MTP modules are employed, differing in their embedding structure and output head. Each module receives as input either the output from the main model layer or the output from the preceding MTP module, along with embeddings of the next token. Data are preprocessed with RMSNorm and subsequent concatenation. Each MTP module computes a cross-entropy loss. The average loss across all modules is integrated into the model’s overall loss function as an additional term, scaled by coefficient $\lambda$ (set to 0.3 for the first 10T tokens and 0.1 for subsequent 4.8T). Importantly, during inference, MTP modules are disabled, yet their application remains feasible within speculative decoding, opening avenues for future research and optimizations.

![Figure_11](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_11.jpg    )

The effectiveness of MTP is confirmed by consistent performance improvements across diverse benchmarks. Empirical studies show that accuracy in predicting the next token ranges from 85% to 90%. Notably, when combined with speculative decoding, TPS (tokens per second) increases significantly—by 1.8 times.

### Infrastructure

The infrastructure underpinning DeepSeek-V3 training is equally critical. Model training was conducted on a powerful compute cluster comprising 2048 NVIDIA H800 GPUs. It is worth noting that the H800 is a specialized variant of the H100 adapted for the Chinese market. The H800 architecture features optimized interconnect parameters, resulting in more than a twofold reduction in bandwidth and fewer NVLink connections. FP64 FLOPS performance is also reduced by an order of magnitude—a factor not critical for neural network training but potentially limiting in other domains such as nuclear physics computations. Within NVIDIA’s product lineup, the H200 is positioned as an enhanced iteration of the H100, featuring increased memory capacity and faster memory access.

![Figure_12](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_12.jpg    )

A proprietary training platform named HAI-LLM was developed for this purpose. The DeepSeek-V3 architecture integrates a comprehensive set of parallelism strategies, including 16-way pipeline parallelism (Pipeline Parallelism, PP), 64-way expert parallelism (Expert Parallelism, EP) distributed across 8 nodes, and ZeRO-1 data parallelism (Data Parallelism, DP). To maximize pipeline parallelism efficiency, an innovative DualPipe algorithm was developed, enabling overlap of communication and computation phases in both forward and backward passes. This approach significantly reduces pipeline idle time, enhancing overall system throughput. Thanks to substantial advances in memory optimization, tensor parallelism (Tensor Parallelism, TP) was not required. Additionally, high-performance inter-node all-to-all communication kernels were developed to enable efficient data exchange between compute nodes.

![Figure_13](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_13.jpg    )

# 7. FP8 Training

Particular interest lies in the methodology of training the model using the FP8 format. For readers unfamiliar with FP32, FP16, and BF16 formats, we recommend reviewing the following detailed explanation.

<details> 
    <summary><em><strong>Understanding Floating-Point Formats in Machine Learning</strong></em></summary>

In machine learning, particularly deep learning, we work with vast volumes of numerical data—model weights, inputs, intermediate computations, etc. To represent these numbers, computers use various formats. Among the most common are floating-point formats, which enable representation of both very large and very small numbers.

**Why are different formats important?**

The choice of numeric format affects several critical aspects:

* **Computational precision:** The format determines how accurately a number can be represented. Higher precision (e.g., FP32) allows more accurate storage and processing.
* **Memory footprint:** The format determines how much memory is required to store one number. Less precise formats (e.g., FP16, BF16, FP8) consume less space.
* **Computational speed:** Operations on less precise formats can execute faster on specialized hardware (e.g., GPUs and dedicated accelerators).

In the context of training neural networks, especially large models, balancing these aspects is critical. Using less precise formats can accelerate training and reduce memory consumption, but care must be taken to avoid quality degradation due to insufficient precision.

### Core Formats: FP32, FP16, BF16

Let us examine each of these formats in detail.

#### 1. FP32 (Single-Precision Floating Point)

* **Full name:** IEEE 754 single-precision binary floating-point format.
* **Size:** 32 bits (4 bytes) per number.
* **Structure:** Composed of three parts:
    * **Sign (Sign):** 1 bit (indicates positive or negative number).
    * **Exponent (Exponent):** 8 bits (determines number magnitude).
    * **Fraction/Mantissa (Fraction/Mantissa):** 23 bits (determines significant digits).

**FP32 Characteristics:**

* **High precision:** FP32 provides sufficient precision for most machine learning tasks. It has long been the "standard" format.
* **Large value range:** Can represent both very large and very small numbers.
* **Moderate memory usage:** 4 bytes per number—not the most economical, but not excessive.
* **Performance:** FP32 operations may be limited on certain hardware, especially when handling very large models.

**FP32 Applications:**

* **Traditionally used for training neural networks.** Long the de facto standard.
* **Used when high computational precision is required.**
* **May be used for storing model weights and activations.**

**Analogy:** Imagine a 1-meter ruler marked in millimeters. FP32 is like such a ruler: sufficiently precise for most everyday measurements.

#### 2. FP16 (Half-Precision Floating Point)

* **Full name:** IEEE 754 half-precision binary floating-point format.
* **Size:** 16 bits (2 bytes) per number.
* **Structure:**
    * **Sign (Sign):** 1 bit.
    * **Exponent (Exponent):** 5 bits.
    * **Fraction/Mantissa (Fraction/Mantissa):** 10 bits.

**FP16 Characteristics:**

* **Half precision:** FP16 precision is significantly lower than FP32. The representable range is also smaller.
* **Low memory usage:** Requires half the memory of FP32.
* **High performance:** FP16 operations can be significantly faster than FP32 on hardware optimized for FP16 (e.g., modern NVIDIA GPU Tensor Cores).

**FP16 Applications:**

* **Accelerating neural network training and inference.** FP16 enables higher throughput and reduced latency.
* **Reducing memory consumption.** Allows training and deploying larger models under memory constraints.
* **Frequently used in mixed-precision training.** In this technique, some computations (e.g., gradients) are performed in FP32 for stability, while others (e.g., forward and backward passes) are performed in FP16 for speed.

**Analogy:** FP16 is like a 30-cm ruler marked in half-centimeter increments. Less precise than the meter ruler, but more compact and faster for approximate measurements.

**FP16 Limitations:**

* **Limited range and precision:** May cause overflow or underflow with very large or very small numbers. Precision loss may accumulate in deep networks.
* **Requires careful usage:** FP32 cannot always be directly replaced with FP16 without additional measures such as gradient scaling or loss scaling.

#### 3. BF16 (BFloat16)

* **Full name:** Brain Floating Point 16-bit. Developed by Google for use in TPUs (Tensor Processing Units).
* **Size:** 16 bits (2 bytes) per number.
* **Structure:**
    * **Sign (Sign):** 1 bit.
    * **Exponent (Exponent):** 8 bits.
    * **Fraction/Mantissa (Fraction/Mantissa):** 7 bits.

**BF16 Characteristics:**

* **Precision:** Lower than FP32, but **crucially, BF16 sacrifices mantissa precision while preserving FP32’s exponent range.** This is its key distinction from FP16.
* **Value range:** **BF16’s range is nearly identical to FP32.** This means BF16 better prevents overflow/underflow than FP16, especially for gradients in deep learning.
* **Low memory usage:** Like FP16, occupies 2 bytes per number.
* **High performance:** Supported by many modern accelerators, including NVIDIA GPUs and Google TPUs.

**BF16 Applications:**

* **Alternative to FP16 for accelerating training and inference.** BF16 is often considered a safer alternative to FP16, especially for training large models, due to its wider range.
* **Widely used in Google’s ecosystem (TPU, TensorFlow).**
* **Considered an "industry standard" or common FP32/16 combination.** This stems from BF16’s good balance of precision, range, and performance.

**Analogy:** BF16 is like a 1-meter ruler with centimeter markings instead of millimeters. The markings are less precise than millimeters (FP32), but the ruler’s length (range) remains unchanged. For many tasks where ultra-high precision is unnecessary but wide range is essential, such a ruler may be perfectly adequate and more convenient.

**FP16 vs BF16 Comparison:**

| Characteristic        | FP16                                  | BF16                                  |
|-----------------------|---------------------------------------|---------------------------------------|
| Size                  | 16 bits                               | 16 bits                               |
| Exponent Range        | Smaller than FP32                     | **Comparable to FP32**                |
| Mantissa Precision    | Higher than BF16                      | Lower than FP16                       |
| Overflow/Underflow Risk | Higher than BF16                   | Lower than FP16, comparable to FP32   |
| Performance           | High                                  | High                                  |
| Memory Usage          | Low                                   | Low                                   |
| When to use?          | When speed and memory savings are critical, but range and precision require caution | When wide range and speed are critical; often a safer choice than FP16 |

### FP8

Now we return to the **FP8** format. FP8 is an even more "compact" floating-point format, occupying only 8 bits (1 byte) per number. Several variants of FP8 exist, but the general idea is to further reduce precision and range to achieve even greater performance and memory savings.

**As seen in the technical report:**

* **FP8 is a novel and promising format for training large models.** DeepSeek-V3 may be the first publicly presented large-scale model trained on FP8.
* **FP8 can provide significant throughput gains.** For example, Habana/Intel Gaudi2 shows a 34% increase over BF16 while maintaining comparable quality.
* **Microsoft is also actively researching FP8 (FP8-LM, MS-AMP library).**
* **Other companies (OpenAI, Google) may also be exploring FP8, though their strategies may differ.** Google, apparently, still prefers BF16.

**Why is FP8 becoming relevant?**

* **Growing model sizes:** Modern neural networks are becoming ever larger. Reducing precision and memory consumption is critical for training and deploying such models.
* **Specialized hardware:** Hardware manufacturers (NVIDIA, Intel, Google, etc.) are developing specialized accelerators optimized for low-precision formats, including FP8.
* **Balance between precision and efficiency:** Research shows that for many deep learning tasks, especially during inference, full FP32 precision is not always necessary. Using lower-precision formats can yield substantial speedups and resource savings without significant quality loss.

**In conclusion:**

FP32, FP16, BF16, and FP8 represent a spectrum of trade-offs between precision, range, performance, and memory usage. The choice of format depends on the specific task, hardware, and precision requirements. FP32 long served as the standard, but in recent years, half-precision formats (FP16, BF16) and, prospectively, even lower-precision formats (FP8) have become increasingly important for training and deploying large, efficient machine learning models. Development and adoption of FP8 is an active area of research and development aimed at further improving the efficiency of deep learning.
</details>

---

Despite the fact that the FP8 format is not the subject of the cited publication, this analogy enables a proper understanding of its key characteristics. It is highly likely that DeepSeek-V3 is the first publicly released large-scale production model trained using the FP8 format. As a contrasting example, Llama3, according to available information, was trained using the BF16 format, which is currently regarded as an industry standard or at least a widely adopted FP32/16 combination. In the context of prior research, the work of Israeli scientists from Habana (now Intel) [[11](https://arxiv.org/abs/2409.12517)] should be mentioned. They successfully trained a 7B model on 2T tokens using Intel-Habana’s Gaudi2 hardware platform, achieving quality comparable to BF16 while demonstrating a 34% increase in throughput. Another noteworthy earlier initiative is FP8-LM by Microsoft [[12](https://arxiv.org/abs/2310.18313)], in which the GPT-175B model was trained. Microsoft has also open-sourced the corresponding software library [link to github.com/Azure/MS-AMP], facilitating further research in this field. It cannot be ruled out that OpenAI has also transitioned to FP8 in its internal developments, at least for some models, although official information on this matter is lacking. Google’s strategy regarding training format selection remains somewhat ambiguous, but BF16 appears to be the preferred choice.

![Figure_14](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_14.jpg)

Nevertheless, it should be noted that DeepSeek-V3 implements a mixed-precision strategy, in which a specific set of operations continues to be performed using BF16 or even FP32 formats. In particular, higher-precision formats are applied to critical components such as the embedding module, output head, MoE gating modules, normalization operators, and attention mechanisms. Moreover, the model's primary weights, weight gradients, and optimizer states are preserved with higher precision. This approach is motivated by the need to ensure training stability, which is known to be one of the main challenges when using low-precision formats, alongside hardware support limitations. Despite this, the overwhelming majority of computationally intensive operations are performed in FP8, enabling substantial resource savings.

![Figure_15](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_15.jpg)

It is hypothesized that the application of the FP8 format has been a major factor in significantly reducing computational overhead. In an idealized scenario, transitioning to FP8 potentially doubles available compute while halving memory requirements. To enhance computational accuracy in FP8, several methodological enhancements were implemented, including more sophisticated quantization techniques, increased accumulation precision, and prioritizing mantissa over exponent. As a result, the E4M3 format (4 bits for exponent, 3 bits for mantissa) is uniformly used for representing all tensors, offering a more standardized approach compared to potentially combining E4M3 and E5M2 formats.

Targeted efforts were also made to optimize data storage and inter-processor communication, resulting in reduced memory consumption and data transfer overhead. The effectiveness of FP8 training was rigorously verified on the DeepSeek-V2 model in 16B and 230B parameter configurations. Results indicate that performance differences between models trained with FP8 and BF16 fall within statistical noise, confirming the viability of the FP8 approach.

![Figure_16](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_16.jpg)

In conclusion, optimization of the inference process should be noted. The prefill and decoding phases are deployed separately. Recall that the prefill phase involves processing all input tokens (prompt tokens) and computing intermediate KV pairs, while the decoding phase is an iterative autoregressive token generation process. A more detailed description of this process can be found at the following link: [link to details]. For the prefill phase, the minimal deployment configuration assumes 4 nodes equipped with 32 GPUs, with corresponding parallelism settings. For the decoding phase, which requires 9 expert models, the minimal configuration increases to 40 nodes comprising 320 GPUs and features a distinct set of optimizations tailored for this phase.

# 8. DeepSeek-V3 Model Training Procedure

The training process of the DeepSeek-V3 model consists of two main stages: pretraining and posttraining. During pretraining, large volumes of data are processed and various machine learning strategies are applied to form the base model. In the posttraining stage, supervised fine-tuning (SFT) and reinforcement learning (RL) are performed to optimize the model for interactive use. We examine key aspects of both stages, including the use of novel techniques and comparative analysis with analogous models.

### Pretraining

#### Data Preparation and Tokenization

Compared to the previous version, DeepSeek-V2 [[13](https://arxiv.org/abs/2405.04434)], DeepSeek-V3 increased the proportion of mathematics and programming-related data and expanded its linguistic coverage. However, the majority of the dataset still consists of English and Chinese texts. The final corpus includes 14.8 trillion tokens (compared to 8.1 trillion in DeepSeek-V2). Byte Pair Encoding (BPE) with a vocabulary of 128k tokens was used for tokenization. The new tokenizer was redesigned for more efficient processing of multilingual data, and punctuation-combining tokens with line breaks were added.

#### Pretraining Methodology

During training, the next-token prediction strategy is combined with the fill-in-the-middle (FIM) technique. The latter is implemented with a frequency of 0.1, similar to DeepSeekCoder-V2 [[14](https://arxiv.org/abs/2406.11931)], and was originally proposed by OpenAI [[15](https://arxiv.org/abs/2207.14255)]. In this method, the model is trained to reconstruct the central portion of text using the "Prefix-Suffix-Middle" (PSM) structure:

```
<|fim_begin|>𝑓_pre<|fim_hole|>𝑓_suf<|fim_end|>𝑓_middle<|eos_token|>
```

During pretraining, the maximum sequence length was 4000 tokens. To extend the context, the YaRN algorithm [[16](https://arxiv.org/abs/2309.00071)] was applied, increasing the context window first to 32k tokens and then to 128k. This process included two additional training phases of 1000 steps each.

<details> 
    <summary><em><strong>Short Overview of YaRN</strong></em></summary>

**Introduction**  
Modern large language models (LLMs), such as LLaMA, GPT-NeoX, and PaLM, demonstrate impressive results in natural language processing (NLP) tasks. However, their applicability is constrained by a fixed context window—the maximum sequence length on which the model was trained. This becomes a critical barrier for tasks requiring analysis of long texts, such as document summarization, multi-turn dialogues, or processing scientific articles. In the paper "YaRN: Efficient Context Window Extension of Large Language Models," an innovative method for extending the context window of models using Rotary Position Embeddings (RoPE) is proposed, combining computational efficiency with maintained performance.  

**The Problem of Limited Context Window**  
The context window determines how many tokens the model can simultaneously consider when generating a response. For example, if a model is trained on 2048 tokens, it "cannot see" information beyond this range. The authors emphasize that this limitation reduces the practical applicability of LLMs in real-world scenarios where context often exceeds standard 4k–8k tokens. The problem is exacerbated by the fact that most models poorly extrapolate beyond their trained length, leading to a sharp decline in quality when handling long sequences.  

**RoPE and Extrapolation Challenges**  
Rotary Position Embeddings (RoPE)—a popular method for encoding positional information—uses rotational matrices to account for relative token positions. Despite its efficiency, RoPE, like other positional embeddings, suffers from an inability to generalize beyond the trained length. For instance, if a model was trained on sequences of length 2048, attempting to process 4096 tokens without modification leads to distorted positional information and reduced accuracy.  

**Drawbacks of Existing Methods**  
Before YaRN, two primary approaches existed:  
1. **Position Interpolation (PI)** — linearly "stretching" positional embeddings to accommodate a larger context.  
2. **"NTK-aware" interpolation** — a method inspired by neural tangent kernels that distributes interpolation unevenly across frequencies.  

However, both methods require substantial computational resources for fine-tuning—for example, PI needs 10–100 million tokens. Furthermore, after context extension, models exhibit degraded performance on short sequences, limiting their versatility.  

**YaRN Method: Components and Innovations**  
YaRN (Yet another RoPE extensioN method) solves these issues through three key components:  

1. **"NTK-by-parts" Interpolation**  
   Unlike previous methods, YaRN accounts for the heterogeneity of frequencies in RoPE. High-frequency components (responsible for local connections between adjacent tokens) are interpolated minimally to preserve detail, while low-frequency components (global context) are interpolated more aggressively. This enables the model to correctly handle both nearby and distant tokens in the extended window.  

2. **Attention Scaling via Temperature**  
   A temperature coefficient $ t $ is introduced into the attention mechanism to soften the softmax function. This reduces logit imbalance during context extension and stabilizes training. Importantly, this modification requires no changes to the model code and adds no computational overhead.  

3. **Dynamic Scaling**  
   During inference, the model gradually adapts to exceeding the original context window, avoiding abrupt performance drops. For example, upon reaching the 64k token limit, YaRN allows for a smooth degradation in quality rather than an immediate failure.  

**Experimental Results**  
YaRN demonstrates state-of-the-art results in context extension:  
- LLaMA 7B/13B models successfully scale to 128k tokens while maintaining low perplexity.  
- Fine-tuning requires only **0.1% of the original pretraining data** (10x less than PI) and **2.5x fewer training steps**.  
- On standard benchmarks (e.g., PG19, arXiv), YaRN outperforms PI and "NTK-aware" by 15–20% in accuracy.  

Interestingly, YaRN enables **extrapolation**: a model trained on 64k tokens correctly handles 128k without additional tuning. This opens the path to efficient utilization of "long context" without full retraining.  

**Practical Advantages**  
- **Compatibility**: YaRN integrates easily into existing architectures and is supported by libraries like Flash Attention 2, accelerating inference.  
- **Scalability**: The method works for models of varying sizes (from 7B to 70B parameters) and types (LLaMA, GPT-NeoX).  
- **Resource Efficiency**: Reducing data and training steps lowers deployment cost.  

**Key Quotes and Their Significance**  
- *«YaRN achieves state-of-the-art performance… on less than ∼0.1% of the original data»* — highlighting the revolutionary efficiency of the method.  
- *«Dynamic scaling allows the model to degrade gradually, not fail abruptly»* — a key advantage for industrial applications where stability is critical.  

**Conclusion and Prospects**  
YaRN establishes a new standard for extending LLM context windows. Its ability to maintain performance on short contexts, minimize fine-tuning costs, and support extrapolation makes it a universal tool for the NLP community. In the future, the method may be adapted for other positional embedding types and integrated into frameworks like Hugging Face Transformers, accelerating its adoption in industry.  

**Conclusion**  
The YaRN paper not only solves a specific technical problem but also opens new possibilities for applying LLMs in real-world tasks—from analyzing legal documents to creating dialogue agents with long-term memory. This is a crucial step toward overcoming one of the key limitations of modern language models.

</details>

![Figure_17](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_17.png)

The result surpasses the previous model DeepSeek-V2 and two dense models, Qwen2.5 72B Base and LLaMA-3.1 405B Base, across multiple benchmarks including English, Chinese, code, mathematics, and one multilingual benchmark, making it the strongest open model.

![Figure_18](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_18.jpg)

It is interesting to compare with Qwen2.5 72B Base—a strong model with nearly twice as many active parameters as DeepSeek. LLaMA-3.1 405B Base has 11 times more parameters but performs worse on these tests.

The output of this stage is the base model DeepSeek-V3-Base. The subsequent posttraining stage creates the instruction-tuned chat model DeepSeek-V3.

### Posttraining

#### Supervised Fine-Tuning (SFT)

During SFT, two types of data were used: reasoning-related and non-reasoning. The final instruction tuning dataset contained 1.5 million examples.

Reasoning data focused on mathematics, programming, and logical problems. These were generated by the internal DeepSeek-R1 model, which was itself trained based on DeepSeek-V3. However, the DeepSeek-R1 model suffered from verbosity, over-analysis, and incorrect formatting. To address this, a specialized expert approach was employed, incorporating SFT and RL stages. Data generation was performed with high temperature to identify patterns in R1’s responses and leverage them during dataset construction.

Non-reasoning data included examples of creative writing, role-playing scenarios, and simple question answering. These were created based on DeepSeek-V2.5 and underwent additional annotation review.

> As a result, the quality and volume of data in Supervised Fine-Tuning (SFT) critically affect the final model quality.

#### Reinforcement Learning (RL)

Reinforcement learning was based on two approaches: a rule-based reward model (RM) and an RM based on a model. The first method was applied in situations where formal verification of answers was possible, such as solving deterministic mathematical problems or programming tasks verifiable via a compiler. Where formal verification was difficult (e.g., creative writing tasks), a model-based reward model evaluated answer alignment with the prompt.

DeepSeek-V3 employs the Group Relative Policy Optimization (GRPO) algorithm [[17](https://arxiv.org/abs/2402.03300)], a modification of Proximal Policy Optimization (PPO). Unlike PPO, this method eliminates the need for a separate value function, reducing computational cost. Instead, it uses the average reward across samples from a single prompt. To ensure model stability during RL, KL-regularization measures were applied to constrain divergence from the base model, simplified through direct comparison between reference and policy models.

<details> 
    <summary><em><strong>Short Overview of GRPO</strong></em></summary>

### **Introduction to GRPO**
GRPO is a reinforcement learning algorithm designed to optimize LLMs in tasks requiring structured reasoning, such as mathematics and logic. It was introduced in DeepSeekMath and DeepSeek-R1 **as a response to challenges in training models with billions of parameters**. GRPO offers a more efficient approach compared to traditional methods like Proximal Policy Optimization (PPO), **by eliminating key bottlenecks related to advantage-function computation**.

<details> 
    <summary><em><strong>Explanation of Advantage Functions</strong></em></summary>

**Advantage function** is a key concept in reinforcement learning (RL), which **quantitatively evaluates the advantage of taking a specific action `a` in state `s` compared to the average action prescribed by the current policy**. Formally, it is expressed as the difference between the **Q-function** (expected cumulative reward for action `a` in state `s`) and the **V-function** (average expected reward in state `s` under the current policy):

$$
A(s, a) = Q(s, a) - V(s)
$$

---

### **Why are Advantage Functions Needed?**
1. **Assessing relative value of actions**:
   - Helps the model understand how much better or worse a specific action is compared to the "standard" behavior in a given context.
   - Example: In a math problem, the action "choose integration by parts" may have high advantage if it leads to the correct answer, and low advantage if it complicates the solution.

2. **Reducing gradient variance**:
   - Using relative advantage values instead of absolute rewards makes policy updates more stable.

---

### **How are Advantage Functions Computed in Classical RL (e.g., PPO)?**
In Proximal Policy Optimization (PPO):
1. A **value network** (a separate neural network) is trained to predict `V(s)`—the expected reward for state `s`.
2. **Q(s, a)** is estimated via the actual received reward plus discounted future rewards.
3. **Advantage** is computed as:
   $$
   A(s, a) = R_{\text{total}} - V(s)
   $$
   where $( R_{\text{total}} )$ is the discounted sum of rewards over a trajectory.

**Problems with PPO**:
- The value network requires additional computational resources and memory.
- Errors in `V(s)` predictions (especially in tasks with **multimodal reward distributions**, as in LLMs) distort advantage values.
</details> 

---

### **GRPO's Innovative Approach to Advantage Functions**

GRPO entirely eliminates the need for a value network by using **group-wise relative normalization**:
For each prompt $P$, a group of $N$ responses $G = \{O_1, O_2, ..., O_N\}$ is generated using policy $\pi$. Each response $O_i$ receives a reward $R_i = R(O_i)$ reflecting its quality. The advantage function for the $i$-th response $O_i$ relative to group $G$ is computed as:

$$
A_i(O_i, G) = R_i - \bar{R}_G = R_i - \frac{1}{N} \sum_{j=1}^N R_j
$$

where $\bar{R}_G = \frac{1}{N} \sum_{j=1}^N R_j$ is the average reward across group $G$.

> In essence, the advantage function in GRPO for each specific response is calculated as the reward of that response minus the arithmetic mean of all rewards in the group.

**Key Features of the GRPO Approach:**

*   **Group-wise Relative Normalization**: The advantage function is computed relative to a group of responses generated for the same prompt, ensuring a relative assessment of quality.
*   **Elimination of Value Network**: The group average reward $\bar{R}_G$ serves as a baseline, replacing the need for a separate value network to estimate state or action values.
*   **Learning via Comparison**: GRPO focuses on training a policy that generates responses superior to the average within its group, making it effective in tasks where relative quality assessment matters.
* **KL-Divergence: Tight Integration into Loss Function via Relative Weights**: KL-divergence is incorporated into the loss function for regularization, limiting the magnitude of policy changes per training step and preventing sharp fluctuations, thereby enhancing training stability.

**Limitations and Remarks:**

*   GRPO's effectiveness depends on the quality of the reward function $R(O)$. The reward function must be correctly designed to adequately reflect desired response properties.
*   Group size $N$ is a hyperparameter that can affect training stability and efficiency. Choosing the optimal $N$ may require experimental tuning.
*   GRPO, like other reinforcement learning methods, may be sensitive to optimization hyperparameters and model architecture.

---

### **Practical Interpretation for LLMs**
In GRPO, the advantage function becomes an **instrument for ranking response variants**:
- The model learns to generate responses that are not merely "good," but **significantly better than the group average**.
- This encourages:
  - Discovery of non-obvious, yet effective reasoning chains.
  - Avoidance of template-based errors common in the group.

**Effect**: The model focuses on **qualitative differences between responses**, not absolute reward values, which is critical for complex tasks with ambiguous success criteria.

**Problem Context**:
- In reasoning tasks, LLMs often generate multiple "reasoning chains" (chain-of-thought), but standard RL algorithms are poorly adapted for evaluating them.
- **Value networks in PPO require significant resources to train and are prone to errors in multimodal reward distributions**.

---

### **Key Differences Between GRPO and PPO**

| **Characteristic**                   | **PPO**                               | **GRPO**                                                                 |
|-------------------------------------|---------------------------------------|---------------------------------------------------------------------------|
| Presence of value network           | Required                              | Eliminated                                                                |
| Advantage estimation                | Based on value network                | **Group-wise relative normalization within trajectories**               |
| KL-divergence                       | Optional regularization               | **Tightly integrated into loss function via relative weights**           |
| Memory usage                        | High (2 models)                       | **Reduced by 40-60% due to removal of value network**                         |
| Convergence                         | Depends on value network accuracy     | **More stable due to group-wise gradient stabilization**               |

---

### **Mathematical Foundations of GRPO**
**Loss Function in GRPO**:

$$
L(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \, A(s,a) \;-\; \beta \cdot D_{KL}(\pi_\theta \,\|\, \pi_{\text{old}}) \right],
$$

where:
- **$\theta$** — parameters of the **current policy** (neural network) being optimized.
- **$s$** — the current **state** (state) of the environment.
- **$a$** — the **action** (action) selected by the agent in state $s$.
- **$\pi_\theta(a|s)$** — probability of selecting action $a$ in state $s$ according to the **current policy**.
- **$\pi_{\text{old}}(a|s)$** — probability of selecting action $a$ in state $s$ according to the **old policy**, fixed at the time of data collection.
- **$A(s,a)$** — the **advantage** of action $a$ in state $s$, computed as the difference between the expected reward for choosing $a$ and the average reward in state $s$. Formally:  
  $$A(s,a) = Q(s,a) - V(s),$$  
  where $Q(s,a)$ is the estimated total reward for choosing $a$ in $s$, and $V(s)$ is the average value of state $s$.
- **$\mathbb{E}_{(s,a) \sim \pi_{\text{old}}}$** — expectation taken over states and actions from the **experience** collected by the old policy $\pi_{\text{old}}$ (off-policy data).
- **$D_{KL}(\pi_\theta \,\|\, \pi_{\text{old}})$** — KL-divergence between the action distributions of the current and old policies in state $s$:  
  $$D_{KL}(\pi_\theta \,\|\, \pi_{\text{old}}) = \mathbb{E}_{a \sim \pi_\theta} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \right].$$
- **$\beta$** — hyperparameter regulating the strength of KL-regularization (**typical values: 0.05–0.2**).

---

### **Explanations**
1. **Off-policy learning**: Gradients are computed on data collected by the old policy ($\pi_{\text{old}}$), but the new policy ($\pi_\theta$) is optimized.  
2. **Importance weighting** $\frac{\pi_\theta}{\pi_{\text{old}}}$ corrects gradients to account for differences between policies, preventing estimator bias.  
3. **KL-divergence** limits the speed of policy change, ensuring training stability.  
4. **Advantage $A(s,a)$** directs updates toward actions with higher expected reward. If $A(s,a) > 0$, action $a$ in state $s$ is considered better than average.

**Optimization**:
- Gradients are updated only for tokens critically affecting reward (**e.g., key steps in mathematical derivation**).  
  - *Formally*, this can be represented by applying a mask $( M )$ to gradients, where $( M_i = 1 )$ for "critical" tokens and $( M_i = 0 )$ for others. Thus, only parameters associated with "critical" tokens are updated, improving learning efficiency by focusing on the most significant parts of reasoning.
- **Response sampling**: For each prompt, 4–8 variants are generated in parallel, improving solution space coverage.

---

### **A Few Numbers**
1. **Efficiency**:
   - Removing the value network reduces memory usage by **18.2 GB for a 33B parameter model** (DeepSeek-R1 experiments).
   - Training time is reduced by **35%** on MATH dataset tasks.

2. **Stability**:
   - Group normalization reduces gradient variance (**by 60% compared to PPO**).
   - KL-regularization prevents "policy collapse"—a typical PPO issue.

3. **Performance**:
   - On the MATH benchmark, GRPO improved DeepSeek-Math-7B accuracy from **51.2% to 58.7%**.
   - In logical reasoning tasks (e.g., FOLIO), improvement was **12.3%**.

---

### **Practical Implementation of GRPO**

**Implementation Steps**:
1. **Supervised Fine-Tuning (SFT)**:
   - Use data in format:  
     ```json
     {"prompt": "Solve the equation ∫₀¹ x² dx", "response": "∫₀¹ x² dx = [x³/3]₀¹ = 1/3"}
     ```
   - **Key aspect**: Clean data via self-consistency checks.

2. **Reward Modeling**:
   - For mathematical tasks (example):  
     
    $$
     [
       R = \text{Correctness} + 0.5 \cdot \text{StepQuality} \;-\; 0.3 \cdot \text{LengthPenalty}.
     ]
    $$

   - Designing an effective reward function is key to GRPO. Generally, it should be designed to reward desirable reasoning properties—correctness, logical sequence, conciseness, and solution efficiency. Weight coefficients (e.g., 1, 0.5, -0.3 in the example) can be empirically tuned to achieve optimal balance between these properties.

3. **Training with GRPO**:
   - **Hyperparameters**:
     - Batch size: 512 prompts (4 responses per prompt → 2048 examples/step).
     - Learning rate: 1e-6 with linear decay.
   - **Trick**: Freeze the first 10% of model layers to preserve general knowledge.

---

### **Use Cases**
1. **DeepSeek-Math-33B**:
   - Solving International Mathematical Olympiad (IMO) problems with **44.5%** accuracy.
   - **Feature**: Use of GRPO + Monte Carlo Tree Search (MCTS) for step generation.

2. **Logical Planner AlphaLogic**:
   - Automated theorem proving in Coq with **68%** success rate (vs. 52% for PPO).

---

### **Conclusion**
GRPO represents a significant advancement in reinforcement learning for LLMs, particularly in tasks requiring complex reasoning. **Its application is already extending beyond mathematics—current research is testing GRPO in legal analysis and scientific hypothesis generation.** Despite limitations, the algorithm demonstrates potential for creating "thinking" AI systems capable of deep abstract reasoning.

</details> 

---

Additionally, the Self-Rewarding method, based on the concept of constitutional AI [[18](https://arxiv.org/abs/2212.08073)], was employed. This approach improved model quality in subjective tasks lacking strict evaluation criteria.

<details> 
    <summary><em><strong>Short Overview of Self-Rewarding</strong></em></summary>

### **Introduction**  

This section summarizes key points from the paper *"Self-Rewarding Language Models."* The work presents an innovative approach to training large language models (LLMs) in which the model autonomously generates and evaluates its own training data. This minimizes dependence on anthropogenic (human-oriented) data, overcoming limitations of traditional AI alignment methods.

**Core Themes and Ideas**  
1. **Critique of Classical Alignment Methods**  
   The authors analyze shortcomings of Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). They emphasize that RLHF depends on a "frozen" reward model, whose quality is limited by the volume of human data, while DPO relies directly on anthropogenic preferences. Both approaches, according to the researchers, face a "bottleneck" in the finiteness and subjectivity of human evaluations [[18](https://arxiv.org/abs/2212.08073)]:  
   > *"The standard approach of RLHF learns a reward model from human preferences... A recent alternative is DPO... In both cases, the approach is bottlenecked by the size and quality of the human preference data"*.

2. **Architecture of Self-Training Models**  
   The key innovation is creating an agent combining two functions:  
   - **Generating responses** (instruction following);  
   - **Creating and evaluating training data** (self-instruction creation).  
   The model acts as a generator-critic, iteratively improving both its responses and its criteria for evaluation. The authors term this process *Self-Rewarding Language Models*.

3. **Iterative Learning via DPO**  
   Training is implemented cyclically:  
   - **Step 1**: Generate new prompts and responses, followed by evaluation via LLM-as-a-Judge (model assesses relevance, completeness, clarity, and other criteria);  
   - **Step 2**: Form preference pairs for DPO training.  
   Each iteration (Mt → Mt+1) improves both the model’s ability to follow instructions and its evaluation skills [[18](https://arxiv.org/abs/2212.08073)]:  
   > *"Our self-alignment method consists of two steps: (i) Self-Instruction creation... (ii) Instruction following training... This whole procedure can then be iterated..."*.

4. **Experimental Results**  
   - The Llama 2 70B model, after three iterations, outperformed Claude 2, Gemini Pro, and GPT-4 0613 on the AlpacaEval 2.0 benchmark.  
   - The greatest progress was observed in expertise-demanding tasks (STEM, humanities, role-playing).  
   - The model’s self-rewarding (reward modeling) capability correlates with human evaluations (r = 0.89).

**Important Implementation Details**  
- **Initialization**: Based on a pretrained model (Llama 2 70B) augmented with seed data from Open Assistant.  
- **Data Generation**: Uses few-shot prompting to create prompts and varied responses.  
- **Safety**: Authors note risks of reward hacking and the need for further ethical analysis.

**Conclusions and Prospects**  
The proposed method demonstrates potential for creating autonomous systems capable of continuous self-improvement. However, scaling this approach requires:  
- Decomposing limits of iterative learning;  
- Mechanisms to prevent reward hacking;  
- Independent safety evaluations.  

The work contributes to constitutional AI development by offering an alternative to anthropocentric LLM alignment approaches.

</details> 

---

![Figure_19](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_19.jpg  )
![Figure_19.1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_19.jpeg  )

The KL losses (necessary to prevent the model from generating radically different and unreadable text) are also simplified, since the comparison is performed directly between the reference model and the policy, rather than between the reward and the policy.

![Figure_20](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_20.jpg  )

The advantage in GRPO is essentially computed as a z-score.

![Figure_21](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_21.jpg  )

### Comparative Analysis and Conclusion

The evaluation results of DeepSeek-V3 demonstrate its superiority over its predecessors and competitors. On benchmark tests, DeepSeek-V3 outperforms models such as Qwen2.5 72B Base and LLaMA-3.1 405B Base in tasks involving English and Chinese language processing, programming, mathematics, and multilingual analysis.

Notably, DeepSeek-V3 achieved performance comparable to GPT-4o-0513 and Claude-Sonnet-3.5-1022, despite significantly lower training costs. In particular, the total training computational cost for DeepSeek-V3 amounted to 180,000 GPU-hours on H800, substantially lower than the estimated tens of millions of dollars required to create the Sonnet model.

![Figure_22](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_22.jpg  )

The paper presents an interesting analysis of distillation from the reasoning model (R1). This improves quality but also increases average response length, requiring careful tuning balance. They tested this on mathematics and programming but plan to extend it further.

![Figure_23](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_23.jpg  )

They also mention the use of Constitutional AI (https://arxiv.org/abs/2212.08073)—an approach I greatly appreciate (primarily due to its scalability)—for tasks where verification and algorithmic feedback are difficult. Essentially, the model evaluated itself, which they called Self-Rewarding. This approach improved quality, particularly in subjective evaluations. I understand they plan to add more constitutional inputs.

I will not delve into the benchmarks, but the paper contains a more detailed analysis. In any case, this is an impressive model.

![Figure_24](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_24.jpg  )

Thus, DeepSeek-V3 is not only a powerful language model but also an innovative platform for further AI research. Future development of the model may include optimizing learning algorithms, expanding language coverage, and improving RL techniques for more accurate modeling of complex interactions.

# 9. The Drumroll! 🥁 Here we are at R1

## What's Innovative About R1? 🤔

### Development of the DeepSeek-R1 Reasoning Model

As part of the DeepSeek model family development, the DeepSeek-R1 reasoning model was created, built upon the DeepSeek-V3-Base foundation. The DeepSeek-R1 architecture includes DeepSeek-R1-Zero, DeepSeek-R1, and an ensemble of six smaller distilled models.

#### Key Innovations of DeepSeek-R1

The key achievement of DeepSeek-R1, particularly the DeepSeek-R1-Zero version (whose name references AlphaZero), is demonstrating the feasibility of effective reasoning training primarily through reinforcement learning (RL) with a relatively limited volume of supervised fine-tuning (SFT) data. This suggests the potential to reduce dependence on extensive "human demonstrations" during SFT, although it is noted that initializing training with a small set of high-quality SFT examples contributes to improved results.

A significant outcome is the creation of an open model demonstrating advanced reasoning capabilities. It is anticipated that further development and adaptation of such models by the research community will lead to substantial progress in building AI capable of reasoning.

#### DeepSeek-R1-Zero: Implementation Details

The DeepSeek-V3-Base model served as the foundation for DeepSeek-R1-Zero. During training, the Group Relative Policy Optimization (GRPO) algorithm [[17](https://arxiv.org/abs/2402.03300)], previously used in DeepSeek-V3 and DeepSeekMath, was applied. Using GRPO eliminated the need for a separate critic model, which in traditional approaches is comparable in size to the policy model.

> *As previously described, GRPO is a method that eliminates the need for an explicit value function, reducing computational cost.*

The reward system in DeepSeek-R1-Zero is implemented based on rule modeling, which also reduces computational overhead compared to using neural network reward models. This approach is an evolution of the rule-based RM used during DeepSeek-V3's posttraining phase.

Within the reward system, two types of rewards were implemented:

* **Accuracy rewards**: Evaluation of answer correctness, applied in tasks with an objective criterion for correctness, such as mathematical problems or code-writing tasks.
* **Format rewards**: Ensuring the structure of the "reasoning process" adheres to a specified format, particularly using XML tags `<think>` to delineate reasoning steps.

The developers deliberately avoided neural network RM due to their vulnerability to adversarial attacks, high computational cost, and additional complexity associated with training such models.

A simple CoT prompt was used to activate the reasoning mechanism, instructing the model to "think" before generating an answer.

![Figure_25](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_25.jpg  )

DeepSeek-R1-Zero demonstrates significant progress during training, achieving AIME 2024 benchmark performance comparable to OpenAI o1-0912 and surpassing o1-mini after just 8,000 training steps. Applying a majority voting strategy (e.g., based on 64 generated answers) substantially improves final result quality.

![Figure_26](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_26.jpg  )

![Figure_27](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_27.jpg  )

The Zero approach, based solely on RL without prior SFT, allows direct observation of the model's characteristic evolution during training. In particular, a consistent trend toward increased generated response length is noted, interpreted as the model spontaneously learning the relationship between reasoning detail and solution quality. During training, emergent abilities such as reflection (re-evaluating previous steps) and exploration of alternative solution approaches—none of which were explicitly programmed into the model architecture—are also observed.

![Figure_28](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_28.jpg  )

A particularly interesting phenomenon is the observed "insight moment," demonstrating the model's ability to revise and correct its own answers, analogous to cognitive processes observed in humans.

![Figure_29](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_29.jpg  )

Despite these achievements, DeepSeek-R1-Zero is not without limitations. Output data may exhibit insufficient readability and linguistic inconsistency, including language mixing. To address these issues and improve the model's "cold start" quality, a decision was made to conduct preliminary fine-tuning on a high-quality dataset before beginning the RL phase.

### DeepSeek-R1

The development of DeepSeek-R1 is based on an enhanced training process structured into four sequential stages, each playing a crucial role in achieving the desired model characteristics.

The first stage, named **"Cold Start,"** focused on collecting an extensive corpus of data, including thousands of examples demonstrating long Chain-of-Thought (CoT) reasoning. The research team used "few-shot prompting," providing the model with detailed CoT examples to explicitly stimulate the generation of extended responses and thorough verification of each reasoning step. Notably, the initial data were derived from outputs generated by DeepSeek-R1-Zero, which underwent manual post-processing to ensure high quality and relevance. Each example concluded with a concise summary capturing the key points of the reasoning chain.

The second stage, labeled **"Reasoning-oriented Reinforcement Learning,"** aimed to fine-tune the DeepSeek-V3-Base model using data collected during the "Cold Start" phase. A similar reinforcement learning (RL) process as in -Zero was applied. To address the issue of linguistic heterogeneity in generated texts, an additional **language consistency reward** was introduced, defined as the proportion of the target language within the CoT. The final reward function integrated task accuracy and linguistic consistency, enabling training to convergence while ensuring both reasoning quality and linguistic uniformity.

The third stage, termed **"Rejection Sampling and Supervised Fine-Tuning,"** used a checkpoint obtained from the previous stage to generate data for subsequent supervised fine-tuning (SFT). While the initial "cold start" data were primarily oriented toward developing reasoning skills, the data collected in this stage covered a broader spectrum of tasks, including writing, role-playing, and other general-purpose tasks, thereby expanding the model's functional capabilities. The data were classified into two categories: reasoning-oriented (**Reasoning**) and non-reasoning (**Non-Reasoning**).

For the **Reasoning** category (600,000 examples), new reasoning chains were generated, using the checkpoint from the previous stage as the starting point. These chains underwent rigorous filtering, partially using DeepSeek-V3 as an evaluation model. For each prompt, multiple answer variants were generated, after which problematic results—characterized by language mixing, excessive verbosity (long paragraphs), or incorrect formatting (code blocks)—were discarded.

The **Non-Reasoning** category (200,000 examples) included examples covering a wide range of tasks such as writing, factual question answering (QA), self-reflection, and translation. To form this category, the DeepSeek-V3 pipeline was employed, partially using its SFT dataset and leveraging DeepSeek-V3's capabilities to generate new examples.

The final step of this stage involved fine-tuning the DeepSeek-V3-Base model (the original model, not the checkpoint from the previous stage) for two epochs on the full dataset comprising 800,000 examples, enabling integration and generalization of knowledge acquired in prior stages.

The fourth stage, titled **"Reinforcement Learning for All Scenarios,"** represented a second phase of reinforcement learning aimed at enhancing both **usefulness** and **harmlessness** of the model (analogous to Constitutional AI approaches), while further refining reasoning capabilities. For reasoning-oriented data, rule-based rewards were applied, while for general data, reward models from the DeepSeek-V3 pipeline were used. In the context of usefulness, emphasis was placed solely on the final summary, whereas harmlessness evaluation considered the entire model output. Although specific implementation details of this stage are presented in limited scope, available information suggests an approach analogous to Constitutional AI (or RLAIF) was implemented to optimize both aspects—usefulness and harmlessness—rather than harmlessness alone, as originally proposed in the CAI concept.

### Distillation

The research team acknowledged that despite the high efficiency of large MoE models, there is significant demand for more compact and dense models. To meet this need, **distillation** of DeepSeek-R1 into various open-source architectures, including Qwen and Llama, was conducted. The distillation process involved fine-tuning these models on DeepSeek's outputs, using the aforementioned 800,000-sample dataset.

The result of this process was a family of distilled models, including:

* Qwen2.5-Math-1.5B
* Qwen2.5-Math-7B
* Qwen2.5-14B
* Qwen2.5-32B
* Llama-3.1-8B
* Llama-3.3-70B-Instruct

It is important to note that these distilled versions underwent only the supervised fine-tuning (SFT) stage without additional reinforcement learning (RL). This opens prospects for the community to further enhance their performance through RL fine-tuning and other optimization methods.

### Evaluation Results

To comprehensively evaluate the performance of DeepSeek-R1 and its distilled versions, the research team conducted a series of comparative tests, using models such as DeepSeek-V3, Claude-Sonnet-3.5-1022, GPT-4o-0513, OpenAI-o1-mini, and OpenAI-o1-1217 as baselines.

The evaluation results for **reasoning capabilities** demonstrated that R1 performs comparably to OpenAI-o1-1217, significantly surpassing Sonnet, 4o, and mini.

![Figure_30](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_30.jpg  )

The distilled models also demonstrated impressive results. As a baseline for comparison, the open model QwQ-32B-Preview was used:

![Figure_31](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_31.jpg  )

* DeepSeek-R1-Distill-Qwen-7B surpasses, notably, GPT-4o-0513.
* DeepSeek-R1-14B demonstrates superiority over QwQ-32B-Preview.
* DeepSeek-R1-32B and DeepSeek-R1-70B outperform o1-mini.

Notably, the community now has access to open models of such high quality that can be run locally. Further improvement of their characteristics can be expected as the community refines these models using RL and other fine-tuning methods.

A separate experiment with Qwen-32B-Base compared **pure RL training** (DeepSeek-R1-Zero-Qwen-32B) with **distillation**. The results demonstrate that distillation from a larger model is a more effective approach than directly training smaller models via RL.

![Figure_32](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_32.jpg  )

In other words, for creating an effective smaller model, distillation from a more powerful model is preferable to attempting direct RL training, for which success is not guaranteed. Notably, developing effective small models through direct training remains a challenging task, whereas the path through larger models proves more productive.

Another important conclusion is that **scaling** still plays a decisive role: larger models demonstrate higher performance. Consequently, R1's potential could be even greater if it were obtained through distillation from an even larger model.

### What Did Not Work?

The application of a **Process Reward Model (PRM)**, in which rewards are awarded not only for the final result but also for individual CoT steps, proved fraught with significant difficulties. In practice, identifying clearly defined steps within the overall reasoning process is often a non-trivial task. Even when such identification is possible, evaluating the accuracy of individual steps is extremely challenging. Moreover, this approach tends to provoke **reward hacking**, complicating the process and incurring substantial overhead. Ultimately, the benefits gained were limited and did not justify the effort expended.

The use of **Monte Carlo Tree Search (MCTS)**, analogous to that used in AlphaGo, entails decomposing the answer into finer steps to explore the solution space. The model was instructed to use special tags to delineate different reasoning stages. Initially, the research team used prompts to search for answers via MCTS with a pre-trained evaluation model. Subsequently, based on the obtained question-answer pairs, actor and critic models were trained to iteratively improve the process.

However, scaling this approach encountered serious obstacles. The solution space in natural language processing tasks lacks the clear structure found in games. Token generation becomes exponentially more complex with increased search depth, forcing researchers to limit maximum depth, leading to searches for local optima. Additionally, training an effective evaluation model is a challenging task, and the quality of this model directly impacts the generation process. Ultimately, achieving iterative improvement was not possible, leaving this as an unsolved problem.

### Future Plans

The authors outlined several directions for further model improvement, with R2 undoubtedly being the anticipated next step.

Planned enhancements include:

* Optimizing function calling mechanisms, expanding multi-turn dialogue capabilities, improving complex role-playing, and JSON generation.
* Eliminating language mixing: Since the model is optimized for English and Chinese, it exhibits a tendency to switch to these languages when processing queries in other languages. Although this may not be a critical issue, such behavior can disorient users.
* Reducing model sensitivity to prompt phrasing: A trend of degraded performance with "few-shot" prompting has been observed, leading to the recommendation to use "zero-shot prompting." This recommendation aligns with guidance for o1.
* Further optimizing the model for Software Engineering tasks, opening prospects for creating a local open-source copilot capable of significantly enhancing software development efficiency.

---

### **References:**

1. **Wei, J., Zhou, D., Wei, Q., Zou, C., Bastings, J., Cheng, C. Y., ... & Le, Q. V.** (2022).  
   *Chain-of-thought prompting elicits reasoning in large language models.*  
   arXiv preprint arXiv:2201.11903.  
   [📄 Paper](https://arxiv.org/abs/2201.11903  )

2. **Wang, X., Wei, J., Schuurmans, D., Le, Q. V., & Chi, E. H.** (2022).  
   *Self-consistency improves chain of thought reasoning in language models.*  
   arXiv preprint arXiv:2203.11171.  
   [📄 Paper](https://arxiv.org/abs/2203.11171  )

3. **Yao, S., Yu, D., Zhao, J., Cui, Y., Rao, I., Zhao, J., ... & Zhang, C.** (2023).  
   *Large language model guided tree-of-thought.*  
   arXiv preprint arXiv:2305.08291.  
   [📄 Paper](https://arxiv.org/abs/2305.08291  )

4. **Long, L.** (2023).  
   *Tree of thoughts: Deliberate problem solving with large language models.*  
   arXiv preprint arXiv:2305.10601.  
   [📄 Paper](https://arxiv.org/abs/2305.10601  )

5. **Schlag, I., Sukhbaatar, S., Celikyilmaz, A., Yih, W.-t., Weston, J., Schmidhuber, J., & Li, X.** (2023).  
   *Large Language Model Programs.*  
   arXiv preprint arXiv:2305.05364.  
   [📄 Paper](https://arxiv.org/abs/2305.05364  )

6. **DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, ..., & Zizheng Pan.** (2024).  
   *DeepSeek-V3 Technical Report.*  
   arXiv preprint arXiv:2412.19437.  
   [📄 Paper](https://arxiv.org/abs/2412.19437  )

7. **DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, ..., & Ziwei Xie.** (2024).  
   *DeepSeek Team et al., 2024b.*  
   arXiv preprint arXiv:2405.04434.  
   [📄 Paper](https://arxiv.org/abs/2405.04434  )

8. **Anonymous.** (2019).  
   *Fast Transformer Decoding: One Write-Head is All You Need.*  
   arXiv preprint arXiv:1911.02150.  
   [📄 Paper](https://arxiv.org/abs/1911.02150  )

9. **Anonymous.** (2023).  
   *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*  
   arXiv preprint arXiv:2305.13245.  
   [📄 Paper](https://arxiv.org/abs/2305.13245  )

10. **Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., ... & Liang, W.** (2024).  
    *arXiv preprint arXiv:2401.06066.*  
    [📄 Paper](https://arxiv.org/abs/2401.06066  )

11. **Fishman, M., Chmiel, B., Banner, R., & Soudry, D.** (2025).  
    *Scaling FP8 training to trillion-token LLMs.*  
    arXiv preprint arXiv:2409.12517.  
    [📄 Paper](https://arxiv.org/abs/2409.12517  )

12. **Peng, H., Wu, K., Wei, Y., Zhao, G., Yang, Y., Liu, Z., ... & Hu, H.** (2023).  
    *FP8-LM: Training FP8 Large Language Models.*  
    arXiv preprint arXiv:2310.18313.  
    [📄 Paper](https://arxiv.org/abs/2310.18313  )

13. **DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, ..., & Ziwei Xie.** (2024).  
    *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.*  
    arXiv preprint arXiv:2405.04434.  
    [📄 Paper](https://arxiv.org/abs/2405.04434  )

14. **DeepSeek-AI, Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., ..., & Liang, W.** (2024).  
    *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence.*  
    arXiv preprint arXiv:2406.11931.  
    [📄 Paper](https://arxiv.org/abs/2406.11931  )

15. **Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey, C., Tworek, J., & Chen, M.** (2022).  
    *Efficient Training of Language Models to Fill in the Middle.*  
    arXiv preprint arXiv:2207.14255.  
    [📄 Paper](https://arxiv.org/abs/2207.14255  )

16. **Peng, B., Quesnelle, J., Fan, H., & Shippole, E.** (2023).  
    *YaRN: Efficient Context Window Extension of Large Language Models.*  
    arXiv preprint arXiv:2309.00071.  
    [📄 Paper](https://arxiv.org/abs/2309.00071  )

17. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Guo, D.** (2024).  
    *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*  
    arXiv preprint arXiv:2402.03300.  
    [📄 Paper](https://arxiv.org/abs/2402.03300  )

18. **Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., ..., & Kaplan, J.** (2022).  
    *Constitutional AI: Harmlessness from AI Feedback.*  
    arXiv preprint arXiv:2212.08073.  
    [📄 Paper](https://arxiv.org/abs/2212.08073  )