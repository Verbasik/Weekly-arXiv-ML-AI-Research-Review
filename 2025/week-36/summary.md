# Schema-Guided Scene-Graph Reasoning based on Multi-Agent Large Language Model System

## System Architecture and Methodology

SG² (Schema-Guided Scene-Graph Reasoning) is a multi-agent framework that overcomes the fundamental limitations of large language models in performing spatial reasoning over complex scene graphs. The system operates under an iterative "Reason-while-Retrieve" paradigm, where specialized agents collaborate to solve tasks without ever directly processing the full scene graph.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-01.png)

> Figure 1: Comparison of reasoning paradigms. (a) "Reason-only" directly processes the full scene graph, often leading to hallucinations and distractions. (b) "Retrieve-then-Reason" performs static retrieval before reasoning. (c) SG²'s "Reason-while-Retrieve" approach enables dynamic, iterative information gathering through specialized agents.

The architecture consists of two primary modules: a Reasoner responsible for task planning and solution generation, and a Retriever that programmatically extracts information from scene graphs. Each module contains specialized sub-agents that perform distinct duties while maintaining separated contexts to prevent information overload.

The Reasoner module includes a Task Planner, which organizes problem-solving by generating information retrieval queries, invoking external tools, or providing final solutions. It works in tandem with a Tool Caller, which transforms high-level queries into executable Python code. The Retriever module includes a Code Writer, which generates executable Python programs for programmatic querying of scene graphs, and a Verifier, which checks whether the retrieved information satisfies the original query requirements.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-02.png)

> Figure 2: Detailed workflow illustrating how the multi-agent system processes a task. The Task Planner generates queries based on the scene graph schema, while Code Writer produces executable code to extract relevant information, maintaining context separation between reasoning and retrieval operations.

## Schema-Guided Information Processing

The key innovation of SG² is the use of scene graph schemas as structural guidance for both reasoning and retrieval operations. Instead of overwhelming agents with raw graph data, the system provides each agent with an abstract description of the graph structure, including node types, edge relationships, and attribute specifications.

The schema performs several critical functions: it enables the Task Planner to reason abstractly about spatial relationships without processing irrelevant details, guides Code Writer in generating structurally correct graph traversal code, and ensures that queries between modules are well-formed and parsable. This schema-based approach prevents the common issue where large language models become distracted by irrelevant information in large, complex environments.

The programmatic retrieval mechanism represents a significant departure from traditional fixed-API approaches. Rather than relying on pre-defined query functions, Code Writer generates custom Python code capable of performing complex graph traversals, filtering operations, and data aggregation. This flexibility allows the system to adapt to diverse informational needs without requiring extensive manual API curation for specific tasks.

## Experimental Evaluation and Results

Researchers evaluated SG² across diverse environments and task types to demonstrate its effectiveness. Testing was conducted in BabyAI (a 2D grid environment) for numerical question answering and path planning tasks, and in VirtualHome for complex household planning scenarios.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-03.png)

> Figure 3: Example tasks from test environments. (a-b) BabyAI object pickup tasks, (c) numerical reasoning about spatial relationships, (d) household planning tasks in VirtualHome requiring multi-step action sequences.

Results demonstrate SG²’s superior performance across all tested scenarios. For numerical question answering in BabyAI, SG² achieved 98% success compared to 86% for ReAct and traditional graph-prompting methods, which typically scored below 70%. In path planning tasks, SG² maintained high performance (96–97% success), while many baseline methods completely failed on domain variations, dropping to 0% success.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-04.png)

> Figure 4: Visualization of how scene graphs represent spatial environments. The hierarchical graph structure captures relationships between rooms, objects, and spatial connections, enabling systematic reasoning about environmental layout.

Perhaps most significantly, ablation studies reveal the specific contribution of SG²’s design decisions. When ReAct was restricted to limited APIs (ReAct-limit), its performance plummeted (from 86% to 40% in numerical QA). However, SG² under the same API constraints (SG²-limit) still achieved 47% success, demonstrating that the multi-agent architecture itself provides substantial advantages by preventing context accumulation and maintaining focused reasoning.

## Performance Analysis and Computational Efficiency

Analysis of computational efficiency reveals SG²’s adaptive information processing capabilities. For logically simple tasks on large graphs, the system processes fewer tokens per iteration than would be required to analyze the full graph, demonstrating effective information filtering. For complex tasks requiring comprehensive analysis, SG² appropriately scales its computational effort while maintaining performance.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-05.png)

> Figure 5: Performance comparison using small language models. Although all methods show performance degradation with smaller models, SG² retains a relative advantage, achieving 60% success with Phi4-14B compared to baseline methods, which typically scored below 30%.

Evaluation using small language models (SLMs) provides insight into the framework’s accessibility. While performance declines significantly for all methods when using models such as Phi4-14B, Qwen3-14B, and DeepSeek-7B, SG² still outperforms baseline approaches. With Phi4-14B, SG² achieved 60% success compared to baseline methods scoring below 30%, indicating that the schema-based approach makes complex reasoning more accessible to smaller, more efficient models.

## Task Execution Examples

The paper presents detailed execution traces illustrating how SG² handles complex multi-step reasoning. In path planning tasks, the system demonstrates sophisticated understanding of environmental constraints, such as the necessity to collect keys before opening doors or removing obstacles before navigation.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-06.png)

> Figure 6: Example execution trace for an object pickup task. The system iteratively queries relevant information, generates tool calls for navigation, and maintains a clear reasoning chain while avoiding irrelevant environmental details.

For household planning tasks in VirtualHome, SG² successfully handles implicit action preconditions that confuse other approaches. For example, given the task "put soap in the cabinet," the system correctly determines that the cabinet must first be opened before the soap can be placed inside, demonstrating nuanced understanding of action sequences and environmental constraints.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-07.png)

> Figure 7: VirtualHome task execution showing how SG² handles action preconditions. The system correctly identifies that the bathroom cabinet must be opened before placing items inside, demonstrating understanding of implicit action requirements.

## Computational Costs and Scalability

Token consumption analysis reveals SG²’s efficiency advantages. The system demonstrates adaptive computational scaling, processing information proportionally to task complexity rather than environment size. For simple queries in large environments, SG² uses significantly fewer tokens than approaches processing full scene graphs.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-08.png)

> Figure 8: Token consumption for a simple numerical query. SG² processes fewer tokens than the size of the full graph (green line), demonstrating efficient information filtering for simple tasks.

For complex tasks requiring extensive reasoning, the system appropriately scales its computational effort while maintaining efficiency through targeted information retrieval rather than processing irrelevant environmental details.

![](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-36/assets/Image-09.png)

> Figure 9: Token consumption for complex path planning. Although SG² requires higher computational costs, it maintains efficiency by processing only task-relevant information via iterative retrieval, rather than analyzing the entire environment.

## Significance and Future Directions

SG² addresses critical limitations in current LLM-based reasoning systems for structured environments. The multi-agent architecture with schema-guided programmatic retrieval offers a robust solution to problems such as hallucinations, context overload, and inflexible information access patterns inherent in existing approaches.

The framework’s success across diverse tasks and its sustained performance advantages even under constrained conditions suggest broad applicability for embodied AI applications. The demonstrated ability to operate with smaller language models—albeit with reduced performance—points to potential for creating more accessible and deployable systems.

Future research directions include integrating additional specialized agents (e.g., solution verifiers), exploring multimodal capabilities for richer environmental understanding, and optimizing reasoning trace lengths for improved efficiency. The programmatic retrieval paradigm can be extended to other structured data types beyond scene graphs, potentially enabling similar improvements in database queries, knowledge graph reasoning, and other structured information processing tasks.

This work lays the foundation for more sophisticated multi-agent LLM systems capable of effectively navigating the complexity of the real world while preserving the reliability and efficiency required for practical deployment in robotics, virtual assistants, and smart environment applications.