# Kimi-K2

![Figure_0](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/Figure_0.png  )

## Introduction

Modern large language models (LLMs) have become a cornerstone tool across diverse domains—from automated text and code generation to supporting intelligent agents capable of executing complex tasks. As parameter counts and pretraining token volumes grow, we observe qualitative leaps in model capabilities: improved context understanding, generation accuracy, and specialized problem-solving skills.

Kimi-K2, developed by Moonshot AI, represents one of the most ambitious projects in the open LLM ecosystem. It employs a Mixture-of-Experts (MoE) architecture with a trillion parameters, yet activates only approximately 32 billion parameters per token due to its sparse activation mechanism. Kimi-K2 integrates cutting-edge attention optimizations for processing ultra-long contexts (up to 128k tokens), an innovative optimizer called MuonClip for stable and efficient training on an unprecedented scale of data (15.5 trillion tokens), and a comprehensive post-training pipeline to transform the base model into an interactive, agent-oriented assistant.

In this review, we provide a detailed analysis of:

1. **Kimi-K2’s architecture** — MoE principles, modified attention mechanisms, key parameters, and engineering solutions for accelerated inference.
2. **The training process** — pretraining on a massive corpus, distributed learning techniques, optimizers used, and fine-tuning and RLHF stages to enable “agentic” capabilities.
3. **Key results and comparisons** — against prior versions and industry leaders, on academic benchmarks, programming tasks, and agent-based scenarios.


## Kimi-K2 Model Architecture

### General Characteristics:

Kimi-K2 is a large language model with a *Mixture-of-Experts* (MoE) architecture, as previously discussed [here](https://verbasik.github.io/Weekly-arXiv-ML-AI-Research-Review/#2025/week-06). The total model size is **1 trillion parameters**, yet at any given moment, only approximately **32 billion parameters** are active—that is, only a small subset participates in processing each token. The model contains **61 transformer layers** (one of which is a “dense” layer without expert partitioning). The hidden representation (embedding) size is **7168**, with **64 attention heads**. The vocabulary size is **160k tokens**, and the maximum context length is **128k tokens**. The model is an autoregressive decoder (similar to GPT), generating text by sequentially predicting the next token based on previous ones.

![Figure_01](  https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/Figure_01.jpg  )

### Mixture-of-Experts:

In each transformer layer implemented as MoE, there are **384 separate experts** (specialized MLP subnetworks). For each token, a dynamic router (gating mechanism) selects the **top-8 experts** out of these 384 to process the token, plus one special **shared expert** that is always activated. Thus, each token undergoes computation through 9 expert subnetworks (8 selected + 1 shared), significantly improving model efficiency without linearly increasing computational cost. Unused experts remain “sleeping,” enabling the massive total parameter count (1T) while maintaining manageable inference costs (only 32B active). Routing follows a *top-k* principle (Kimi-K2 uses k=8), likely employing top-2 or top-8 gating, allowing each token to receive a combination of multiple expert “opinions” rather than a single one. The presence of a single **shared expert**, active in every layer, improves robustness and baseline quality—this component ensures that each layer includes a dense subnetwork to complement the specialized experts. **Sparse activation**—activating only a subset of subnetworks—reduces computation: not all parameters participate in every forward pass, only those most relevant to the current token.

### Attention Mechanisms:

The model employs a modified self-attention mechanism optimized for long contexts. First, the number of attention heads has been **reduced** compared to standard transformers: Kimi-K2 uses 64 heads per layer with an embedding size of 7168, resulting in an atypical projection size of ~112 per head. Larger heads in smaller numbers help stabilize attention computations over long sequences. Second, the model adopts the **Multi-Head *Latent* Attention (MLA)** method, originally demonstrated in DeepSeek V3, which we covered in exhaustive detail [here](https://verbasik.github.io/Weekly-arXiv-ML-AI-Research-Review/#2025/week-07_&_08). This approach dramatically reduces memory and computational demands when handling long contexts. In traditional multi-head attention, large key and value matrices must be stored per token (dimensionality proportional to the number of heads and their size). In MLA, each context position is stored as a **compressed latent vector** of fixed dimensionality (e.g., ~512–576 dimensions, including positional information), independent of the number of heads. Essentially, keys and values are not stored separately per head but compressed into a unified representation; the input *query* vector is then projected into this compressed space to compute scalar attention weights, and the result is projected back. This mechanism reduces the key/value cache size by approximately **60 times compared to standard MHA** and ~12 times compared to Grouped Query Attention (GQA), making the practical use of 128k-token contexts feasible without memory overflow. Despite additional projection operations, the overall computational complexity of attention over long sequences is drastically reduced—MLA requires orders of magnitude fewer FLOPs and memory for KV-cache updates and attention steps than standard attention.

To implement attention at such an enormous context length, the developers also employed **FlashAttention**—a highly efficient fused GPU algorithm for computing attention matrices. Publications note that during autoregressive generation (token-by-token), MLA integrates with a FlashAttention-like kernel for the *QK* and *AV* matrix multiplication stages, performing them in so-called “tiles” to enhance performance. Additionally, Kimi-K2 uses *Rotary Positional Embeddings* (RoPE) to encode token positions in the sequence. Thanks to **RoPE scaling** (linear or dynamic scaling of the rotational phase function step), the model maintains its ability to distinguish nearby positions even with extended context windows. Finally, attention projections **do not use bias terms**, and dropout is likely disabled in attention layers—these simplifications are commonly adopted in large LLMs to conserve parameters and improve stability. For activation functions, the model employs **SwiGLU** (Swish + Gated Linear Unit) in position-independent MLP layers—a function proven effective in large transformers (e.g., used in PaLM).

## Training Process

### Datasets and Data Volumes:

Kimi-K2 was trained on a *massive corpus of textual data* totaling **15.5 trillion tokens**—one of the largest datasets ever used to train an LLM to date. In essence, the model “read” nearly all accessible internet content (including numerous sources in English, Chinese, and other languages, code repositories, scientific texts, etc.) multiple times. This training task forces the model to form a generalized representation of language and knowledge contained in the data. Training was performed on variable-length sequences, likely with increasing maximum context length throughout training (to effectively leverage the 128k context by the end). The pretraining corpus included texts from diverse domains: from encyclopedias and news articles to code and mathematical solutions. Judging from results, special emphasis was placed on programming and mathematical datasets—the model demonstrates outstanding performance in coding and math, with benchmark results detailed below.

### The Muon Optimizer and Its Limitations

Muon is an optimization algorithm based on matrix orthogonalization principles, specifically using the Newton-Schulz iteration to orthogonalize gradient matrices. The core idea is to encourage diverse update directions, preventing weight matrices from collapsing into low-rank structures that could limit model expressiveness.

The original Muon algorithm applies the following update rule:

![Image_01](  https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/Image_01.png  )

#### Limitations of Muon

**Scalability Issues in the Original Version**

- Muon initially showed strong results on small models, but its efficacy when scaling to large models (billions of parameters) remained unproven.

**Attention Logit Instability in MoE Models**

- Muon, optimizing Query and Key projection matrices, could generate weights with anomalously large values, especially during late training. This led to explosive attention logits (up to $10^3$–$10^5$), breaking softmax and causing loss divergence.

- Unlike AdamW, where learning rate and momentum indirectly constrain update steps, Muon (particularly when combined with techniques like weight decay) sometimes scaled weights too aggressively.

### Optimizer **MuonClip**

Training such a large MoE model presents significant challenges—primarily **training instability**, manifested in exploding attention logits. The standard approach for LLMs is **AdamW**, but the **Moonshot AI** team developed a more token-efficient optimizer—[**Muon**](https://arxiv.org/abs/2502.16982  )—which has demonstrated superiority over AdamW in training large language models. However, when scaled (e.g., in the **Kimi K2** model, built on an architecture similar to DeepSeek-V3), instability emerged: attention logits became excessively high, particularly during later training stages. This led to "divergence"—a sharp spike in loss values and abrupt training termination.

To resolve this issue, a modification named **MuonClip** was proposed, with its key innovation being the **QK-clip** technique. Its essence lies in **directly scaling the Query and Key projection weights after optimizer updates**. This controls attention logits at the source—before softmax is applied. This proved more stable than post-hoc logit clipping, query/key normalization, or other heuristics.

![loss_vs_tokens](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/loss_vs_tokens.png  )

Formally, MuonClip introduces an adaptive scaling factor $\eta$ and a balancing hyperparameter $\alpha$, following these formulas:

![Image_02](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/Image_02.png  )

where $t$ is a predefined threshold. This ensures no logit exceeds the allowable value, even under accumulated gradients. Such adaptation prevents softmax explosions, preserving gradient stability and attention energy control.

In practice, **Kimi K2** was successfully pretrained on **15.5 trillion tokens** using MuonClip—**without a single crash, loss spike, or training halt**. This was made possible by precise control over attention logits and adaptive weight scaling. Note that training likely occurred in **BF16** or **FP16** format with dynamic loss scaling, enabling efficient GPU memory usage. During inference, weights were converted to **FP8 with block quantization**, but training itself was conducted in high precision.

Thus, **MuonClip** is not merely another optimizer, but an **engineering solution to LLM scalability**. It unites Muon’s token efficiency with precise attention stabilization—becoming one of the key factors enabling training of a model of this scale without failure.

### Distributed Training

One trillion parameters is far beyond the memory capacity of a single device; therefore, Kimi-K2 training was distributed across hundreds of GPUs. Moonshot has not disclosed the exact configuration, but expert estimates suggest hundreds of high-performance cards (e.g., NVIDIA A100/H100) and costs on the order of tens of millions of dollars. For efficient model parallelization, the training stack leveraged [**DeepSpeed**](https://github.com/deepspeedai/DeepSpeed  ) and techniques like [Zero Redundancy Optimizer (**ZeRO**)](https://arxiv.org/abs/1910.02054  ). Specifically, at least **ZeRO Stage-1** or Stage-2 was employed, distributing optimizer states and gradients across nodes to reduce per-device memory requirements. The model was likely also partitioned across experts among different nodes—a natural solution for MoE (different experts stored on different devices, with tokens routed to them). This approach scales nearly linearly: adding more GPUs allows inclusion of more experts. Additionally, standard techniques like **gradient checkpointing** (activations not stored fully but recomputed during backpropagation) were used to significantly reduce memory consumption during long-sequence training at the cost of minor additional computation. Together, these engineering solutions made training a model of unprecedented size feasible.

### Fine-tuning and RLHF:

After pretraining, the developers conducted additional staged fine-tuning to imbue the model with *agentic* capabilities and a user interface. Two versions were released: **Kimi-K2-Base**—the base model after pretraining (intended for researchers to fine-tune independently), and **Kimi-K2-Instruct**—a model that underwent specialized post-tuning and is ready for interactive use as a chatbot or agent system.

In post-tuning, special emphasis was placed on training the model to *perform actions*, not merely generate text. This stage can be broadly divided into **supervised fine-tuning on synthetic tasks** and **reinforcement learning with feedback**.

* **Simulating Tool Usage**: The Moonshot team generated an extensive dataset of tasks requiring interaction with external tools (APIs, databases, shell commands, web search, etc.) to teach the model sequences of actions. Instead of manual annotation, they employed *Large-Scale Agentic Data Synthesis*: auxiliary AI agents simulated thousands of scenarios across hundreds of domains, where the agent (model) had to use various tools to achieve goals. All steps (tool queries, received responses, final decisions) were recorded as pseudo-dialogues. An independent model-judge (LLM-critic) then evaluated these generated episodes against predefined quality criteria, selecting only the highest-quality, most successful attempts. These filtered, high-quality action sequences were used for *supervised learning*—Kimi-K2 was fine-tuned to replicate such multi-step solutions, effectively internalizing patterns for planning and tool invocation. This process laid the foundation for “agentive thinking” directly within the base model weights.

![workflow-agent](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/workflow-agent.png  )

* **RL with Self-Critique (in the spirit of RLHF)**: Beyond imitation learning, the developers implemented a **Reinforcement Learning** mechanism to further enhance the model’s ability to solve tasks, especially those without a single correct answer. The primary limitation of classical RLHF (learning from human feedback) is the limited and narrow scope of reward signals for creative or analytical tasks. In Kimi-K2, they took a creative approach: the model was trained to *self-evaluate* its outputs against predefined quality criteria. A **self-critique system** was implemented: the model generates an answer and, either in parallel or as a subsequent step, generates an evaluation of that answer based on predefined “quality rubrics.” Since this critic itself may be imperfect, it was periodically improved on verifiable tasks (e.g., math problems or code tests)—these *verifiable tasks* were used to *train the critic* to more accurately predict quality. The improved critic was then applied to non-verifiable tasks (e.g., essay writing, analysis) to provide reward/penalty signals to the main model. Thus, iterative reinforcement learning occurred without direct human involvement: the model learned to improve its actions by relying on an internal “judicial system” calibrated on solvable tasks. This approach is analogous to RLHF but replaces human feedback with scalable AI feedback. As a result, **Kimi-K2-Instruct** acquired “reflexive” skills: it immediately outputs actions or responses close to optimal, without requiring lengthy chain-of-thought deliberation (*reflex-grade model*).

The outcome of this final training was a model capable of **following instructions**, maintaining dialogue, and **autonomously executing complex action sequences**. Note that, at present, Kimi-K2-Instruct is not multimodal—unlike its predecessor (Kimi k1.5), it cannot process visual data directly and lacks a dedicated “thinking mode.” The team focused on textual and agentic capabilities, planning to add image/audio support and more advanced reasoning mechanisms (“long thinking”) in future versions.

## Additional Details and Comparison with Predecessors

### Evolution Compared to Kimi k1.5

The new model Kimi-K2 marks a significant leap forward compared to prior Moonshot AI models. The predecessor (Kimi k1.5), released earlier in 2025, was a multimodal LLM supporting images and extended context up to 128k. Kimi k1.5 also employed RL techniques and had substantial size, yet was significantly smaller than Kimi-K2: approximately **389 billion parameters (52 billion active)** under an MoE architecture—roughly one-third the scale of the current model. Kimi-K2 expanded scale: 1 trillion parameters (+157% over K1.5) and introduced novel technological solutions—specifically, the MLA mechanism for attention, whereas Kimi k1.5 relied on more traditional approaches (positional interpolation) for long-context capabilities. Furthermore, K1.5 emphasized multimodality and dialogue, while K2 prioritized **agentic capabilities** (autonomous task execution). Kimi-K2 in its current version does not support image or audio processing (multimodal aspects are planned for later), but substantially outperforms K1.5 in *textual* and *coding* tasks, as well as tool usage. Another distinction is **openness**: Kimi-K2 was released with open-source code and weights (Modified MIT License), whereas Kimi k1.5 was primarily accessible via API/interface without a fully open model. Thus, Kimi-K2 represents a more scalable, agent-focused evolution of the Kimi family.

### Performance on Benchmarks

![Figure_02](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/Figure_02.png  )

At release, Kimi-K2 demonstrates *state-of-the-art* results among open models and closely approaches closed-source leaders. On the academic knowledge and reasoning benchmark **MMLU** (57 subjects), the model achieves approximately **87.8%** accuracy, surpassing all prior open-source LLMs (for comparison, OpenAI GPT-4 was evaluated at ~86.4% on MMLU). On the Chinese-language equivalent **C-Eval**, Kimi-K2 scored ~**92.5%**, decisively outperforming previous Chinese-language models—confirming its deep understanding of Chinese data. In complex mathematical problems (**MATH**—school-level problem-solving), it achieved **70.2%** correct solutions—a notable leap over previous-generation models (for comparison, GPT-4 ~85%, Llama-2 70B ~50%). On elementary arithmetic (**GSM8K**), the model correctly solves **92.1%** of problems, nearly eliminating earlier errors in multi-step calculations.

Particularly impressive are performance metrics in programming. In code generation benchmarks, Kimi-K2 sets new records among open models. For example, on **LiveCodeBench v6** (a realistic competitive programming benchmark), the base Kimi-K2-Base model achieves ~**26.3%** *pass@1* accuracy, while the final instruct version, Kimi-K2-Instruct, reaches **53.7%** *pass@1*, **outperforming** even GPT-4.1 (~44.7%) on these tasks. In multilingual programming (**MultiPL-E** benchmark), the model approaches the top tier with ~85–86% accuracy, and on the internal **SWE-bench (Software Engineering)** test, it achieved **65.8%** successful solutions—comparable to some proprietary models from Anthropic and significantly better than most open-source models.

Kimi-K2 also leads on specialized agent benchmarks: it ranked first among open models on **Tau** and **AceBench** (evaluating tool usage capability). For instance, in Tau scenarios (solving tasks in retail, flight booking, telecom, etc., via tools), Kimi-K2-Instruct achieves 70–75% success, approaching Claude 2 levels and surpassing other open alternatives.

Collectively, these results indicate that **Kimi-K2 has established a new quality standard** for open models. On many metrics, it **catches up to, and occasionally surpasses**, leading closed systems. For example, developers note that Kimi-K2-Instruct outperforms versions of Claude 4 (Anthropic) and even updated GPT-4.1 on several key tests. VentureBeat also highlights that Kimi-K2 surpassed GPT-4 in certain “pain points,” such as mathematical proofs and complex code.

Of course, the model is not flawless—developers acknowledge that Kimi-K2 can still err in very long reasoning chains, may produce overly verbose answers to simple questions, and currently **lacks multimodal capabilities** (cannot "see" images). However, these shortcomings are acknowledged and actively addressed (improvements to "long thinking" and vision are planned for future versions).

### Conclusion

Kimi-K2 represents a technically outstanding LLM: an innovative MoE architecture with top-8 expert routing and QK-clip optimization enabled the creation of an *open* model with 1 trillion parameters, trained on an unprecedented data volume without failure. The training process incorporated advanced stability techniques (MuonClip, BF16), distributed workload (ZeRO), and imitation/reinforcement learning to cultivate agentic skills. The resulting model sets a new benchmark for open-source AI, excelling particularly in programming, mathematics, and autonomous task execution. Kimi-K2-Base provides researchers with a powerful foundation for their own experiments and fine-tuning, while Kimi-K2-Instruct is already available for direct use—deployable locally or via API without any paid subscriptions. Kimi-K2 demonstrates that open initiatives can compete with industry leaders and paves the way for even more advanced and accessible AI systems in the near future.