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

$$
\begin{aligned}
M_t &= \mu M_{t-1} + \nabla \mathcal{L}(W_{t-1}) \\
O_t &= \text{NewtonSchulz}(M_t) \\
W_t &= W_{t-1} - \eta_t \left( \gamma \cdot O_t \cdot \sqrt{\max(A,B)} + \lambda W_{t-1} \right)
\end{aligned}
$$

### Momentum Formation

![momentum_visualization](  https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/momentum_visualization.png  )

$$
M_t = \mu M_{t-1} + \nabla \mathcal{L}(W_{t-1})
$$

#### Key Components:
1. **$\mu$ (momentum coefficient)**  
   - Analogous to $\beta_1$ in Adam (typically chosen in the range **0.9–0.99**).  
   - Determines the proportion of the previous momentum retained. For example, with $\mu=0.9$, gradient history is weighted at each step.

2. **$\nabla \mathcal{L}(W_{t-1})$ (loss gradient)**  
   - A matrix of the same dimension as the trainable parameters $W_{t-1}$.  
   - Indicates how weights should be adjusted to reduce the model’s error on the current step.

3. **$M_t$ (accumulated momentum)**  
   - The resulting matrix combining the current gradient with the "history" of prior updates.  
   - Dimensionality: identical to $W_{t-1}$ (e.g., $7168 \times 7168$ for attention matrices in Kimi-K2).

---

#### Detailed Explanation:

*   **What is it?**  
    - The momentum mechanism is analogous to **inertia** in physics.  
    - Instead of abruptly changing direction at each step (as in standard SGD), the model "accumulates" gradients, averaging them with a coefficient $\mu$.  
    - The formula $M_t$ represents a **weighted sum** of the previous momentum ($M_{t-1}$) and the new gradient.

*   **How does it work?**  
    - At step $t=0$: $M_0 = \nabla \mathcal{L}(W_0)$ (no history).  
    - At step $t=1$: $M_1 = \mu M_0 + \nabla \mathcal{L}(W_1)$.  
    - At step $t=k$: $M_k$ contains contributions from all previous gradients, but older gradients have exponentially diminishing influence (due to multiplication by $\mu^k$).

*   **Why is it needed?**  
    1. **Noise suppression**: Neural network gradients are often "noisy" (especially with mini-batch training). Momentum smooths these fluctuations.  
    2. **Faster convergence**: In "ravines" of the loss landscape (narrow regions with sharp gradients), momentum helps accelerate movement along the ravine floor.  
    3. **Avoiding "sticking"**: Without momentum, the model may oscillate around a local minimum without reaching it (see visualization above).

---

#### Analogy:
Imagine descending a hill:  
- **Without momentum (SGD)**: You take small steps strictly downhill. If you hit a rock (noise), you abruptly change direction.  
- **With momentum**: You build up speed like a ball rolling down the slope. Minor obstacles (noise) don’t alter your trajectory, and overall motion becomes smoother and faster.

---

#### Distinctions in Muon:
- Unlike Adam, where momentum is combined with adaptive step sizes, Muon uses "pure" momentum before orthogonalization.  
- The matrix $M_t$ is later transformed into an orthogonal matrix $O_t$ (via Newton-Schulz iteration), preventing weight collapse into a low-rank subspace.

> **Important**: In Muon, momentum is applied to all model parameter matrices (e.g., $W^Q$, $W^K$ in attention layers), not just scalar quantities.

### Gradient Orthogonalization via Newton-Schulz Iteration

![orthogonalization_iterations_3d](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/orthogonalization_iterations_3d.png  )

![orthogonalization_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/orthogonalization_visualization.png  )

$$
O_t = \text{Newton-Schulz}(M_t)
$$

#### Key Components:  
1. **$M_t$ (accumulated momentum)**  
   - The input gradient matrix incorporating historical information (from the prior momentum step).  
   - Dimensionality: e.g., $7168 \times 7168$ for attention matrices in Kimi-K2.  

2. **Newton-Schulz Iteration**  
   - A fast numerical method for approximate matrix orthogonalization.  
   - Alternative to the computationally expensive SVD decomposition.  

3. **$O_t$ (orthogonalized gradient)**  
   - The resulting matrix with mutually perpendicular update directions.  
   - Ensures diverse optimization steps.  

---  

#### Detailed Explanation:  

* **What is it?**  
  - The process of transforming gradient matrix $M_t$ into matrix $O_t$, where all update directions become **mutually perpendicular**.  
  - Analogous to "mapping the terrain" along multiple axes instead of moving along a single line.  

* **How does it work?**  
  1. **Normalization**:  
     - $M_t$ is scaled by the Frobenius norm (analogous to "dividing by vector length" for matrices):  
     $$  
     X_0 = \frac{M_t}{\|M_t\|_F}  
     $$  
  2. **Iterative refinement**:  
     - Over 5 steps ($k=1..5$), a sequence of matrices $X_k$ is computed using the formula:  
     $$  
     X_{k} = 3.4445 \cdot X_{k-1} - 4.7750 \cdot (X_{k-1}X_{k-1}^T)X_{k-1} + 2.0315 \cdot (X_{k-1}X_{k-1}^T)^2 X_{k-1}  
     $$  
     - Coefficients are tuned for stable convergence.  
  3. **Result**:  
     - After 5 iterations, $O_t = X_5$—an almost orthogonal matrix.  

* **Why is it needed?**  
  1. **Combating rank collapse**:  
     - Prevents weight matrices from collapsing into low-dimensional subspaces.  
  2. **Directional diversity**:  
     - Each update explores a new direction rather than repeating prior ones.  
  3. **Efficiency**:  
     - Significantly cheaper than SVD (5–10x faster for large matrices).  

---  

#### Distinctions in Muon:  
- **Locality**: Orthogonalization is applied separately to each weight matrix (e.g., $W_Q$, $W_K$ in attention layers).  
- **Fixed cost**: Always exactly 5 iterations, regardless of matrix size.  
- **BF16 compatibility**: The method is robust to low-precision arithmetic errors.  

> **Important**: Orthogonalization does not alter the "magnitude" of the gradient (its norm), only its **directions**. This is akin to redistributing the same step budget across different coordinate axes.

### Weight Update

![weight_update_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/weight_update_visualization.png  )

![weight_decay_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/weight_decay_visualization.png  )

![scale_normalization_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/scale_normalization_visualization.png  )

Considering weight decay and update scale normalization:

$$
W_t = W_{t-1} - \eta_t \left( \gamma \cdot O_t \cdot \sqrt{\max(A,B)} + \lambda W_{t-1} \right)
$$

where:

* $\gamma = 0.2$ — scale normalization factor aligning Muon with RMS updates in AdamW (typically in the range 0.2–0.4);
* $\lambda$ — weight decay coefficient;
* $\sqrt{\max(A,B)}$ compensates for inconsistent update scales due to matrix dimensions.

#### Key Components:  
1. **$O_t$ (orthogonalized gradient)**  
   - Result of the prior step (Newton-Schulz iteration).  
   - Contains perpendicular update directions.  

2. **$\sqrt{\max(A,B)}$ (scale normalization)**  
   - $A$ and $B$ are matrix dimensions (e.g., for $W_Q$ of size $7168 \times 7168$: $A=B=7168$).  
   - Compensates for scale differences in updates across differently shaped matrices.  

3. **$\lambda W_{t-1}$ (weight decay)**  
   - Regularization to prevent overfitting.  
   - Analogous to "friction"—gradually reduces weight magnitudes.  

4. **$\gamma$ (scale coefficient)**  
   - Empirically tuned (0.2 for Kimi-K2).  
   - Aligns Muon’s step size with typical AdamW updates.  

---  

#### Detailed Explanation:  

* **What happens?**  
  The formula updates model weights by combining:  
  - **Intelligent direction** ($O_t$ — orthogonalized gradient)  
  - **Stabilizing corrections** (scale normalization + weight decay)  

* **Step-by-step logic:**  
  1. **Orthogonal step**:  
     $\gamma \cdot O_t$ defines the primary update direction, where:  
     - $\gamma=0.2$ reduces step size for stability  
     - $O_t$ guarantees diverse update directions  

  2. **Scale correction**:  
     Multiplication by $\sqrt{\max(A,B)}$ resolves the issue:  
     - For a $1024 \times 4096$ matrix ($\max=4096$):  
       $\sqrt{4096} = 64$ increases the step  
     - For a $128 \times 128$ matrix ($\max=128$):  
       $\sqrt{128} \approx 11.3$ decreases the step  

  3. **Regularization**:  
     $\lambda W_{t-1}$ acts as:  
     - A "brake" for large weights (L2 regularization)  
     - Prevents parameter explosion  

  4. **Final update**:  
     All components are summed and multiplied by the learning rate $\eta_t$  

* **Why such complexity?**  
  1. **For large models**:  
     - Without $\sqrt{\max(A,B)}$, wide matrices would receive gigantic updates  
     - Without $\lambda$, weights would exceed bf16 bounds  

  2. **For training quality**:  
     - Orthogonality of $O_t$ improves parameter space exploration  
     - Weight decay preserves generalization capacity  

---  

#### Practical Nuances:  
- **For attention matrices**:  
  $W_Q$, $W_K$ additionally undergo **QK-clip** (separate norm constraints)  
- **Parameter values in Kimi-K2**:  
  - $\gamma = 0.2$  
  - $\lambda \approx 0.01$ (typical for LLMs)  
  - $\eta_t$ decays according to a cosine schedule

### Optimizer **MuonClip**

Training such a large MoE model presents significant challenges—primarily **training instability**, manifested in exploding attention logits. The standard approach for LLMs is **AdamW**, but the **Moonshot AI** team developed a more token-efficient optimizer—[**Muon**](https://arxiv.org/abs/2502.16982  )—which has demonstrated superiority over AdamW in training large language models. However, when scaled (e.g., in the **Kimi K2** model, built on an architecture similar to DeepSeek-V3), instability emerged: attention logits became excessively high, particularly during later training stages. This led to "divergence"—a sharp spike in loss values and abrupt training termination.

To resolve this issue, a modification named **MuonClip** was proposed, with its key innovation being the **QK-clip** technique. Its essence lies in **directly scaling the Query and Key projection weights after optimizer updates**. This controls attention logits at the source—before softmax is applied. This proved more stable than post-hoc logit clipping, query/key normalization, or other heuristics.

![loss_vs_tokens](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/loss_vs_tokens.png  )

Formally, MuonClip introduces an adaptive scaling factor $\eta$ and a balancing hyperparameter $\alpha$, following these formulas:

$$
q_i = \eta^\alpha W_q x_i,\quad
k_i = \eta^{1 - \alpha} W_k x_i,
$$

where $W_q$, $W_k$ are attention layer weights, $x_i$ is the input vector, and the resulting attention logit becomes:

$$
(\eta^\alpha q_i)^T (\eta^{1 - \alpha} k_j) = \eta \, q_i^T k_j.
$$

Thus, the scale of attention logits $q_i^T k_j$ is directly regulated by $\eta$, which is updated at each step:

$$
\eta = \min\left( \frac{t}{\max_{i,j} (q_i^T k_j)}, 1 \right),
$$

where $t$ is a predefined threshold. This ensures no logit exceeds the allowable value, even under accumulated gradients. Such adaptation prevents softmax explosions, preserving gradient stability and attention energy control.

**Symptoms**:  
- On late training stages, logits $q_i^T k_j$ reach anomalously high values (e.g., $10^3$ instead of typical $[-10, 10]$).  
- This leads to:  
  - **Numerical instability**: softmax outputs NaN due to exponential overflow.  
  - **Loss divergence**: Sudden loss spikes and training collapse.  

**Causes**:  
1. Gradient accumulation in $W_q$ and $W_k$ due to deep network architecture.  
2. Absence of natural constraints on weight norm growth in Muon (unlike AdamW, where adaptive step sizes partially mitigate this).  

---

### Solution: QK-clip — Controlling Logits at the Source

![qk_clip_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/qk_clip_visualization.png  )

Instead of post-processing logits (e.g., via `torch.clamp`), MuonClip **embeds the constraint directly into Query and Key projection weights**. This is achieved by introducing an adaptive parameter $\eta$ and a balancing coefficient $\alpha$:

$$
q_i = \eta^\alpha W_q x_i, \quad  
k_i = \eta^{1 - \alpha} W_k x_i
$$

**How it works**:  
1. **Logit Scaling**:  
   The resultant logit $q_i^T k_j$ becomes $\eta \cdot (W_q x_i)^T (W_k x_j)$.  
   - If $\eta = 0.5$, all logits are halved.  
   - When $\eta = 1$, the system operates "as-is".  

2. **Automatic Adjustment**:  
   At each step, $\eta$ is recalculated as:  
   $$
   \eta = \min\left( \frac{t}{\max_{i,j} (q_i^T k_j)}, 1 \right),  
   $$  
   where $t$ is a threshold value (e.g., $t=50$).  

3. **Balance Between Query and Key**:  
   The hyperparameter $\alpha \in [0,1]$ distributes the scaling responsibility:  
   - $\alpha=0.5$: equal scaling of both projections.  
   - $\alpha=0.8$: heavier burden on $W_q$ (useful if Keys must remain stable).  

---

### Practical Implementation in Kimi K2

1. **Placement in Pipeline**:  
   - QK-clip is applied **after every weight update** by Muon, but **before the forward pass**.  
   - Computed independently for **each attention head**.  

2. **Hyperparameters**:  
   - Threshold $t$: selected from range $[30, 100]$ (depends on network depth).  
   - $\alpha$: typically $0.5$ or $0.6$ (tuned on validation).  

3. **Compatibility with BF16/FP16**:  
   - Scaling via $\eta$ prevents overflow during softmax computation.  
   - No gradients required for $\eta$—it is a deterministic operation.  

4. **Results**:  
   - Training on **15.5 trillion tokens** completed without incidents.  
   - No NaNs even in layers with 7168-dimensional embeddings.  

---

### Additional Techniques in MuonClip

1. **Dynamic Threshold $t$**:  
   - Early training: $t=100$ (allows exploration).  
   - Late training: $t=30$ (strict control).  

2. **Exponential Smoothing of $\eta$**:  
   To avoid abrupt jumps, use:  
   $$
   \eta_{\text{new}} = 0.9 \cdot \eta_{\text{old}} + 0.1 \cdot \eta_{\text{computed}}  
   $$  

3. **Integration with Weight Decay**:  
   QK-clip complements, but does not replace, L2 regularization. The full update formula:  
   $$
   W_t = (1 - \lambda) W_{t-1} - \eta_t \cdot \text{MuonGrad} \quad \text{→} \quad \text{QK-clip}  
   $$  

![optimizer_comparison_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/optimizer_comparison_visualization.png  )

![practical_implementation_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/practical_implementation_visualization.png  )

![qk_clip_mechanism_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/qk_clip_mechanism_visualization.png  )

![exploding_logits_visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-29/assets/exploding_logits_visualization.png  )

#### Limitations of Muon

**Scalability Issues in the Original Version**

- Muon initially showed strong results on small models, but its efficacy when scaling to large models (billions of parameters) remained unproven.

**Attention Logit Instability in MoE Models**

- Muon, optimizing Query and Key projection matrices, could generate weights with anomalously large values, especially during late training. This led to explosive attention logits (up to $10^3$–$10^5$), breaking softmax and causing loss divergence.

- Unlike AdamW, where learning rate and momentum indirectly constrain update steps, Muon (particularly when combined with techniques like weight decay) sometimes scaled weights too aggressively.

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