# Computer Science > Computation and Language
# Title: Diffusion Language Models: A New Paradigm in NLP

## TL;DR

<details> 
    <summary><em><strong>A New Paradigm in NLP</strong></em></summary>

### 1. Introduction: A Paradigm Shift in Text Generation

Diffusion Language Models (DLMs) represent a revolutionary alternative to traditional autoregressive large language models (AR-LLMs), such as GPT. While AR-LLMs generate text sequentially, token by token, from left to right—leading to linear growth in time and computational cost with increasing response length—DLMs borrow ideas from successful diffusion models for images and audio. They learn to reconstruct text from a noisy version, gradually "denoising" and refining the output. This allows DLMs to generate text holistically and iteratively improve its quality, enabling faster and more coherent generation through parallel updates of multiple tokens and the ability to correct errors during the process.

This review examines four key DLM architectures: Gemini Diffusion (Google DeepMind), Mercury Coder (Inception Labs), LLaDA (Chinese researchers), and Eso-LM (NVIDIA & Cornell University), analyzing their architectural features, diffusion mechanisms, generation algorithms, and experimental results on quality and performance.

### 2. Architectural Analysis: Key Diffusion LM Models

#### 2.1. Gemini Diffusion (Google DeepMind)

**Gemini Diffusion** is an experimental text model from Google DeepMind that uses a diffusion approach to generate entire text fragments and iteratively refine them.

**Diffusion Mechanism:** The model is trained to transform random noise into meaningful text. During the forward process, random noise (masking or token replacement) progressively corrupts the original text. In the reverse process, the model iteratively removes noise, restoring the original content. Each denoising step is a transformer pass.

**Generation and Architecture:** It begins with a fully noisy sequence of fixed length and transforms it into a meaningful response over several iterations. All tokens are updated in parallel, generating an entire text block simultaneously.

**Speed:** Internal Google evaluations show speeds of 1000–2000 tokens/s, significantly surpassing AR models like Gemini Flash (~272 tokens/s).

**Bidirectional Attention:** During denoising, tokens see context from both sides, improving global phrase coherence and enabling consideration of future words when selecting earlier ones.

**Engineering Innovations:** Uses a specialized token recovery schedule and a modified transformer with full attention across the entire block to enhance coherence. The "Instant Edit" mode allows real-time editing of existing text, correcting grammar or changing style on the fly.

#### 2.2. Mercury Coder (Inception Labs)

**Mercury** is the first commercial-scale diffusion LLM, developed by Inception Labs and designed for code generation.

**Diffusion Mechanism:** Implements coarse-to-fine generation. In the first stage, a rough sketch is created, then refined in parallel over several iterations, gradually removing noise and filling gaps.

**Generation Algorithm:** Generates a full response in a fixed number of denoising steps. The key feature is parallelism, allowing multiple tokens to be updated in a single step.

**Speed:** Demonstrates over 1000 tokens/s on NVIDIA H100 GPUs, 5–10 times faster than optimized AR counterparts (e.g., GPT-4o Mini ~59 tokens/s).

**Innovations:** Based on score-based diffusion for discrete data, using score entropy for stable training. The model learns to estimate the likelihood ratio between the correct and current (noisy) token.

**Progressive Random Masking:** During the forward process, an increasing percentage of tokens are masked. The reverse process starts with a fully masked draft and progressively uncovers masks using an adaptive recovery schedule: easier-to-predict elements are revealed first, harder ones later. This makes generation robust and allows error correction.

**Optimizations:** Likely employs key-value (KV) caching of transformer activations between steps to improve efficiency.

#### 2.3. LLaDA (Large Language Diffusion Models)

**LLaDA** is an open research initiative demonstrating that diffusion models can compete with autoregressive models at scale.

**Architecture and Mechanism:** This is a Masked Diffusion Model (MDM). The forward process implements discrete random token masking. The reverse process is iterative mask restoration, where the transformer predicts tokens at masked positions (performing Masked Language Modeling) with bidirectional attention across the entire sequence. The model optimizes the variational lower bound on log-likelihood (ELBO).

**Generation Algorithm:** Generates a sequence in a fixed number of steps $T$. At each reverse step $t$, the model parallelly fills a subset of masks. The order of mask uncovering is randomized. Once a position is revealed, it remains fixed. This ensures parallelism and stability, allowing the total number of network calls to be significantly less than sequence length ($T \ll L$).

**Engineering Achievements:** First diffusion model successfully scaled to 8 billion parameters trained from scratch on 2.3 trillion tokens. LLaDA solves the "curse of diffusion in discrete space," demonstrating scalability and key LLM capabilities (knowledge generalization, instruction following) at the level of AR models. LLaDA-8B successfully performs complex interactive tasks after SFT and outperformed GPT-4o in reverse reasoning tasks requiring bidirectional processing.

#### 2.4. Eso-LM (NVIDIA & Cornell)

**Eso-LM (Esoteric Language Models)** is an experimental hybrid architecture combining autoregression and diffusion to achieve superior quality and speed.

**Two-Phase Generation:** The process is divided into a diffusion phase and a sequential phase. First, an MDM model parallelly generates a draft phrase, leaving some positions masked. Then, an AR model sequentially fills the remaining masks left-to-right. Parameter $\alpha_0$ controls the proportion: $\alpha_0=1$ for pure diffusion, $\alpha_0=0$ for pure AR.

> "Eso-LMs ... successfully unified autoregressive (AR) and masked diffusion model (MDM) paradigms."
>
> "The diffusion phase (orange) gradually removes noise from tokens in parallel, while the sequential phase (green) fills remaining masked tokens autoregressively."

**Unified Architecture and Attention:** Trained on a single transformer capable of operating in both modes via a custom attention mechanism with a bias matrix $A$.

**Eso-LM (A):** Uses bidirectional attention among clean tokens and causal attention to masked tokens during the diffusion phase.

**Eso-LM (B):** Introduces causal constraints even for "clean" tokens during diffusion, enabling KV activation caching for them, achieving up to 65% acceleration compared to baseline MDM without caching.

**Architecture Summary:** Eso-LM demonstrated that a symbiotic union of AR and diffusion is possible. The model achieved the best perplexity today among diffusion models and provides a continuous spectrum of quality-speed trade-offs between AR and MDM, landing on a new Pareto frontier of quality and speed.

> "This is the best result among nonlinear (non-AR) methods on corpus datasets."

### 3. Mathematical and Probabilistic Formalization

All discussed DLMs aim to model the probability distribution over text via a Markovian diffusion process, rather than autoregressive factorization $P(X) = \prod_{i=1}^L P(x_i \mid x_{<i})$.

#### 3.1. Cross-Entropy and Its Role

Cross-entropy ($H(P, Q) = -\sum_{k=1}^K P(k) \log Q(k)$) is the central loss function in classification tasks. It measures the model's "surprise" relative to the true distribution. In machine learning, minimizing cross-entropy is equivalent to maximizing log-likelihood. For one-hot labels, cross-entropy equals KL divergence ($D_{KL}(P \mid Q) = H(P, Q) - H(P)$), since the entropy of the true one-hot distribution $H(P)$ is zero.

#### 3.2. The Essence of Diffusion Models for Text

In DLMs, a latent variable—the diffusion step index $t$—is introduced.

**Forward (forward) process $q$:** Gradual addition of noise to data, e.g., via token masking with increasing probability $\beta_t$. At $t=T$, data is fully corrupted. This process is known exactly.

**Reverse (reverse) process $p_\theta$:** A learned process of noise removal. A neural network (parameterized by weights $\theta$) approximates the restoration distribution $p_\theta(x_{t-1} \mid x_t)$—the probability of obtaining a less noisy state $x_{t-1}$ from current $x_t$.

For computational tractability, this distribution is assumed to factorize fully over positions containing noise at step $t$:

$$ p_\theta(x_{t-1} \mid x_t) = \prod_{i\in \mathcal{M}_t} p_\theta(x^{(i)}_{t-1} \mid x_t) $$

This means the model predicts each masked token independently (conditionally on the current state of the entire sequence $x_t$). This is equivalent to a Masked Language Modeling (MLM) task, similar to BERT.

**Training:** Each diffusion step is optimized using cross-entropy on the correct token instead of the mask. The loss function takes the form of Negative ELBO (NELBO)—the negative variational lower bound on data log-likelihood.

$$ ELBO = \mathcal{L}_{\text{diff}} = \sum_{t=1}^T w_t \mathbb{E}_{x_t \sim q}\left[ -\log p_\theta\left(x_{t-1}^{\mathcal{M}_t} = X^{\mathcal{M}_t} \mid x_t\right) \right] $$

Where $X^{\mathcal{M}_t}$ are the true tokens at masked positions, and $w_t$ are weighting coefficients.

Mercury optimizes an ELBO analog via score entropy, training restoration of probability ratios. Eso-LM introduced a hybrid likelihood model, where the overall loss decomposes into NELBO for the diffusion part and standard AR cross-entropy.

**Key Insight:** DLMs expand the solution space for modeling $P(X)$, confirming that key LLM properties are enabled by generative modeling with likelihood maximization via an alternative factorization.

### 4. Experimental Results

#### 4.1. Generation Quality and Metrics

Modern DLMs have already reached quality comparable to AR transformers.

**LLaDA-8B:** Matches AR-LLMs of similar size (LLaMA2-7B) in perplexity and zero-shot tasks, approaching LLaMA3-8B. Solved the "reversal curse" better than GPT-4(open).

**Gemini Diffusion:** Comparable in quality to much larger Gemini 2.x models, nearly matching Gemini Flash-Lite on code benchmarks (HumanEval ~89.6% vs 90.2%). Outperformed on the mathematical test AIME 2025 (23.3% vs 20.0%). Lags on MMLU and Big-Bench Hard.

**Mercury Coder:** Outperformed Gemini 2.0 Flash-Lite, Claude 3.5 Haiku, GPT-4o Mini, and Qwen 2.5 Coder on 4 of 6 standard programming benchmarks. Occupies a Pareto-optimal zone in the "solution quality vs tokens/s" trade-off.

**Eso-LM:** Achieved perplexity of 26.2 on LM1B and ~30.5 on OpenWebText, better than prior diffusion LMs and approaching AR models. Generates more meaningful and diverse texts without mode collapse.

#### 4.2. Speed, Latency, and Efficiency

The main advantage of DLMs is generation speed, especially for long outputs.

**Gemini Diffusion:** Responds to queries in 1–3 seconds versus ~7 seconds for Gemini 2.5 Flash. Performance of 600–1300 tokens/s, with peak claimed values of 1000–2000 tokens/s, far exceeding GPT-3/4 (100–200 tokens/s).

**Mercury:** Achieved ~1100 tokens/s on H100, comparable to best specialized hardware accelerators, but on standard GPUs, highlighting algorithmic efficiency.

**Latency:** DLMs excel at generating large text fragments. Time grows closer to $O(\text{const})$ due to fixed iteration count, whereas AR models scale linearly $O(N)$.

**Time to First Token (TTFT):** Typically higher for DLMs, as the entire iteration cycle must complete before delivering a full answer, unlike AR models that output the first token immediately.

**Energy Efficiency:** Diffusion generation better parallelizes and fully utilizes GPUs. "Energy per token" for DLMs is potentially lower due to higher throughput. Adaptive mechanisms allow DLMs to save resources on simple queries.

| Model                         | Speed (tokens/s) | Code Quality                          | Features                                                                 |
|-------------------------------|------------------|----------------------------------------|--------------------------------------------------------------------------|
| Gemini Diffusion (Google)     | ~1000 tokens/s   | HumanEval ~90%                         | —                                                                        |
| Mercury Coder (Inception)     | 737 (Small), 1109 (Mini) on H100 | Comparable to GPT-4o Mini and Claude 3.5 | 5–10× faster than analogs                                                |
| LLaDA 8B                      | No exact data (potential $T \ll L$) | —                                      | Strong in zero-shot and instructions; outperforms GPT-4o in "reverse" tasks |
| Eso-LM (A/B)                  | Up to 65% speedup (B) | Close to AR                            | Best speed-quality trade-off on lengths ~1K tokens                       |

### 5. Comparison with Classical Transformers

#### 5.1. Advantages of the Diffusion LM Paradigm

- **High Speed and Low Latency on Long Outputs:** Parallel generation enables DLMs to be an order of magnitude faster, making LLMs viable for real-time applications (chatbots, voice assistants, IDE assistants).
- **Iterative Improvement and Self-Correction:** DLMs can correct errors and hallucinations on subsequent denoising steps, unlike irreversible token selection in AR models.
- **Greater Global Text Coherence:** Bidirectional attention allows consideration of future words when generating earlier ones, improving overall text consistency.
- **Flexibility and Editability:** DLMs naturally support interactive editing, mid-text additions, and stylistic transformations, as their training is based on MLM.
- **Adaptive Computation:** DLMs can dynamically adjust the number of generation iterations based on task complexity, potentially enabling more efficient resource usage.

#### 5.2. Limitations and Challenges

- **Complexity and Training Cost:** Require large datasets and computational resources (LLaDA 8B trained on 2.3 trillion tokens, 130k GPU-hours). Optimizing ELBO with multi-step masking is more complex than simple AR cross-entropy.
- **Time to First Token:** DLMs deliver the full text block at once, which may seem slow for scenarios where users expect a streaming text output (e.g., typing simulation).
- **Memory and Compute Overhead:** At each step, DLMs process the entire context, potentially requiring more memory. However, developments like Eso-LM (B) show KV caching is feasible.
- **Precision on Detailed Tasks:** In some cases, AR models show superior quality on tasks requiring precise knowledge or multi-step logical reasoning (e.g., MMLU for Gemini Diffusion).

#### 5.3. Potential to Replace AR Models

Many experts believe the diffusion paradigm may become the new standard for LLMs. Google openly states intentions to explore this path. Mercury's commercial release confirms industry interest in fast solutions. Advantages in speed and scalability are too significant. Hybrid variants like Eso-LM may serve as a smooth bridge, leveraging the best of both approaches and offering a spectrum of models with varying speed/accuracy ratios.

### 6. Conclusion and Future Directions

Diffusion Language Models represent a promising step forward in NLP, demonstrating that the iterative denoising paradigm can achieve results comparable to AR-LLMs with significantly better performance.

A paradigm shift is possible, but will take time. Coexistence of AR (for step-by-step prediction, maximum accuracy on small models) and Diffusion (for long texts, interactive scenarios) is likely.

Scalability is encouraging. There are no theoretical barriers to training DLMs with 70B+ parameters. If Gemini Diffusion scales to GPT-4 levels while retaining its speed advantage, it will cement the paradigm.

Practical applications will begin in narrow domains. Mercury chose code generation. Expect applications in real-time systems (speech translation, AR assistants, chatbots) and enterprise solutions due to point-editing capability.

Promising architectures: Gemini Diffusion may become the locomotive. Mercury demonstrated the value of new optimization objectives (score entropy). LLaDA is vital as proof of DLM adoption by the open-source community. Eso-LM points toward hybrid solutions and already demonstrates optimal speed-quality trade-offs with KV caching.

Diffusion Language Models today stand on the brink of transforming from a scientific experiment into a new industry standard. The competition between the two approaches drives better solutions, and in the coming years, we expect new hybrid paradigms to emerge, combining the strengths of all approaches.

</details>

## Introduction

Diffusion Language Models (**DLMs**) represent a new paradigm for text generation, an alternative to classical autoregressive transformers. In traditional large language models (LLMs), text is generated sequentially, token by token, left to right. This leads to high time and computational costs: long responses require proportionally more steps, and parallel acceleration is limited by sequential dependencies. The diffusion approach offers a different mechanism: the model learns to reconstruct text from a noisy version, gradually **"denoising" and refining the output**. The idea is borrowed from successful diffusion models for images and audio, where generation proceeds through repeated denoising transformations of random noise into a meaningful signal. Applied to language, this means DLMs generate a phrase holistically and **iteratively improve its quality**, rather than predicting the next word one at a time. This approach opens possibilities for **faster and more coherent** generation: the model can create entire text blocks at once, parallelly updating multiple tokens, and has the chance to correct errors during the process.

In this review, we examine four advanced DLM architectures demonstrating this new approach: **Gemini Diffusion** from Google DeepMind, **Mercury Coder** from startup Inception Labs, **LLaDA** (Large Language Diffusion Models) developed by a group of Chinese researchers, and **Eso-LM** (Esoteric Language Models)—a joint effort by NVIDIA and Cornell University. For each model, we analyze the architecture and diffusion mechanism, describe the generation algorithm (parallel, sequential, or hybrid) and key engineering innovations (e.g., token recovery schedules, attention modifications, and key-value caching). We then present the mathematical formalization of diffusion LLMs: how they define the distribution over text and which loss functions they optimize. Next, we compare experimental results—both in text quality (perplexity, BLEU, MMLU, etc.) and performance (speed in tokens/s, latency, energy efficiency)—on modern hardware platforms (GPU H100/A100, TPU, etc.). Finally, we contrast diffusion models with classical transformers, noting their advantages, current limitations, and potential to displace autoregressive models, as well as discussing prospects for scaling and practical application.

> Denoising (denoising) — the process of gradually removing random noise from a noisy sequence to restore meaningful text.

## Architectural Analysis: Key Diffusion LM Models

### Gemini Diffusion (Google DeepMind)

**Gemini Diffusion** is an experimental text model from Google DeepMind demonstrating a diffusion approach to language generation. Unlike standard LLMs that predict tokens sequentially, Gemini Diffusion generates an entire text fragment at once and iteratively refines it. **Diffusion Mechanism:** The model is trained to transform random noise into meaningful text, similar to how Stable Diffusion generates images from noise. During the *forward* process, random noise is applied to the data: original sentences are progressively corrupted (e.g., tokens are masked or replaced with noise) until unrecognizable. The model then learns the *reverse* process—step by step removing noise to restore the original textual content. Each denoising step is a transformer pass.

**Generation and Architecture:** During text synthesis, Gemini Diffusion begins with a *fully noisy sequence* of fixed length (e.g., all tokens masked or filled with random symbols) and transforms it into a meaningful response over several iterations. All tokens are updated in parallel—essentially, the model generates an *entire text block simultaneously*, rather than one word at a time. As a result, response speed increases dramatically: internal Google evaluations show Gemini Diffusion achieves **1000–2000 tokens/s**, whereas even the fastest autoregressive version of Gemini (Flash mode) yields only ~272 tokens/s. Additionally, the diffusion model can use *bidirectional attention* within the generated block: during denoising, tokens freely see context from both sides, unlike the strictly causal (left-to-right) attention in standard transformers. This gives Gemini Diffusion the ability for **non-local "view" of the sentence**—the model considers future words when selecting earlier ones, improving global phrase coherence.

**Engineering Innovations:** For efficient generation, Gemini Diffusion uses a specialized *token recovery schedule* and a modified transformer. Although full implementation details are not disclosed, it is known that hundreds of noise and recovery steps are used during training. Presumably, as in other MDMs (masked diffusion models), a random permutation of positions is applied: the order in which tokens are "cleared" at each step can be set by a *permutation*, allowing words to be revealed in any sequence. This eliminates the rigid left-to-right constraint and helps the model correct errors: if a token is poorly generated early on, a subsequent denoising iteration can replace it with a better variant. The denoiser uses a standard transformer but with **full attention across the entire block**, enhancing response coherence through global text optimization rather than local optimization as in AR models. Developers also note an "Instant Edit" mode: the model can take existing text as input and **edit it "on the fly"**, correcting grammar, changing style, or adding code—all naturally supported by the iterative diffusion mechanism. Altogether, Gemini Diffusion demonstrates that, with comparable quality to classical models, a diffusion LM can provide significantly lower latency and higher text coherence.

> Bidirectional attention (bidirectional attention) — a mechanism in neural networks where each token in a sequence can consider information from both preceding and succeeding tokens.

### Mercury Coder (Inception Labs)

**Mercury** is the first commercial-scale diffusion LLM, introduced by startup Inception Labs under Professor Stefano Ermon. Its version, **Mercury Coder**, is designed for code generation and is already available for testing, marking the practical application of dLLMs in industry. **Diffusion Mechanism:** Mercury implements a so-called *coarse-to-fine* ("from sketch to detail") generative process. In the first stage, the model generates a rough sketch of the output sequence by filling it with "white noise" text. For example, for a coding task, Mercury might start with a template where some characters are missing (masked) or random. Then, over several iterations, this sketch is *parallelly refined*: the model removes noise and fills gaps, gradually approaching the final solution. This process is analogous to diffusion models for images, where a grainy sketch evolves into a detailed image.

**Generation Algorithm:** Mercury generates a full response in a fixed number of denoising steps. **Parallelism** is the key distinction: the model can update multiple tokens simultaneously in one step (e.g., several words or code symbols). As a result, speed is impressive—**over 1000 tokens/s on NVIDIA H100 GPU**. Tests show Mercury Coder generates code 5–10 times faster than optimized autoregressive analogs (e.g., GPT-4o Mini ~59 tokens/s), achieving ~1100 tokens/s on H100. This is accomplished using a relatively small transformer (Small and Mini versions)—the acceleration is achieved purely through algorithmic improvements.

**Innovations and Features:** Exact architectural details of Mercury are not fully disclosed (model parameters, training corpus, etc., are kept secret). However, from scientific publications by co-authors, it is known that Mercury is based on the **score-based diffusion** method for discrete data. As early as October 2023, Ermon's group proposed a novel approach—*score entropy*, a discrete analog of score distribution alignment, ensuring stable diffusion training for text and forming a variational lower bound on log-likelihood (ELBO). In Mercury, the model learns not to directly predict tokens, but to **estimate the likelihood ratio** between the correct token and the current (noisy) token. This allows introducing a specific uncertainty metric for each symbol.

In practice, Mercury performs *progressive random masking*: during the *forward* phase, an increasing percentage of input text tokens are randomly masked at each step. By the end of the forward process, a significant portion of the sequence is hidden. The *reverse* process then begins with a fully masked draft and progressively **uncovers masks**. At each denoising step, Mercury computes for each token the *relative confidence* (that same transition ratio) that the token should be a specific word `y` instead of the current `x`. If confidence is high, the token "unmasks" (mask replaced with prediction); if not, it remains masked until a later step. Thus, an *adaptive recovery schedule* is applied: easier-to-predict elements are revealed first, harder ones later, allowing the model to spend more iterations precisely on difficult text segments. Ingenieurs at Inception Labs describe that during one diffusion step, Mercury updates **multiple tokens in parallel**, using a pre-trained mask distribution over steps. This makes generation not only fast but also robust—errors can be corrected on subsequent iterations. Unlike autoregression, where one incorrectly chosen letter sets a wrong context for all subsequent ones, here an error is not fatal: on the next step, the mask can be redefined correctly.

To improve efficiency, Mercury also implements classical techniques used by AR models: for example, **key-value (KV) caching** of transformer activations between steps to avoid recalculating unchanged parts of the sequence. While not explicitly stated in open sources whether Mercury uses caching, the general DLM principle allows it—especially if the order of mask uncovering is fixed. It is known that another team (NVIDIA/Cornell in Eso-LM) achieved significant acceleration by introducing causal constraints for caching during diffusion. Perhaps Mercury also applies attention optimizations, given its high achievable speed without specialized hardware. Overall, Mercury Coder demonstrated the viability of diffusion LLMs in real-world applications, particularly where **instant generation of long responses** is critical (e.g., code autocompletion).

### LLaDA (Large Language Diffusion Models)

**LLaDA** is an open research initiative that first demonstrated diffusion models can compete with autoregressive models at scale. The LLaDA authors (Nie et al., 2025) set out to train a diffusion model **from scratch** on a massive corpus (trillions of tokens) and test whether it could achieve key LLM capabilities—knowledge generalization, instruction understanding, in-context learning, etc.

**Architecture and Mechanism:** The name LLaDA stands for *Large Language Diffusion with Masking*—that is, it is a **masked diffusion model** for text. The forward process is implemented via *discrete random masking*: the original sequence is progressively "destroyed" by replacing individual tokens with a special MASK symbol independently until, at the final step, all tokens become masks. Each mask is interpreted as "noise" in the data. The reverse process is *iterative mask restoration*: a trained transformer receives a partially masked text and predicts what token should be at each mask position (i.e., performs **masked language modeling** at each diffusion step). LLaDA uses a **standard Transformer** without any special modules: the difference is only that during denoising, **bidirectional attention** across the entire sequence is allowed (no causal mask), since the order of mask filling is not fixed left-to-right. Thus, the model learns the distribution over text by optimizing the **variational lower bound on log-likelihood**: instead of directly maximizing $\log P(text)$, a sequence of auxiliary distributions (diffusion steps) is introduced, bound by the ELBO inequality. In practice, LLaDA's final loss function is a weighted sum of cross-entropy on predicting masked tokens at each diffusion step. Simply put, the model learns to perform mask filling well at all levels of data "noisiness"—from nearly clean sentences to fully masked ones.

**Generation Algorithm:** LLaDA generates a sequence of length $L$ in $T$ steps. Initially, $z_T$ is taken—a fully masked input (all $L$ positions = \[MASK\]). At each reverse step $t=T, T-1, \dots, 1$, the model parallelly fills some *subset* of masks with its predictions. The order in which masks are uncovered is either random or follows a special schedule. In the original work, independent random masking is applied: at each forward step, each token is masked with probability $p_t$. This corresponds to the fact that reverse uncovering proceeds not strictly left-to-right but in a random order—critical for removing AR constraints. Mathematically, the reverse step $t$ is modeled by the conditional distribution $p_\theta(x_{t-1} | x_t)$, parameterized by a transformer, which factorizes over all mask positions. After predicting some masks at step $t$, the model moves to step $t-1$, where fewer masks remain, and so on until $t=0$, when no masks remain and the final text is obtained. **Crucially**: once a position is revealed (mask replaced with a token), **it remains fixed** on all subsequent steps. This prevents cycling and improves stability: each step adds new "clean" tokens, and by the end, everything is restored. Such a parallel process is much faster than autoregression, as the total number of network calls (NFEs) can be far less than $L$ (e.g., $T=50$ steps for $L=100$ tokens versus 100 steps for AR).

**Engineering Achievements:** LLaDA became the first language diffusion model successfully scaled to **8 billion parameters**, trained from scratch on a massive corpus (~2.3 trillion tokens). This required ~0.13 million GPU-hours on NVIDIA H800 accelerators. Architecturally, the model is similar to GPT/LLaMA-family transformers (which served as baseline comparisons). However, the authors had to solve the problem known as the *"curse of diffusion in discrete space"*: early attempts to apply diffusion to text yielded much worse perplexities than AR models. LLaDA shows that with proper tuning (random masking, ELBO optimization, subsequent supervised fine-tuning), these limitations are overcome. Notably, LLaDA demonstrated the **scalability** of the diffusion approach: model quality steadily improved with increasing parameters and data, similar to autoregression. It was confirmed that key LLM capabilities do not critically depend on the AR paradigm: after standard instruction fine-tuning (SFT), the diffusion model LLaDA-8B successfully performs complex interactive tasks (dialogues, instruction following) at the level of top 8-billion-parameter ARMs. An interesting experiment involved *reverse reasoning*—when a question is posed "backwards" (e.g., a poem with its words reversed). AR models typically fail due to rigid context direction, while LLaDA succeeded and even outperformed GPT-4o in restoring a reversed poetic text. This demonstrates the natural advantage of the diffusion approach in tasks requiring bidirectional sequence processing.

### Eso-LM (NVIDIA & Cornell)

**Eso-LM** (*Esoteric Language Models*) is an experimental hybrid architecture combining the best features of autoregression and diffusion. The developers posed the question: Can we unify the high quality of AR models with the high speed and flexibility of MDMs, creating a unified approach trained "on all fronts"? The result was Eso-LM variants A and B—models that generate text in **two phases** and introduce special modifications to the transformer attention mechanism.

**Two-Phase Generation:** The process is divided into a *diffusion phase* and a *sequential phase*. First, an **MDM model parallelly generates a draft** phrase, filling some positions while leaving others masked. In simpler terms, at this stage, a sentence is produced with some words already in place and other spots left as blank "fields." Then the second phase begins: an **AR model completes the sentence** by sequentially filling the remaining masks left-to-right via standard autoregression. Formally, if $z_0$ is the partially masked sequence after the first stage, then in the second phase, the final output $x$ is generated as $x = \text{AR}(z_0)$, where the AR model sees already revealed "clean" tokens and completes the missing parts. Importantly, in this approach, *some tokens are generated in parallel, others sequentially*. The partitioning parameter is $\alpha_0$: if $\alpha_0 = 1$, the entire text is generated by diffusion (pure parallel mode); if $\alpha_0 = 0$, the entire text is generated by AR (classic mode). Usually, an intermediate value is chosen, e.g., $\alpha_0 = 0.5$—half the tokens are placed immediately by diffusion, half are filled in by AR. Such a hybrid allows **interpolating between AR and MDM** in quality and speed and adds additional flexibility: for example, the first and last words of a sentence can be determined in parallel (considering global context), while details in the middle are refined sequentially. Eso-LM clearly combines strengths: high modeling quality (AR component ensures smooth transitions, especially for complex fragments) and high speed over most of the sequence (parallel MDM component saves time).

**Example implementation from code:**
```python
# algo.py, lines 146-147
do_sequential = self.config.algo.alpha_0 != 1
do_diffusion = self.config.algo.alpha_0 != 0

# Two-phase generation in EsoLM._loss():
if do_sequential:
    # AR phase: sequential mask filling
    alpha_start = self.config.algo.alpha_0
    z0 = self.q_xt(x0_reconstruction, alpha_start)
    reconstruction_loss, sort_idx = self._reconstruction_loss(x0_reconstruction, z0)
    
if do_diffusion:
    # Diffusion phase: parallel generation
    diffusion_loss, sort_idx = self.nll(x0_diffusion, None, ...)
```

Formally, if $z_0$ is a partially masked sequence after the first stage, then in the second stage the final output $x$ is generated as $x = \text{AR}(z_0)$, where the AR model sees already revealed "clean" tokens and completes the missing ones. Importantly, with this approach, *some tokens are generated in parallel, while others are generated sequentially*. The partitioning parameter is the fraction $\alpha_0$: if $\alpha_0 = 1$, the entire text is generated by diffusion only (pure parallel mode); if $\alpha_0 = 0$, the entire text is generated by AR (classic mode). Typically, an intermediate value is chosen, e.g., $\alpha_0 = 0.5$: half the tokens are placed immediately by diffusion, and the other half are filled in by AR.

**Specific configurations from the project:**
```yaml
# configs/algo/esolm.yaml - base configuration
alpha_0: 0.0  # Default: pure AR mode

# Experiments with different values of α₀:
# scripts/esolm/train_owt_esolmb_alpha0_0d125.sh -> α₀ = 0.125
# scripts/esolm/train_owt_esolmb_alpha0_0d25.sh  -> α₀ = 0.25
# scripts/esolm/train_owt_esolmb_alpha0_0d5.sh   -> α₀ = 0.5
# scripts/esolm/train_owt_esolmb_alpha0_1.sh     -> α₀ = 1.0 (pure diffusion)
```

In `trainer_base.py`, a noise scheduler is defined:
```python
class LogLinear:
    def __init__(self, alpha_0=1):
        self.alpha_0 = alpha_0
    
    def alpha_t(self, t):
        alpha_t = self.alpha_0 * (1 - t)  # Linear noise schedule
```

This hybrid enables **interpolation between AR and MDM** in terms of quality and speed, and introduces additional flexibility: for example, the first and last words of a sentence can be determined in parallel (leveraging global context), while details in the middle are refined sequentially. Eso-LM explicitly combines the strengths: high modeling quality (the AR component ensures smooth transitions, especially for complex fragments) and high speed over most of the sequence (the parallel MDM component saves time).

**Unified architecture and attention:** The key challenge is training a **single transformer** capable of operating in both diffusion and autoregressive modes. Typically, these requirements conflict: AR requires causal masking (a token sees only preceding tokens), while MDM requires full attention over all tokens (masks can be uncovered in any order). The authors of Eso-LM solved this problem with a **custom attention mechanism** using mask $A$.

**Implementation of custom attention masks:**
```python
# models/dit.py - implementation of various mask types
@lru_cache
def _causal_mask(b, h, q_idx, kv_idx):
    """Causal mask for AR mode"""
    causal = q_idx >= kv_idx
    return causal

@lru_cache  
def _bidirectional_mask(b, h, q_idx, kv_idx):
    """Full attention for MDM mode"""
    bidirectional = q_idx == q_idx  # Always True
    return bidirectional

@lru_cache
def _mixed_mask(b, h, q_idx, kv_idx, cutoffs):
    """Mixed mask for EsoLM"""
    causal = q_idx >= kv_idx
    block_identity = q_idx >= cutoffs[b]
    return causal | block_identity

# Usage in EsoLMDiT:
def _get_attention_mask(self, seq_len, attn_mode=None, cutoffs=None):
    if attn_mode == 'causal':
        return _get_causal_mask(seq_len)
    elif attn_mode == 'bidirectional':
        return _get_bidirectional_mask(seq_len)
    elif attn_mode == 'mixed':
        return _get_mixed_mask(seq_len, cutoffs)
```

A matrix of attention offsets $A_{i,j}$ is introduced into the transformer, where $A_{i,j} = 0$ permits attention from position $i$ to $j$, and $-\infty$ prohibits it. By configuring this matrix, any attention pattern can be emulated. For example, for the AR component, $A$ defines a triangular causal mask, while for the MDM component, it enables full attention between "clean" tokens and restricted attention for masks. Specifically, **Eso-LM (A)** removes bidirectional attention *between masks* during the diffusion phase. This means masks do not "see" each other during denoising—thereby eliminating redundant dependencies and enabling faster transformer operation. The authors further enhance this efficiency with *sparse attention*: on each diffusion step, only the masks selected for unveiling at that step, plus all already unveiled tokens, are processed—not all positions.

**Implementation of token sorting for two-phase generation:**
```python
# algo.py - EsoLM._sort_indices()
def _sort_indices(self, indices, shuffle, keep_masks_unshuffled=False):
    """Sorting to determine token generation order"""
    masked = (indices == self.mask_index)
    
    if shuffle:
        # Random offsets for diffusion phase
        offsets = torch.rand(indices.shape).to(indices.device) * 0.9
        
        if keep_masks_unshuffled:
            # For AR phase: strict left-to-right order for masks
            offsets[masked] = torch.linspace(
                0, 1, torch.sum(masked)).to(indices.device)
    else:
        # Fixed order
        offsets = torch.linspace(0, 0.9, indices.shape[1]).to(indices.device)
    
    # Sorting: masks come first + offsets
    sort_idx = (masked + offsets).argsort(descending=False)
    return sort_idx
```

This approach significantly reduces costs for long sequences: instead of processing all 10k tokens, for example, only 1k masks are updated, leaving 9k unchanged. **Eso-LM (B)** goes further: it introduces a causal constraint even on "clean" tokens during diffusion, enabling **KV caching** for them. In simpler terms, variant B sacrifices part of the bidirectional context (clean tokens see only preceding clean tokens), but in return can store their representations and avoid recomputing them at each step. This yields additional speedup—estimates suggest that KV caching in the diffusion phase increases speed by up to **65%** compared to a baseline MDM without caching.

**Implementation of KV caching:**
```python
# models/dit.py - DDiTBlock with KV caching
class DDiTBlock(nn.Module):
    def reset_kv_cache(self):
        """Reset KV cache"""
        self.k_cache = None
        self.v_cache = None
    
    def _process_and_update_kv(self, k, v, num_clean):
        """Update KV cache only for clean tokens"""
        if self.k_cache is None:
            self.k_cache = k[:, :num_clean]
            self.v_cache = v[:, :num_clean]
        else:
            # Concatenate with previous cache
            self.k_cache = torch.cat([self.k_cache, k[:, :num_clean]], dim=1)
            self.v_cache = torch.cat([self.v_cache, v[:, :num_clean]], dim=1)
    
    @torch.no_grad()
    def _attention_with_kv_cache(self, qkv, rotary_cos_sin, num_clean, num_clean_and_mask):
        """Attention with KV cache"""
        # num_clean: number of clean tokens
        # num_clean_and_mask: clean + masks for generation
        
        # Apply rotary embeddings and split into q, k, v
        qkv = split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Update cache
        self._process_and_update_kv(k, v, num_clean)
        
        # Use cached values for attention
        attention_output = fused_flex_attention(
            q[:, :num_clean_and_mask], 
            self.k_cache, 
            self.v_cache,
            mask=None
        )
        return attention_output
```


**Unified architecture and attention:** The key challenge is training a **single transformer** capable of operating in both diffusion and autoregressive modes. Typically, these requirements conflict: AR requires causal masking (a token sees only preceding tokens), while MDM requires full attention over all tokens (masks can be uncovered in any order). The authors of Eso-LM solved this problem with a **custom attention mechanism** using mask $A$. A matrix of attention offsets $A_{i,j}$ is introduced into the transformer, where $A_{i,j} = 0$ permits attention from position $i$ to $j$, and $-\infty$ prohibits it. By configuring this matrix, any attention pattern can be emulated. For example, for the AR component, $A$ defines a triangular causal mask, while for the MDM component, it enables full attention between "clean" tokens and restricted attention for masks. Specifically, **Eso-LM (A)** removes bidirectional attention *between masks* during the diffusion phase. This means masks do not "see" each other during denoising—thereby eliminating redundant dependencies and enabling faster transformer operation. The authors further enhance this efficiency with *sparse attention*: on each diffusion step, only the masks selected for unveiling at that step, plus all already unveiled tokens, are processed—not all positions. This approach significantly reduces costs for long sequences: instead of processing all 10k tokens, for example, only 1k masks are updated, leaving 9k unchanged. **Eso-LM (B)** goes further: it introduces a causal constraint even on "clean" tokens during diffusion, enabling **KV caching** for them. In simpler terms, variant B sacrifices part of the bidirectional context (clean tokens see only preceding clean tokens), but in return can store their representations and avoid recomputing them at each step. This yields additional speedup—estimates suggest that KV caching in the diffusion phase increases speed by up to **65%** compared to a baseline MDM without caching. The slight quality degradation (perplexity) is acceptable: variant B shows slightly worse PPL than A, but still better than pure diffusion models, and is the **only** known model capable of caching in parallel generation.

**Architecture summary:** Eso-LM demonstrated that a **symbiotic fusion of AR and diffusion** is possible within a single transformer. In practice, both variants achieve the **best perplexity to date among diffusion models** (on LM1B and OpenWebText datasets) and provide a continuous quality/speed spectrum between AR and MDM. Notably, Eso-LM (A) approaches the PPL of pure AR models while generating significantly faster, and Eso-LM (B) trades a slight PPL reduction for being the **fastest** (thanks to caching). In speed benchmarks, Eso-LM outperforms prior diffusion models, reaching a new **Pareto frontier of quality and speed** (i.e., no prior model achieved the same perplexity at the same speed). With few generation steps, the hybrid does not suffer collapse (unlike some simplified interpolation schemes); with many steps, it produces higher-quality samples than all prior diffusion LMs. These results position Eso-LM as a critical benchmark for future architectures: the combination of parallel coarse generation followed by sequential refinement may be the optimal path forward.

## Mathematical and Probabilistic Formalization

All discussed diffusion language models aim to model the *probability distribution over text*, analogous to traditional LLMs, but via a **Markovian diffusion process** rather than autoregressive factorization.

Formally, the goal is to maximize the probability $P_\theta(X)$ for a sequence $X = (x_1, \dots, x_L)$ from the training corpus. In autoregressive models, factorization over tokens is applied:  
$$
P(X) = \prod_{i=1}^L P(x_i \mid x_{<i})
$$  
and training reduces to minimizing cross-entropy of next-token prediction.

<details> 
    <summary><em><strong>Cross-Entropy</strong></em></summary>

## Cross-Entropy (Cross-Entropy Loss)

The cross-entropy loss function is a central loss function in **classification** tasks, particularly binary and multiclass. It is closely tied to **maximum likelihood estimation** and fundamental concepts in information theory.

### 1. Classification Problem Setup

Let the training set be:

$$
D = \{(x_i, y_i)\}_{i=1}^n,\quad x_i \in \mathcal{X} \subseteq \mathbb{R}^d,\ y_i \in \{1, 2, \dots, K\}
$$

Goal: Find a parameterized function $f_\theta(x)$ that approximates the class probability distribution:

$$
f_\theta(x) = \hat{\mathbf{p}}(x) = (\hat{p}_1(x), \hat{p}_2(x), \dots, \hat{p}_K(x)), \quad \sum_{k=1}^K \hat{p}_k(x) = 1,\ \hat{p}_k(x) \ge 0
$$

(e.g., softmax output).

Let $y_i$ be the true class; then the target one-hot label vector is:

$$
\mathbf{y}_i = (0,\dots, 1, \dots, 0), \text{ where 1 is at the } y_i\text{-th position}
$$

### 2. Definition of Cross-Entropy

Cross-entropy between the true distribution $P$ and predicted distribution $Q$:

$$
\boxed{
H(P, Q) = -\sum_{k=1}^K P(k) \log Q(k)
}
$$

In supervised learning context:

* $P(k) = \mathbb{I}[y_i = k]$ — one-hot distribution;
* $Q(k) = \hat{p}_k(x_i)$ — predicted probability;

- Intuitively, cross-entropy measures how far our predictions deviate from true values;
- The logarithm transforms products of probabilities into sums, simplifying computation and improving numerical stability;
- The negative sign ensures the loss is positive, since log probabilities are non-positive; summing them yields a negative value, which we flip to obtain a minimizable positive loss.

Then:

$$
\text{Loss}(x_i, y_i) = - \log \hat{p}_{y_i}(x_i)
$$

Over the entire dataset:

$$
\boxed{
\mathcal{L}_{CE}(\theta) = -\frac{1}{n} \sum_{i=1}^n \log \hat{p}_{y_i}(x_i)
}
$$

### 3. Interpretations

#### (a) Information Theory

Cross-entropy measures the **average number of bits** required to encode true labels $P$ using a code based on distribution $Q$:

* If $Q \approx P$, then $H(P,Q) \approx H(P)$ — the entropy.
* If $Q$ diverges significantly from $P$, then $H(P,Q)$ increases.

> ⇒ **Minimizing cross-entropy ⇔ maximizing accuracy of probability predictions.**

#### (b) Probabilistic Interpretation

Assume the model predicts probabilities $Q = f_\theta(x)$, and data labels $y_i$ are independent. Then:

$$
\log L(\theta) = \sum_{i=1}^n \log P(y_i | x_i, \theta) = \sum_{i=1}^n \log \hat{p}_{y_i}(x_i)
$$

Thus:

$$
\boxed{
\mathcal{L}_{CE} = - \log L(\theta)
}
$$

That is, **cross-entropy is the negative log-likelihood**. Hence, it arises naturally from the maximum likelihood estimation (MLE) principle.

#### (c) Relation to KL Divergence

Recall KL divergence:

$$
D_{KL}(P \| Q) = \sum_{k=1}^K P(k) \log \frac{P(k)}{Q(k)} = H(P, Q) - H(P)
$$

With one-hot labeling $P(k) = \delta_{ky}$ ⇒ $H(P)$ = 0 ⇒

$$
\boxed{
D_{KL}(P \| Q) = H(P, Q)
}
$$

<details> 
    <summary><em><strong>Cross-Entropy vs KL-Divergence</strong></em></summary>

## 🔍 **Optimizing Cross-Entropy vs KL-Divergence: From Simple to Complex**

When working with machine learning tasks, especially classification, we often need to measure how far a model's predictions deviate from ground truth. Two popular ways to do this are **cross-entropy (Cross-Entropy, CE)** and **KL-divergence (Kullback-Leibler Divergence, KLD)**.

At first glance, they seem very similar, but there are important distinctions. Let’s break it down step by step!

### **1️⃣ What is entropy $H(P)$?**

Entropy of a distribution $ P $ is a measure of its **uncertainty**. The formula:

$$
H(P) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

where:
- $ \mathcal{X} $ — all possible tokens (words/symbols),
- $ P(x) $ — probability of token $ x $ in the true distribution.

The higher $ H(P) $, the more "spread out" the distribution (greater uncertainty).

### **2️⃣ Example: Next-Token Prediction**

Suppose we have:
- **Context:** `"The cat is lying on ___"`
- **Possible next tokens:** `"rug"` (0.7), `"floor"` (0.2), `"sofa"` (0.1)

Then the **true distribution $ P $** might be:

#### **🔹 Case 1: One-hot (deterministic)**
If the correct token is only `"rug"`, then:

$$
P = [1, 0, 0]
$$

Entropy:

$$
H(P) = - \left( 1 \cdot \log 1 + 0 \cdot \log 0 + 0 \cdot \log 0 \right) = 0
$$

(since $ \lim_{p \to 0} p \log p = 0 $)

**Conclusion:**

- $ H(P) = 0 $ → no uncertainty.
- In this case, **KLD and CE are identical**:

$$
D_{KL}(P \| Q) = H(P, Q) - H(P) = H(P, Q)
$$

#### **🔹 Case 2: Probabilistic (soft)**
Suppose the correct tokens have probabilities:

$$
P = [0.7, 0.2, 0.1]
$$

Then entropy:

$$
H(P) = - (0.7 \log 0.7 + 0.2 \log 0.2 + 0.1 \log 0.1)
$$

Assume log base 2 (bits):

$$
H(P) \approx - (0.7 \cdot (-0.514) + 0.2 \cdot (-2.321) + 0.1 \cdot (-3.321)) \approx 1.157 \text{ bits}
$$

**What does this mean?**
- Entropy is **not zero**, meaning there is uncertainty in the correct answer.
- If the model predicts $ Q = [0.6, 0.3, 0.1] $, then:

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

Here, $ H(P, Q) $ is the cross-entropy, and $ H(P) $ is the "baseline" uncertainty of the data.

### **3️⃣ Cross-Entropy (Cross-Entropy, CE)**

#### **🔹 What is it?**

Cross-entropy measures how "surprising" the model’s predictions are relative to the true distribution. The lower the CE, the better the model predicts.

#### **🔹 Formula**

For discrete case (e.g., classification):

$$
H(P, Q) = -\sum_{i} P(i) \log Q(i)
$$

where:
- $ P $ — true distribution (usually one-hot encoded, e.g., $[0, 1, 0]$ for class 2).
- $ Q $ — predicted distribution (e.g., $[0.1, 0.8, 0.1]$).

#### **🔹 Characteristics**
✅ **Simplicity**: In ML, CE is often used because if $ P $ is one-hot, the formula simplifies to $ -\log Q(\text{true class}) $.  
✅ **Efficiency**: Gradients are easy to compute, accelerating training.

## **4️⃣ KL-Divergence (Kullback-Leibler Divergence, KLD)**

#### **🔹 What is it?**

KLD measures how much one distribution $ Q $ differs from another $ P $. It is **not a distance metric** (asymmetric: $ D_{KL}(P \| Q) \neq D_{KL}(Q \| P) $).

#### **🔹 Formula**

$$
D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} = H(P, Q) - H(P)
$$

where:
- $ H(P, Q) $ — cross-entropy between $ P $ and $ Q $,
- $ H(P) $ — entropy of $ P $ (measure of uncertainty).

#### **🔹 Characteristics**
✅ **Informational difference**: KLD shows how many extra bits are needed to encode $ P $ using $ Q $.  
❌ **Depends on $ H(P) $**: If $ P $ is fixed (e.g., one-hot), then $ H(P) = 0 $, and KLD becomes equal to CE!

### **5️⃣ Relationship Between CE and KLD**

From the KLD formula:
$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

#### **🔹 If $ P $ is one-hot (as in classification):**
- $ H(P) = 0 $ (entropy of a deterministic distribution is zero),
- Then **KLD = CE**!

#### **🔹 If $ P $ is not one-hot (e.g., smoothed labels):**
- $ H(P) > 0 $, so KLD and CE differ.
- Optimizing KLD accounts for the entropy of $ P $, while CE does not.

### **6️⃣ When to Use What?**

| **Criterion**       | **Cross-Entropy (CE)** | **KL-Divergence (KLD)** |
|--------------------|------------------------|--------------------------|
| **One-hot labels**  | ✅ Better (simpler and faster) | ⚠️ Same (KLD = CE) |
| **Smoothed labels** | ❌ Does not account for $ H(P) $ | ✅ Accounts for distributional difference |
| **Interpretation**  | "Surprise" of the model | "Informational cost" of error |

#### **🔹 Practical takeaway:**
- **In most classification tasks, CE and KLD are equivalent** (since labels are one-hot).
- **If labels are probabilistic (e.g., soft targets in distillation) — KLD is better**, as it accounts for the entropy of the true distribution.

### **🎯 Summary**

- **Cross-entropy** is the model’s "surprise" relative to true labels.
- **KL-divergence** is the "cost" of using $ Q $ instead of $ P $.
- **If $ P $ is deterministic (one-hot) → CE = KLD.**
- **If $ P $ is probabilistic → KLD accounts for its entropy.**

Now you know the difference and can consciously choose your loss function! 🚀

</details>

### 4. Special Cases

#### 🔹 Binary Cross-Entropy (Logistic Regression)

If $y_i \in \{0, 1\}$ and $f(x_i) = \hat{p}_i \in (0, 1)$, then:

$$
\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]
$$

### 5. Properties of Cross-Entropy

| Property              | Description                                                                        |
| --------------------- | ---------------------------------------------------------------------------------- |
| 📈 Convexity          | If $\hat{p}_k$ is an affine function of parameters, $\mathcal{L}_{CE}$ is convex. |
| ⚙️ Smoothness         | Differentiable w.r.t. $\hat{p}_k$, suitable for gradient descent.                |
| 🎯 Interpretability   | Loss equals $-\log$ of predicted probability of the correct class.               |
| ⚠️ Sensitivity        | Heavily penalizes high confidence in incorrect classes.                          |

> **Example:** If the correct class is predicted with probability 0.9: $-\log(0.9) ≈ 0.105$, and if 0.01 — $-\log(0.01) ≈ 4.6$.

### 6. Gradient of Cross-Entropy

Consider the softmax output of the model:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}},\quad z_k = \text{logits}
$$

And the loss function:

$$
\mathcal{L}_{CE}(z, y) = -\log \hat{p}_y
$$

Gradient w.r.t. $z_j$:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = \hat{p}_j - \mathbb{I}[j = y]
$$

That is:

$$
\nabla_z \mathcal{L} = \hat{\mathbf{p}} - \mathbf{y}
$$

> 💡 This is very convenient: the gradient is simply the difference between the predicted distribution and the true one-hot vector.

### 7. Practical Aspects

**When to apply:**

* Classification (binary / multiclass / multi-label).
* When accurate probability estimation is important.
* When inference is based on MLE.

**When to avoid:**

* Tasks with noisy labels: cross-entropy is sensitive to label noise.
* Class imbalance without correction (risk of over-representing frequent classes).

### 8. Conclusion

Cross-entropy is a fundamental loss function in classification, rigorously derived from probability and information theory. Its convexity, simple gradient, and interpretability make it the preferred choice in most supervised learning tasks. However, its sensitivity to overconfident errors requires careful model monitoring and tuning.

> 🧪 In real-world tasks, its variants are often applied: **focal loss** (to combat imbalance), **label smoothing**, **soft targets**, **weighted CE**, etc.

</details>

---

In diffusion models, a *latent variable* — the **diffusion step index $t$** — is introduced, and two processes are defined:
- **Forward (forward)** $q$ — gradual addition of noise to the data,
- **Reverse (reverse)** $p_\theta$ — a learnable process of noise removal.

The forward process is constructed so that at $t = T$, the data is completely destroyed (e.g., all tokens are masked or replaced with uniform noise), and at $t = 0$, the data is clean (original text). Specifically, at each step $t \to t-1$, we define the distribution $q(x_{t-1} \mid x_t, X)$, where $X$ is the original data.

In discrete diffusion models, **masking** is typically used as the noise mechanism: with an increasing probability $\beta_t$, each token is independently replaced with `[MASK]`. At the limit $t = T$, we obtain a fully masked sequence (complete noise). This forward process $q$ is known exactly (mask distribution is fixed). According to diffusion theory, the *posterior* distribution $q(x_{t-1} \mid x_t, X)$ can also be derived analytically (for masking, it is proportional either to a delta function on the true token or on the mask).

The reverse process $p_\theta$ is parameterized by a neural network. It must approximate the restoration distribution: $p_\theta(x_{t-1} \mid x_t)$ — the probability of obtaining a less noisy state at step $t-1$, given the state at step $t$. To make the task tractable, it is typically assumed that this distribution is **fully factorized** over positions containing noise at step $t$. For example, if at step $t$ the masked indices are $\mathcal{M}_t$, then  
$$
p_\theta(x_{t-1} \mid x_t) = \prod_{i\in \mathcal{M}_t} p_\theta(x^{(i)}_{t-1} \mid x_t)
$$  
— the model predicts each masked token independently (conditionally on the current state of the entire sequence $x_t$). This is exactly equivalent to the task of **masked language modeling** on the current context $x_t$.

<details> 
    <summary><em><strong>The Essence of a Diffusion Model for Text</strong></em></summary>

## **The Essence of a Diffusion Model for Text:**

Imagine you have a meaningful text. The **"diffusion"** process gradually **corrupts** this text, for example, by replacing random words (tokens) with "masks" (`[MASK]`) or pure noise. This is similar to someone erasing words from a page. The model’s ultimate goal is to learn the **reverse process**: take this noisy, "corrupted" text and **restore** the original meaningful version.

**Key component: reverse process ($p_\theta$)**

*   **What is it?** This is the model’s "brain" for restoration. It is a neural network (parameterized by weights $\theta$) that **learns to predict what the text looked like on the previous, *less* noisy step ($x_{t-1}$), given the current, *more* noisy state ($x_t$)**.
*   **Formally:** $p_\theta(x_{t-1} \mid x_t)$ is a *conditional probability distribution*. It answers: "If the current text looks like $x_t$ (contains some noise/masks), what is the probability that on the previous step it looked like $x_{t-1}$?"
*   **Training goal:** Adjust the neural network’s parameters $\theta$ so that this distribution $p_\theta$ is as close as possible to the *true* (but unknown) restoration distribution $q(x_{t-1} \mid x_t)$.

**Problem: Too complex!**

*   Predicting the *entire* vector $x_{t-1}$ (representing the whole sequence of words/tokens) at once, based on $x_t$ — is an *extremely* difficult task. It would require the model to consider all possible word combinations across the entire sequence simultaneously. Computationally infeasible.

**Solution: Full factorization (conditional independence)**

*   To make the task **tractable**, a key **assumption** is introduced:
    > **Assume that predicting each *individual* noisy token at position $i$ for step $t-1$ ($x^{(i)}_{t-1}$) depends *only* on the *current* noisy state of the *entire* sequence ($x_t$), but *not* on what we predict for *other* noisy positions ($j \neq i$) on this *same* step $t-1$.**

*   **In simpler terms:** The model looks at the *entire* current noisy text $x_t$ (including known *unnoised* words and current masks/noise). Based on this **full context** $x_t$, it **independently** predicts what should be placed at the location of *each individual* mask/noise to obtain $x_{t-1}$.
*   **Analogy 1 (Crossword):** Imagine $x_t$ is a crossword puzzle where some cells are filled with letters (unnoised tokens) and others are blank (masks `[MASK]`). The model looks at the *entire* crossword — and word intersections — and predicts what letter should go in *each* blank cell ($x^{(i)}_{t-1}$). The factorization assumption says: predicting the letter for cell (1,1) depends on *all* already filled cells around it, but *not* directly on what you predict *simultaneously* for cell (5,5). You predict them independently, but each prediction relies on the *same* overall context (the entire current crossword $x_t$).
*   **Analogy 2 (Chess):** Imagine a chessboard $x_t$, where some pieces are on their starting positions (unnoised tokens) and some squares are empty (masks). The model must restore the previous state of the board $x_{t-1}$, where pieces may have been on the empty squares. It looks at the *entire* current board (location of remaining pieces) and **independently** decides which piece is *most likely* to have been on *each* empty square before it was removed. The decision for one square depends on the overall layout, but not directly on the decision for another square *at this exact moment*.

**Mathematical expression:**

*   Let $\mathcal{M}_t$ denote the **set of indices** in the sequence that are **noised** (masked or distorted) at the current step $t$.
*   The full factorization assumption allows us to write the complex joint restoration distribution as a **product** of independent distributions for **each noised position**:
    $$
    p_\theta(x_{t-1} \mid x_t) = \prod_{i \in \mathcal{M}_t} p_\theta(x^{(i)}_{t-1} \mid x_t)
    $$
*   **What this means:**
    *   $\prod_{i \in \mathcal{M}_t}$: Multiply probabilities over all positions $i$ that are noised at step $t$.
    *   $p_\theta(x^{(i)}_{t-1} \mid x_t)$: Probability that at position $i$ in the *less* noised state $x_{t-1}$, a *specific* token (word, letter) $x^{(i)}_{t-1}$ appears, **given that we observe the *entire* current noised text $x_t$**.
    *   **Key point:** The distribution for position $i$ ($p_\theta(x^{(i)}_{t-1} \mid x_t)$) depends **ONLY** on $x_t$ (the full current context), but **NOT** on what the model predicts for $x^{(j)}_{t-1}$ (another noised position $j$) when computing *this same* distribution $p_\theta(x_{t-1} \mid x_t)$. They are considered **conditionally independent** given fixed $x_t$.

**Connection to Masked Language Modeling (MLM) a la BERT:**

*   **This is the most critical analogy!** Look closely at $p_\theta(x^{(i)}_{t-1} \mid x_t)$.
*   **What is this?** It is the task of predicting **one** token (the one that was at position $i$ in $x_{t-1}$) based on the **entire** current context $x_t$.
*   **How does this look in practice?** At step $t$, we have a sequence $x_t$, where at positions $\mathcal{M}_t$ are masks `[MASK]` (or other noise symbols). The neural network ($p_\theta$) takes $x_t$ as input and for **each** position $i$ in $\mathcal{M}_t$, outputs a probability distribution ($p_\theta(x^{(i)}_{t-1} \mid x_t)$) over the *entire vocabulary* — which word/token with what probability should replace *this specific mask* to obtain state $x_{t-1}$.
*   **This is exactly the task of Masked Language Modeling (MLM)!** The same task solved by models like BERT. BERT receives text with masks and learns to predict the original words under the masks, using the context of the *entire* sentence (and $x_t$ in diffusion is precisely such a masked context).

**Optimization (Training) via Cross-Entropy:**

*   How do we train models like BERT to solve MLM? We use **cross-entropy loss**.
*   **How this works in diffusion:**
    1.  During training, for **each** diffusion step $t$ and for **each** noised position $i \in \mathcal{M}_t$ at that step, we know the **original, correct token** that was at this position *before* noise was applied (this is $x^{(i)}_{t-1}$).
    2.  The neural network ($p_\theta$) for position $i$ outputs a **predicted probability distribution** $p_\theta(x^{(i)}_{t-1} \mid x_t)$ over all possible tokens.
    3.  We compute the **cross-entropy** between:
        *   **Ideal distribution:** probability 1.0 for the *correct* token and 0.0 for all others.
        *   **Predicted distribution:** $p_\theta(x^{(i)}_{t-1} \mid x_t)$ from the model.
    4.  This cross-entropy measures how well the model predicted the correct token *for this specific position $i$*.
    5.  Losses from *all* noised positions $i \in \mathcal{M}_t$ at step $t$ are **summed** (or averaged). This is the total loss for step $t$.
    6.  Gradients of this total loss are backpropagated through the neural network, updating its weights $\theta$ to improve predictions.
*   **In summary:** Training the reverse diffusion process at *each* step $t$ **reduces to solving multiple independent Masked LM tasks on context $x_t$**, optimized using familiar **cross-entropy** for each mask.

**Key Takeaways:**

1.  **Reverse process ($p_\theta$)** — a neural network that learns to "fix" text step by step.
2.  **Factorization ($p_\theta = \prod p_\theta(...)$)** — a *necessary simplification* making training possible. It means: "Predict each mask independently, but use the *entire* current text as context for each prediction."
3.  **$p_\theta(x^{(i)}_{t-1} \mid x_t)$** — the **core of the process**. This is exactly the task solved by BERT (Masked LM): "What is hidden under this specific mask `[MASK]` in this context $x_t$?"
4.  **Cross-Entropy** — the standard and efficient way to *train* the neural network to solve many such MLM tasks *in parallel* on a single diffusion step $t$.

</details>

---

Thus, *each diffusion step is optimized using familiar cross-entropy* on the correct token instead of the mask.** However, unlike BERT, masking is applied repeatedly at multiple levels, so weighting coefficients are introduced per step. As a result, the loss function takes the form of **Negative ELBO (NELBO)** — the negative variational lower bound on the data log-likelihood. In Sahoo et al. (2024), for discrete diffusion, the NELBO expression was derived as a sum over losses on masked positions:

$$
ELBO = \mathcal{L}_{\text{diff}} = \sum_{t=1}^T w_t \, \mathbb{E}_{x_t \sim q}\left[ -\log p_\theta\left(x_{t-1}^{\mathcal{M}_t} = X^{\mathcal{M}_t} \mid x_t\right) \right],
$$

where:
- $X^{\mathcal{M}_t}$ — true tokens at positions masked in state $x_t$;
- Coefficients $w_t$ depend on the chosen noise schedule (e.g., $w_t = 1$ for all $t$ in the simplest case, or increasing/decreasing to reflect the importance of each step).

Intuitively, the model learns to *simultaneously predict masks of varying "depths"* — when 10% of the text is masked, 20%, ..., up to 100%. At the limit $t = T$, it solves the task "guess the entire text from context = zero" (which is nearly impossible, but this term trains the network to produce a reasonable prior distribution).

All described models (Gemini, Mercury, LLaDA) follow this paradigm with some variations. For example, **Mercury**, through the concept of *score entropy*, effectively also optimizes an analog of ELBO, but not directly through token log-likelihoods, rather by learning to *restore probability ratios* (i.e., instead of predicting $P(y)$, the model estimates $\log \frac{P(y)}{P(x)}$ for tokens $y$ and current $x$). It has been shown that this approach is equivalent to a new formulation of score matching in discrete space and provides more stable training than direct token prediction. Nevertheless, the outcome is the same: Mercury learns to restore masked sequences over several steps by optimizing maximum likelihood (via ELBO) and using variants of cross-entropy loss to update weights.

Notably, **Eso-LM** introduced a novel likelihood model unifying AR and diffusion. If $z_0$ denotes a partially generated MDM sequence (with masks) and $x$ the final text, the full decomposition distribution can be written as a mixture:

$$
P_\theta(x) = \sum_{z_0} P_\theta(x \mid z_0)\,P_\theta(z_0),
$$

where:
- $P_\theta(z_0)$ — probability of obtaining the draft $z_0$ via the diffusion component, and $P_\theta(x \mid z_0)$ — probability of extending it to $x$ via the AR model. Direct computation of this sum is infeasible, so the authors applied a variational approach: they introduced a simple posterior distribution (which masks random tokens from the full $x$ to obtain $z_0$) and derived the **ELBO for the hybrid generator**.

Interestingly, the resulting loss function again decomposes into two terms:
1. NELBO of the diffusion component (sum of masked LM losses over steps, as above),
2. Standard autoregressive loss on tokens remaining on the AR stage (also cross-entropy).

This means Eso-LM can be trained via a single end-to-end procedure: for each example, first apply **stochastic masking** (to isolate future AR tokens), then simulate diffusion to restore the rest, and finally add the AR model’s loss on the remaining tokens. This approach preserves probabilistic grounding (a lower bound on log-likelihood exists) and enables efficient training of a unified transformer on the combined task.

Overall, mathematically, diffusion LLMs expand the solution space for modeling $P(X)$. They confirm the general principle that **key properties of LLMs (scalability, context-based learning, instruction following)** are not inherently tied to autoregression, but to the more fundamental power of generative modeling via maximum likelihood. Diffusion models, by optimizing ELBO, implement the same principles through a different factorization. The distinction lies only in that AR models are a special case ($T = L$, each $x_{t-1}$ contains one new token), while DLMs are the general case ($T < L$ or even $T \ll L$, with parallel updates).

## Experimental Results

### Generation Quality and Metrics

Despite radically different mechanisms, modern diffusion LMs already achieve text quality comparable to transformers across many metrics. Authors of LLaDA reported that their 8-billion-parameter model **matches autoregressive LLMs of similar size in perplexity and zero-shot tasks**. In particular, LLaDA-8B outperformed LLaMA2-7B in nearly all 15 standard zero-shot/low-shot learning tasks and **approached the level of LLaMA3-8B**. After instruction fine-tuning, LLaDA demonstrated confident instruction following and dialogue skills comparable to strong LLMs of similar scale. On applied benchmarks, LLaDA also performed convincingly: for example, on the knowledge exam MMLU and the math problem set GSM8K, its results matched those of the autoregressive baseline trained on the same data. Moreover, the diffusion model **solved the “reversal curse” problem**, handling reversed poetry better than GPT-4(open). However, it should be noted that LLaDA has been evaluated only up to 8B. No direct figures exist for larger scales — perhaps AR models still lead at hundreds of billions of parameters, but the study suggests the gap is not fundamental.

**Gemini Diffusion** (Google) in its current implementation is comparable in quality to the Gemini 2.x family. According to DeepMind, on several external tests, **Gemini Diffusion achieves quality comparable to much larger models**, while being faster. On code benchmarks, the diffusion Gemini nearly matches the autoregressive Gemini Flash-Lite: for example, HumanEval (percentage of successful program solutions on first attempt) \~**89.6% vs 90.2%**, MBPP (Python benchmark) \~76.0% vs 75.8% — effectively on par. On BigCodeBench, parity is also observed \~45.4% vs 45.8%. Only on some complex tasks did the older architecture outperform the new: for instance, Gemini Flash-Lite performed better on the multilingual MMLU test (79.0% vs 69.1%) and on complex logical tasks in BIG-Bench Hard (21.0% vs 15.0%). However, Gemini Diffusion unexpectedly outperformed autoregression on the mathematical test AIME 2025 (23.3% vs 20.0%). On scientific Q\&A (GPQA Diamond), the diffusion model lagged (40.4% vs 56.5%), indicating a need for further fine-tuning on facts and knowledge. Overall, **the quality gap between diffusion and AR is minimal** at this stage. According to the lead engineer, “on metrics for relatively small models, the difference is virtually eliminated.” O’Donoghue (Google DeepMind) notes that in some areas diffusion already has advantages — for example, tasks requiring **global consistency** (programming, complex reasoning) may benefit from the non-local attention of the diffusion approach.

**Mercury Coder** was oriented toward code tasks and achieved impressive results given its small size. In independent tests, Mercury Coder Small outperformed models such as Gemini 2.0 Flash-Lite, Claude 3.5 Haiku, GPT-4o Mini, and Qwen 2.5 Coder (7B) on **at least 4 of 6** standard programming benchmarks. Mercury Coder Mini (a larger version) surpassed these competitors on at least 2 of 6 task sets. Among the metrics used were HumanEval (code generation from description), MBPP (Python tasks), MultiPL-E (multilingual HumanEval), HumanEval+ (extended version), and coding competitions. Notably, Mercury lost only to the specialized model **DeepSeek Coder V2 Lite**, which led on all 6 tests. This indicates that diffusion models can already compete with top-optimized AR models in a narrow domain (code generation) — though there is still room to reach absolute leadership. Mercury’s qualitative performance is also evaluated via aggregate metrics “accuracy vs speed”: in the space of “solution score vs tokens/second,” Mercury resides in a **preferably advantageous zone (Pareto-optimal)**, delivering both high solution scores and performance. Visually, on graphs, Mercury Coder achieves code quality close to GPT-4o but with an order-of-magnitude faster generation time.

**Eso-LM** was primarily measured by perplexity (language model quality) and sampling characteristics. On classic datasets LM1B and OpenWebText, Eso-LM (A) achieved perplexities of 26.2 and ~30.5 respectively, significantly better than prior diffusion LMs on these corpora. This is the **best result among nonlinear (non-AR) methods** on these datasets. The transformer still leads slightly (e.g., a GPT-2 analog achieves ~23.0 PPL on OWT), but the gap has narrowed. Moreover, including the sequential phase allows Eso-LM to surpass pure diffusion models not only in perplexity but also in *quality of generated text*. The authors conducted comparative sample evaluations: at equal diffusion steps, Eso-LM generated **more meaningful and diverse texts**, without exhibiting mode collapse (observed in some simplified schemes like BD3-LM at low step counts). Also noted is that Eso-LM can flexibly adjust the proportion of parallel and sequential generation to achieve the desired trade-off. For example, variant B (with caching) slightly lags behind variant A in perplexity but is **much faster** on long sequences, so on practical generation of long text (e.g., 1–2 thousand characters), it may deliver a better overall result (if measuring “perplexity per second”).

### Speed, Latency, and Efficiency

The main advantage of diffusion LLMs is **generation speed**, especially noticeable on long outputs. Practical tests show DLMs can deliver responses several times faster than comparable autoregressive models. For example, **Gemini Diffusion** responds to complex queries (such as generating an HTML application with code) in **1–3 seconds**, whereas autoregressive Gemini 2.5 Flash takes ~7 seconds for the same request. Measured throughput of Gemini Diffusion varies from 600 to 1300 tokens/s depending on the task. Maximum claimed values (1000–2000 tokens/s) far exceed classical LLMs: for comparison, GPT-3/4 typically do not exceed 100–200 tokens/s even with optimized decoding, and many large models output only dozens of tokens per second. **Mercury** broke a record of sorts, achieving ~1100 tokens/s on H100 (for the Mini version), comparable to the best specialized hardware accelerators (previously such figures were only achieved on non-standard chips like Groq or SambaNova). Importantly, Mercury runs on standard GPUs, highlighting the **algorithmic efficiency**: as Ermon notes, the model loads GPU much more fully than AR-LMs, eliminating idle time between sequential steps.

From the perspective of *response latency*, diffusion models are especially advantageous when generating **large text fragments**. An AR model must generate each of the N tokens sequentially — time grows linearly $O(N)$. DLMs, however, can generate very long texts in a fixed number of iterations (e.g., 20–50), so theoretical complexity is closer to $O(\text{const})$. In practice, complexity still increases, but more slowly: for example, Gemini Diffusion generates 10k tokens in seconds (a few iterations over 10k tokens at once), whereas 10k tokens from autoregression require thousands of decoder steps. Therefore, for scenarios requiring **low latency with long output length**, diffusion LMs open new possibilities. Such scenarios include: streaming chatbot interactions (fast response without “typing” letter-by-letter), code autocompletion systems in IDEs (need near-instant output of large code chunks), live speech translation and transcription, generation of long narratives, etc.

However, *time-to-first-token (TTFT)* for diffusion models is typically higher than for AR. This is because AR can immediately output the first symbol with almost no delay (after one step), whereas diffusion models must complete the entire iteration cycle before the full answer is ready. Users may perceive this as a slight pause before the response, but then the entire answer appears instantly. In interactive applications, this requires a different UX — for example, instead of showing “typing” character-by-character, diffusion systems can display a generation progress indicator and then instantly output the full text.

Regarding *energy efficiency*, direct measurements are still limited, but some inferences can be drawn. Diffusion generation, though requiring multiple iterations, **parallelizes better and utilizes computational resources more effectively**. In AR decoding, due to its sequential nature, modern GPUs often idle (only one “condition” is loaded per step, leaving other tensor capacities unused). DLMs, conversely, process large volumes of data at each step (the entire block at once), which better fills the GPU. Inception Labs claim Mercury reduces inference cost and makes it more predictable (no timing spikes on long queries). One can expect that *energy per token* for diffusion models will be lower than for AR, precisely due to higher throughput — the model performs slightly more operations but outputs an order of magnitude more symbols per second, meaning **operation per token is cheaper**. Furthermore, adaptive mechanisms, such as variable numbers of steps depending on task complexity, allow DLMs to conserve resources: simple queries require only a few iterations (saving time and energy), while complex ones automatically trigger more steps, allocating more computation where needed. Such *adaptive compute* is difficult to implement in standard LLMs, which always run the full decoder for every new token and cannot “speed up” on easy segments. Thus, diffusion approaches have potential to be more energy-efficient at scale, though precise figures will emerge as further research progresses.

The table below summarizes key results on quality and speed for the discussed models:

| Model                        | Speed (tokens/s)                     | Code Quality                      | Features                      |
|---------------------------- |--------------------------------------|-----------------------------------|-------------------------------|
| **Gemini Diffusion (Google)** | ~1000 tokens/s                       | HumanEval ~90%                    | —                             |
| **Mercury Coder (Inception)** | 737 (Small), 1109 (Mini) on H100     | On par with GPT-4o Mini and Claude 3.5 | **5–10× faster** than analogs |
| **LLaDA 8B**                | No exact figures (potential $T ≪ L$)| —                                 | Strong in zero-shot and instructions, outperforms GPT-4o in "reverse" tasks |
| **Eso-LM (A/B)**            | Up to 65% speedup (B)                | Close to AR                       | Best speed/quality trade-off at ~1K token lengths |

## Comparison with Classical Transformers

### Advantages of the Diffusion LM Paradigm:

**High speed and low latency on long outputs.** 
- Parallel generation of text blocks allows DLMs to deliver responses an order of magnitude faster than autoregressive models of similar size. This opens the path for using LLMs in **real-time applications**, where AR models were too slow (chatbots with complex responses, voice assistants, IDE helpers).

**Iterative improvement and self-correction.** 
- Unlike AR, diffusion models are not permanently bound to initial token choices — each subsequent step can correct earlier flaws. This reduces the risk of error accumulation and **hallucinations**: if the early-stage response veers off course, the model can “rewrite” problematic areas during denoising.

**Greater global text coherence.** 
- Thanks to bidirectional attention within generated blocks, DLMs consider future word context when generating earlier ones. This helps ensure phrase-end consistency with beginnings, formatting compliance, tense and number agreement across the text, etc. — areas where AR models sometimes struggle due to their left-context limitation.

**Flexibility and editability.** 
- Diffusion models naturally handle filling text gaps and inserting fragments, as their training is based on masked-language modeling. Thus, they are naturally suited for tasks of **interactive editing, mid-text completion, and stylistic transformation of existing text**. For example, Gemini Diffusion has a mode for instant editing of arbitrary text with minimal prompting. In AR models, such capabilities require special tricks (instruction-tuning, infilling training) and remain less reliable.

**Adaptive computation.** 
- DLMs can dynamically adjust the number of generation iterations to the task: simple queries — fewer steps (resource and time savings), complex ones — more steps (quality enhancement). AR models always perform $L$ steps for $L$ tokens, regardless of task difficulty. Adaptability implies potentially **more efficient use of computation**, especially when deploying systems across diverse user queries.

### Limitations and Challenges:

**Complexity and cost of training.** 
- Training diffusion LLMs requires maintaining a delicate balance. Very large datasets and computational resources are needed — for example, LLaDA 8B was trained on 2.3 trillion tokens, consuming 130 thousand GPU hours. Moreover, optimizing ELBO with repeated masking is more complex than simple AR cross-entropy. Careful tuning of noise schedules, weighting coefficients, and possibly new techniques (like score entropy) for stability is required. All this makes DLM development more labor-intensive.

**Time-to-first-token.** 
- When used in interfaces where users expect streamed text (e.g., simulating typing responses), DLMs may appear “slow,” as nothing is output until the full block is complete. AR models begin outputting text almost immediately, token by token. This requires a shift in interaction paradigm: **DLMs are preferable where receiving the full answer at once is better than receiving it piecemeal**. For tasks where gradualness matters (e.g., word completion after the first letter), AR may retain an advantage.

**Memory and computation allocation.** 
- Diffusion models process the entire context at each step, which may require more memory (to store activations, though fewer steps). However, developments like Eso-LM (B) show this barrier can be overcome: caching is possible even in diffusion. Generally, if very long context is needed (e.g., 100k tokens), AR models face similar challenges. DLMs still need to prove efficiency on extremely large contexts.

**Accuracy on detailed tasks.**
- Although the gap is narrowing, in some cases AR models still show superior quality. In Gemini tests, the diffusion model lagged on questions requiring precise knowledge or multi-step logical reasoning (e.g., complex Big-Bench questions). This may be due to the currently limited size of diffusion models. Experiments with models >20B parameters are needed to confirm that DLMs maintain their trend of improvement and catch up to AR across all metrics.

### Potential to Replace AR Models:

Many experts believe the diffusion approach could become the **new standard** for LLMs in the future. Google openly states its intent to explore this path to reduce latency across all Gemini models. The emergence of Mercury in the commercial sphere signals that industry is seeking faster solutions for LLM deployment. Andrew Ng noted that the entry of diffusion models into text is “a cool attempt to explore diffusion models as an alternative, generating entire text at once... Congrats Stefano & team!” The advantages in speed and scalability are too significant to ignore. Of course, classical transformers will remain in use for a long time, especially given their mature ecosystem (optimized frameworks, hardware accelerators, extensive fine-tuning experience). But if DLMs continue progressing — especially proving themselves on 50B+ models — a **paradigm shift is entirely plausible**.

One can envision such an evolution: first, diffusion models will occupy niches critical for speed (code autocompletion, chats with strict response-time limits, mobile apps, perhaps real-time code generation). Then, as quality improves, they will begin displacing AR from other domains. Hybrid variants (like Eso-LM) may serve as a smooth bridge: leveraging the best of both approaches, they ease the transition. Eso-LM already demonstrates that it is possible to achieve a **spectrum of models with varying speed/accuracy ratios**, which is more valuable than a rigid “either fast or accurate” alternative.

## Conclusion and Outlook

Diffusion language models represent a promising step forward in NLP. While the autoregressive transformer architecture seemed the irreplaceable foundation of LLMs in recent years, new research has proven otherwise: **the iterative denoising paradigm can achieve comparable results with significantly better performance**. This opens vast opportunities for innovation. In conclusion, we note several forecasts and observations:

* **A paradigm shift is possible but will take time.** Just as diffusion models revolutionized image generation, we can expect them to become the dominant text generation method, especially when the issue is **scaling**. Users and developers always seek faster model responses — this is a strong driver for technological change. However, full replacement of AR models won’t happen immediately: in the near term, coexistence and specialization are likely (AR — for tasks requiring step-by-step prediction or maximum accuracy on small models; Diffusion — for long texts, interactive scenarios, fast response prototyping).

* **Scalability of diffusion models looks promising.** Already at 8B parameters, LLaDA showed trends analogous to AR: more data and parameters — better results. There are no theoretical barriers to training a diffusion model with 70B+ parameters, except for higher computational costs. Google is likely already working on this (Gemini family). If Gemini Diffusion scales to GPT-4 levels (hundreds of billions of parameters) and retains its speed advantage, it could cement the paradigm. In academia, DLM research is accelerating: variants with multimodality (e.g., **LLaDA-V** for simultaneous image and text processing) and improved training algorithms are emerging.

* **Practical applications will start in narrow domains.** Mercury chose the code generation niche — and not by accident: programmers highly value fast autocompletion. We can expect diffusion models to first establish themselves in IDEs and code assistants. Other likely areas are **real-time systems** where latency is critical: speech translation, AR assistance, customer support chatbots. In enterprise solutions where control and customization matter, DLMs may also attract interest due to their ability to precisely edit generated responses (e.g., correcting a generated report without rebuilding it entirely).

* **Promising architectures:** among the approaches discussed, each has strong points. **Gemini Diffusion**, as a Google development, likely gains an advantage from access to scale of data and computation — this architecture could become the **locomotive of the paradigm** if integrated into products. **Mercury** demonstrated the value of *new optimization targets* (score entropy) — its methods may form the foundation of many future models, not just commercial ones. **LLaDA** is important as proof that the **open-source community** can also implement diffusion LLMs; its code and model are already available, enabling many researchers to experiment with diffusion. **Eso-LM** points toward hybrid solutions — likely, major framework producers (Hugging Face, PyTorch team) will implement such “universal transformers” with flexible attention, since this delivers gains. The most promising from a practical standpoint is the **combination of speed and quality**: models that can, when needed, work as AR for fine-tuning text, yet instantly deliver a rough draft. Eso-LM (B) has already demonstrated this approach with KV caching and acceleration, and this may become the basis of future industrial DLMs.

In summary, diffusion language models today stand on the brink of transforming from a scientific experiment into a **new industry standard**. If remaining limitations — improving “first-token latency,” simplifying training, and matching the quality of top GPT systems — can be overcome, AR models will have no competitive advantage left. As noted in the community, Google, with its resources, may truly **replace the LLM standard from autoregression to diffusion**. Even if this does not happen instantly, the mere existence of this alternative stimulates progress: competition between the two approaches leads to better solutions. Already, we see ideas from diffusion (masking, iterative refinement) beginning to permeate “classical” models, and AR models adopting some adaptive tricks. In the coming years, we will witness an exciting evolution in this field, and possibly the emergence of entirely **new hybrid paradigms** that unify the strengths of all approaches to achieve even smarter and faster artificial intelligences.