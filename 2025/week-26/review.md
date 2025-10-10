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

**Пример реализации из кода:**
```python
# algo.py, строки 146-147
do_sequential = self.config.algo.alpha_0 != 1
do_diffusion = self.config.algo.alpha_0 != 0

# Двухфазная генерация в EsoLM._loss():
if do_sequential:
    # AR-фаза: последовательное заполнение масок
    alpha_start = self.config.algo.alpha_0
    z0 = self.q_xt(x0_reconstruction, alpha_start)
    reconstruction_loss, sort_idx = self._reconstruction_loss(x0_reconstruction, z0)
    
if do_diffusion:
    # Диффузионная фаза: параллельная генерация
    diffusion_loss, sort_idx = self.nll(x0_diffusion, None, ...)
```

Формально, если $z_0$ – частично замаскированная последовательность после первого этапа, то на втором этапе генерируется финальный вывод $x$ как $x = \text{AR}(z_0)$, где AR-модель видит уже раскрытые «чистые» токены и дополняет недостающее. Важно, что при таком подходе *часть токенов генерируется параллельно, а часть – последовательно*. Параметром разбиения служит доля $\alpha_0$: если $\alpha_0 = 1$, то весь текст генерирует только диффузия (чисто параллельный режим), если $\alpha_0 = 0$ – весь текст порождается AR (классический режим). Обычно выбирают промежуточное значение, например, $\alpha_0 = 0.5$ – половина токенов сразу ставится диффузией, половина дозаполняется AR.

**Конкретные конфигурации из проекта:**
```yaml
# configs/algo/esolm.yaml - базовая конфигурация
alpha_0: 0.0  # По умолчанию чисто AR режим

# Эксперименты с разными значениями α₀:
# scripts/esolm/train_owt_esolmb_alpha0_0d125.sh -> α₀ = 0.125
# scripts/esolm/train_owt_esolmb_alpha0_0d25.sh  -> α₀ = 0.25
# scripts/esolm/train_owt_esolmb_alpha0_0d5.sh   -> α₀ = 0.5
# scripts/esolm/train_owt_esolmb_alpha0_1.sh     -> α₀ = 1.0 (чисто диффузия)
```

В `trainer_base.py` определен noise scheduler:
```python
class LogLinear:
    def __init__(self, alpha_0=1):
        self.alpha_0 = alpha_0
    
    def alpha_t(self, t):
        alpha_t = self.alpha_0 * (1 - t)  # Линейное расписание шума
```

Такой гибрид позволяет **интерполировать между AR и MDM** по качеству и скорости, а также вносит дополнительную гибкость: к примеру, можно первые и последние слова предложения определить параллельно (учитывая глобальный контекст), а детали середины уточнить последовательно. Eso-LM явно сочетает сильные стороны: высокое качество моделирования (AR-часть гарантирует гладкость переходов, особенно для сложных фрагментов) и высокую скорость на большую часть последовательности (параллельная MDM-часть экономит время).

**Единая архитектура и внимание:** ключевая сложность – обучить **один трансформер**, способный работать и в режиме диффузии, и в режиме авторегрессии. В обычном случае требования конфликтуют: для AR нужно causal-маскирование (токен видит только предшествующие), для MDM – полное внимание по всем токенам (маски можно раскрывать в любом порядке). Авторы Eso-LM решили эту проблему с помощью **кастомного механизма внимания** с маской $A$.

**Реализация кастомных масок внимания:**
```python
# models/dit.py - реализация различных типов масок
@lru_cache
def _causal_mask(b, h, q_idx, kv_idx):
    """Causal маска для AR режима"""
    causal = q_idx >= kv_idx
    return causal

@lru_cache  
def _bidirectional_mask(b, h, q_idx, kv_idx):
    """Полное внимание для MDM режима"""
    bidirectional = q_idx == q_idx  # Всегда True
    return bidirectional

@lru_cache
def _mixed_mask(b, h, q_idx, kv_idx, cutoffs):
    """Смешанная маска для EsoLM"""
    causal = q_idx >= kv_idx
    block_identity = q_idx >= cutoffs[b]
    return causal | block_identity

# Использование в EsoLMDiT:
def _get_attention_mask(self, seq_len, attn_mode=None, cutoffs=None):
    if attn_mode == 'causal':
        return _get_causal_mask(seq_len)
    elif attn_mode == 'bidirectional':
        return _get_bidirectional_mask(seq_len)
    elif attn_mode == 'mixed':
        return _get_mixed_mask(seq_len, cutoffs)
```

В трансформер вводится матрица смещений внимания $A_{i,j}$, где $A_{i,j} = 0$ разрешает внимание от позиции $i$ к $j$, а $-\infty$ запрещает. Настраивая эту матрицу, можно эмулировать любой шаблон внимания. Например, для AR-части $A$ будет задавать треугольную каузальную маску, а для MDM-части – полное внимание между «чистыми» токенами и ограниченное для масок. Конкретно, **Eso-LM (A)** убирает двунаправленное внимание *между масками* в диффузионной фазе. Это означает, что маски при денойзинге не «видят» друг друга – тем самым исключается избыточная зависимость, и трансформер может работать быстрее. Эту экономию авторы усиливают с помощью *разреженного внимания*: на каждом шаге диффузии обрабатываются не все позиции, а только те маски, которые выбраны для раскрытия на данном шаге, плюс все уже раскрытые токены.

**Реализация сортировки токенов для двухфазной генерации:**
```python
# algo.py - EsoLM._sort_indices()
def _sort_indices(self, indices, shuffle, keep_masks_unshuffled=False):
    """Сортировка для определения порядка генерации токенов"""
    masked = (indices == self.mask_index)
    
    if shuffle:
        # Случайные смещения для диффузионной части
        offsets = torch.rand(indices.shape).to(indices.device) * 0.9
        
        if keep_masks_unshuffled:
            # Для AR части: строгий left-to-right порядок масок
            offsets[masked] = torch.linspace(
                0, 1, torch.sum(masked)).to(indices.device)
    else:
        # Фиксированный порядок
        offsets = torch.linspace(0, 0.9, indices.shape[1]).to(indices.device)
    
    # Сортировка: маски идут первыми + смещения
    sort_idx = (masked + offsets).argsort(descending=False)
    return sort_idx
```

Такой подход существенно снижает затраты при длинных последовательностях, потому что вместо полного прохода по 10k токенам, например, можно обновить только 1k масок, оставив 9k постоянных. **Eso-LM (B)** идёт ещё дальше: он вводит каузальное ограничение даже для «чистых» токенов во время диффузии, позволяя **кешировать KV**-активации для них. Проще говоря, вариант B жертвует частью двунаправленного контекста (чистые токены видят только предшествующие чистые), но взамен может сохранять их представления и не пересчитывать на каждом шаге. Это даёт дополнительное ускорение – по оценкам, поддержка KV-кеша в диффузионной фазе увеличивает скорость до **65%** по сравнению с базовым MDM без кеширования.

**Реализация KV кэширования:**
```python
# models/dit.py - DDiTBlock с KV кэшированием
class DDiTBlock(nn.Module):
    def reset_kv_cache(self):
        """Сброс KV кэша"""
        self.k_cache = None
        self.v_cache = None
    
    def _process_and_update_kv(self, k, v, num_clean):
        """Обновление KV кэша только для чистых токенов"""
        if self.k_cache is None:
            self.k_cache = k[:, :num_clean]
            self.v_cache = v[:, :num_clean]
        else:
            # Конкатенация с предыдущим кэшем
            self.k_cache = torch.cat([self.k_cache, k[:, :num_clean]], dim=1)
            self.v_cache = torch.cat([self.v_cache, v[:, :num_clean]], dim=1)
    
    @torch.no_grad()
    def _attention_with_kv_cache(self, qkv, rotary_cos_sin, num_clean, num_clean_and_mask):
        """Attention с использованием KV кэша"""
        # num_clean: количество чистых токенов
        # num_clean_and_mask: чистые + маски для генерации
        
        # Применение rotary embeddings и разделение на q, k, v
        qkv = split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Обновление кэша
        self._process_and_update_kv(k, v, num_clean)
        
        # Использование кэшированных значений для attention
        attention_output = fused_flex_attention(
            q[:, :num_clean_and_mask], 
            self.k_cache, 
            self.v_cache,
            mask=None
        )
        return attention_output
```


**Единая архитектура и внимание:** ключевая сложность – обучить **один трансформер**, способный работать и в режиме диффузии, и в режиме авторегрессии. В обычном случае требования конфликтуют: для AR нужно causal-маскирование (токен видит только предшествующие), для MDM – полное внимание по всем токенам (маски можно раскрывать в любом порядке). Авторы Eso-LM решили эту проблему с помощью **кастомного механизма внимания** с маской $A$. В трансформер вводится матрица смещений внимания $A_{i,j}$, где $A_{i,j} = 0$ разрешает внимание от позиции $i$ к $j$, а $-\infty$ запрещает. Настраивая эту матрицу, можно эмулировать любой шаблон внимания. Например, для AR-части $A$ будет задавать треугольную каузальную маску, а для MDM-части – полное внимание между «чистыми» токенами и ограниченное для масок. Конкретно, **Eso-LM (A)** убирает двунаправленное внимание *между масками* в диффузионной фазе. Это означает, что маски при денойзинге не «видят» друг друга – тем самым исключается избыточная зависимость, и трансформер может работать быстрее. Эту экономию авторы усиливают с помощью *разреженного внимания*: на каждом шаге диффузии обрабатываются не все позиции, а только те маски, которые выбраны для раскрытия на данном шаге, плюс все уже раскрытые токены. Такой подход существенно снижает затраты при длинных последовательностях, потому что вместо полного прохода по 10k токенам, например, можно обновить только 1k масок, оставив 9k постоянных. **Eso-LM (B)** идёт ещё дальше: он вводит каузальное ограничение даже для «чистых» токенов во время диффузии, позволяя **кешировать KV**-активации для них. Проще говоря, вариант B жертвует частью двунаправленного контекста (чистые токены видят только предшествующие чистые), но взамен может сохранять их представления и не пересчитывать на каждом шаге. Это даёт дополнительное ускорение – по оценкам, поддержка KV-кеша в диффузионной фазе увеличивает скорость до **65%** по сравнению с базовым MDM без кеширования. При этом небольшое снижение качества (перплексии) приемлемо: вариант B показывает чуть хуже PPL, чем A, но всё ещё лучше, чем чисто диффузионные модели, и **единственный** среди известных, кто может кешировать в параллельной генерации.

**Итоги архитектуры:** Eso-LM продемонстрировал, что **симбиотическое объединение AR и диффузии возможно** в рамках одного трансформера. На практике оба варианта модели достигли **лучшей на сегодня перплексии среди диффузионных моделей** (на датасетах LM1B и OpenWebText) и предоставили непрерывный спектр качество/скорость между AR и MDM. Особо отмечается, что Eso-LM (A) может приблизиться по PPL к чисто AR-модели, но генерирует существенно быстрее, а Eso-LM (B) слегка уступает по PPL, зато **работает быстрее всех** (благодаря кешированию). В тестах на скорость генерации Eso-LM превосходил предыдущие диффузионные модели, попадая на новую **Pareto-границу качества и быстродействия** (т. е. ни одна прежняя модель не давала одновременно такой же перплексии при такой же скорости). При малом числе шагов генерации гибрид не страдает коллапсом (в отличие от некоторых упрощённых интерполирующих схем), а при большом числе шагов выдает образцы лучшего качества, чем все прошлые диффузионные LM. Эти результаты делают Eso-LM важным ориентиром в дизайне будущих архитектур: возможно, именно комбинация параллельной черновой генерации с последовательной доводкой позволит достичь наилучшего баланса.

## Математическая и вероятностная формализация

Все рассмотренные диффузионные языковые модели стремятся моделировать *распределение вероятностей над текстом*, аналогично традиционным LLM, но не через авторегрессионное разложение, а посредством **марковского диффузионного процесса**.

Формально цель — максимизировать вероятность $P_\theta(X)$ для последовательности $X = (x_1, \dots, x_L)$ из обучающего корпуса. В авторегрессионных моделях применяется факторизация по токенам:  
$$
P(X) = \prod_{i=1}^L P(x_i \mid x_{<i})
$$  
и обучение сводится к минимизации кросс-энтропии предсказания следующего токена.

<details> 
    <summary><em><strong>Кросс-энтропия (Cross-Entropy)</strong></em></summary>

## Кросс-энтропия (Cross-Entropy Loss)

Функция кросс-энтропии (Cross-Entropy Loss) — это центральная функция потерь в задачах **классификации**, особенно бинарной и многоклассовой. Она тесно связана с **максимизацией логарифма правдоподобия**, а также с фундаментальными концепциями информационной теории.

### 1. Постановка задачи классификации

Пусть дана обучающая выборка:

$$
D = \{(x_i, y_i)\}_{i=1}^n,\quad x_i \in \mathcal{X} \subseteq \mathbb{R}^d,\ y_i \in \{1, 2, \dots, K\}
$$

Цель: найти параметризованную функцию $f_\theta(x)$, которая аппроксимирует распределение вероятностей классов:

$$
f_\theta(x) = \hat{\mathbf{p}}(x) = (\hat{p}_1(x), \hat{p}_2(x), \dots, \hat{p}_K(x)), \quad \sum_{k=1}^K \hat{p}_k(x) = 1,\ \hat{p}_k(x) \ge 0
$$

(например, выход softmax'а).

Пусть $y_i$ — истинный класс, тогда целевая one-hot вектор-метка:

$$
\mathbf{y}_i = (0,\dots, 1, \dots, 0), \text{ где 1 стоит на } y_i\text{-й позиции}
$$

### 2. Определение кросс-энтропии

Кросс-энтропия между истинным распределением $P$ и предсказанным $Q$:

$$
\boxed{
H(P, Q) = -\sum_{k=1}^K P(k) \log Q(k)
}
$$

В контексте supervised learning:

* $P(k) = \mathbb{I}[y_i = k]$ — one-hot распределение;
* $Q(k) = \hat{p}_k(x_i)$ — предсказанная вероятность;

- Если говорить интуитивно, кросс-энтропия помогает нам понять, насколько наши предсказания отличаются от истинных значений;
- Логарифм используется для того, чтобы преобразовать произведение вероятностей в сумму, что упрощает расчёты и делает их более стабильными;
- Знак минус используется, потому что кросс-энтропия учитывает логарифмы вероятностей, которые всегда меньше или равны нуля. Когда мы суммируем эти логарифмы, результат получается отрицательным. Чтобы сделать функцию потерь положительной и минимизировать её, мы добавляем этот минус.

Тогда:

$$
\text{Loss}(x_i, y_i) = - \log \hat{p}_{y_i}(x_i)
$$

А по всей выборке:

$$
\boxed{
\mathcal{L}_{CE}(\theta) = -\frac{1}{n} \sum_{i=1}^n \log \hat{p}_{y_i}(x_i)
}
$$

### 3. Интерпретации

#### (а) Информационная теория

Кросс-энтропия измеряет **среднее количество бит**, необходимое для кодирования истинных меток $P$, если используется кодировка, основанная на распределении $Q$:

* Если $Q \approx P$, то $H(P,Q) \approx H(P)$ — энтропия.
* Если $Q$ сильно отличается от $P$, то $H(P,Q)$ увеличивается.

> ⇒ **Минимизация кросс-энтропии ⇔ максимизация точности предсказания вероятностей.**

#### (б) Вероятностная интерпретация

Предположим, что модель предсказывает вероятности $Q = f_\theta(x)$, и данные метки $y_i$ независимы, тогда:

$$
\log L(\theta) = \sum_{i=1}^n \log P(y_i | x_i, \theta) = \sum_{i=1}^n \log \hat{p}_{y_i}(x_i)
$$

Тогда:

$$
\boxed{
\mathcal{L}_{CE} = - \log L(\theta)
}
$$

То есть, **кросс-энтропия — это отрицательный логарифм правдоподобия**. Поэтому она возникает естественно при выводе из принципа максимального правдоподобия (MLE).

#### (в) Связь с KL-дивергенцией

Напомним, KL-дивергенция:

$$
D_{KL}(P \| Q) = \sum_{k=1}^K P(k) \log \frac{P(k)}{Q(k)} = H(P, Q) - H(P)
$$

При one-hot разметке $P(k) = \delta_{ky}$ ⇒ $H(P)$ = 0 ⇒

$$
\boxed{
D_{KL}(P \| Q) = H(P, Q)
}
$$

<details> 
    <summary><em><strong>Кросс-энтропия vs KL-дивергенция</strong></em></summary>

## 🔍 **Оптимизация кросс-энтропии vs KL-дивергенции: от простого к сложному**

Когда мы работаем с задачами машинного обучения, особенно в классификации, нам часто нужно измерять, насколько предсказания модели отличаются от истинных значений. Два популярных способа сделать это — **кросс-энтропия (Cross-Entropy, CE)** и **KL-дивергенция (Kullback-Leibler Divergence, KLD)**.

На первый взгляд, они кажутся очень похожими, но есть важные различия. Давайте разберёмся по шагам!

### **1️⃣ Что такое энтропия $H(P)$?**

Энтропия распределения $ P $ — это мера его **неопределённости**. Формула:

$$
H(P) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

где:
- $ \mathcal{X} $ — все возможные токены (слова/символы),
- $ P(x) $ — вероятность токена $ x $ в истинном распределении.

Чем выше $ H(P) $, тем более "размазано" распределение (больше неопределённости).

### **2️⃣ Пример: Next-Token Prediction**

Допустим, у нас есть:
- **Контекст:** `"Кошка лежит на ___"`
- **Возможные следующие токены:** `"ковре"` (0.7), `"полу"` (0.2), `"диване"` (0.1)

Тогда **истинное распределение $ P $** может быть:

#### **🔹 Случай 1: One-hot (детерминированное)**
Если правильный токен только `"ковре"`, то:

$$
P = [1, 0, 0]
$$

Энтропия:

$$
H(P) = - \left( 1 \cdot \log 1 + 0 \cdot \log 0 + 0 \cdot \log 0 \right) = 0
$$

(поскольку $ \lim_{p \to 0} p \log p = 0 $)

**Вывод:**

- $ H(P) = 0 $ → нет неопределённости.
- В этом случае **KLD и CE совпадают**:

$$
D_{KL}(P \| Q) = H(P, Q) - H(P) = H(P, Q)
$$

#### **🔹 Случай 2: Вероятностное (soft)**
Пусть правильные токены имеют вероятности:

$$
P = [0.7, 0.2, 0.1]
$$

Тогда энтропия:

$$
H(P) = - (0.7 \log 0.7 + 0.2 \log 0.2 + 0.1 \log 0.1)
$$

Допустим, логарифм по основанию 2 (биты):

$$
H(P) \approx - (0.7 \cdot (-0.514) + 0.2 \cdot (-2.321) + 0.1 \cdot (-3.321)) \approx 1.157 \text{ бит}
$$

**Что это значит?**
- Энтропия **не нулевая**, значит, есть неопределённость в правильном ответе.
- Если модель предсказывает $ Q = [0.6, 0.3, 0.1] $, то:

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

Здесь $ H(P, Q) $ — кросс-энтропия, а $ H(P) $ — "базовая" неопределённость данных.

### **3️⃣ Кросс-энтропия (Cross-Entropy, CE)**

#### **🔹 Что это?**

Кросс-энтропия измеряет, насколько "удивительны" предсказания модели относительно истинного распределения. Чем меньше CE, тем лучше модель предсказывает.

#### **🔹 Формула**

Для дискретного случая (например, классификация):

$$
H(P, Q) = -\sum_{i} P(i) \log Q(i)
$$

где:
- $ P $ — истинное распределение (обычно one-hot encoded, например, $[0, 1, 0]$ для класса 2).
- $ Q $ — предсказанное распределение (например, $[0.1, 0.8, 0.1]$).

#### **🔹 Особенности**
✅ **Простота**: в ML чаще используют CE, потому что если $ P $ — one-hot, то формула упрощается до $ -\log Q(\text{true class}) $.  
✅ **Эффективность**: градиенты легко вычисляются, что ускоряет обучение.

## **4️⃣ KL-дивергенция (Kullback-Leibler Divergence, KLD)**

#### **🔹 Что это?**

KLD измеряет, насколько одно распределение $ Q $ отличается от другого $ P $. Это **не метрика расстояния** (не симметрична: $ D_{KL}(P \| Q) \neq D_{KL}(Q \| P) $).

#### **🔹 Формула**

$$
D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} = H(P, Q) - H(P)
$$

где:
- $ H(P, Q) $ — кросс-энтропия между $ P $ и $ Q $,
- $ H(P) $ — энтропия $ P $ (мера неопределённости).

#### **🔹 Особенности**
✅ **Информационная разница**: KLD показывает, сколько дополнительных бит информации нужно для кодирования $ P $, если использовать $ Q $.  
❌ **Зависит от $ H(P) $**: если $ P $ фиксировано (например, one-hot), то $ H(P) = 0 $, и KLD становится равным CE!

### **5️⃣ Связь между CE и KLD**

Из формулы KLD видно:
$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

#### **🔹 Если $ P $ — one-hot (как в классификации):**
- $ H(P) = 0 $ (энтропия детерминированного распределения нулевая),
- Тогда **KLD = CE**!

#### **🔹 Если $ P $ не one-hot (например, сглаженные метки):**
- $ H(P) > 0 $, значит, KLD и CE различаются.
- Оптимизация KLD учитывает ещё и энтропию $ P $, а CE — нет.

### **6️⃣ Когда что использовать?**

| **Критерий**       | **Кросс-энтропия (CE)** | **KL-дивергенция (KLD)** |
|--------------------|------------------------|--------------------------|
| **One-hot метки**  | ✅ Лучше (проще и быстрее) | ⚠️ То же самое (KLD = CE) |
| **Сглаженные метки** | ❌ Не учитывает $ H(P) $ | ✅ Учитывает разницу распределений |
| **Интерпретация**  | "Удивление" модели | "Информационная стоимость" ошибки |

#### **🔹 Практический вывод:**
- **В большинстве задач классификации CE и KLD эквивалентны** (так как метки one-hot).
- **Если метки вероятностные (например, soft targets в Distillation) — KLD лучше**, так как учитывает энтропию истинного распределения.

### **🎯 Итог**

- **Кросс-энтропия** — это "удивление" модели относительно истинных меток.
- **KL-дивергенция** — это "стоимость" использования $ Q $ вместо $ P $.
- **Если $ P $ детерминировано (one-hot) → CE = KLD.**
- **Если $ P $ вероятностно → KLD учитывает его энтропию.**

Теперь вы знаете разницу и можете осознанно выбирать функцию потерь! 🚀

</details>

### 4. Частные случаи

#### 🔹 Бинарная кросс-энтропия (логистическая регрессия)

Если $y_i \in {0, 1}$ и $f(x_i) = \hat{p}_i \in (0, 1)$, тогда:

$$
\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]
$$

### 5. Свойства кросс-энтропии

| Свойство              | Описание                                                                        |
| --------------------- | ------------------------------------------------------------------------------- |
| 📈 Выпуклость         | Если $\hat{p}*k$ — аффинная функция параметров, $\mathcal{L}*{CE}$ выпукла. |
| ⚙️ Гладкость          | Дифференцируема по $\hat{p}_k$, подходит для градиентного спуска.            |
| 🎯 Интерпретируемость | Потери равны $-log$ предсказанной вероятности правильного класса.            |
| ⚠️ Чувствительность   | Наказывает сильнее за большую уверенность в неправильных классах.               |

> **Пример:** если правильный класс предсказан с вероятностью 0.9: $-log(0.9) ≈ 0.105$, а если 0.01 — $-\log(0.01) ≈ 4.6$.

### 6. Градиент кросс-энтропии

Рассмотрим softmax выход модели:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}},\quad z_k = \text{логиты}
$$

И функцию потерь:

$$
\mathcal{L}_{CE}(z, y) = -\log \hat{p}_y
$$

Градиент по $z_j$:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = \hat{p}_j - \mathbb{I}[j = y]
$$

То есть:

$$
\nabla_z \mathcal{L} = \hat{\mathbf{p}} - \mathbf{y}
$$

> 💡 Это очень удобно: градиент — просто разность между предсказанным распределением и истинным one-hot вектором.

### 7. Практические аспекты

**Когда применять:**

* Классификация (бинарная / многоклассовая / multi-label).
* Когда важно корректно оценивать вероятности.
* При выводе через MLE.

**Когда избегать:**

* Задачи с шумной разметкой: кросс-энтропия чувствительна к label noise.
* Имбаланс классов без корректировки (возможна переоценка частых классов).

### 8. Заключение

Кросс-энтропия — фундаментальная функция потерь в классификации, строго выведенная из теории вероятностей и информационной теории. Её выпуклость, простота градиента и интерпретируемость делают её предпочтительным выбором в большинстве задач обучения с учителем. Однако чувствительность к переуверенным ошибкам требует внимательного контроля и настройки моделей.

> 🧪 В реальных задачах часто применяют её модификации: **focal loss** (для борьбы с дисбалансом), **label smoothing**, **soft targets**, **weighted CE** и др.

</details>

---

В диффузионных же моделях вводится *скрытая переменная* — **индекс шага диффузии $t$**, и определяются два процесса:
- **прямой (forward)** $q$ — постепенное добавление шума к данным,
- **обратный (reverse)** $p_\theta$ — обучаемый процесс удаления шума.

Forward-процесс строится так, что при $t = T$ данные полностью разрушены (например, все токены замаскированы или заменены на равномерный шум), а при $t = 0$ — данные чистые (оригинальный текст). Конкретно, на каждом шаге $t \to t-1$ мы можем определять распределение $q(x_{t-1} \mid x_t, X)$, где $X$ — исходные данные.

В дискретных диффузионных моделях обычно используется **маскирование** как вид шума: с некоторой возрастающей вероятностью $\beta_t$ каждый токен заменяется на `[MASK]` независимо. В пределе $t = T$ получаем полностью замаскированную последовательность (полный шум). Этот forward-процесс $q$ известен точно (задано распределение масок). Согласно теории диффузии, *апостериорное* распределение $q(x_{t-1} \mid x_t, X)$ тоже может быть получено в аналитическом виде (для маскирования оно пропорционально либо дельта-функции на истинном токене, либо на маске).

Обратный процесс $p_\theta$ параметризуется нейросетью. Он должен аппроксимировать распределение восстановления: $p_\theta(x_{t-1} \mid x_t)$ — вероятность получить менее зашумленное состояние на шаге $t-1$, исходя из состояния на шаге $t$. Чтобы сделать задачу tractable, обычно предполагается *полная факторизация* этого распределения по позициям, содержащим шум на шаге $t$. Например, если на шаге $t$ замаскированы индексы $\mathcal{M}_t$, то  
$$
p_\theta(x_{t-1} \mid x_t) = \prod_{i\in \mathcal{M}_t} p_\theta(x^{(i)}_{t-1} \mid x_t)
$$  
— модель предсказывает каждый замаскированный токен независимо (условно на текущее состояние всей последовательности $x_t$). Это в точности эквивалентно задаче **маскированного языкового моделирования** на текущем контексте $x_t$.

<details> 
    <summary><em><strong>Суть диффузионной модели для текста</strong></em></summary>

## **Суть диффузионной модели для текста:**

Представьте, что у вас есть осмысленный текст. Процесс **"диффузии"** постепенно **портит** этот текст, например, заменяя случайные слова (токены) на "маски" (`[MASK]`) или просто на случайный шум. Это похоже на то, как будто кто-то стирает слова на странице. Конечная цель модели — научиться выполнять **обратный процесс**: взять этот зашумленный, "испорченный" текст и **восстановить** изначальный осмысленный вариант.

**Ключевой компонент: обратный процесс ($p_\theta$)**

*   **Что это?** это "мозг" модели восстановления. Это нейронная сеть (параметризованная весами $\theta$), которая **учится предсказывать, как выглядел текст на предыдущем, *менее* зашумленном шаге ($x_{t-1}$), исходя из текущего, *более* зашумленного состояния ($x_t$)**.
*   **Формально:** $p_\theta(x_{t-1} \mid x_t)$ — это *условное распределение вероятностей*. Оно говорит: "Если текущий текст выглядит как $x_t$ (содержит какие-то шумы/маски), то какова вероятность того, что на предыдущем шаге он выглядел как $x_{t-1}$?".
*   **Цель обучения:** настроить параметры $\theta$ нейронной сети так, чтобы это распределение $p_\theta$ было как можно ближе к *истинному* (но неизвестному нам) распределению восстановления $q(x_{t-1} \mid x_t)$.

**Проблема: слишком сложно!**

*   Предсказывать *весь* вектор $x_{t-1}$ (который представляет всю последовательность слов/токенов) сразу, на основе $x_t$ — это *чрезвычайно* сложная задача. Это потребовало бы от модели учитывать все возможные комбинации слов во всей последовательности одновременно. Вычислительно это неподъемно.

**Решение: полная факторизация (условная независимость)**

*   Чтобы сделать задачу **решаемой (tractable)**, вводится ключевое **допущение**:
    > **Предположим, что предсказание каждого *отдельного* зашумленного токена на позиции $i$ для шага $t-1$ ($x^{(i)}_{t-1}$) зависит *только* от *текущего* зашумленного состояния *всей* последовательности ($x_t$), но *НЕ* зависит от того, что мы предскажем для *других* зашумленных позиций ($j \neq i$) на этом *же* шаге $t-1$.**

*   **Проще говоря:** модель смотрит на *весь* текущий зашумленный текст $x_t$ (включая известные *незашумленные* слова и текущие маски/шумы). На основе этого **полного контекста** $x_t$ она **независимо** предсказывает, что должно стоять на месте *каждой отдельной* маски/шума, чтобы получить $x_{t-1}$.
*   **Аналогия 1 (Кроссворд):** представьте, что $x_t$ — это кроссворд, где некоторые клетки заполнены буквами (незашумленные токены), а некоторые пусты (маски `[MASK]`). Модель смотрит на *весь* кроссворд — и на пересечения слов — и предсказывает, какая буква должна быть в *каждой* пустой клетке ($x^{(i)}_{t-1}$). Предположение факторизации говорит: предсказание буквы для клетки (1,1) зависит от *всех* уже заполненных клеток вокруг нее, но *не* зависит напрямую от того, что вы предскажете *одновременно* для клетки (5,5). Вы предсказываете их независимо, но каждое предсказание опирается на *один и тот же* общий контекст (весь текущий кроссворд $x_t$).
*   **Аналогия 2 (Шахматы):** представьте шахматную доску $x_t$, где некоторые фигуры стоят на своих начальных позициях (незашумленные токены), а некоторые клетки пусты (маски). Модель должна восстановить предыдущее состояние доски $x_{t-1}$, где на пустых клетках могли стоять фигуры. Она смотрит на *всю* текущую доску (расположение оставшихся фигур) и **независимо** решает, какая фигура *наиболее вероятно* стояла на *каждой* пустой клетке перед тем, как ее убрали. Решение для одной клетки зависит от общего положения, но не зависит напрямую от решения для другой клетки *в этот самый момент*.

**Математическое выражение:**

*   Обозначим $\mathcal{M}_t$ как **множество индексов позиций** в последовательности, которые **зашумлены** (замаскированы или искажены) на текущем шаге $t$.
*   Допущение полной факторизации позволяет записать сложное совместное распределение восстановления как **произведение** независимых распределений для **каждой зашумленной позиции**:
    $$
    p_\theta(x_{t-1} \mid x_t) = \prod_{i \in \mathcal{M}_t} p_\theta(x^{(i)}_{t-1} \mid x_t)
    $$
*   **Что это значит:**
    *   $\prod_{i \in \mathcal{M}_t}$: перемножаем вероятности по всем позициям $i$, которые зашумлены на шаге $t$.
    *   $p_\theta(x^{(i)}_{t-1} \mid x_t)$: вероятность того, что на позиции $i$ в *менее* зашумленном состоянии $x_{t-1}$ находится *конкретный* токен (слово, буква) $x^{(i)}_{t-1}$, **при условии, что мы видим *весь* текущий зашумленный текст $x_t$**.
    *   **Ключевой момент:** распределение для позиции $i$ ($p_\theta(x^{(i)}_{t-1} \mid x_t)$) зависит **ТОЛЬКО** от $x_t$ (от всего текущего контекста), но **НЕ** зависит от того, что модель предскажет для $x^{(j)}_{t-1}$ (для другой зашумленной позиции $j$) при расчете *этого же* распределения $p_\theta(x_{t-1} \mid x_t)$. Они считаются **условно независимыми** при фиксированном $x_t$.

**Связь с Masked Language Modeling (MLM) a la BERT:**

*   **Это самая важная аналогия!** посмотрите внимательно на $p_\theta(x^{(i)}_{t-1} \mid x_t)$.
*   **Что это?** это задача предсказания **одного** токена (того, что был на позиции $i$ в $x_{t-1}$) на основе **всего** текущего контекста $x_t$.
*   **Как это выглядит на практике?** на шаге $t$ у нас есть последовательность $x_t$, в которой на позициях $\mathcal{M}_t$ стоят маски `[MASK]` (или другие символы шума). Нейронная сеть ($p_\theta$) принимает на вход $x_t$ и для **каждой** позиции $i$ из $\mathcal{M}_t$ выдает распределение вероятностей ($p_\theta(x^{(i)}_{t-1} \mid x_t)$) по *всему* словарю — какое слово/токен с какой вероятностью должно стоять *вместо этой конкретной маски*, чтобы получить состояние $x_{t-1}$.
*   **Это в точности задача Masked Language Modeling (MLM)!** та самая, на которой обучаются такие модели, как BERT. BERT получает на вход текст с масками и учится предсказывать оригинальные слова под масками, используя контекст *всего* предложения (и $x_t$ в диффузии — это и есть такой контекст с масками).

**Оптимизация (Обучение) с помощью Cross-Entropy:**

*   Как мы обучаем модели типа BERT решать задачу MLM? Мы используем **кросс-энтропийную потерю (cross-entropy loss)**.
*   **Как это работает в диффузии:**
    1.  Во время обучения, для **каждого** шага диффузии $t$ и для **каждой** зашумленной позиции $i \in \mathcal{M}_t$ на этом шаге, мы знаем **оригинальный, правильный токен**, который был на этой позиции *до* наложения шума (это и есть $x^{(i)}_{t-1}$).
    2.  Нейронная сеть ($p_\theta$) для позиции $i$ выдает **предсказанное распределение вероятностей** $p_\theta(x^{(i)}_{t-1} \mid x_t)$ по всем возможным токенам.
    3.  Мы вычисляем **кросс-энтропию** между:
        *   **Идеальным распределением:** вероятность 1.0 для *правильного* токена и 0.0 для всех остальных.
        *   **Предсказанным распределением:** $p_\theta(x^{(i)}_{t-1} \mid x_t)$ от модели.
    4.  Эта кросс-энтропия измеряет, насколько хорошо модель предсказала правильный токен *для этой конкретной позиции $i$*.
    5.  Потери (loss) со *всех* зашумленных позиций $i \in \mathcal{M}_t$ на шаге $t$ **суммируются** (или усредняются). Это и есть общая потеря для шага $t$.
    6.  Градиенты этого общего лосса распространяются назад по нейронной сети, обновляя ее веса $\theta$, чтобы улучшить предсказания.
*   **Итог:** обучение обратного процесса диффузии на *каждом* шаге $t$ **сводится к решению множества независимых задач Masked LM на контексте $x_t$**, и оптимизируется это с помощью привычной **кросс-энтропии** для каждой маски.

**Ключевые выводы:**

1.  **Обратный процесс ($p_\theta$)** — это нейросеть, которая учится "чинить" текст шаг за шагом.
2.  **Факторизация ($p_\theta = \prod p_\theta(...)$)** — это *необходимое упрощение*, делающее обучение возможным. Оно означает: "Предсказывай каждую маску независимо, но используй для каждой предсказания *весь* текущий текст".
3.  **$p_\theta(x^{(i)}_{t-1} \mid x_t)$** — это **ядро процесса**. Это ровно та задача, которую решает BERT (Masked LM): "Что скрывается под этой конкретной маской `[MASK]` в данном контексте $x_t$?".
4.  **Cross-Entropy** — это стандартный и эффективный способ *обучить* нейросеть решать множество таких задач MLM *параллельно* на одном шаге диффузии $t$.

</details>

---

Таким образом, *каждый шаг диффузии оптимизируется с помощью привычной cross-entropy* на правильный токен вместо маски. Однако в отличие от BERT, здесь маскирование применяется многократно и на разных уровнях, поэтому вводятся весовые коэффициенты на каждом шаге. В итоге функция потерь принимает вид **Negative ELBO (NELBO)** — отрицательной вариационной нижней оценки лог-правдоподобия данных. В работе Sahoo et al. (2024) для дискретной диффузии было выведено выражение NELBO через сумму потерь на маскированные позиции:

$$
ELBO = \mathcal{L}_{\text{diff}} = \sum_{t=1}^T w_t \, \mathbb{E}_{x_t \sim q}\left[ -\log p_\theta\left(x_{t-1}^{\mathcal{M}_t} = X^{\mathcal{M}_t} \mid x_t\right) \right],
$$

где:
- $X^{\mathcal{M}_t}$ — истинные токены на позициях, замаскированных в состоянии $x_t$;
- Коэффициенты $w_t$ зависят от выбранного расписания шума (например, $w_t = 1$ для всех $t$ в простейшем случае, либо растут/убывают, отражая важность каждого шага).

Интуитивно модель учится *одновременно предсказывать маски разной "глубины"* — когда замаскировано 10% текста, 20%, … 100%. В пределе $t = T$ она решает задачу «угадай весь текст целиком по контексту = нулю» (что почти невозможно, но этот термин обучает сеть выдавать разумное априорное распределение).

Все описанные модели (Gemini, Mercury, LLaDA) следуют этой парадигме с некоторыми вариациями. Например, **Mercury** через концепцию *score entropy* фактически тоже оптимизирует аналог ELBO, но не напрямую через лог-правдоподобия токенов, а через обучение *восстановления соотношений вероятностей* (то есть вместо предсказания $P(y)$ модель оценивает $\log \frac{P(y)}{P(x)}$ для токенов $y$ и текущего $x$). Было показано, что этот подход эквивалентен новой формулировке задачи score matching в дискретном пространстве и обеспечивает более стабильное обучение, чем прямое предсказание токенов. Тем не менее результат тот же: Mercury обучается восстанавливать маскированную последовательность за несколько шагов, оптимизируя максимальное правдоподобие (через ELBO) и используя виды кросс-энтропийного loss для обновления весов.

Отдельно стоит отметить, что **Eso-LM** ввел новую модель правдоподобия, объединяющую AR и диффузию. Если обозначить через $z_0$ частично сгенерированную MDM-последовательность (с масками), а через $x$ — финальный текст, то полное распределение разложения можно записать как смешанное:

$$
P_\theta(x) = \sum_{z_0} P_\theta(x \mid z_0)\,P_\theta(z_0),
$$

где:
- $P_\theta(z_0)$ — вероятность получить черновик $z_0$ с помощью диффузионной части, а $P_\theta(x \mid z_0)$ — вероятность дописать его до $x$ AR-моделью. Вычислять эту сумму напрямую невозможно, поэтому авторы применили вариационный подход: ввели простое апостериорное распределение (которое маскирует случайные токены у полного $x$, чтобы получить $z_0$) и получили **ELBO для гибридного генератора**.

Интересно, что итоговая функция потерь опять распадается на два слагаемых:
1. NELBO диффузионной части (сумма masked LM потерь по шагам, как выше),
2. Обычная потеря автодополнения на токены, которые остаются на AR-этапе (тоже кросс-энтропия).

Это означает, что Eso-LM можно обучать единой процедурой end-to-end: на каждом примере сначала применить **стохастическое маскирование** (чтобы отделить будущие AR-токены), затем имитировать диффузию для восстановления остальных, и наконец добавить loss AR-модели на оставшиеся токены. Такой подход сохранил обоснованность с точки зрения вероятности (имеется нижняя оценка лог-правдоподобия) и позволил эффективно тренировать единый трансформер на комбинированную задачу.

В целом, математически диффузионные LLM расширяют пространство решений для моделирования $P(X)$. Они подтверждают общий принцип, что **ключевые свойства LLM (масштабируемость, обучение в контексте, следование инструкциям)** обеспечиваются не конкретно авторегрессией, а более фундаментально — мощью генеративного моделирования с максимизацией правдоподобия. Диффузионные модели, оптимизируя ELBO, реализуют те же принципы, только через иную факторизацию. Отличие лишь в том, что AR-модели — частный случай ($T = L$, каждое $x_{t-1}$ содержит один новый токен), а DLM — общий вариант ($T < L$ или даже $T \ll L$, при параллельном обновлении).

## Экспериментальные результаты

### Качество генерации и метрики

Несмотря на радикально другой механизм, современные диффузионные LM уже достигают сопоставимого с трансформерами качества текста по многим показателям. Авторы LLaDA сообщили, что их 8-миллиардная модель **по перплексии и zero-shot задачам** не уступает авторегрессионным LLM аналогичного размера. В частности, LLaDA-8B превзошла LLaMA2-7B почти во всех из 15 стандартных задач нулевого/малого-shot обучения и **приблизилась к уровню LLaMA3-8B**. После fine-tuning на инструкциях LLaDA показала уверенное следование инструкциям и диалоговые навыки, сравнимые с сильными LLM аналогичного масштаба. На прикладных бенчмарках LLaDA также убедительно выступила: например, на экзамене знаний MMLU и наборе математических задач GSM8K ее результаты оказались на одном уровне с авто-регрессионным базовым моделью, обученной на тех же данных. Более того, диффузионная модель **решила проблему “reversal curse”**, справившись с перевернутым стихотворением лучше, чем GPT-4(open). Однако стоит отметить, что LLaDA пока проверена только до 8B. Нет прямых цифр для большего масштаба – возможно, AR-модели всё ещё лидируют на сотнях миллиардов параметров, но исследование намекает, что разрыв не принципиален.

**Gemini Diffusion** (Google) в текущей реализации сопоставима по качеству с семейством Gemini 2.x. По заявлению DeepMind, на ряде внешних тестов **Gemini Diffusion показывает качество, сравнимое с гораздо более крупными моделями**, оставаясь быстрее их. На кодовых бенчмарках диффузионная Gemini почти не уступает авто-регрессионной Gemini Flash-Lite: например, HumanEval (процент успешных решений программных задач с первой попытки) \~**89.6% против 90.2%**, MBPP (бенчмарк по Python) \~76.0% vs 75.8% – фактически на одном уровне. На BigCodeBench также паритет \~45.4% vs 45.8%. Лишь на некоторых сложных задачах старой архитектуре удалось превзойти новую: например, Gemini Flash-Lite показала себя лучше на мульти-языковом тесте MMLU (79.0% против 69.1%) и на сложных логических задачах BIG-Bench Hard (21.0% vs 15.0%). Зато Gemini Diffusion неожиданно опередила авто-регрессию в математическом тесте AIME 2025 (23.3% vs 20.0%). На научном Q\&A (GPQA Diamond) диффузионная модель отстала (40.4% vs 56.5%), что свидетельствует о необходимости дообучения на фактах и знаниях. В целом, **разрыв в качестве между диффузией и AR минимален** на данном этапе. По словам ведущего инженера проекта, “по метрикам на относительно небольших моделях разница практически нивелирована”. О’Донохью (Google DeepMind) отмечает, что в некоторых областях у диффузии уже есть преимущества – например, задачи, требующие **глобальной согласованности** (программирование, сложное рассуждение) могут выиграть от нелокального внимания диффузионного подхода.

**Mercury Coder** был ориентирован на кодовые задачи, и там он достигает впечатляющих результатов, учитывая небольшой размер модели. В независимых тестах версия Mercury Coder Small уверенно обошла такие модели, как Gemini 2.0 Flash-Lite, Claude 3.5 Haiku, GPT-4о Mini и Qwen 2.5 Coder (7B) на **как минимум 4 из 6** стандартных бенчмарков по программированию. Mercury Coder Mini (более крупная версия) превзошла этих же конкурентов хотя бы на 2 из 6 наборов задач. Среди используемых метрик были упомянуты HumanEval (тест генерации кода по описанию), MBPP (Python-задания), MultiPL-E (мульти-язычный HumanEval), HumanEval+ (расширенная версия) и кодовые соревнования. Примечательно, что Mercury уступил лишь специализированной модели **DeepSeek Coder V2 Lite**, которая лидировала на всех 6 тестах. Это говорит о том, что диффузионная модель уже способна конкурировать с лучшими оптимизированными AR-моделями в узкой области (генерация кода) – хотя до абсолютного лидерства ещё есть пространство. Качественные характеристики Mercury также оцениваются через совокупные метрики “точность vs скорость”: в пространстве “качество решения vs токенов/с” Mercury находится в **преференциально выгодной зоне (Pareto-optimal)**, обеспечивая одновременно высокий балл за решения и производительность. Визуально на графиках видно, что Mercury Coder даёт близкое к GPT-4o качество кода, но при порядково меньшем времени генерации.

**Eso-LM** в основном измерялся перплексией (качество языковой модели) и характеристиками выборки. На классических датасетах LM1B и OpenWebText, Eso-LM (A) достиг перплексии 26.2 и \~30.5 соответственно, что значительно лучше, чем у прежних диффузионных LM на этих наборах. Это **лучший результат среди нелинейных (не-AR) методов** на данных корпусах. Трансформер всё ещё чуть впереди (например, условно 23.0 PPL на OWT у GPT-2 аналога), но отставание сократилось. Более того, включение последовательной фазы позволяет Eso-LM превзойти чисто диффузионные модели не только по перплексии, но и по *качеству сгенерированного текста*. Авторы проводили сравнительную оценку сэмплов: при равном числе шагов диффузии, Eso-LM генерировал **более осмысленные и разнообразные тексты**, не проявляя модового коллапса (который наблюдался у некоторых упрощенных схем типа BD3-LM на малом числе шагов). Также отмечено, что Eso-LM может гибко менять пропорцию параллельного и последовательного генерирования, добиваясь нужного компромисса. Например, вариант B (с кешированием) слегка уступает по перплексии варианту A, но при этом **гораздо быстрее** на длинных последовательностях, поэтому на практической генерации длинного текста (например, 1-2 тысячи символов) может давать лучший общий результат (если измерять условную “перплексию за секунду”).

### Скорость, латентность и эффективность

Главное преимущество диффузионных LLM – **скорость генерации**, особенно заметная на длинных ответах. Практические тесты показывают, что DLM могут выдавать ответ в несколько раз быстрее, чем сравнимые авторегрессионные модели. Например, **Gemini Diffusion** отвечает на сложные запросы (как генерация HTML-приложения с кодом) за **1–3 секунды**, тогда как автогрессия Gemini 2.5 Flash на тот же запрос тратит \~7 секунд. Измеренная производительность Gemini Diffusion варьируется от 600 до 1300 токенов/с в зависимости от задачи. Максимальные заявленные значения (1000–2000 токенов/с) в разы превосходят классические LLM: для сравнения, GPT-3/4 обычно не превышают 100–200 токенов/с даже при оптимизированном выводе, а многие большие модели выдают всего десятки токенов в секунду. **Mercury** побил своего рода рекорд, показав \~1100 токенов/с на H100 (для версии Mini), что сопоставимо с лучшими специализированными аппаратными ускорителями (ранее подобные цифры достигались только на нестандартных чипах вроде Groq или SambaNova). При этом Mercury работает на стандартных GPU, что подчёркивает **эффективность алгоритма**: как отмечает Эрмон, модель загружает GPU намного полнее, чем AR-LM, убирая простой между последовательными шагами.

С точки зрения *латентности ответа*, диффузионные модели особенно выгодны при генерации **больших фрагментов текста**. AR-модель вынуждена генерировать каждый из N токенов последовательно – время растет линейно $O(N)$. DLM же могут за фиксированное число итераций (например, 20–50) сгенерировать очень длинный текст, поэтому теоретическая сложность ближе к $O(\text{const})$. На практике, конечно, сложность тоже растет, но медленнее: например, 10k токенов Gemini Diffusion сгенерирует за считанные секунды (несколько итераций по 10k токенов сразу), тогда как 10k токенов автогрессии – это тысячи шагов декодера. Поэтому для сценариев, требующих **низкой задержки при большой длине ответа**, диффузионные LMs открывают новые возможности. К таким сценариям относятся: потоковое общение в чат-ботах (быстрый отклик без “напечатывания” по буквам), системы автодополнения кода в IDE (нужен почти мгновенный вывод большого куска кода), живой перевод и транскрипция речи, генерация длинных повествований и др..

Однако *время до первого токена (TTFT)* у диффузионных моделей обычно выше, чем у AR. Это связано с тем, что AR может сразу выдать первый символ практически без задержки (после одного шага), а диффузионной модели нужно закончить весь цикл итераций, прежде чем будет готов полный ответ. Пользователь может ощутить это как легкую паузу перед ответом, зато потом ответ появляется целиком сразу. В интерактивных приложениях это требует иного UX – например, вместо показа “печати” символов по одному, диффузионные системы могут отображать индикатор прогресса генерации и затем мгновенно выдать весь текст.

Что касается *энергоэффективности*, то прямые измерения пока мало представлены, но можно сделать некоторые выводы. Диффузионная генерация, хотя и требует несколько итераций, **лучше параллелится и задействует вычислительные ресурсы**. В AR-декодировании из-за последовательной природы современный GPU часто простаивает (загружено лишь одно “условие” на каждый шаг, а остальные тензорные мощности не используются). DLM наоборот, на каждом шаге обрабатывает большой объем данных (весь блок сразу), что ближе к полному заполнению GPU. Inception Labs утверждают, что Mercury за счёт алгоритма снижает стоимость инференса и делает его более предсказуемым (никаких взрывов по времени на длинных запросах). Можно ожидать, что *энергия на один токен* у диффузионных моделей выйдет меньше, чем у AR, именно из-за более высокой производительности – модель тратит чуть больше операций, но выдаёт на порядок больше символов в секунду, а значит, **операция на токен дешевле**. Кроме того, адаптивные механизмы, вроде переменного числа шагов в зависимости от сложности задачи, позволяют DLM экономить ресурсы: на простых запросах достаточно пары итераций (экономия времени и энергии), а на сложных система автоматически сделает больше шагов, потратив больше вычислений там, где нужно. Такой *адаптивный compute* трудно реализовать в обычных LLM, которые всегда прогоняют полный декодер для каждого нового токена, не умея “ускориться” на лёгких участках. Таким образом, диффузионные подходы имеют потенциал быть более энергоэффективными при масштабной генерации, хотя точные цифры появятся по мере дальнейших исследований.

В таблице ниже суммируем ключевые результаты по качеству и скорости для рассматриваемых моделей:

| Модель                        | Скорость (токенов/с)                     | Качество кода                      | Особенности                      |
|----------------------------   |------------------------------------------|----------------------------------- |----------------------------------|
| **Gemini Diffusion (Google)** | ~1000 токенов/с                          | HumanEval ~90%                     | —                                |
| **Mercury Coder (Inception)** | 737 (Small), 1109 (Mini) на H100         | На уровне GPT-4o Mini и Claude 3.5 |В **5–10×** быстрее аналогов      |
| **LLaDA 8B**                  | Нет точных данных (потенциал $T ≪ L$)    | —                                  |Силён в zero-shot и инструкциях, превосходит GPT-4o в "reverse" задачах |
| **Eso-LM (A/B)**              | До 65% ускорения (B)                     | Близко к AR                        |Лучший компромисс скорость/качество на длинах ~1K токенов |

## Сравнение с классическими трансформерами

### Преимущества парадигмы Diffusion LM:

**Высокая скорость и низкая задержка на длинных выводах.** 
- Параллельная генерация блоков текста позволяет DLM выдавать ответы на порядок быстрее, чем авторегрессионные модели аналогичного размера. Это открывает путь к использованию LLM в **реально-временных приложениях**, где AR-модели были слишком медленны (чат-боты со сложными ответами, голосовые ассистенты, IDE-помощники).

**Итеративное улучшение и самокоррекция.** 
- В отличие от AR, диффузионная модель не “навсегда” привязана к первоначальному выбору токена – каждый следующий шаг может исправить огрехи предыдущего. Это снижает риск накопления ошибок и **галлюцинаций**: если на ранней стадии ответ ушел не в ту сторону, модель может “переписать” проблемные места по ходу денойзинга.

**Более глобальная когерентность текста.** 
- Благодаря двунаправленному вниманию внутри генерируемого блока, DLM учитывают контекст будущих слов при генерации предыдущих. Это помогает обеспечивать согласованность окончания фразы с ее началом, соблюдение требований форматирования, согласование времён и чисел по всему тексту и т.д. – того, с чем иногда не справляются AR-модели, будучи ограничены только левым контекстом.

**Гибкость и редактируемость.** 
- Диффузионные модели по своей природе умеют заполнять пропуски в тексте и встраивать фрагменты, поскольку их обучение основано на masked-language modeling. Поэтому они естественно подходят для задач **интерактивного редактирования, дополнения в середине текста, стилистической трансформации готового текста**. Например, Gemini Diffusion имеет режим мгновенного редактирования произвольного текста с минимальным промптом. В AR-моделях такие возможности требуют специальных трюков (инструктирование, обучение на инфиллинг) и всё равно не столь надёжны.

**Адаптивные вычисления.** 
- DLM способны динамически менять число итераций генерации под задачу: простые запросы – меньше шагов (экономия ресурсов и времени), сложные – больше (повышение качества). AR-модель всегда делает $L$ шагов для $L$ токенов, не различая простоту задач. Адаптивность означает потенциально **более эффективное использование вычислений**, особенно при развёртывании системы на разнообразные запросы пользователей.

### Ограничения и вызовы:

**Сложность и стоимость обучения.** 
- Обучение диффузионных LM требует соблюдения тонкого баланса. Необходимы очень большие наборы данных и вычислительные ресурсы – например, LLaDA 8B обучалась на 2.3 трлн токенов, потратив 130 тыс. часов GPU. Кроме того, оптимизация ELBO с многократным маскированием сложнее, чем простая кросс-энтропия AR. Требуется тщательная настройка расписания шума, весовых коэффициентов, возможно, новых техник (как score entropy) для стабильности. Всё это делает разработку DLM более трудоёмкой.

**Время до первого символа.** 
- При использовании в интерфейсах, где пользователь ожидает увидеть поток текста (например, имитация набора ответа), DLM могут показаться “медленными”, так как ничего не выводят, пока не закончен весь блок. AR-модель начинает выдавать текст почти сразу токен за токеном. Это требует смещения парадигмы взаимодействия: **DLM предпочтительны там, где лучше сразу получить весь ответ целиком**, а не по частям. Для задач, где важна именно постепенность (например, авто-дополнение слова по первой букве), AR может иметь преимущество.

**Выделение памяти и вычислений.** 
- Диффузионные модели на каждом шаге обрабатывают весь контекст, что может потребовать больше памяти (для хранения активаций, хотя и меньше шагов). Однако разработки вроде Eso-LM (B) показывают, что этот барьер снимается: можно кешировать и в диффузии. В общем случае, если требуется очень длинный контекст (например, 100k токенов), AR-модель тоже столкнется с проблемами. DLM ещё предстоит доказать свою эффективность на экстремально больших контекстах.

**Точность на детальных задачах.**
- Хотя разрыв сокращается, в ряде случаев AR-модели пока показывают лучшее качество. В тестах Gemini, диффузионная модель отставала в вопросах, требующих точных знаний или многошагового логического вывода (например, сложные вопросы Big-Bench). Возможно, это из-за ограниченного пока размера диффузионных моделей. Нужны эксперименты с параметрами >20B, чтобы удостовериться, что DLM на больших масштабах сохраняют тенденцию к улучшению и догонят AR по всем метрикам.

### Потенциал замещения AR-моделей:

Многие эксперты считают, что диффузионный подход может стать **новым стандартом** для LLM в будущем. Google открыто заявляет о намерении исследовать этот путь для снижения латентности во всех моделях Gemini. Появление Mercury в коммерческом поле – знак, что индустрия ищет более быстрые решения для внедрения LLM. Andrew Ng отметил, что вход диффузионных моделей в текст – «cool attempt to explore diffusion models as an alternative, generating entire text at once... Congrats Stefano & team!». Преимущества по скорости и масштабируемости слишком значимы, чтобы ими пренебречь. Конечно, классические трансформеры ещё долго будут использоваться, тем более что их экосистема отточена (есть оптимизированные фреймворки, аппаратные ускорители, много опыта в fine-tuning). Но если DLM продолжат прогресс – особенно докажут себя на моделях 50B+ – то **парадигмальный сдвиг вполне вероятен**.

Можно предположить такую эволюцию: сперва диффузионные модели займут ниши, критичные к скорости (автодополнение, чаты с жёстким ограничением времени отклика, мобильные приложения, возможно, генерация кода в реальном времени). Затем, по мере оптимизации качества, они начнут вытеснять AR и из других областей. Гибридные варианты (как Eso-LM) могут послужить плавным мостом: используя лучшее от обоих подходов, они облегчат переход. Уже сейчас Eso-LM показывает, что можно получить **спектр моделей с разным соотношением скорость/точность**, и это ценнее жесткой альтернативы “либо быстро, либо точно”.

## Заключение и перспективы

Диффузионные языковые модели представляют собой многообещающий шаг вперед в NLP. Если за последние годы архитектура трансформера с авторегрессией казалась безальтернативной основой LLM, то новые исследования доказали обратное: **итеративная денойзинг-парадигма способна достичь сопоставимых результатов** при значительно лучшей производительности. Это открывает широкий простор для инноваций. В заключение отметим несколько прогнозов и наблюдений:

* **Парадигмальный сдвиг возможен, но потребует времени.** Подобно тому, как диффузионные модели революционизировали генерацию изображений, можно ожидать, что и в тексте они станут основным методом генерации, особенно когда вопрос стоит в **масштабировании**. Пользователи и разработчики всегда стремятся к более быстрому отклику моделей – это сильный драйвер к смене технологий. Тем не менее, полная замена AR-моделей случится не сразу: в ближайшее время вероятно сосуществование и специализация (AR – для задач, где требуется пошаговое предсказание или максимальная точность на малых моделях; Diffusion – для длинных текстов, интерактивных сценариев, быстрого прототипирования ответов).

* **Масштабируемость диффузионных моделей выглядит обнадеживающе.** Уже на 8B параметрах LLaDA показала тренды, аналогичные AR: больше данных и параметров – лучше результаты. Нет теоретических барьеров, чтобы обучить диффузионную модель с 70B+ параметров, разве что потребуются большие вычислительные затраты. Google, вероятно, этим уже занимается (семейство Gemini). Если Gemini Diffusion будет масштабирован до уровня GPT-4 (порядка сотен миллиардов параметров) и сохранит преимущество в скорости, это могло бы закрепить парадигму. В академической среде тоже набирает обороты исследование DLM: появляются варианты с мультимодальностью (например, **LLaDA-V** для одновременной работы с изображениями и текстом) и улучшенные алгоритмы обучения.

* **Практическое применение начнется с узких областей.** Mercury выбрал нишу кодогенерации – и не случайно: программисты особо ценят скорость автодополнения. Мы можем ожидать, что в IDE и кодовых ассистентах диффузионные модели закрепятся первыми. Другие вероятные области – системы **реального времени**, где задержка критична: автоперевод речи, подсказки в дополненной реальности, чат-боты службы поддержки. В корпоративных решениях, где важен контроль и кастомизация, DLM могут привлечь еще и способностью к точечному редактированию ответов (например, корректировать сгенерированный отчет без полной его перестройки).

* **Перспективные архитектуры:** среди рассмотренных подходов каждый имеет сильные стороны. **Gemini Diffusion**, будучи разработкой Google, вероятно получит преимущество доступа к масштабам данных и вычислений – эта архитектура может стать **локомотивом парадигмы**, если будет интегрирована в продукты. **Mercury** показал ценность *новых оптимизационных целей* (score entropy) – возможно, его методы лягут в основу многих последующих моделей, и не только коммерческих. **LLaDA** важна как доказательство, что **open-source сообщество** тоже способно внедрить диффузионные LLM, ее код и модель уже доступны, что позволит многим исследователям экспериментировать с диффузией. **Eso-LM** указывает направление гибридных решений – вероятно, крупные производители фреймворков (Hugging Face, PyTorch team) реализуют подобные “универсальные трансформеры” с гибким вниманием, раз это дает выигрыш. Наиболее перспективным с точки зрения практики выглядит **сочетание быстродействия и качества**: модели, которые смогут при необходимости работать как AR для тонкой доработки текста, но при этом выдавать черновой набросок мгновенно. Eso-LM (B) уже продемонстрировала такой подход с KV-кешем и ускорением, и это может стать базисом будущих промышленных DLM.

В итоге, диффузионные языковые модели сегодня находятся на грани превращения из научного эксперимента в **новый стандарт индустрии**. Если удастся преодолеть оставшиеся ограничения – улучшить “first-token latency”, упростить процесс обучения и довести качество до уровня лучших GPT-систем – у AR-моделей не останется конкурентных преимуществ. Как отмечено в сообществе, Google, обладая ресурсами, действительно может **сменить стандарт LLM с автогрессии на диффузию**. Даже если этого не случится мгновенно, сама альтернатива стимулирует прогресс: соревнование двух подходов ведет к лучшим решениям. Уже сейчас мы видим, как идеи из диффузии (маскирование, итеративное уточнение) начинают проникать в “классические” модели, а AR-модели перенимают некоторые трюки адаптивности. В ближайшие годы нас ждёт увлекательное развитие этой области, и возможно, появление совершенно **новых гибридных парадигм**, которые объединят сильные стороны всех подходов для достижения ещё более разумных и быстрых искусственных интеллектов.
