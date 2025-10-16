# From GPT-2 to gpt-oss: An Analysis of Architectural Advancements

> And how they stack up against Qwen3

On August 5, 2025, OpenAI released new open-weight LLMs: gpt-oss-120b and gpt-oss-20b—the first fully open models since the release of GPT-2 in 2019. And yes, thanks to some clever optimizations, you can run them locally (but more on that later).

This is the first time since GPT-2 that OpenAI has shared a large, fully open model. Early GPT models demonstrated how transformer architecture scales. Then, the release of ChatGPT in 2022 brought these models into the mainstream, showcasing their practical utility for writing, knowledge retrieval (and later, programming). Now, the company has shared the long-awaited model weights, and the architecture contains several interesting details.

I spent the past few days studying the code and technical reports to distill the most compelling insights. (Just a few days after this, OpenAI also announced GPT-5—I'll briefly touch on it in the context of gpt-oss models at the end of the article.)

Below is a brief overview of what this article covers. For convenient navigation, I recommend using the table of contents on the left side of the article page.

- Architecture comparison with GPT-2  
- MXFP4 optimization enabling gpt-oss models to run on a single GPU  
- Width vs. depth trade-offs (gpt-oss vs. Qwen3)  
- Attention biases and "sinks"  
- Benchmarks and comparison with GPT-5  

I hope you find this article helpful!

## 1. Model Architecture Overview

Before delving into detailed architecture comparisons, let's begin with an overview of the two models—`gpt-oss-20b` and `gpt-oss-120b`—shown in Figure 1 below.

![Two gpt-oss models side by side](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-01.png)

> Figure 1: Two gpt-oss models side by side.

If you've previously seen schematics of modern LLMs or read my earlier article "A Grand Architecture Comparison," you may notice that at first glance, there's nothing fundamentally new or unusual here.

This isn't surprising: leading LLM developers typically use the same base architecture, making only minor refinements. This is my personal hypothesis, but I believe the reasons are:

- Significant personnel rotation occurs between labs.
- We still haven't found anything better than the transformer architecture. While state space models and text diffusion models exist, to my knowledge, no one has demonstrated they perform as well as transformers at this scale. (Most comparisons I've found focus solely on benchmark scores. It remains unclear how well these models handle real-world multi-step writing and programming tasks. At the time of writing, the highest-ranked non-fully-transformer model on LM Arena—Jamba, a hybrid transformer and state space model—is ranked 96th. *NOTE: I was kindly pointed out that there is a higher-ranked hybrid model—Hunyuan-TurboS at 22nd place.*)
- Most improvements are likely achieved through data and fine-tuning algorithms, not radical architectural changes.

Nevertheless, their design choices contain many interesting aspects. Some are shown in the figure above (others aren't, but we'll discuss them later). In the remainder of this article, I'll sequentially highlight these features and compare them with other architectures.

I also want to emphasize that I am in no way affiliated with OpenAI. My information is based solely on studying the published model code and technical reports. If you want to learn how to run these models locally, the best place to start is OpenAI's official model pages on the Hugging Face Hub:

- https://huggingface.co/openai/gpt-oss-20b  
- https://huggingface.co/openai/gpt-oss-120b  

The 20-billion-parameter model (`gpt-oss-20b`) can run on a consumer GPU with 16 GB VRAM. The 120-billion-parameter model (`gpt-oss-120b`) can run on a single NVIDIA H100 GPU with 80 GB VRAM or newer hardware. I'll return to this later, as there are several important caveats.

---

## 2. The Legacy of GPT-2

Before diving into comparisons with modern architectures, let’s take a journey back in time and compare gpt-oss side-by-side with GPT-2 (Figure 2) to vividly see the journey taken.

![Comparison of gpt-oss-20b and GPT-2 XL 1.5B architectures](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-02.jpg)

> Figure 2: Comparison of gpt-oss-20b and GPT-2 XL 1.5B architectures.

Both gpt-oss and GPT-2 are LLMs using only the decoder and built on the transformer architecture introduced in the seminal paper "Attention Is All You Need" (2017). Over the years, many details of this architecture have evolved.

However, these changes are not unique to gpt-oss. As we'll see later, they are common across many modern language models. Since I've already discussed many of these aspects in my previous article "A Grand Architecture Comparison," I'll aim to be concise and focus on key points.

### 2.1 Abandonment of Dropout

Dropout (2012) is a classic method for preventing overfitting that randomly "turns off" (i.e., zeros out) parts of layer activations or attention scores (Figure 3) during training. However, in modern large language models, dropout is rarely used, and most models released after GPT-2 have abandoned it.

![Illustration of dropout applied to attention score matrix](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-03.png)

> Figure 3: Illustration of dropout applied to attention score matrix.

One might assume dropout was initially used in GPT-2 as an inheritance from the original transformer architecture. Researchers likely noticed it provided no real performance improvement for LLMs (I observed the same in my small GPT-2 reproduction experiments). This is because LLMs are typically trained for just one epoch on massive datasets, unlike the hundreds of epochs for which dropout was originally designed. Since LLMs see each token only once during training, the risk of overfitting is low.

Interestingly, although dropout has been largely ignored in LLM architecture design for years, I found a 2025 research paper with experiments on relatively small models (Pythia 1.4B) confirming that under one-epoch training, dropout degrades final model quality.

### 2.2 RoPE Replaces Absolute Positional Embeddings

In transformer LLMs, positional encoding is necessary due to the attention mechanism. By default, attention treats input tokens as if they have no order. In the original GPT architecture, this problem was solved with absolute positional embeddings: a learned vector corresponding to the token's position in the sequence was added to the token vector (Figure 4).

![Illustration of absolute positional embeddings](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-04.jpg)

> Figure 4: Illustration of absolute positional embeddings.

RoPE (Rotary Position Embedding) proposed a different approach: instead of adding positional information as separate vectors, it encodes position by rotating the query and key vectors, with the rotation depending on each token's position. (The idea behind RoPE is elegant, but explaining it is complex—I plan to break it down separately.)

First introduced in 2021, RoPE gained widespread adoption with the release of the original Llama model in 2023 and has since become standard for modern LLMs.

> ⚓ [Example code implementation of RoPE](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/main/2025/week-39-40/experiments/domain/positional_encoding/rope.py)

### 2.3 Swish/SwiGLU Replaces GELU

Early GPT architectures used the activation function GELU. Why use Swish instead of GELU now? Swish (also known as the Sigmoid Linear Unit, SiLU) is computationally slightly cheaper, and in my view, that's the entire reason. Depending on which paper you read, you'll find one function slightly better than the other in terms of modeling performance. In my opinion, these small differences likely fall within standard error margins, and the specific result will heavily depend on hyperparameter fine-tuning.

Activation functions were a hot topic for debate until the deep learning community largely settled on ReLU over a decade ago. Since then, researchers have proposed and tested many variants similar to ReLU but with smoother curves; GELU and Swish (Figure 5) are among those that stuck.

![Comparison of Swish and GELU activation functions—smoother versions of ReLU](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-05.jpg)

> Figure 5: Comparison of Swish and GELU activation functions—smoother versions of ReLU.

Early GPT architectures used GELU, defined as:

$$
\frac{x}{2} \cdot \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$ 

Here, $\text{erf}$ (from English *error function*) is the integral of the Gaussian function, computed via polynomial approximations, making it computationally more expensive than simpler functions like the sigmoid used in Swish ($x * \text{sigmoid}(x)$).

In practice, Swish is slightly cheaper to compute than GELU, and this is likely the main reason it replaced GELU in most new models. Depending on the paper, one function may appear slightly better in modeling quality. But I would say these improvements often fall within error margins, and the winner will heavily depend on hyperparameter tuning.

Swish is used in most modern architectures. However, GELU is not entirely forgotten; for example, Google's Gemma models still use GELU.

However, a more significant change is that the feed-forward module (a small multi-layer network) has been replaced by its "gated" variant—GLU (Gated Linear Unit), proposed in a 2020 paper. Specifically, two fully connected layers are replaced with three, as shown in Figure 6 below.

![Comparison of a standard feed-forward layer with its gated variants SwiGLU and GEGLU](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-06.jpg)

> Figure 6: Comparison of a standard feed-forward layer with its gated variants SwiGLU and GEGLU.

At first glance, SwiGLU/GEGLU variants may seem better than standard layers simply because they have more parameters due to the additional layer. But this is misleading because in practice, the weight matrices $W$ and $V$ in SwiGLU/GEGLU are usually chosen to be half the size of the matrix $W_1$ in the traditional feed-forward layer.

To illustrate this better, consider concrete code implementations:

![Standard feed-forward module (top) and SwiGLU variant (bottom) side by side](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-07.jpg)

> Figure 7: Standard feed-forward module (top) and SwiGLU variant (bottom) side by side. Note that the Swish function is implemented as "silu" in PyTorch.

Suppose the embedding dimension is 1024. For a standard feed-forward layer:
*   `fc1`: 1024 × 4096 = 4,194,304 parameters
*   `fc2`: 4096 × 1024 = 4,194,304 parameters
*   Total: 8,388,608 parameters.

For the GLU variant:
*   `fc1`: 1024 × 1024 = 1,048,576 parameters
*   `fc2`: 1024 × 1024 = 1,048,576 parameters
*   `fc3`: 1024 × 1024 = 1,048,576 parameters
*   Total: 3 × 1,048,576 = 3,145,728 parameters.

Thus, using GLU variants ultimately results in *fewer* total parameters while still demonstrating better performance. The reason is that these variants provide additional multiplicative interactions, increasing the network's expressiveness (for the same reason deep and narrow networks can outperform wide and shallow ones with quality training).

> ⚓ [Example code implementation of SwiGLU](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/main/2025/week-39-40/experiments/domain/activations/swiglu.py)

### 2.4 Mixture-of-Experts Instead of a Single FeedForward Module

In addition to updating the feed-forward module to SwiGLU, as discussed in the previous section, gpt-oss replaces the single feed-forward module with multiple such modules, using only a subset for each token generation step. This approach is known as Mixture-of-Experts (MoE) and is illustrated in Figure 8 below.

![Feed-forward module replaced with a Mixture-of-Experts (MoE) module](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-08.png)

> Figure 8: Feed-forward module replaced with a Mixture-of-Experts (MoE) module.

Thus, replacing a single feed-forward module with multiple ones (as implemented in the MoE architecture) significantly increases the model's total number of parameters. However, the key trick is that not all "experts" are activated (used) for each token. Instead, a special router selects only a small subset of experts for each specific token.

Since only a few experts are active simultaneously, MoE modules are often called sparse, in contrast to dense modules that always use the full set of parameters. While the large total parameter count enabled by the MoE architecture enhances the language model's capacity to absorb more knowledge during training, sparsity preserves inference efficiency because not all parameters are activated simultaneously.

(An interesting fact: In most MoE models, expert weights constitute over 90% of the model's total parameters.)

> ⚓ [Example code implementation of Mixture-of-Experts](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/tree/main/2025/week-39-40/experiments/domain/moe)

### 2.5 Grouped Query Attention Instead of Multi-Head Attention

As mentioned in my previous articles, Grouped Query Attention (GQA) has become a more computationally and parameter-efficient alternative to classical Multi-Head Attention (MHA) in recent years.

In MHA, each "head" has its own key and value projections. GQA reduces memory consumption by grouping multiple heads together to share the same key and value projections.

For example, as shown in Figure 9, if we have 2 key-value groups and 4 attention heads, heads 1 and 2 can share one key-value pair, while heads 3 and 4 share another. This grouping reduces the total number of key and value computations, leading to lower memory usage and improved efficiency without noticeable degradation in model quality, according to ablation studies.

![Comparison of MHA and GQA](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-09.png)

> Figure 9: Comparison of MHA and GQA. Here, the group size is 2, where a key-value pair is shared by two queries.

Thus, the core idea of GQA is to reduce the number of key and value heads by having multiple query heads share the same keys and values. This (1) reduces the model's total parameter count and (2) lowers memory bandwidth consumption during inference, since fewer key-value (KV) pairs are stored and retrieved.

(If you're curious how GQA looks in code, see my guide converting GPT-2 to Llama 3—it includes a version without KV cache and my version with KV cache.)

Although GQA primarily serves as a tool for improving computational efficiency compared to MHA, ablation studies (e.g., in the original GQA paper and the Llama 2 paper) show that it is comparable to standard MHA in modeling quality.

> ⚓ [Example code implementation of Grouped Query Attention](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/main/2025/week-39-40/experiments/domain/attention/gqa.py)

### 2.6 Sliding Window Attention

Sliding window attention (Figure 10 below) was first proposed in the LongFormer paper (2020) and later gained widespread adoption thanks to Mistral. Notably, in gpt-oss, it is applied in every second layer. It can be viewed as a variant of multi-head attention (in this case, Grouped Query Attention) where the attention context is limited to a small window, reducing both memory and computational costs.

![Comparison of standard attention (left) and sliding window attention (right)](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-10.jpg)

> Figure 10: Comparison of standard attention (left) and sliding window attention (right).

Specifically, gpt-oss alternates between layers of GQA with full context access and layers of GQA with a sliding window limited to 128 tokens.

As I discussed in my previous article, Gemma 2 (2024) used a similar 1:1 ratio. Gemma 3, released earlier this year, went even further and switched to a 5:1 ratio—meaning only one layer with full attention per five layers with local (windowed) attention.

According to ablation studies within the Gemma project, using sliding window attention has virtually no impact on modeling quality, as shown in the figure below. Note that the window size in Gemma 2 was 4096 tokens, and in Gemma 3, it was reduced to 1024. In gpt-oss, the window is only 128 tokens—surprisingly small.

As an interesting fact: the official announcement paper notes that sliding window attention appears to have been used in GPT-3:

> “The models use alternating dense and locally-sparse sparse attention patterns, similar to GPT-3.”

Who would have thought! I reread the original GPT-3 paper, and it indeed mentions:

> “We use the same model and architecture as in GPT-2 [RWC+19], including the modified initialization, pre-normalization, and reversible tokenization described in that work, except that in the transformer layers we apply alternating dense and locally-sparse sparse attention patterns, similar to Sparse Transformer [CGRS19].”

### 2.7 RMSNorm Instead of LayerNorm

Finally, the last minor improvement compared to GPT-2 is replacing LayerNorm (2016) with RMSNorm (2019), which has become a general trend in recent years.

Similar to replacing GELU with Swish and SwiGLU, RMSNorm is another small but sensible efficiency improvement. RMSNorm, like LayerNorm, is designed to normalize layer activations, as shown in Figure 11 below.

Perhaps you recall that not long ago, BatchNorm was the de facto standard. However, it lost popularity primarily because it is difficult to efficiently parallelize (due to the need to compute batch statistics—mean and variance) and performs poorly with small batch sizes.

![Comparison of LayerNorm (left) and RMSNorm (right) on a small linear layer](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-11.jpg)

> Figure 11: Comparison of LayerNorm (left) and RMSNorm (right) on a small linear layer.

As seen in Figure 11, both LayerNorm and RMSNorm scale layer outputs to a reasonable range of values.

LayerNorm subtracts the mean and divides by the standard deviation so that layer outputs have zero mean and unit variance (variance = 1, standard deviation = 1).

RMSNorm divides inputs by the root-mean-square. This scales activations to a comparable magnitude but does not force them to zero mean or unit variance. In the example above, the mean is 0.77 and the variance is 0.41.

Both normalizations stabilize activation scales and improve trainability, but RMSNorm is often preferred in large-scale LLMs because it is cheaper to compute. Unlike LayerNorm, RMSNorm has no bias term and replaces expensive mean and variance calculations with a single root-mean-square computation. This reduces the number of inter-feature reductions from two to one, lowering GPU communication overhead and improving training efficiency.

Figure 12 shows how this looks in code:

![Implementations of LayerNorm and RMSNorm in code](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-12.jpg)

> Figure 12: Implementations of LayerNorm and RMSNorm in code, demonstrating that RMSNorm is computationally simpler.

> ⚓ [Example code implementation of RMSNorm](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/main/2025/week-39-40/experiments/domain/normalization/rmsnorm.py)

### 2.8 The Legacy of GPT-2

I still believe GPT-2 is an excellent architecture for beginners learning about LLMs. It is simple enough to avoid getting lost in layers of optimization tricks yet complex enough to provide a solid understanding of how modern transformer models work.

Starting with GPT-2 allows you to focus on fundamental concepts (attention mechanisms, positional embeddings, normalization, and the overall training pipeline) without being overwhelmed by additional features and refinements characteristic of newer architectures.

Moreover, I believe it's worthwhile to spend time studying and even implementing GPT-2 yourself *before* layering on more modern changes. You won't only understand these innovations more easily but are likely to appreciate them more—you'll have a clear understanding of the limitations or problems they aim to solve.

For example, building on my GPT-2 codebase, I recently implemented Qwen3 from scratch, which, as we'll see, is very similar to gpt-oss. This leads us to the next topic: comparing gpt-oss with a more modern architecture.

---

## 3. Comparing gpt-oss with a Modern Architecture (Qwen3)

Now that we've traced the evolution from GPT-2 to GPT OSS, we can move to the next step and compare GPT OSS with a more modern architecture—Qwen3, released three months earlier in May 2025.

The reason I chose Qwen3 is that, at the time of writing, it is one of the best open-weight models available. Moreover, one of Qwen3's MoE models is broadly comparable in size to gpt-oss.

Figure 13 below shows a comparison between gpt-oss-20b and a comparable-sized Qwen3 model.

![gpt-oss and Qwen3 models of comparable size side by side](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-13.png)

> Figure 13: gpt-oss and Qwen3 models of comparable size side by side.

As seen, gpt-oss-20B and Qwen3-30B-A3B are very similar in architectural components. The main difference (aside from dimensions) is that gpt-oss uses sliding window attention, as discussed in Section 2.6 (not shown in the figure), whereas Qwen3 does not employ this mechanism.

We'll now examine the most notable details in order.

### 3.1 Width vs. Depth

A careful comparison of both models reveals that Qwen3 is a significantly deeper architecture: it has 48 transformer blocks compared to 24 in gpt-oss-20b (Figure 14).

![Qwen3 has twice as many transformer blocks as gpt-oss-20b](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-14.png)

> Figure 14: Qwen3 has twice as many transformer blocks as gpt-oss-20b.

Conversely, gpt-oss is a much wider architecture:

- Embedding dimension: 2880 vs. 2048
- Intermediate projection dimension of the experts (feed-forward): also 2880 vs. 768

It is also worth noting that gpt-oss uses twice as many attention heads, although this does not directly increase model width—width is determined by the embedding dimension.

Does one approach offer an advantage at a fixed parameter count? Generally, deeper models possess greater flexibility but are harder to train due to instability issues—exploding and vanishing gradients—which are mitigated by RMSNorm and skip connections.

Wider architectures benefit from faster generation speed (more tokens per second) due to better parallelization, at the cost of higher memory consumption.

Regarding modeling quality, unfortunately, I am unaware of any good "apple-to-apple" comparisons (where model size and training data are strictly fixed) except for an ablation study in the Gemma 2 paper (Table 9). There, for a 9B-parameter architecture, the wider configuration slightly outperformed the deeper one: on average across four benchmarks, the wide model scored 52.0 versus 50.8 for the deep model.

### 3.2 A Few Large Experts vs. Many Small Ones

As shown in Figure 14 above, it is notable that gpt-oss has surprisingly few experts (only 32 instead of 128), and only 4 are activated per token (instead of 8). However, each of these experts is significantly larger than those in Qwen3.

This is interesting because recent trends indicate benefits from a greater number of smaller experts. This change, at a fixed total parameter count, is well illustrated by Figure 15 from the DeepSeekMoE paper.

![Annotated figure from the DeepSeekMoE paper](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-15.jpg)

> Figure 15: Annotated figure from "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", https://arxiv.org/abs/2401.06066

It should be noted that, unlike DeepSeek models, neither gpt-oss nor Qwen3 use shared experts.

To be fair, the small number of experts in gpt-oss may be a side effect of its 20B parameter size. Looking at the 120B model (Figure 16 below), we see that the number of experts (and transformer blocks) has indeed been increased, while everything else remains unchanged.

![Two gpt-oss architectures side by side](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-16.png)

> Figure 16: Two gpt-oss architectures side by side: in the larger 120B model, only the number of transformer blocks and experts are scaled.

The most mundane explanation for this similarity between the 20B and 120B models is likely that primary focus was on the 120B model, and the 20B version was obtained by simply shortening (fewer blocks) and reducing the number of experts—since these components contain the bulk of the parameters. One might also speculate that they began training the 120B model and then "cut off" portions of blocks and experts for fine-tuning (rather than initializing from scratch).

In any case, scaling *only* these two components is highly unusual. For example, examining MoE models of various sizes in Qwen3 (Figure 17 below), it is clear they are scaled more proportionally across parameters.

![Architectural differences among various Qwen3 models](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-17.jpg)

> Figure 17: Architectural differences among various Qwen3 models.

### 3.3 Attention Biases and "Sinks"

Both gpt-oss and Qwen3 utilize Grouped Query Attention. The primary difference is that gpt-oss limits context length using sliding window attention in every second layer, as previously mentioned.

However, another interesting nuance caught my attention: it appears that gpt-oss employs bias terms in its attention weights, as shown in the figure below.

![gpt-oss models use bias units in attention layers](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-18.png)

> Figure 18: gpt-oss models use bias units in attention layers. See code example here.

I have not encountered such biases since the GPT-2 era, and they are generally considered redundant. Indeed, a recent paper mathematically demonstrates this is at least true for the key projection (k_proj). Moreover, empirical results show nearly zero difference in performance between models with and without biases (see Figure 19 below).

![Table from a paper showing average test error when training models with and without bias units](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-19.jpg)

> Figure 19: Table from https://arxiv.org/pdf/2302.08626, showing average test error when training models with and without bias units.

Another detail you may have noticed in the code screenshot (Figure 18) is the definition of "sinks." In general, attention sinks are special tokens at the beginning of a sequence to which attention is always applied to stabilize its operation, particularly in long-context scenarios. When the context becomes very long, a token at the beginning remains in the attention focus and can learn to store useful general information about the entire sequence. (This idea was first proposed in the paper "Efficient Streaming Language Models with Attention Sinks.")

In gpt-oss's implementation, attention sinks are not real tokens in the input sequence. Instead, they are learnable bias logits added to the attention scores for each head (Figure 20). The goal is the same, but without altering the tokenized input.

![Use of attention sinks in gpt-oss](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-20.jpg)

> Figure 20: Use of attention sinks in gpt-oss; based on Hugging Face code here.

### 3.4 License

Finally, like Qwen3, the gpt-oss models are released under the open Apache 2.0 license—this is excellent (it's the same license I prefer for my own open-source projects). This means the models can be used for distillation into other models or in commercial products without restrictions.

**Open Weights vs. Open Source Code.** This question has been debated for years, but it is worth clarifying to avoid confusion around this release. Some model developers release only weights and inference code (e.g., Llama, Gemma, gpt-oss), while others (e.g., OLMo) release everything—including training code, datasets, and weights—meeting the strict definition of "open source code."

By this strict criterion, gpt-oss is a model with open weights (like Qwen3), as it includes weights and inference code but not training code or datasets. However, industry terminology is inconsistent.

I assume "oss" in "gpt-oss" stands for "open source software"; however, I was pleasantly surprised that OpenAI itself, in its official announcement, clearly refers to gpt-oss as an open-weight model.

---

## 4. Other Interesting Details

While the previous sections described the evolution of the architecture since GPT-2 and discussed its similarities with Qwen3 (and most other modern models), there are several important nuances I have not yet mentioned. They don't quite fit into the previous sections but deserve attention.

### 4.1 Training Overview

Unfortunately, information about dataset sizes and training algorithms is scarce. Below I have compiled the most interesting fragments from the model card (1) and the announcing post (2):

> The gpt-oss models were trained using our most advanced pretraining and post-training methods [...] (1)  
> [...] training took 2.1 million hours on H100 GPUs, with gpt-oss-20b requiring nearly 10 times less. (1)  
> [...] including supervised learning and high-cost reinforcement learning stages [...] (2)  
> We trained the models on a predominantly English text dataset with an emphasis on STEM, programming, and general knowledge. (2)

Thus, we know gpt-oss are reasoning-oriented models. The computational volume—2.1 million H100 hours—is roughly comparable to the 2.788 million H800 hours spent training DeepSeek V3, which is nearly 5.6 times larger. Unfortunately, data on Qwen3's training time is currently unavailable.

Interestingly, the training time estimate for gpt-oss includes both supervised learning for instruction following and reinforcement learning for reasoning, whereas DeepSeek V3 is merely a pre-trained base model upon which DeepSeek R1 was separately trained.

### 4.2 Reasoning Levels

As mentioned, gpt-oss are reasoning-oriented models. But particularly interesting is that they are trained such that users can easily control the degree of reasoning directly during inference.

Specifically, gpt-oss models can receive instructions like "Reasoning effort: low/medium/high" in the system prompt, which directly affects the length and precision of the response, as shown in Figure 21.

![Length and quality of gpt-oss model responses at different reasoning levels](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-21.jpg)

> Figure 21: Length and quality of gpt-oss model responses at different reasoning levels (annotated figure from model card).

This flexibility is useful as it allows balancing cost, computation, and accuracy. For example, if the task is simple—answering a direct question or correcting a typo—extended reasoning can be skipped. This saves time and resources, avoiding overly long answers and verbose reasoning chains.

It is somewhat unfortunate that, unlike Qwen3 or OLMo, OpenAI did not release base models *before* the reinforcement learning stage for reasoning. Base models are especially valuable for researchers working on reasoning methods (which is why I currently prefer using Qwen3 Base). Presumably, OpenAI's decision was driven by industrial and production scenarios rather than research needs.

Note that the original Qwen3 models also had a switch to enable/disable reasoning mode (via the `enable_thinking=True/False` parameter in the tokenizer, which simply added `think` tags to disable reasoning). However, the Qwen3 team recently updated their models and abandoned the hybrid approach in favor of specialized variants: Instruct / Thinking / Coder.

The reason is that the hybrid mode showed inferior performance compared to separate models:

> After discussion with the community and reflection, we decided to abandon the hybrid reasoning mode. We will now train Instruct and Thinking models separately to achieve the best quality. Source

### 4.3 MXFP4 Optimization: A Small but Important Detail

One interesting surprise was that OpenAI released the gpt-oss models with MoE expert quantization in MXFP4 format.

Previously, quantization formats were a niche topic relevant mainly to mobile and embedded AI, but everything changed with the growth of model sizes. In this case, MXFP4 optimization enables running the model on a single GPU.

Here's how it looks in practice:

- The large model (120B) fits on a single 80GB GPU (H100 or newer). This is not consumer hardware, but renting a machine with a single H100 is much cheaper than one with multiple GPUs. Plus, there's no need to worry about model distribution across GPUs and communication overhead. It's pleasant that AMD MI300X support is available from the start!
- The smaller model (20B) fits even within 16GB of VRAM; however, this requires an RTX 50-series or newer GPU supporting MXFP4. (Note: a recent patch added support for older cards like the RTX 4090.)

It should be noted that the models will still work on older hardware without MXFP4, but with significantly higher RAM consumption. Without MXFP4 optimization, the models in bfloat16 format would require approximately 48 GB (gpt-oss-20b) and 240 GB (gpt-oss-120b).

Incidentally, I smoothly run gpt-oss-20b on my Mac Mini via Ollama. It uses about 13.5 GB of memory—quite reasonable.

### 4.4 Benchmarks

The models are still too new for independent benchmarks. Checking the LM Arena leaderboard, I found that gpt-oss is not yet represented. Thus, Qwen3-Instruct remains the best open-weight model according to LM Arena users (Figure 22).

![Current state of the LM Arena leaderboard](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-22.png)

> Figure 22: Current state of the LM Arena leaderboard (as of August 8, 2025).

However, according to reasoning benchmarks from the gpt-oss announcement post, these models are comparable to both proprietary OpenAI models and Qwen3 (Figure 23).

![Main benchmark charts from the official gpt-oss announcement](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-23.png)

> Figure 23: Main benchmark charts taken from the official gpt-oss announcement. Data for gpt-oss-120b without tools is from the official model card, and Qwen3 figures are from the official Qwen3 repository.

However, it should be noted that gpt-oss-120b is nearly half the size of the Qwen3 A235B-A22B-Thinking-2507 model and yet runs on a single GPU.

Benchmark performance, however, does not always reflect real-world usability. After a few days of limited testing, I found gpt-oss to be quite competent. Nevertheless, as with others, I noticed its relatively high tendency toward hallucinations (a point also mentioned in its model card).

This may be related to the strong focus on reasoning tasks—math, puzzles, code—which may have led to some "forgetting" of general knowledge. However, since gpt-oss was originally designed with tool use in mind, this limitation may become less significant over time. Tool integration in open-source LLMs is still in its early stages, but as it develops, I expect models will increasingly rely on external sources (e.g., search engines) when answering factual questions.

If this happens, it will be more logical to prioritize reasoning ability over memorization. This resembles education (or life itself), where problem-solving skills are often more important than rote memorization.

---

## 5. gpt-oss and GPT-5

OpenAI had a busy week: shortly after releasing gpt-oss, the company unveiled the long-awaited GPT-5 model. The release of GPT-5 was intriguing. And if I must say one thing about it, I was genuinely surprised by how good their open-source models are compared to their own best commercial product—judging by the benchmarks (Figure 24).

![Main benchmark charts from the official GPT-5 announcement](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-39-40/assets/Figure-24.jpg)

> Figure 24: Main benchmark charts taken from the official GPT-5 announcement. gpt-oss data is from the model card and announcement; Qwen3 figures are from the official Qwen3-Coder repository.

Overall, despite some calling the release overhyped, I am pleased that we now have a new set of genuinely strong open-weight models that are not so far behind the best proprietary counterparts. Of course, benchmarks often don't reflect real-world usage, and it is still too early to draw conclusions based on limited experience. But I believe now is an excellent time for those who enjoy working with open-weight models—locally or in private infrastructures.