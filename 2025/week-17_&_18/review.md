# **Mamba 2 + Transformer = Nemotron H**

## **Introduction**

Transformers today are the gold standard of neural networks, especially large language models. They became the first truly scalable architecture, meaning that for the first time, it became possible to reliably improve model performance by increasing the amount of data and parameters without hitting hardware performance or neural network memory ceilings.

It was the transformer that transformed the artificial intelligence industry into the powerful force we see today. Before 2017, when researchers at Google Brain invented this architecture, the cornerstone of the AI industry was finding the right model structure. Now, scientists face entirely different challenges, and companies and researchers barely think about architecture anymore—because the transformer exists!

This is how renowned Andrej Karpathy, former ML director at Tesla and co-founder and former chief scientist at OpenAI, describes this architecture: "The transformer is not just another method—it’s an approach that completely changed our view of AI. We were incredibly lucky to stumble upon it in the vast space of algorithms. I believe the transformer is superior to the human brain in many ways."

However, despite all its strengths, the transformer has its shortcomings. Therefore, some research groups continue searching for a better algorithm that could surpass the transformer or at least match its performance. In this article, we will explore why this task is so non-trivial and what exactly in the transformer still leaves room for improvement.

## **Why Transformers Are So Hard to Replace**

To understand this, let’s dive deeper into this architecture. What exactly is a transformer?

The transformer’s origins lie in the now-iconic paper "Attention Is All You Need," published in 2017 by eight researchers from Google. Notably, all eight authors are listed as equal contributors—an unusual rarity in scientific papers. Interestingly, none of these eight researchers now work at Google. Almost all of them became founders of well-known AI startups, including Cohere, Character.ai, Adept, Inceptive, Essential AI, and Sakana AI.

Historically, before transformers, the dominant LLM architecture was recurrent neural networks (RNNs). RNNs, along with their advanced variants like LSTM and GRU, processed information sequentially, like a person reading left to right. Yet, this algorithm is a significant simplification of human reading. At the core of these architectures is a hidden state that is recursively updated at each step (hence the name). However, as we know, relationships between words can be more complex: they don't always manifest sequentially. Therefore, processing words (or rather, tokens) strictly one after another causes us to lose the ability to capture relationships between words that are not adjacent. The model may simply "forget" something important before it gets the chance to recognize its relevance to later text.

Thus, the next major milestone in NLP development was the attention mechanism. Traditionally, it is believed that this mechanism was invented in 2014 by one of the fathers of deep learning, Yoshua Bengio. The essence of the mechanism lies in "weighting" the relevance of all tokens in a sequence relative to each other: each token with every other. In practice, this is implemented as the multiplication of three tensors: Query, Key, and Value. Each of these matrices is obtained by multiplying the input embeddings X by learnable weights W. Query, Key, and Value can be thought of as components necessary for "intelligent search" across the sequence: queries, keys, and values. Through sequential multiplication of these matrices (as shown in the image below), we obtain the attention mechanism, which reveals the significance of relationships between words. Thus, with attention, we can account for relationships between words in a passage regardless of how far apart they are.

![Figure_03](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_03.png)

However, the mere appearance of the attention mechanism did not revolutionize artificial intelligence. Before the transformer paper, researchers used attention only as an addition to RNN architectures. The breakthrough achieved by Google’s team was precisely that they invented an architecture that completely abandoned the RNN concept and relied entirely on the attention mechanism. Hence the paper’s title: "Attention Is All You Need" (of course, without a nod to the famous Beatles song, it wouldn’t have been complete). Incidentally, the established terms Query, Key, and Value were also introduced in this research. Thus, the transformer was born, whose fundamental innovation was the ability to process sequences in parallel rather than sequentially. This gives the model the capacity not only to globally understand the texts it reads and writes but also to train and scale efficiently. The transformer can "digest" massive amounts of information and grow to enormous parameter counts. Meanwhile, its performance does not plateau—it continues to improve. This is another crucial distinguishing feature of this architecture.

![Figure_04](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_04.webp)

Today, transformers have fully captured the AI industry and research. All popular chatbots today—ChatGPT from OpenAI, Gemini from Google, Claude from Anthropic, Grok from xAI—are based on transformers. The same applies to image generation tools: Midjourney, Stable Diffusion, Runway, and others. These networks are built on diffusion models, which internally, in turn, use transformers. Additionally, the architecture is applied in molecular structure prediction models, robotics, and self-driving cars. Co-author of the transformer paper, Ashish Vaswani, aptly described this model: "The transformer is a way to very quickly simultaneously capture all relationships between different parts of any input. These can be parts of a sentence, musical notes, pixels, or protein molecules. It’s suitable for any task." However, transformers are not without drawbacks. Today we will examine the Mamba architecture, which aspires to become a competitor to transformers and address their vulnerabilities, as well as the Nemotron-H family of models from NVIDIA, which represent a hybrid architecture combining the strengths of the transformer with the efficiency of Mamba layers.

NVIDIA’s Nemotron-H models strategically replace the majority of self-attention layers in transformers with Mamba layers, which are based on State Space Models (SSM). Unlike self-attention, whose computational and memory complexity scales quadratically with sequence length, Mamba layers offer constant computational and memory complexity per token, making them especially efficient for generating long sequences.

![Figure_01](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_01.jpeg)

*Comparison of Throughput and Accuracy  
Figure 1: Comparison of Nemotron-H models against other modern LLMs in terms of throughput (tokens/s/GPU) and accuracy on the MMLU benchmark. Nemotron-H-56B offers 2.4x higher throughput than Llama-3.1-70B while achieving higher accuracy levels.*

The key innovation of Nemotron-H lies in carefully balancing these two architectural paradigms to maintain or improve accuracy while significantly increasing inference speed. This approach addresses the critical community need for LLMs capable of efficiently handling long contexts without sacrificing performance.

## **Mamba Architecture Overview**

Mamba is an innovative architecture based on Structured State Space Models (SSM). It was designed for efficient identification of complex dependencies in sequential data and is positioned as a serious transformer competitor. The architecture combines advantages of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), achieving linear or nearly linear computational scaling with respect to sequence length.

### **Key Advantages of Mamba**

1. **Selective Mechanism**  
   - A simple and efficient mechanism for filtering out irrelevant information has been introduced.  
   - Allows retention of necessary data through parameterized SSM parameters.

2. **Hardware-Oriented Algorithm**  
   - Uses recursive scanning instead of traditional convolutional computations.  
   - Optimized for GPUs, delivering up to 3x speedup on A100 GPUs.

3. **Modeling Capabilities**  
   - Maintains performance comparable to transformers.  
   - Exhibits nearly linear scalability, making it suitable for handling long and complex data sequences.

### **Applications of Mamba**

Mamba demonstrates outstanding performance across various domains:  
- **Computer Vision**:  
  The Vim model, based on Mamba, is 2.8x faster than DeiT at extracting features from high-resolution images and saves 86.8% GPU memory.  
- **Natural Language Processing (NLP)**:  
  Enhanced selective SSM architecture provides 2–8x speedups.  
- **Code Generation (Text-to-Code)**:  
  Mistral developed Codestral, an SSM-based model that outperforms nearly all other open models on benchmarks.

Before diving into a deep review of Mamba’s architectural specifics, let’s clarify what RNNs, LSTMs, GRUs, and SSMs are 👇

<details> 
    <summary><em><strong> 🔥 Recurrent Neural Network (RNN)</strong></em></summary>

## **1. Introduction and Motivation**

### **1.1 Why Recurrent Networks Are Needed**
- **Sequential data**: language, time series, audio, DNA sequences.  
- **Temporal dependencies**: Fully connected networks treat inputs as independent; RNNs store context in a hidden state $h_t$.

### **1.2 History**

- **1982 — Hopfield Network.**  
  Demonstrated that a neural network with symmetric weights can function as an energy-based associative memory model. John Hopfield’s work was the first demonstration of trainable recurrent connections in neuro-computation.

- **1986 — BPTT Algorithm (Rumelhart & McClelland).**  
  The authors generalized classical back-propagation to temporally unfolded graphs, opening the path to gradient-based learning of long sequences. The book *Parallel Distributed Processing* cemented the idea of distributed representations.

- **1990 — "Simple RNN" (Elman).**  
  D. Elman showed that a recurrent "context" layer could capture grammatical dependencies in a synthetic language. Thus, the basic Elman-net architecture emerged, becoming a textbook RNN benchmark.

- **1997 — LSTM (Hochreiter & Schmidhuber).**  
  Introduction of a memory cell and gating mechanisms solved the vanishing gradient problem, enabling modeling of dependencies hundreds of steps back. LSTM soon became the standard for speech recognition and machine translation.

- **2014 — GRU (Cho et al.).**  
  Reducing the number of gates to two, GRU offered a lighter alternative to LSTM with comparable accuracy. The publication coincided with the boom in seq2seq models for translation and dialogue systems.

- **2020s — RNN + Attention Hybrids (RWKV, S4, Mamba).**  
  Modern works combine linear recurrent operators with attention layers, achieving transformer-like scalability with $O(1)$ memory. Such models successfully compete on long-context and streaming tasks.

## **2. Simple RNN (Elman Cell): How Does It Work?**

### **2.1 Intuition**

Imagine you are reading a sentence word by word. To understand the meaning of the current word, you rely not only on the word itself but also on the context accumulated from previous words. The Simple RNN works similarly:

*   At each time step $t$, it takes:
    1.  **A new input** $x_t$ (e.g., a vector representation of a word).
    2.  **The state from the previous step** $h_{t-1}$ (context, "memory").
*   Based on these two inputs, it computes:
    1.  **A new state** $h_t$, which will be passed to the next step.
    2.  **An output** $y_t$ (e.g., prediction of the next word or label for the current element).

### **2.2 Formalization and Notation**

Let’s describe this mathematically. First, define the tensor (vector/matrix) dimensions:

| **Object** | **Dimension**        | **Meaning**                                    |
| :--------- | :------------------- | :--------------------------------------------- |
| $x_t$      | $\mathbb{R}^{d_x}$   | Input vector at time $t$                       |
| $h_t$      | $\mathbb{R}^{d_h}$   | Hidden state vector at time $t$                |
| $y_t$      | $\mathbb{R}^{d_y}$   | Model output vector at time $t$                |
| $W_{xh}$   | $\mathbb{R}^{d_x \times d_h}$ | Input → hidden state weight matrix         |
| $W_{hh}$   | $\mathbb{R}^{d_h \times d_h}$ | Previous state → current state weight matrix (recurrent connection) |
| $W_{hy}$   | $\mathbb{R}^{d_h \times d_y}$ | Hidden state → output weight matrix        |
| $b_h$      | $\mathbb{R}^{d_h}$   | Hidden layer bias vector                       |
| $b_y$      | $\mathbb{R}^{d_y}$   | Output layer bias vector                       |

> **Why track dimensions?** This helps avoid errors in matrix operations and when writing code (especially with broadcasting in libraries like NumPy/PyTorch).

### **2.3 Dynamics of One Step**

Now write the formulas describing the transition from step $t-1$ to step $t$:

{% raw %}

$$
\boxed{%
\begin{aligned}
h_t &= \sigma_h\!\bigl(W_{xh}x_t + W_{hh}h_{t-1} + b_h\bigr), & h_0&=\mathbf0,\\[4pt]
y_t &= \sigma_y\!\bigl(W_{hy}h_t + b_y\bigr).
\end{aligned}}
$$

{% endraw %}

**Explanations:**

1.  **Computing hidden state $h_t$:**
    *   $W_{xh}x_t$: Influence of current input $x_t$ on the new state.
    *   $W_{hh}h_{t-1}$: Influence of previous state $h_{t-1}$ (memory) on the new state. This is the **key recurrent connection**.
    *   $b_h$: Bias.
    *   $\sigma_h$: Hidden layer activation function. Often **tanh** or **sigmoid** is used, as they "compress" values into bounded ranges ([-1, 1] for tanh, [0, 1] for sigmoid), which can help stabilize gradients during training.
    *   $h_0 = \mathbf{0}$: Start with a zero hidden state vector before processing the first sequence element.

2.  **Computing output $y_t$:**
    *   $W_{hy}h_t$: Transformation of current hidden state $h_t$ into output representation.
    *   $b_y$: Output layer bias.
    *   $\sigma_y$: Output layer activation function. Its choice **depends on the task**:
        *   `softmax`: for classification tasks (e.g., predicting the next symbol/word from a vocabulary).
        *   `sigmoid`: for binary classification (e.g., sentiment analysis: positive/negative).
        *   `id` (linear activation, i.e., none): for regression tasks (predicting a numerical value).

![Image_01](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/RNN/Image_01.webp)

```python
"""
This code implements a simple recurrent neural network (RNN) for processing word sequences.
It includes model parameter initialization, a softmax function, and the main RNN loop that processes
an input word sequence and outputs predictions for each word in the sequence.

Functional Purpose:
The code demonstrates RNN operation on a text sequence example. It initializes weights and biases,
performs embedded operations (one-hot encoding, embedding, hidden state computation, softmax),
and outputs the top-2 predictions for each word in the sequence.
"""

import numpy as np
import pandas as pd

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Description:
    ---------------
        Computes softmax for the input array.

    Args:
    ---------------
        x: Input array for which to compute softmax.

    Returns:
    ---------------
        Array with softmax applied.

    Raises:
    ---------------
        ValueError: If input array is empty.

    Examples:
    ---------------
        >>> softmax(np.array([1, 2, 3]))
        array([0.09003057, 0.24472847, 0.66524096])
    """
    if x.size == 0:
        raise ValueError("Input array cannot be empty")

    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

# ---------------- Model Parameters ----------------
vocab = ["the", "students", "opened", "their", "books", "laptops", "zoo"]
V = len(vocab)
d_e, d_h = 8, 16  # embedding and hidden state dimensions

# Dictionary: word → index
word2idx = {w: i for i, w in enumerate(vocab)}

# Initialize weights
np.random.seed(0)
E = np.random.randn(d_e, V) * 0.1      # embeddings
W_e = np.random.randn(d_h, d_e) * 0.1  # hidden ← embedding
W_h = np.random.randn(d_h, d_h) * 0.1  # hidden ← hidden
b1 = np.zeros((d_h, 1))                # hidden layer bias
U = np.random.randn(V, d_h) * 0.1      # hidden → logits projection
b2 = np.zeros((V, 1))                  # output layer bias

# --------------- Matrix Visualization ----------------
df_E = pd.DataFrame(
    E, index=[f"e{i}" for i in range(d_e)], columns=vocab
)
df_We = pd.DataFrame(
    W_e, index=[f"h{i}" for i in range(d_h)], columns=[f"e{j}" for j in range(d_e)]
)
df_Wh = pd.DataFrame(
    W_h, index=[f"h{i}" for i in range(d_h)], columns=[f"h{j}" for j in range(d_h)]
)
df_U = pd.DataFrame(
    U, index=vocab, columns=[f"h{j}" for j in range(d_h)]
)

print("\nMatrix E (embeddings):")
print(df_E)
print("\nMatrix W_e (hidden ← embedding):")
print(df_We)
print("\nMatrix W_h (hidden ← hidden):")
print(df_Wh)
print("\nMatrix U (projection to output):")
print(df_U)

# --------------- Main RNN Loop ----------------
sequence = ["the", "students", "opened", "their"]
h_prev = np.zeros((d_h, 1))

print("\nStep  t    Word       Top-2 (word, prob)")
print("-" * 60)
for t, word in enumerate(sequence, 1):
    print(f"\n## Step-by-step breakdown for t={t}, word = '{word}'")

    # 1) One-hot
    x = np.zeros((V, 1))
    x[word2idx[word], 0] = 1.0
    print("1) One-hot vector x:")
    print(x.T)

    # 2) Embedding
    e = E @ x
    print("\n2) Embedding e = E @ x:")
    print(e.T)

    # 3) Hidden state
    h = np.tanh(W_h @ h_prev + W_e @ e + b1)
    print("\n3) Hidden state h:")
    print(h.T)

    # 4) Logits and softmax
    o = U @ h + b2
    y = softmax(o)
    print("\n4) Logits o = U @ h + b2:")
    print(o.T)
    print("   Softmax y:")
    print(y.T)

    # Top-2 candidates
    top2 = np.argsort(-y.flatten())[:2]
    probs = [(vocab[i], float(y[i])) for i in top2]
    print(f"\nTop-2 candidates: {probs}")

    # Update hidden state
    h_prev = h
```

### **Explanation of the "Simple RNN Language Model" Diagram (step by step)**

1. **Input Feeding**  
   - At each step $t$, we have a word as a one-hot vector  
     $$x^{(t)} \in \mathbb{R}^{|V|}$$  
     where $|V|$ is the vocabulary size.
     
   - Example: For vocabulary $\{\text{the}, \text{students}, \text{opened}, \dots\}$, the word "students" is encoded as a vector with 1 at the "students" position and 0 elsewhere.

2. **Embedding Transformation**  
   - Multiply one-hot $x^{(t)}$ by the embedding matrix  
     $$E \in \mathbb{R}^{d_e \times |V|}$$  
     to obtain a dense vector  
     $$e^{(t)} = E \, x^{(t)} \in \mathbb{R}^{d_e}$$

3. **Hidden State Update**  
   - Recurrent formula:  

    $$
      h^{(t)} = \sigma\bigl(W_h \, h^{(t-1)} + W_e \, e^{(t)} + b_1\bigr)
    $$  
     
     where  
     - $h^{(t)} \in \mathbb{R}^{d_h}$ — hidden state at step $t$,  
     - $W_h \in \mathbb{R}^{d_h \times d_h}$ — hidden state transition matrix,  
     - $W_e \in \mathbb{R}^{d_h \times d_e}$ — input embedding matrix,  
     - $b_1 \in \mathbb{R}^{d_h}$ — bias vector,  
     - $\sigma$ — nonlinearity (usually $\tanh$ or ReLU).

   Initialization:  

     $$h^{(0)} = \mathbf{0}\quad(\text{or random vector}).$$  
   - When computing output, bias $b_1$ is added to $W_h\,h^{(t-1)} + W_e\,e^{(t)}$, and bias $b_2$ is added to $U\,h^{(t)}$, followed by softmax on logits.

4. **Output Computation**  
   - Construct logits for vocabulary distribution:  
     $$
       o^{(t)} = U \, h^{(t)} + b_2,\qquad U\in\mathbb{R}^{|V|\times d_h},\;b_2\in\mathbb{R}^{|V|}.
     $$  
   - Apply softmax to obtain probability distribution:  
     $$
       \hat y^{(t)} = \mathrm{softmax}\bigl(o^{(t)}\bigr)\in[0,1]^{|V|},\quad\sum_i \hat y^{(t)}_i = 1.
     $$  
   - Vector $\hat y^{(t)}$ indicates which word the model considers most likely to follow at position $t+1$.

5. **Repeat Over Time**  
   - Weight matrices ($W_{xh}, W_{hh}, W_{hy}$) and bias vectors ($b_h, b_y$) are **identical across all time steps $t$**. The network uses the same set of parameters to process each sequence element. This makes RNNs compact in parameter count, independent of sequence length $T$.

## **3. Training RNN: Backpropagation Through Time (BPTT)**

We have defined how an RNN makes predictions (forward pass). But how do we adjust its weights $W_{xh}, W_{hh}, W_{hy}, b_h, b_y$ to make predictions accurate? For this, we need an error backpropagation algorithm adapted for recurrent structure — **Backpropagation Through Time (BPTT)**.

### **3.1 Idea: Temporal Unrolling**

To apply gradient descent, we need to compute gradients of the loss function $L$ with respect to all model parameters. The challenge is that output $y_t$ depends on $h_t$, which depends on $h_{t-1}$, which depends on $h_{t-2}$, and so on, all the way back to $h_0$. Moreover, all $h_k$ (for $k < t$) depend on the same weights $W_{hh}$ and $W_{xh}$.

The idea of BPTT is to **mentally "unroll" the RNN in time** for a sequence of length $T$. Imagine you have $T$ copies of the same RNN cell connected in sequence. Input $x_t$ and previous state $h_{t-1}$ are fed into the $t$-th copy, which produces $h_t$ and $y_t$, and $h_t$ is passed to the $(t+1)$-th copy.

![Image_02](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/RNN/Image_02.png)

### **Explanation of the "Simple RNN Language Model" Diagram (step by step)**

Below is illustrated how forward and backward passes (BPTT) occur in an unrolled RNN model for the phrase "the students opened their".

#### **1. Temporal Unrolling**

- Each rectangle on the diagram corresponds to one time step $t=0,1,2,3$.  
- Inputs: $x_0,x_1,x_2,x_3$ — one-hot vectors for words "the", "students", "opened", "their".  
- Initial hidden state $h_{-1}$ is initialized to zeros.  
- Hidden states $h_0\ldots h_3$ are sequentially passed along the edge $W_{hh}$.  
- At each step, output $\hat y_t$ is computed from $h_t$ via $W_{hy}$.

#### **2. Forward Pass**

#### **Step 0 ($t=0$), word "the"**

1. **One-hot representation**:  
   $ x_0 $ — a unit vector with 1 at the position corresponding to the word "the".

2. **Embedding (vector representation)**:  
   $ e_0 = E\,x_0 $,  
   where $ E $ is the embedding matrix.

3. **Hidden state**:  
   $$
   h_0 = \tanh\bigl(W_{xh} e_0 + W_{hh} h_{-1} + b_1\bigr),
   $$  
   where:
   - $ W_{xh}, W_{hh} $ — weight matrices,
   - $ b_1 $ — bias,
   - $ h_{-1} $ — initial hidden state (usually a zero vector).

4. **Model output and softmax**:  
   $$
   o_0 = W_{hy}h_0 + b_2,\quad \hat{y}_0 = \mathrm{softmax}(o_0),
   $$  
   where:
   - $ W_{hy} $ — matrix transforming hidden state to logits,
   - $ b_2 $ — bias,
   - $ \hat{y}_0 $ — probability distribution over the vocabulary.

5. **Loss function**:  
   Target word — "students". Loss is computed as:  
   $$
   L_0 = -\log\hat{y}_0[\text{students}].
   $$

#### **Detailed Explanation of the Loss Function**

The model uses the **cross-entropy loss function** for multiclass classification. Consider its steps:

1. **Logits and probabilities**:  
   The model outputs a vector of logits:  
   $$
   o_0 = W_{hy}h_0 + b_2,
   $$  
   which is then converted to probabilities via softmax:  
   $$
   \hat{y}_0 = \mathrm{softmax}(o_0) \in [0, 1]^{|V|}, \quad \sum_i \hat{y}_0[i] = 1.
   $$

2. **Target label**:  
   Target label $ y^{(0)} $ is a one-hot vector with 1 at the position of the target word "students":  
   $$
   y^{(0)}_{\text{students}} = 1.
   $$

3. **Cross-entropy**:  
   Cross-entropy formula:  
   $$
   L_0 = -\sum_{i=1}^{|V|} y^{(0)}_i \log\hat{y}_0[i] = -\log\hat{y}_0[\text{students}].
   $$

4. **Intuition**:  
   The lower the predicted probability $ \hat{y}_0[\text{students}] $, the higher the penalty (loss value).

#### **Step 1 ($t=1$), word "students"**
- Similarly: $x_1$ → $e_1$ →  
  $$h_1 = \tanh(W_{xh}e_1 + W_{hh}h_0 + b_1).$$  
- Output $\hat y_1 = \mathrm{softmax}(W_{hy}h_1+b_2)$,  
  target word "opened", $L_1=-\log\hat y_1[opened]$.

#### **Step 2 ($t=2$), word "opened"**
- $x_2$ → $e_2$ →  
  $$h_2 = \tanh(W_{xh}e_2 + W_{hh}h_1 + b_1).$$  
- $\hat y_2$, target "their", $L_2=-\log\hat y_2[their]$.

#### **Step 3 ($t=3$), word "their"**
- $x_3$ → $e_3$ →  
  $$h_3 = \tanh(W_{xh}e_3 + W_{hh}h_2 + b_1).$$  
- $\hat y_3$, target "books", $L_3=-\log\hat y_3[books]$.

- **Total loss**:  
  $$L = L_0 + L_1 + L_2 + L_3.$$  

#### **3. Backward Pass (Backward pass — BPTT)**

- Gradients from each $L_t$ (red arrows) flow through:
  - Output layer $W_{hy}$ to hidden states,
  - Recurrent connections $W_{hh}$ to previous $h_{t-1}$.
- At each step, accumulate $
  \frac{\partial L}{\partial W_{xh}},
  \frac{\partial L}{\partial W_{hh}},
  \frac{\partial L}{\partial W_{hy}},
  \frac{\partial L}{\partial b_1},
  \frac{\partial L}{\partial b_2}$.
- Finally, weights are updated considering the contribution of errors from all time steps.

**Conclusion:** BPTT unrolls the RNN in time, computes local losses at each step, and propagates errors through all temporal connections, enabling training that accounts for context from previous tokens.

Although we create $T$ copies for computation, remember: **weights $W_{xh}, W_{hh}, W_{hy}$ are shared across all these copies**.

#### **3.2 Overall Loss Function**

Typically, the total loss $L$ for the entire sequence is the sum or average of local losses $\ell$ at each step:

$$
L \;=\;\sum_{t=1}^{T}\,\ell\bigl(y_t,\widehat y_t\bigr),
$$

where $y_t$ is the model's prediction at step $t$, and $\widehat y_t$ is the true value (target) at step $t$. Function $\ell$ can be, for example, cross-entropy for classification or mean squared error (MSE) for regression.

#### **3.3 Gradient Computation (Example for $W_{hh}$)**

Consider how to compute the gradient of the total loss $L$ with respect to one element $w$ from matrix $W_{hh}$. Using the chain rule, the gradient of $L$ with respect to $w$ is the sum of contributions from each time step $t$:

$$
\frac{\partial L}{\partial w}\;=\; \sum_{t=1}^{T}\,\frac{\partial \ell(y_t, \widehat y_t)}{\partial w}
$$

To find $\frac{\partial \ell(y_t, \widehat y_t)}{\partial w}$, we must account for how $w$ affects $y_t$. This influence occurs through the hidden state $h_t$:

$$
\frac{\partial \ell(y_t, \widehat y_t)}{\partial w} = \frac{\partial \ell}{\partial y_t} \frac{\partial y_t}{\partial h_t} \frac{\partial h_t}{\partial w}
$$

The most complex part is $\frac{\partial h_t}{\partial w}$. State $h_t$ depends on $w$ directly (via the term $W_{hh}h_{t-1}$ in the formula for $h_t$) and indirectly, through all previous states $h_{t-1}, h_{t-2}, \dots, h_1$, since they too depend on $w$.

$$
\frac{\partial h_t}{\partial w} = \underbrace{\frac{\partial h_t}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial w}}_{\text{via } h_{t-1}} + \underbrace{\frac{\partial h_t}{\partial w}}_{\text{direct influence}}
$$

Expanding this recursion further, we see that the gradient includes a **sum of paths** of varying lengths from the past to the present. Each such path includes products of Jacobians $\frac{\partial h_k}{\partial h_{k-1}}$.

$$
\frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial}{\partial h_{k-1}} \sigma_h(W_{xh}x_k + W_{hh}h_{k-1} + b_h) = \operatorname{diag}\!\bigl[\sigma_h'(a_k)\bigr]\,W_{hh}
$$
where $a_k = W_{xh}x_k + W_{hh}h_{k-1} + b_h$ is the argument of activation function $\sigma_h$ at step $k$. Denote this Jacobian as $J_k$.

Then the contribution to the gradient from a path of length $k$ (from $h_{t-k}$ to $h_t$) includes the product of $k$ such Jacobians: $J_t J_{t-1} \dots J_{t-k+1}$.

#### **3.4 Problems: Vanishing and Exploding Gradients**

It is precisely these **long products of Jacobians** $J_k = \operatorname{diag}[\sigma_h'(a_k)] W_{hh}$ that cause problems in training RNNs:

1.  **Vanishing Gradient**: If the eigenvalues of matrix $W_{hh}$ (or norms of Jacobians $J_k$) are **less than 1** in magnitude, multiplying many such matrices causes the result to decay exponentially toward zero as $k$ increases. This means gradients from distant past steps ($t-k$ for large $k$) barely reach parameters $W_{hh}$, and the network cannot learn **long-term dependencies**. Simple RNNs are especially susceptible to this problem.
2.  **Exploding Gradient**: If the eigenvalues of $W_{hh}$ (or norms of $J_k$) are **greater than 1** in magnitude, the product of Jacobians grows exponentially. This leads to enormous gradient values, making gradient descent steps unstable and potentially causing training divergence (NaN/Inf in losses or weights).

#### **3.5 Classical Solutions**

*   **Gradient Clipping**: Artificially constraining the gradient norm. If $\|\nabla\theta\| > \tau$ (some threshold), scale the gradient: $\nabla\theta \leftarrow \frac{\tau}{\|\nabla\theta\|} \nabla\theta$. This helps combat *exploding*, but not *vanishing*.
*   **Proper Weight Initialization**: For example, orthogonal initialization for $W_{hh}$ can help keep eigenvalues close to 1.
*   **Use of More Complex Cells**: **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** were specifically designed to combat vanishing gradients. They introduce "gates" that control the flow of information and gradients through the cell, allowing information to be retained over long periods.
*   **Activation Functions**: Using ReLU may exacerbate exploding gradients but is less prone to vanishing than sigmoid/tanh (if activation is non-zero). However, in recurrent parts, tanh is often preferred.


## **4. BPTT in Practice: Pseudo-code and PyTorch**

Modern deep learning frameworks (PyTorch, TensorFlow/Keras) implement BPTT automatically. You only need to define the RNN architecture and run the backward pass (`loss.backward()` in PyTorch).

Here is a typical training loop using BPTT in PyTorch (using `tanh` as $\sigma_h$ and linear $\sigma_y$):

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Hyperparameters and data (example values) ---
T = 10      # Sequence length
batch_size = 32
d_x = 20    # Input dimension
d_h = 50    # Hidden state dimension
d_y = 5     # Output dimension

# --- Model (define parameters) ---
W_xh = torch.randn(d_x, d_h, requires_grad=True)
W_hh = torch.randn(d_h, d_h, requires_grad=True)
W_hy = torch.randn(d_h, d_y, requires_grad=True)
b_h  = torch.zeros(d_h, requires_grad=True)
b_y  = torch.zeros(d_y, requires_grad=True)
params = [W_xh, W_hh, W_hy, b_h, b_y]

# --- Example data ---
x_sequence = torch.randn(T, batch_size, d_x) # [Time, Batch, Features]
y_true_sequence = torch.randn(T, batch_size, d_y)

# --- Optimizer ---
optimizer = optim.Adam(params, lr=0.001)

# --- Training loop (one iteration) ---
optimizer.zero_grad()

# == Forward pass (manual unrolling for clarity) ==
h_t = torch.zeros(batch_size, d_h) # Initial hidden state h_0
outputs = []
for t in range(T):
    # Simple RNN formula
    h_t = torch.tanh(x_sequence[t] @ W_xh + h_t @ W_hh + b_h)
    y_t = h_t @ W_hy + b_y # Linear output layer
    outputs.append(y_t)

# Stack outputs into one tensor [T, Batch, d_y]
y_pred_sequence = torch.stack(outputs)

# == Compute loss ==
# Example: MSE at each step, then average over time and batch
loss = F.mse_loss(y_pred_sequence, y_true_sequence)

# == Backward pass (BPTT) ==
loss.backward() # PyTorch automatically computes gradients ∂L/∂params via BPTT

# == Optional: Gradient Clipping ==
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0) # Clamp gradient norm

# == Optimizer step ==
optimizer.step()

print(f"Loss: {loss.item()}")
# print(f"Gradient norm for W_hh: {W_hh.grad.norm().item()}") # Can inspect gradient norm
```

**Key Points:**

*   PyTorch builds a dynamic computational graph during the forward pass.
*   When `loss.backward()` is called, PyTorch traverses this graph in reverse order, applying the chain rule (implementing BPTT) to compute gradients for all parameters (`requires_grad=True`) that affect `loss`.
*   `torch.nn.utils.clip_grad_norm_` is standard practice to prevent gradient explosion.

## **5. The Problem of Long-Term Dependencies**

One of the appealing ideas behind RNNs is their potential to link prior information with the current task—for example, knowledge of a previous video frame can aid in understanding the current frame. If RNNs possessed this capability, they would be extremely useful. But do RNNs truly provide us with this ability? It depends on certain circumstances.

Sometimes, only recent information is needed to perform the current task. Consider, for instance, a language model attempting to predict the next word based on preceding ones. If we want to predict the final word in the sentence “Облака плывут по небу,” we don’t need broader context; it’s obvious that the last word will be “небу.” In such cases, where the distance between relevant information and the point where it is needed is small, RNNs can learn to utilize past information.

But there are cases where more context is required. Suppose we wish to predict the final word in the text: “Я вырос во Франции… Я бегло говорю по-французски.” The immediate context suggests the last word is the name of a language, but to determine which one, we need the distant context of “Франции.” Thus, the gap between relevant information and its point of application can become very large.

Unfortunately, as this distance grows, RNNs lose their ability to link information.

![Image_03](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/RNN/Image_03.png)

Fortunately, LSTM does not suffer from these problems!

## **Conclusion**

Recurrent neural networks, with their compact memory and natural causality, remain indispensable for streaming tasks and resource-constrained scenarios. A deep understanding of their mathematical foundations, engineering techniques, and modern variations equips researchers with a tool that harmoniously complements the family of Transformer-like models.

</details>

<details> 
    <summary><em><strong> 🔥🔥 Long Short-Term Memory (LSTM)</strong></em></summary>

## 1. Introduction and Motivation

### 1.1 History of LSTM: Key Works by Hochreiter and Schmidhuber, Evolution of the Idea

**Origins (1991–1997)**

The vanishing gradient problem was formally identified in the early 1990s by Sepp Hochreiter in his doctoral thesis. Hochreiter and Jürgen Schmidhuber began searching for an architecture capable of overcoming this issue.

**Key Milestones:**

- **1991–1995**: Early experiments and theoretical developments. Hochreiter and Schmidhuber explored various ways to allow gradients to flow through long sequences without vanishing.

- **1997**: Publication of the original paper "Long Short-Term Memory" in Neural Computation. This foundational work introduced the LSTM architecture with gating mechanisms and the Constant Error Carousel (CEC).

**Evolution of LSTM:**

- **1999–2000**: Felix Gers and colleagues introduced "peephole connections"—a modification allowing gates to "peek" into the cell state.

- **2000**: Gers and Schmidhuber introduced the "forget gate"—a critical improvement enabling LSTMs to reset their state and learn on sequences of arbitrary length.

- **2005**: Graves and Schmidhuber presented bidirectional LSTM (BiLSTM), which processes sequences in both directions.

- **2013–2014**: Era of widespread industrial adoption of LSTM, especially in speech recognition and machine translation.

- **2014**: Development of GRU (Gated Recurrent Unit) by Cho et al. as a simpler alternative to LSTM, preserving most advantages.

**Schmidhuber’s Quote on LSTM’s Creation (2015):**

> "The problem was that gradients either vanished or exploded, and we needed to create an architecture that allowed information and gradients to flow across many time steps."

### 1.2 Key Advantages: Why LSTM Became the Industry Standard

**1. Solving the Vanishing Gradient Problem**

LSTM effectively solves the vanishing gradient problem through its unique cell memory mechanism with controlled gates. The key component is the **Constant Error Carousel (CEC)**, which ensures a constant gradient flow through the cell state.

**2. Long-Term Memory**

LSTMs can retain information over hundreds or even thousands of time steps:
- Demonstrate superior ability to capture long-range dependencies
- Can selectively preserve important information and discard irrelevant details
- Enable modeling of context across multiple temporal scales simultaneously

**3. Adaptability to Diverse Data Types**

LSTMs operate effectively on various types of sequential data:
- Text and speech (machine translation, speech recognition)
- Time series (financial forecasting, sensor data)
- Biological sequences (DNA, protein analysis)
- Multimodal data (image captions, video analytics)

**4. Scalability and Flexibility**

- Ability to build deep architectures by stacking LSTM layers
- Compatibility with other neural network types (CNNs, Attention)
- Efficient parallelization of training on modern hardware

**5. Industrial Impact**

Before the emergence of transformers (2017–2018), LSTM was the absolute industry and research standard:

- **Google** (2015–2016): Used LSTM in speech recognition systems, reducing error rates by 30%
- **Apple**: Integrated LSTM into Siri to improve contextual understanding
- **Facebook**: Applied LSTM for automated message translation
- **Amazon**: Utilized LSTM in recommendation systems and demand forecasting

Even after the advent of transformers, LSTM remains relevant in several domains:
- Processing streaming data in real time
- Tasks with limited computational resources
- Applications requiring model interpretability

**The Evolution of LSTM Popularity** reflects their significance: from academic research in the late 1990s, to industrial dominance in the mid-2010s, and subsequent integration with attention-based architectures.

## 2. LSTM Architecture: How It Works

### 2.1 Intuition: Metaphor of a Conveyor Belt with Controlled Gates

To understand how LSTM operates, imagine it as a smart production line with conveyor belts and gates:

**1. Main Conveyor — Cell State**

The core component of LSTM is the **cell state**—a horizontal line running through the entire chain. Think of a long conveyor belt stretching through the entire factory (sequence). On this belt moves a "container of information" (cell state, $C_t$).

![Image_01.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_01.png)

**Characteristics of the Cell State:**
- It passes directly through the entire chain, undergoing only a few linear transformations.
- Information can flow easily along it without alteration.
- This container can preserve information nearly unchanged over long periods—this is the core secret of LSTM.

However, LSTM can **remove information** from the cell state; this process is regulated by structures called **gates**.

**2. Controlled Gate System**

At each step (time point), our conveyor passes through three checkpoints:

- **Forget Gate**: Acts as a filter deciding what information to discard from the container. Imagine a worker examining the container’s contents and the current input, then deciding what to throw out: "Do we still need to remember the subject’s gender to correctly align pronouns? Yes, keep it. But the color of their car? No, discard it."

- **Input Gate**: Determines what new information to add to the container. Imagine another worker, looking at the current input and previous output, deciding which new facts are important enough to remember: "We just learned the name of a new character in the story? That’s important—add it to the container."

- **Output Gate**: Controls which part of the container’s contents to pass to the outside world (as the hidden state). Imagine a third worker deciding which accumulated information is most relevant right now: "We need to predict the next word—important to know the subject is singular, but irrelevant to recall which country was mentioned earlier."

**3. Dual State System**

Unlike vanilla RNNs, LSTM has two information pathways:

- **Cell State** $C_t$: The main conveyor, designed for long-term information storage.
- **Hidden State** $h_t$: A filtered version of the cell state, containing only the information the network deems relevant at the current moment.

**Notebook Metaphor:**

Another way to visualize LSTM is as a person with a notebook (cell state) who constantly decides:
- Which old notes to erase (forget gate)
- Which new notes to write (input gate)
- Which information from the notebook to use when answering a question (output gate)

This system allows LSTM to function as a smart information accumulator, selectively remembering important facts and ignoring irrelevant details—exactly what is needed for processing long sequences.

### 2.2 Formalization and Notation: Defining Dimensions and Variables

Let us formalize the LSTM architecture, clearly defining all components and their dimensions. This aids both in understanding the structure and subsequent implementation.

**Main Notations:**

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|-----------------|
| $x_t$ | $\mathbb{R}^{d_x}$ | Input vector at step $t$ |
| $h_t$ | $\mathbb{R}^{d_h}$ | Hidden state at step $t$ |
| $C_t$ | $\mathbb{R}^{d_h}$ | Cell state at step $t$ |
| $f_t$ | $\mathbb{R}^{d_h}$ | Forget gate activation at step $t$ |
| $i_t$ | $\mathbb{R}^{d_h}$ | Input gate activation at step $t$ |
| $o_t$ | $\mathbb{R}^{d_h}$ | Output gate activation at step $t$ |
| $\tilde{C}_t$ | $\mathbb{R}^{d_h}$ | Candidate vector for new cell values |

**Weight Matrices and Bias Vectors:**

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|-----------------|
| $W_f$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for forget gate |
| $W_i$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for input gate |
| $W_C$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for candidate vector |
| $W_o$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for output gate |
| $b_f$ | $\mathbb{R}^{d_h}$ | Bias for forget gate |
| $b_i$ | $\mathbb{R}^{d_h}$ | Bias for input gate |
| $b_C$ | $\mathbb{R}^{d_h}$ | Bias for candidate vector |
| $b_o$ | $\mathbb{R}^{d_h}$ | Bias for output gate |

**Activation Functions:**
- $\sigma$: Sigmoid function, maps inputs to range [0, 1]
- $\tanh$: Hyperbolic tangent, maps inputs to range [-1, 1]

**Input and Output Data Dimensions:**
- $d_x$: Dimension of input vector $x_t$
- $d_h$: Dimension of hidden state and cell state

**Concatenation of Input and Previous State:**

For simplified notation, we often concatenate the input vector $x_t$ and previous hidden state $h_{t-1}$:

$$[x_t, h_{t-1}] \in \mathbb{R}^{d_x + d_h}$$

This allows us to define weights as a single matrix per gate, rather than separate matrices for $x_t$ and $h_{t-1}$.

**Notes on Dimensions:**

1. In standard LSTM architecture, cell state dimension $C_t$ equals hidden state dimension $h_t$. In some variants, they may differ.

2. All gates ($f_t$, $i_t$, $o_t$) have the same dimension $d_h$, enabling element-wise control over the cell state.

3. Total parameters in standard LSTM:
   - Weights: $4 \times d_h \times (d_x + d_h)$
   - Biases: $4 \times d_h$
   - Total: $4 \times d_h \times (d_x + d_h + 1)$

We will use these notations in subsequent sections to describe LSTM’s mathematical formulas and dynamics.

### 2.3 Dynamics of One Step

Now we examine how all LSTM components are computed precisely at a single time step $t$. We will detail each architectural element.

### Forget Gate

The forget gate $f_t$ determines which information from the previous cell state $C_{t-1}$ should be retained and which discarded:

$$
f_t = \sigma\big(W_f \cdot [x_t, h_{t-1}] + b_f\big)
$$

Here:
- $[x_t, h_{t-1}]$ — concatenation of current input and previous hidden state
- $W_f$ — weight matrix for forget gate
- $b_f$ — bias vector
- $\sigma$ — sigmoid function returning values in [0, 1]

Result $f_t$ is a vector of values between 0 and 1, where:
- **1** means "retain this information fully"
- **0** means "discard this information entirely"

#### **Example:**

If the model processes text and at some point recognizes a new sentence has begun, the forget gate may "zero out" part of the information from the previous sentence.

![Image_02.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_02.png)

**Context:** Processing sequence "Я люблю море. На небе светит солнце."

1. **State after first sentence:**
   - $C_{t-1}$ (cell state) encodes information from first sentence: [0.8, 0.6, -0.2]
     - 0.8 → fact "sea" (noun)
     - 0.6 → emotion "love"
     - -0.2 → pronoun "I"

2. **Processing word "На" (start of new sentence):**
   - Input $x_t$ = embedding of "На" [0.1, -0.3, 0.5]
   - $h_{t-1}$ = previous hidden state [0.7, 0.5, -0.1]
   - Forget gate computes:
     $$
     f_t = \sigma\left(
     \begin{bmatrix}
     0.2 & 0.4 & -0.1 \\
     -0.3 & 0.6 & 0.2 \\
     0.1 & -0.2 & 0.3
     \end{bmatrix}
     \cdot
     \begin{bmatrix}
     0.1 \\ -0.3 \\ 0.5 \\ 0.7 \\ 0.5 \\ -0.1
     \end{bmatrix}
     +
     \begin{bmatrix}
     0.1 \\ -0.2 \\ 0.3
     \end{bmatrix}
     \right) = [0.1, 0.9, 0.8]
     $$
     - First neuron (0.1) → forget pronoun information (no longer needed)
     - Second neuron (0.9) → retain emotional context
     - Third neuron (0.8) → retain noun information

3. **Updated cell state:**
   - $C_t = f_t \odot C_{t-1} = [0.1, 0.9, 0.8] \odot [0.8, 0.6, -0.2] = [0.08, 0.54, -0.16]$
     - Value 0.8 ("sea") reduced to 0.08 → forgotten
     - Emotion 0.6 preserved as 0.54
     - Pronoun -0.2 became -0.016 → nearly forgotten

**Interpretation:** The model decided:
- Preserve emotional context (may be useful for sentiment analysis)
- Discard specific nouns from previous sentence
- Prepare for new syntactic structure (new sentence)

**Vector Visualization:**
```
Before forget gate:    [ 0.80  0.60  -0.20 ]
After forget gate:     [ 0.08  0.54  -0.02 ]
                       │      │      └── Nearly forgotten ("I")
                       │      └────────── Preserved ("love")
                       └───────────────── Forgotten ("sea")
```

---

### Input Gate

The input gate $i_t$ determines what new information to add to the cell state:

$$
i_t = \sigma\big(W_i \cdot [x_t, h_{t-1}] + b_i\big)
$$

As with the forget gate, the result is a vector of values in [0, 1], where:
- **1** means "add this new information fully"
- **0** means "do not add this information"

#### Candidate Cell State Vector

![Image_03.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_03.png)

Parallel to the input gate, a candidate vector $\tilde{C}_t$ is created—a "draft" of new information that may potentially be added to the cell state:

$$
\tilde{C}_t = \tanh\big(W_C \cdot [x_t, h_{t-1}] + b_C\big)
$$

Here, the $\tanh$ function normalizes values to the range [-1, 1].

#### Updating the Cell State

![Image_04.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_04.png)

Now we are ready to update the cell state $C_t$, using the forget gate, input gate, and candidate vector:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

where $\odot$ denotes element-wise multiplication (Hadamard product).

This equation describes LSTM’s key mechanism:
1. $f_t \odot C_{t-1}$ — old information we chose to retain
2. $i_t \odot \tilde{C}_t$ — new information we chose to add

**Crucially!** The cell state $C_t$ is updated using only linear operations (multiplication and addition). This ensures gradients can flow through the cell without vanishing, solving the vanilla RNN problem.

### **Example:**

Continue processing the sequence "Я люблю море. На небе светит солнце." after applying the forget gate.

1. **Current state after forget gate:**
   - $f_t \odot C_{t-1} = [0.08, 0.54, -0.16]$ — part of the previous sentence’s information preserved

2. **Processing word "На" (start of new sentence):**
   - Input $x_t$ = embedding of "На" [0.1, -0.3, 0.5]
   - $h_{t-1}$ = previous hidden state [0.7, 0.5, -0.1]
   
   - Input gate computes:
     $$
     i_t = \sigma\left(
     \begin{bmatrix}
     0.3 & -0.2 & 0.1 \\
     0.5 & 0.4 & -0.3 \\
     -0.1 & 0.7 & 0.2
     \end{bmatrix}
     \cdot
     \begin{bmatrix}
     0.1 \\ -0.3 \\ 0.5 \\ 0.7 \\ 0.5 \\ -0.1
     \end{bmatrix}
     +
     \begin{bmatrix}
     -0.1 \\ 0.2 \\ 0.1
     \end{bmatrix}
     \right) = [0.7, 0.6, 0.9]
     $$
     
   - Candidate vector of new state:
     $$
     \tilde{C}_t = \tanh\left(
     \begin{bmatrix}
     0.4 & 0.1 & -0.3 \\
     -0.2 & 0.5 & 0.3 \\
     0.3 & -0.4 & 0.2
     \end{bmatrix}
     \cdot
     \begin{bmatrix}
     0.1 \\ -0.3 \\ 0.5 \\ 0.7 \\ 0.5 \\ -0.1
     \end{bmatrix}
     +
     \begin{bmatrix}
     0.2 \\ -0.1 \\ 0.4
     \end{bmatrix}
     \right) = [0.6, -0.3, 0.7]
     $$
     - First neuron (0.6) → information about place "sky" (noun)
     - Second neuron (-0.3) → neutral emotional tone
     - Third neuron (0.7) → preposition "on" (indicates location)

3. **Updated cell state with new information:**
   - $i_t \odot \tilde{C}_t = [0.7, 0.6, 0.9] \odot [0.6, -0.3, 0.7] = [0.42, -0.18, 0.63]$
   
   - Final cell state:
     $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t = [0.08, 0.54, -0.16] + [0.42, -0.18, 0.63] = [0.50, 0.36, 0.47]$
     - Value 0.08 (forgotten "sea") + 0.42 (new "sky") = 0.50 → new location information
     - Emotion 0.54 (preserved "love") + (-0.18) (neutrality) = 0.36 → reduced emotional tone
     - Value -0.16 (nearly forgotten "I") + 0.63 (new "on") = 0.47 → shift from subject to location

**Interpretation:** The model:
- Added new information about the sky, which becomes the new noun
- Reduced emotional tone, transitioning to a more neutral description
- Shifted focus from the subject ("I") to spatial relation ("on")

**Vector Visualization:**
```
After forget gate:   [ 0.08  0.54  -0.16 ]
New information:     [ 0.42  -0.18  0.63 ]
Final state:         [ 0.50  0.36  0.47 ]
                      │      │      └── New focus (location "on")
                      │      └───────── Reduced emotionality
                      └──────────────── New noun ("sky")
```
---

#### Output Gate

![Image_05.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_05.png)

The output gate $o_t$ determines which portion of the updated cell state to pass to the output hidden state:

$$
o_t = \sigma\big(W_o \cdot [x_t, h_{t-1}] + b_o\big)
$$

Like the other gates, $o_t$ contains values in the range [0, 1].

#### Hidden State

Finally, compute the new hidden state $h_t$ by applying the output gate to the normalized cell state:

$$
h_t = o_t \odot \tanh(C_t)
$$

Here:
- $\tanh(C_t)$ normalizes cell state values to the range [-1, 1]
- $o_t$ determines which components of this normalized state to pass forward

The hidden state $h_t$ is used both for predicting the current step’s output and as input for the next step of the network.

**Summary: Full Set of LSTM Equations**

For convenience, here is the complete set of equations describing one LSTM step:

$$
\begin{align}
f_t &= \sigma(W_f \cdot [x_t, h_{t-1}] + b_f) \\
i_t &= \sigma(W_i \cdot [x_t, h_{t-1}] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [x_t, h_{t-1}] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [x_t, h_{t-1}] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{align}
$$

These six equations fully describe the dynamics of an LSTM cell at a single time step.

## 3. Mathematical Foundations and Functioning of LSTM

### 3.1 Role of Sigmoid Functions: Why Sigmoid for Gates

The sigmoid function plays a pivotal role in the LSTM architecture, especially in the gate mechanism. Let us examine why this specific activation function is used for all three gates (forget, input, and output).

![Image_06.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/LSTM/Image_06.jpg)

**Mathematical definition of the sigmoid function:**

$\sigma(x) = \frac{1}{1 + e^{-x}}$

**Key properties of the sigmoid that make it ideal for gates:**

1. **Bounded output range [0, 1]**
   - This property is critical for gates, as they must function as filters
   - Value 0 means "completely block information"
   - Value 1 means "completely pass information"
   - Intermediate values allow partial passage of information

2. **Smoothness and differentiability**
   - The sigmoid is continuous and differentiable everywhere
   - Its derivative has a simple form: $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$
   - This is essential for backpropagation during training

3. **Nonlinearity and saturation**
   - For large positive $x$, the function approaches 1
   - For large negative $x$, it approaches 0
   - This creates a "saturation" effect that stabilizes network dynamics

**Practical application in LSTM gates:**

- **Forget gate ($f_t$)**: Sigmoid determines what percentage of each cell state element to retain. Value 0 means "forget completely," 1 means "retain completely."

- **Input gate ($i_t$)**: Sigmoid controls how much of the new information ($\tilde{C}_t$) to add to the cell state. Value 0 means "add nothing," 1 means "add fully."

- **Output gate ($o_t$)**: Sigmoid regulates how much information from the cell state to pass to the hidden state $h_t$. Value 0 means "pass nothing," 1 means "pass everything."

**Note on bias initialization:**

It is important to note that gate biases are often initialized specially:
- The forget gate bias ($b_f$) is frequently initialized to positive values (e.g., 1 or 2), so that early in training the network tends to "remember" information
- Biases of other gates are typically initialized to zero or small random values

Such initialization helps LSTM learn long-term dependencies more quickly.

### 3.2 Role of the tanh Activation Function: In Candidate Vector and Output

The hyperbolic tangent (tanh) is the second key activation function in the LSTM architecture. It is used in two critical places: generating the candidate vector and forming the output hidden state.

![Image_07.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_%26_18/assets/LSTM/Image_07.JPG)

**Mathematical definition of tanh:**

$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Key properties of tanh important for LSTM:**

1. **Bounded output range [-1, 1]**
   - Unlike the sigmoid, tanh is symmetric about the origin
   - The [-1, 1] range allows representation of both positive and negative activations with equal amplitude

2. **Steep gradient**
   - The derivative of tanh at zero is 1, which is higher than that of sigmoid (0.25)
   - This provides stronger gradients during backpropagation

3. **Zero mean output**
   - Tanh outputs have approximately zero mean
   - This helps mitigate the problem of covariate shift during training

**Role of tanh in the candidate vector $\tilde{C}_t$:**

$\tilde{C}_t = \tanh(W_C \cdot [x_t, h_{t-1}] + b_C)$

1. **Value normalization**
   - Tanh scales all values to [-1, 1], creating stable dynamics in the cell
   - This prevents uncontrolled growth of cell state values

2. **Bipolar representation**
   - Negative values can represent "inhibitory" signals
   - Positive values can represent "excitatory" signals
   - This enables rich internal data representation

3. **Balance with input gate sigmoid**
   - Tanh generates candidate values in [-1, 1]
   - The input gate sigmoid determines how much of these values to add
   - This allows both addition and subtraction of information from the cell state

**Role of tanh in forming the hidden state $h_t$:**

$h_t = o_t \odot \tanh(C_t)$

1. **Output normalization**
   - Cell state $C_t$ may contain large-amplitude values
   - Tanh normalizes these before output, ensuring stability for subsequent layers

2. **Activation balancing**
   - Tanh provides symmetric output for positive and negative values in $C_t$
   - This benefits downstream layers that often perform better with centered inputs

3. **Interpretability of output**
   - Hidden state $h_t$ is used for predictions and as input to the next step
   - The normalized [-1, 1] range ensures consistent scaling of these signals

**Comparison with Other Activation Functions:**

Why tanh, and not other functions such as ReLU?

- **ReLU** does not bound upper output limits, potentially leading to uncontrolled activation growth
- **Leaky ReLU** has an unbounded range and asymmetric response, less suitable for cell state
- **Sigmoid** restricts outputs to positive values only, reducing model expressiveness

**Practical Aspect:**

In some modern LSTM variants, tanh may be replaced with other activations for the candidate vector, but in the classical architecture and most practical implementations, tanh remains the standard choice due to its mathematical properties aligning well with the nature of sequential data processing.

### 3.3 Gradient Flow in LSTM: How the Architecture Solves the Vanishing Gradient Problem

The key advantage of LSTM over vanilla RNNs is its ability to effectively handle long sequences without the vanishing gradient problem. Let us examine precisely how LSTM resolves this fundamental issue at the level of gradient flow.

**Recall the problem in vanilla RNNs:**

In a vanilla RNN, the gradient of the loss with respect to hidden state $h_{t-k}$ involves a product of many Jacobians:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t-k+1}^{t} \text{diag}(\tanh'(a_i)) \cdot W_{hh}$$

These Jacobians typically have eigenvalues less than 1, causing exponential gradient decay as $k$ increases.

**Key LSTM Innovation: Constant Error Carousel**

LSTM’s defining feature is the **Constant Error Carousel (CEC)**, enabled by the direct linear connection through the cell state $C_t$.

Consider the gradient flow through the cell state from time $t$ to $t-1$:

$$\frac{\partial C_t}{\partial C_{t-1}} = \frac{\partial (f_t \odot C_{t-1} + i_t \odot \tilde{C}_t)}{\partial C_{t-1}} = f_t$$

This expression reveals LSTM’s critical property: **the gradient from $C_t$ to $C_{t-1}$ flows via simple element-wise multiplication by the forget gate $f_t$**. No nonlinear activation functions or matrix multiplications are involved—only direct element-wise multiplication.

**Recurrent gradient propagation across multiple steps:**

When propagating the gradient backward over $k$ steps, we have:

$$\frac{\partial C_t}{\partial C_{t-k}} = \prod_{i=t-k+1}^{t} \frac{\partial C_i}{\partial C_{i-1}} = \prod_{i=t-k+1}^{t} f_i$$

This is a (element-wise) product of forget gate vectors.

**How this solves the vanishing gradient problem:**

1. **Controlled gradient flow via $f_t$**
   - If components of $f_t$ are close to 1, the gradient flows with minimal decay
   - LSTM learns to set $f_t \approx 1$ for important information

2. **Additive cell state update**
   - Unlike multiplicative recurrent connections in vanilla RNNs, LSTM uses additive updates:
     $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
   - This allows gradients to flow without mandatory multiplication by recurrent weights

3. **Learnable "forgetting" mechanism**
   - Instead of fixed decay, LSTM learns what to remember and what to discard
   - This acts as a selective "gate" that passes important gradients and blocks irrelevant ones

**Mathematical Modeling of Gradient Flow:**

The full gradient of loss $L$ with respect to cell state $C_{t-k}$ can be decomposed:

$$\frac{\partial L}{\partial C_{t-k}} = \sum_{j=t-k+1}^{T} \frac{\partial L}{\partial C_j} \frac{\partial C_j}{\partial C_{t-k}}$$

Here, the first term $\frac{\partial L}{\partial C_j}$ is the backpropagation from loss to cell state at time $j$, and the second term $\frac{\partial C_j}{\partial C_{t-k}}$ is the product of forget gates along the path from $t-k$ to $j$.

**Practical Consequences:**

1. **Long-range dependencies**
   - LSTM can learn dependencies over hundreds or even thousands of steps, impossible for vanilla RNNs
   - For example, LSTM can link "Франция" at the start of text to "французский" at the end, even with a large gap

2. **Selective sensitivity**
   - LSTM learns to be sensitive only to important long-term dependencies
   - This is more efficient than attempting to memorize everything

3. **Training stability**
   - Controlled gradient flow makes LSTM training more stable
   - Less frequent need for gradient clipping or specialized weight initialization

Thus, LSTM’s unique architecture—with its constant error carousel through the cell state and learnable gates—effectively resolves the vanishing gradient problem, enabling modeling of long-term dependencies in sequential data.

### 3.4 Comparison with Vanilla RNN: A Mathematical Perspective on Advantages

Let us conduct a rigorous mathematical comparison between LSTM and vanilla RNN to better understand why LSTM handles sequential processing significantly better.

**1. Architectural Comparison of Core Equations**

**Vanilla RNN:**
$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$

**LSTM:**
$
\begin{align}
f_t &= \sigma(W_f \cdot [x_t, h_{t-1}] + b_f) \\
i_t &= \sigma(W_i \cdot [x_t, h_{t-1}] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [x_t, h_{t-1}] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [x_t, h_{t-1}] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{align}
$

**Key Differences:**

- **One vs. multiple equations**: RNN uses a single equation, whereas LSTM decomposes the update into several specialized components.
- **One vs. two states**: RNN has only a hidden state $h_t$, whereas LSTM separates information between the hidden state $h_t$ and the cell state $C_t$.
- **Simple vs. adaptive update**: RNN always updates the entire hidden state at once, while LSTM selectively updates components of the cell state.

**2. Information Flow Over Time**

**Vanilla RNN:**

Information from input $x_{t-k}$ to the current hidden state $h_t$ passes through a chain of nonlinear transformations:

$h_t = F(h_{t-1}, x_t) = F(F(h_{t-2}, x_{t-1}), x_t) = ... = F(F(...F(h_{t-k-1}, x_{t-k})...), x_t)$

Here $F(h, x) = \tanh(W_{xh}x + W_{hh}h + b_h)$. Each application of the nonlinearity $\tanh$ may lead to information loss.

**LSTM:**

Information can flow through the cell state $C_t$ with controlled forgetting:

$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t = f_t \odot (f_{t-1} \odot C_{t-2} + i_{t-1} \odot \tilde{C}_{t-1}) + i_t \odot \tilde{C}_t = ...$

Expanding this expression further yields:

$C_t = \left( \prod_{j=t-k+1}^{t} f_j \right) \odot C_{t-k} + \sum_{j=t-k+1}^{t} \left( i_j \odot \tilde{C}_j \odot \prod_{l=j+1}^{t} f_l \right)$

This shows that the cell state $C_t$ is a weighted sum of all previous inputs, where weights are determined by products of forget gates $f_j$. If all $f_j \approx 1$, information flows with minimal distortion.

**3. Mathematical Analysis of Gradient Flow**

**Vanilla RNN:**

The gradient of loss $L$ with respect to weights $W_{hh}$ is computed via the chain rule:

$\frac{\partial L}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial h_{t-k}} \frac{\partial h_{t-k}}{\partial W_{hh}}$

Where the second derivative contains a product of Jacobians:

$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{j=t-k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=t-k+1}^{t} \text{diag}(\tanh'(W_{xh}x_j + W_{hh}h_{j-1} + b_h)) \cdot W_{hh}$

The eigenvalues of this product are typically less than 1, leading to gradient vanishing.

**LSTM:**

For LSTM, the gradient from $C_t$ to $C_{t-k}$ is computed as:

$\frac{\partial C_t}{\partial C_{t-k}} = \prod_{j=t-k+1}^{t} \frac{\partial C_j}{\partial C_{j-1}} = \prod_{j=t-k+1}^{t} f_j$

Since $f_j$ is the output of a sigmoid function trained specifically to control information flow, LSTM can maintain values $f_j \approx 1$ for important components, preventing gradient vanishing.

**4. Quantitative Comparison of Parameters and Computational Complexity**

**Vanilla RNN:**
- Number of parameters: $d_h \times (d_x + d_h + 1)$
- Computational complexity per step: $O(d_h \times (d_x + d_h))$

**LSTM:**
- Number of parameters: $4 \times d_h \times (d_x + d_h + 1)$
- Computational complexity per step: $O(4 \times d_h \times (d_x + d_h))$

LSTM requires approximately 4 times more parameters and computations, but this is compensated by significant performance improvements on tasks with long-term dependencies.

**5. Empirical Comparison of Capabilities**

| **Property** | **Vanilla RNN** | **LSTM** |
|--------------|-----------------|----------|
| Maximum dependency length | 5–10 steps | Hundreds or thousands of steps |
| Noise robustness | Low | High |
| Ability to forget irrelevant information | Limited | High |
| Adaptability to different time scales | Low | High |

**6. Geometric Interpretation**

If we represent the space of hidden states as a multidimensional space:

- **Vanilla RNN** creates complex nonlinear dynamics where trajectories rapidly converge to attractors, leading to loss of information about past states.

- **LSTM** creates controlled dynamics where important directions in the state space can be preserved nearly unchanged, allowing information to flow over long distances, while unimportant directions decay rapidly.

Overall, LSTM is a deeply thought-out extension of the RNN architecture that deliberately eliminates key mathematical limitations of vanilla RNNs—particularly the vanishing gradient problem—making LSTM a significantly more powerful tool for modeling sequences with long-term dependencies.

</details>

<details> 
    <summary><em><strong> 🔥🔥🔥 Gated Recurrent Unit (GRU)</strong></em></summary>

## 1. Introduction and Motivation

### 1.1 Context of GRU's Emergence: History as an LSTM Simplification

Gated Recurrent Unit (GRU) is a variant of recurrent neural network introduced to the world in 2014. GRU emerged during active research in neural machine translation and natural language processing, when LSTM (Long Short-Term Memory) had already proven themselves as effective models for sequential data.

**Chronology of GRU's emergence:**

- **2013–2014**: As part of neural machine translation architecture development, researchers from Montreal began experimenting with variations of recurrent networks.
  
- **June 2014**: KyungHyun Cho and colleagues published the paper ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078), introducing the GRU architecture for the first time.

- **September 2014**: In the paper ["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"](https://arxiv.org/abs/1412.3555), Junyoung Chung and others conducted the first systematic comparison of LSTM and GRU.

GRU arose from the need for a simpler yet equally effective alternative to LSTM. Researchers sought an architecture that could:

- Handle long-term dependencies in sequences
- Be computationally more efficient and easier to train
- Require fewer parameters while preserving expressive power

The key insight was that not all components of the complex LSTM architecture were necessary to achieve strong performance. This led to the creation of GRU, which merges the functionality of several LSTM gates and simplifies the overall state update mechanism.

### 1.2 Authors and Key Publications: Cho and Colleagues' Work, Connection to Machine Translation

GRU is inextricably linked to the name of **Kyunghyun Cho** and his colleagues from the University of Montreal. The research group developing GRU included prominent deep learning researchers such as Yoshua Bengio and Dzmitry Bahdanau.

**Foundational Publications:**

1. **"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (Cho et al., 2014)**
   - First presentation of GRU in the context of an encoder-decoder architecture for machine translation
   - Demonstration of the ability to learn phrase representations at the sentence level
   - Comparison with baseline RNN and demonstration of advantages of gating mechanisms

2. **"Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (Chung et al., 2014)**
   - Systematic comparison of LSTM and GRU on polyphonic music modeling and speech signal processing tasks
   - Conclusion: GRU achieves performance comparable to LSTM with lower computational complexity

3. **"Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau, Cho, and Bengio, 2014)**
   - Application of GRU in an attention-based architecture for machine translation
   - Demonstration of GRU's capabilities in combination with attention mechanisms

**Connection to Machine Translation:**

The emergence of GRU is closely tied to the advancement of neural machine translation (NMT). In 2014, researchers actively sought alternatives to statistical machine translation methods, and recurrent neural networks showed promising results.

GRU was specifically developed within the context of an encoder-decoder model for NMT, where:
- The **encoder** encodes the input sentence into a fixed vector
- The **decoder** generates the translation based on this vector

Cho and colleagues' work showed that GRU could effectively encode semantic and syntactic information of sentences, which is critical for high-quality translation. Moreover, GRU's simplified structure enabled training of deeper models and scaling to larger datasets, which was essential for practical machine translation applications.

Quote from KyungHyun Cho on GRU's creation:
> "We aimed to create a simpler alternative to LSTM that could be easier to train and more computationally efficient, while retaining the ability to model long-term dependencies in sequential data."

### 1.3 Balance of Complexity and Efficiency: Why There Was a Need for More Compact Models

In the early 2010s, LSTM became the de facto standard for sequential processing tasks, but they had several limitations that stimulated the search for more compact alternatives:

**Problems Associated with LSTM Complexity:**

1. **Computational Requirements**
   - LSTM has four sets of weights and biases (for forget, input, output gates, and candidate vector)
   - Training large LSTM models required substantial computational resources
   - Processing time per sequence element was critical for real-time applications

2. **Training Complexity**
   - More parameters meant a larger search space during optimization
   - LSTM often required meticulous hyperparameter tuning
   - Convergence was harder to achieve on limited datasets

3. **Memory and Power Consumption**
   - LSTM models consumed more memory during deployment
   - This limited their use on mobile and embedded devices
   - High computational energy consumption

4. **Theoretical Understanding**
   - The complex LSTM architecture made behavior analysis difficult
   - It was not always clear which components contributed most to effectiveness

**Why More Compact Models Were Needed:**

1. **Growth of Mobile and Embedded Systems**
   - Rising demand for models capable of operating on resource-constrained devices
   - Need to balance accuracy with energy efficiency

2. **Scaling to Large Datasets**
   - Increasing availability of data required more efficient models
   - Simpler models could process more data with the same resources

3. **Experiments in Architecture Simplification**
   - Researchers began systematically studying which LSTM components were truly necessary
   - Experiments showed that some LSTM elements could be merged without performance loss

4. **Pursuit of Design Elegance**
   - Occam's Razor: if two models show equal performance, the simpler one is preferable
   - Simpler models often generalize better to new data

**How GRU Solves These Problems:**

1. **Fewer Parameters**
   - GRU has only two gates instead of three in LSTM
   - Merges input and forget gate functionalities into a single update gate
   - No separate cell state (uses only the hidden state)

2. **Preservation of Key Functionality**
   - Despite simplifications, GRU retains the ability to model long-term dependencies
   - The gating mechanism still allows control over information flow through the network

3. **Empirical Results**
   - On many tasks, GRU showed results comparable to LSTM
   - In some cases, it even outperformed LSTM, especially on limited datasets

GRU represents an elegant balance between complexity and efficiency, making recurrent models more accessible and applicable across a wide range of tasks.

## 2. GRU Architecture: Key Components

### 2.1 Intuition: Metaphor of "Economical Memory" with Two Control Points

To understand how GRU works, imagine it as a system of "economical memory" with two primary control points. This metaphor helps intuitively grasp how GRU manages information.

**Imagine a librarian working with one large book (hidden state):**

Unlike LSTM, which has a separate long-term storage (cell state) and a notebook for current notes (hidden state), the GRU librarian works with only one book. In this book, they continuously update information following two simple rules:

**1. Reset Gate — "What Should Be Re-read?"**

Imagine the librarian deciding which parts of their current knowledge (stored in the book) are relevant for understanding new information:

- When the reset gate is close to 1: "This part of my current knowledge is important for understanding the new information"
- When the reset gate is close to 0: "This part of my knowledge is irrelevant to the new information; I temporarily ignore it"

For example, if you read the sentence "The weather in Paris..." and the text continues with "...the capital of France," the reset gate might decide that knowledge about "Paris" remains relevant, while information about "weather" is no longer needed to understand the continuation.

**2. Update Gate — "How Strongly Should the Book Be Updated?"**

After determining which parts of current knowledge are relevant, the librarian must decide to what extent to update the book's content:

- When the update gate is close to 1: "I will fully replace this part of the book with new information"
- When the update gate is close to 0: "I will preserve the existing information unchanged"

In the text example: if the book previously contained information about Rome, and the new sentence discusses Paris, the update gate might set a high value to replace information about Rome with information about Paris.

**GRU Process in Metaphor:**

1. **Receiving new information**: The librarian receives a new information chunk (input vector $x_t$)

2. **Determining relevance of old knowledge**: The librarian uses the reset gate to determine which parts of the existing book (hidden state $h_{t-1}$) are relevant for processing new information

3. **Creating a draft update**: Based on relevant parts of old knowledge and new information, the librarian creates a draft update (candidate vector $\tilde{h}_t$)

4. **Deciding update intensity**: Using the update gate, the librarian decides how strongly to update each part of the book

5. **Updating the book**: The book is updated according to the update gate decisions, creating a new book state ($h_t$)

**Key Difference from LSTM:**

LSTM can be compared to a more complex system with:
- A long-term storage warehouse (cell state)
- A workbench (hidden state)
- Three workers making decisions (three gates)

GRU simplifies this system by merging the "warehouse" and "workbench" into one storage and reducing the number of "workers" to two, making the system more economical while preserving its core functionality.

This economy—GRU’s main advantage—enables achieving comparable results to LSTM with lower computational costs.

### 2.2 Formalization and Notation: Defining Variables and Dimensions

Let us formalize the GRU architecture by defining all its components and their dimensions. This aids both in understanding the structure and in subsequent implementation.

**Main Notations:**

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|-----------------|
| $x_t$ | $\mathbb{R}^{d_x}$ | Input vector at step $t$ |
| $h_t$ | $\mathbb{R}^{d_h}$ | Hidden state at step $t$ |
| $z_t$ | $\mathbb{R}^{d_h}$ | Update gate activation at step $t$ |
| $r_t$ | $\mathbb{R}^{d_h}$ | Reset gate activation at step $t$ |
| $\tilde{h}_t$ | $\mathbb{R}^{d_h}$ | Candidate vector for new hidden state |

**Weight Matrices and Bias Vectors:**

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|-----------------|
| $W_z$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for update gate |
| $W_r$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for reset gate |
| $W_h$ | $\mathbb{R}^{d_h \times (d_x + d_h)}$ | Weights for candidate vector |
| $b_z$ | $\mathbb{R}^{d_h}$ | Bias for update gate |
| $b_r$ | $\mathbb{R}^{d_h}$ | Bias for reset gate |
| $b_h$ | $\mathbb{R}^{d_h}$ | Bias for candidate vector |

In an alternative formulation, weights for input and hidden state can be separated:

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|-----------------|
| $W_{xz}$ | $\mathbb{R}^{d_h \times d_x}$ | Weights for input in update gate |
| $W_{hz}$ | $\mathbb{R}^{d_h \times d_h}$ | Weights for hidden state in update gate |
| $W_{xr}$ | $\mathbb{R}^{d_h \times d_x}$ | Weights for input in reset gate |
| $W_{hr}$ | $\mathbb{R}^{d_h \times d_h}$ | Weights for hidden state in reset gate |
| $W_{xh}$ | $\mathbb{R}^{d_h \times d_x}$ | Weights for input in candidate vector |
| $W_{hh}$ | $\mathbb{R}^{d_h \times d_h}$ | Weights for hidden state in candidate vector |

Both formulations are equivalent, but one may be more convenient depending on context.

**Activation Functions:**

- $\sigma$: Sigmoid function, maps inputs to range [0, 1]
- $\tanh$: Hyperbolic tangent, maps inputs to range [-1, 1]

**Input and Output Data Dimensions:**

- $d_x$: Dimension of input vector $x_t$
- $d_h$: Dimension of hidden state

**Concatenation of Input and Previous State:**

For simplified notation, we often concatenate the input vector $x_t$ and previous hidden state $h_{t-1}$:

$$[x_t, h_{t-1}] \in \mathbb{R}^{d_x + d_h}$$

**Notes on Dimensions:**

1. GRU uses only the hidden state $h_t$, unlike LSTM, which has an additional cell state $C_t$.

2. Both gates ($z_t$, $r_t$) have the same dimension $d_h$, enabling element-wise control over hidden state updates.

3. Total parameters in standard GRU:
   - Weights: $3 \times d_h \times (d_x + d_h)$
   - Biases: $3 \times d_h$
   - Total: $3 \times d_h \times (d_x + d_h + 1)$

Note that GRU has only 3/4 the number of parameters of LSTM, which is one of this architecture’s key advantages.

### 2.3 Dynamics of One Step

Now we examine how all GRU components are computed precisely at a single time step $t$. We will detail each architectural element.

![Image_01.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/GRU/Image_01.jpg)

#### Reset Gate

The reset gate $r_t$ determines which elements of the previous state $h_{t-1}$ should be considered when computing the new candidate vector:

$$
r_t = \sigma\big(W_r \cdot [x_t, h_{t-1}] + b_r\big)
$$

or, in expanded form:

$$
r_t = \sigma\big(W_{xr} \cdot x_t + W_{hr} \cdot h_{t-1} + b_r\big)
$$

Here:
- $[x_t, h_{t-1}]$ — concatenation of current input and previous hidden state
- $W_r$ (or $W_{xr}$ and $W_{hr}$) — weight matrices for reset gate
- $b_r$ — bias vector
- $\sigma$ — sigmoid function returning values in [0, 1]

Result $r_t$ is a vector of values between 0 and 1, where:
- **1** means "fully consider this information from the previous state"
- **0** means "fully ignore this information from the previous state"

**Example:** When processing text, if a sentence changes topic, the reset gate may set low values to "forget" the context of the previous topic when forming the new candidate vector.

#### Update Gate

![Image_02.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/GRU/Image_02.jpg)

The update gate $z_t$ determines how strongly each element of the hidden state should be updated:

$$
z_t = \sigma\big(W_z \cdot [x_t, h_{t-1}] + b_z\big)
$$

or:

$$
z_t = \sigma\big(W_{xz} \cdot x_t + W_{hz} \cdot h_{t-1} + b_z\big)
$$

As with the reset gate, the result is a vector of values in [0, 1], where:
- **1** means "fully replace old information with new"
- **0** means "fully preserve old information unchanged"

**Important Observation:** The update gate in GRU plays a role analogous to the combined forget and input gates in LSTM. A high value of $z_t$ means "forget" old information and "remember" new information.

#### Candidate Hidden State Vector

The candidate vector $\tilde{h}_t$ represents a "proposal" for the new hidden state, but considering only the part of the previous state determined by the reset gate:

$$
\tilde{h}_t = \tanh\big(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h\big)
$$

or:

$$
\tilde{h}_t = \tanh\big(W_{xh} \cdot x_t + W_{hh} \cdot (r_t \odot h_{t-1}) + b_h\big)
$$

Here:
- $r_t \odot h_{t-1}$ — element-wise multiplication of reset gate with previous hidden state
- $\tanh$ — hyperbolic tangent normalizing values to range [-1, 1]

**Key Point:** The reset gate is applied to $h_{t-1}$ before its use in computing the candidate vector, not to the full hidden state. This allows the network to "forget" part of the past state when computing new information, without necessarily altering the full state update.

#### Final Hidden State Update

Finally, compute the new hidden state $h_t$ as a weighted combination of the previous state and the candidate vector, where the weight is determined by the update gate:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

Here:
- $(1 - z_t) \odot h_{t-1}$ — part of old information we chose to retain
- $z_t \odot \tilde{h}_t$ — part of new information we chose to add

**Important Observation:** This formula is equivalent to interpolating between the old state $h_{t-1}$ and the new candidate state $\tilde{h}_t$, where $z_t$ determines the interpolation point for each vector element.

**Summary: Full Set of GRU Equations**

For convenience, here is the complete set of equations describing one GRU step:

$$
\begin{align}
z_t &= \sigma(W_z \cdot [x_t, h_{t-1}] + b_z) \\
r_t &= \sigma(W_r \cdot [x_t, h_{t-1}] + b_r) \\
\tilde{h}_t &= \tanh(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{align}
$$

These four equations fully describe the dynamics of a GRU cell at a single time step.

## 3. Mathematical Comparison of GRU with LSTM

### 3.1 Architectural Simplifications: Which LSTM Components Were Merged or Removed

GRU can be viewed as a simplified version of LSTM, where certain components were merged or entirely removed. Let us systematically examine these simplifications.

**1. Merging States: Elimination of Separate Cell State**

LSTM has two types of states:
- Cell state ($C_t$) — long-term memory
- Hidden state ($h_t$) — short-term memory and output

In GRU, these two states are merged into one — the hidden state $h_t$, which performs both functions.

Mathematical consequence:
- LSTM: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, then $h_t = o_t \odot \tanh(C_t)$
- GRU: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

**2. Merging Input and Forget Gates**

In LSTM, the forget gate $f_t$ determines how much old information to retain, and the input gate $i_t$ determines how much new information to add. These decisions are made independently.

In GRU, the update gate $z_t$ simultaneously determines how much old information to replace with new. This creates a rigid coupling: if you add X% new information, you must forget X% old information.

Mathematical consequence:
- LSTM: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ (where $f_t$ and $i_t$ are independent)
- GRU: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ (where $z_t$ and $(1-z_t)$ are complementary)

**3. Modification of Reset Gate Application**

In LSTM, the output gate $o_t$ is applied after computing all other components to determine which information from the cell state to pass to the output hidden state.

In GRU, the reset gate $r_t$ is applied before computing the candidate vector, determining which part of the previous state to consider when creating the new state.

Mathematical consequence:
- LSTM: $h_t = o_t \odot \tanh(C_t)$ (output gate controls final output)
- GRU: $\tilde{h}_t = \tanh(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h)$ (reset gate influences candidate vector creation)

**4. Elimination of Output Gate**

LSTM has a separate output gate $o_t$ controlling which information from the cell memory should be visible in the external hidden state.

GRU has no such gate — the entire hidden state is always fully accessible.

**5. Comparative Table of Components**

| **LSTM Component** | **GRU Equivalent** | **Notes** |
|--------------------|------------------|-----------|
| Cell state $C_t$ | Absent (merged into $h_t$) | GRU has single state |
| Hidden state $h_t$ | Hidden state $h_t$ | Analogous in both architectures |
| Forget gate $f_t$ | Part of update gate $(1-z_t)$ | In GRU complementary to input gate |
| Input gate $i_t$ | Update gate $z_t$ | In GRU complementary to forget gate |
| Output gate $o_t$ | Absent | GRU has no output filtering |
| Candidate vector $\tilde{C}_t$ | Candidate vector $\tilde{h}_t$ | Analogous, but in GRU influenced by reset gate |
| — | Reset gate $r_t$ | Unique to GRU |

**6. Equation Comparison**

**LSTM**:
$$
f_t = \sigma(W_f \cdot [x_t, h_{t-1}] + b_f)
$$
$$
i_t = \sigma(W_i \cdot [x_t, h_{t-1}] + b_i)
$$
$$
o_t = \sigma(W_o \cdot [x_t, h_{t-1}] + b_o)
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [x_t, h_{t-1}] + b_C)
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

**GRU**:
$$
z_t = \sigma(W_z \cdot [x_t, h_{t-1}] + b_z)
$$
$$
r_t = \sigma(W_r \cdot [x_t, h_{t-1}] + b_r)
$$
$$
\tilde{h}_t = \tanh(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h)
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**Key Observation**: GRU has fewer equations and parameters, providing a more compact architecture that nonetheless preserves the core functionality of controlling information flow.

### 3.2 Gradient Flow in GRU: Analysis of Solving the Vanishing Gradient Problem

GRU, like LSTM, successfully addresses the vanishing gradient problem, but does so slightly differently. Let us analyze precisely how gradient flow is organized in GRU.

**Recall the problem in vanilla RNNs:**

In vanilla RNNs, gradients vanish due to repeated multiplication by Jacobians with eigenvalues less than 1:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t-k+1}^{t} \text{diag}(\tanh'(a_i)) \cdot W_{hh}$$

**Analysis of Gradient Flow in GRU:**

In GRU, the hidden state is updated by:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Compute the gradient $\frac{\partial h_t}{\partial h_{t-1}}$ using the chain rule:

$$\frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial}{\partial h_{t-1}} [(1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t]$$

Expanding this expression:

$$\frac{\partial h_t}{\partial h_{t-1}} = \underbrace{(1 - z_t)}_{\text{direct path}} + \underbrace{\frac{\partial z_t}{\partial h_{t-1}} \odot (\tilde{h}_t - h_{t-1})}_{\text{via } z_t} + \underbrace{z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}}_{\text{via } \tilde{h}_t}$$

**Key Observation 1: Direct Gradient Path**

The first term $(1 - z_t)$ represents the direct gradient path. If $z_t$ is close to 0 (i.e., the decision to retain most old information), the gradient can flow almost unchanged through this path.

Comparison with LSTM:
- In LSTM, the direct path goes through the cell state: $\frac{\partial C_t}{\partial C_{t-1}} = f_t$
- In GRU, the direct path goes through the hidden state: $(1 - z_t)$

**Key Observation 2: Adaptive Update**

GRU does not merely prevent gradient vanishing—it does so adaptively. The network learns to set $z_t$ close to 0 for elements where long-term dependencies are important.

**Gradient Flow Across Multiple Steps:**

For $k$ steps back, the gradient can be expressed as:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

If for each step $i$, the values $(1 - z_i)$ are close to 1 for certain elements, the gradient can flow through these elements with minimal vanishing. This creates "information highways" for backpropagation.

**Reset Gate and Gradients:**

The reset gate $r_t$ influences gradients through the second path:

$$\frac{\partial \tilde{h}_t}{\partial h_{t-1}} = \frac{\partial}{\partial h_{t-1}} \tanh(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h)$$

This includes:
- Direct influence: $r_t \odot \frac{\partial}{\partial h_{t-1}} \tanh(...)$
- Indirect influence via $\frac{\partial r_t}{\partial h_{t-1}}$

The reset gate allows GRU to "forget" certain components of the previous state when computing the candidate vector, but this does not impede gradient flow through the main path $(1 - z_t)$.

**Comparison with LSTM:**

| **Aspect** | **LSTM** | **GRU** |
|------------|----------|---------|
| Primary gradient path | Through cell state: $\frac{\partial C_t}{\partial C_{t-1}} = f_t$ | Through hidden state: $(1 - z_t)$ |
| Flow control | Three independent gates | Two gates, one with a complementary effect |
| Number of gradient paths | Multiple paths through $C_t$ and $h_t$ | Fewer paths, but with a direct mainline |

Thus, GRU addresses the vanishing gradient problem through an adaptive mechanism that creates direct paths for gradient flow, similar to LSTM, but using fewer components.

**3. Memory Consumption**

**LSTM**:
- State storage: $O(2d_h)$ for $h_t$ and $C_t$
- Intermediate results storage for backpropagation: $O(4d_h T)$ for a sequence of length $T$

**GRU**:
- State storage: $O(d_h)$ for $h_t$ only
- Intermediate results storage: $O(3d_h T)$

GRU achieves significant memory savings compared to LSTM for long sequences.

**4. Parallelization and Hardware Acceleration**

**LSTM**:
- Four matrix multiplications can be parallelized or fused into a single large multiplication
- More complex dependencies between components may reduce pipelining efficiency

**GRU**:
- Three matrix multiplications are also amenable to parallelization
- Fewer dependencies between components potentially better for pipelining
- Computation of $r_t \odot h_{t-1}$ introduces an additional dependency

**Practical effect**:
On modern GPUs and specialized hardware accelerators (TPUs, NPUs), GRU’s speed advantage over LSTM often amounts to 20–30% for the same hidden state size.

**5. Scaling to Large Models**

As the hidden state dimension $d_h$ increases, the difference in computational efficiency becomes more pronounced:

- For $d_h = 1024$, the parameter difference: ~1.7 million
- For $d_h = 2048$, the parameter difference exceeds 6 million

This is especially important for deep models with multiple layers, where the savings multiply by the number of layers.

**Table of Efficiency Comparison for Various Dimensions**

| **Dimension** | **LSTM Parameters** | **GRU Parameters** | **Savings** | **Savings (%)** |
|---------------|---------------------|--------------------|-------------|-----------------|
| $d_h = 128$ | 0.22M | 0.16M | 0.06M | 25% |
| $d_h = 256$ | 0.57M | 0.43M | 0.14M | 25% |
| $d_h = 512$ | 1.67M | 1.25M | 0.42M | 25% |
| $d_h = 1024$ | 6.03M | 4.52M | 1.51M | 25% |
| $d_h = 2048$ | 23.14M | 17.35M | 5.79M | 25% |

This computational efficiency makes GRU an especially attractive choice for resource-constrained tasks and models requiring real-time operation.

## Conclusion

Gated Recurrent Unit (GRU) represents an important advancement in recurrent neural network architectures, successfully balancing efficiency and computational complexity. Developed in 2014 as a simplified alternative to LSTM, GRU retains the key ability to model long-term dependencies in sequences while significantly reducing the number of parameters and computational costs.

The main advantages of GRU are:
1. Simplified architecture with two gates instead of three in LSTM
2. Merging of hidden state and memory cell into a single state
3. Comparable performance to LSTM with fewer computational resources
4. Better scalability for large models and long sequences

Mathematical analysis shows that GRU effectively addresses the vanishing gradient problem through an adaptive information flow mechanism. Furthermore, the GRU architecture demonstrates substantial savings in memory and computational resources compared to LSTM, particularly as the hidden state dimension increases.

Thus, GRU represents an optimal choice for sequence processing tasks requiring a balance between efficiency and computational performance, especially under resource constraints or real-time operation requirements.

</details>

<details> 
    <summary><em><strong> 🔥🔥🔥🔥 State Space Models (SSM)</strong></em></summary>

## 1. Introduction to State Space Models (SSM)

### 1.1 Context of SSM Emergence: Evolution of Architectures for Sequence Modeling

State Space Models (SSM) are a class of dynamical systems that entered deep learning from control theory and signal processing. Their emergence as neural network architectures can be viewed as part of a broader evolution in sequence modeling.

**Chronology of sequence model evolution:**

- **1980s–1990s**: Emergence of classical recurrent neural networks (RNNs) for processing sequential data.
  
- **1997**: Introduction of LSTM (Hochreiter & Schmidhuber) as a solution to the vanishing gradient problem.
  
- **2014**: Appearance of GRU (Cho et al.) as a simplified version of LSTM.
  
- **2017**: Transformer architecture (Vaswani et al.) with attention mechanisms revolutionized NLP.
  
- **2019–2020**: First experiments adapting classical SSM for deep learning (Gu et al.).
  
- **2021–2022**: Emergence of the first efficient SSM implementations — S4 (Structured State Space Sequence Model) and S4D (Diagonal State Space).
  
- **2023**: Introduction of the selective SSM architecture Mamba (Gu & Dao).

SSMs arose as an attempt to combine the advantages of recurrent networks (linear complexity) and transformers (long-range dependency modeling) while overcoming their drawbacks. The key insight was that classical system theory already provided a formalism for modeling dynamic processes with long-term dependencies in the form of linear state-space systems.

The primary motivations for developing SSMs for deep learning included:

- **Scalability**: Need for architectures capable of efficiently processing very long sequences (thousands or millions of tokens).
  
- **Computational efficiency**: Requirement for models with linear scaling relative to sequence length, unlike the quadratic complexity of transformers.
  
- **Long-range dependencies**: Need to model dependencies over very large distances without vanishing gradient issues.
  
- **Theoretical grounding**: Desire to leverage well-established mathematical tools from control theory and signal processing.

The transition from classical RNNs to SSMs can be viewed as a natural evolution, where SSMs offer a more formal and powerful framework for modeling dynamical systems while preserving computational efficiency.

### 1.2 Authors and Key Publications in SSM

The development of SSMs for deep learning is associated with several research groups that gradually advanced and refined this paradigm.

**Key researchers and publications:**

1. **Albert Gu and colleagues (2021–2023)**
   - "Efficiently Modeling Long Sequences with Structured State Spaces" — the first work introducing model S4
   - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — a revolutionary paper introducing selective SSMs
   - Gu and his colleagues from Stanford University, later from Carnegie Mellon University, developed the foundational structure of modern SSMs for deep learning

2. **Tri Dao and colleagues (2022–2023)**
   - Co-author of the Mamba architecture
   - Significant contributions to optimizing SSMs for modern hardware platforms

3. **Aviv Regev, Krishnan Prasad, Noam Rot (2022)**
   - "On the Parameterization and Initialization of Diagonal State Space Models" — introduced S4D, a diagonal variant of SSM
   - Key optimizations making SSMs more practical for training and inference

4. **Sirena Dan and Yakov Kerner (2023)**
   - "Simplified State Space Layers for Sequence Modeling" — introduced S5, a simplified and improved version of SSM
   - Simplifications making SSMs more accessible for widespread use

**Connection to signal processing and control theory:**

Notably, SSMs in neural networks demonstrate strong ties to classical signal processing methods. Researchers adapted techniques known for decades in technical literature:

> "SSMs unify recurrent network methods with classical linear systems theory, creating a bridge between deep learning and traditional signal processing. This allows us to leverage the rich mathematical apparatus of control theory for modern neural architectures." — Albert Gu

**Key papers influencing SSM development:**

1. **"HiPPO: Recurrent Memory with Optimal Polynomial Projections" (Gu et al., 2020)**
   - Precursor to SSMs, establishing theoretical foundations for modeling long-range dependencies
   - Laid the mathematical groundwork for subsequent developments of S4 and other SSMs

2. **"Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2021)**
   - First presentation of S4, a structured state space model
   - Demonstrated superior performance on tasks with long-range dependencies

3. **"Diagonal State Spaces are as Effective as Structured State Spaces" (Gupta et al., 2022)**
   - Showed that simpler diagonal SSMs can be as effective as full SSMs
   - Significantly simplified SSM computational complexity and implementation

4. **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)**
   - Introduced selective SSMs that dynamically adapt parameters based on input data
   - A breakthrough making SSMs competitive with transformers even in language modeling tasks

The evolution of SSMs exemplifies effective cross-disciplinary idea exchange, where classical system theory has been successfully adapted to solve modern challenges in deep learning.

### 1.3 Trade-off Balance: Why SSMs Emerged Between RNNs and Transformers

State Space Models emerged in response to fundamental limitations of existing architectures for sequence modeling: recurrent neural networks (including LSTM and GRU) and transformers. SSMs aim to find a "golden middle ground" that combines the best properties of both approaches.

**Problems with existing architectures:**

1. **Limitations of RNN/LSTM/GRU:**
   - **Sequential processing**: Requirement to process sequence elements one after another, limiting parallelism
   - **Difficulty with long-range dependencies**: Although LSTM and GRU partially mitigate vanishing gradients, they remain limited in capturing dependencies over very large distances
   - **Training difficulties**: Gradient instability, especially in deep architectures
   - **Limited scalability**: Challenges in scaling to very deep models

2. **Limitations of transformers:**
   - **Quadratic complexity**: Self-attention mechanism has O(n²) complexity relative to sequence length, making long-sequence processing computationally expensive
   - **Memory constraints**: High memory consumption for long sequences
   - **Fixed context window**: Practical limits on the context length a model can process

**Why SSMs were needed:**

1. **Theoretical elegance:**
   - SSMs are grounded in well-established mathematics from control theory
   - Provide a formal framework for modeling dynamical systems with continuous time
   - Enable analytical methods for model behavior analysis

2. **Scaling to large datasets:**
   - Linear O(n) complexity with respect to sequence length
   - Ability to efficiently process very long sequences (thousands or millions of elements)
   - Lower memory consumption compared to transformers

3. **Balance between efficiency and expressiveness:**
   - Retain ability to model long-range dependencies (like transformers)
   - Computational efficiency comparable to RNNs
   - Enable parallel processing, unlike RNNs

4. **Need for processing diverse modalities:**
   - Need for a unified approach to modeling audio, video, text, and other sequential data
   - SSMs are well-suited for continuous signals and can handle data of varying nature

**Comparative architecture table:**

| **Characteristic** | **RNN/LSTM/GRU** | **Transformers** | **SSM** |
|-------------------|-------------------|-------------------|---------|
| Computational complexity | O(n) | O(n²) | O(n) |
| Parallelism | Limited | High | High |
| Long-range dependencies | Limited capacity | Excellent capacity | Excellent capacity |
| Scalability to long sequences | Good | Limited | Superior |
| Memory usage | Low | High | Low |
| Theoretical grounding | Empirical | Empirical | Formal, from control theory |
| Interpretability | Limited | Via attention | Via system theory |

Quote from Albert Gu on the motivation for creating SSM:

> "We aimed to create an architecture that could scale linearly with sequence length while retaining the ability to model long-range dependencies characteristic of transformers. SSMs offer a theoretically grounded approach to this problem, building on decades of research in control theory and signal processing."

Thus, SSMs represent not merely another architecture, but a fundamentally new approach to sequence modeling, striving to overcome the principled limitations of existing methods.

## 2. Mathematical Foundations and SSM Architecture

### 2.1 Intuition: Metaphor of a "Dynamic System with Memory"

To understand the intuitive essence of State Space Models, it is helpful to think of them as dynamical systems with internal memory that process incoming signals and generate outputs. The metaphor of a "dynamic system with memory" helps clarify the key components and operational principles of SSM.

**Imagine a physical dynamical system:**

Picture a reservoir with liquid into which an input flow ($x(t)$) is fed. The reservoir’s state (level, temperature, pressure — the hidden state $h(t)$) changes under the influence of the input flow according to physical laws. Meanwhile, we measure certain parameters at the output ($y(t)$), which depend on the reservoir’s current state.

**Key distinction from RNNs:**

Unlike RNNs, SSMs are formulated in continuous time and then discretized for computation. This can be visualized as follows: while RNNs (including LSTM and GRU) operate on discrete time steps (e.g., word to word in text), SSMs model a continuous process that is then sampled at discrete intervals.

This continuous nature gives SSMs several theoretical advantages:
- Ability to apply the rich apparatus of differential equations
- More natural modeling of processes occurring in continuous time
- Better adaptability to data with varying temporal resolutions

**Intuitive example: Language modeling**

![SSM workflow diagram](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_01.png    )

In language modeling, SSM can be intuitively understood as follows:
- Input sequence $x(t)$: a stream of words or tokens
- Hidden state $h(t)$: the "understanding" of the text, updated with each new word
- Matrix $A$: determines how quickly context is forgotten over time
- Matrix $B$: determines how strongly each new word influences understanding
- Matrix $C$: determines how current understanding generates predictions for the next words

Thus, SSM can be intuitively understood as a system that continuously processes an information stream while preserving memory of past events and generating outputs based on its current internal state.

### 2.2 Formalization and Notation: Defining Variables and Dimensions

Let us formalize the SSM architecture by defining all its components and corresponding dimensions. This will help better understand the model structure and aid in subsequent implementation.

**Basic notation for linear SSM:**

| **Symbol** | **Dimension** | **Description** |
|------------|---------------|----------------|
| $x(t)$ | $\mathbb{R}^{d_x}$ | Continuous-time input signal |
| $h(t)$ | $\mathbb{R}^{d_h}$ | Continuous-time hidden state |
| $y(t)$ | $\mathbb{R}^{d_y}$ | Continuous-time output signal |
| $A$ | $\mathbb{R}^{d_h \times d_h}$ | State dynamics matrix |
| $B$ | $\mathbb{R}^{d_h \times d_x}$ | Input transformation matrix |
| $C$ | $\mathbb{R}^{d_y \times d_h}$ | Output transformation matrix |
| $D$ | $\mathbb{R}^{d_y \times d_x}$ | Direct feedthrough matrix (optional) |

**Continuous-time State Space Model:**

A linear continuous-time SSM is described by the following differential equations:

$$
\begin{align}
h'(t) &= Ah(t) + Bx(t) \\
y(t) &= Ch(t) + Dx(t)
\end{align}
$$

Where:
- $h'(t)$ is the derivative of the hidden state with respect to time
- $x(t)$ is the input signal at time $t$
- $h(t)$ is the hidden state at time $t$
- $y(t)$ is the output signal at time $t$

The state equation, via matrices $A$ and $B$, describes how the state evolves under the influence of inputs.

![State equation visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_02.png    )

The output equation describes how the state is translated into output (via matrix $C$) and how the input directly affects the output (via matrix $D$).

![Output equation visualization](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_03.png    )

> Note: Matrices $A$, $B$, $C$, and $D$ are trainable parameters.

**Discretization for Practical Computation:**

The original state equations are presented in continuous form and must be transformed for processing discrete inputs. This is achieved using the Zero-Order Hold (ZOH) method, which holds the value of a discrete input signal until the next sample arrives.

![Visual explanation of Zero-Order Hold in transitioning from discrete to continuous](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_04.png    )

In neural networks, discretization is performed with a trainable step $\Delta$. The ZOH method approximates the input signal by quantization. Formally, for a system with parameters $(A, B, C)$, the transformation is described by:

$$
\begin{align}
h_t &= \bar{A}h_{t-1} + \bar{B}x_t \\
y_t &= Ch_t + Dx_t
\end{align}
$$

Where:
- $h_t$ is the discrete hidden state at time $t$
- $x_t$ is the discrete input signal at time $t$
- $y_t$ is the discrete output signal at time $t$
- $\bar{A} = \exp(\Delta A)$ — discretized state dynamics matrix
- $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$ — discretized input transformation matrix

```python
"""
Module for converting continuous systems to discrete form and analyzing them.

Functional Purpose:
-----------------------------
This code provides tools for:
1. Converting continuous systems defined in state-space into discrete systems using the Zero-Order Hold (ZOH) method
2. Simulating the behavior of discrete systems
3. Visualizing discretization results and analyzing the impact of discretization step size
4. Demonstrating the principle of operation of the Zero-Order Hold interpolator

Main Functions:
- continuous_to_discrete: converts a continuous system to discrete form
- simulate_discrete_system: simulates the behavior of a discrete system
- visualize_zoh: visualizes the ZOH principle of operation
- plot_system_response: displays input/output signals and system states
- example_discretization_effect: demonstrates the effect of discretization step size
"""

# Standard libraries
from typing import Tuple, Optional, List, Any

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Scientific computing
from scipy.linalg import expm


def continuous_to_discrete(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
    delta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a continuous system to discrete form using Zero-Order Hold (ZOH).

    Description:
    ---------------
        Discretizes a state-space system using the Zero-Order Hold method.
        This method assumes the input signal remains constant between sampling instants.

    Args:
    ---------------
        A: Continuous system state dynamics matrix (n x n)
        B: Continuous system input transformation matrix (n x m)
        C: Output transformation matrix (p x n)
        D: Direct feedthrough matrix (p x m), defaults to zero matrix
        delta: Discretization step size (default: 1.0)

    Returns:
    ---------------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A_d: Discretized state dynamics matrix (n x n)
            B_d: Discretized input transformation matrix (n x m)
            C_d: Output transformation matrix (p x n)
            D_d: Direct feedthrough matrix (p x m)

    Raises:
    ---------------
        ValueError: If matrix dimensions are inconsistent

    Examples:
    ---------------
        >>> A = np.array([[0, 1], [-1, -0.5]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> A_d, B_d, C_d, D_d = continuous_to_discrete(A, B, C)
    """
    # Validate matrix dimensions
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of rows in matrices A and B must match")
    if C.shape[1] != A.shape[0]:
        raise ValueError(
            "Number of columns in matrix C must equal number of rows in matrix A"
        )

    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Input dimension
    p = C.shape[0]  # Output dimension

    # Compute discretized state dynamics matrix: A_d = exp(delta * A)
    A_d = expm(delta * A)

    # Compute discretized input matrix
    # For cases where A is close to singular, use an alternative approach
    if np.linalg.cond(A) > 1e12:
        # Approximation for ill-conditioned matrices
        B_d = delta * B
    else:
        # Extend matrix for computation via matrix exponential
        n_aug = n + m
        M = np.zeros((n_aug, n_aug))
        M[:n, :n] = delta * A
        M[:n, n:] = delta * B

        # Compute exp(M)
        EM = expm(M)

        # Extract B_d from result
        B_d = EM[:n, n:]

    # C and D remain unchanged during discretization
    C_d = C
    D_d = D if D is not None else np.zeros((p, m))

    return A_d, B_d, C_d, D_d


def simulate_discrete_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    x: np.ndarray,
    h0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a discrete state-space system.

    Description:
    ---------------
        Simulates the behavior of a discrete state-space system governed by:
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t

    Args:
    ---------------
        A: Discretized state dynamics matrix (n x n)
        B: Discretized input transformation matrix (n x m)
        C: Output transformation matrix (p x n)
        D: Direct feedthrough matrix (p x m)
        x: Input signal (T x m) or (T,) for single-input
        h0: Initial system state (n,), defaults to zero vector

    Returns:
    ---------------
        Tuple[np.ndarray, np.ndarray]:
            h: System states at each time step (T x n)
            y: System outputs at each time step (T x p)

    Raises:
    ---------------
        ValueError: If input data dimensions are inconsistent

    Examples:
    ---------------
        >>> A = np.array([[0.9, 0.1], [0, 0.8]])
        >>> B = np.array([[1], [0.5]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> x = np.ones((100, 1))
        >>> h, y = simulate_discrete_system(A, B, C, D, x)
    """
    # Validate matrix dimensions
    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Input dimension
    p = C.shape[0]  # Output dimension

    # Convert x to 2D array if 1D
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Validate input signal dimension
    T = x.shape[0]  # Number of time steps
    if x.shape[1] != m:
        raise ValueError(
            f"Input signal dimension ({x.shape[1]}) does not match expected ({m})"
        )

    # Initialize arrays for states and outputs
    h = np.zeros((T, n))
    y = np.zeros((T, p))

    # Set initial state
    if h0 is not None:
        if len(h0) != n:
            raise ValueError(
                f"Initial state dimension ({len(h0)}) does not match expected ({n})"
            )
        h[0] = h0

    # Compute output at initial time
    y[0] = C @ h[0] + D @ x[0]

    # Simulate system for each time step
    for t in range(1, T):
        h[t] = A @ h[t-1] + B @ x[t]
        y[t] = C @ h[t] + D @ x[t]

    return h, y


def visualize_zoh(
    continuous_time: np.ndarray,
    discrete_time: np.ndarray,
    signal: np.ndarray,
    title: str = "Zero-Order Hold (ZOH) Interpolation"
) -> plt.Figure:
    """
    Visualizes the principle of Zero-Order Hold (ZOH) interpolation.

    Description:
    ---------------
        Demonstrates how a discrete signal is converted to continuous form
        using Zero-Order Hold, which holds the signal value constant between
        sampling instants.

    Args:
    ---------------
        continuous_time: Array of continuous time for plotting
        discrete_time: Array of discrete sampling times
        signal: Discrete signal corresponding to discrete_time
        title: Plot title (default: "Zero-Order Hold (ZOH) Interpolation")

    Returns:
    ---------------
        plt.Figure: Matplotlib figure object with the plot

    Examples:
    ---------------
        >>> continuous_time = np.linspace(0, 2, 1000)
        >>> discrete_time = np.arange(0, 2.1, 0.1)
        >>> signal = np.sin(2 * np.pi * 0.5 * discrete_time)
        >>> fig = visualize_zoh(continuous_time, discrete_time, signal)
    """
    # Flatten signal to 1D if 2D
    if signal.ndim > 1:
        signal = signal.flatten()

    # Create ZOH-interpolated signal
    zoh_signal = np.zeros_like(continuous_time)
    for i, t in enumerate(continuous_time):
        idx = np.searchsorted(discrete_time, t, side='right') - 1
        if idx >= 0 and idx < len(signal):
            zoh_signal[i] = signal[idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot discrete samples
    ax.scatter(
        discrete_time, signal, color='red', s=80, zorder=3, label='Discrete samples'
    )

    # Plot ZOH-interpolated signal
    ax.step(
        continuous_time, zoh_signal, where='post', color='blue',
        linestyle='-', linewidth=2, alpha=0.7, label='ZOH interpolation'
    )

    # Add vertical lines at discrete time points
    for t in discrete_time:
        ax.axvline(x=t, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    return fig


def plot_system_response(
    time: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    title: str = "System Response"
) -> plt.Figure:
    """
    Visualizes system inputs, states, and outputs.

    Description:
    ---------------
        Creates three plots showing:
        1. System input signal
        2. System states
        3. System output signal

    Args:
    ---------------
        time: Array of time samples
        x: System input signal (T x m)
        h: System states (T x n)
        y: System outputs (T x p)
        title: Plot title (default: "System Response")

    Returns:
    ---------------
        plt.Figure: Matplotlib figure object with plots

    Examples:
    ---------------
        >>> time = np.arange(0, 10, 0.1)
        >>> x = np.sin(time).reshape(-1, 1)
        >>> h = np.random.randn(len(time), 2)
        >>> y = np.cos(time).reshape(-1, 1)
        >>> fig = plot_system_response(time, x, h, y)
    """
    # Convert to 2D arrays if 1D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot input signal
    for i in range(x.shape[1]):
        axs[0].plot(
            time, x[:, i], linewidth=2, label=f'x_{i+1}' if x.shape[1] > 1 else 'x'
        )
        axs[0].step(time, x[:, i], linewidth=1, linestyle='--', alpha=0.7, where='post')
    axs[0].set_title('Input Signal', fontsize=14)
    axs[0].set_ylabel('Amplitude', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    if x.shape[1] > 1:
        axs[0].legend()

    # Plot system states
    for i in range(h.shape[1]):
        axs[1].plot(time, h[:, i], linewidth=2, label=f'h_{i+1}')
    axs[1].set_title('System States', fontsize=14)
    axs[1].set_ylabel('Amplitude', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # Plot output signal
    for i in range(y.shape[1]):
        axs[2].plot(
            time, y[:, i], linewidth=2, label=f'y_{i+1}' if y.shape[1] > 1 else 'y'
        )
    axs[2].set_title('Output Signal', fontsize=14)
    axs[2].set_xlabel('Time', fontsize=12)
    axs[2].set_ylabel('Amplitude', fontsize=12)
    axs[2].grid(True, alpha=0.3)
    if y.shape[1] > 1:
        axs[2].legend()

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)

    return fig


def example_discretization_effect(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    delta_values: List[float],
    T_sim: float = 10
) -> plt.Figure:
    """
    Demonstrates the effect of discretization step size on system response.

    Description:
    ---------------
        Compares system responses under different discretization step sizes,
        illustrating how the choice of delta affects modeling accuracy.

    Args:
    ---------------
        A: Continuous system state dynamics matrix (n x n)
        B: Continuous system input transformation matrix (n x m)
        C: Output transformation matrix (p x n)
        D: Direct feedthrough matrix (p x m)
        delta_values: List of discretization step sizes to compare
        T_sim: Simulation time in seconds (default: 10)

    Returns:
    ---------------
        plt.Figure: Matplotlib figure object with comparison plot

    Examples:
    ---------------
        >>> A = np.array([[0, 1], [-1, -0.5]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> delta_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        >>> fig = example_discretization_effect(A, B, C, D, delta_values)
    """
    # Create figure for comparison
    plt.figure(figsize=(14, 8))

    # Color scheme for different delta values
    colors = plt.cm.viridis(np.linspace(0, 1, len(delta_values)))

    for i, delta in enumerate(delta_values):
        # Generate input signal with current delta
        T = int(T_sim / delta) + 1
        t = np.arange(0, T) * delta

        # Step input followed by sine wave
        x = np.zeros((T, 1))
        # Step input starting at 0.5 seconds
        x[int(0.5/delta):] = 1.0

        # Discretize system
        A_d, B_d, C_d, D_d = continuous_to_discrete(A, B, C, D, delta)

        # Simulate system
        h, y = simulate_discrete_system(A_d, B_d, C_d, D_d, x)

        # Plot output signal
        plt.plot(t, y, color=colors[i], label=f'delta = {delta}', linewidth=2)

    plt.title('Effect of Discretization Step Size on System Response', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('System Output', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def run_example() -> None:
    """
    Runs a demonstration of the module functionality.

    Description:
    ---------------
        Demonstrates:
        1. The discretization process of state-space equations
        2. Simulation of a discrete system
        3. Visualization of the ZOH principle
        4. The effect of discretization step size on system response
    """
    print("Demonstration of state-space discretization using ZOH")
    print("=" * 60)

    # Define parameters of continuous system
    # Example: damped oscillator
    A = np.array([
        [0, 1],
        [-1, -0.5]  # ω² = 1, ζ = 0.25
    ])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    print("Continuous system:")
    print("A =\n", A)
    print("B =\n", B)
    print("C =\n", C)
    print("D =\n", D)

    # Discretization step
    delta = 0.1

    # Discretize system
    A_d, B_d, C_d, D_d = continuous_to_discrete(A, B, C, D, delta)

    print("\nDiscretized system (delta =", delta, "):")
    print("A_d =\n", A_d)
    print("B_d =\n", B_d)
    print("C_d =\n", C_d)
    print("D_d =\n", D_d)

    # Generate input signal
    T = 100  # Number of time steps
    time = np.arange(0, T) * delta

    # Step input followed by sine wave
    x = np.zeros((T, 1))
    # Step input from step 10 to 50
    x[10:50] = 1.0
    # Sine wave
    x[50:] = np.sin(2 * np.pi * 0.1 * (np.arange(50, T))).reshape(-1, 1)

    # Simulate system
    h, y = simulate_discrete_system(A_d, B_d, C_d, D_d, x)

    # Visualize results
    fig1 = plot_system_response(time, x, h, y, title="Response of Discretized System")

    # Demonstrate Zero-Order Hold interpolation
    continuous_time = np.linspace(0, 2, 1000)
    discrete_time = np.arange(0, 2.1, delta)
    discrete_signal = np.sin(2 * np.pi * 0.5 * discrete_time)

    fig2 = visualize_zoh(
        continuous_time, discrete_time, discrete_signal,
        title=f"Zero-Order Hold (ZOH) with delta = {delta}"
    )

    # Show effect of different discretization steps
    delta_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    fig3 = example_discretization_effect(A, B, C, D, delta_values)

    plt.show()


if __name__ == "__main__":
    run_example()
```

---

## Resolving the contradiction between the continuity of SSM and the discreteness of the text

1. **Theoretical Foundation and Practical Implementation**

   SSMs are originally formulated in continuous time through differential equations. This is their theoretical foundation, rooted in control theory and signal processing. However, for practical application, these continuous models are discretized.

2. **Abstract Representation of Discrete Data**

   When applying SSMs to text, the following occurs: discrete text tokens (words or subwords) are transformed into continuous vector representations (embeddings). These continuous representations are then used as input signals for the SSM.

3. **Conceptual Bridge via Embeddings**

   Text tokens → Embeddings (continuous vectors) → Processing via SSM → Output continuous representations → Conversion back to discrete predictions

4. **Discretization for Computation**

   Even when working with continuous representations, the model is still discretized for practical computation, as described earlier via the Zero-Order Hold (ZOH) method.

### Deeper Understanding of the Process:

When SSMs are applied to text, each processing step corresponds to one token (e.g., a word or subword). In this context:

- **$x(t)$ becomes $x_t$** — the embedding vector of the current token
- **$h(t)$ becomes $h_t$** — the model’s hidden state after processing the current token
- **$y(t)$ becomes $y_t$** — the output vector used to predict the next token

Thus, although the model is designed based on differential equations in continuous time, in practice we use its discretized version, processing text token by token as if these were discrete time steps.

---

**Parameterization in SSM for Deep Learning:**

In models such as S4 and Mamba, specialized parameterizations of matrices $A$, $B$, and $C$ are employed to improve trainability and computational efficiency:

1. **Structured Matrix $A$**:
   - In S4: A specialized parameterization based on HiPPO (Hierarchical Polynomial Projections) is used.
   - In later variants (S4D): A diagonal matrix $A = \text{diag}(a_1, a_2, ..., a_{d_h})$ is often used.

2. **Parameterization of $B$**:
   - In S4: $B$ may be low-rank or specially structured.
   - In the simplest case: $B$ may be a column vector.

3. **Parameterization of $C$**:
   - Typically, matrix $C$ is parameterized directly.
   - In some SSM variants, constraints on $C$ are applied to improve stability.

**Dimensions in Multi-Layer SSM:**

| **Parameter** | **Dimension** | **Description** |
|---------------|---------------|----------------|
| $d_x$ | Scalar | Dimension of input vector |
| $d_h$ | Scalar | Dimension of hidden state (typically 64 to 1024) |
| $d_y$ | Scalar | Dimension of output vector (usually equal to $d_x$) |
| $L$ | Scalar | Number of SSM layers |
| $N$ | Scalar | Length of input sequence |

**Selective SSM (Mamba):**

In selective SSMs such as Mamba, model parameters become functions of the input data:

$$
\begin{align}
h'(t) &= A(x)h(t) + B(x)x(t) \\
y(t) &= C(x)h(t) + D(x)x(t)
\end{align}
$$

Where $A(x)$, $B(x)$, $C(x)$, and $D(x)$ are functions of the input $x(t)$, typically implemented via neural networks.

**Total Parameters in a Standard SSM Layer:**
- Matrix $A$: $d_h \times d_h$ (or $d_h$ for diagonal parameterization)
- Matrix $B$: $d_h \times d_x$
- Matrix $C$: $d_y \times d_h$
- Matrix $D$ (if used): $d_y \times d_x$
- Total: $d_h \times d_h + d_h \times d_x + d_y \times d_h + d_y \times d_x$ parameters

In selective SSMs, the number of parameters increases due to additional projection layers that generate parameters depending on the input data.

### 2.3 Dynamics of SSM: From Continuous to Discrete Time

Now, let us examine in detail how the State Space Model operates over time, beginning with its continuous formulation and concluding with its discrete implementation used in neural networks.

#### Continuous Time: Formulation via Differential Equations

SSMs in continuous time are described by a system of first-order linear differential equations:

$$
\begin{align}
h'(t) &= Ah(t) + Bx(t) \\
y(t) &= Ch(t) + Dx(t)
\end{align}
$$

The first equation describes how the hidden state evolves over time; the second describes how the hidden state is transformed into the output signal.

**Interpretation of Components:**

- **Matrix $A$** determines the system’s intrinsic dynamics. Its eigenvalues indicate system stability:
  - Negative real parts of eigenvalues → stable system
  - Positive real parts → unstable system
  - Imaginary parts → oscillatory behavior

- **Matrix $B$** determines how the input signal influences changes in the hidden state.

- **Matrix $C$** determines how the hidden state influences the output signal.

- **Matrix $D$** (if used) allows the input signal to directly affect the output signal.

![Final SSM workflow diagram](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_05.png  )

Thus, the entire system operates as follows:

- The input signal is first multiplied by matrix $B$, which describes how inputs influence the system;

- The hidden state is updated. We multiply the state by matrix $A$, which describes how all internal states are interconnected. Matrix $A$ is applied before generating state representations and is updated after representation updates;

- Then, we use matrix $C$ to describe the transformation into the output signal;

- Matrix $D$ is a skip connection used to mitigate gradient vanishing within the network.

#### Analytical Solution in Continuous Time

For a linear system, an analytical solution exists:

$$
\begin{align}
h(t) &= e^{A(t-t_0)}h(t_0) + \int_{t_0}^{t}e^{A(t-\tau)}Bx(\tau)d\tau \\
y(t) &= Ch(t) + Dx(t)
\end{align}
$$

Where $e^{At}$ is the matrix exponential, the solution to the homogeneous equation $h'(t) = Ah(t)$.

This solution reveals that the current system state depends on:
1. The initial state, transformed via the matrix exponential
2. The convolution of the input signal with the kernel $e^{A(t-\tau)}B$

#### Transition to Discrete Time

For practical implementation, the continuous model must be discretized. Several discretization methods exist:

1. **Euler Method (simplest)**:
   $$h_{t} = h_{t-1} + \Delta \cdot (Ah_{t-1} + Bx_t)$$
   
   Where $\Delta$ is the discretization step. This method is simple but inaccurate for rapidly changing systems.

2. **Zero-Order Hold (ZOH) Method**, which assumes the input remains constant during each discretization step:

$$
\begin{align}
h_t &= \bar{A}h_{t-1} + \bar{B}x_t \\
\bar{A} &= e^{A\Delta} \\
\bar{B} &= (A)^{-1}(e^{A\Delta} - I)B
\end{align}
$$

This method provides an exact solution when the input signal is constant between discretization steps.

3. **Bilinear Transformation (Tustin)**, which provides better approximation for systems with oscillatory dynamics.

#### Computational Aspects of Discretization

The key computational challenge is the efficient computation of the matrix exponential $e^{A\Delta}$. Models such as S4 employ specialized techniques for efficient computation:

- **For diagonal matrix $A$** (as in S4D): The exponential is computed element-wise: $e^{A\Delta} = \text{diag}(e^{a_1\Delta}, e^{a_2\Delta}, ..., e^{a_n\Delta})$

- **For general matrix $A$**: Techniques based on Schur decomposition or series approximations are used.

#### Convolutional Interpretation of SSM

One key insight: a discretized SSM can be represented as a one-dimensional convolution:

$$y_t = \sum_{i=0}^{t-1} K_{t-i} x_i + Dx_t$$

where $K_i = C\bar{A}^{i-1}\bar{B}$ is the system’s impulse response.

This convolutional interpretation enables efficient implementation of SSMs via Fast Fourier Transform (FFT), especially beneficial for long sequences:

1. Compute the impulse response $K = [K_1, K_2, ..., K_L]$
2. Use the convolution property in the frequency domain: FFT(x ∗ K) = FFT(x) ⊙ FFT(K)
3. Compute convolution via:
   - $X = \text{FFT}(x)$
   - $K' = \text{FFT}(K)$
   - $Y = \text{IFFT}(X \odot K')$

This reduces complexity from $O(N^2)$ to $O(N \log N)$.

#### Parallel Recurrent Algorithm

For selective SSMs (such as Mamba), the convolutional approach is inapplicable because parameters depend on input data. Instead, a specialized recurrent algorithm is used:

```
h_0 = 0
for t = 1 to N:
    Compute A_t, B_t, C_t based on x_t
    h_t = Ā_t * h_{t-1} + B̄_t * x_t
    y_t = C_t * h_t
```

Although the algorithm appears sequential, Mamba employs a specialized parallel implementation that efficiently leverages GPU architecture.

#### Key Differences from RNN

| **Aspect** | **RNN/LSTM/GRU** | **SSM** |
|------------|-------------------|---------|
| Theoretical Foundation | Discrete recurrent equations | Continuous differential equations |
| Formulation | Initially discrete | Continuous, then discretized |
| Parameterization | Free weight matrices | Structured matrices with theoretical constraints |
| Computation | Strictly sequential | Can be optimized via FFT or parallel scanning |
| Stability | Empirical methods (gates) | Theoretical guarantees via constraints on matrix A |

Thus, SSMs offer a theoretically grounded alternative to traditional RNNs, with direct connections to control theory and signal processing, granting them several theoretical and practical advantages.

## 3. Mathematical Comparison of SSM with Other Architectures

### 3.1 SSM and Classical RNNs: Formal Comparison and Differences

To better understand how SSMs relate to classical recurrent neural networks (including LSTM and GRU), we perform a formal mathematical comparison, highlighting key differences in formulation, parameterization, and behavior.

**Basic Formulations:**

**Standard RNN:**
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$y_t = W_o h_t + b_o$$

**LSTM:**
$$
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
h_t &= o_t \odot \tanh(C_t)
\end{align}
$$

**GRU:**
$$
\begin{align}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
\tilde{h}_t &= \tanh(W_h \cdot [x_t, r_t \odot h_{t-1}] + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{align}
$$

**Discretized SSM:**
$$
\begin{align}
h_t &= \bar{A}h_{t-1} + \bar{B}x_t \\
y_t &= Ch_t + Dx_t
\end{align}
$$

**Key Differences:**

**1. Origin and Theoretical Foundations:**

- **RNN/LSTM/GRU**: Developed empirically as recurrent neural networks for sequential data processing. Gating mechanisms in LSTM and GRU were introduced to address specific learning challenges.

- **SSM**: Based on well-developed theory of linear systems and control. The model is originally formulated in continuous time and then discretized for computation.

**2. Nonlinearity:**

- **RNN/LSTM/GRU**: Explicit nonlinear activation functions (tanh, sigmoid) are applied at each step.

- **SSM (basic)**: Linear system at its core. Nonlinearity is typically introduced via external layers or more complex architectures.

**3. State Update:**

- **RNN**: Simple recurrent update with full replacement of the previous state.

- **LSTM/GRU**: Complex gating mechanisms for selective updating of state components.

- **SSM**: Linear dynamics where influence of the previous state is controlled by matrix $\bar{A}$, and influence of the input by matrix $\bar{B}$.

**4. Parameterization:**

- **RNN/LSTM/GRU**: Arbitrary weight matrices without special structure.

- **SSM**: Structured matrices with specialized parameterization (diagonal, HiPPO-like, etc.) ensuring theoretical guarantees of stability and efficiency.

**5. Mathematical Interpretation:**

- **RNN/LSTM/GRU**: Discrete mapping from input and previous state to new state.

- **SSM**: Approximation of integrating a continuous differential equation. The state represents an accumulated "imprint" of the entire history, modulated by exponential decay.

**6. Connection to LSTM via Linearization:**

Interestingly, a connection between LSTM and linear SSMs can be shown via linearization. If LSTM is linearized around an equilibrium point, we obtain:

$$h_t \approx Ah_{t-1} + Bx_t$$

Where $A$ and $B$ are Jacobian matrices corresponding to partial derivatives with respect to $h_{t-1}$ and $x_t$. This form resembles a discretized SSM.

**7. Gradient Flow and Stability:**

- **RNN**: Susceptible to vanishing/exploding gradients due to repeated multiplication by the same matrix.

- **LSTM/GRU**: Mitigate this issue via gating mechanisms that create "gradient highways".

- **SSM**: Stability is controlled via eigenvalues of matrix $A$. Theoretically grounded parameterization ensures training stability.

**Formal Comparison of Gradient Flow:**

For a basic RNN, the gradient during backpropagation is:
$$\frac{\partial \mathcal{L}}{\partial h_t} = \sum_{k=t+1}^{T} \frac{\partial \mathcal{L}}{\partial h_k} \prod_{j=t+1}^{k} \text{diag}(\tanh'(W_h h_{j-1} + W_x x_j + b)) W_h$$

For SSM:
$$\frac{\partial \mathcal{L}}{\partial h_t} = \sum_{k=t+1}^{T} \frac{\partial \mathcal{L}}{\partial h_k} \prod_{j=t+1}^{k} \bar{A}_j$$

In SSM with diagonal matrix $\bar{A}$, each state component is updated independently, making gradient flow more controllable.

**8. Computational Scaling:**

- **RNN/LSTM/GRU**: Strictly sequential computation, limiting parallelism.

- **SSM** (standard): Can be computed via convolution with FFT, achieving $O(N \log N)$ complexity for a sequence of length $N$.

- **Selective SSM** (Mamba): Require specialized parallel scanning algorithms but still maintain linear $O(N)$ complexity.

Thus, SSMs represent a fundamentally different approach to sequence modeling compared to classical RNNs, with stronger theoretical foundations and potentially superior learning and scaling properties.

### 3.2 SSM and Transformers: Mechanisms for Modeling Long-Range Dependencies

State Space Models (SSM) and transformers represent two distinct approaches to modeling long-range dependencies in sequences. They achieve similar goals but use fundamentally different mechanisms. We now conduct a comparative analysis.

**  
- **Трансформеры**:  
  Влияние позиции $j$ на позицию $i$ выражается как:  
  $$z_i = \sum_{j=1}^{N} \alpha_{ij} \cdot (W_V x_j)$$  
  где $\alpha_{ij}$ — весовые коэффициенты внимания между позициями $i$ и $j$.  

- **SSM**:  
  Влияние всех предыдущих позиций на текущую позицию $t$ выражается как:  
  $$h_t = \bar{A}h_{t-1} + \bar{B}x_t = \sum_{i=0}^{t-1} \bar{A}^{t-1-i}\bar{B}x_i$$  
  $$y_t = Ch_t = C\sum_{i=0}^{t-1} \bar{A}^{t-1-i}\bar{B}x_i = \sum_{i=0}^{t-1} K_{t-i}x_i$$  
  где $K_j = C\bar{A}^{j-1}\bar{B}$ — импульсная характеристика системы.  

2. **Затухание влияния с расстоянием**:  

- **Трансформеры**:  
  В принципе, нет ограничений на затухание — вес внимания для дальних зависимостей может быть таким же высоким, как и для ближних.  

- **Стандартные SSM**:  
  Затухание определяется собственными значениями матрицы $\bar{A}$. Для устойчивой системы влияние затухает экспоненциально:  
  $$K_j \sim e^{\lambda j}$$  
  где $\lambda < 0$ — доминирующее собственное значение матрицы $A$.  

- **Селективные SSM (Mamba)**:  
  Затухание становится зависимым от содержания, так как параметры матрицы $\bar{A}$ меняются в зависимости от входа:  
  $$K_j(x) \sim e^{\lambda(x) j}$$  

3. **Теоретическая ёмкость памяти**:  

- **Трансформеры**:  
  Определяется размером контекстного окна. Каждая позиция имеет прямой доступ ко всем позициям в окне.  

- **SSM**:  
  Определяется размерностью скрытого состояния $d_h$ и спектральными свойствами матрицы $A$. Чем ближе собственные значения к мнимой оси, тем дольше сохраняется информация.  

4. **Содержательная селективность**:  

- **Трансформеры**:  
  Высокая селективность через механизм внимания, который явно вычисляет отношения между всеми парами позиций.  

- **Стандартные SSM**:  
  Низкая содержательная селективность — параметры фиксированы и не зависят от содержания.  

- **Селективные SSM (Mamba)**:  
  Высокая селективность через параметризацию, зависящую от входа, которая позволяет модели динамически адаптировать память.  

5. **Аналитическое сравнение для конкретных задач**:  

**Задача: копирование на дальнюю дистанцию**  
Задача копирования требует "запоминания" информации с определенной позиции и её воспроизведения позже.  

- **Трансформеры**:  
  Решают задачу через прямое внимание к нужной позиции с высоким весом $\alpha_{ij}$.  

- **SSM**:  
  Стандартные SSM затрудняются с точным копированием из-за экспоненциального затухания.  
  Mamba решает проблему, адаптируя скорость затухания в зависимости от содержания.  

**Задача: индукционные головки**  
Индукционные головки требуют определения закономерностей и их экстраполяции.  

- **Трансформеры**:  
  Формируют индукционные головки через механизм внимания, явно моделируя отношения между текущей позицией и предыдущими вхождениями шаблона.  

- **SSM**:  
  Mamba формирует аналог индукционных головок через селективное сохранение информации в скрытом состоянии, регулируя параметры $\bar{A}$ и $\bar{B}$ в зависимости от входа.  

**Визуализация механизмов моделирования долговременных зависимостей**:  
Для наглядности, можно представить обработку последовательности "a b c a d e a f":  

- **Трансформеры**:  
  При предсказании токена после последнего "a", модель может напрямую обратить внимание на предыдущие вхождения "a" и извлечь следующие за ними токены ("b", "d"), что способствует прогнозированию "f".  

- **SSM (Mamba)**:  
  При обработке последнего "a", модель адаптирует параметры так, чтобы сохранить информацию о контексте после предыдущих "a". Эта информация накапливается в скрытом состоянии и используется для прогнозирования "f".  

**Сравнительная таблица механизмов**:  

| **Аспект** | **Трансформеры** | **Стандартные SSM** | **Селективные SSM (Mamba)** |
|------------|------------------|--------------------|------------------------------|
| Основной механизм | Явное внимание | Рекуррентная передача | Адаптивная рекуррентная передача |
| Сложность с длиной | O(N²) | O(N) | O(N) |
| Содержательная селективность | Высокая | Низкая | Высокая |
| Затухание влияния | Произвольное | Фиксированное экспоненциальное | Адаптивное экспоненциальное |
| Параллелизм | Высокий | Высокий через FFT | Умеренный через специальные алгоритмы |
| Интерпретируемость | Веса внимания | Импульсная характеристика | Зависящая от входа характеристика |

Таким образом, SSM и трансформеры представляют два разных, но мощных подхода к моделированию долговременных зависимостей. SSM (особенно в версии Mamba) достигают многих возможностей трансформеров при линейной сложности, что делает их особенно привлекательными для обработки длинных последовательностей.

### 3.3 Вычислительная эффективность: линейное масштабирование и оптимизации

Одним из ключевых преимуществ моделей пространства состояний (SSM) является их вычислительная эффективность по сравнению с трансформерами, особенно при работе с длинными последовательностями. Рассмотрим подробно аспекты вычислительной эффективности SSM и специальные оптимизации, применяемые в их реализациях.

**Сравнение асимптотической сложности**:  

| **Архитектура** | **Временная сложность** | **Пространственная сложность** |
|-----------------|-------------------------|--------------------------------|
| Трансформеры (стандартные) | O(N²) | O(N²) |
| Трансформеры с линейным вниманием | O(N) | O(N) |
| RNN/LSTM/GRU | O(N) | O(1) |
| Стандартные SSM | O(N log N) через FFT | O(N) |
| Селективные SSM (Mamba) | O(N) | O(N) |

**Анализ вычислительных паттернов**:  

1. **Трансформеры**:  
   - Доминирующая операция: матричное умножение для расчета внимания $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$  
   - Узкое место: вычисление матрицы внимания $QK^T$ размера $N \times N$  
   - Проблема: квадратичный рост с длиной последовательности  

2. **Стандартные SSM**:  
   - Доминирующая операция: свертка через FFT  
   - Операции:   
     - Вычисление импульсной характеристики $K = [C\bar{A}^{0}\bar{B}, C\bar{A}^{1}\bar{B}, ..., C\bar{A}^{L-1}\bar{B}]$  
     - FFT преобразование: $X = \text{FFT}(x)$, $K' = \text{FFT}(K)$  
     - Поэлементное умножение: $Y = X \odot K'$  
     - Обратное FFT: $y = \text{IFFT}(Y)$  
   - Преимущество: сложность O(N log N) вместо O(N²)  

3. **Селективные SSM (Mamba)**:  
   - Невозможность использования FFT из-за параметров, зависящих от входа  
   - Доминирующая операция: параллельное рекуррентное сканирование  
   - Специальные оптимизации для параллельного выполнения на GPU  

**Конкретные оптимизации в реализациях SSM**:  

1. **Параллельное рекуррентное сканирование в Mamba**:  
   
   Основной алгоритм выглядит последовательным:  
   ```
   h_0 = 0
   for t = 1 to N:
       Вычислить A_t, B_t, C_t на основе x_t
       h_t = A_t * h_{t-1} + B_t * x_t
       y_t = C_t * h_t
   ```  
   
   Однако в Mamba применяется специальная техника параллельного сканирования:  
   - Разделение последовательности на блоки, которые могут обрабатываться параллельно  
   - Использование алгоритма с несколькими проходами для объединения результатов  
   - Оптимизация использования GPU-памяти путем хранения промежуточных состояний в быстрой SRAM  

2. **Оптимизация расчета матричной экспоненты**:  
   
   Для диагональной параметризации (как в S4D):  
   - Замена вычисления полной матричной экспоненты на поэлементную экспоненту диагональных элементов  
   - Снижение сложности с O(d_h³) до O(d_h)  

3. **Кэширование и повторное использование**:  
   
   В стандартных SSM с фиксированными параметрами:  
   - Предварительное вычисление импульсной характеристики $K$ для повторного использования  
   - Кэширование дискретизированных матриц $\bar{A}$ и $\bar{B}$  

4. **Оптимизации размера ядра свертки**:  
   
   - Ограничение длины импульсной характеристики определенным "окном" L << N  
   - Применение усечения для коэффициентов с малым влиянием  
   - Использование разреженных представлений для импульсной характеристики  

**Специализированные реализации для аппаратного ускорения**:  

1. **CUDA-реализации для Mamba**:  
   
   - Использование регистровой памяти GPU для хранения расширенного состояния  
   - Разработка специализированных CUDA-ядер для параллельного сканирования  
   - Оптимизации для специфических особенностей архитектуры NVIDIA (тензорные ядра, разделяемая память)  

2. **Оптимизации для инференса**:  
   
   - Квантизация параметров модели (int8, float16)  
   - Слияние операций для снижения накладных расходов  
   - Специализированные реализации для edge-устройств  

3. **Применение TensorCore и специализированных акселераторов**:  
   
   - Использование тензорных ядер для ускорения матричных операций  
   - Оптимизации под специализированные NPU/TPU  

**Сравнение времени выполнения в реальных условиях**:  
Для последовательности длиной 2048 токенов (данные из статьи о Mamba):  

| **Модель** | **Время прямого прохода (мс)** | **Время обратного прохода (мс)** | **Использование памяти (МБ)** |
|------------|--------------------------------|----------------------------------|-------------------------------|
| Трансформер | 14.2 | 28.7 | 640 |
| Трансформер (FlashAttention-2) | 9.8 | 19.5 | 400 |
| Стандартный SSM (S4) | 7.3 | 16.2 | 180 |
| Mamba | 7.9 | 17.1 | 210 |

Для последовательности длиной 8192 токенов:  

| **Модель** | **Время прямого прохода (мс)** | **Время обратного прохода (мс)** | **Использование памяти (МБ)** |
|------------|--------------------------------|----------------------------------|-------------------------------|
| Трансформер | 198.5 | 401.6 | 9500 |
| Трансформер (FlashAttention-2) | 41.2 | 83.7 | 1600 |
| Стандартный SSM (S4) | 28.6 | 62.1 | 720 |
| Mamba | 31.4 | 67.8 | 830 |

Масштабирование к очень длинным последовательностям (64K токенов) показывает еще более значительное преимущество SSM-архитектур, особенно в использовании памяти, что критично для обработки длинного контекста.  

**Практические выводы**:  

1. **Для последовательностей средней длины** (до 2K токенов), оптимизированные реализации трансформеров и SSM показывают сопоставимую производительность.  
2. **Для длинных последовательностей** (4K-16K токенов), SSM начинают демонстрировать существенное преимущество в эффективности.  
3. **Для очень длинных последовательностей** (>16K токенов), преимущество SSM становится определяющим фактором, позволяя эффективно обрабатывать контексты, недоступные стандартным трансформерам.  
4. **С точки зрения энергоэффективности**, SSM-архитектуры требуют значительно меньше энергии для обработки сопоставимого объема данных, что особенно важно для масштабных моделей и мобильных приложений.  

Таким образом, вычислительная эффективность SSM-архитектур делает их особенно привлекательными для задач с длинным контекстом и для приложений с ограниченными вычислительными ресурсами. Линейное масштабирование по длине последовательности позволяет преодолеть фундаментальное ограничение трансформеров, открывая новые возможности для моделирования последовательностей.  

## **Вывод**  

Модели пространств состояний (SSM) являются не просто альтернативой трансформерам или RNN, а новой парадигмой, объединяющей строгую математику, эффективность и гибкость. Их развитие открывает путь к более интерпретируемым, масштабируемым и вычислительно доступным архитектурам, способным обрабатывать контексты длиной в десятки тысяч токенов и далее. В условиях, когда модели становятся всё более громоздкими, SSM возвращают внимание к эффективности, устойчивости и фундаментальности.  

</details>  

---  

В статье «**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**» авторы **Albert Gu** (Carnegie Mellon University) и **Tri Dao** (Princeton University) представляют новую архитектуру, которая устраняет это ограничение, сохраняя при этом мощные возможности моделирования **Transformers**.  

![Selective State Space Model with Hardware-Aware State Expansion](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Image_01.png  )  
*Figure 1: Architecture of the selective state space model with hardware-aware state expansion. Input data influences model parameters through a selection mechanism, enabling content-aware reasoning.*  

**Mamba** represents a fundamental shift in sequence modeling, combining the linear scaling properties of State Space Models (**SSM**) with a novel selection mechanism that enables content-aware reasoning — a capability previously exclusive to attention mechanisms. This development has profound implications for the efficiency and capabilities of AI systems processing sequential data.  

## **Understanding Traditional Sequence Modeling**  

To appreciate the innovation of **Mamba**, it is essential to understand the evolution of sequence modeling approaches and their limitations.  

**Transformers** revolutionized sequence modeling with their self-attention mechanism, which creates direct connections between all positions in a sequence. This provides exceptional modeling power, but at the cost of quadratic computational complexity relative to sequence length — a problem known as the "attention bottleneck."  

Several approaches have been developed to alleviate this bottleneck:  

- **Linear attention**: Approximations of inner attention with linear complexity.  
- **Convolutional algorithms**: Extended convolutional models with strobing mechanisms.  
- **Recurrent Neural Networks (RNN)**: Sequential processing with hidden state updates.  
- **Structured State Space Models (SSM)**: Continuous systems discretized for sequence modeling.  

While these alternatives achieve linear scaling with sequence length, they typically lack the content-aware reasoning capabilities of attention, significantly limiting their effectiveness in language modeling tasks.  

**State Space Models (SSM)**, in particular, have shown promising results in efficiently modeling long-range dependencies, but traditional **SSM** use fixed parameters independent of input content. This time-invariance restricts their ability to perform tasks such as selective copying and induction heads — fundamental operations for language understanding.  

![Comparison of Copying Tasks](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Image_02.png  )  
*Figure 2: Comparison of copying tasks. Left: Standard copying task solvable by time-invariant models. Right: Selective copying and induction heads require content-aware reasoning.*  

**Mamba** introduces the concept of "selective state space models" (selective SSM), which allow model parameters to be functions of input data. This enables the model to dynamically decide what information to store and propagate based on the content of current inputs.  

The core innovation is that SSM parameters (**A**, **B**, **C**, and **D**) become functions of the input data, rather than fixed values. This is achieved through a selection mechanism that projects inputs to determine parameter values at each step.  

### **Continuous-Time SSM**  

A linear continuous-time SSM is described by the following differential equations:  

$$
\begin{align}
h'(t) &= Ah(t) + Bx(t) \\
y(t) &= Ch(t) + Dx(t)
\end{align}
$$  

Where:  
- $h'(t)$ is the derivative of the hidden state with respect to time  
- $x(t)$ is the input signal at time $t$  
- $h(t)$ is the hidden state at time $t$  
- $y(t)$ is the output signal at time $t$  

The state equation, via matrices $A$ and $B$, describes how the state evolves under the influence of inputs.  

<div align="center">  
  <img src="https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_02.png  " alt="Visualization of state equation">  
</div>  

The output equation describes how the state is translated into output (via matrix $C$) and how the input directly affects the output (via matrix $D$).  

<div align="center">  
  <img src="https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_03.png  " alt="Visualization of output equation">  
</div>  

> Note: Matrices $A$, $B$, $C$, and $D$ are trainable parameters.  

**Interpretation of components**:  

- **Matrix $A$** determines the system’s intrinsic dynamics. Its eigenvalues indicate system stability:  
  - Negative real parts of eigenvalues → stable system  
  - Positive real parts → unstable system  
  - Imaginary parts → oscillatory behavior  

- **Matrix $B$** determines how the input signal influences changes in the hidden state.  

- **Matrix $C$** determines how the hidden state influences the output signal.  

- **Matrix $D$** (if used) allows the input signal to directly affect the output signal.  

<div align="center">  
  <img src="https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_05.png  " alt="Final SSM workflow diagram">  
</div>  

Thus, the entire system operates as follows:  

- The input signal is first multiplied by matrix $B$, which describes how inputs influence the system;  

- The hidden state is updated. We multiply the state by matrix $A$, which describes how all internal states are interconnected. Matrix $A$ is applied before generating state representations and is updated after representation updates;  

- Then, we use matrix $C$ to describe the transformation into the output signal;  

- Matrix $D$ is a skip connection used to mitigate gradient vanishing within the network.  

**In the selective formulation of SSM, these parameters become dependent on input data**:  

$$
\begin{align}
h'(t) &= A(x)h(t) + B(x)x(t) \\
y(t) &= C(x)h(t) + D(x)x(t)
\end{align}
$$  

Where $A(x)$, $B(x)$, $C(x)$, and $D(x)$ are functions of the input $x(t)$, typically implemented via neural networks.  

**Total parameters in a standard SSM layer**:  
- Matrix $A$: $d_h \times d_h$ (or $d_h$ for diagonal parameterization)  
- Matrix $B$: $d_h \times d_x$  
- Matrix $C$: $d_y \times d_h$  
- Matrix $D$ (if used): $d_y \times d_x$  
- Total: $d_h \times d_h + d_h \times d_x + d_y \times d_h + d_y \times d_x$ parameters  

In selective SSMs, the number of parameters increases due to additional projection layers that generate parameters depending on the input data.

### **Projection Layers for Dynamic Parameter Generation**

To make the parameters of an SSM layer (matrices $A$, $B$, $C$, and $D$) functions of the input data $x(t)$, **projection layers** are employed. These are essentially small neural networks (sometimes called hypernetworks) that, at each step, "look at" the current input and generate new parameter values.

1. **Concept of the Projection Layer**  
   Instead of storing a fixed matrix $A$, we learn an additional network $f_A$ that, given an input vector $x$, outputs a "flattened" parameter vector $\theta_A$. This vector is then reshaped into a matrix of the same dimensions as $A$. Similarly, networks $f_B$, $f_C$, and $f_D$ are used.  
   $$
     \theta_A = f_A(x), \quad A(x) = \mathrm{reshape}(\theta_A)
   $$

2. **Structure of a Single Projection Layer**  
   Typically, $f_A$ is a one- or two-layer MLP (fully connected network):  
   $$
   \begin{aligned}
   z_1 &= W_1 x + b_1,\\
   a_1 &= \sigma(z_1),\\
   \theta_A &= W_2 a_1 + b_2,
   \end{aligned}
   $$
   where  
   - $W_1, b_1$ are parameters of the first layer (dimensions $d_{\text{proj}}\times d_x$ and $d_{\text{proj}}$),  
   - $W_2, b_2$ are parameters of the output layer ($d_h^2\times d_{\text{proj}}$ and $d_h^2$),  
   - $\sigma$ is a nonlinearity (ReLU, GELU, etc.),  
   - $d_{\text{proj}}$ is the "hidden" dimension of the projection layer.  

   The vector $\theta_A\in\mathbb{R}^{d_h^2}$ is then reshaped into a matrix $A(x)\in\mathbb{R}^{d_h\times d_h}$.

3. **Advantages and Overhead**  
   - **Flexibility.** Network $f_A$ "learns" to produce different system dynamics depending on the content of $x$.  
   - **Local Adaptation.** The model can immediately respond to new events in the input by changing its internal mechanics.  
   - **Overhead.** Instead of one set of parameters, we store parameters of the hypernetwork:  
     $$
       \underbrace{d_{\text{proj}}\cdot d_x + d_{\text{proj}}}_{\text{first layer}}
       \;+\;
       \underbrace{d_h^2\cdot d_{\text{proj}} + d_h^2}_{\text{second layer}}
     $$
     But since $d_{\text{proj}}\ll d_h^2$, the increase in parameter count remains moderate.

4. **Practical Example**  
   Let $d_x = 128$, $d_h = 64$, and choose $d_{\text{proj}} = 32$.  
   - First layer of hypernetwork: $32\times128 + 32 = 4128$ parameters.  
   - Second layer: $64^2\times32 + 64^2 = 131\,072 + 4096 = 135\,168$.  
   - Total ≈139,296 parameters instead of a fixed $A$ of size $64^2 = 4096$.

<u>Thus, projection layers (hypernetworks) transform the input signal $x(t)$ into settings for the internal parameters of the SSM layer, enabling dynamic, content-dependent sequence processing. This is the key to combining the linear scaling of SSM with the "intelligent" reasoning characteristic of attention mechanisms.</u>

### **Discretized Version**

The discretized version used in **Mamba** is expressed as follows:

![Transition from continuous SSM to discrete. Now we feed discrete values as input and obtain discrete output.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/SSM/Image_06.png  )

$$
h_t = \bar A_t\,h_{t-1} + \bar B_t\,x_t
$$

$$
y_t = C_t\,h_t + D_t\,x_t
$$

Where parameters $\bar A_t$, $\bar B_t$, $C_t$, and $D_t$ at each step are computed by projection layers (hypernetworks) based on the current input signal $x_t$.

### Integration into the Model

The architecture of **Mamba** integrates these selective SSMs into an optimized model structure that is remarkably simple compared to **Transformers**. The model consists of alternating layers of selective SSM and simple projections, requiring no separate attention or MLP blocks.

![Architectural Evolution to Mamba](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Image_03.png  )

*Figure 3: Comparison of architectures showing evolution from H3 (an SSM variant) to Mamba via closed MLP blocks.*

# **Introducing: Nemotron-H**

## **Hybrid Mamba-Transformer Architecture**

**Nemotron-H** implements a *hybrid Mamba-Transformer architecture*, where the majority of layers are **Mamba-2** Structured State-Space Model (SSM) layers, and a small fraction are classical Transformer *self-attention* layers, interleaved with Feed-Forward Network (FFN) layers. The model structure is carefully designed to leverage the strengths of both approaches: SSM layers provide efficient processing of long sequences with linear (or even constant) complexity relative to sequence length, while a few self-attention layers add the model’s ability for precise "gluing" of global context and superior **in-context learning** capabilities.

In the **Nemotron-H** architecture, only about **8% of layers** are self-attention layers — they are *uniformly distributed* across the entire depth of the network. The remaining layers alternate: approximately half are Mamba-2 layers and half are FFN layers. For example, the Nemotron-H model with ~8 billion parameters contains *52 layers*: of these, **4 self-attention layers** (~7.7% of all layers), and 24 Mamba-2 and 24 FFN layers respectively. In the larger version (~56 billion parameters), there are a total of **118 layers**, of which **10 are self-attention**, and the remaining 108 are evenly split between Mamba-2 and FFN (54 of each type).

![Architectures of Nemotron-H-8B/56B models](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_02.png  )

*Architectures of Nemotron-H-8B/56B models. Approximately 8% of the total number of layers in the model are self-attention layers; these layers are uniformly distributed across the entire model. The rest of the model consists of alternating Mamba-2 and FFN layers.*

Special placement rules are enforced in the layer stack: **the first layer is always Mamba-2**, and **the last layer is FFN**. Each self-attention layer is placed *before* its corresponding FFN (as in a standard Transformer block). This arrangement allows leveraging the fact that **the first SSM layer can independently learn positional dependencies**, eliminating the need for explicit positional embeddings. Indeed, Nemotron-H **uses no external positional encodings whatsoever** — the model orients itself within the input sequence through the inherent computation of Mamba layers and global attention coverage, which further enables generalization to sequences longer than those seen during training.

The hybrid Mamba-Transformer architecture has proven highly effective. In NVIDIA’s experiments, the 8-billion-parameter hybrid model **Mamba-2-Hybrid** outperformed a pure Transformer model of the same size by an average of 12 points across 12 standard NLP tasks, and is predicted to achieve up to **8x faster token generation** during inference. On very long inputs (up to 128k tokens), the hybrid also maintains quality at or above Transformer levels, demonstrating successful integration of both architectures’ advantages.

## **Training Principles: FP8 Training and Distillation**

**Training Nemotron-H** employed specialized techniques to accelerate and optimize the process: first, **low-precision FP8 format** was used for computations during training; second, specialized *distillation* and model compression techniques were applied to obtain more compact versions without significant quality loss.

### **FP8 Training**

The largest model, Nemotron-H-56B, became NVIDIA’s first model trained from scratch using **FP8 (8-bit float)** numbers in all major matrix operations. This significantly accelerates training and reduces memory consumption compared to traditional BF16/FP16, but requires careful handling to avoid quality degradation due to reduced precision. A *hybrid FP8 scheme* was applied: all linear layers (including QKV projections in attention and feed-forward layers) are computed in 8-bit, **except the first 4 and last 4 layers** of the model — these extreme layers are retained in higher-precision BF16 to preserve stability. In FP8 mode, two number formats are used: **E4M3** (4 bits exponent, 3 bits mantissa) for *weights and activations*, and the wider **E5M2** (5 bits exponent, 2 bits mantissa) for *gradients*. This choice is motivated by the fact that gradients have a larger dynamic range (but can be coarsely quantized in mantissa), while weights/activations can be stored with a smaller dynamic range but higher relative precision.

![Relative loss difference between FP8 and BF16 during training](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_06.jpeg  )

*Relative loss difference between FP8 and BF16 during training, showing convergence over training progress.*

To effectively leverage FP8, **per-tensor dynamic quantization** of activations is applied. Specifically, for each tensor, a quantization scale is computed as the ratio of the maximum representable FP8 value to the maximum absolute value in the tensor. All elements are then multiplied by this scale and rounded to FP8 format. This normalization maximizes the use of the available 8-bit float range and minimizes rounding errors. Equally important is the choice of rounding scheme during quantization: experiments showed that **truncation (rounding toward zero)** yields better results during subsequent fine-tuning — likely because it distorts gradient distributions less, especially on later training stages.

![Comparison of FP8 and BF16 training](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_07.jpeg  )

*Comparison of FP8 and BF16 training across various benchmarks, showing comparable or superior performance with FP8 training.*

Despite this aggressive reduction in bit-width, model quality suffered almost no degradation. The log-loss during FP8 training was only a fraction of a percent higher than with BF16, and although the gap slightly increased near the end of training (possibly due to accumulation of very small gradients being zeroed out in E5M2), the final model’s ability to solve tasks was no worse — and in some tests even better — than an equivalent BF16-trained model. Graphs showed that the FP8 model achieved the same or higher accuracy on downstream tasks despite a slightly higher training loss. Thus, the **FP8 recipe** proved its viability, enabling the training of a 56-billion-parameter Nemotron-H without extending the development cycle or degrading quality relative to conventional half-precision training.

### **Model Distillation and Compression**

To adapt large models for diverse deployment scenarios, NVIDIA applied *compression techniques with minimal quality loss*. In particular, a more compact **Nemotron-H-47B** model was derived from the top-tier Nemotron-H-56B using a combination of **pruning and distillation**, named ***MiniPuzzle***. The goal was to reduce the number of parameters (~16%) and required memory/latency while preserving nearly identical accuracy. MiniPuzzle combines ideas from *Minitron* and *Puzzle* approaches.

The overall algorithm proceeds as follows:

- a. First, the *importance* of various components of the large model (each layer, FFN neurons, etc.) is evaluated based on their impact on final error:

![Layer importance scores](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_09.jpeg  )

*Layer importance scores, showing the varying contributions of different layer types.*

- b. Then, an *architecture search* is performed — which layers can be entirely removed and how much to reduce the width of FFN layers — to fit within resource constraints while preserving the most important components:

![Layer selection patterns for candidate architectures](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_10.jpeg  )

*Layer selection patterns for candidate architectures, showing which layers are preserved in each potential compressed model.*

- c. Based on these decisions, the least important layers are **removed** (entire Mamba/Attention/FFN layers are deleted) and the sizes of certain layers are reduced (e.g., internal FFN dimensions are shortened). This yields a compressed architecture: in the 56B→47B case, several layers were discarded (in particular, half of the self-attention layers — 5 out of 10, and parts of others) and widths of some FFN layers were reduced to remove ~9 billion parameters.

![MiniPuzzle compression workflow](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_08.jpeg  )

*MiniPuzzle compression workflow, showing transition from a pre-trained model to a compressed model via importance evaluation, neural architecture search, and distillation.*

After architectural compression, the resulting model (student) must be trained to approximate the quality of the original (teacher). This is achieved via *knowledge distillation* — training on outputs of the large model. In the distillation process, **progressive unfreezing** of layers helps transfer knowledge more smoothly. Initially, the student’s main weights can be frozen, and only new or modified parameters trained. 

For the language model, MiniPuzzle proceeded as follows:

1. Initialize student weights using surviving weights from the teacher (copied unchanged); new parameters are randomly initialized;  
2. First, train only the output layer and topmost layers, ensuring the student’s logits distribution approaches the teacher’s (minimizing, for example, **Kullback–Leibler divergence** $D_\text{KL}(p_\text{teacher} \parallel p_\text{student})$ between next-token probability distributions) — this ensures correct "calibration" of the output to the teacher;  
3. Then **unfreeze the next layers** and continue training, progressing toward the network’s beginning. This *stepwise unblocking* avoids abrupt feature-space jumps: the student model gradually inherits knowledge, starting from high-level representations and proceeding deeper. Simultaneously, alongside distillation loss, standard language modeling loss (cross-entropy on ground-truth tokens) may also be used — this combines **soft-distillation** (using teacher’s "soft" probability labels) with supervised learning on true data. Formally, the distillation loss is often taken as:

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{\text{CE}}(y^\text{true}, p_\text{student}) + \lambda\, T^2\, D_{\text{KL}}\!\big(p_\text{teacher}^{(T)} \parallel p_\text{student}^{(T)}\big),$$ 

where 

- $\mathcal{L}_{\text{CE}}$ is standard cross-entropy on the true next token;
- $D_{\text{KL}}$ is Kullback–Leibler divergence between teacher and student distributions, both scaled by a common temperature $T$ (higher temperature smooths distributions for more stable training);
- $\lambda$ is a coefficient determining the distillation contribution. $\lambda$ and the unfreezing schedule are chosen experimentally.

Ultimately, after several fine-tuning stages with the teacher, the Nemotron-H-47B model achieved **nearly identical accuracy** to the 56B model, gaining ~20% in inference speed and resource efficiency.

<details> 
    <summary><em><strong>Formalization of Kullback–Leibler Divergence</strong></em></summary>

---

The **Kullback–Leibler divergence** (KL divergence) is a measure of the difference between two probability distributions. In knowledge distillation, it quantifies how much the student model’s probability distribution deviates from the teacher’s. Mathematically, for discrete distributions, KL divergence is defined as:

$$D_\text{KL}(p_\text{teacher} \parallel p_\text{student}) = \sum_{i=1}^{|V|} p_\text{teacher}(i) \log \frac{p_\text{teacher}(i)}{p_\text{student}(i)}$$

where:
- $|V|$ — vocabulary size (number of all possible tokens)
- $p_\text{teacher}(i)$ — probability of token $i$ according to the teacher model
- $p_\text{student}(i)$ — probability of token $i$ according to the student model

Intuitively, KL divergence measures the "surprise" of using $p_\text{student}$ instead of $p_\text{teacher}$. The smaller this value, the better the student reproduces the teacher’s knowledge. Importantly, KL divergence is asymmetric: $D_\text{KL}(p_\text{teacher} \parallel p_\text{student}) \neq D_\text{KL}(p_\text{student} \parallel p_\text{teacher})$. In distillation, the direction $p_\text{teacher} \to p_\text{student}$ is used, forcing the student to pay special attention to tokens the teacher assigns high probability to.

When using temperature $T$, distributions are modified as follows:

$$p_\text{model}^{(T)}(i) = \frac{\exp(z_i/T)}{\sum_{j=1}^{|V|} \exp(z_j/T)}$$

where $z_i$ are the original logits for token $i$. Higher temperature $(T > 1)$ makes distributions smoother, reducing the gap between probabilities of different tokens, helping the student capture subtle preferences of the teacher among alternative continuations.

</details>

<details> 
    <summary><em><strong>Formalization of Cross-Entropy</strong></em></summary>

---

**Cross-entropy** between the true distribution and the student model’s distribution is used for training on real data. If $y^{\text{true}}$ represents the true next token, then cross-entropy is defined as:

$$\mathcal{L}_{\text{CE}}(y^{\text{true}}, p_\text{student}) = -\sum_{i=1}^{|V|} \mathbb{1}[i = y^{\text{true}}] \log p_\text{student}(i) = -\log p_\text{student}(y^{\text{true}})$$

where $\mathbb{1}[i = y^{\text{true}}]$ is an indicator function equal to 1 if $i$ is the true next token, and 0 otherwise.

In standard language model training, this quantity is minimized, equivalent to maximizing likelihood on training data. However, this approach teaches the model only on "hard" labels, ignoring information about other plausible continuations that may be nearly as good as the true token.

Mathematically, there is an interesting relationship between KL divergence and cross-entropy:

$$D_\text{KL}(p_\text{teacher} \parallel p_\text{student}) = H(p_\text{teacher}, p_\text{student}) - H(p_\text{teacher})$$

where $H(p_\text{teacher}, p_\text{student})$ is the cross-entropy between distributions, and $H(p_\text{teacher})$ is the entropy of the teacher’s distribution. Since $H(p_\text{teacher})$ does not depend on student parameters, minimizing KL divergence is equivalent to minimizing cross-entropy between teacher and student distributions.

</details>

---

**Differences Between Optimizing KL Divergence and Shannon Cross-Entropy**

**Optimization Goal**  
   - **KL divergence** ($D_\text{KL}(p_\text{teacher}\parallel p_\text{student})$) directly measures the "distance" between two soft distributions — teacher and student. During optimization, the student learns to reproduce the full shape of the teacher’s distribution, considering probabilities of all tokens.  
   - **Cross-entropy** ($\mathcal{L}_\text{CE}(y^\text{true}, p_\text{student})$) works with "hard" labels: only the correct token contributes non-zero loss. The student seeks to maximize the probability of a single true token, ignoring alternatives.

## **Optimizations for Long Context**

A primary goal of Nemotron-H is efficient operation with *long contexts* (tens of thousands or more tokens). To this end, specialized optimizations were applied during training and inference:

### **Dynamic Sequence Packing**

**Dynamic sequence packing** — a technique used during training of models with large context windows. The idea is to more efficiently fill each training sequence (up to maximum length) with multiple original samples, minimizing padding losses. Unlike simple random batching, dynamic packing analyzes the lengths of individual texts in the training corpus and concatenates several short texts sequentially into one long example, separating them with special delimiter tokens. **Thus, the model sees contexts containing multiple unrelated or weakly related parts**, teaching it to be robust to topic shifts and not rely on positional anchoring at the end of training examples.

Practically, for a given maximum context $L_{\max}$, the packing algorithm can operate greedily: take the longest texts and pad them with shorter ones to achieve a total length close to $L_{\max}$ without overflow. These *composite sequences* are fed to the model as a single input. The model is trained to ignore special separators and predict the **next word while also accounting for the possibility of context switches within the sequence**. This is especially useful for long-context models: they learn to *efficiently distribute attention between relevant parts of a large input*. As a result, **no context interval is wasted** — even if real data contains few very long documents, packing synthesizes long sequences from multiple fragments.

### **Example of Dynamic Sequence Packing**

Consider a practical example of dynamic sequence packing when training a language model with a maximum context window of 4096 tokens.

**Original texts in the training corpus:**

1. **Climate news article:** 850 tokens  
2. **AI research paper:** 1200 tokens  
3. **Italian pasta recipe:** 300 tokens  
4. **Smartphone review:** 560 tokens  
5. **Poem:** 180 tokens  
6. **Technical documentation excerpt:** 920 tokens  

**Standard approach (without dynamic packing):**

```
Batch 1: [Climate article + padding to 4096 tokens]
Batch 2: [AI paper + padding to 4096 tokens]
Batch 3: [Recipe + padding to 4096 tokens]
Batch 4: [Smartphone review + padding to 4096 tokens]
Batch 5: [Poem + padding to 4096 tokens]
Batch 6: [Technical doc + padding to 4096 tokens]
```

**Approach with dynamic packing:**

```
Batch 1: [Climate article (850) + <SEP> + Recipe (300) + <SEP> + Poem (180) + <SEP> + Technical doc (920) + padding]
        = 2253 tokens + padding to 4096
        
Batch 2: [AI paper (1200) + <SEP> + Smartphone review (560) + <SEP> + (other texts) + padding]
        = 1763 tokens + padding to 4096
```

Moreover, dynamic packing enables *cumulative long-context skill training*: for example, the maximum length of packed sequences can be gradually increased during training (curriculum learning by context length). Initially, the model is trained on relatively short sequences (but diverse in content), then the window expands — up to target lengths of 32k or 100k tokens. This approach ensures **phased learning** by length: the model does not immediately face extremely long dependencies but gradually adapts to them. The Nemotron-H report mentions a *phased approach* in training: dividing training into phases with different data mixtures. This applied to data diversity, but similarly can be applied to length. Altogether, dynamic packing maximizes memory and time usage per training iteration, *simulating long dialogues or documents* from fragments, improving the model’s capabilities on long contexts.

### Extended RoPE Positional Encodings (with θ Parameter)

To support long sequences, it is crucial to correctly convey positional information beyond the training range. One effective solution is **Rotary Positional Embeddings (RoPE)** with a modified rotation parameter (θ) for extrapolation to longer contexts. RoPE introduces a *multiplicative positional shift* into attention: elements $q_i, k_i$ for each position $i$ are represented as complex rotations of vectors in planes dependent on position $i$. Formally, if a $d$-dimensional vector is split into pairs of coordinates $(u, v)$, RoPE applies the transformation: $(u, v)$ at position $i$ becomes $(u\cos \theta_i + v\sin \theta_i,\; -u\sin \theta_i + v\cos \theta_i)$, where $\theta_i$ is defined, for example, as $\theta_i = i \cdot \theta_{\text{base}}^{2j/d}$ for the $j$-th coordinate pair (here $\theta_{\text{base}}$ is a base coefficient usually tied to maximum length). The resulting dot products $q_i \cdot k_j$ then obtain an intrinsic cosine factor $\cos(\theta_i - \theta_j)$, depending on the positional difference $i$ and $j$. This is equivalent to relative positional embeddings, allowing the model to generate beyond seen positions without a fixed window.

**Extending RoPE with θ** means adapting the base rotation step for a longer range. For example, if a model was trained with maximum length $N$, the base scale θ is typically chosen so that rotations at position $N$ cover a full phase. To extend the limit to $k \cdot N$, we can **reduce the rotation step**: effectively take $\theta_{\text{base,new}} = \theta_{\text{base}} / c$ for some coefficient $c > 1$. Then, for the original maximum $N$, the rotation becomes only $1/c$ of a full cycle, and the full cycle stretches to $c \cdot N$. Essentially, this *compresses the frequency spectrum* of positional rotations, enabling larger positions before phases begin to repeat. This trick is often called *NTK preservation* (Neural Tangent Kernel projection) for long contexts: it attempts to preserve the relative positioning ratios in the new range as they were in the training range.

For instance, LLaMA 2 developers, when extending context to 32k, applied **RoPE Scaling**: multiplying position indices by a factor <1 before computing rotational phases, effectively stretching the positional space. In the context of Nemotron models with self-attention layers, such schemes can also be applied. Although Nemotron-H base models do not use explicit positional embeddings, when adapting purely Transformer-based models (e.g., Nemotron-T) to 128k length, RoPE scaling approaches were employed. Experiments show that *without retraining*, the attention window can be extended by correctly selecting θ for rotary encodings: the model continues to understand token order without encountering unknown positional patterns.

**Conclusion:** *RoPE with θ extension* is a way to inform Transformer attention about positions >> the original maximum, *extrapolating* the rotational period. This technique, compatible with the hybrid architecture (for the few attention layers in Nemotron-H), enables safe operation on 100k+ token contexts without noticeable quality drop or costly retraining on such lengths.

<details> 
    <summary><em><strong>Formalization of Rotary Positional Embedding (RoPE)</strong></em></summary>

### **1. Why are positional embeddings needed?**
- **Problem**: In the classical Transformer self-attention mechanism, token order is lost because all tokens are processed in parallel.
- **Goal**: Give the model knowledge of each token’s position while preserving flexibility in handling relative distances.

### **2. Core idea of RoPE: rotation instead of addition**
- Instead of **adding** a separate positional vector to the token vector (as in absolute embeddings), RoPE **rotates** the feature vector itself.
- Each two consecutive numbers in the vector (a coordinate pair) are treated as a point on a 2D plane.  
- For a token at position *m*, each such pair is rotated by an angle proportional to *m*.

![RoPE](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_15.webp  )

> **Analogy.** Imagine you have an arrow (feature vector) on a circle, and you rotate it by an angle depending on the word’s position. The farther the word, the greater the rotation.

### **3. How it works simply**

**Splitting a 1024-dimensional vector into pairs**

1. **Number of pairs:**  
   $$
     \frac{1024}{2} = 512\text{ pairs}
   $$

2. **Indexing formula:**  
   For each $(i\in\{1,2,\dots,512\})$, define the pair as  
   
   $$
     \text{pair}_i = \bigl[x_{2i-1},\;x_{2i}\bigr]
   $$

3. **Examples:**  
   - **Pair 1:** $([x_1,\,x_2])$ 
   - **Pair 2:** $([x_3,\,x_4])$  
   - …  
   - **Pair 512:** $([x_{1023},\,x_{1024}])$

4. **Next:** apply a rotation matrix to each pair  
   
   $$
     R(m,\theta_i)=
     \begin{pmatrix}
       \cos(m\theta_i) & -\sin(m\theta_i)\\
       \sin(m\theta_i) &  \cos(m\theta_i)
     \end{pmatrix}
   $$

   where  
   
   $$
     \theta_i = 10000^{-2i/1024},\qquad m = \text{token position}
   $$

Thus, for a 1024-dimensional embedding vector, you:
- Form 512 two-dimensional vectors.
- Apply a unique rotation to each, encoding the token’s position.
- Obtain a final vector of the same dimension, now "numbered" via rotations.

**Result:** RoPE simply "rotates" each small vector by an angle dependent on position, thereby simultaneously preserving absolute and relative positional information.

</details> 

## **Accuracy Across Benchmark Tests**

Despite architectural changes, Nemotron-H models maintain high performance across a wide range of tests:

- Nemotron-H-56B outperforms Llama-3.1-70B in 16 of 17 evaluated tasks
- Models demonstrate particularly high performance in mathematical reasoning tasks

![Comparison of Nemotron-H and other models on MMLU](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_13.jpeg  )

*Comparison of Nemotron-H and other models on MMLU, showing competitive performance.*

Models were evaluated using a comprehensive benchmark suite including MMLU, GSM8K, MATH, HumanEval, and various reasoning tasks, consistently demonstrating competitive or superior performance compared to Transformer models of similar size.

## **Conclusion**

In this analysis, we traced the evolution of sequence modeling architectures: from classical RNNs and their improvements (LSTM, GRU) to the revolutionary attention mechanism and hybrid approaches based on selective State Space Models (SSM). Recurrent networks demonstrated their value in streaming data and resource-constrained settings thanks to compact memory and natural causality. However, their limited ability to model very long dependencies spurred the development of more sophisticated architectures.

The emergence of the Transformer with self-attention enabled efficient handling of global token relationships, but faced quadratic complexity with sequence length. SSM family models (S4, S5) proposed linear (and often constant) scaling, yet lacked content-aware selectivity. The Mamba architecture unified SSM advantages with selective mechanisms, allowing state parameters to adapt to input data and achieving high performance on long contexts.

Finally, the hybrid model Nemotron-H strategically combines Mamba-2 SSM layers with a small fraction of Transformer self-attention layers, ensuring linear complexity during long-sequence generation while preserving global context strength for in-context learning. This balanced approach signals a new stage in the evolution of language and multimodal models, where efficient long-dependency processing is combined with deep content understanding.

Thus, further development of hybrid architectures that combine the best properties of RNNs, SSMs, and Transformers promises significant breakthroughs in NLP, computer vision, and biological signal modeling, enabling the creation of faster, more scalable, and more accurate artificial intelligence systems.