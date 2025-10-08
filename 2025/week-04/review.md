# ModernBERT: The Successor to BERT 🚀

**ModernBERT** — a new generation of bidirectional encoder that combines long sequence processing, code understanding, and efficient computation. 🌟

## Introduction 📜

BERT was released in **2018**, but it is still widely used. In fact, it is the second most popular model on the **Hugging Face Hub** with over **68 million monthly downloads**! 🚀 This is because its encoder-only architecture makes it ideal for many real-world tasks, such as:

- **Search** (e.g., RAG)
- **Classification** (e.g., content moderation)
- **Named Entity Recognition** (e.g., for privacy and compliance)

Finally, after **6 years**, we have a successor! 🎉 **ModernBERT**. This new family of models surpasses BERT and its analogs in both speed and accuracy. 🚀

ModernBERT builds upon decades of advancements in large language models (LLMs) and applies them to BERT-style models, including architectural updates and training processes. 🧠

Beyond being faster and more accurate, ModernBERT also increases context length to **8,000 tokens** (compared to 512 for most encoders) and is the first encoder-only model designed to include a large amount of code in its training data. 💻

These capabilities open new application areas previously inaccessible through open models, such as:

- **Large-scale code search**
- **New IDE capabilities**
- **New full-text search pipelines**, based on full-text extraction rather than small snippets

But to explain what we're doing, let's take a step back and look at where we came from. 🔍

> We expect ModernBERT to become the new standard in many applications where encoder-only models are currently used, such as in **RAG** (Retrieval-Augmented Generation) pipelines and recommendation systems. 📊

## Encoder-Only Model 🤖

Recent major developments in LLMs have focused on models like **GPT**, **Claude**, **Llama**, **Mistral**, and **DeepSeek**. These are decoder-only or generative models. Their ability to generate human-like content has led to new amazing application areas for **GenAI**, such as generative art and interactive chat. 🎨💬

These attractive applications have attracted significant investment, funded explosive research, and led to rapid technological progress. Essentially, we've brought these achievements back to encoder-only models. 🚀

**Why?** Because for many practical applications, you need an optimized and powerful model! And it doesn't have to be a generative model. 💡

Roughly speaking, decoder-only models are too large, too slow, too patented, and too expensive for many tasks. Consider that the original **GPT-1** was a model with **117 million parameters**. For comparison, the **Llama 3.1** model has **405 billion parameters**, and its technical report describes synthesis methods and control mechanisms that are too complex and expensive for most companies to reproduce. 💸

Thus, to use a model like **ChatGPT**, you need to pay a fee and wait several seconds to get an API response from a heavy server you can't control. ⏳

Of course, the boundless capabilities of these massive generative models mean you can reluctantly use them for non-generative or discriminative tasks, such as classification. This is because you can describe the classification task in simple language and then... just let the model do the classification. But although this workflow is excellent for prototyping, you won't want to pay the prototyping price once you move to production. 💼

The obsession with the popularity of **GenAI** has overshadowed the capabilities of encoder-only models. These are the foundation of real-world language processing, and these models are actually used in many scientific and commercial applications for such workloads. 🧑‍💻

## Encoder-Only Model 🛠️

The output of an encoder-only model is a list of numbers (embedding vectors). You can say that instead of generating a text answer, the encoder model encodes its "answer" into this compressed numerical form. This vector represents a compressed representation of the model's input data, so encoder-only models are sometimes called embedding models. 📊

Although decoder-only models (e.g., GPT) can perform the work of encoder-only models (e.g., BERT), they are limited by a key constraint: since they are generative models, they are mathematically "not allowed" to "look ahead" beyond the token. They can only look backward. This is the difference from encoder-only models, which are trained to look forward and backward (in a bidirectional manner) for each token. They are designed for this, making them very efficient at the task. 🚀

Essentially, advanced models like **O1** from **OpenAI** are like **Ferrari SF-23**. This is clearly a triumph of engineering designed to win races, so we talk about it. But to change a tire, you need a special repair crew, and you can't buy one yourself. In contrast, the **BERT** model looks like a **Honda Civic**. This is also a triumph of engineering, but more subtle, as it is designed to be accessible, economical, reliable, and very useful. That's why they are absolutely everywhere. 🚗

# It will be very stuffy, this block can be skipped 😌

<details>
  <summary>Click to expand</summary>

### Let's Recall How an Encoder Works 🤖

> This block will cover the basic mathematical architecture of the encoder and decoder. We will also draw a parallel between the original BERT architecture and the new ModernBERT, as ModernBERT is a modernized version of BERT with improved architecture, including RoPE, GeGLU, Flash Attention, and other optimizations.

In the original Transformer model described in the paper "Attention Is All You Need," the architecture is divided into two main parts: the encoder and the decoder. Both parts consist of layers with the same general structure but serve different purposes.

The figure below shows the architecture of the Transformer model. It consists of two main parts: the **encoder (encoder)** and the **decoder (decoder)**.

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Figure_1.png  )

### Encoder (Encoder)
The encoder is usually located on the left side of the architecture. It consists of several layers, each of which includes:
1. **Multi-Head Attention** — an attention mechanism that allows the model to focus on different parts of the input data.
2. **Add & Norm** — a layer that adds the input data to the attention result (residual connection) and applies normalization.
3. **Feed Forward** — a fully connected layer that is applied to each sequence element independently.
4. **Add & Norm** — again adds the input data to the result and normalizes.

These layers are repeated several times (usually 6 or more) to create a deep model.

### Decoder (Decoder)
The decoder is usually located on the right side of the architecture. It also consists of several layers but has additional components:
1. **Masked Multi-Head Attention** — an attention mechanism that masks future tokens to prevent "peeking" ahead.
2. **Add & Norm** — a layer that adds the input data to the attention result and normalizes.
3. **Multi-Head Attention** — an attention mechanism that considers the output of the encoder.
4. **Add & Norm** — again adds the input data to the result and normalizes.
5. **Feed Forward** — a fully connected layer, similar to that used in the encoder.
6. **Add & Norm** — a final add and normalize layer.

### Inputs and Outputs
- **Input Embedding** and **Positional Encoding** relate to the input data fed into the encoder.
- **Output Embedding** and **Outputs (shifted right)** relate to the output data processed by the decoder.

### **Encoder**:

#### Roles:
The role of the encoder is to process the input data and create a representation reflecting the relationships between elements (e.g., words in a sentence). This part of the transformer does not generate any new content; it simply transforms the input data into a state that the decoder can use.

#### System Functionality:
Each encoder layer has self-attention mechanisms and feed-forward neural networks. The self-attention mechanism allows each position in the encoder to process all positions in the previous encoder layer — thus, it can learn the context around each word.

#### Contextual Embeddings:
The output of the encoder is a series of vectors that represent the input sequence in a multidimensional space. These vectors are often called contextual embeddings because they encode not only individual words but also their context within the sentence.

#### Mathematical Description:

1. **Self-Attention (Self-Attention)**:
   - **Input Vector $X$**: This is a matrix representing the input sequence (e.g., words in a sentence) as embeddings.
   - **Keys $K$**, **Queries $Q$**, and **Values $V$**: These matrices are obtained by multiplying the input vector $X$ by weight matrices $W_K$, $W_Q$, and $W_V$ respectively:

     $$
     Q = XW_Q, \quad K = XW_K, \quad V = XW_V
     $$

   - **Attention $A$**: Calculated by taking the dot product of queries and keys, normalized by the key dimension $d_k$:

     $$
     A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
     $$

   - **Self-Attention Output $Z$**: This is the result of multiplying the attention matrix by the values:

     $$
     Z = AV
     $$

```python
import numpy as np

def self_attention(X, d_k):
    """
    Description:
      Implements the Self-Attention mechanism.

    Args:
        X (numpy.ndarray): Input vector (matrix of embeddings), shape (n_samples, embedding_dim).
        d_k (int): Dimension of keys.

    Returns:
        numpy.ndarray: Self-attention output Z.
    """
    # Initialize weight matrices with random values
    embedding_dim = X.shape[1]
    W_Q = np.random.rand(embedding_dim, d_k)
    W_K = np.random.rand(embedding_dim, d_k)
    W_V = np.random.rand(embedding_dim, d_k)

    # Calculate query, key, and value matrices
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # Calculate attention matrix
    attention_scores = Q @ K.T
    scaled_attention_scores = attention_scores / np.sqrt(d_k)

    # Apply softmax function to get attention weights
    attention_weights = np.exp(scaled_attention_scores) / np.sum(np.exp(scaled_attention_scores), axis=-1, keepdims=True)

    # Calculate output vector
    Z = attention_weights @ V

    return Z

# Example usage
# Suppose we have a sequence of 3 words, each represented as a 4-dimensional vector
X = np.array([[1.0, 0.5, 0.2, 0.8],
              [0.3, 0.9, 0.1, 0.4],
              [0.6, 0.2, 0.7, 0.5]])

d_k = 2  # Example key dimension

# Get the self-attention output
output_Z = self_attention(X, d_k)

print("Input vector X:\n", X)
print("\nSelf-attention output Z:\n", output_Z)
```

2. **Feedforward Neural Network (Feedforward Neural Network)**:
   - The self-attention output $Z$ passes through two fully connected layers with an activation function (e.g., ReLU) between them:

     $$
     \text{FFN}(Z) = \max(0, ZW_1 + b_1)W_2 + b_2
     $$

   - **$W_1$, $W_2$**: Weight matrices of the first and second fully connected layers.
   - **$b_1$, $b_2$**: Biases of the first and second fully connected layers.

#### **Weight Matrices $W_K$, $W_Q$, and $W_V$ in the Context of Transformers**

1. General Context

  In the Transformer architecture, the attention mechanism plays a crucial role, allowing the model to consider context when processing each element of the sequence (e.g., words in a sentence). For this, the attention mechanism uses three key components: **queries (queries)**, **keys (keys)**, and **values (values)**. These components are formed using weight matrices $W_Q$, $W_K$, and $W_V$ respectively.

2. Formation of Weight Matrices

  2.1 Input Data.
  
  Suppose we have an input sequence $X$ of size $n \times d_{model}$, where:
  - $n$ — the number of tokens in the sequence (sentence length).
  - $d_{model}$ — the dimension of embeddings (e.g., 512 or 1024).

  This sequence $X$ is fed into the Transformer, where it first undergoes linear transformations to form queries $Q$, keys $K$, and values $V$.

  2.2 Linear Transformations.

  To create queries, keys, and values, the input sequence is multiplied by three different weight matrices $W_Q$, $W_K$, and $W_V$:

  $$
  Q = XW_Q, \quad K = XW_K, \quad V = XW_V
  $$

  where:
  - $W_Q$ — weight matrix for queries of size $d_{model} \times d_k$.
  - $W_K$ — weight matrix for keys of size $d_{model} \times d_k$.
  - $W_V$ — weight matrix for values of size $d_{model} \times d_v$.

  The dimensions $d_k$ and $d_v$ can be chosen differently, but usually $d_k = d_v = d_{model}$. These weight matrices are trained during the model's training process, and their task is to find such representations of queries, keys, and values that allow for optimal context consideration.

3. Why Different Matrices Are Used?

  3.1 Queries ($W_Q$)

  The matrix $W_Q$ transforms the input vectors into queries, which are used to evaluate the importance of other sequence elements relative to the current one. This means that each query tries to "figure out" which other words in the sentence it should pay attention to.

  3.2 Keys ($W_K$)

  The matrix $W_K$ transforms the input vectors into keys. Keys are used to match with queries. Essentially, keys contain information about "how important" each sequence element is when matching with a query. The higher the similarity between a query and a key (measured by the dot product), the more attention will be paid to the corresponding element.

  3.3 Values ($W_V$)

  The matrix $W_V$ transforms the input vectors into values. Values convey the actual information that will be used after applying the attention mechanism. The final value for each token will be a weighted sum of all token values, where the weights are determined by the degree of attention (calculated from queries and keys).

4. Attention Mechanism

  4.1 Calculating Attention

  Queries $Q$ and keys $K$ are used to calculate the attention matrix. For this, the dot product of queries and keys is calculated, normalized by the dimension $d_k$, and then the softmax function is applied:

  $$
  A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
  $$

  Where $A$ is the attention matrix, showing how much each element of the sequence influences others.

  4.2 Applying Attention to Values

  After obtaining the attention matrix, it is multiplied by the value matrix $V$ to obtain the final representation for each token:

  $$
  Z = AV
  $$

5. Conclusion

  The weight matrices $W_Q$, $W_K$, and $W_V$ serve to transform input data into queries, keys, and values necessary for the attention mechanism. These matrices are trained during the model's training process and play a crucial role in determining how the model interprets the context of each word in the sentence.

  Such attention to context allows Transformers to efficiently solve natural language processing tasks, achieving high performance in translation, text understanding, and many others.

## **From Input to Output Inside the Encoder**

### 1. **Input Data:**

  - **Input Sequence $X$**: A matrix of size $(N, L, D_{model})$, 

  where:

  **Batch Size ($N$) - Size of the Batch:**

  * Batch Size defines how many **independent** sequences the model processes **simultaneously** in one full training or inference pass.

  **Sequence Length ($L$) - Length of the Sequence:**

  * Sequence length defines the number of **elements** (tokens) in each **individual** sequence within the batch.

  **Embedding Dimension ($D_{model}$) - Dimension of the Embeddings:**

  * Fixed vector representation of any object (e.g., a word, a token).

  **How is the matrix X obtained:**

  1. **Token Embedding:** Each element in the sequence (e.g., a word in a sentence) is converted into an embedding - a vector of size $D_{model}$.

  2. **Forming a Sequence:** For each sequence in the batch, we obtain a matrix of size $(L, D_{model})$. Each row of this matrix corresponds to the embedding of one element of the sequence. Thus, we have $L$ rows (by the number of elements in the sequence) and $D_{model}$ columns (by the embedding dimension).

  3. **Forming a Batch:** Since we have $N$ independent sequences in the batch, we "stack" these $(L, D_{model})$ matrices on top of each other. This creates a three-dimensional tensor (or matrix) $X$ of size $(N, L, D_{model})$.

### **In Summary:**

The matrix $X$ of size $(N, L, D_{model})$ represents a three-dimensional tensor. If viewed as a set of two-dimensional matrices, it consists of $N$ matrices of size $(L, D_{model})$, "stacked" on top of each other. Each of these internal matrices (corresponding to one sequence in the batch) has $L$ rows (by the number of elements in the sequence) and $D_{model}$ columns (by the embedding dimension).

### **Example:**

```Python
import torch

# Example data: 3 sentences in the batch
sentences = [
    "Hello everyone! I am interested in artificial intelligence",  # 7 tokens
    "Hello, how are you?",                                   # 4 tokens
    "AI is interesting!"                                  # 5 tokens
]

# 1. Simplified tokenization for each sentence
batch_tokens = [
    ["[CLS]", "Hello", "everyone", "!", "I", "am", "interested", "in", "artificial", "intelligence", "[SEP]"],  # L=11
    ["[CLS]", "Hello", ",", "how", "are", "you", "?", "[SEP]", "[PAD]", "[PAD]"],                      # L=11 (with padding)
    ["[CLS]", "AI", "—", "is", "interesting", "!", "[SEP]", "[PAD]", "[PAD]"]                      # L=11 (with padding)
]

N = len(sentences)  # Batch size = 3
L = 11               # Maximum sequence length (after padding)
D_model = 512       # Embedding dimension

# 2. Create random embeddings for the entire batch
embeddings = torch.randn(N, L, D_model)

# 3. Form the input tensor X
X = embeddings

print("Shape of tensor X:", X.shape)  # (3, 11, 512)
print("\nStructure of data:")
print(f"• Batch size (N): {N}")
print(f"• Sequence length (L): {L} (with padding)")
print(f"• Embedding dimension (D_model): {D_model}\n")

# ================= Visualization of the process =================
# Visualize the content
for batch_idx in range(N):
    print(f"Batch {batch_idx + 1} ({sentences[batch_idx][:20]}...):")
    print(f"Tokens: {batch_tokens[batch_idx]}")
    
    # Show the first 3 elements of embeddings for key positions
    print("\nExample embeddings:")
    print(f"• [CLS] token: {X[batch_idx, 0, :3].detach().numpy().round(4)}...")
    print(f"• 3rd token:   {X[batch_idx, 2, :3].detach().numpy().round(4)}...")
    print(f"• [SEP] token: {X[batch_idx, -3, :3].detach().numpy().round(4)}...")
    print(f"• [PAD] token: {X[batch_idx, -1, :3].detach().numpy().round(4)}...")
    print("-" * 60)
```

```Python
# ================= Detailed output for the first batch =================
# Visualize the content of tensor X
for batch_idx in range(N):
    print(f"Batch {batch_idx + 1} ({sentences[batch_idx][:20]}...):")
    print(f"Tokens: {batch_tokens[batch_idx]}")
    
    # Output all embeddings for the current sequence
    print("\nEmbeddings:")
    for token_idx in range(L):
        print(f"• Token {token_idx}: {X[batch_idx, token_idx].detach().numpy().round(4)}...")
    print("-" * 60)
```

### 2. **Positional Encodings (Positional Encodings):**

  **Why are positional encodings needed at all?** Traditional recurrent neural networks (RNNs), such as LSTM or GRU, process the sequence token by token, and the order of processing naturally accounts for the position of the token. However, attention-based architectures like Transformer process all tokens in the sequence in parallel. Because of this, they lose information about the order of tokens. Positional encodings are introduced to "tell" the model where each token is located in the sequence.

  When a trained model receives a sentence for analysis (e.g., for classification or machine translation), the encoder Transformer also processes all tokens in the sequence in parallel, forming contextualized representations for each token.

  It is also important to note that positional encodings ($PE$) have the same dimension as word embeddings ($D_{model}$). This is a key point, as it allows them to be added element-wise.

  ![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Figure_2.png  )

  - To account for the order of words in the sequence, positional encodings $PE$ of the same dimension $D_{model}$ are added to the word embeddings.
  - $PE$ are calculated using the following formulas:
    $$
    PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D_{model}}}\right) \\
    PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D_{model}}}\right)
    $$
    
    where:

  *   **$pos$ (position):**
      *   Represents an integer indicating the position of the token in the sequence.
      *   Numbering starts from 0. The first token has $pos = 0$, the second has $pos = 1$, and so on.
      *   For example, in the sentence "The dog barks loudly", "The" has $pos = 0$, "barks" has $pos = 1$, "loudly" has $pos = 2$.
  *   **$i$ (dimension index):**
      *   This is an index that determines a specific dimension in the positional encoding vector.
      *   $i$ takes values from 0 to $D_{model}/2 - 1$.
      *   For each value of $i$, a pair of values is calculated: one with sine, the other with cosine.
      *   For example, if $D_{model} = 512$, then $i$ will take values from 0 to 255.
  *   **$D_{model}$ (model dimension):**
      *   This is the dimension of word embeddings and positional encodings.
      *   Usually this value is 512, but it can be different.
      *   $D_{model}$ defines the length of the positional encoding vector.

  **Why use sine and cosine?**

  *   **Uniqueness:** Sine and cosine allow generating unique positional encodings for each position.
  *   **Periodicity:** The periodicity of these functions allows the model to easily distinguish relative positions of tokens.
  *   **Extrapolation:** The model can extrapolate to longer sequences than those on which it was trained.
  *   **Relative positions:** The difference between positional encodings for adjacent positions remains relatively constant, helping the model understand the relative position of tokens.

  **Generating unique positional encodings:**

  For each position $pos$ and each dimension index $i$ relative to the three-dimensional tensor $X$ of size $(N, L, D_{model})$, a unique value is calculated. Since $i$ ranges from 0 to $D_{model}/2 - 1$, for each position $pos$, a vector of size $D_{model}$ is obtained. The first $D_{model}/2$ elements of this vector are calculated using sine, and the remaining $D_{model}/2$ elements are calculated using cosine.

  **Size Compatibility:**

  Positional encodings have the same dimension as word embeddings ($D_{model}$), allowing them to be added element-wise. This allows the model to consider both the semantic meaning of a word (from the embedding) and its position in the sequence (from the positional encoding).

### **In Summary:**

The final input for the first encoder layer is obtained by element-wise addition of word embeddings $X$ and positional encodings $PE$: $X_{embedded} = X + PE$.

# Example:

Suppose $D_{model} = 4$. Then for position $pos = 1$ and $i = 0$, we get:

$$PE_{(1, 0)} = \sin\left(\frac{1}{10000^{0}}\right) = \sin(1) \approx 0.84$$
$$PE_{(1, 1)} = \cos\left(\frac{1}{10000^{0}}\right) = \cos(1) \approx 0.54$$

For $i = 1$:

$$PE_{(1, 2)} = \sin\left(\frac{1}{10000^{2/4}}\right) = \sin\left(\frac{1}{100}\right) \approx 0.01$$
$$PE_{(1, 3)} = \cos\left(\frac{1}{10000^{2/4}}\right) = \cos\left(\frac{1}{100}\right) \approx 1$$

Thus, the positional encoding vector for position 1 will be approximately [0.84, 0.54, 0.01, 1].

- Final input for the first encoder layer: $X_{embedded} = X + PE$.

```Python
import torch
import math

def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Description:
        Generation of positional encodings according to the formula from the original Transformer paper.

    Args:
        max_len: Maximum sequence length.
        d_model: Model dimension (number of features).

    Returns:
        Tensor of positional encodings with shape (max_len, d_model).

    Examples:
        >>> pe = positional_encoding(10, 512)
        >>> pe.shape
        torch.Size([10, 512])
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def print_embeddings(tensor: torch.Tensor, tokens: list, title: str, max_elements: int = 3) -> None:
    """
    Description:
        Visualization of embeddings with token labels.

    Args:
        tensor: Tensor of embeddings.
        tokens: List of tokens for visualization.
        title: Title for output.
        max_elements: Maximum number of elements to display.

    Returns:
        None

    Examples:
        >>> embeddings = torch.randn(5, 512)
        >>> tokens = ["token1", "token2", "token3", "token4", "token5"]
        >>> print_embeddings(embeddings, tokens, "Example embeddings")
    """
    print(f"\n{title}:")
    for idx, (vec, token) in enumerate(zip(tensor, tokens)):
        elements = vec[:max_elements].detach().numpy().round(4)
        print(f"{idx:2d} {token:15}: [{', '.join(f'{x:7.4f}' for x in elements)}...]")

# Example data
sentences = [
    "Hello everyone! I am interested in artificial intelligence",
    "Hello, how are you?",
    "AI is interesting!"
]

# Model parameters
N = len(sentences)  # Batch size
L = 9               # Maximum sequence length
D_model = 512       # Embedding dimension

# 1. Tokenization with padding
batch_tokens = [
    ["[CLS]", "Hello", "everyone", "!", "I", "am", "interested", "in", "artificial", "intelligence", "[SEP]"],  # L=11
    ["[CLS]", "Hello", ",", "how", "are", "you", "?", "[SEP]", "[PAD]", "[PAD]"],                      # L=11 (with padding)
    ["[CLS]", "AI", "—", "is", "interesting", "!", "[SEP]", "[PAD]", "[PAD]"]                      # L=11 (with padding)
]

# 2. Create embeddings
embeddings = torch.randn(N, L, D_model)

# 3. Generate positional encodings
pe = positional_encoding(L, D_model)

# 4. Combine embeddings with positional encodings
X_embedded = embeddings + pe  # Broadcasting for batch

# ================= Visualization of the process =================
print("="*60)
print("Step 1: Input embeddings")
print(f"Shape of tensor: {embeddings.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(embeddings[batch_idx], batch_tokens[batch_idx], "Input embeddings")

print("\n" + "="*60)
print("Step 2: Positional encodings")
print(f"Shape of tensor: {pe.shape}")
print_embeddings(pe, [f"Position {i}" for i in range(L)], "Example encodings")

print("\n" + "="*60)
print("Step 3: Combined embeddings (X + PE)")
print(f"Shape of tensor: {X_embedded.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(X_embedded[batch_idx], batch_tokens[batch_idx], "Result of addition")
```

```Python
# ================= Detailed output for the first batch =================
print("\n" + "="*60)
print("Detailed analysis of the first batch:")
batch_idx = 0

# Original data
print(f"\nText: '{sentences[batch_idx]}'")
print(f"Tokens: {batch_tokens[batch_idx]}")

# Comparison for key positions
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Original embedding:  {embeddings[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Positional encoding: {pe[pos, :3].detach().numpy().round(4)}")
    print(f"Combined:           {X_embedded[batch_idx, pos, :3].detach().numpy().round(4)}")
```

### 3. Multi-Head Self-Attention:
   - The input to the Multi-Head Attention sublayer is $X_{embedded} = X + PE$.
   - **Linear projections:** The input $X_{embedded}$ is linearly projected into queries $Q$, keys $K$, and values $V$ for each head:
     $$
     Q_i = X_{embedded} W_{Q_i}, \quad K_i = X_{embedded} W_{K_i}, \quad V_i = X_{embedded} W_{V_i}
     $$
     
      where:
      
      - $W_{Q_i} \in \mathbb{R}^{D_{model} \times D_k}$
      - $W_{K_i} \in \mathbb{R}^{D_{model} \times D_k}$
      - $W_{V_i} \in \mathbb{R}^{D_{model} \times D_v}$ - weight matrices for the $i$-th attention head
      - $D_k$ - dimension of keys and queries
      - $D_v$ - dimension of values
      
      Usually $D_k = D_v = D_{model} / h$, where $h$ is the number of heads.

      From a linear algebra perspective, linear projection is a linear transformation that maps a vector from one vector space to another. In the context of multi-head attention, the input vector $X_{embedded}$, belonging to a $D_{model}$-dimensional space, undergoes linear transformations to create three new vectors: $Q_i$, $K_i$, and $V_i$. These vectors reside in their own subspaces.

      **Formal description:**

      1. **$X_{embedded} \in \mathbb{R}^{D_{model}}$:** Input vector in $D_{model}$-dimensional space.
      2. **$W_{Q_i} \in \mathbb{R}^{D_{model} \times D_k}$, $W_{K_i} \in \mathbb{R}^{D_{model} \times D_k}$, $W_{V_i} \in \mathbb{R}^{D_{model} \times D_v}$:** Weight matrices defining linear mappings.
      3. **Linear mappings (projections):**
         - $Q_i = X_{embedded} W_{Q_i}$: Projection of $X_{embedded}$ into $D_k$-dimensional query subspace.
         - $K_i = X_{embedded} W_{K_i}$: Projection of $X_{embedded}$ into $D_k$-dimensional key subspace.
         - $V_i = X_{embedded} W_{V_i}$: Projection of $X_{embedded}$ into $D_v$-dimensional value subspace.

      **Why is this needed?**

      - **Separation into subspaces:** Linear projections create separate subspaces for queries, keys, and values. This allows the model to process input data from different perspectives.
      - **Specialization:** Each subspace has its own role: queries search for relevant keys, and values are used for information aggregation.
      - **Trainability:** Weight matrices $W_{Q_i}$, $W_{K_i}$, and $W_{V_i}$ are trainable parameters, allowing the model to adapt to specific tasks.
      - **Multi-head:** Using multiple heads (different sets of weight matrices) allows the model to simultaneously consider different subspaces, enhancing its effectiveness.

      Thus, linear projection in the multi-head attention mechanism is a way to transform input data into different subspaces, each with its own role in the information processing. This is achieved by applying linear mappings defined by trainable weight matrices.

   - **Attention for each head:** Calculated as a weighted sum of values, where weights are determined by the softmax function of the dot product of queries and keys:
     $$
     Z_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{D_k}}\right) V_i
     $$

   - **Softmax function:**

      **Definition:** The softmax function is a function that transforms a vector of real numbers into a vector of probabilities. It takes an input vector $z = [z_1, z_2, ..., z_n]$ and returns a vector $\sigma(z) = [\sigma(z_1), \sigma(z_2), ..., \sigma(z_n)]$, where each element $\sigma(z_i)$ is calculated by the formula:

      $$
      \sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
      $$

      where:
      - $z_i$ — this is the $i$-th element of the input vector $z$.
      - $e$ — this is the base of the natural logarithm (approximately 2.71828).
      - $n$ — this is the dimension of the vector $z$.

      **Properties of Softmax:**
      - **Normalization:** The sum of all elements in the output vector $\sigma(z)$ is 1: $\sum_{i=1}^{n} \sigma(z_i) = 1$.
      - **Probabilities:** Each element of the output vector $\sigma(z_i)$ lies in the range from 0 to 1: $0 \leq \sigma(z_i) \leq 1$.
      - **Transformation:** Softmax transforms arbitrary real numbers into probabilities, making it useful for classification and attention tasks.

   - **Softmax in Attention Mechanism:**

      In the attention mechanism, softmax is used to calculate attention weights, which show how important each token is when calculating the contextualized representation. In the expression $\text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{D_k}}\right) V_i$:

      - $Q_i$ — this is the matrix of queries for the $i$-th head.
      - $K_i$ — this is the matrix of keys for the $i$-th head.
      - $V_i$ — this is the matrix of values for the $i$-th head.
      - $D_k$ — this is the dimension of keys and queries.

      **Calculation of attention weights:**
      1. **Dot product:** The dot product of queries and keys is calculated: $Q_i K_i^T$. This gives a matrix where each element shows how "compatible" the query of one token is with the key of another.
      2. **Scaling:** The result of the dot product is divided by $\sqrt{D_k}$. This scaling helps stabilize training, preventing very large values that could cause gradient problems.
      3. **Softmax:** The softmax function is applied to the scaled result. This transforms the values into probabilities, which represent attention weights.

   - **Concatenation of heads:** The outputs of all heads are concatenated:
      $$
      \text{Concat}(Z_1, Z_2, ..., Z_h)
      $$
      where $Z_i = \text{Attention}(Q_i, K_i, V_i)$.
   - **Linear projection of output:** The result of concatenation is projected back into the $D_{model}$ space:
      $$
      \text{MultiHead}(Q, K, V) = \text{Concat}(Z_1, Z_2, ..., Z_h) W^O
      $$
      where $W^O \in \mathbb{R}^{h D_v \times D_{model}}$ - weight matrix for output projection.

### Let's take a closer look at how softmax works inside Multi-Head Attention using the code example below.

Now let's move on to the most interesting part — Multi-Head Self-Attention, where softmax is used.

**1. Linear projections:**

  - The input to Multi-Head Attention is the combined embeddings (shape `torch.Size([3, 9, 512])`).
  - For each head (in your example, there are 8), the input data is linearly projected into three matrices:
    - **Q (queries):** `torch.Size([3, 8, 9, 64])`
    - **K (keys):** `torch.Size([3, 8, 9, 64])`
    - **V (values):** `torch.Size([3, 8, 9, 64])`
  - These projections are performed using trainable weight matrices $W_Q$, $W_K$, and $W_V$.
  - In your example, for the first position of the first batch (token `[CLS]`), you see examples of matrices Q, K, and V.

  **Note!**

```
# This is before the Multi-Head Attention sublayer!

Position 0 ([CLS]):
Original embedding X:  [ 0.9007 -2.1055  0.6784]
Positional encoding PE:  [0. 1. 0.]
Combined X + PE:  [ 0.9007 -1.1055  0.6784]

# This is after Multi-Head Attention!
  Position 0 ([CLS]):
    Q: [[ 17.2809  14.7748 -11.5202]
[ -7.0419 -29.7085  54.2687]
[ 29.6011  11.744   -0.8485]
[ -6.7934 -25.4389  67.2095]
[ 28.5684 -16.4333 -13.8622]
[ -4.5139 -61.0905  -0.8532]
[-24.339   -9.4282  -5.367 ]
[  7.8296 -14.4175  16.9908]]
    K: [[  6.637  -10.3847 -29.3882]
[-22.8757 -18.3149 -65.0343]
[ 18.4063  29.4638 -34.1548]
[  5.1229   5.5592  66.0818]
[  9.9801 -20.4229  -7.4216]
[-22.0776   4.2677 -32.6255]
[-40.8423  19.4702   0.3407]
[  8.7071  27.0544 -13.8258]]
    V: [[ 25.8466  12.3776  -7.7585]
[ 32.6146  -2.0634  32.7602]
[ 25.009   11.0889  28.2676]
[ 19.9813 -11.8157  22.7189]
[ 12.4848   6.3136 -28.9884]
[ -1.6635  15.4315 -23.0705]
[ 14.6102  -1.6098 -15.4584]
[ -9.4451 -38.6892  42.4362]]
```

*   **"Q for CLS token" - this is a set of vectors in the form of a matrix.**
*   **The number of vectors equals the number of attention heads** (in your example, 8).
*   **Each vector corresponds to a separate attention head.**
*   **Each vector is obtained by linear projection:** The combined embedding of the CLS token **is multiplied by the weight matrix $W_q$** of the specific attention head.

**2. Calculation of attention weights:**

  - For each head and for each position in the sequence, attention weights are calculated.
  - **Dot product:** First, the dot product of queries and keys is calculated: $Q_i K_i^T$.
    - In your example, for the first head and the first position (token `[CLS]`), this will be the dot product of the query vector of `[CLS]` with the transpose of the key vector of each position in the sequence.
    - The result will be a matrix of size (9, 9), where each element shows how "compatible" the query of `[CLS]` is with the key of each of the 9 tokens in the sequence.

    <div style="border: 1px solid #000; padding: 10px; margin: 10px;">
    
    **Purpose of the dot product:**

    The main purpose of the dot product at this stage is to **determine how "compatible" or "relevant" the query (Q) of one token is to the keys (K) of all other tokens in the sequence.** As a result, we get "raw" attention weights, which are then normalized by Softmax.

    **What happens for token `[CLS]` (first position) and the first head:**

    1.  **Take the query vector Q for `[CLS]` from the first head:**
        *   In your example for position 0 ([CLS]) and the first head (first row in the Q block) the vector is shown (only the first 3 elements are shown): `[ 17.2809  14.7748 -11.5202 ...]`. Actually, this is a vector of size 64. Let's denote it as `Q_cls_head1`.

    2.  **Take the key vectors K for ALL positions (from 0 to 8) from the first head:**
        *   For each position from 0 to 8 in the sequence (tokens `[CLS]`, `Всем`, `привет`, `!`, `Я`, `увлекаюсь`, `искусственным`, `интеллектом`, `[SEP]`) there is a corresponding key vector K from the first head. In your example for position 0 ([CLS]) and the first head (first row in the K block) the vector is shown: `[  6.637  -10.3847 -29.3882 ...]`. Let's denote these key vectors as `K_pos0_head1`, `K_pos1_head1`, `K_pos2_head1`, ..., `K_pos8_head1`. Each of these is also a vector of size 64.

    3.  **Calculate the dot product between `Q_cls_head1` and each key vector `K_pos_j_head1`:**
        *   For each position `j` from 0 to 8 we calculate the dot product:
            *   `score_0 = Q_cls_head1 * (K_pos0_head1)^T`  (attention `[CLS]` on `[CLS]`)
            *   `score_1 = Q_cls_head1 * (K_pos1_head1)^T`  (attention `[CLS]` on `Всем`)
            *   `score_2 = Q_cls_head1 * (K_pos2_head1)^T`  (attention `[CLS]` on `привет`)
            *   ...
            *   `score_8 = Q_cls_head1 * (K_pos8_head1)^T`  (attention `[CLS]` on `[SEP]`)

        *   **Each dot product `score_j` — this is a single number (scalar).** It shows how "compatible" the query of `[CLS]` is with the key of the token at position `j`. The higher the value `score_j`, the more attention (yet "raw") the token `[CLS]` should pay to the token at position `j`.

        In our example for position [CLS]:
          ```python
          Q: [[-22.9001 -31.6346  6.0742]    # vector of head 0
              [ 41.4631   5.2998  6.2346]    # vector of head 1
              [ 29.6049 -43.8211 -13.9067]   # vector of head 2
              [ -9.0778  17.0357  -0.9468]   # vector of head 3
              [ 19.0137  -6.5111 -15.9635]   # vector of head 4
              [-42.3292 -31.1711  -1.0993]   # vector of head 5
              [ 22.1916 -19.8376  24.6427]   # vector of head 6
              [-11.865  -57.7867 -35.5895]]  # vector of head 7
          ```

          Process of calculation:
          
            - For head 0:
              * Take the Q vector of head 0: [-22.9001 -31.6346 6.0742]
              * Multiply it by all K vectors ONLY of head 0
            - For head 1:
              * Take the Q vector of head 1: [41.4631 5.2998 6.2346]
              * Multiply it by all K vectors ONLY of head 1
            - And so on for each head

    4.  **Result - a series of scalar values:**
        *   As a result of these 9 dot products, we get a series of numbers: `[score_0, score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8]`.
        *   **This series of scalar values (after scaling and Softmax) will become the attention weights for token `[CLS]` in the first head.** In your example, after Softmax, these weights became `[1. 0. 0. 0. 0. 0. 0. 0. 0.]`.
      </div>

  - **Scaling:** The result of the dot product is divided by $\sqrt{D_k}$, where $D_k$ is the dimension of keys and queries (in your example $D_k = 64$). This scaling helps stabilize training, preventing very large values that could cause gradient problems.
  - **Softmax:** The softmax function is applied to the scaled result.
    - Softmax transforms the values into probabilities, which represent attention weights.
    - Softmax is applied to each row of the matrix (9, 9), i.e., for each token in the sequence, the attention weights relative to all other tokens are calculated.
    - **Important:** Softmax normalizes the weights so that their sum is 1. This means that the attention weights show how important each token is when calculating the contextualized representation of the current token.
    - In your example, for the first head and the first position, you see that the attention weights are `[1. 0. 0. 0. 0. 0. 0. 0. 0.]`. This means that when calculating the contextualized representation of token `[CLS]` (first position), the maximum attention is paid to the token `[CLS]` itself, and the other tokens have no value.

**3. Weighted value summation:**

  - After calculating the attention weights, they are used to weight the values.
  - The attention weights are multiplied by the corresponding values.
  - The result is the weighted sum of values, which represents the contextualized representation of the token.
  - In your example, for the first head and the first position, you see the matrix Z, which is the result of the weighted summation.

**4. Concatenation of heads:**

  - The outputs of all heads are concatenated:
      $$
      \text{Concat}(Z_1, Z_2, ..., Z_h)
      $$
      where $Z_i = \text{Attention}(Q_i, K_i, V_i)$.
   - **Linear projection of output:** The result of concatenation is projected back into the $D_{model}$ space:
      $$
      \text{MultiHead}(Q, K, V) = \text{Concat}(Z_1, Z_2, ..., Z_h) W^O
      $$
      where $W^O \in \mathbb{R}^{h D_v \times D_{model}}$ - output projection weight matrix.

### Let's take a closer look at how softmax works inside Multi-Head Attention using the code example below.

Now let's move on to the most interesting part — Multi-Head Self-Attention, where softmax is used.

**1. Linear projections:**

  - The input to Multi-Head Attention is the combined embeddings (shape `torch.Size([3, 9, 512])`).
  - For each head (in your example, there are 8), the input data is linearly projected into three matrices:
    - **Q (queries):** `torch.Size([3, 8, 9, 64])`
    - **K (keys):** `torch.Size([3, 8, 9, 64])`
    - **V (values):** `torch.Size([3, 8, 9, 64])`
  - These projections are performed using trainable weight matrices $W_Q$, $W_K$, and $W_V$.
  - In your example, for the first position of the first batch (token `[CLS]`), you see examples of matrices Q, K, and V.

  **Note!**

```
# This is before the Multi-Head Attention sublayer!

Position 0 ([CLS]):
Original embedding X:  [ 0.9007 -2.1055  0.6784]
Positional encoding PE:  [0. 1. 0.]
Combined X + PE:  [ 0.9007 -1.1055  0.6784]

# This is after Multi-Head Attention!
  Position 0 ([CLS]):
    Q: [[ 17.2809  14.7748 -11.5202]
[ -7.0419 -29.7085  54.2687]
[ 29.6011  11.744   -0.8485]
[ -6.7934 -25.4389  67.2095]
[ 28.5684 -16.4333 -13.8622]
[ -4.5139 -61.0905  -0.8532]
[-24.339   -9.4282  -5.367 ]
[  7.8296 -14.4175  16.9908]]
    K: [[  6.637  -10.3847 -29.3882]
[-22.8757 -18.3149 -65.0343]
[ 18.4063  29.4638 -34.1548]
[  5.1229   5.5592  66.0818]
[  9.9801 -20.4229  -7.4216]
[-22.0776   4.2677 -32.6255]
[-40.8423  19.4702   0.3407]
[  8.7071  27.0544 -13.8258]]
    V: [[ 25.8466  12.3776  -7.7585]
[ 32.6146  -2.0634  32.7602]
[ 25.009   11.0889  28.2676]
[ 19.9813 -11.8157  22.7189]
[ 12.4848   6.3136 -28.9884]
[ -1.6635  15.4315 -23.0705]
[ 14.6102  -1.6098 -15.4584]
[ -9.4451 -38.6892  42.4362]]
```

*   **"Q for CLS token" - this is a set of vectors in the form of a matrix.**
*   **The number of vectors equals the number of attention heads** (in your example, 8).
*   **Each vector corresponds to a separate attention head.**
*   **Each vector is obtained by linear projection:** The combined embedding of the CLS token **is multiplied by the weight matrix $W_q$** of the specific attention head.

**2. Calculation of attention weights:**

  - For each head and for each position in the sequence, attention weights are calculated.
  - **Dot product:** First, the dot product of queries and keys is calculated: $Q_i K_i^T$.
    - In your example, for the first head and the first position (token `[CLS]`), this will be the dot product of the query vector of `[CLS]` with the transpose of the key vector of each position in the sequence.
    - The result will be a matrix of size (9, 9), where each element shows how "compatible" the query of `[CLS]` is with the key of each of the 9 tokens in the sequence.

    <div style="border: 1px solid #000; padding: 10px; margin: 10px;">
    
    **Purpose of the dot product:**

    The main purpose of the dot product at this stage is to **determine how "compatible" or "relevant" the query (Q) of one token is to the keys (K) of all other tokens in the sequence.** As a result, we get "raw" attention weights, which are then normalized by Softmax.

    **What happens for token `[CLS]` (first position) and the first head:**

    1.  **Take the query vector Q for `[CLS]` from the first head:**
        *   In your example for position 0 ([CLS]) and the first head (first row in the Q block) the vector is shown (only the first 3 elements are shown): `[ 17.2809  14.7748 -11.5202 ...]`. Actually, this is a vector of size 64. Let's denote it as `Q_cls_head1`.

    2.  **Take the key vectors K for ALL positions (from 0 to 8) from the first head:**
        *   For each position from 0 to 8 in the sequence (tokens `[CLS]`, `Всем`, `привет`, `!`, `Я`, `увлекаюсь`, `искусственным`, `интеллектом`, `[SEP]`) there is a corresponding key vector K from the first head. In your example for position 0 ([CLS]) and the first head (first row in the K block) the vector is shown: `[  6.637  -10.3847 -29.3882 ...]`. Let's denote these key vectors as `K_pos0_head1`, `K_pos1_head1`, `K_pos2_head1`, ..., `K_pos8_head1`. Each of these is also a vector of size 64.

    3.  **Calculate the dot product between `Q_cls_head1` and each key vector `K_pos_j_head1`:**
        *   For each position `j` from 0 to 8 we calculate the dot product:
            *   `score_0 = Q_cls_head1 * (K_pos0_head1)^T`  (attention `[CLS]` on `[CLS]`)
            *   `score_1 = Q_cls_head1 * (K_pos1_head1)^T`  (attention `[CLS]` on `Всем`)
            *   `score_2 = Q_cls_head1 * (K_pos2_head1)^T`  (attention `[CLS]` on `привет`)
            *   ...
            *   `score_8 = Q_cls_head1 * (K_pos8_head1)^T`  (attention `[CLS]` on `[SEP]`)

        *   **Each dot product `score_j` — this is a single number (scalar).** It shows how "compatible" the query of `[CLS]` is with the key of the token at position `j`. The higher the value `score_j`, the more attention (yet "raw") the token `[CLS]` should pay to the token at position `j`.

        In our example for position [CLS]:
          ```python
          Q: [[-22.9001 -31.6346  6.0742]    # vector of head 0
              [ 41.4631   5.2998  6.2346]    # vector of head 1
              [ 29.6049 -43.8211 -13.9067]   # vector of head 2
              [ -9.0778  17.0357  -0.9468]   # vector of head 3
              [ 19.0137  -6.5111 -15.9635]   # vector of head 4
              [-42.3292 -31.1711  -1.0993]   # vector of head 5
              [ 22.1916 -19.8376  24.6427]   # vector of head 6
              [-11.865  -57.7867 -35.5895]]  # vector of head 7
          ```

          Process of calculation:
          
            - For head 0:
              * Take the Q vector of head 0: [-22.9001 -31.6346 6.0742]
              * Multiply it by all K vectors ONLY of head 0
            - For head 1:
              * Take the Q vector of head 1: [41.4631 5.2998 6.2346]
              * Multiply it by all K vectors ONLY of head 1
            - And so on for each head

    4.  **Result - a series of scalar values:**
        *   As a result of these 9 dot products, we get a series of numbers: `[score_0, score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8]`.
        *   **This series of scalar values (after scaling and Softmax) will become the attention weights for token `[CLS]` in the first head.** In your example, after Softmax, these weights became `[1. 0. 0. 0. 0. 0. 0. 0. 0.]`.
      </div>

  - **Scaling:** The result of the dot product is divided by $\sqrt{D_k}$, where $D_k$ is the dimension of keys and queries (in your example $D_k = 64$). This scaling helps stabilize training, preventing very large values that could cause gradient problems.
  - **Softmax:** The softmax function is applied to the scaled result.
    - Softmax transforms the values into probabilities, which represent attention weights.
    - Softmax is applied to each row of the matrix (9, 9), i.e., for each token in the sequence, the attention weights relative to all other tokens are calculated.
    - **Important:** Softmax normalizes the weights so that their sum is 1. This means that the attention weights show how important each token is when calculating the contextualized representation of the current token.
    - In your example, for the first head and the first position, you see that the attention weights are `[1. 0. 0. 0. 0. 0. 0. 0. 0.]`. This means that when calculating the contextualized representation of token `[CLS]` (first position), the maximum attention is paid to the token `[CLS]` itself, and the other tokens have no value.

**3. Weighted value summation:**

  - After calculating the attention weights, they are used to weight the values.
  - The attention weights are multiplied by the corresponding values.
  - The result is the weighted sum of values, which represents the contextualized representation of the token.
  - In your example, for the first head and the first position, you see the matrix Z, which is the result of the weighted summation.

**4. Concatenation of heads:**

  - The outputs of all heads are concatenated:
      $$
      \text{Concat}(Z_1, Z_2, ..., Z_h)
      $$
      where $Z_i = \text{Attention}(Q_i, K_i, V_i)$.
   - **Linear projection of output:** The result of concatenation is projected back into the $D_{model}$ space:
      $$
      \text{MultiHead}(Q, K, V) = \text{Concat}(Z_1, Z_2, ..., Z_h) W^O
      $$
      where $W^O \in \mathbb{R}^{h D_v \times D_{model}}$ - output projection weight matrix.

**Influence of Softmax:**

- Softmax plays a key role in the attention mechanism, transforming the results of the dot product of queries and keys into probabilities.
- These probabilities (attention weights) show how important each token is when calculating the contextualized representation of the current token.
- Softmax ensures that the sum of attention weights is 1, allowing the model to effectively distribute attention among different tokens.

**In summary:**

Softmax within Multi-Head Attention allows the model to dynamically determine which parts of the input sequence to focus on when processing each token. This makes the model more flexible and capable of capturing complex dependencies in the data.

```python
# Standard libraries
import math

# Third-party libraries
import numpy as np
import torch


def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Description:
        Generation of positional encodings according to the formula from the original Transformer paper.

    Args:
        max_len: Maximum sequence length.
        d_model: Model dimension (number of features).

    Returns:
        Tensor of positional encodings with shape (max_len, d_model).

    Examples:
        >>> pe = positional_encoding(10, 512)
        >>> pe.shape
        torch.Size([10, 512])
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def print_embeddings(
    tensor: torch.Tensor, tokens: list, title: str, max_elements: int = 3
) -> None:
    """
    Description:
        Visualization of embeddings with token labels.

    Args:
        tensor: Tensor of embeddings.
        tokens: List of tokens for visualization.
        title: Title for output.
        max_elements: Maximum number of elements to display.

    Returns:
        None

    Examples:
        >>> embeddings = torch.randn(5, 512)
        >>> tokens = ["token1", "token2", "token3", "token4", "token5"]
        >>> print_embeddings(embeddings, tokens, "Example embeddings")
    """
    print(f"\n{title}:")
    for idx, (vec, token) in enumerate(zip(tensor, tokens)):
        elements = vec[:max_elements].detach().numpy().round(4)
        print(f"{idx:2d} {token:15}: [{', '.join(f'{x:7.4f}' for x in elements)}...]")


def print_attention_details(
    batch_idx: int,
    head_idx: int,
    pos_idx: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    attention_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    tokens: list,
    num_elements: int = 5,
) -> None:
    """
    Description:
        Output detailed information about the attention process for a specific position.

    Args:
        batch_idx: Batch index.
        head_idx: Head index.
        pos_idx: Position in the sequence.
        Q: Query tensor.
        K: Key tensor.
        attention_scores: Tensor of raw attention scores.
        attention_weights: Tensor of attention weights after softmax.
        tokens: List of tokens.
        num_elements: Number of elements to output.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print(
        f"Attention details for batch {batch_idx}, head {head_idx}, "
        f"position {pos_idx} ({tokens[batch_idx][pos_idx]}):"
    )

    # Output Q vector
    q_vec = Q[batch_idx, head_idx, pos_idx, :num_elements].detach().numpy()
    print(f"Q vector (first {num_elements} elements):")
    print(f"{q_vec.round(4)}")

    # Output K vectors
    print(f"K vectors (first {num_elements} elements of each):")
    for i, token in enumerate(tokens[batch_idx]):
        k_vec = K[batch_idx, head_idx, i, :num_elements].detach().numpy()
        print(f"{i:2d} {token:15}: {k_vec.round(4)}")

    # Manual calculation of dot products
    manual_scores = []
    q = Q[batch_idx, head_idx, pos_idx]
    for i in range(len(tokens[batch_idx])):
        k = K[batch_idx, head_idx, i]
        score = torch.dot(q, k) / math.sqrt(D_k)
        manual_scores.append(score.item())

    # Get automatically calculated scores
    auto_scores = attention_scores[batch_idx, head_idx, pos_idx].detach().numpy()

    # Compare results
    print("\nRaw attention scores:")
    print(f"Manual calculation: {np.array(manual_scores).round(4)}")
    print(f"Automatic:          {auto_scores.round(4)}")

    # Output weights after softmax
    weights = attention_weights[batch_idx, head_idx, pos_idx].detach().numpy()
    print(f"\nAttention weights after Softmax:")
    print(f"{weights.round(4)}")


# Example data
sentences = [
    "Hello everyone! I am interested in artificial intelligence",
    "Hello, how are you?",
    "AI is interesting!"
]

# Model parameters
N = len(sentences)  # Batch size
L = 9               # Maximum sequence length
D_model = 512       # Embedding dimension
h = 8               # Number of heads
D_k = D_model // h  # Dimension of keys and queries
D_v = D_model // h  # Dimension of values

# 1. Tokenization with padding
batch_tokens = [
    ["[CLS]", "Hello", "everyone", "!", "I", "am", "interested", "in", "artificial", "intelligence", "[SEP]"],  # L=11
    ["[CLS]", "Hello", ",", "how", "are", "you", "?", "[SEP]", "[PAD]", "[PAD]"],                      # L=11 (with padding)
    ["[CLS]", "AI", "—", "is", "interesting", "!", "[SEP]", "[PAD]", "[PAD]"]                      # L=11 (with padding)
]

# 2. Create embeddings
embeddings = torch.randn(N, L, D_model)

# 3. Generate positional encodings
pe = positional_encoding(L, D_model)

# 4. Combine embeddings with positional encodings
X_embedded = embeddings + pe  # Broadcasting for batch

# ================= Visualization of the process =================
print("=" * 60)
print("Step 1: Input embeddings")
print(f"Shape of tensor: {embeddings.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(embeddings[batch_idx], batch_tokens[batch_idx], "Input embeddings")

print("\n" + "=" * 60)
print("Step 2: Positional encodings")
print(f"Shape of tensor: {pe.shape}")
print_embeddings(pe, [f"Position {i}" for i in range(L)], "Example encodings")

print("\n" + "=" * 60)
print("Step 3: Combined embeddings (X + PE)")
print(f"Shape of tensor: {X_embedded.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(X_embedded[batch_idx], batch_tokens[batch_idx], "Result of addition")

# ================= Detailed output for the first batch =================
print("\n" + "=" * 60)
print("Detailed analysis of the first batch:")
batch_idx = 0

# Original data
print(f"\nText: '{sentences[batch_idx]}'")
print(f"Tokens: {batch_tokens[batch_idx]}")

# Comparison for key positions
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Original embedding:  {embeddings[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Positional encoding: {pe[pos, :3].detach().numpy().round(4)}")
    print(f"Combined:           {X_embedded[batch_idx, pos, :3].detach().numpy().round(4)}")

# ================= Multi-Head Self-Attention =================
print("\n" + "=" * 60)
print("Step 4: Multi-Head Self-Attention")

# 1. Linear projections
W_Q = torch.randn(h, D_model, D_k)
W_K = torch.randn(h, D_model, D_k)
W_V = torch.randn(h, D_model, D_v)

Q = torch.einsum('nlk,hkd->nhld', X_embedded, W_Q)
K = torch.einsum('nlk,hkd->nhld', X_embedded, W_K)
V = torch.einsum('nlk,hkd->nhld', X_embedded, W_V)

print("\nLinear projections:")
print(f"Shape of Q: {Q.shape}")
print(f"Shape of K: {K.shape}")
print(f"Shape of V: {V.shape}")

# Output first 3 elements for Q, K, V
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    for pos in [0, 2, 4]:
        print(f"  Position {pos} ({batch_tokens[batch_idx][pos]}):")
        print(f"    Q: {Q[batch_idx, :, pos, :3].detach().numpy().round(4)}")
        print(f"    K: {K[batch_idx, :, pos, :3].detach().numpy().round(4)}")
        print(f"    V: {V[batch_idx, :, pos, :3].detach().numpy().round(4)}")

# After calculating attention_weights add:
print_attention_details(
    batch_idx=0,
    head_idx=0,
    pos_idx=0,
    Q=Q,
    K=K,
    attention_scores=attention_scores,
    attention_weights=attention_weights,
    tokens=batch_tokens,
)

# 2. Attention for each head
attention_scores = torch.einsum('nhld,nhmd->nhlm', Q, K) / math.sqrt(D_k)

# Masking padding
mask = torch.ones(N, 1, L, L, dtype=torch.bool)
for batch_idx, tokens in enumerate(batch_tokens):
    for i, token in enumerate(tokens):
        if token == "[PAD]":
            mask[batch_idx, :, i:, :] = False
            mask[batch_idx, :, :, i:] = False
attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

attention_weights = torch.softmax(attention_scores, dim=-1)
Z = torch.einsum('nhlm,nhmd->nhld', attention_weights, V)

print("\nAttention for each head:")
print(f"Shape of Z: {Z.shape}")

# Output first 3 elements for Z
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    for pos in [0, 2, 4]:
        print(f"  Position {pos} ({batch_tokens[batch_idx][pos]}):")
        print(f"    Z: {Z[batch_idx, :, pos, :3].detach().numpy().round(4)}")

# Visualization of attention weights for the first head and first position
print("\nVisualization of attention weights for the first head and first position:")
print(f"Attention weights (first head, first position): {attention_weights[0, 0, 0, :].detach().numpy().round(4)}")

# 3. Concatenation of heads
Z_concat = Z.transpose(1, 2).reshape(N, L, h * D_v)

print("\nConcatenation of heads:")
print(f"Shape of Z_concat: {Z_concat.shape}")

# 4. Linear projection of output
W_O = torch.randn(h * D_v, D_model)
multi_head_output = torch.einsum('nlk,kd->nld', Z_concat, W_O)

print("\nLinear projection of output:")
print(f"Shape of MultiHead Output: {multi_head_output.shape}")

# Output for the first batch
print("\nDetailed analysis of the first batch after Multi-Head Attention:")
batch_idx = 0
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Combined:     {X_embedded[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Multi-Head Output: {multi_head_output[batch_idx, pos, :3].detach().numpy().round(4)}")
```

4. **Add & Norm layer (after Multi-Head Attention):**
   - **Add (residual connection):**

        *   **Essence of residual connection:** A residual connection, also known as a skip-connection or residual connection, involves **adding the output of the Multi-Head Attention layer to its original input**.

            In formula, this looks like:
            $$
            \text{Output}_{Add1} = X_{embedded} + \text{MultiHead}(Q, K, V)
            $$
            where:
            *   $X_{embedded}$ - this is the **input** to the Multi-Head Attention sub-layer. In the context of Transformer, this could be the embeddings of the input tokens, possibly having passed through previous Transformer layers.
            *   $\text{MultiHead}(Q, K, V)$ - this is the **output** of the Multi-Head Attention layer.
            *   $\text{Output}_{Add1}$ - this is the **result of addition**, which becomes the input for the next step - the layer normalization.

            > So, essentially, it's just a regular addition from linear algebra of two matrices, or more precisely, two tensors of the same dimension.

        *   **Why are residual connections needed?**

            *   **Combatting the vanishing gradient problem:** In deep neural networks, such as Transformers, gradients (signals for training) can diminish as they propagate through many layers. Residual connections help **"jump over"** layers, providing a more direct path for gradients. This facilitates the training of deep networks and allows them to learn effectively.
            *   **Improving training of deep networks:** Residual connections allow for the training of **deeper and more complex models**. Without them, adding new layers to a deep network often does not improve performance, and may even degrade it. Residual connections allow for the effective use of the advantages of deep architectures.
            *   **Preserving input information:** By adding the original input to the output of the attention layer, we **preserve information about the original embeddings**. The attention layer focuses on *changes* and *refining* the input representations, and the residual connection ensures that the original information is not completely lost.

   - **Norm (layer normalization):**

        *   **Essence of layer normalization:** Layer normalization is a normalization technique applied **to the outputs of a neural network layer within a single training example**. Unlike Batch Normalization, which normalizes across a batch, Layer Normalization normalizes **across features within a single example**.

            Layer Normalization is used in Transformer. The formula looks like this:
            $$
            \text{Output}_{Norm1} = \text{LayerNorm}(\text{Output}_{Add1}) = \gamma \frac{\text{Output}_{Add1} - \mu}{\sigma} + \beta
            $$
            where:
            *   $\text{Output}_{Add1}$ - this is the **input** to the normalization layer, which is the result of the residual connection.
            *   $\mu$ - this is the **mean** of the elements of the input $\text{Output}_{Add1}$ **across the feature dimension** (for each example separately).
            *   $\sigma$ - this is the **standard deviation** of the elements of the input $\text{Output}_{Add1}$ **across the feature dimension** (for each example separately).
            *   $\gamma$ (gamma) and $\beta$ (beta) - these are **trainable scaling and shifting parameters**. They allow the network to **adjust** the degree of normalization and to restore the optimal value range after normalization. Initially, $\gamma$ is usually initialized to ones, and $\beta$ to zeros.
            *   $\text{Output}_{Norm1}$ - this is the **output** of the normalization layer, which becomes the input for the next sub-layer (in this case, the Feed Forward Network).

            > So, essentially, layer normalization, or more precisely, all the weights of the output matrix, is very similar to Z-score normalization (also known as standardization), which in statistics is used to transform data to a standard normal distribution. The formula for Layer Normalization we are considering:

            $$
            \text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta
            $$

        *   **Why is layer normalization needed?**

            *   **Stabilizing training:** Layer normalization **stabilizes the training process**, making it faster and more robust. It helps to **reduce internal covariate shift**, i.e., the change in the distribution of inputs to each layer during training. This happens because normalization brings the input data to a more standard range of values (close to zero mean and unit standard deviation).
            *   **Accelerating convergence:** Stabilizing training allows for the use of **higher learning rates** and **accelerates the convergence** of the model to the optimal solution.
            *   **Improving generalization:** Layer normalization can also contribute to **better generalization** of the model, i.e., its ability to perform well on new, previously unseen data.
            *   **Less dependence on batch size:** Unlike Batch Normalization, Layer Normalization **does not depend on batch size**. This makes it particularly useful in situations where the batch size is small or when using recurrent neural networks, where sequence length can vary.

            **Analogy:** Imagine you are adjusting the volume on different devices. Each device has its own volume range. Layer normalization is like **standardizing the volume across all devices**. This makes it easier to compare and process the sound, making the system more stable and predictable. The parameters $\gamma$ and $\beta$ allow for a slight "tuning" of this standard to account for the specific characteristics of each device.

5. **Feed Forward Network:**

    **Purpose of FFN:**

    FFN is a key component in each Transformer block, responsible for **non-linear transformation of token representations at the level of individual positions**. While Multi-Head Attention allows tokens to interact with each other and consider context, FFN processes the representation of each token **individually**, but already with the context obtained from the attention layer.

   - The input to the FFN sub-layer is $\text{Output}_{Norm1}$.
   - FFN consists of two linear layers with an activation function (e.g., ReLU, GeLU) between them:
     $$
     \text{FFN}(\text{Output}_{Norm1}) = \text{Activation}(\text{Output}_{Norm1} W_1 + b_1) W_2 + b_2
     $$
     where:

        - $W_1 \in \mathbb{R}^{D_{model} \times D_{ff}}$
        - $W_2 \in \mathbb{R}^{D_{ff} \times D_{model}}$ - weight matrices
        - $b_1 \in \mathbb{R}^{D_{ff}}$, $b_2 \in \mathbb{R}^{D_{model}}$ - bias vectors
        - $D_{ff}$ - internal dimension of FFN (usually $4 \times D_{model}$).

        **Output:** FFN transforms the input and outputs a tensor of the **same dimensionality** $(N, L, D_{model})$, where:

        - $N$ — batch size (number of examples in the batch),
        - $L$ — sequence length (number of tokens),
        - $D_{model}$ — hidden dimension (embedding dimension).

        This output is then passed to the next Transformer layer or used for the task (e.g., classification or text generation).

    **Structure of FFN:**

    FFN consists of **two sequential linear layers** with an **activation function** between them. This can be represented as a two-layer fully connected neural network applied to each position in the sequence.

    **Components of FFN and formula:**

    $$
    \text{FFN}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
    $$

    Let's break down each component of the formula:

    1.  **First linear layer (Expansion Layer):**  `(x W_1 + b_1)`
        *   **Input:**  $x$ - this is the input to FFN, i.e., $\text{Output}_{Norm1}$ of dimensionality `[batch_size, sequence_length, hidden_size]` ($D_{model}$).
        *   **Weight matrix $W_1$**:  $W_1 \in \mathbb{R}^{D_{model} \times D_{ff}}$ - this is the **weight matrix of the first linear layer**. It is a **trainable parameter**.
        *   **Bias vector $b_1$**: $b_1 \in \mathbb{R}^{D_{ff}}$ - this is the **bias vector of the first linear layer**. It is also a **trainable parameter**.
        *   **Internal dimension $D_{ff}$**: $D_{ff}$ - this is the **internal (intermediate) dimension of FFN**. It is usually **larger than $D_{model}$**, often 4 times larger ($D_{ff} = 4 \times D_{model}$). For example, if $D_{model} = 512$, then $D_{ff} = 2048$. This increase in dimensionality at this stage is called **"expansion" (expansion)**.
        *   **Operation:** A **linear transformation** of the input $x$ is performed by matrix multiplication with $W_1$ and adding the bias $b_1$.
        *   **Output of the first linear layer:** The result is a tensor of dimensionality `[batch_size, sequence_length, D_{ff}]`. The feature space dimensionality **increases** from $D_{model}$ to $D_{ff}$.

    2.  **Activation function (Activation Function):**  `Activation(...)`
        *   **Input:**  The output of the first linear layer of dimensionality `[batch_size, sequence_length, D_{ff}]`.
        *   **Activation function:**  $\text{Activation}$ - this is a **non-linear activation function**. In Transformer, the following are typically used:
            *   **ReLU (Rectified Linear Unit):**  $\text{ReLU}(z) = \max(0, z)$. A simple and efficient function that sets negative values to zero.
            *   **GeLU (Gaussian Error Linear Unit):** A smoother activation function that in some cases shows better results than ReLU. The formula for GeLU is slightly more complex, but the essence is that it introduces non-linearity.
        *   **Purpose of the activation function:** The activation function **introduces non-linearity** into the transformation. Without it, FFN would be just another linear layer, and the Transformer as a whole would be equivalent to a linear model, which would severely limit its expressiveness. Non-linearity allows the model to learn **complex, non-linear dependencies** in the data.
        *   **Output of the activation function:** The dimensionality of the tensor **does not change** after applying the activation function. The output still has dimensionality `[batch_size, sequence_length, D_{ff}]`.

    3.  **Second linear layer (Contraction Layer):**  `(... ) W_2 + b_2`
        *   **Input:**  The output of the activation function of dimensionality `[batch_size, sequence_length, D_{ff}]`.
        *   **Weight matrix $W_2$**:  $W_2 \in \mathbb{R}^{D_{ff} \times D_{model}}$ - this is the **weight matrix of the second linear layer**. It is also a **trainable parameter**.
        *   **Bias vector $b_2$**: $b_2 \in \mathbb{R}^{D_{model}}$ - this is the **bias vector of the second linear layer**. It is also a **trainable parameter**.
        *   **Operation:** A **linear transformation** of the output of the activation function is performed by matrix multiplication with $W_2$ and adding the bias $b_2$.
        *   **Output of the second linear layer (and FFN as a whole):** The result is a tensor of dimensionality `[batch_size, sequence_length, D_{model}]`. The feature space dimensionality **returns** to the original $D_{model}$. This is called **"contraction" (contraction)**.

    **Dimensions in FFN on an example:**

    Suppose $D_{model} = 512$ and $D_{ff} = 4 \times D_{model} = 2048$.

    1.  **Input $x$**:  `[batch_size, sequence_length, 512]`
    2.  **First linear layer $(x W_1 + b_1)$**:
        *   $W_1$ has dimensionality `[512, 2048]`
        *   Output: `[batch_size, sequence_length, 2048]` (dimensionality expanded)
    3.  **Activation function $\text{Activation}$**:
        *   Input: `[batch_size, sequence_length, 2048]`
        *   Output: `[batch_size, sequence_length, 2048]` (dimensionality does not change)
    4.  **Second linear layer $(... ) W_2 + b_2)$**:
        *   $W_2$ has dimensionality `[2048, 512]`
        *   Output: `[batch_size, sequence_length, 512]` (dimensionality contracted back to original)

    **Purpose of matrices $W_1$ and $W_2$:**

    *   **$W_1$ (expansion matrix):** The matrix $W_1$ is responsible for **projecting the input space of dimensionality $D_{model}$ into a wider space of dimensionality $D_{ff}$**. This allows FFN to **increase its expressiveness** and "remember" more information on the intermediate stage.
    *   **$W_2$ (contraction matrix):** The matrix $W_2$ is responsible for **projecting back from the space of dimensionality $D_{ff}$ to the original space of dimensionality $D_{model}$**. This is necessary so that the FFN output has the same dimensionality as the input, and can be integrated into the rest of the Transformer architecture. Also, the matrix $W_2$ allows for **mixing and aggregating information** obtained on the intermediate stage.

    **Why is FFN needed in Transformer?**

    *   **Introducing non-linearity:** FFN introduces **non-linearity** into the model, which is critical for learning complex dependencies in the data.
    *   **Processing information at the level of positions:** FFN is applied **independently to each position** in the sequence. This allows the model to perform **more complex, non-linear transformation** of each token's representation after the context has been considered by the attention layer.
    *   **Increasing model expressiveness:** By expanding the dimensionality to $D_{ff}$ and then contracting it back to $D_{model}$, FFN allows the model to **increase its expressiveness** and ability to learn more complex patterns. The intermediate space of larger dimensionality acts as a kind of "hidden space" where the model can more flexibly manipulate data representations.

```python
# Standard libraries
import math

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn  # Import nn module for LayerNorm


def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Description:
        Generation of positional encodings according to the formula from the original Transformer paper.

    Args:
        max_len: Maximum sequence length.
        d_model: Model dimension (number of features).

    Returns:
        Tensor of positional encodings with shape (max_len, d_model).

    Examples:
        >>> pe = positional_encoding(10, 512)
        >>> pe.shape
        torch.Size([10, 512])
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def print_embeddings(
    tensor: torch.Tensor, tokens: list, title: str, max_elements: int = 3
) -> None:
    """
    Description:
        Visualization of embeddings with token labels.

    Args:
        tensor: Tensor of embeddings.
        tokens: List of tokens for visualization.
        title: Title for output.
        max_elements: Maximum number of elements to display.

    Returns:
        None

    Examples:
        >>> embeddings = torch.randn(5, 512)
        >>> tokens = ["token1", "token2", "token3", "token4", "token5"]
        >>> print_embeddings(embeddings, tokens, "Example embeddings")
    """
    print(f"\n{title}:")
    for idx, (vec, token) in enumerate(zip(tensor, tokens)):
        elements = vec[:max_elements].detach().numpy().round(4)
        print(f"{idx:2d} {token:15}: [{', '.join(f'{x:7.4f}' for x in elements)}...]")


def print_attention_details(
    batch_idx: int,
    head_idx: int,
    pos_idx: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    attention_scores: torch.Tensor,
    attention_weights: torch.Tensor,
    tokens: list,
    num_elements: int = 5,
) -> None:
    """
    Description:
        Outputs detailed information about the attention process for a specific position.

    Args:
        batch_idx: Batch index.
        head_idx: Attention head index.
        pos_idx: Position in the sequence.
        Q: Tensor of queries.
        K: Tensor of keys.
        attention_scores: Tensor of raw attention scores.
        attention_weights: Tensor of attention weights after softmax.
        tokens: List of tokens.
        num_elements: Number of elements to output.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print(
        f"Attention details for batch {batch_idx}, head {head_idx}, "
        f"position {pos_idx} ({tokens[batch_idx][pos_idx]}):"
    )

    # Output Q vector
    q_vec = Q[batch_idx, head_idx, pos_idx, :num_elements].detach().numpy()
    print(f"Q vector (first {num_elements} elements):")
    print(f"{q_vec.round(4)}")

    # Output K vectors
    print(f"K vectors (first {num_elements} elements of each):")
    for i, token in enumerate(tokens[batch_idx]):
        k_vec = K[batch_idx, head_idx, i, :num_elements].detach().numpy()
        print(f"{i:2d} {token:15}: {k_vec.round(4)}")

    # Manual calculation of dot products
    manual_scores = []
    q = Q[batch_idx, head_idx, pos_idx]
    for i in range(len(tokens[batch_idx])):
        k = K[batch_idx, head_idx, i]
        score = torch.dot(q, k) / math.sqrt(D_k)
        manual_scores.append(score.item())

    # Getting automatically calculated scores
    auto_scores = attention_scores[batch_idx, head_idx, pos_idx].detach().numpy()

    # Comparing results
    print("\nRaw attention scores:")
    print(f"Manual calculation: {np.array(manual_scores).round(4)}")
    print(f"Automatic:          {auto_scores.round(4)}")

    # Output weights after softmax
    weights = attention_weights[batch_idx, head_idx, pos_idx].detach().numpy()
    print(f"\nAttention weights after Softmax:")
    print(f"{weights.round(4)}")


# Example data
sentences = [
    "Всем привет! Я увлекаюсь искусственным интеллектом",
    "Привет, как дела?",
    "ИИ — это интересно!",
]

# Model parameters
N = len(sentences)  # Batch size
L = 9               # Maximum sequence length
D_model = 512       # Embedding dimension
h = 8               # Number of heads
D_k = D_model // h  # Key and query dimension
D_v = D_model // h  # Value dimension
D_ff = 4 * D_model  # Feed Forward Network dimension

# 1. Tokenization with padding
batch_tokens = [
    ["[CLS]", "Всем", "привет", "!", "Я", "увлекаюсь", "искусственным", "интеллектом", "[SEP]"],
    ["[CLS]", "Привет", ",", "как", "дела", "?", "[SEP]", "[PAD]", "[PAD]"],
    ["[CLS]", "ИИ", "—", "это", "интересно", "!", "[SEP]", "[PAD]", "[PAD]"],
]

# 2. Create embeddings
embeddings = torch.randn(N, L, D_model)

# 3. Generate positional encodings
pe = positional_encoding(L, D_model)

# 4. Combine embeddings with positional encodings
X_embedded = embeddings + pe  # Broadcasting for batch

# ================= Visualization of the process =================
print("=" * 60)
print("Step 1: Initial token embeddings")
print(f"Tensor shape: {embeddings.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(embeddings[batch_idx], batch_tokens[batch_idx], "Initial embeddings")

print("\n" + "=" * 60)
print("Step 2: Positional encodings")
print(f"Tensor shape: {pe.shape}")
print_embeddings(pe, [f"Position {i}" for i in range(L)], "Example encodings")

print("\n" + "=" * 60)
print("Step 3: Combined embeddings (X + PE)")
print(f"Tensor shape: {X_embedded.shape}")
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    print_embeddings(X_embedded[batch_idx], batch_tokens[batch_idx], "Result of addition")

# ================= Detailed analysis for the first batch =================
print("\n" + "=" * 60)
print("Detailed analysis of the first batch:")
batch_idx = 0

# Initial data
print(f"\nText: '{sentences[batch_idx]}'")
print(f"Tokens: {batch_tokens[batch_idx]}")

# Comparison for key positions
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Initial embedding:  {embeddings[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Positional encoding: {pe[pos, :3].detach().numpy().round(4)}")
    print(f"Combined:           {X_embedded[batch_idx, pos, :3].detach().numpy().round(4)}")

# ================= Multi-Head Self-Attention =================
print("\n" + "=" * 60)
print("Step 4: Multi-Head Self-Attention")

# 1. Linear projections
W_Q = torch.randn(h, D_model, D_k)
W_K = torch.randn(h, D_model, D_k)
W_V = torch.randn(h, D_model, D_v)

Q = torch.einsum('nlk,hkd->nhld', X_embedded, W_Q)
K = torch.einsum('nlk,hkd->nhld', X_embedded, W_K)
V = torch.einsum('nlk,hkd->nhld', X_embedded, W_V)

print("\nLinear projections:")
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# Output first 3 elements for Q, K, V
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    for pos in [0, 2, 4]:
        print(f"  Position {pos} ({batch_tokens[batch_idx][pos]}):")
        print(f"    Q: {Q[batch_idx, :, pos, :3].detach().numpy().round(4)}")
        print(f"    K: {K[batch_idx, :, pos, :3].detach().numpy().round(4)}")
        print(f"    V: {V[batch_idx, :, pos, :3].detach().numpy().round(4)}")

# 2. Attention for each head
attention_scores = torch.einsum('nhld,nhmd->nhlm', Q, K) / math.sqrt(D_k)

# Masking padding
mask = torch.ones(N, 1, L, L, dtype=torch.bool)
for batch_idx, tokens in enumerate(batch_tokens):
    for i, token in enumerate(tokens):
        if token == "[PAD]":
            mask[batch_idx, :, i:, :] = False
            mask[batch_idx, :, :, i:] = False
attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

attention_weights = torch.softmax(attention_scores, dim=-1)
Z = torch.einsum('nhlm,nhmd->nhld', attention_weights, V)

print("\nAttention for each head:")
print(f"Z shape: {Z.shape}")

# Output first 3 elements for Z
for batch_idx in range(N):
    print(f"\nBatch {batch_idx + 1}: '{sentences[batch_idx]}'")
    for pos in [0, 2, 4]:
        print(f"  Position {pos} ({batch_tokens[batch_idx][pos]}):")
        print(f"    Z: {Z[batch_idx, :, pos, :3].detach().numpy().round(4)}")

# Visualization of attention weights for the first head and first position
print("\nVisualization of attention weights for the first head and first position:")
print(f"Attention weights (first head, first position): {attention_weights[0, 0, 0, :].detach().numpy().round(4)}")

# 3. Concatenation of heads
Z_concat = Z.transpose(1, 2).reshape(N, L, h * D_v)

print("\nConcatenation of heads:")
print(f"Z_concat shape: {Z_concat.shape}")

# 4. Linear projection of output
W_O = torch.randn(h * D_v, D_model)
multi_head_output = torch.einsum('nlk,kd->nld', Z_concat, W_O)

print("\nLinear projection of output:")
print(f"MultiHead Output shape: {multi_head_output.shape}")

# Output for the first batch
print("\nDetailed analysis of the first batch after Multi-Head Attention:")
batch_idx = 0
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Combined:             {X_embedded[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Multi-Head Output:    {multi_head_output[batch_idx, pos, :3].detach().numpy().round(4)}")

# ================= Add & Norm (after Multi-Head Attention) =================
print("\n" + "=" * 60)
print("Step 5: Add & Norm (after Multi-Head Attention)")

# Add (residual connection)
# The output of the Multi-Head Attention layer (multi_head_output) is added to the input of the Multi-Head Attention sub-layer (X_embedded)
output_add_norm_1_add = X_embedded + multi_head_output
print("\nAdd (residual connection):")
print(f"Output shape after Add: {output_add_norm_1_add.shape}")

# Norm (layer normalization)
# Apply Layer Normalization to the result of addition
layer_norm_1 = nn.LayerNorm(D_model) #  D_model - dimensionality to normalize
output_add_norm_1_norm = layer_norm_1(output_add_norm_1_add)
print("\nNorm (layer normalization):")
print(f"Output shape after LayerNorm: {output_add_norm_1_norm.shape}")

# Output for the first batch after Add & Norm
print("\nDetailed analysis of the first batch after Add & Norm:")
batch_idx = 0
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Multi-Head Output:      {multi_head_output[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Output Add:             {output_add_norm_1_add[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Output Add & Norm:      {output_add_norm_1_norm[batch_idx, pos, :3].detach().numpy().round(4)}")


# ================= Feed Forward Network =================
print("\n" + "=" * 60)
print("Step 6: Feed Forward Network")

# 1. First linear layer (Expansion Layer)
W_ff_1 = torch.randn(D_model, D_ff)
b_ff_1 = torch.randn(D_ff)
output_ffn_layer_1 = torch.relu(torch.einsum('nlk,kd->nld', output_add_norm_1_norm, W_ff_1) + b_ff_1)
print("\nFirst linear layer (Expansion Layer):")
print(f"Output shape after first linear layer: {output_ffn_layer_1.shape}")

# 2. Second linear layer (Contraction Layer)
W_ff_2 = torch.randn(D_ff, D_model)
b_ff_2 = torch.randn(D_model)
output_ffn = torch.einsum('nlk,kd->nld', output_ffn_layer_1, W_ff_2) + b_ff_2
print("\nSecond linear layer (Contraction Layer):")
print(f"Output shape after second linear layer (FFN Output): {output_ffn.shape}")

# Output for the first batch after Feed Forward Network
print("\nDetailed analysis of the first batch after Feed Forward Network:")
batch_idx = 0
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"Output Add & Norm:      {output_add_norm_1_norm[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"FFN Output:             {output_ffn[batch_idx, pos, :3].detach().numpy().round(4)}")

# ================= Add & Norm (after Feed Forward Network) =================
print("\n" + "=" * 60)
print("Step 7: Add & Norm (after Feed Forward Network)")

# Add (residual connection)
# The output of FFN (output_ffn) is added to the input of the FFN sub-layer (output_add_norm_1_norm)
output_add_norm_2_add = output_add_norm_1_norm + output_ffn
print("\nAdd (residual connection after FFN):")
print(f"Output shape after Add: {output_add_norm_2_add.shape}")

# Norm (layer normalization)
# Apply Layer Normalization to the result of addition
layer_norm_2 = nn.LayerNorm(D_model) #  D_model - dimensionality to normalize
output_add_norm_2_norm = layer_norm_2(output_add_norm_2_add)
print("\nNorm (layer normalization after FFN):")
print(f"Output shape after LayerNorm: {output_add_norm_2_norm.shape}")

# Output for the first batch after Add & Norm (after FFN)
print("\nDetailed analysis of the first batch after Add & Norm (after FFN):")
batch_idx = 0
for pos in [0, 2, 4, 6, 8]:
    print(f"\nPosition {pos} ({batch_tokens[batch_idx][pos]}):")
    print(f"FFN Output:             {output_ffn[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Output Add (after FFN):   {output_add_norm_2_add[batch_idx, pos, :3].detach().numpy().round(4)}")
    print(f"Output Add & Norm (after FFN): {output_add_norm_2_norm[batch_idx, pos, :3].detach().numpy().round(4)}")
```

6. **Add & Norm layer (after Feed Forward):**
   - **Add (residual connection):** The output of FFN (output_ffn) is added to the input of the FFN sub-layer (output_add_norm_1_norm):
     $$
     \text{Output}_{Add2} = \text{Output}_{Norm1} + \text{FFN}(\text{Output}_{Norm1})
     $$
   - **Norm (layer normalization):** Layer normalization is applied:
     $$
     \text{Output}_{Norm2} = \text{LayerNorm}(\text{Output}_{Add2})
     $$

7. **Encoder output:**
   - The output of each encoder layer is $\text{Output}_{Norm2}$ of dimensionality $(N, L, D_{model})$.
   - After passing through all $N_{layers}$ encoder layers, the final output represents a matrix of contextualized embeddings of dimensionality $(N, L, D_{model})$.

Continuing the paper review...

</detail>

# 🚀 Accelerating Your Encoder

### Supporting Generative Models 🤝

One way to understand the popularity of encoder-only models is to note how often they are combined with decoder-only models to create safe and efficient systems.

A clear example is **RAG**. Instead of relying on the knowledge that the LLM learned from its model parameters, the system uses a document store to provide the LLM with information relevant to the query. But, of course, this only postpones the problem. If the LLM doesn't know which documents are relevant to the query, then the system needs some other process to select these documents? For this, a model is needed that is fast and cheap enough to be used for encoding large amounts of information necessary to make the LLM useful. Usually, this model is an encoder-only model, such as **BERT**. 📚

Another example is a controlled architecture where inexpensive classifiers can be used to ensure that the generated text does not violate content safety requirements. 🔒

In short, every time you see a decoder-only model in deployment, there is a reasonable probability that an encoder-only model is also part of the system. But the reverse is not true. 🔄

### Encoder-Based Systems 🌐

Before **GPT**, content recommendations existed in social networks and on platforms like **Netflix**. Advertising targeting was done on these platforms, in search, and other places. There are also content classifications, such as spam detection and abuse detection. These systems are built not on generative models, but on representation models, such as encoder-only models. All these systems still exist and operate on a massive scale. Imagine how many ads are targeted every second around the world! 🌍

**Downloads:** on **HuggingFace** **RoBERTa**, the leading model based on **BERT**, has more downloads than the 10 most popular LLMs on **HuggingFace** combined. In fact, an encoder-only model currently provides more than **1 billion monthly downloads**, almost three times the **397 million monthly downloads** of a decoder-only model. In fact, the category of masked language models, consisting of "base" encoder-only models such as **ModernBERT**, ready for fine-tuning for other subsequent applications, is the most downloaded category of models of all model categories. 📥

**Inference Cost:** The above shows that in terms of each inference, encoder-only models require far more inferences per year than decoder-only models or generative models. An interesting example is **FineWeb-Edu**, where quality filtering based on a model must be performed for more than **15 trillion tokens**. The **FineWeb-Edu** team decided to use the decoder-only model **Llama-3-70b-Instruct** to create annotations and use a fine-tuned **BERT**-based model to perform most of the filtering. This filtering took **6000 H100 hours** at a total cost of **$60,000** at a HuggingFace inference point price of **$10 per hour**. In contrast, even using the cheapest option of **Google Gemini Flash** and its low inference cost of **$0.075 per million tokens**, transmitting **15 trillion tokens** to a popular decoder-only model would cost more than **a million dollars**! 💸

## Performance 🚀

### Overview 📊

Figure 1 shows the accuracy results of the **ModernBERT** model and several other models on various tasks, measured using standard academic benchmarks. 📈 The data demonstrates that **ModernBERT** achieves superior results in most categories considered, making it a universal model for encoder-based tasks. 🏆

![Table_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Table_1.png)

The **DeBERTaV3** model long served as the benchmark in natural language processing (NLP) competitions, particularly on the Kaggle platform. 🥇 However, **ModernBERT** not only became the first base-sized model to surpass **DeBERTaV3** on the GLUE benchmark, but also demonstrates significantly less memory consumption, less than 1/5 of the memory required by **DeBERTa**. 💾

**ModernBERT** is characterized by high speed. ⏩ In particular, it demonstrates twice the speed compared to **DeBERTa**, and in typical use cases with mixed-length inputs, the speedup can reach fourfold. 🚀 The processing speed of long contexts is also significantly higher, almost three times faster than other high-performance models such as **NomicBERT** and **GTE-en-MLM**. 🏎️

The context length of the **ModernBERT** model reaches **8192 tokens**, more than 16 times greater than the capabilities of most existing encoder models. 📏 This property plays a key role in Retrieval-Augmented Generation (RAG) pipelines, where limited context can lead to information fragmentation and hinder semantic understanding. 🧩 **ModernBERT** also efficiently integrates with the ColBERT long-context extraction method, demonstrating a 9 percentage point advantage over other long-context models. 📊 Notably, this model, with its high training speed and adapted for tasks compared to other base models, surpasses even widely used search models in tasks requiring long-context processing. 🎯

**ModernBERT** demonstrates unique capabilities in code search tasks. 💻 Currently, there are no similar encoder models trained on a comparable volume of code-containing data. 📚 As an example, the StackOverflow-QA (SQA) dataset, a hybrid resource combining code and natural language. 🌐 In this dataset, due to specialized code understanding and the ability to process long contexts, **ModernBERT** is one of the few models to achieve a score above 80 points. 🎉

These functional capabilities of **ModernBERT** open up prospects for creating a whole range of new applications. 🌟 As an example, consider integrating with an intelligent Integrated Development Environment (IDE) that indexes the entire enterprise codebase, using **ModernBERT**'s capabilities to quickly and accurately retrieve relevant code from various repositories, taking into account long contexts. 🔍 Another example could be a code chatbot service capable of providing application functionality descriptions by aggregating information from multiple separate projects. 🤖

In comparison with base models, **ModernBERT** demonstrates higher efficiency in three key task categories: search, natural language understanding, and code search. 🔎 In the area of natural language understanding, the model slightly lags behind **DeBERTaV3**, but significantly outperforms it in speed. ⚡ It is important to note that **ModernBERT**, like any base model, is initially designed for the masked language modeling task. 🎭 Additional fine-tuning of the model is required for other tasks. 🛠️

In comparison with advanced models, **ModernBERT** demonstrates comparable or superior results in most tasks. 📈 Moreover, **ModernBERT** surpasses most models in processing speed and can handle input sequences up to 8192 tokens, significantly exceeding the capabilities of base models. 🚀

## Efficiency ⚡

Figure 2 shows the data on memory efficiency (maximum batch size, BS) and inference speed (thousands of tokens per second) for **ModernBERT** and several other decoder models tested on an NVIDIA RTX 4090 graphics card. 📊 First, it should be noted that the efficiency analysis was conducted on widely available consumer-class graphics cards, not on the latest and hard-to-obtain equipment. 💻 This is due to the fact that **ModernBERT** development is oriented towards practical applicability and usefulness, not just creating an exclusive advertising product. 🎯

![Table_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Table_2.png)

According to the principle of practical orientation, the developers of **ModernBERT** aimed to ensure its applicability in real-world applications, not just in laboratory test conditions. 🧪 Traditionally, model testing is often conducted under optimal conditions, for example, at maximum context length, as reflected in the "fixed" column in the table. 📏 However, in real-world scenarios, input data sizes vary, so special attention was paid to optimizing performance under variable-length inputs, as reflected in the "variable" column. 🔄 The data shows that **ModernBERT** demonstrates significantly higher processing speed compared to other models when handling variable-length input data. 🚀

**ModernBERT** demonstrates a speed 2-3 times higher than the nearest competing model when processing long contextual inputs, which are likely to play a key role in the most promising future applications. 🌟 From a practical implementation perspective, **ModernBERT** does not require installing additional complex dependencies, except for the widely used Flash Attention library. 📚

The efficiency of **ModernBERT** allows for using a larger batch size compared to most other models, which enables the effective use of graphics cards with lower computational power and cost. 💰 In particular, the high efficiency of the base version of the model opens up prospects for developing new applications that can be deployed directly in web browsers, on mobile phones, and other devices. 📱

## Why ModernBERT, well, modern? 🤔

The data presented above demonstrates the need to pay more attention to encoder models. 🧠 Despite their significance, encoder architecture has evolved less intensively compared to decoder architecture, especially after the emergence of the BERT model in 2018. 📅

Notably, after the emergence of the RoBERTa model, further development of encoders did not lead to overall improvement (so-called "Pareto improvement") without any trade-offs. 📉 For example, the DeBERTaV3 model demonstrates higher performance in the GLUE benchmark and classification tasks, but lags in efficiency and search tasks. 🔍 Similarly, other models, such as AlBERT and GTE-en-MLM, improve specific aspects of the original BERT and RoBERTa models but lose in others. 🎲

Nevertheless, since the emergence of the first BERT and RoBERTa models, significant progress has been made in the field of language model development. 🚀 In particular, in the field of decoder models, unlike encoders, there is a trend towards "Pareto improvement," where new models surpass previous ones in all key parameters. 📈 Advancement in models is the result of both scientific research and engineering efforts. 🛠️

Thus, the main goal of the **ModernBERT** project is to integrate modern engineering approaches into the development of encoder models. 🎯 This is achieved by implementing the following three key principles:

1. Use of modern Transformer architecture. 🤖
2. Prioritizing efficiency. ⚡
3. Applying modern data scaling methods and expanding data sources. 📊

### Meet the new Transformer, which is no different from the old Transformer 🤖

The Transformer architecture has become dominant and is currently used in the vast majority of modern models. 🌍 It is important to note that there are many variations of the Transformer architecture. 🔄 A common principle for all of them is that the attention mechanism plays a key role, and further improvements are built around optimizing this mechanism. 🎯

**ModernBERT** is based on the Transformer++ architecture (developed by Mamba), which was first applied in the Llama2 series of models. 🦙 In particular, in **ModernBERT**, several components of the BERT architecture have been replaced with their improved counterparts, specifically:

- Replacing traditional positional encoding with rotary positional embedding (RoPE), which provides improved understanding of relative positions between tokens and allows scaling to longer sequences. 🔄
- Replacing the MLP layer with a GeGLU layer and improving the GeLU activation function used in the original BERT model. 🧠
- Simplifying the architecture by removing redundant bias parameters, allowing more efficient use of computational resources. 💻
- Adding an additional normalization layer after embedding, which helps stabilize the training process. 📊

### Improving Efficiency

As noted earlier, encoder models, including **ModernBERT**, do not possess characteristics comparable to high-performance models. 🏎️ However, this does not mean they are not capable of demonstrating high speed. ⏩ In most practical scenarios, similar to how an ordinary car is used for daily highway driving, it is expected that a reliable encoder model will efficiently handle data processing tasks within established performance requirements. 🚗

Indeed, in the scenarios considered, data processing speed plays a key role. ⏱️ Encoder models are especially popular in cases where it is necessary to process large volumes of data, where even minor speed increases can quickly accumulate or where latency is very important, for example, **RAG**. In many cases, the encoder even operates on a central processing unit (CPU), and efficiency is even more important if we want to get results in a reasonable time. ⚙️

Consistent with common scientific research practice, the development of **ModernBERT** leverages the achievements of previous work, in particular, the advantages provided by the speed optimization of Flash Attention 2. 🚀 The efficiency of **ModernBERT** is achieved through the implementation of the following three key components:

1. Using the attention switching mechanism to improve processing efficiency. 🔄
2. Using padding removal and sequence packing methods to reduce computational costs. 📦
3. Designing the model architecture with hardware considerations for optimal use of computing equipment. 💻

### Upgrading your Honda Civic for the track 🏎️

We've already discussed this: encoders are not **Ferraris**, and **ModernBERT** is no exception. However, this does not mean it cannot work fast. When you go on the highway, you usually don't go out and change your car to a race car, but rather expect that your everyday, reliable car will comfortably handle the speed limit. 🚗

Indeed, for all the use cases we mentioned above, speed is crucial. Coders are very popular in situations where it is necessary to process large amounts of data, where even small speed increases can quickly accumulate or where delay is very important, for example, **RAG**. In many cases, the encoder even runs on the CPU, and efficiency is even more important if we want to get results in a reasonable time. ⏱️

As in most other research, we build on the shoulders of giants and benefit from the speed improvements of **Flash Attention 2**. Our efficiency improvement is based on the following three key components:

1. **Attention switching** for improved processing efficiency
2. **Padding removal and sequence packing** for reduced computational costs
3. **Model design with hardware considerations** for maximum equipment utilization

### Global and Local Attention 🌍

One of the most effective features of **ModernBERT** is its alternating attention, unlike exclusively global attention. Technically, this means that the model's attention mechanism pays attention to the full input only every **3 layers** (global attention), while the remaining layers use a sliding window where each token pays attention only to the **128 closest tokens** (local attention). Since the computational complexity of attention increases sharply with each additional token, this allows **ModernBERT** to process long input sequences faster than any other model. ⚡

Conceptually, the reason for this efficiency is quite simple: imagine you are reading a book. Do you need to fully understand the entire plot of each sentence to understand most of it (global attention)? Or is it enough to be aware of the current chapter (local attention), if you periodically review its meaning for the main plot (global attention)? In the vast majority of cases, the latter is true. 📚

### Sequence Unpacking and Packing 📦

Another key mechanism enhancing the efficiency of **ModernBERT** is the use of padding removal and sequence packing methods.

To process multiple sequences in a single batch, encoder models require all sequences to have the same length to ensure parallel computation. Traditionally, for this, padding is used: the longest sequence is determined, and padding tokens (padding tokens) are added to the remaining sequences to align the length. 🧩

Although padding solves the problem, this solution is not optimal: a significant portion of computational resources is spent on processing padding tokens, which carry no semantic load. 💡

In contrast to padding, sequence packing (unpacking) allows avoiding unnecessary computations on padding tokens, and the number of meaningful tokens becomes more uniform across different batches. When using masking, samples can be processed individually. 🎯

Padding removal effectively solves this problem: instead of storing padding tokens, they are removed, and sequences are combined into mini-batches of size **1**, which avoids unnecessary computations. When using **Flash Attention**, the implementation of padding removal is even faster than previous approaches, which largely relied on unpacking and repadding sequences as they passed through the model. This is achieved through a custom unpacking implementation based on the latest advancements in **RoPE Flash Attention** support. This approach allows **ModernBERT** to remove sequence padding once and, if necessary, repad it after processing, making the model **10–20% faster** compared to previous methods. ⚡

For further acceleration during pre-training, padding removal is effectively used in combination with sequence packing. Sequence packing is a logical next step: since input data is combined into a sequence, and graphics processors efficiently perform parallelization, it is necessary to maximize the computational efficiency obtained from a single forward pass of the model. For this, a greedy algorithm is used, which groups individual sequences into combined sequences whose length is as close as possible to the model's maximum input length. 🧠

### Pay Attention to Hardware 💻

Finally, the third aspect of **ModernBERT**'s efficiency is hardware consideration.

When designing the model architecture, two insights from previous research were considered:

1. **Depth and width of layers.** Research shows that deeper models with narrower layers generally demonstrate better performance than shallower models with wider layers. However, increasing model depth has a downside: the deeper the model, the less opportunity for parallelization, and thus, with the same number of parameters, it runs slower.
2. **Hardware efficiency.** To achieve maximum performance, the model dimensions must match the capabilities and limitations of the target graphics processor, and different graphics processor models have different limitations.

There is no optimal solution that provides equally high model performance on different graphics processors. However, there are useful recommendations, for example, "Examples of joint model architecture design with hardware," which provides a detailed description of optimizing model architecture for specific graphics processors. As a heuristic approach, it is proposed to extend these recommendations for different groups of graphics processors, while adhering to a certain set of constraints. Logically, the first step is to define these constraints, which in this case include:

- Defining target graphics processors as common inference models (**RTX 3090/4090**, **A10**, **T4**, **L4**).
- Defining approximate target model sizes: **130 to 150 million parameters** for **ModernBERT-Base** and **350 to 420 million parameters** for **ModernBERT-Large**.
- Ensuring the final embedding size matches the original **BERT** model sizes (**768** for base, **1024** for large) for maximum backward compatibility.
- Establishing general performance constraints for graphics processor groups.

Then, various model architectures were tested using limited grid search, varying the number and width of layers. After determining the most efficient configurations, the heuristic approach was confirmed to match the actual performance of graphics processors, and the final architecture model was selected. 🛠️

Another important area where encoders have potential for improvement is training data. Often, this refers only to the volume of training data, but this is not entirely accurate. Previous encoder models, such as **DeBERTaV3**, were trained for a sufficiently long time and may have even exceeded the trillion-token threshold! 📚

The problem lies in the diversity of training data: many earlier models were trained on limited datasets, often consisting of **Wikipedia** and **Wikibooks**. These datasets primarily represent textual modality and contain only high-quality natural text. 📖

In contrast, **ModernBERT** is trained on data from diverse English sources, including web documents, code, and scientific articles. The total volume of training data is **2 trillion tokens**, most of which are unique, not repeated **20–40 times**, as in previous encoders. 📊

The result of this approach is evident: among all existing open-source code encoders, **ModernBERT** stands out in solving programming-related tasks. Particular interest is the potential use of this model to improve programming assistance tools. 💻

## Process 🛠️

We follow the training methodology used for the original **BERT** model, with some minor improvements inspired by subsequent research. In particular, we abandoned the goal of predicting the next sentence, as it was found that its addition creates unnecessary load without noticeable improvement in results. Moreover, we changed the masking token ratio, increasing it from **15%** to **30%**. 📈

Both models are trained in three stages, ensuring comprehensive preparation. Initially, the models are trained on **1.7 trillion tokens** with a sequence length of **1024**. Then follows the long-context adaptation stage, during which training continues on **250 billion tokens** with a sequence length of **8192**. During this, to maintain computational stability, the total number of tokens processed in each batch remains relatively constant by proportionally reducing the batch size. On the final stage, "annealing" is performed on **50 billion tokens** selected using various strategies to achieve an optimal balance of long-context advantages, as highlighted in the **ProLong** study. 🧠

This three-stage training approach ensures high efficiency of the model in various tasks, which is confirmed by its results: **ModernBERT** demonstrates competitiveness in tasks requiring long-context processing and does not lag in performance when working with short contexts. 📊

… but there is another important advantage: on the first two stages, after completing the warm-up phase, training is conducted at a constant speed. The learning rate decay is applied only on the last **50 billion tokens**, according to a trapezoidal scheme (or "warm-up-stabilization-decay" scheme). Moreover, inspired by the **Pythia** approach, we deliberately remove each intermediate checkpoint created during the stable training phases. This decision is driven by the desire to support future research and practical applications: any researcher or developer can resume training from any of the provided checkpoints before the decay phase and perform further training on specialized data corresponding to their specific tasks! 🚀

### Know-how – the key to success! 🧠

If you have been following the narrative closely, you probably already anticipated: to further accelerate the training process, we, of course, apply a number of effective techniques. In particular, we have two key methods in our arsenal.

Let's start with the first, a fairly common one: since the initial training phase is associated with fine-tuning random weights, we use a gradual increase in batch size strategy. At the initial stage, we work with a small batch size to ensure more frequent model weight updates during the processing of a given amount of data. Then, as training progresses, we gradually increase the batch size to the target value. This approach significantly speeds up the initial training phase, when the model is actively learning fundamental language patterns. 📚

The second technique, conversely, is less trivial: initializing weights for larger models using a "mosaic" approach, inspired by the **Microsoft Phi** series of models. The basis of this method is a simple but important idea: why initialize **ModernBERT-large** weights randomly if there is already a high-quality (we allow ourselves such an assessment) set of **ModernBERT-base** weights? 🧩

Practical experience shows that using the base model weights **ModernBERT** as a starting point for **ModernBERT-large** provides more effective training than random initialization. Moreover, this method works well in conjunction with the gradual batch size increase strategy, which together allows for significant acceleration of the initial training phase. ⚡

## Conclusion 🎯

In this publication, we present **ModernBERT** – a new family of modern, compact, and high-performance models developed with an architecture exclusively focused on encoding. **ModernBERT** represents a long-awaited update to the **BERT** paradigm. 🚀