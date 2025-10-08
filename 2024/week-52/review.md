## Briefing Document: "Super Weights" in Large Language Models

## Introduction

This review covers research on "super weights" in large language models (LLMs). The authors discovered that a very small number of parameters—even as few as a single scalar!—in LLMs play a disproportionately critical role in their ability to generate high-quality text. Researchers from Apple claim that a tiny subset of at most six scaling factors matters more than all others. The authors refer to these as super weights, and pruning them destroys model quality.

Several prior papers have shown that, at scale, a small set of hidden state features contains outliers with enormous magnitude. These outliers constitute a small percentage of all activations but are crucial for preserving the quality of compressed models. In the context of LLMs, these outliers manifest as "super activations"—anomalously large activations that are also critical for model quality. Removing these "super weights" can completely break the model, reducing accuracy to random-guessing levels and increasing perplexity by several orders of magnitude.

The study also demonstrates that these "super weights" and "super activations" can be identified using a simple, data-free method. This method is proposed for improving model quantization, enabling quality retention even under significant reductions in computational complexity.


## Key Findings and Ideas

### Super Weights
- The authors discovered that a single parameter ("super weight") in an LLM exerts a disproportionately large influence on model quality.

- Removing this parameter can lead to the generation of meaningless text, both qualitatively and quantitatively (demonstrated on Llama-7B in Figure 1 and Table 1).

- Notably, removing even 7,000 other largest-magnitude parameters has negligible impact on quality compared to removing just one "super weight."

**_Inside trained LLMs lies a group of outlier weights with large magnitude, comprising roughly 0.01% of all model weights—still hundreds of thousands in billion-parameter models. This was known previously. The current work shows that within this group resides a single weight (the super weight, SW), not necessarily the largest in magnitude, whose importance exceeds the combined importance of thousands of other outliers. It is essential for quality; without it, the LLM cannot generate coherent text. Perplexity increases by several orders of magnitude, and zero-shot task accuracy drops to random levels._**

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A0%D0%B8%D1%81%D1%83%D0%BD%D0%BE%D0%BA_1.png)

![Table_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A2%D0%B0%D0%B1%D0%BB%D0%B8%D1%86%D0%B0_1.png)

> “In Llama-7B, removing the super weight—a single scalar—completely destroys the model’s ability to generate text. Average zero-shot task accuracy effectively drops to zero. Conversely, removing the other 7,000 largest outliers, including outliers larger than the super weight, affects performance by no more than a few percentage points.”


## Identifying Super Weights

### Core Method

A data-free method for identifying super weights is proposed, requiring no validation dataset or usage examples. The method is based on the following principles:

- Analyzing the distribution of activations during a forward pass
- Detecting jumps in the input and output distributions of `mlp.down_proj` layers
- Using only a single input prompt for detection
- The authors provide a catalog of super weight coordinates for several publicly available LLMs (Table 2).


**Super weight coordinate definition:**
  - Row is determined by the channel index of the input activation distribution
  - Column is determined by the channel index of the output activation distribution

**Super weight characteristics:**
  - Not necessarily the largest in absolute magnitude within the weight matrix
  - Detectable using an arbitrary input prompt
  - Reducing activation requires pruning only one weight

**Distribution across models:**
  - The maximum number of super weights (six) was found in Phi-3-mini-4k-instruct
  - Super weight positions remain stable after instruction fine-tuning

![Table_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A2%D0%B0%D0%B1%D0%BB%D0%B8%D1%86%D0%B0_2.png)

> “Based on the above analysis, we present an efficient way to localize super weights: SWs can be found by detecting jumps in the input and output distributions of down_proj across layers. This detection requires only a single input prompt, not a validation dataset or usage examples.”


### Super Activations
- Super weights induce "super activations"—very large activations that persist across many layers at the same position, regardless of input.
- These super activations play a key role in model functioning.
- Removing the super weight sharply reduces the magnitude of the super activation, confirming a causal relationship.

**_Earlier work (https://arxiv.org/abs/2402.17762) identified super activations critical for quality. They exist across various layers, have constant magnitude, and always appear at the same position regardless of input. The current work finds that the activation channel coincides with that of the super weight, and the activation first appears immediately after the super weight. Pruning this super weight significantly reduces the activation, suggesting it is caused by the weight—not merely correlated. Such activations are termed super activations (SAs)._**

**_Prior work explained super activations via bias terms but did not explain how they arise or why they consistently appear at the same locations. Now, the authors empirically find that, prior to the down projection (down_proj), the Hadamard product of gate and up projections (gate_proj, up_proj) produces a relatively large activation. The super weight then further amplifies it, yielding the super activation._**

**Recall that the MLP block in Llama looks like this:**

```
out = down_proj( act_fn(gate_proj(input)) x up_proj(input) )
```

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A0%D0%B8%D1%81%D1%83%D0%BD%D0%BE%D0%BA_4.png)

> “We discover another intriguing property: the activation channel matches our super weight, and the activation appears immediately after our super weight. To confirm whether this is correlation or causation, we remove the super weight and check the magnitude of the massive activation. In Figure 4, we find that removing the super weight sharply reduces the massive activation’s magnitude. This indicates that massive activations are created by super weights. For a sequence, we call these massive activations ‘super activations.’”


### Mechanisms of Super Weight Action
- Beyond generating super activations, super weights suppress stop-word probabilities in model outputs (Figures 2, 5).
- Removing super weights increases stop-word probabilities and decreases probabilities of meaningful words.
- Restoring super activations partially recovers model quality after super weight removal—but not fully.
- Experiments were conducted by zeroing out the SW, including restoring the SA to its original value, to test the SW’s influence on other activations. This recovers 42% of the loss, indicating the SW’s impact on quality extends beyond just the SA.
- Analysis of 500 diverse prompts from the Lambada validation set shows that removing the SW drastically increases stop-word probabilities (while correspondingly decreasing regular word probabilities). For “the” it’s 2×, for “.” it’s 5×, and for “,” it’s 10×. Thus, the presence of the SW appears to suppress stop words and enable coherent text generation.

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A0%D0%B8%D1%81%D1%83%D0%BD%D0%BE%D0%BA_2.png)

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/main/2024/week-52/assets/%D0%A0%D0%B8%D1%81%D1%83%D0%BD%D0%BE%D0%BA_5.png)

> “Specifically, when we restore super activations, average accuracy recovers to 49.94 from 35.14, indicating that restoring super activations recovers approximately 42% of the quality loss. These results show that although super activations contribute significantly to model performance, they do not fully explain the overall impact of the super weight on quality.”


### Super Weights and Quantization
- Super weights and super activations strongly negatively impact model quantization.
- A quantization method is proposed that explicitly preserves super weights, improving quality over traditional quantization.
- For activation quantization, the approach replaces the super activation with its median value, quantizes, then restores the original value.
- For weight quantization, the method clips (constrains) outliers—including the super weight—quantizes, then restores the super weight.


### Experimental Results
- Experiments were conducted on various LLMs, including Llama, Mistral, and OLMo.
- The proposed super-weight-aware quantization method achieved competitive results compared to state-of-the-art methods like SmoothQuant.
- The method enables scaling block-wise weight quantization to larger block sizes without significant quality degradation.


## Practical Significance
- Identifying critically important parameters in LLMs can lead to more efficient model compression and optimization techniques.
- The proposed data-free super weight identification method can be used to optimize model quantization without requiring additional training data.
- Improved quantization that accounts for super weights enables more compact and efficient models suitable for resource-constrained environments.


## Conclusion
The study demonstrates the importance of "super weights" and "super activations" in LLM operation. Despite their scarcity, these parameters exert a disproportionately large influence on model quality. The authors propose practical methods for their identification and utilization to improve model quantization. The findings underscore the need for further investigation and explicit consideration of super weights in developing more efficient and robust LLMs.

---

## Glossary
- **Large Language Model (LLM)**: A machine learning model trained on vast amounts of textual data, capable of generating and understanding natural language.
- **Super weight**: A single scalar parameter in an LLM that, despite not being the largest in magnitude, plays a disproportionately critical role in model performance.
- **Super activation**: An anomalously large activation value resulting from the influence of a super weight.
- **Zero-shot**: A model’s ability to perform a task without prior training on that specific task.
- **Perplexity**: A measure of how well a model predicts the next token in a sequence. Lower perplexity indicates better performance.
- **Quantization**: A technique for reducing numerical precision to decrease model size and accelerate computation.
- **mlp.down_proj**: The down-projection layer in a multilayer perceptron (MLP), part of the LLM architecture.
- **Clipping**: A method of constraining value ranges to mitigate outlier effects during quantization.
- **Stop words**: Frequently occurring words (e.g., “and,” “but,” “on”) that typically carry little semantic information.
- **Hadamard product**: Element-wise multiplication of two matrices.
- **SmoothQuant**: An LLM quantization method that scales activations to reduce outlier impact.
- **AWQ (Activation-aware Weight Quantization)**: A weight quantization method that considers activations to optimize scaling parameters.
- **SqueezeLLM**: A quantization method using sparse matrices to preserve the most important parameters at higher precision.
- **Skip connection**: A direct connection between layers that bypasses one or more intermediate layers.
- **Per-tensor quantization**: A quantization method applying identical quantization parameters to an entire tensor.
- **Per-token quantization**: A quantization method applying quantization parameters individually to each token.
- **Gaussian Distribution**: The normal distribution describing the spread of random variables.
- **Z-score**: A measure of how many standard deviations an observation lies from the mean.

---

## Quick Quiz

1. **What is a "super weight" in the context of large language models (LLMs)?**  
   A super weight is a single scalar parameter in an LLM that, although not the largest in magnitude, plays a disproportionately critical role in model quality. Its removal can completely destroy the LLM’s ability to generate text.

2. **Where are super weights typically located in LLM architecture, according to the study?**  
   Super weights are typically found in the `mlp.down_proj` layer in the early stages of the LLM architecture.

3. **What are "super activations," and how are they related to super weights?**  
   Super activations are anomalously large activation values in LLMs that persist across many layers. They arise as a result of super weights amplifying input activations.

4. **What is the impact of removing a super weight on LLM performance?**  
   Removing a super weight causes a sharp drop in zero-shot task accuracy and increases perplexity by orders of magnitude.

5. **How can knowledge of super weights be leveraged during LLM quantization?**  
   Knowledge of super weights allows them to be preserved during quantization, while other weights can be quantized using methods like clipping. This improves the quality of the quantized model.

6. **Describe the method for identifying super weights presented in the study.**  
   Super weights can be identified by detecting peak values in the input and output distributions of `mlp.down_proj` layers using a single input prompt.

7. **How do super weights affect the probability distribution of output tokens?**  
   Removing super weights increases stop-word probabilities and decreases probabilities of meaningful words, impairing the model’s ability to make accurate and confident predictions.

8. **How does the study investigate the relationship between super weights and super activations?**  
   Researchers show that removing a super weight leads to a significant reduction in super activation magnitude, indicating that super weights generate these anomalous activations.

9. **What were the results of experiments restoring super activations after super weight removal?**  
   Restoring super activations after super weight removal partially recovers LLM quality but not fully, indicating that super weights affect the model through mechanisms beyond just super activations.

10. **How do the study’s findings relate to other LLM quantization methods like SmoothQuant, AWQ, and SqueezeLLM?**  
    The study shows that preserving super weights can be competitive with methods like SmoothQuant for activation quantization and can enable larger block sizes in weight quantization, similar to AWQ and SqueezeLLM, which also implicitly account for the importance of these parameters.

---

## **Mathematical Formalization**

### 1. Basic Operations and Notation

1. **Input data**:  
   - $X$ — matrix/tensor of dimension $(L \times H)$.  
     - $L$ — number of positions in the input sequence (or batch size $\times$ sequence length).  
     - $H$ — hidden dimension.  
   - Element $X_{i,k}$ denotes the value of the input activation at row $i$ and column (channel) $k$.  

2. **Weight matrix**:  
   - $W$ — weight matrix of dimension $(D \times H)$.  
     - $D$ — output dimension, smaller or larger than $H$ depending on the architecture.  
   - Element $W_{j,k}$ denotes the weight value multiplied by $X_{i,k}$ when computing the corresponding output.  

3. **Output activations**:  
   - $Y$ — result of multiplying $X$ by $W^\mathsf{T}$, i.e., a matrix of dimension $(L \times D)$.  
   - Element $Y_{i,j}$ is obtained from the dot product of the $i$-th row of $X$ with the $j$-th row of $W$.  

### 2. Element-wise Computation of Output Activation

Each element $Y_{i,j}$ is defined by the formula:

$$
  Y_{i,j} 
  = \sum_{k=1}^{H} \left( X_{i,k} \times W_{j,k} \right).
$$

- Index $i$ ranges over $1 \le i \le L$.  
- Index $j$ ranges over $1 \le j \le D$.  
- Index $k$ ranges over $1 \le k \le H$.

### **_Let’s illustrate this concept with a concrete 3x3 matrix example, mimicking computations inside an LLM Transformer._**

  **Assume we have the following matrices:**
  
  **1. Input matrix X (dimension 3x3):**
  
  Imagine this represents activations for three consecutive words in a sentence, where each word is represented by a hidden-dimension vector of size 3.
  
  ```
  X = [[1, 2, 3],  # Activations for the first word
       [4, 5, 6],  # Activations for the second word
       [7, 8, 9]]  # Activations for the third word
  ```
  
  Here:
  - $L = 3$ (number of positions/words)
  - $H = 3$ (hidden dimension)
  - $X_{1,1} = 1$, $X_{1,2} = 2$, $X_{1,3} = 3$
  - $X_{2,1} = 4$, $X_{2,2} = 5$, $X_{2,3} = 6$
  - $X_{3,1} = 7$, $X_{3,2} = 8$, $X_{3,3} = 9$
  
  **2. Weight matrix W (dimension 3x3):**
  
  Imagine this represents the weights of a linear transformation layer in a Transformer that maps input activations into a new space of the same dimensionality.
  
  ```
  W = [[0.1, 0.2, 0.3],  # Weights for the first output channel
       [0.4, 0.5, 0.6],  # Weights for the second output channel
       [0.7, 0.8, 0.9]]  # Weights for the third output channel
  ```
  
  Here:
  - $D = 3$ (output dimension)
  - $H = 3$ (hidden dimension)
  - $W_{1,1} = 0.1$, $W_{1,2} = 0.2$, $W_{1,3} = 0.3$
  - $W_{2,1} = 0.4$, $W_{2,2} = 0.5$, $W_{2,3} = 0.6$
  - $W_{3,1} = 0.7$, $W_{3,2} = 0.8$, $W_{3,3} = 0.9$
  
  **3. Computing the output matrix Y (dimension 3x3):**
  
  Now let’s compute the elements of matrix $Y$ element by element using the formula:
  
  $$
    Y_{i,j} = \sum_{k=1}^{H} \left( X_{i,k} \times W_{j,k} \right)
  $$
  
  **Computing $Y_{1,1}$:**
  
  Here $i=1$, $j=1$. We take the first row of matrix $X$ and the first row of matrix $W$:
  
  $Y_{1,1} = (X_{1,1} \times W_{1,1}) + (X_{1,2} \times W_{1,2}) + (X_{1,3} \times W_{1,3})$
  
  $Y_{1,1} = (1 \times 0.1) + (2 \times 0.2) + (3 \times 0.3)$
  
  $Y_{1,1} = 0.1 + 0.4 + 0.9$
  
  $Y_{1,1} = 1.4$
  
  **Computing $Y_{1,2}$:**
  
  Here $i=1$, $j=2$. We take the first row of matrix $X$ and the second row of matrix $W$:
  
  $Y_{1,2} = (X_{1,1} \times W_{2,1}) + (X_{1,2} \times W_{2,2}) + (X_{1,3} \times W_{2,3})$
  
  $Y_{1,2} = (1 \times 0.4) + (2 \times 0.5) + (3 \times 0.6)$
  
  $Y_{1,2} = 0.4 + 1.0 + 1.8$
  
  $Y_{1,2} = 3.2$
  
  **Computing $Y_{1,3}$:**
  
  Here $i=1$, $j=3$. We take the first row of matrix $X$ and the third row of matrix $W$:
  
  $Y_{1,3} = (X_{1,1} \times W_{3,1}) + (X_{1,2} \times W_{3,2}) + (X_{1,3} \times W_{3,3})$
  
  $Y_{1,3} = (1 \times 0.7) + (2 \times 0.8) + (3 \times 0.9)$
  
  $Y_{1,3} = 0.7 + 1.6 + 2.7$
  
  $Y_{1,3} = 5.0$
  
  **Computing $Y_{2,1}$:**
  
  Here $i=2$, $j=1$. We take the second row of matrix $X$ and the first row of matrix $W$:
  
  $Y_{2,1} = (X_{2,1} \times W_{1,1}) + (X_{2,2} \times W_{1,2}) + (X_{2,3} \times W_{1,3})$
  
  $Y_{2,1} = (4 \times 0.1) + (5 \times 0.2) + (6 \times 0.3)$
  
  $Y_{2,1} = 0.4 + 1.0 + 1.8$
  
  $Y_{2,1} = 3.2$
  
  **Computing $Y_{2,2}$:**
  
  Here $i=2$, $j=2$. We take the second row of matrix $X$ and the second row of matrix $W$:
  
  $Y_{2,2} = (X_{2,1} \times W_{2,1}) + (X_{2,2} \times W_{2,2}) + (X_{2,3} \times W_{2,3})$
  
  $Y_{2,2} = (4 \times 0.4) + (5 \times 0.5) + (6 \times 0.6)$
  
  $Y_{2,2} = 1.6 + 2.5 + 3.6$
  
  $Y_{2,2} = 7.7$
  
  **Computing $Y_{2,3}$:**
  
  Here $i=2$, $j=3$. We take the second row of matrix $X$ and the third row of matrix $W$:
  
  $Y_{2,3} = (X_{2,1} \times W_{3,1}) + (X_{2,2} \times W_{3,2}) + (X_{2,3} \times W_{3,3})$
  
  $Y_{2,3} = (4 \times 0.7) + (5 \times 0.8) + (6 \times 0.9)$
  
  $Y_{2,3} = 2.8 + 4.0 + 5.4$
  
  $Y_{2,3} = 12.2$
  
  **Computing $Y_{3,1}$:**
  
  Here $i=3$, $j=1$. We take the third row of matrix $X$ and the first row of matrix $W$:
  
  $Y_{3,1} = (X_{3,1} \times W_{1,1}) + (X_{3,2} \times W_{1,2}) + (X_{3,3} \times W_{1,3})$
  
  $Y_{3,1} = (7 \times 0.1) + (8 \times 0.2) + (9 \times 0.3)$
  
  $Y_{3,1} = 0.7 + 1.6 + 2.7$
  
  $Y_{3,1} = 5.0$
  
  **Computing $Y_{3,2}$:**
  
  Here $i=3$, $j=2$. We take the third row of matrix $X$ and the second row of matrix $W$:
  
  $Y_{3,2} = (X_{3,1} \times W_{2,1}) + (X_{3,2} \times W_{2,2}) + (X_{3,3} \times W_{2,3})$
  
  $Y_{3,2} = (7 \times 0.4) + (8 \times 0.5) + (9 \times 0.6)$
  
  $Y_{3,2} = 2.8 + 4.0 + 5.4$
  
  $Y_{3,2} = 12.2$
  
  **Computing $Y_{3,3}$:**
  
  Here $i=3$, $j=3$. We take the third row of matrix $X$ and the third row of matrix $W$:
  
  $Y_{3,3} = (X_{3,1} \times W_{3,1}) + (X_{3,2} \times W_{3,2}) + (X_{3,3} \times W_{3,3})$
  
  $Y_{3,3} = (7 \times 0.7) + (8 \times 0.8) + (9 \times 0.9)$
  
  $Y_{3,3} = 4.9 + 6.4 + 8.1$
  
  $Y_{3,3} = 19.4$
  
  **4. Resulting matrix Y:**
  
  After all computations, we obtain the output matrix $Y$:
  
  ```
  Y = [[1.4, 3.2, 5.0],
       [3.2, 7.7, 12.2],
       [5.0, 12.2, 19.4]]
  ```

### 3. What Is a "Super Weight"

1. **Core idea**: A "super weight" is an element $W_{j,m}$ such that, when multiplied by the corresponding activation $X_{i,m}$, it contributes a **disproportionately large** amount to the output $Y_{i,j}$.  
2. **Dominance threshold**: For a given element $Y_{i,j}$, examine all terms $X_{i,k} \times W_{j,k}$. If there exists an index $m$ such that

$$
  \left|X_{i,m} \times W_{j,m}\right| 
    \gg \sum_{k \neq m} \left| X_{i,k} \times W_{j,k} \right|,
$$

then the pair $(j,m)$ "dominates" in $Y_{i,j}$.

3. **Generalization**: If, across different inputs $X$, the same index pair $(j,m)$ (i.e., weight $W_{j,m}$) repeatedly produces such dominant contributions, this $W_{j,m}$ is called a **super weight**.

## **_Let’s illustrate the concept of a "super weight" using 3x3 matrices, as in the previous example._**

**We’ll use the same input matrix X:**

```
X = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
```

**Now we’ll modify the weight matrix W to demonstrate a "super weight."** Suppose we want weight $W_{1,1}$ to exert dominant influence. We’ll make it significantly larger than the other weights in the first row of $W$:

```
W = [[10.0, 0.2, 0.3],  # Significantly increased weight W_{1,1}
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9]]
```

Here, $W_{1,1} = 10.0$ is much larger than $W_{1,2} = 0.2$ and $W_{1,3} = 0.3$.

**We’ll now compute the output matrix Y using this modified weight matrix W:**

Let’s focus on computing the elements of the first row of matrix $Y$, since the first row of $W$ contains the "super weight."

**Computing $Y_{1,1}$:**

$Y_{1,1} = (X_{1,1} \times W_{1,1}) + (X_{1,2} \times W_{1,2}) + (X_{1,3} \times W_{1,3})$

$Y_{1,1} = (1 \times 10.0) + (2 \times 0.2) + (3 \times 0.3)$

$Y_{1,1} = 10.0 + 0.4 + 0.9$

$Y_{1,1} = 11.3$

Now examine the contribution of each term:
- $|X_{1,1} \times W_{1,1}| = |1 \times 10.0| = 10.0$
- $|X_{1,2} \times W_{1,2}| = |2 \times 0.2| = 0.4$
- $|X_{1,3} \times W_{1,3}| = |3 \times 0.3| = 0.9$

Compare the dominant contribution to the sum of the others:

$|X_{1,1} \times W_{1,1}| = 10.0$

$\sum_{k \neq 1} |X_{1,k} \times W_{1,k}| = |0.4| + |0.9| = 1.3$

Clearly, $10.0 \gg 1.3$. Thus, for $Y_{1,1}$, the index pair $(j=1, m=1)$ dominates.

**Computing $Y_{1,2}$:**

$Y_{1,2} = (X_{1,1} \times W_{2,1}) + (X_{1,2} \times W_{2,2}) + (X_{1,3} \times W_{2,3})$

$Y_{1,2} = (1 \times 0.4) + (2 \times 0.5) + (3 \times 0.6)$

$Y_{1,2} = 0.4 + 1.0 + 1.8$

$Y_{1,2} = 3.2$

Here, no single term clearly dominates.

**Computing $Y_{1,3}$:**

$Y_{1,3} = (X_{1,1} \times W_{3,1}) + (X_{1,2} \times W_{3,2}) + (X_{1,3} \times W_{3,3})$

$Y_{1,3} = (1 \times 0.7) + (2 \times 0.8) + (3 \times 0.9)$

$Y_{1,3} = 0.7 + 1.6 + 2.7$

$Y_{1,3} = 5.0$

Again, no single term dominates here.

**Now consider another element where the "super weight" may manifest, for example, $Y_{2,1}$:**

$Y_{2,1} = (X_{2,1} \times W_{1,1}) + (X_{2,2} \times W_{1,2}) + (X_{2,3} \times W_{1,3})$

$Y_{2,1} = (4 \times 10.0) + (5 \times 0.2) + (6 \times 0.3)$

$Y_{2,1} = 40.0 + 1.0 + 1.8$

$Y_{2,1} = 42.8$

Compare contributions:
- $|X_{2,1} \times W_{1,1}| = |4 \times 10.0| = 40.0$
- $|X_{2,2} \times W_{1,2}| = |5 \times 0.2| = 1.0$
- $|X_{2,3} \times W_{1,3}| = |6 \times 0.3| = 1.8$

Compare the dominant contribution to the sum of the others:

$|X_{2,1} \times W_{1,1}| = 40.0$

$\sum_{k \neq 1} |X_{2,k} \times W_{1,k}| = |1.0| + |1.8| = 2.8$

Again, $40.0 \gg 2.8$, and the index pair $(j=1, m=1)$ dominates in $Y_{2,1}$.

**Generalizing to a "super weight":**

In our example, weight $W_{1,1} = 10.0$ is a candidate "super weight." If, when feeding different input matrices $X$, we repeatedly observe that the term involving $W_{1,1}$ (i.e., $X_{i,1} \times W_{1,1}$) vastly exceeds the sum of all other terms in computing various $Y_{i,j}$ elements, we would classify $W_{1,1}$ as a "super weight."

**Example with different inputs:**

Suppose we have another input matrix $X'$:

```
X' = [[2, 1, 0],
      [5, 0, 1],
      [9, 2, 3]]
```

Compute $Y'_{1,1}$ using the same weight matrix $W$:

$Y'_{1,1} = (X'_{1,1} \times W_{1,1}) + (X'_{1,2} \times W_{1,2}) + (X'_{1,3} \times W_{1,3})$

$Y'_{1,1} = (2 \times 10.0) + (1 \times 0.2) + (0 \times 0.3)$

$Y'_{1,1} = 20.0 + 0.2 + 0.0$

$Y'_{1,1} = 20.2$

Contributions:
- $|X'_{1,1} \times W_{1,1}| = |2 \times 10.0| = 20.0$
  
- $|X'_{1,2} \times W_{1,2}| = |1 \times 0.2| = 0.2$
  
- $|X'_{1,3} \times W_{1,3}| = |0 \times 0.3| = 0.0$

Comparison: $20.0 \gg 0.2 + 0.0$. The pair $(j=1, m=1)$ dominates again.

If this pattern repeats across many different input matrices $X$, weight $W_{1,1}$ would be confidently classified as a "super weight," as it exerts a disproportionately large influence on output values.

### 4. What Is a "Super Activation"

1. **From weight to activation**: A super weight produces an anomalously large output $Y_{i,j}$. This output is called a "super activation":

$$
  Y_{i,j} \approx X_{i,m} \times W_{j,m},
$$

when the contribution of a single pair $(k = m)$ vastly exceeds the combined contribution of all other components.

2. **Outlier definition**: Formally, $Y_{i,j}$ is a super activation if its absolute value lies many standard deviations away from the mean across all elements. For example:

$$
  \left|Y_{i,j}\right| > \mu_Y + \gamma \,\sigma_Y,
$$

where $\mu_Y$ and $\sigma_Y$ are the mean and standard deviation over the set $\{Y_{i',j'}\}$. The parameter $\gamma$ determines how extreme an "outlier" must be relative to the mean.

## **_Let’s now unpack the concept of a "super activation" using the previous "super weight" example._**

**We’ll use the results from the earlier example with super weight $W_{1,1} = 10.0$.**

Recall, the weight matrix $W$ was:

```
W = [[10.0, 0.2, 0.3],
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9]]
```

And the input matrix $X$:

```
X = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
```

The resulting output matrix $Y$, computed with this "super weight," was:

```
Y = [[11.3, 3.2, 5.0],
     [42.8, 7.7, 12.2],
     [74.3, 12.2, 19.4]]
```

### 1. Super Activation as a Result of Dominant Contribution

As shown earlier, in the computation of $Y_{1,1}$, the contribution from the pair $(j=1, m=1)$ dominated:

$Y_{1,1} = (1 \times 10.0) + (2 \times 0.2) + (3 \times 0.3) = 10.0 + 0.4 + 0.9 = 11.3$

Here, $10.0$ is significantly larger than $0.4 + 0.9 = 1.3$. Thus, $Y_{1,1} \approx X_{1,1} \times W_{1,1} = 1 \times 10.0 = 10.0$. In this context, $Y_{1,1} = 11.3$ can be considered a **super activation**, as it arises from the dominant contribution of the "super weight" $W_{1,1}$.

Similarly, in computing $Y_{2,1}$:

$Y_{2,1} = (4 \times 10.0) + (5 \times 0.2) + (6 \times 0.3) = 40.0 + 1.0 + 1.8 = 42.8$

Here, $40.0$ is significantly larger than $1.0 + 1.8 = 2.8$. Therefore, $Y_{2,1} \approx X_{2,1} \times W_{1,1} = 4 \times 10.0 = 40.0$. $Y_{2,1} = 42.8$ is also a **super activation**.

And in computing $Y_{3,1}$:

$Y_{3,1} = (7 \times 10.0) + (8 \times 0.2) + (9 \times 0.3) = 70.0 + 1.6 + 2.7 = 74.3$

Here, $70.0$ is significantly larger than $1.6 + 2.7 = 4.3$. Hence, $Y_{3,1} \approx X_{3,1} \times W_{1,1} = 7 \times 10.0 = 70.0$. $Y_{3,1} = 74.3$ is also a **super activation**.

In these cases, the "super weight" $W_{1,1}$, combined with the corresponding activations $X_{i,1}$, yields output values $Y_{i,1}$ where the contribution of a single term is disproportionately large.

### 2. Super Activation as an Outlier

Now let’s consider the definition of a "super activation" as an outlier. For this, we need to compute the mean ($\mu_Y$) and standard deviation ($\sigma_Y$) across all elements of matrix $Y$:

```
Y = [[11.3, 3.2, 5.0],
     [42.8, 7.7, 12.2],
     [74.3, 12.2, 19.4]]
```

Elements of matrix $Y$: 11.3, 3.2, 5.0, 42.8, 7.7, 12.2, 74.3, 12.2, 19.4

**Computing the mean ($\mu_Y$):**

$\mu_Y = \frac{11.3 + 3.2 + 5.0 + 42.8 + 7.7 + 12.2 + 74.3 + 12.2 + 19.4}{9}$  
$\mu_Y = \frac{188.1}{9} \approx 20.9$

**Computing the standard deviation ($\sigma_Y$):**

First, compute the variance ($\sigma_Y^2$):

$\sigma_Y^2 = \frac{1}{9} \sum_{i,j} (Y_{i,j} - \mu_Y)^2$

$\sigma_Y^2 = \frac{1}{9} [ (11.3-20.9)^2 + (3.2-20.9)^2 + (5.0-20.9)^2 + (42.8-20.9)^2 + (7.7-20.9)^2 + (12.2-20.9)^2 + (74.3-20.9)^2 + (12.2-20.9)^2 + (19.4-20.9)^2 ]$

$\sigma_Y^2 = \frac{1}{9} [ (-9.6)^2 + (-17.7)^2 + (-15.9)^2 + (21.9)^2 + (-13.2)^2 + (-8.7)^2 + (53.4)^2 + (-8.7)^2 + (-1.5)^2 ]$

$\sigma_Y^2 = \frac{1}{9} [ 92.16 + 313.29 + 252.81 + 479.61 + 174.24 + 75.69 + 2851.56 + 75.69 + 2.25 ]$

$\sigma_Y^2 = \frac{1}{9} [ 4317.3 ] \approx 479.7$

Now compute the standard deviation:

$\sigma_Y = \sqrt{479.7} \approx 21.9$

**Identifying super activations as outliers:**

Let $\gamma = 2$. Then the threshold for a super activation is:

$|\text{Super activation}| > \mu_Y + \gamma \sigma_Y = 20.9 + 2 \times 21.9 = 20.9 + 43.8 = 64.7$

Now evaluate the elements of matrix $Y$:

- $|Y_{1,1}| = 11.3 \ngtr 64.7$  
- $|Y_{1,2}| = 3.2 \ngtr 64.7$  
- $|Y_{1,3}| = 5.0 \ngtr 64.7$  
- $|Y_{2,1}| = 42.8 \ngtr 64.7$  
- $|Y_{2,2}| = 7.7 \ngtr 64.7$  
- $|Y_{2,3}| = 12.2 \ngtr 64.7$  
- $|Y_{3,1}| = 74.3 > 64.7$  **Super activation**  
- $|Y_{3,2}| = 12.2 \ngtr 64.7$  
- $|Y_{3,3}| = 19.4 \ngtr 64.7$  

According to this criterion, only $Y_{3,1} = 74.3$ qualifies as a super activation.

If we chose a smaller $\gamma$, say $\gamma = 1$, the threshold would be:

$|\text{Super activation}| > 20.9 + 1 \times 21.9 = 42.8$

In this case, super activations would be:

- $|Y_{2,1}| = 42.8 \ngtr 42.8$ (on the boundary)  
- $|Y_{3,1}| = 74.3 > 42.8$  **Super activation**

### 5. Verifying the Presence of a Super Weight

1. **Zeroing operation**: Let $\widehat{W}$ be a copy of $W$ with candidate weight $W_{j,m}$ set to 0:

```math
  \widehat{W}_{j',k'} = 
  \begin{cases}
  0, & \text{if } (j' = j) \text{ and } (k' = m),\\
  W_{j',k'}, & \text{otherwise}.
  \end{cases}
```

2. **Perplexity comparison**: If the model with weights $\widehat{W}$ yields a perplexity $PPL_\text{new}$ that is orders of magnitude higher than the original $PPL_\text{orig}$, then $W_{j,m}$ is a super weight.

$$
  \text{If } 
  \frac{PPL_\text{new}}{PPL_\text{orig}} \gg 1,
  \quad 
  \text{then } W_{j,m} \text{ is a super weight.}
$$

### 6. Propagation of Super Activations

In a multi-layer architecture, the output $Y$ is fed into subsequent blocks (through nonlinear functions, skip connections, etc.). A super activation in $Y_{i,j}$ may be partially or fully preserved in later layers, triggering a cascading effect:

$$
  \tilde{X} = \phi(Y), \quad 
  \tilde{Y} = \tilde{X} \, W_{\text{next}}^\mathsf{T},
$$

where $\phi(\cdot)$ denotes an activation function (e.g., ReLU, GeLU).

---

### Final Remarks on the Role of Super Weights and Super Activations

1. **Super weights** may not be the largest in magnitude within weight matrix $W$, but they can "resonate" with large input values $X_{i m}$, producing "super activations."  
2. **Super activations** can be traced across layers and often persist through activation functions and skip connections.  
3. Removing a super weight causes super activations to vanish (or drastically diminish), leading to a sharp drop in performance (generation quality, zero-shot accuracy, etc.).  
4. Preserving super weights during quantization (or treating them specially—for example, storing them in higher precision) helps avoid catastrophic model degradation.

---

## Final Summary

- A **super weight** is a scalar parameter $W_{j m}$ in the $\mathrm{down\_proj}$ matrix (or another layer) whose contribution to activations $Y_{i j}$ is **dominant**, leading to "super activations."  
- A **super activation** is an extremely large activation value (element $Y_{i j}$) caused by the dominance of the pair $(X_{i m}, W_{j m})$.  
- Removing a single super weight collapses model performance, as super activations disappear or sharply decrease—confirming a causal relationship.  
- In the linear transformation formalism $Y = XW^\mathsf{T}$, super weights correspond to index pairs $(j,m)$ for which the product $X_{i m} \cdot W_{j m}$ is orders of magnitude larger than the sum of all other components for a given position $(i,j)$.

Thus, "super weights" and "super activations" represent a phenomenon of interaction between a large (or "resonant") weight and specific input activations, which ultimately has a critical impact on the performance of the entire large language model.