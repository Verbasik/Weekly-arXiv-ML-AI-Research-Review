# Continuous Thought Machines

## Table of Contents  
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Continuous Thought Machine Architecture](#continuous-thought-machine-architecture)
3. [Dissecting the Mathematical Apparatus Under the Hood of CTM](#dissecting-the-mathematical-apparatus-under-the-hood-of-ctm)
4. [Internal Recurrent Time and Synapses](#internal-recurrent-time-and-synapses)
5. [Neuron-Level Models and Neural Timing](#neuron-level-models-and-neural-timing)
6. [Representation of Neuron Synchronization](#representation-of-neuron-synchronization)
7. [Formalization of the Loss Function](#formalization-of-the-loss-function)
8. [Adaptive Reasoning Depth](#adaptive-reasoning-depth)
9. [Conclusion](#conclusion)

## **1. Introduction and Motivation**

Modern neural networks have achieved remarkable success, yet remain simplified compared to the biological brain. In particular, they typically do not account for **neuronal temporal dynamics**—the precise timing of spikes and synchronization of activity, which play a critical role in biological neural networks. Artificial neurons usually output only a single static activation value, ignoring *when* a neuron fires relative to others. Biological principles such as spike-timing-dependent plasticity (STDP) indicate that timing is essential for learning and information processing in the brain. The gap between flexible human thought and current AI suggests that certain fundamental components related to **temporal signal processing** are missing from existing models.

**Continuous Thought Machine (CTM)** is a novel neural network architecture proposed by Sakana AI that *restores time to the foundation of neural computation*. The CTM model is explicitly designed to use neuron synchronization as a mechanism for reasoning. Unlike traditional networks, CTM equips each neuron with information about its past activations, enabling it to adapt its current behavior based on temporal patterns. This allows CTM to "think" through a task step-by-step, coordinating neurons over time and making its decision process interpretable to humans. Research demonstrates that this approach enhances performance on complex tasks and improves model efficiency across diverse benchmarks. CTM represents a significant step toward bridging artificial and biological neural networks, unlocking new possibilities for AI.

## **2. Continuous Thought Machine Architecture**

**Continuous Thought Machine** introduces three key innovations to neural network architecture:

1. *Internal recurrent time dimension* (separate from input data time), upon which the model's "thinking" dynamics unfold;
2. *Neuron-Level Models (NLMs)*—individual parameters per neuron that process the temporal history of input signals;
3. *Representation of neuron synchronization*—using **synchronization in activations** directly as the latent feature space for decision-making. Below, we detail CTM components and their mathematical formalization.

![Figure 1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-20/assets/Image_01.jpeg  )

**Fig. 1:** Architecture of the Continuous Thought Machine (CTM), labeling key components.

1. **Synapse model** — Synaptic model (blue connections) computes *pre-activations* $a^\tau$ for each neuron, modeling inter-neuronal connections;
2. **History (pre-activations)** — Buffer of $M$ most recent pre-activations for each neuron $A^\tau$ (shown as waves);
3. **Neuron-Level Models** — Individual neuron models (red) $g_{\theta_d}$ process history $A^\tau_d$ and output *post-activations* $z^{\tau+1}_d$;
4. **Post-activations** — Output activation vector $z^{\tau+1}$ of all $D$ neurons at current step;
5. **History (post-activations)** — Cumulative history of post-activations $Z^\tau$ since initialization;
6. **Synchronization matrix $S^\tau$** — Synchronization matrix computed as dot products between temporal activation series of neurons;
7. **Selected neuron pairs** — Selection of a subset of elements from $S^\tau$ (neuron pairs) corresponding to latent synchronization features;
8. **Latent representation** — Synchronization vector (green) derived from selected $S^\tau$ elements, split into two parts: for output and attention;
9. **OUT/ATTN projections** — Linear layers: $W_{\text{out}}$ projects the latent vector to output (e.g., classes), $W_{\text{in}}$ to attention query vector *q*;
10. **Data modulation (Attention output)** — Using *q*, the model extracts relevant information from inputs (via attention mechanism *ATTN*, yellow block $o^\tau$); this modulated information is combined with current post-activations, closing the loop for the next internal tick.

## **3. Dissecting the Mathematical Apparatus Under the Hood of CTM**

**Applying CTM to an NLP Task: Next-Token Prediction**

Consider a typical natural language processing task: **predicting the next token** in a sequence. Let the model's input be a tokenized text sequence:

$$
\text{Input: } \quad x = (x_1, x_2, \dots, x_{t})
$$

The model's task is to predict token $x_{t+1}$. To do so, CTM will **iteratively reason over the input sequence** across internal steps $\tau = 1, 2, \dots, T$, gradually forming and refining its prediction as neurons synchronize.

### Applying CTM Architecture Components:

<details> 
    <summary><em><strong>1. Synapse model</strong></em></summary>

### **1. Synapse model**

At each internal step $\tau$, the synapse model computes a vector of *pre-activations* $a^\tau \in \mathbb{R}^D$ via the synaptic model:

$$
a^\tau = f_{\theta_{\text{syn}}}([z^\tau, o^\tau])
$$

where:

* $z^\tau \in \mathbb{R}^D$ — post-activations of all neurons from the previous step;
* $o^\tau \in \mathbb{R}^{d_{\text{attn}}}$ — attention vector (modulated information from input);
* $f_{\theta_{\text{syn}}}$ — MLP with U-Net-like architecture.

In NLP tasks:

* Input $o^\tau$ is formed from tokens $x = (x_1, ..., x_t)$ encoded as embeddings via encoder $F(x) \in \mathbb{R}^{t \times d_{\text{attn}}}$.
* Output $a^\tau$ can be interpreted as a "neuron evaluation" based on attention state and self-context.

<details> 
    <summary><em><strong>🧠 MLP with U-Net-like Architecture</strong></em></summary>

### **MLP with U-Net-like Architecture for Synapse Model**

In the context of CTM, a **U-Net-like MLP** refers to a fully connected network organized with a "contracting–expanding" structure and *skip connections*, analogous to the classical U-Net in computer vision, but applied to a one-dimensional input feature vector.

Below, we step-by-step dissect how such a module $f_{\theta_{\text{syn}}}$ may be constructed.

---

### 1. Input Vector and Its Dimensions

At step $\tau$, we have two vectors:

* $z^\tau \in \mathbb{R}^D$ — post-activations of all $D$ neurons from the prior step.
* $o^\tau \in \mathbb{R}^{d_{\text{attn}}}$ — attention output, extracted from input data.

We concatenate them into a single vector:

$$
v^\tau = 
\begin{bmatrix}
z^\tau \\[4pt]
o^\tau
\end{bmatrix}
\;\in\;
\mathbb{R}^{\,D + d_{\text{attn}}}\
$$

#### **Example:**

1. Set dimensions: $D=3$, $d_{\text{attn}}=2$

2. Choose concrete vectors:

   $$
   z^τ = (0.1,\;-0.4,\;0.7)
   $$

   $$
   o^τ = (0.5,\;-0.2)
   $$

3. Concatenate:

   $$
   v^τ = 
   \begin{bmatrix}
     0.1\\
     -0.4\\
     0.7\\
     0.5\\
     -0.2
   \end{bmatrix}
   \in \mathbb R^5
   $$

**Python pseudocode:**

```python
import numpy as np

# Dimensions
D = 3
d_attn = 2

# Sample data
z = np.array([0.1, -0.4, 0.7])       # shape (3,)
o = np.array([0.5, -0.2])            # shape (2,)

# Concatenation
v = np.concatenate([z, o])           # shape (5,)
print("v =", v)                      # [ 0.1 -0.4  0.7  0.5 -0.2]
print("Shape of v:", v.shape)        # (5,)
```

Thus, vector $v^τ$ of dimension $D + d\_{\text{attn}}$ contains both information from previous neuron post-activations and details from the attention mechanism.

---

### 2. Contracting Path

The goal of the contracting path is to progressively reduce the dimensionality of the feature space, extracting high-level representations. Suppose we have $L$ contraction levels. At each level $\ell = 1,2,\dots,L$, two operations are performed:

1. **Fully connected layer** (Linear) with dimensionality reduction:

   $$
     e^\ell = \sigma\bigl(W_e^\ell\,e^{\ell-1} + b_e^\ell\bigr),
     \quad
     e^0 \equiv v^\tau,
   $$

   where:

   - $\sigma(\cdot)$ — nonlinearity (ReLU, GELU, etc.);
   - $W_e^\ell$ has dimensions $\;d_{\ell}\times d_{\ell-1}$;
   - $d_0 = D + d_{\text{attn}}$;
   - $d_\ell < d_{\ell-1}$.

2. **Additional contraction** (optional)—e.g., a second Linear or BatchNorm+ReLU, but key is to record the result as the "contracting" output of the level:

   $$
     \tilde e^\ell = \sigma\bigl(W_{\tilde e}^\ell\,e^\ell + b_{\tilde e}^\ell\bigr).
   $$

Thus, after $L$ levels, we obtain a "bottleneck":

$$
b = \tilde e^L \in \mathbb{R}^{d_L},
$$

where $d_L$ is the minimum dimension.

#### **Example:**

Take:

- $d_0 = D + d_{\mathrm{attn}} = 5$;  
- Number of levels $L = 2$;  
- At first level $d_1 = 4$, at second $d_2 = 2$.

Let input vector be:

$$
e^0 = v^\tau = \begin{pmatrix}0.1\\ -0.4\\ 0.7\\ 0.5\\ -0.2\end{pmatrix}
$$

Define simple matrices and biases:

1. **Level $\ell=1$:**

   $$
   W_e^1 = 
   \begin{pmatrix}
     1 & 0 & 0 & 0 & 0 \\
     0 & 1 & 0 & 0 & 0 \\
     0 & 0 & 1 & 0 & 0 \\
     0 & 0 & 0 & 1 & 0 
   \end{pmatrix},\quad
   b_e^1 = \begin{pmatrix}0\\0\\0\\0\end{pmatrix}
   $$

   Then:

   $$
   e^1 = \mathrm{ReLU}(W_e^1 e^0 + b_e^1)
       = \mathrm{ReLU}\!\bigl([0.1,\,-0.4,\,0.7,\,0.5]^\top\bigr)
       = [0.1,\,0,\,0.7,\,0.5]^\top
   $$

   Additional contraction:

   $$
   W_{\tilde e}^1 = I_{4},\quad b_{\tilde e}^1=0,\qquad
   \tilde e^1 = \mathrm{ReLU}(e^1) = [0.1,\,0,\,0.7,\,0.5]^\top
   $$

2. **Level $\ell=2$:**

   $$
   W_e^2 = 
   \begin{pmatrix}
     0 & 1 & 0 & 0 \\
     0 & 0 & 1 & 0 
   \end{pmatrix},\quad
   b_e^2 = \begin{pmatrix}0\\0\end{pmatrix}
   $$

   Then:

   $$
   e^2 = \mathrm{ReLU}(W_e^2 \tilde e^1 + b_e^2)
       = \mathrm{ReLU}\!\bigl([0,\,0.7]^\top\bigr)
       = [0,\,0.7]^\top
   $$

   And additional:

   $$
   W_{\tilde e}^2 = I_{2},\quad b_{\tilde e}^2=0,\qquad
   \tilde e^2 = \mathrm{ReLU}(e^2) = [0,\,0.7]^\top
   $$

Final "bottleneck":

$$
b = \tilde e^2 = \begin{pmatrix}0\\0.7\end{pmatrix}
\;\in\;\mathbb R^{d_2},\quad d_2=2.
$$


#### **Conclusion:**

1. Concatenation:

   $$
   v^\tau = 
   \begin{bmatrix}
     z^\tau \\[4pt]
     o^\tau
   \end{bmatrix}
   \;\in\;
   \mathbb{R}^{D + d_{\text{attn}}}
   $$

   where upper components are $z^\tau\in\mathbb R^D$, lower are $o^\tau\in\mathbb R^{d\_{\text{attn}}}$

2. "Bottleneck":

   After two contraction levels $L=2$, we obtain:

   $$
   b = \tilde e^2 = 
   \begin{pmatrix}
     0\\
     0.7
   \end{pmatrix}
   \;\in\;
   \mathbb{R}^{d_2},
   \quad d_2 = 2
   $$

This is precisely the goal of the contracting path: progressively compress the input vector of dimension $d\_0=D+d\_{\mathrm{attn}}$ down to $d\_L$, here to 2.

---

### 3. Expanding Path with Skip Connections

Now we begin to **expand** the representation back to dimensionality $D$. At each level, a **skip connection** from the contracting path at the same level is used:

For $\ell = L, L-1, \dots, 1$:

1. **Concatenate** current decoder activation with corresponding contracting activation:

   $$
     c^\ell = 
     \begin{bmatrix}
       \tilde e^\ell \\[4pt]
       d^{\ell+1}
     \end{bmatrix},
     \quad
     d^{L+1} \equiv b.
   $$

   Here $c^\ell \in \mathbb{R}^{\,d_\ell + d_\ell}$.

2. **Expanding fully connected layer**:

   $$
     d^\ell = \sigma\bigl(W_d^\ell\,c^\ell + b_d^\ell\bigr),
   $$

   where $W_d^\ell\colon \mathbb{R}^{d_\ell + d_\ell} \to \mathbb{R}^{d_{\ell-1}}$.

The result after level $\ell=1$ is a vector:

$$
d^1 \in \mathbb{R}^{d_0} = \mathbb{R}^{\,D + d_{\text{attn}}}.
$$

#### **Example:**

During decoding, we reconstruct the representation back to dimensionality $d_0 = D + d_{\mathrm{attn}}$, using skip connections from the contracting path.

> **Notation for this example:**
>
> * $D = 3$, $d_{\mathrm{attn}} = 2$ → $d_0 = 5$
> * Number of levels $L = 2$
> * Contracting level dimensions: $d_1 = 4$, $d_2 = 2$
> * Bottleneck $b = \tilde e^2 = [0,\;0.7]^\top$

#### Step ℓ = 2 (Level $L$)

1. **Skip from contracting path:**
   $\tilde e^2 = [0,\;0.7]^\top$

2. **Previous decoder state:**
   $d^3 \equiv b = [0,\;0.7]^\top$

3. **Concatenation:**

   $$
     c^2 =
     \begin{bmatrix}
       \tilde e^2 \\[4pt]
       d^3
     \end{bmatrix}
     =
     \begin{bmatrix}
       0 \\[2pt]
       0.7 \\[2pt]
       0 \\[2pt]
       0.7
     \end{bmatrix}
     \;\in\;\mathbb R^{\,d_2 + d_2} = \mathbb R^4
   $$

4. **Expanding linear layer:**
   For simplicity, choose:

   $$
   W_d^2 =
   \begin{pmatrix}
     1 & 0 & 0 & 0 \\
     0 & 1 & 0 & 0 \\
     0 & 0 & 1 & 0 \\
     0 & 0 & 0 & 1 \\
     0 & 0 & 0 & 0
   \end{pmatrix}_{5\times 4},\quad
   b_d^2 = \mathbf{0}_{5}.
   $$

   Then:

   $$
   d^2 = \mathrm{ReLU}(W_d^2\,c^2 + b_d^2)
       = \mathrm{ReLU}\!\bigl([\,0,\;0.7,\;0,\;0.7,\;0\,]^\top\bigr)
       = [\,0,\;0.7,\;0,\;0.7,\;0\,]^\top
   \;\in\;\mathbb R^5.
   $$

#### Step ℓ = 1

1. **Skip from contracting path:**
   $\tilde e^1 = [0.1,\;0,\;0.7,\;0.5]^\top$

2. **Previous decoder state:**
   $d^2 = [0,\;0.7,\;0,\;0.7,\;0]^\top$

3. **Concatenation:**

   $$
     c^1 =
     \begin{bmatrix}
       \tilde e^1 \\[4pt]
       d^2
     \end{bmatrix}
     =
     \begin{bmatrix}
       0.1 \\[2pt]
       0 \\[2pt]
       0.7 \\[2pt]
       0.5 \\[2pt]
       0 \\[2pt]
       0 \\[2pt]
       0.7 \\[2pt]
       0 \\[2pt]
       0.7 \\[2pt]
       0
     \end{bmatrix}
     \;\in\;\mathbb R^{\,d_1 + d_1} = \mathbb R^8
   $$

4. **Expanding linear layer:**
   Let:

   $$
   W_d^1 \colon \mathbb R^8 \to \mathbb R^5,\quad
   W_d^1 = 
   \begin{pmatrix}
     I_5 & \mathbf{0}_{5\times3}
   \end{pmatrix},
   \quad b_d^1 = \mathbf{0}_5.
   $$

   Then:

   $$
   d^1 = \mathrm{ReLU}(W_d^1\,c^1 + b_d^1)
       = \mathrm{ReLU}\!\bigl([\,0.1,\;0,\;0.7,\;0.5,\;0\,]^\top\bigr)
       = [\,0.1,\;0,\;0.7,\;0.5,\;0\,]^\top
   = v^\tau.
   $$

Finally, after level ℓ=1, we recover the vector $d^1\in\mathbb R^5$, identical to the original $v^\tau$, as required.

```python
import numpy as np

# Example decoding for L=2, D=3, d_attn=2
tilde_e2 = np.array([0, 0.7])
d3 = tilde_e2.copy()  # bottleneck

# Step ℓ=2
c2 = np.concatenate([tilde_e2, d3])       # shape (4,)
d2 = np.maximum(np.dot(np.vstack([np.eye(4), np.zeros((1,4))]), c2), 0)

# Step ℓ=1
tilde_e1 = np.array([0.1, 0, 0.7, 0.5])
c1 = np.concatenate([tilde_e1, d2])      # shape (8,)
d1 = np.maximum(np.dot(np.hstack([np.eye(5), np.zeros((5,3))]), c1), 0)

print("d1 =", d1)  # restores original v = [0.1, 0, 0.7, 0.5, 0]
```

#### **Conclusion:**

1. **Concatenation**

   $$
   v^\tau = 
   \begin{bmatrix}
     z^\tau \\[4pt]
     o^\tau
   \end{bmatrix}
   \;\in\;
   \mathbb{R}^{D + d_{\text{attn}}}
   $$

   where upper components are $z^\tau\in\mathbb R^D$, lower are $o^\tau\in\mathbb R^{d\_{\text{attn}}}$.

2. **"Bottleneck"**

   After two contraction levels $L=2$, we obtained:

   $$
   b = \tilde e^2 = 
   \begin{pmatrix}
     0\\
     0.7
   \end{pmatrix}
   \;\in\;
   \mathbb{R}^{d_2},
   \quad d_2 = 2.
   $$

3. **Expanding with skip connections**

   After decoding at levels $\ell=2$ and $\ell=1$, combining hidden representations from the contracting path and applying expanding linear layers, we obtain:

   $$
   d^1 = 
   \mathrm{ReLU}\bigl(W_d^1\,c^1 + b_d^1\bigr)
   = v^\tau
   \;\in\;
   \mathbb{R}^{D + d_{\text{attn}}}.
   $$

   Thus, at the output of the expanding path, we **reconstruct the original vector** $v^\tau$, completing the "compression–expansion" cycle with information preserved via skip connections.

---

### 4. Output Layer

To obtain the **pre-activations** $a^\tau \in \mathbb{R}^D$, extract the first $D$ components from the expanded vector (or apply a separate linear layer):

$$
a^\tau = W_{\text{out}}^{\text{syn}}\;d^1 + b_{\text{out}}^{\text{syn}},
\qquad
W_{\text{out}}^{\text{syn}}\colon \mathbb{R}^{D + d_{\text{attn}}}\to\mathbb{R}^D.
$$

#### **Example:**

* Dimensions: $D=3$, $d_{\mathrm{attn}}=2$ → $D + d_{\mathrm{attn}} = 5$.
* Suppose after the expanding path we obtain:

  $$
    d^1 = 
    \begin{pmatrix}
      0.1\\
      0\\
      0.7\\
      0.5\\
      0
    \end{pmatrix}
    \in\mathbb R^5.
  $$
* Define output layer parameters:

  $$
    W_{\text{out}}^{\text{syn}}
      = \bigl[I_3\;\bigm|\;\mathbf{0}_{3\times2}\bigr]
      = 
      \begin{pmatrix}
        1 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0
      \end{pmatrix},
    \quad
    b_{\text{out}}^{\text{syn}}
      = 
      \begin{pmatrix}
        0.05\\
       -0.05\\
        0.10
      \end{pmatrix}.
  $$
* Then:

  $$
    a^\tau
    = W_{\text{out}}^{\text{syn}}\,d^1 + b_{\text{out}}^{\text{syn}}
    = 
    \begin{pmatrix}
      1 & 0 & 0 & 0 & 0\\
      0 & 1 & 0 & 0 & 0\\
      0 & 0 & 1 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
      0.1\\0\\0.7\\0.5\\0
    \end{pmatrix}
    +
    \begin{pmatrix}
      0.05\\-0.05\\0.10
    \end{pmatrix}
    =
    \begin{pmatrix}
      0.1 + 0.05\\
      0   - 0.05\\
      0.7 + 0.10
    \end{pmatrix}
    =
    \begin{pmatrix}
      0.15\\
     -0.05\\
      0.80
    \end{pmatrix}.
  $$

Thus, the output block reconstructs the pre-activation vector $a^\tau$ of dimension $D$, ready for transmission to the next internal CTM step.

---

### 5. Schematic Pseudocode

```python
def f_syn(v: Tensor) -> Tensor:
    # v.shape = (batch_size, D + d_attn)
    
    # Contracting path
    e = v
    enc_skips = []
    for ℓ in range(1, L+1):
        e = ReLU(Linear_e[ℓ](e))        # d_{ℓ} <-- d_{ℓ-1}
        enc_skips.append(e)             # save for skip
        e = ReLU(Linear_e_tilde[ℓ](e))  

    b = e  # bottleneck vector, shape=(batch, d_L)

    # Expanding path
    d = b
    for ℓ in reversed(range(1, L+1)):
        skip = enc_skips[ℓ-1]          # corresponding level
        d = torch.cat([skip, d], dim=-1)
        d = ReLU(Linear_d[ℓ](d))       # d_{ℓ-1} <-- 2*d_{ℓ}

    # Output
    a = Linear_out(d)                  # (batch, D)
    return a
```

---

### 6. Why So Many Levels and Skip Connections?

* **Contraction** enables $f_{\theta_{\text{syn}}}$ to capture global, multidimensional dependencies between different parts of the vector $[z^\tau, o^\tau]$, compressing information into a narrow "bottleneck".
* **Skip connections** ensure that precise local (low-level) information is not lost during compression: it is directly transmitted to the reconstruction stage, ensuring training stability and preservation of fine details.
* **Expansion** restores the final feature vector dimension, enriched by the results of global aggregation.

#### Final Formula

Collectively:

$$
\begin{aligned}
e^0 &= [z^\tau; o^\tau],\\
e^\ell &= \sigma\bigl(W_e^\ell\,e^{\ell-1} + b_e^\ell\bigr),\quad
\tilde e^\ell = \sigma\bigl(W_{\tilde e}^\ell\,e^\ell + b_{\tilde e}^\ell\bigr),\\
b &= \tilde e^L,\\
c^\ell &= [\,\tilde e^\ell; d^{\ell+1}],\quad
d^\ell = \sigma\bigl(W_d^\ell\,c^\ell + b_d^\ell\bigr),\\
a^\tau &= W_{\text{out}}^{\text{syn}}\,d^1 + b_{\text{out}}^{\text{syn}}.
\end{aligned}
$$

### 7. Conceptual Conclusion

The U-Net-style MLP with contracting-expanding architecture and skip connections in CTM is designed not merely to detect dependencies between vectors $z^\tau$ and $o^\tau$, but to perform **multiscale, flexible, and robust** processing of their joint representation. It forms a rich, hierarchical mapping that serves as the source of pre-activations $a^\tau$, accounting for both global and local patterns while ensuring stable learning within the recurrent loop:

#### 1. Extraction of Multiscale Interactions

* **Global dependencies:** The contracting path compresses the combined vector $\bigl[z^\tau; o^\tau\bigr]$ into a narrow "bottleneck" $b$, where the network extracts generalized, high-level interaction patterns.
* **Local details:** Thanks to skip connections from corresponding levels, raw compression signals are passed directly to the expansion stage, preserving precise, low-level dependencies within each component of the original vector.

#### 2. Hierarchical Integration

The expanding path "unfolds" the representation back to dimensionality $D + d_{\text{attn}}$, at each level blending the global generalization from the bottleneck $b$ with local features from skip connections. Consequently, each element of the output vector $a^\tau$ incorporates both the "big picture" and fine-grained nuances of neuron synchronization.

#### 3. Training Stability and Efficiency

Skip connections provide a direct path for gradients from deep layers back to the input, eliminating the vanishing gradient problem when the module is applied repeatedly within CTM's recurrent loop. This is critical for reliable learning of the model's "thought" iterations.

#### 4. Scalability Flexibility

The number of levels $L$ and dimensions $d_\ell$ can be increased for complex tasks or reduced for simpler ones, while preserving rich representational capacity. This adaptability allows the synaptic U-Net architecture to be applied across diverse volumes and types of input signals.

#### 5. Balance of Generalization and Detail

The "bottleneck" enables learning generalized relationships between neurons, while skip connections preserve critical fine details. This balance prevents both excessive simplification and overfitting, which is essential for forming accurate pre-activations $a^\tau$ and the entire subsequent reasoning process in CTM.

---

### 8. Conclusion

Thus, the **U-Net-type MLP** in CTM is a compact yet powerful mechanism for "synaptic" processing that multiscale integrates both global and local connections within the combined space $[z^\tau, o^\tau]$, ensures stable gradient flow, and flexibly adapts to task complexity, generating expressive pre-activations for subsequent stages of internal reasoning.

</details>

<!-- Checkpoint: 🧠 U-Net-MLP -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — 🧠 U-Net-MLP:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    The contracting path learns global dependencies, the "bottleneck" provides generalization, and the expanding path with skip connections restores local details; this topology stabilizes gradients and ensures a rich hierarchical representation for computing <em>a<sup>τ</sup></em>.
  </p>
</div>

</details>

On each internal tick $\tau$, the **Synapse model**:

1. **Accepts** the combined state of current internal neuron activations $z^{\tau}$ and the filtered attention output $o^{\tau}$;
2. **Processes** it through a U-Net-MLP with contracting-expanding topology and skip connections;
3. **Outputs** the vector of pre-activations $a^{\tau}\in\mathbb{R}^{D}$.

This $a^{\tau}$:

* encodes *synaptic interactions* between neurons at step $\tau$ (accounting for both global and local dependencies aggregated in the "bottleneck");
* becomes the **anchor point of the internal time scale**: the sequence $\{a^{1},a^{2},\dots\}$ forms a discrete "internal chronotick" of the model, independent of the number of input tokens or their real-world timing.

Thus, the **vector $a^{\tau}$ itself — is a "snapshot" of the synaptic state at the current tick**; the *temporal axis* is formed precisely by the recurrent application of the Synapse model ($\tau \to \tau+1$) in conjunction with the History buffers and subsequent processing in Neuron-Level Models.

<!-- Checkpoint: Synapse model -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — Synapse model:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    A U-Net-like MLP takes the combined past post-activations <em>z<sup>τ</sup></em> and attention output <em>o<sup>τ</sup></em>, multiscale compresses and expands them via skip connections, and outputs pre-activations <em>a<sup>τ</sup></em>; it is here that "synaptic" interactions and the model's internal time scale are formed.
  </p>
</div>

<details> 
    <summary><em><strong>2. History (pre-activations)</strong></em></summary>

### **2. History (pre-activations)**

For each neuron $d = 1, \dots, D$, a **history of the last $M$** pre-activations is maintained:

$$
A^\tau_d = [\, a^{\tau - M + 1}_d,\; \dots,\; a^{\tau}_d \,] \in \mathbb{R}^M
$$

and the full history for all neurons:

$$
A^\tau \in \mathbb{R}^{D \times M}
$$

This buffer enables each neuron to analyze the temporal dynamics of its activation signal (internal steps).

**What is "History (pre-activations)"?**  
1. **Definition**  
   - For each neuron $d$ (where $d=1,\dots,D$), a buffer stores the $M$ most recent values of its pre-activations $a_d^\tau$ across the model's internal steps.  
   - This buffer is denoted as the vector  
     $$
       A_d^\tau = [\,a_d^{\tau-M+1},\,a_d^{\tau-M+2},\,\dots,\,a_d^\tau] \in \mathbb{R}^M.
     $$  
   - For all $D$ neurons together, we obtain the matrix  
     $$
       A^\tau = 
       \begin{pmatrix}
         A_1^\tau\\
         A_2^\tau\\
         \vdots\\
         A_D^\tau
       \end{pmatrix}
       \in \mathbb{R}^{D\times M}.
     $$

2. **Numerical Example**  
   Let $D=2$ (two neurons), $M=3$ (three latest steps). At internal steps $\tau=5$, the pre-activations might be:

   * Neuron 1: 
   - $a\_1^3=0.2$
   - $a\_1^4=-0.1$
   - $a\_1^5=0.5$

   * Neuron 2: 
   - $a\_2^3=1.0$
   - $a\_2^4=0.8$
   - $a\_2^5=0.9$

   Then

   $$
     A_1^5 = [\,0.2,\,-0.1,\,0.5\,],\quad
     A_2^5 = [\,1.0,\,0.8,\,0.9\,],
   $$

   and

   $$
     A^5 = 
     \begin{pmatrix}
       0.2 & -0.1 & 0.5\\
       1.0 &  0.8 & 0.9
     \end{pmatrix}.
   $$

3. **Why is this needed?**

   * The Neuron-Level Model $g_{\theta_d}$ receives as input not only the current $a_d^\tau$, but the entire vector $A_d^\tau$.
   * This enables accounting for **temporal dynamics**: patterns of activation change over the last $M$ steps influence the next activation $z_d^{,\tau+1}$.
   * Such history is essential for forming an **internal temporal context** and neuron synchronization in CTM.

Thus, the "History (pre-activations)" block describes the mechanism for storing and representing the temporal sequence of each neuron's pre-activations, which is key to implementing the stepwise "reasoning" process in the Continuous Thought Machine architecture.

```python
"""
Implementation of a neural network architecture for modeling synaptic 
signal transmission. The code includes two main components:
1. SynapseModel - a neural network class with detailed intermediate output
2. PreActivationHistory - a class for tracking pre-activation history

Functional purpose: Modeling and debugging a synaptic neural network with 
visualization of intermediate neuron states for signal processing analysis.
"""

# Standard libraries
from typing import List, Tuple

# Neural network libraries
import torch
import torch.nn as nn


# Neural network model for debugging synaptic connections
class SynapseModel(nn.Module):
    """
    Description:
    ---------------
        A neural network model with an encoder-decoder architecture and detailed 
        intermediate output for debugging. The model accepts input data, 
        processes it through encoder and decoder layers using skip connections.

    Args:
    ---------------
        d_model: Dimensionality of input vectors
        d_attn: Dimensionality of attention vectors
        hidden_dims: List of hidden layer dimensions in the encoder

    Returns:
    ---------------
        A neural network model object

    Raises:
    ---------------
        ValueError: If hidden_dims dimensions are incompatible with model architecture

    Examples:
    ---------------
        >>> model = SynapseModel(d_model=2, d_attn=2, hidden_dims=[3, 2])
        >>> z = torch.ones(1, 2)
        >>> o = torch.ones(1, 2)
        >>> output = model(z, o)
    """

    def __init__(
        self,
        d_model: int,
        d_attn: int,
        hidden_dims: List[int] = [3, 2],
    ) -> None:
        super().__init__()
        in_dim = d_model + d_attn
        self.d_model = d_model

        # Create encoder layers
        enc_layers = []
        prev = in_dim
        for i, h in enumerate(hidden_dims, 1):
            enc_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True)
            ]
            prev = h
        self.encoder = nn.ModuleList(enc_layers)

        # Create decoder layers
        rev = hidden_dims[::-1]  # Reverse order for decoder
        dec_layers = []
        prev = rev[0] * 2        # Size after first concatenation
        for i, h in enumerate(rev[1:], 1):
            dec_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True)
            ]
            # Concatenate with corresponding skip connection
            prev = h + rev[i]
        dec_layers.append(nn.Linear(prev, d_model))
        self.decoder = nn.ModuleList(dec_layers)

    # Forward pass through the neural network
    def forward(
        self,
        z: torch.Tensor,
        o: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Performs a forward pass through the neural network, combining input
            tensors z and o, and outputs detailed debugging information for 
            intermediate results at each layer.

        Args:
        ---------------
            z: Input tensor of model data
            o: Input tensor of attention vectors

        Returns:
        ---------------
            Tensor of pre-activations at the network output

        Examples:
        ---------------
            >>> model = SynapseModel(2, 2)
            >>> z = torch.ones(1, 2)
            >>> o = torch.ones(1, 2)
            >>> a = model(z, o)
        """
        # Concatenate input tensors along the last dimension
        x = torch.cat([z, o], dim=-1)
        print(f"\n=== Step τ ===")
        print(f">>> Input x (z||o): {x.tolist()}")

        # Pass through encoder layers
        skips = []  # List to store skip connections
        cur = x     # Current tensor
        layer_idx = 0
        for layer in self.encoder:
            cur = layer(cur)
            if isinstance(layer, nn.Linear):
                print(
                    f"  Enc Linear {layer_idx:02d}: shape {tuple(cur.shape)} "
                    f"→ {cur.tolist()}"
                )
            elif isinstance(layer, nn.LayerNorm):
                print(f"  Enc LayerNorm {layer_idx:02d}: → {cur.tolist()}")
            elif isinstance(layer, nn.ReLU):
                print(f"  Enc ReLU      {layer_idx:02d}: → {cur.tolist()}")
                skips.append(cur)  # Save output after ReLU for skip connections
            layer_idx += 1

        # Pass through decoder layers
        # Start by concatenating with the last skip connection
        cur = torch.cat([cur, skips[-1]], dim=-1)
        print(f"  Dec Input (with skip[-1]): → {cur.tolist()}")

        skip_idx = -2  # Index of previous skip connection
        layer_idx = 0
        for layer in self.decoder:
            cur = layer(cur)
            if isinstance(layer, nn.Linear):
                print(f"  Dec Linear {layer_idx:02d}: → {cur.tolist()}")
            elif isinstance(layer, nn.LayerNorm):
                print(f"  Dec LayerNorm {layer_idx:02d}: → {cur.tolist()}")
            elif isinstance(layer, nn.ReLU):
                print(f"  Dec ReLU      {layer_idx:02d}: → {cur.tolist()}")
                # Add skip connection if available
                if skip_idx >= -len(skips):
                    cur = torch.cat([cur, skips[skip_idx]], dim=-1)
                    print(f"    + skip[{skip_idx}]: → {cur.tolist()}")
                    skip_idx -= 1
            layer_idx += 1

        a = cur  # Output pre-activations
        print(f"<<< Output pre-activations a: {a.tolist()}")
        return a


# Class for tracking pre-activation history
class PreActivationHistory(nn.Module):
    """
    Description:
    ---------------
        Module for tracking the history of neural network pre-activations.
        Stores the last M pre-activation values for each neuron and provides
        methods to reset and update the history buffer.

    Args:
    ---------------
        d_model: Dimensionality of pre-activation vector
        M: Number of time steps to store in history

    Returns:
    ---------------
        Object for tracking pre-activation history

    Examples:
    ---------------
        >>> history = PreActivationHistory(d_model=2, M=3)
        >>> history.reset(batch_size=1)
        >>> a = torch.tensor([[0.1, 0.2]])
        >>> history.update(a)
    """

    def __init__(self, d_model: int, M: int) -> None:
        super().__init__()
        self.M = M
        # Initialize history buffer
        buf = torch.zeros(1, d_model, M)
        self.register_buffer('history', buf)

    # Reset the history buffer
    def reset(self, B: int) -> None:
        """
        Description:
        ---------------
            Resets the pre-activation history buffer, initializing it to zeros
            for the specified batch size.

        Args:
        ---------------
            B: Batch size

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> history = PreActivationHistory(2, 3)
            >>> history.reset(1)
        """
        self.history = torch.zeros(
            B, *self.history.shape[1:], device=self.history.device
        )
        print(f"\n*** History reset → shape {tuple(self.history.shape)}")

    # Update the history buffer
    def update(self, a: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Updates the pre-activation history buffer by adding new values 
            and discarding the oldest ones. Outputs the current buffer state.

        Args:
        ---------------
            a: Tensor of new pre-activations to add to history

        Returns:
        ---------------
            Updated pre-activation history buffer

        Examples:
        ---------------
            >>> history = PreActivationHistory(2, 3)
            >>> history.reset(1)
            >>> a = torch.tensor([[0.1, 0.2]])
            >>> updated_history = history.update(a)
        """
        # If batch size changed, reset history
        if self.history.size(0) != a.size(0):
            self.reset(a.size(0))
        
        # Update buffer by appending new values and removing oldest
        self.history = torch.cat(
            [self.history[:, :, 1:], a.unsqueeze(-1)], dim=2
        )
        
        # Output current buffer state as a matrix
        mat = self.history[0].tolist()
        print(f"*** History buffer (last {self.M} steps):")
        for d, row in enumerate(mat, 1):
            print(f"    Neuron {d}: {row}")
        
        return self.history


# Initialization and execution
def main() -> None:
    """
    Description:
    ---------------
        Main function to initialize and run the SynapseModel and track 
        pre-activation history.

    Args:
    ---------------
        None

    Returns:
    ---------------
        None

    Examples:
    ---------------
        >>> main()
    """
    # Set initial random seed
    torch.manual_seed(0)

    # Initialize model parameters
    d_model, d_attn, M = 2, 2, 3
    model = SynapseModel(d_model, d_attn, hidden_dims=[3, 2])
    history = PreActivationHistory(d_model, M)

    # Batch size
    B = 1
    history.reset(B)
    
    # Run model over several time steps
    for tau in range(1, 4):
        print(f"\n>>> Internal Step τ = {tau}")
        # Create input data for current step
        z = torch.full((B, d_model), tau, dtype=torch.float32)
        o = torch.full((B, d_attn), tau * 0.1, dtype=torch.float32)
        # Forward pass through model
        a = model(z, o)
        # Update pre-activation history
        history.update(a)

main()
```

</details>

At each step, a "memory cube" of the last $M$ pre-activation vectors is formed. Formally,

$$
A^{\tau}\;=\;\bigl[a^{\tau-M+1},\,a^{\tau-M+2},\,\dots,\,a^{\tau}\bigr]\;\in\;\mathbb R^{D\times M},
\qquad a^{t}\in\mathbb R^{D}.
$$

* **Columns** are the individual "slices" of synaptic state: each column $a^{t}$ aggregates all $D$ neurons at internal tick $t$.
* **Rows** are individual temporal trajectories of a single neuron $d$ over $M$ steps:

  $$
  A^{\tau}_{d,\: :}\;=\;\bigl[a^{\tau-M+1}_{d},\,\dots,\,a^{\tau}_{d}\bigr].
  $$

It is precisely this set of $M$ latest columns (i.e., $M$ "slices" $a^{t}$) that is fed into the **Neuron-Level Models**, allowing each neuron to analyze its own dynamics and participate in computing subsequent synchronization.

<!-- Checkpoint: History (pre-activations) -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — History (pre-activations):</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    Each neuron maintains a sliding window of its last <em>M</em> pre-activations <a href="https://example.com  ">(A<sup>τ</sup>)</a>, providing the model with information about its own temporal dynamics and forming the basis for synchronization and subsequent processing in Neuron-Level Models.
  </p>
</div>

<details> 
    <summary><em><strong>3. Neuron-Level Models</strong></em></summary>

### **3. Neuron-Level Models**

Each neuron $d$ has its own function $g_{\theta_d}$ that transforms its history $A^\tau_d$ into a post-activation:

$$
z^{\tau+1}_d = g_{\theta_d}(A^\tau_d)
$$

Here, $g_{\theta_d}$ is an individual neuron MLP processing an $M$-length temporal window. For example, it may learn a pattern: "if the neuron was active three times consecutively → activate again." This approximates the **spiking activity** of biological neurons.

#### **Example**

To illustrate the operation of $g_{\theta_d}$, consider a simplified case with one "Neuron-Level Model" — a single-layer MLP processing a history of length $M=3$.

For a specific neuron $d$, the model

$$
z_d^{\tau+1} = g_{\theta_d}\bigl(A_d^\tau\bigr),
$$

where

$$
A_d^\tau = \bigl[a_d^{\tau-2},\,a_d^{\tau-1},\,a_d^{\tau}\bigr]^\top\in\mathbb R^3.
$$

#### MLP Parameters

Let $g_{\theta_d}$ be a two-layer MLP with one hidden unit and ReLU activation:

$$
\begin{aligned}
h &= \mathrm{ReLU}\bigl(W^{(1)}\,A_d^\tau + b^{(1)}\bigr),\quad
W^{(1)}\in\mathbb R^{1\times3},\;b^{(1)}\in\mathbb R,\\
z_d^{\tau+1} &= W^{(2)}\,h + b^{(2)},\quad
W^{(2)}\in\mathbb R^{1\times1},\;b^{(2)}\in\mathbb R.
\end{aligned}
$$

Use concrete numerical values:

$$
W^{(1)} = \begin{pmatrix}0.4 & -0.3 & 0.5\end{pmatrix},\quad
b^{(1)} = 0.1;\qquad
W^{(2)}=1.2,\quad
b^{(2)}=-0.05.
$$

#### Pre-activation History

Assume that on the three most recent internal steps $\tau$, the pre-activations of neuron $d$ were:

$$
a_d^{\tau-2} = 0.2,\quad
a_d^{\tau-1} = -0.1,\quad
a_d^{\tau}   = 0.5.
$$

Then

$$
A_d^\tau = \begin{pmatrix}0.2\\-0.1\\0.5\end{pmatrix}.
$$

#### Step-by-step Calculation

1. **Linear transformation + bias (first layer):**

   $$
   u = W^{(1)}\,A_d^\tau + b^{(1)}
     = [\,0.4,\,-0.3,\;0.5\,]\;\begin{pmatrix}0.2\\-0.1\\0.5\end{pmatrix}
     + 0.1
     = (0.08 + 0.03 + 0.25) + 0.1
     = 0.46.
   $$

2. **ReLU activation:**

   $$
   h = \mathrm{ReLU}(u) = \max(0,\,0.46) = 0.46.
   $$

3. **Second layer output:**

   $$
   z_d^{\tau+1} = W^{(2)}\,h + b^{(2)}
                = 1.2 \times 0.46 - 0.05
                = 0.552 - 0.05
                = 0.502.
   $$

#### Explanation

* **First layer** "convolves" the three-step history $A_d^\tau$ into a single number $u$, weighting past activations by their significance (weights $W^{(1)}$) and adding bias $b^{(1)}$.
* **ReLU** discards negative "noise" and retains only useful patterns (here $u>0$).
* **Second layer** scales the extracted feature $h$ and adds final bias $b^{(2)}$, outputting the new post-activation $z_d^{\tau+1}$.

Thus, even in this simple example, the neuron-level MLP learns to respond to sequences of past activations and form an output based on learned temporal patterns.

</details>

On each internal tick $\tau$, the **Neuron-Level Models (NLM)** block:

1. **Receives** for each neuron $d$ its individual history row
   $A^{\tau}_{d}=[\,a^{\tau-M+1}_{d},\dots ,a^{\tau}_{d}] \in\mathbb R^{M}$.
2. **Processes** this $M$-dimensional vector through a tiny MLP
   $g_{\theta_d}\!=\!\text{(Linear → Act → Linear)}$ — parameters may be unique per neuron.
3. **Returns** the new post-activation

   $$
     z^{\tau+1}_{d}=g_{\theta_d}\bigl(A^{\tau}_{d}\bigr)\in\mathbb R,
     \qquad
     z^{\tau+1}=[z^{\tau+1}_{1},\dots ,z^{\tau+1}_{D}]\in\mathbb R^{D}.
   $$

These $z^{\tau+1}_{d}$:

* implement **temporal filtering**: each neuron decides whether to "spike" now, looking at its own recent pre-activation pattern (analogous to STDP);
* transform the "memory cube" $A^{\tau}$ into the next **dynamics slice** $z^{\tau+1}$, which then participates in
  – accumulating the global history $Z^{\tau+1}=[z^{1},\dots ,z^{\tau+1}]$ and
  – computing the synchronization matrix $S^{\tau+1}=Z^{\tau+1}(Z^{\tau+1})^{\!\top}$.

Thus, the **NLM block acts as an individual "temporal detector"**: it encodes each neuron’s short-term memory into a new state, enabling the network to step-by-step build collective synchronization and progress along the internal time scale.

<!-- Checkpoint: Neuron-Level Models -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — Neuron-Level Models:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    An individual MLP <em>g<sub>θd</sub></em> reads each neuron's history <A<sub>d</sub><sup>τ</sup></A> and generates its new post-activation <em>z<sub>d</sub><sup>τ+1</sup></em>, learning to detect temporal patterns (analogous to STDP) and transforming "raw" signals into dynamics suitable for synchronization.
  </p>
</div>

### **4. Post-activations**

On each internal tick $\tau{+}1$, the **Post-activations** block:

1. **Gathers** outputs from Neuron-Level Models into a single vector

   $$
     z^{\tau+1}=\bigl[z^{\tau+1}_{1},\dots ,z^{\tau+1}_{D}\bigr]\in\mathbb R^{D}.
   $$
2. **Records** this vector as a "snapshot" of the network's *neuronal state*: it reflects which neurons activated after accounting for their short-term memory.
3. **Passes** $z^{\tau+1}$ further down the pipeline:

   * to the long-term memory matrix $Z$ (see figure);
   * to the computation of the synchronization matrix $S^{\tau+1}=Z^{\tau+1}(Z^{\tau+1})^{\!\top}$;
   * back to the **Synapse model** along with the new attention vector $o^{\tau+1}$, closing the recursive cycle.

Thus, **$z^{\tau+1}$ is a "slice" of collective activity** across all neurons, serving as a building block for further synchronization and model reasoning.

The outputs of all neurons at step $\tau+1$ form:

$$
z^{\tau+1} = [z^{\tau+1}_1, z^{\tau+1}_2, \dots, z^{\tau+1}_D] \in \mathbb{R}^D
$$

This vector — the **internal neuronal state of the model** — evolves over time.

<!-- Checkpoint: Post-activations -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — Post-activations:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    The vector <em>z<sup>τ+1</sup></em> aggregates the new outputs of all neurons after their "temporal filtering"; it is an instantaneous snapshot of the network's internal state, forwarded to memory <em>Z</em> and used in synchronization computation.
  </p>
</div>

### **5. History (post-activations)**

At the **History (post-activations)** step:

1. **Extends** the long-term memory matrix by appending a new column:

   $$
     Z^{\tau+1}= \bigl[\,z^{1},\,z^{2},\,\dots ,\,z^{\tau+1}\bigr]\in\mathbb R^{D\times(\tau+1)}.
   $$
2. **Stores** the full trajectory of network behavior: each row is the history of a specific neuron, each column is a "frame" of the entire network.
3. **Used** by two primary blocks:

   * for computing the updated synchronization matrix $S^{\tau+1}$;
   * for evaluating stopping criteria (sufficient output confidence / tick limit reached).

It is precisely the matrix $Z$ that transforms the sequence of discrete "slices" $z^{t}$ into a **continuous activity ribbon**, upon which the model learns to identify long-term co-patterns.

<!-- Checkpoint: History (post-activations) -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — History (post-activations):</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    The matrix <em>Z<sup>τ+1</sup></em> accumulates all <em>z</em> vectors since the start of reasoning, forming the network's "activity ribbon"; it feeds the computation of the synchronization matrix <em>S</em> and serves as the basis for stopping or continuing internal ticks.
  </p>
</div>

### **6. Synchronization matrix $S^{\tau+1}$**

On each internal tick $\tau{+}1$, the **Synchronization matrix** block:

1. **Takes** the accumulated activity ribbon $Z^{\tau+1}\in\mathbb R^{D\times(\tau+1)}$.
2. **Multiplies** it by its own transpose:

   $$
     S^{\tau+1}=Z^{\tau+1}(Z^{\tau+1})^{\!\top}\in\mathbb R^{D\times D},
   $$

   yielding a symmetric matrix of pairwise scalar products.
3. **Interprets** element $S^{\tau+1}_{ij}$ as a measure of similarity between the temporal trajectories of neurons $i$ and $j$:

   * magnitude $\kern0.1em\uparrow$ — neurons activated synchronously;
   * magnitude $\kern0.1em\downarrow$ — their patterns are desynchronized.
4. **Feeds** $S^{\tau+1}$ into the pair selection mechanism, and (during training) applies exponential decay or normalization so that recent "frames" weigh more than distant ones.

Thus, **$S^{\tau+1}$ condenses collective dynamics** into a compact representation of connections, upon which reasoning and attention signals are subsequently built.

<!-- Checkpoint: Synchronization matrix -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — Synchronization matrix:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    The matrix <em>S<sup>τ+1</sup></em> = <em>Z</em><sup>τ+1</sup>(<em>Z</em><sup>τ+1</sup>)<sup>T</sup> records how similar the temporal trajectories of neurons are. A high element indicates strong synchronization; a low one indicates desynchronization. This is the fundamental representation of the network's "collective mind."
  </p>
</div>

### **7. Selected neuron pairs**

At the **Selected neuron pairs** step:

1. **Selects** two fixed subsets of indices

   $$
     \mathcal I_{\text{out}},\,\mathcal I_{\text{action}}\subset\bigl\{(i,j)\,|\,0\!\le i\!<\!j\!<\!D\bigr\},
   $$

   defined once at initialization (random or top-k mode).
2. **Extracts** corresponding elements from the synchronization matrix, forming two vectors:

   $$
     S^{\tau+1}_{\text{out}}\in\mathbb R^{D_{\text{out}}},\qquad
     S^{\tau+1}_{\text{action}}\in\mathbb R^{D_{\text{action}}}.
   $$
3. **Passes**

   * $S^{\tau+1}_{\text{out}}$ → linear projector $W_{\text{out}}$ for prediction (class logits / next token);
   * $S^{\tau+1}_{\text{action}}$ → projector $W_{\text{in}}$ for generating attention query $q^{\tau+1}$.
4. **Ensures** role separation: one subset learns "what to say," the other "where to look" in the data, avoiding model overload from the full $D^2$ number of connections.

In other words, **the high-dimensional matrix $S$ is compressed into two controlled latent vectors**, which feed prediction and attention, making computations scalable.

<!-- Checkpoint: Selected neuron pairs -->
<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Checkpoint — Selected neuron pairs:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">
    From <em>S</em>, two constant subsets of neuron pairs are selected: one (<em>I</em><sub>out</sub>) feeds the output, the other (<em>I</em><sub>action</sub>) feeds the attention query. The resulting vectors <em>S<sub>out</sub></em> and <em>S<sub>action</sub></em> compactly encode key connections, enabling linear (not quadratic) scaling with neuron count.
  </p>
</div>

### **8. Latent representation**

Based on the selected pairs:

* $S^{\tau+1}_{\text{out}}$ serves as a **feature vector** for prediction;
* $S^{\tau+1}_{\text{action}}$ — as an **attention query vector**.

Both vectors are low-dimensional compressed representations of the network's **internal dynamics**.

### **9. OUT/ATTN projections**

Two linear layers:

* For predicting the next token:

  $$
  y^{\tau+1} = W_{\text{out}} \cdot S^{\tau+1}_{\text{out}} \in \mathbb{R}^{V}
  $$

  where $V$ is the token vocabulary size;

* For generating the attention query:

  $$
  q^{\tau+1} = W_{\text{in}} \cdot S^{\tau+1}_{\text{action}} \in \mathbb{R}^{d_{\text{attn}}}
  $$

### **10. Data modulation (Attention output)**

**Attention** is computed over the input token embeddings $F(x) \in \mathbb{R}^{t \times d_{\text{attn}}}$ using $q^{\tau+1}$ as the query:

$$
o^{\tau+1} = \text{Attention}(q^{\tau+1}, K = F(x), V = F(x)) \in \mathbb{R}^{d_{\text{attn}}}
$$

Here, dot-product attention is used:

$$
\text{Attention}(q, K, V) = \text{softmax}\left( \frac{qK^\top}{\sqrt{d}} \right) V
$$

The vector $o^{\tau+1}$ is combined with $z^{\tau+1}$ and fed to the next internal tick $\tau+2$, forming the input to the synapse model:

$$
[z^{\tau+1}, o^{\tau+1}] \longrightarrow f_{\theta_{\text{syn}}}
$$

### **Overall Process at Step $\tau$:**

1. $z^\tau, o^\tau \rightarrow a^\tau$ via synapse
2. Update history $A^\tau$
3. $A^\tau_d \rightarrow z_d^{\tau+1}$ via NLM
4. Update $Z^{\tau+1}$
5. $Z^{\tau+1} \rightarrow S^{\tau+1}$
6. $S^{\tau+1} \rightarrow S_{\text{out}}, S_{\text{action}}$
7. $S_{\text{out}} \rightarrow y^{\tau+1}$, $S_{\text{action}} \rightarrow q^{\tau+1}$
8. $q^{\tau+1} \rightarrow o^{\tau+1}$
9. Feed $[z^{\tau+1}, o^{\tau+1}]$ to next tick.

#### **Example:**

Suppose CTM works on the text:

```
"Albert Einstein was a ..."
```

#### Steps:

* $x = (\text{“Albert”}, \text{“Einstein”}, \text{“was”}, \text{“a”})$
* Task: predict next token $x_5$ (expected: “physicist”)
* On each internal step $\tau$, CTM updates its activations and accumulates neuron synchronization.
* For example, at $\tau=1$: $z^1$ is random, $S^1$ is nearly zero.
* By $\tau=5$: activity stabilizes; pairs of neurons sensitive to patterns like "person → profession" synchronize.
* $y^5 = \text{softmax}(W_{\text{out}} \cdot S^5_{\text{out}})$ — already assigns high probability to “physicist”.
* If $C^5 = 0.95$ and confidence threshold $\tau = 0.9$, the model terminates the cycle.

---

### **4. Internal Recurrent Time and Synapses**

CTM introduces a separate internal time, discretized into steps called **internal ticks** (*internal ticks*). An internal tick is a step of "thinking" by the model, during which it can update its neuronal states even if external data is static. Thus, **CTM can iteratively refine its representation of static input data over time**, approximating the process of thought, where the brain contemplates a task. If input data is sequential (e.g., text), internal ticks may differ from real data time steps, allowing the model to reason longer than the input sequence length.

On each internal step $t$, all neurons update their states through a common **synaptic module**. The synaptic model is a recurrent multi-layer perceptron (MLP) structured as a U-Net (with skip connections between layers). It computes the **pre-activations** ($a^t$) of all $D$ neurons for the next step, based on current neuron activations and external data information. Formally, the synaptic model $f_{\theta_{\text{syn}}}$ takes as input the concatenation of the current post-activation vector $z^t \in \mathbb{R}^D$ and some external signal $o^t$ (attention result on data, detailed below), and outputs a new pre-activation vector:

$$
a^t \;=\; f_{\theta_{\text{syn}}}\!\Big(\big[z^t,\, o^t\big]\Big)\;\in\;\mathbb{R}^D\,,
$$

where $[z^t, o^t]$ denotes concatenation of the two vectors. The vector $a^t = (a^t_1, \dots, a^t_D)$ contains one component for each of the $D$ neurons. This step is analogous to signal propagation through synapses from all neurons to each other: $f_{\theta_{\text{syn}}}$ models inter-neuronal connections simultaneously for the entire neuron layer (its parameters $\theta_{\text{syn}}$ are shared across all layer neurons). Thanks to the U-Net architecture, the synaptic MLP combines deep signal processing with preservation of low-level information via skips, ensuring gradient stability and capturing interactions across multiple scales.

The obtained pre-activations $a^t$ are immediately used to form the **temporal history of each neuron's inputs**. For each neuron $d$, a window of its last $M$ pre-activation values is maintained:

$$
A^t_d \;=\; [\,a^{t-M+1}_d,\; a^{t-M+2}_d,\; \dots,\; a^t_d\,] \;\in\; \mathbb{R}^{M}\,
$$

and the collective set of such windows for all neurons forms the matrix $A^t \in \mathbb{R}^{D\times M}$. It is assumed that for initial steps, when histories are insufficient, zero-padding or incomplete history initiation is used (in implementation, this may be organized as a circular buffer of length $M$ for each component). This history $A^t$ serves as input to the individual neuron models described below. Parameter $M$ determines how far into the past each neuron "remembers" — the length of the temporal context influencing its current activation.

### **5. Neuron-Level Models and Neural Timing**

The key distinction of CTM is that each neuron has its own **neuron model (NLM)**, which determines its output based on the history of its input signals. In standard networks, this role is played by a simple activation function (ReLU, sigmoid, etc.), depending only on the current pre-activation. In CTM, instead of a fixed function, a *personalized MLP is used for each neuron*.

For neuron $d$, denote its model parameters as $\theta_d$. At each step $t$, the model $g_{\theta_d}$ of neuron $d$ takes its input history vector $A^t_d \in \mathbb{R}^M$ and computes the **post-activation** (i.e., final neuron activation) for the next step $t+1$:

$$
z_d^{t+1} \;=\; g_{\theta_d}\!\big(A^t_d\big)\,,
$$

where $z_d^{t+1}$ is a scalar (activation of neuron $d$ at output). The vector of all post-activations at step $t+1$ is $z^{t+1} = (z_1^{t+1}, \dots, z_D^{t+1}) \in \mathbb{R}^D$ (corresponds to block **4** in Figure 1).

Each such neuron model is a small MLP with a unique set of weights $\theta_d$ (not shared with other neurons). **Neural timing and dynamics** are achieved because $g_{\theta_d}$ can learn different responses to patterns in its input sequence $A^t_d$. For example, a neuron may learn to activate only if its inputs show a specific temporal pattern (spike, decay, oscillation, etc.), which a simple ReLU, ignoring the past, cannot do.

Thus, each neuron in CTM is a small *autonomous temporal processor*, analogous to a simplified biological neuron with temporal summation and delay. At the same time, the level of abstraction remains sufficiently high for efficient gradient-based training, as $M$ is typically small, and NLM is a small parametric module. In implementation, NLMs may be constructed as, for example, one-dimensional convolutional filters or small fully connected networks operating independently on each neuron.

After computing $z^{t+1}$, all post-activations are concatenated with the (external) attention output $o^t$ and fed back into the synaptic model on the next tick, closing the recurrent loop. This recurrent cycle enables **iterative evolution of the network's state**, i.e., performing multi-step reasoning over the input.

### **6. Representation of Neuron Synchronization**

After each internal tick $t$, CTM updates not only neuron states but also its *representation of the external world* based on **neuronal activation dynamics over time**. The key idea is to use *synchronization* between neurons as a feature for decision-making.

**Synchronization** here means the degree of simultaneity or alignment of oscillations in activations of different neurons. To compute it, CTM stores the history of all post-activations *throughout the entire internal reasoning process*:

$$
Z^t \;=\; [\,z^1,\; z^2,\; \dots,\; z^t\,] \;\in\; \mathbb{R}^{D\times t}\,,
$$

where the columns of the matrix are the activation vectors at each step from $1$ to $t$. This history $Z^t$ grows continuously as the model reasons (the size of the second dimension equals current $t$).

The **synchronization matrix** $S^t \in \mathbb{R}^{D\times D}$ is defined as the matrix of scalar products of all pairs of neuronal activation time series:

$$
S^t \;=\; Z^t \cdot (Z^t)^\top \,.
$$

Element $S^t_{ij}$ equals $\langle Z^t_{i,\cdot},\, Z^t_{j,\cdot}\rangle$, i.e., the scalar product between the activation time series of neuron $i$ and neuron $j$ (from step 1 to $t$). This value will be high if two neurons exhibited similar dynamics (synchronized activation and decay), and low if their patterns are poorly correlated or temporally shifted.

Thus, $S^t$ encodes **synchronization between each neuron pair** as a feature.

However, the full matrix $S^t$ has size $D \times D$ (symmetric, excluding diagonal — $\frac{D(D-1)}{2}$ unique pairs), which can be very large when the number of neurons is large. For instance, if $D = 512$, the number of pairs is about 130 thousand.

To make the representation compact, the authors propose taking a random subset of $S^t$ elements of fixed size. They select two sets of index pairs $(i,j)$ in advance, of sizes $D_{\text{out}}$ and $D_{\text{action}}$ each. The first set corresponds to features for **model output**, the second for **interaction with input**.

That is, from the full matrix $S^t$, $D_{\text{out}} + D_{\text{action}}$ values are selected, forming two vectors:

- $S_{\text{out}}^t \in \mathbb{R}^{D_{\text{out}}}$ — latent synchronization vector for output,
- $S_{\text{action}}^t \in \mathbb{R}^{D_{\text{action}}}$ — for generating action (e.g., attention query).

These vectors represent a **compressed representation of neuronal dynamics**, used later instead of the full $S^t$. The selection of pairs carries no semantic meaning — they are randomly assigned and fixed at model initialization, and learning discovers how to use these elements.

Note: the dimensionality of the synchronization space grows quadratically with the number of neurons ($\sim D^2/2$), providing potential for richer representation without increasing model parameters (since weight matrices $W_{\text{out}}, W_{\text{in}}$ project into a fixed-dimensional space). The authors note this opens a path to more expressive representations as model width increases.

The two latent synchronization vectors are transformed into final model outputs via trainable **linear layers** (weight matrices): $W_{\text{out}}$ and $W_{\text{in}}$. Projection of the sync-vector to output is performed as:

$$
y^t \;=\; W_{\text{out}} \, S_{\text{out}}^t\,,
$$

where $y^t$ is the model output vector at step $t$ (e.g., class logits for classification or action distribution parameters for reinforcement learning).

Similarly, the other projection yields a vector for *interaction with data*. It can be interpreted as the model's internal "intention" toward the input. In CTM, it is used as an **attention query** (*attention query*) to external data:

$$
q^t \;=\; W_{\text{in}} \, S_{\text{action}}^t\,.
$$

This query $q^t$ is used to extract relevant information from input data via a **cross-attention** mechanism. That is, CTM decides what to attend to, *based on synchronized neuronal activity*.

In implementation, a standard attention module is used: $q^t$ is the query, while *keys* $K$ and *values* $V$ are obtained by passing the input data through a feature extractor network (e.g., ResNet for images). This forms the **attention output** $o^t$:

$$
o^t \;=\; \text{Attention}\!\big(Q = q^t,\; K=F(x),\; V=F(x)\big)\,,
$$

where $F(x)$ are features of input data $x$ (e.g., a set of features for each image patch).

The attention output $o^t$ is a fixed-length vector (usually equal to $z^t$'s dimension) containing "extracted" information from the input relevant to the current query $q^t$. In the simplest case, $o^t$ can be viewed as a weighted sum of input features, where weights are attention coefficients depending on $q^t$.

This vector $o^t$ is then, as described above, **fed back into the synaptic module on the next step** (concatenated with $z^t$).

Thus, CTM on each tick updates its internal state ($z$) *and* adapts its perception of input data ($o$) to this state, analogous to how the brain can actively choose what to look at or think about next.

Collectively, the described cycle defines the **dynamics of the CTM model**. In practice, CTM may not always use all $T$ steps for each task — see the section on the adaptive stopping mechanism for details.

Note: **CTM can be viewed as a special type of recurrent neural network**. Vector $z^t$ plays the role of a hidden state evolving over time, and through $y^t$, the model can emit intermediate results. However, unlike standard RNNs, here the hidden state is updated in a complex way: involving individual neuronal dynamics (NLM) and a cross-attention mechanism to external data.

**CTM's "thinking process"** is an alternation of internal modeling of neuronal interactions (via $f_{\text{syn}}$ and synchronization) and active data reading (via $q^t$ and $o^t$). This architecture leads to a rich space of internal states and, as experiments show, to interpretable problem-solving strategies. For example, when solving mazes, CTM internally "draws" the path during reasoning.

### **7. Formalization of the Loss Function**

Training CTM is complicated by the fact that the model produces predictions at each internal step. This raises the question: how to compute error and update weights while accounting for the full temporal dynamics?

The authors introduce a specialized optimization scheme that encourages the model to learn to solve tasks as quickly as possible (in fewer ticks), without sacrificing accuracy on complex cases.

Let $y^t$ be the model’s prediction at internal step $t$ (e.g., a probability distribution over classes), and $y_{\text{true}}$ be the ground truth. For each tick, a standard loss function $L^t$ can be computed, such as cross-entropy for classification:

$$
L^t = \text{CrossEntropy}(y^t,\; y_{\text{true}})\,.
$$

Additionally, a **confidence measure** at step $t$ is defined. The authors use a simple confidence metric $C^t = 1 - H^t$, where $H^t$ is the normalized entropy of the output distribution $y^t$. The value $C^t$ is close to 1 when the model is confident (the distribution $y^t$ is concentrated on one option), and close to 0 when the distribution is flat (high uncertainty).

Thus, for each $t$, we have a pair $(L^t, C^t)$.

To reduce these values to a single scalar loss for training, two special moments in the internal dynamics are selected:

- $t_1$ — the step of minimum error:  
  $$
  t_1 = \arg\min_{t} L^t
  $$  
  This is the internal tick at which the model most closely approached the correct answer (perhaps for the first time guessing correctly).

- $t_2$ — the step of maximum confidence:  
  $$
  t_2 = \arg\max_{t} C^t
  $$  
  This tick captures the moment when the model is most confident in its prediction.

The final loss function sums the errors at these two steps, with the minimum-error step weighted double:

$$
L_{\text{final}} = 2 \cdot L^{t_1} + L^{t_2}\,.
$$

This approach simultaneously directs gradients to:

- Improve the quality of the answer at the moment the model *first* achieves a good result ($L^{t_1}$),
- Increase confidence when the model already considers the answer correct ($L^{t_2}$).

This scheme offers several advantages:

1. The model does not rely solely on the final internal step $T$: it is incentivized to solve the task earlier if possible (otherwise $L^{t_1}$ will be large and contribute significantly).
2. If on some step the model is already confidently correct, the error on that step is very small, reducing the final loss — thus rewarding the model for an *early confident solution*.
3. If the task is complex and requires all $T$ reasoning steps, the model can still gradually reduce error toward the end and increase confidence. Then both terms $t_1$ and $t_2$ coincide with $T$, and the penalty becomes $2L^T + L^T = 3L^T$. This encourages the model to improve accuracy on the final step, but also to shift success to earlier steps whenever possible, to avoid tripling the final error.

### **8. Adaptive Reasoning Depth**

This training strategy effectively creates an **internal mechanism for adaptive computational complexity**:

- Simple examples are learned by the model in few ticks, since low $L^t$ and high $C^t$ can be achieved long before $T$.
- Complex examples require more internal steps, forcing the model to “think longer.”

As a result, CTM demonstrates **Adaptive Compute** capability: it autonomously regulates how many internal iterations to employ for different inputs.

In practice, when using CTM, one can set a confidence threshold and terminate the internal loop as soon as $C^t$ exceeds, for example, 0.8 or 0.9. Research shows:

- With a high threshold (requiring very high confidence), the model often uses the maximum number of steps.
- With a low threshold, it stops earlier.
- In both cases, high accuracy is achieved.

Thus, the **mechanism for deciding when to stop** in CTM is based on its own confidence in the prediction. This resembles how a human may cease contemplating a problem once confident in the solution.

In experiments on ImageNet, the authors demonstrate that CTM, at a confidence threshold of 0.9, achieves nearly the same accuracy as with a fixed maximum number of steps, but on average performs fewer iterations on simpler image classes. This confirms that CTM efficiently learns not to waste “thinking resources” where unnecessary, and conversely, to deploy more internal computation for difficult cases.

### Comparison with Standard Approaches

It is worth emphasizing that the above approach is merely one possibility. Theoretically, one could train the model solely on the final output $y^T$, or sum errors across all steps. However:

- The first variant would leave intermediate steps untrained (the model could simply “wait” until the final step to output an answer).
- The second variant (averaging over all $t$) might hinder the model from specializing early vs. late steps.

The chosen scheme with $t_1$ and $t_2$ provides a kind of **self-curriculum**:

- The model first learns to solve the task at least by the end of the tick sequence,
- Then strives to achieve this earlier, increasing confidence and reducing intermediate loss.
- This gradually shifts the solution “leftward” in time—not by rigid enforcement, but gently, as the model becomes ready.

Thus, CTM’s internal time becomes meaningful:

> *Early ticks learn to solve simple aspects of the task; later ticks handle more complex aspects or elevate confidence to the required level.*

---

### Conclusion

Continuous Thought Machine is an intriguing and innovative neural network architecture that takes a significant step toward bridging artificial and biological neural networks. Its capacity for internal reasoning, grounded in neuron synchronization and explicit modeling of time, opens new avenues for AI development. However, for a full assessment of its potential and practical applicability, additional research and experiments are required to address current limitations and validate effectiveness across diverse use cases.

**What worked well:**

- **Biological plausibility** — CTM explicitly models key biological mechanisms such as spike-timing-dependent plasticity (STDP) and neuronal synchronization, making it more akin to the biological brain.
- **Novelty of approach** — Introducing an internal recurrent time dimension and using neuron synchronization as the primary decision-making mechanism represents an original and innovative contribution to deep learning.

**What raises concerns:**

- **Lack of comparative analysis** — Absence of detailed comparison with other modern architectures; a comparative analysis across different benchmarks would be valuable.
- **Unjustified choice of random neuron pairs** — The method of selecting random subsets of neuron pairs to represent synchronization. Pure randomness...
- **Limited data on scalability** — No information on how the model scales with increasing numbers of neurons or task complexity.