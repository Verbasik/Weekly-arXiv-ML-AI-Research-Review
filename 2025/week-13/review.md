# **DAPO: A Revolutionary RL Algorithm from ByteDance**

## **Abstract**

This paper introduces **DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)**, an open platform for training large language models (LLMs) using reinforcement learning (RL). Despite significant advances in modern LLMs such as OpenAI o1 and DeepSeek R1, the key technical details of their RL training remain inaccessible to the research community, severely hindering reproducibility and further investigation. In response, the authors propose the innovative DAPO algorithm, which not only demonstrates high efficiency but also provides full openness of code, data, and methodology.

The DAPO system achieved a record score of 50 points on the **AIME 2024** mathematics competition, surpassing the previous record of DeepSeek-R1 (47 points). Crucially, DAPO achieved this result while reducing the number of training steps by half. The algorithm is built upon four key technologies: **Clip-Higher strategy**, **dynamic sampling**, **token-level policy gradient optimization**, and **intelligent length penalty**. These methods address core challenges in RL training, including entropy collapse, reward noise, and inefficiency in learning from long texts.

The authors emphasize that large-scale reinforcement learning is critical for advancing LLMs’ capacity for complex reasoning. However, unlike prior works where RL training details remained hidden (e.g., in OpenAI’s blog posts and DeepSeek R1 technical reports), DAPO provides full transparency. Not only are the training codes, built on the **verl** framework, openly released, but also meticulously curated datasets. This promotes reproducibility and opens new avenues for research in large-scale RL training of LLMs.

Thus, this work makes a significant contribution to the development of open and reproducible methods for training large language models, offering both theoretical innovations and practical tools for the research community.

## **1. Introduction**

The emergence of models with extended test-time reasoning (Test-time Compute), such as OpenAI’s O1, DeepSeek’s R1, and recently Anthropic’s Claude, has marked a fundamental paradigm shift in large language models (LLMs) based on reinforcement learning (Reinforcement Learning, RL). These models have demonstrated unprecedented capabilities in complex reasoning, enabling them to successfully solve challenging mathematical and programming problems at the level of AIME and Codeforces competitions.

The central technology enabling this breakthrough is large-scale reinforcement learning (RL), which fosters the development of sophisticated reasoning forms—including self-checking, iterative refinement, and reflection. Despite impressive results, the specific algorithms and methodological approaches to scalable RL training remain largely obscured in the technical reports of existing models. As the authors note, “key technical details of modern reasoning LLMs are hidden (e.g., in OpenAI’s blog post on model o1 and DeepSeek R1’s technical report),” making it difficult for the research community to reproduce their results.

During experiments, the authors used Qwen2.5-32B as the pretrained model for reinforcement learning based on feedback. Initial runs using the baseline GRPO (Generalized Reward-weighted Policy Optimization) algorithm yielded only 30 points on the AIME test, significantly lagging behind DeepSeek-RL’s 47 points. Deeper analysis revealed that a naive implementation of GRPO encounters several critical problems:

1. **Entropy collapse** — the tendency of the model to narrow the diversity of generated responses;
2. **Reward noise** — incorrect assignment of rewards for partially correct or overly long responses;
3. **Training instability** — difficulties in scaling training across long reasoning chains.

Similar issues were observed more broadly in attempts to reproduce DeepSeek’s results, suggesting possible omissions of critical training details in the published R1 paper—details essential for developing industrial-scale, scalable, and reproducible RL training systems.

To bridge this gap, the authors present a state-of-the-art open-source large-scale RL training system for LLMs that achieves 50 points on AIME 2024 using the Qwen2.5-32B model. This result surpasses the previous record of DeepSeek-RL-Zero-Qwen-32B (47 points), requiring only 50% of the training steps (see Figure 1). The system is built upon the authors’ proposed algorithm, Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), which incorporates four key innovations that fundamentally enhance RL training efficiency in long reasoning chain scenarios:

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_1.png)

- **Clip-Higher** — a strategy promoting generation diversity by enabling adaptive sampling through separate lower and upper clipping bounds (ε-low and ε-high);
- **Dynamic Sampling** — a dynamic sampling method that improves training efficiency and stability by excluding examples with zero gradients;
- **Token-Level Policy Gradient Loss** — computing policy gradients at the individual token level, critically important for efficient learning on long sequences;
- **Overlong Reward Shaping** — an intelligent length penalty system that reduces reward noise and stabilizes the training process.

A key value of this work lies in the full openness of all system components: the entire implementation code, built on the verl framework, along with the carefully curated DAPO-Math-17K dataset, are publicly available. This openness contrasts sharply with prior works that, despite impressive results, withheld critical training details.

During experiments, the authors also observed an intriguing phenomenon: the model not only reinforces existing reasoning patterns but gradually develops fundamentally new capabilities—particularly behaviors related to self-checking and reconsidering prior steps. This opens new perspectives for understanding the fundamental mechanisms by which LLMs learn complex reasoning.

---

<details> 
    <summary><em><strong>Understanding Test-time Compute</strong></em></summary>

### Understanding Test-time Compute

"**Test-time compute**" refers to a paradigm for scaling RL LLMs that emphasizes increasing computational resources available to the model precisely during the processing of a user query (inference time). Unlike the traditional approach, "Test-time compute" enhances the performance of an already-trained model by providing it with more **TIME** and **COMPUTATIONAL POWER** to "think" about each specific query.

### Difference from Traditional Scaling

Traditional LLM scaling focused on the following aspects **during training**:

* **Model size:** increasing the number of parameters and architectural complexity;
* **Data volume:** expanding and diversifying training data;
* **Training computational resources:** using more powerful GPUs and extending training duration.

"Test-time compute" introduces an **additional scaling dimension**, applied **after model training**. This allows improved model efficiency without altering its architecture or parameters, optimizing computational resources at inference time.

### Mechanism and Advantages of Test-time Compute

Providing the model with greater computational resources during inference enables:

* **Deeper query processing:** the model can perform more detailed analysis of input text and context through deeper reasoning chains;
* **Improved reasoning:** additional computations facilitate better planning, search for optimal solutions, and generation of logically sound responses;
* **Use of complex inference algorithms:** the ability to apply resource-intensive but higher-quality decoding and generation methods.

### In Summary

"Test-time compute" represents a significant shift in LLM scaling approaches. It complements traditional methods by focusing on optimizing computational resources at the point of model usage. This opens prospects for creating more intelligent, reasoning-oriented language models, especially in tasks requiring deep analysis and logical deduction.

</details> 

---

We now step-by-step examine the evolution from PPO to GRPO, and then to DAPO, to understand how this new reinforcement learning algorithm was developed.

## **2. Background**

### **2.1 Proximal Policy Optimization (PPO)**

#### Core Concept

Proximal Policy Optimization (PPO) is one of the most popular and effective reinforcement learning algorithms, developed by researchers at OpenAI in 2017. PPO is an improvement over earlier policy optimization methods such as TRPO (Trust Region Policy Optimization), with a simpler implementation and comparable or superior performance.

#### Key Features of PPO

1. **Training stability**: The core idea of PPO is to constrain policy updates between iterations, preventing overly drastic changes that could destabilize learning;
2. **Clipping mechanism**: PPO employs a clipping function to bound the importance sampling ratio, ensuring the new policy does not deviate significantly from the old one;
3. **Sample efficiency**: Compared to other algorithms, PPO makes more efficient use of collected data, achieving better results with fewer environmental interactions.

#### Mathematical Formulation

The PPO objective function is defined as:

$$\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, o_{<t} \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \min \left( \frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})}, 1 - \varepsilon, 1 + \varepsilon \right) \hat{A}_t \right) \right], \quad (1)$$

where:

- $\mathcal{J}_{\text{PPO}}(\theta)$ — the PPO objective function to be maximized by tuning parameters $\theta$;
- $(q, a)$ — a question-answer pair from data distribution $\mathcal{D}_e$, where $q$ represents the query (or environment state), and $a$ the corresponding response (or action);
- $o_t$ — the current output token at step $t$;
- $o_{<t}$ — the sequence of preceding output tokens before step $t$;
- $\pi_\theta(o_t | q, o_{<t})$ — the probability of selecting token $o_t$ under the current policy with parameters $\theta$, given query $q$ and prior tokens $o_{<t}$;
- $\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})$ — the probability of selecting the same token under the old policy with parameters $\theta_{\text{old}}$;
- $\frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})}$ — the importance sampling ratio, indicating the relative likelihood of selecting the token under the new versus old policy;
- $\varepsilon$ — the clipping hyperparameter (typically in range 0.1–0.3), defining the permissible range of change in the importance ratio;
- $\text{clip}(x, a, b)$ — the clipping function, constraining value $x$ within range $[a, b]$;
- $\hat{A}_t$ — the advantage estimate at time step $t$, characterizing the relative value of the selected action compared to the average value of all actions in the current state.

The function $\mathbb{E}$ denotes the mathematical expectation, and the entire expression represents the expected value of selected actions under policy change constraints.

Given the value function $V$ and reward function $R$, $\hat{A}_t$ is computed using the Generalized Advantage Estimation (GAE):

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l} \quad (2)$$

where:

- $\hat{A}_t^{GAE(\gamma,\lambda)}$ — the generalized advantage estimate for time step $t$;
- $\gamma$ — the discount factor for future rewards (typically close to 1, e.g., 0.99), determining how much the agent values future rewards relative to immediate ones;
- $\lambda$ — the exponential weighting parameter for different advantage estimates (typically between 0.9 and 1.0), controlling the bias-variance trade-off in estimation;
- $\delta_t$ — the temporal difference between expected and actual value, computed by formula (3);
- $l$ — the summation index representing the number of steps ahead from current time step $t$.

$$\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t), \quad 0 \leq \gamma, \lambda \leq 1 \quad (3)$$

where:

- $\delta_t$ — the temporal difference at step $t$;
- $R_t$ — the immediate reward received after taking action at step $t$;
- $V(s_t)$ — the estimated value of state $s_t$ according to the current value function;
- $V(s_{t+1})$ — the estimated value of the next state $s_{t+1}$.

**Summary: What’s “under the hood” without mathematical details:**

* **Policy**: This is like an “instruction manual” for the program, dictating which action to take in each situation. In PPO, the policy is represented by a neural network.
* **Value Function**: This estimates how “good” a given state is. It helps the program understand which states to aim for. Also typically a neural network.
* **Advantage**: This is the difference between how good the chosen action was and how good actions typically are in this situation. It helps identify which actions were truly successful.
* **PPO Objective Function (Equation 1)**: This is the main “goal” of learning. The program seeks to maximize it to improve. The most critical element is the “clipping” mechanism, which constrains policy changes.
* **GAE Advantage Estimation (Equation 2)**: A method to evaluate how good the program’s actions were, considering not only immediate outcomes but also future consequences.
* **Temporal Difference (Equation 3)**: Helps the value function learn by comparing the predicted state value with what actually occurred.

#### PPO Workflow

1. **Experience Collection**: Agents interact with the environment following the current policy $\pi_{\theta_{\text{old}}}$, collecting trajectories (sequences of states, actions, rewards).
2. **Advantage Estimation**: For each time step in collected trajectories, the advantage estimate $\hat{A}_t$ is computed using equations (2) and (3).
3. **Policy Optimization**: Policy parameters $\theta$ are updated via stochastic gradient ascent on the objective function (1) over multiple epochs.
4. **Value Function Update**: Value function parameters are updated by minimizing the mean squared error between predicted and actual state values.
5. **Iteration**: Steps 1–4 are repeated until desired performance or a set number of iterations is reached.

#### Advantages of PPO

1. **Simple implementation**: PPO is easier to implement than TRPO and other complex RL algorithms.
2. **High performance**: PPO achieves excellent results across diverse tasks, from Atari games to complex robotics.
3. **Compatibility with neural networks**: PPO works seamlessly with deep neural networks and can train complex policies.
4. **Good scalability**: PPO can be efficiently parallelized to accelerate training.

#### Applications of PPO

PPO is widely used across domains:

1. **Games**: From simple games to complex strategies and simulators.
2. **Robotics**: Training robots to perform complex physical tasks.
3. **Recommendation Systems**: Optimizing recommendation strategies in interactive systems.
4. **Large Language Models**: In recent years, PPO has been actively applied to fine-tune language models via RLHF (Reinforcement Learning from Human Feedback).

#### Conclusion

Proximal Policy Optimization (PPO) is a powerful and flexible reinforcement learning algorithm that, due to its simplicity and effectiveness, has become the de facto standard in deep reinforcement learning. Its applications span from games and robotics to training modern language models, demonstrating the universality and power of this approach.

### **2.2 Group Relative Policy Optimization (GRPO)**

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed for optimizing LLMs in tasks requiring structured reasoning, such as mathematics and logic. It was introduced in the works of DeepSeekMath and DeepSeek-R1 **as a response to the challenges of training models with billions of parameters**. GRPO offers a more efficient approach compared to traditional methods such as Proximal Policy Optimization (PPO), **by eliminating key bottlenecks associated with advantage function computation**.

<details> 
    <summary><em><strong>Explanation of Advantage Functions</strong></em></summary>

**Advantage function** is a fundamental concept in reinforcement learning (RL) that **quantitatively evaluates the advantage of selecting a specific action `a` in state `s` relative to the average action prescribed by the current model policy**. Formally, it is expressed as the difference between the **Q-function** (expected cumulative reward for action `a` in state `s`) and the **V-function** (average expected reward in state `s` under the current policy):

$$
A(s, a) = Q(s, a) - V(s)
$$

---

### **Why is the Advantage Function Needed?**
1. **Assessing Relative Action Value**:
   - Helps the model understand how much better or worse a specific action is compared to the “standard” behavior in a given context under the current policy.
   - Example: In a math problem, the action "choose integration by parts" may have high advantage if it leads to a correct answer, and low advantage if it complicates the solution.

2. **Reducing Gradient Variance**:
   - Using relative advantage values instead of absolute rewards leads to more stable policy updates.

---

### **How are Advantage Functions Computed in Classical RL (e.g., PPO)?**  
In Proximal Policy Optimization (PPO):  
1. The **Value network** (a separate neural network) is trained to predict $V(s)$ — the expected reward for state $s$.  
2. $Q(s, a)$ is estimated via the actual received reward plus discounted future rewards.  
3. The **Advantage** is computed as:  
$$
A(s, a) = R_{\text{total}} - V(s)
$$  
where $R_{\text{total}}$ is the discounted sum of rewards over the trajectory.  

**Problems with PPO**:  
- The Value network requires additional computational resources and memory.  
- Errors in predicting $V(s)$ (especially in tasks with **multimodal reward distributions**, as in LLMs) distort advantage values.  

> Here will be a link to the notebook  

</details>  

---

### **GRPO’s Innovative Approach to Advantage Functions**

![Figure_19](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_19.jpg  )

<details> 
    <summary><em><strong>Step-by-step PPO and GRPO:</strong></em></summary>

## PPO (Proximal Policy Optimization)

**1. Input Query (q):**

- **Description:** The process begins with an input query, denoted as `q`. In the context of language models, this may be a textual prompt or question to which the model must generate a response.  
- **On diagram:** Left block labeled `q`.

**2. Policy Model:**

- **Description:** This is a neural network that takes query `q` as input and generates output `o`. `o` represents the generated response or sequence of actions (tokens, in the case of LLMs), based on the current policy. The policy model aims to maximize expected reward.  
- **On diagram:** Yellow block labeled `Policy Model`, taking input from `q` and outputting `o`. Yellow color indicates this model is **trainable** (Trained Models).

**3. Reference Model and Reward Model:**

- **Description:** After the Policy Model generates output `o`, it is fed into two blocks: `Reference Model` and `Reward Model`.  
    - **Reference Model:** This is a **frozen** (Frozen Models, blue color) model representing either the previous version of the Policy Model or another model to which the current policy should not deviate too strongly. It is used to compute the **KL divergence**, which regulates policy updates to prevent overly drastic changes.  
    - **Reward Model:** This model evaluates the quality of output `o` and assigns it a reward `r`. As described in the text, on the **first iteration** of GRPO training, this reward is taken from an **external reward function**, which may be manual, automated, or rule-based. On the diagram, the Reward Model is also shown as **frozen** (blue color), implying it is not updated during this training stage but serves as an external evaluation source.  
- **On diagram:** Two blue blocks `Reference Model` and `Reward Model`, taking input `o`. An arrow labeled `KL` goes from `Reference Model` to the symbol "⊕", indicating use of **KL divergence**. The Reward Model outputs reward `r`.

**4. Value Model:**

- **Description:** In PPO, a Value Model is used to estimate the “value” of a state. It takes output `o` from the Policy Model as input and predicts a scalar value `v`, representing the expected cumulative reward obtainable starting from this state (or after generating this output). The Value Model is used to compute the baseline for the **advantage function**.  
- **On diagram:** Yellow block `Value Model`, taking input `o` and outputting `v`. Yellow color indicates the Value Model is also **trainable**.

**5. KL Divergence and Operation "⊕":**

- **Description:** **KL divergence** measures the difference between probability distributions—in this case, between the current Policy Model and the Reference Model. On the diagram, the symbol "⊕" denotes an operation in which **KL divergence** is used for regularization. In PPO, **KL divergence** is often added to the loss function to constrain policy changes and ensure training stability.  
- **On diagram:** Arrow labeled `KL` from `Reference Model` to symbol "⊕".

**6. Reward (r) and Value (v):**

- **Description:** `r` is the reward received from the Reward Model for output `o`. `v` is the state value estimate from the Value Model.  
- **On diagram:** Blocks `r` and `v`, representing outputs of the Reward Model and Value Model, respectively.

**7. GAE (Generalized Advantage Estimation) and Advantage (A):**

- **Description:** **GAE** (Generalized Advantage Estimation) is a method for estimating the **advantage function** `A`. The **advantage function** indicates how much better or worse an action taken in a given state is compared to the average action in that state. In PPO, GAE uses both rewards `r` and value estimates `v` to compute `A`. The GAE formula typically includes discounted rewards and the difference between current and next value estimates.  
- **On diagram:** Block `GAE`, taking inputs reward `r` and value estimate `v`, and outputting advantage `A`.

**8. Advantage (A) and Policy Model (Feedback):**

- **Description:** The computed advantage `A` is used to update the Policy Model. If the advantage is positive, it means the action (or generated output) was better than expected, and the Policy Model is updated to increase the likelihood of generating similar outputs in the future. If the advantage is negative, the Policy Model is adjusted to reduce the probability of such outputs.  
- **On diagram:** Arrow from block `GAE` with advantage `A` back to `Policy Model`, indicating the learning and parameter update process based on advantage.

**9. Trained Models and Frozen Models:**

- **Description:** Blocks `Trained Models` (yellow) and `Frozen Models` (blue) indicate which models are updated during training and which remain unchanged. In PPO, the Policy Model and Value Model are trainable, while the Reference Model and Reward Model (on the first iteration of GRPO) are frozen.  
- **On diagram:** Yellow blocks `Policy Model` and `Value Model` reside in the container `Trained Models`. Blue blocks `Reference Model` and `Reward Model` reside in the container `Frozen Models`.

**In summary, the PPO process works as follows:** The query `q` is fed into the Policy Model, which generates output `o`. This output is evaluated by the Reward Model (yielding `r`) and the Value Model (yielding `v`). The Reference Model is also used to compute **KL divergence**. Based on `r` and `v`, the advantage `A` is calculated using GAE. Advantage `A` is then used to update the Policy Model and Value Model to improve the policy and value estimation in the future. **KL divergence** is used to regularize the learning process.

---

## GRPO (Group Relative Policy Optimization)

**1. Input Query (q):**

- **Description:** Analogous to PPO, the process begins with an input query `q`.  
- **On diagram:** Left block labeled `q`.

**2. Policy Model and Group of Outputs (o₁, o₂, ..., o…):**

- **Description:** In GRPO, the Policy Model also takes query `q` as input, but instead of generating a single output, it generates a **group** of `G` distinct outputs: `o₁, o₂, ..., o…`. As explained in the text, these are **horizontally diverse, variant responses to the same prompt**. This means that for a single query, the Policy Model generates multiple alternative responses.  
- **On diagram:** Yellow block `Policy Model`, taking input `q` and outputting a group of outputs represented by blocks labeled `o₁`, `o₂`, ..., `o…`. Yellow color indicates the Policy Model is **trainable**.

**3. Reference Model and Reward Model:**

- **Description:** Each output from the group `oᵢ` is fed into the Reference Model and Reward Model. Analogous to PPO, the Reference Model is **frozen** and used for **KL regularization**. The Reward Model is also **frozen** and evaluates each group output `oᵢ`, assigning it a reward `rᵢ`.  
- **On diagram:** Blue blocks `Reference Model` and `Reward Model`, taking input group of outputs `o₁, o₂, ..., o…`. Arrow labeled `KL` from `Reference Model` to symbol "⊕". Reward Model outputs group of rewards `r₁, r₂, ..., r…`.

**4. KL Divergence and Operation "⊕":**

- **Description:** Similar to PPO, **KL divergence** is used for regularization to constrain policy changes.  
- **On diagram:** Arrow labeled `KL` from `Reference Model` to symbol "⊕".

**5. Rewards (r₁, r₂, ..., r…):**

- **Description:** `r₁, r₂, ..., r…` are the rewards obtained from the Reward Model for each output in the group `o₁, o₂, ..., o…`.  
- **On diagram:** Block `r₁, r₂, ..., r…`, representing the group of rewards.

**6. Group Computation and Advantages (A₁, A₂, ..., A…):**

- **Description:** The key distinction between GRPO and PPO lies in the `Group Computation` block. In GRPO, **there is no Value Model**. Instead, the advantage function is computed via **group-wise relative normalization**. As described in the text, the advantage for each response $O_i$ in group $G = \{O_1, O_2, ..., O_N\}$ is computed as:

$$
A_i(O_i, G) = R_i - \bar{R}_G = R_i - \frac{1}{N} \sum_{j=1}^N R_j
$$

where $\bar{R}_G$ is the average reward across the group. Thus, `Group Computation` takes the group of rewards `r₁, r₂, ..., r…` and computes for each output `oᵢ` the advantage `Aᵢ` as the difference between its reward `rᵢ` and the group’s average reward.

- **On diagram:** Block `Group Computation`, taking input group of rewards `r₁, r₂, ..., r…` and outputting group of advantages `A₁, A₂, ..., A…`.

**7. Advantages (A₁, A₂, ..., A…) and Policy Model (Feedback):**

- **Description:** The group of advantages `A₁, A₂, ..., A…` is used to update the Policy Model. The policy is updated to increase the probability of generating outputs with higher advantages (i.e., outputs better than the group average).  
- **On diagram:** Arrow from block `Group Computation` with advantages `A₁, A₂, ..., A…` back to `Policy Model`, indicating the learning process.

**8. Trained Models and Frozen Models:**

- **Description:** Analogous to PPO, the Policy Model is trainable, while the Reference Model and Reward Model (on the first iteration of GRPO) are frozen.  
- **On diagram:** Yellow block `Policy Model` in container `Trained Models`. Blue blocks `Reference Model` and `Reward Model` in container `Frozen Models`.

**In summary, the GRPO process works as follows:** The query `q` is fed into the Policy Model, which generates a group of outputs `o₁, o₂, ..., o…`. Each output in the group is evaluated by the Reward Model (yielding `r₁, r₂, ..., r…`) and used by the Reference Model for **KL regularization**. Then, in the `Group Computation` block, the advantage `Aᵢ` for each output is computed as the difference between its reward and the group’s average reward. The group of advantages `A₁, A₂, ..., A…` is used to update the Policy Model. **Importantly, GRPO omits the Value Model entirely.**

**Key differences between GRPO and PPO, clearly shown in the diagram:**

1.  **Absence of Value Model in GRPO:** The most noticeable difference—GRPO does not use a Value Model. Instead of estimating absolute state value, GRPO focuses on **relative comparison within a group of responses**.

2.  **Grouped Processing of Outputs and Rewards in GRPO:** GRPO generates and processes a group of outputs per query, then computes advantage based on reward comparisons within that group. PPO processes each output individually and uses a Value Model to estimate the baseline.

3.  **Group Computation in GRPO:** This block is unique to GRPO and implements **group-wise relative normalization**, computing advantage as the difference between an individual response’s reward and the group’s average reward.

</details>

---

![Visualization of Reward Evaluation Process in GRPO for DeepSeek-R1 by Jay Alammar](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_2.png  )

GRPO completely eliminates the need for a value network by using **group-wise relative normalization**:  
For each prompt $P$, a group of $N$ responses $G = \{O_1, O_2, ..., O_N\}$ is generated using policy $\pi$. Each response $O_i$ is assigned a reward $R_i = R(O_i)$ reflecting its quality. The advantage function for the $i$-th response $O_i$ relative to group $G$ is computed as:

$$
A_i(O_i, G) = R_i - \bar{R}_G = R_i - \frac{1}{N} \sum_{j=1}^N R_j
$$

where:

- In GRPO, the reward estimate $R_i$ (reward for response $O_i$) on the **first iteration** is taken from an **external reward function** $R(O_i)$, which:

    1. **Is independent of the current policy $\pi$.**  
    - This may be:  
        - Manual annotation (e.g., expert quality assessments).  
        - Automated algorithm (e.g., a critic model, a separate ML model scoring texts).  
        - Rule-based heuristic (e.g., format compliance, keyword presence).  

    2. **Requires prior configuration.**  
    - If a critic model is used, it must be pre-trained on annotated data.  
    - If annotations are manual, a dataset with human ratings must be prepared.

- $\bar{R}_G = \frac{1}{N} \sum_{j=1}^N R_j$ — the average reward across group $G$.

> In essence, the advantage function in GRPO for each specific response is calculated as: the response’s reward minus the arithmetic mean of all rewards in the group.

The group $G$ in GRPO context represents **horizontally diverse variant responses to the same prompt $P$**, not sequential steps along a trajectory.  

**Explanation:**  
1. **Horizontal Variability:** For each prompt $P$, policy $\pi$ generates $N$ **alternative responses** ($O_1, O_2, \dots, O_N$), which are independent variants—not parts of a single chain (trajectory). This resembles sampling multiple possible answers to a single question.  

2. **Intra-group Comparison:** The advantage function $A_i$ measures how much a specific response $O_i$ is better or worse than the **group average** ($\bar{R}_G$). This requires all responses in $G$ to be parallel alternatives; otherwise, the mean loses its meaning as a relative baseline.  

3. **Elimination of Value Network:** GRPO replaces the assessment of “absolute” utility (via a value network) with **relative comparison within a group**. For this, the group must contain diverse responses to the same prompt so that the average reward $\bar{R}_G$ reflects the group’s overall quality.  

**Key Features of the GRPO Approach:**

*   **Group-wise Relative Normalization:** The advantage function is computed relative to a group of responses generated for the same prompt, ensuring relative quality assessment;  
*   **Elimination of Value Network:** The group’s average reward $\bar{R}_G$ serves as the baseline, replacing the need for a separate value network to estimate state or action values;  
*   **Learning via Comparison:** GRPO focuses on training a policy that generates responses superior to the average within the group, making it effective in tasks where relative quality assessment matters;  
*   **KL Divergence: Strict Integration into Loss Function via Relative Weights**: KL divergence is incorporated into the loss function to regularize, limiting the magnitude of policy changes at each training step and preventing abrupt oscillations, thus enhancing training stability.

**Limitations and Notes:**

*   The effectiveness of the GRPO approach depends on the quality of the reward function $R(O)$. The reward function must be properly designed to accurately reflect desired response properties.  
*   Group size $N$ is a hyperparameter that may influence training stability and efficiency. Optimal $N$ may require experimental tuning.  
*   Like other reinforcement learning methods, GRPO may be sensitive to optimizer hyperparameters and model architecture choices.

---

### **Practical Interpretation for LLMs**

In GRPO, the advantage function becomes a **ranking tool for response variants**:  
- The model learns to generate responses that are not merely “good,” but **significantly better than the group average**.  
- This encourages:  
  - Discovery of non-obvious but effective reasoning chains.  
  - Avoidance of template-based errors common within the group.

**Effect**: The model focuses on **qualitative differences between responses**, not absolute reward values—critical for complex tasks with ambiguous success criteria.

**Problem Context**:  
- In reasoning tasks, LLMs often generate multiple “chain-of-thought” reasoning paths, but standard RL algorithms are poorly adapted to evaluate them.  
- **Value networks in PPO demand significant training resources and are prone to errors under multimodal reward distributions**.

---

### **Key Differences Between GRPO and PPO**

| **Characteristic**                   | **PPO**                               | **GRPO**                                                                 |
|-------------------------------------|---------------------------------------|---------------------------------------------------------------------------|
| Presence of Value Network           | Required                              | Eliminated                                                                |
| Advantage Estimation                | Based on Value Network                | **Group-wise Relative Normalization within Trajectories**                 |
| KL Divergence                       | Optional Regularization               | **Strict Integration into Loss Function via Relative Weights**           |
| Memory Usage                        | High (2 models)                       | **Reduced by 40–60% due to removal of Value Network**                     |
| Convergence                         | Depends on Value Network Accuracy     | **More stable due to group-wise gradient stabilization**                  |

---

### **Mathematical Formulation**

**GRPO Objective Function**:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)} \left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\text{ clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta D_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right)\right],
$$

where:
- **$\theta$** — parameters of the **current policy** (neural network) optimized during training.
- **$q$** — the **query**, input to the language model.
- **$a$** — the **answer**, generated by the model for query $q$.
- **$\mathcal{D}$** — dataset of question-answer pairs used to train the model.
- **$o_i$** — the $i$-th **output** generated by model $\pi_{\theta_{\text{old}}}$ for query $q$.
- **$G$** — number of outputs generated per query.
- **$|o_i|$** — length of the $i$-th output (number of tokens).
- **$r_{i,t}(\theta)$** — ratio of probabilities between current and old policies for the $t$-th token in the $i$-th output:
  $$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$$
- **$\hat{A}_{i,t}$** — **advantage** estimate for the $t$-th token in the $i$-th output, computed as the difference between the expected reward for selecting this token and the average reward in the current context.
- **$\text{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)$** — clipping function constraining the probability ratio within range $[1-\varepsilon, 1+\varepsilon]$ to prevent excessively large policy updates per training step.
- **$\varepsilon$** — hyperparameter defining the allowable range of probability ratio change (typically 0.1–0.2).
- **$D_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})$** — KL divergence between the action distributions of the current policy $\pi_\theta$ and the reference policy $\pi_{\text{ref}}$, used for regularization:  
  $$D_{KL}(\pi_\theta||\pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta} \left[ \log \frac{\pi_\theta(o|q)}{\pi_{\text{ref}}(o|q)} \right].$$
- **$\pi_{\text{ref}}$** — reference policy, typically a pretrained model to which the policy is encouraged to remain close during fine-tuning.
- **$\beta$** — hyperparameter controlling the strength of KL regularization (**typical values: 0.05–0.2**).

Key features of this GRPO formulation:
1. Optimization is performed across multiple generated outputs ($G$) per query.
2. Individual token contributions within output sequences are accounted for.
3. Clipping technique is applied to stabilize training.
4. KL regularization is employed to maintain proximity to the reference model.
5. The formulation is specifically adapted for training generative language models.

---

### **Explanations**

1. **Off-policy learning**: Gradients are computed on data collected by the old policy ($\pi_{\text{old}}$), while optimizing the new policy ($\pi_\theta$).  
2. **Importance weighting** $\frac{\pi_\theta}{\pi_{\text{old}}}$ adjusts gradients to account for policy differences, preventing estimation bias.  
3. **KL divergence** constrains the rate of policy change, ensuring training stability.  
4. **Advantage $A(s,a)$** guides updates toward actions with higher expected reward. If $A(s,a) > 0$, action $a$ in state $s$ is considered better than average.

**Optimization**:  
- Gradients are updated only for tokens critically influencing reward (**e.g., key steps in mathematical derivation**).  
  - *Formally*, this can be represented by applying a mask $( M )$ to gradients, where $( M_i = 1 )$ for "critical" tokens and $( M_i = 0 )$ for others. Thus, only parameters associated with "critical" tokens are updated, improving training efficiency by focusing on the most significant reasoning components.  
- **Response sampling**: For each prompt, 4–8 variants are generated in parallel, enhancing solution space coverage.

---

### **A Few Numbers**

1. **Efficiency**:  
   - Removal of the value network reduces memory usage by **18.2 GB for a 33B-parameter model** (DeepSeek-R1 experiments).  
   - Training time is reduced by **35%** on MATH dataset tasks.

2. **Stability**:  
   - Group normalization reduces gradient variance (**by 60% compared to PPO**).  
   - KL regularization prevents "policy collapse"—a common PPO issue.

3. **Performance**:  
   - On the MATH benchmark, GRPO improved DeepSeek-Math-7B accuracy from **51.2% to 58.7%**.  
   - In logical reasoning tasks (e.g., FOLIO), improvement reached **12.3%**.

---

### **Practical Implementation of GRPO**

**Implementation Steps**:  
1. **Supervised Fine-Tuning (SFT)**:  
   - Use data in format:  
     ```json
     {"prompt": "Solve ∫₀¹ x² dx", "response": "∫₀¹ x² dx = [x³/3]₀¹ = 1/3"}
     ```
   - **Key aspect**: Clean data via self-consistency checks.

2. **Reward Modeling**:  
   - For math tasks (example):  
     
    $$
     [
       R = \text{Correctness} + 0.5 \cdot \text{StepQuality} \;-\; 0.3 \cdot \text{LengthPenalty}.
     ]
    $$

   - Designing an effective reward function is critical to GRPO. Generally, it should encourage desired reasoning properties: correctness, logical coherence, conciseness, and solution efficiency. Weight coefficients (e.g., 1, 0.5, -0.3 in the example) can be empirically tuned to achieve optimal balance among these properties.

3. **GRPO Training**:  
   - **Hyperparameters**:  
     - Batch size: 512 prompts (4 responses per prompt → 2048 examples per step).  
     - Learning rate: 1e-6 with linear decay.  
   - **Trick**: Freeze the first 10% of model layers to preserve general knowledge.

---

### **Use Cases**

1. **DeepSeek-Math-33B**:  
   - Solved International Mathematical Olympiad (IMO) problems with **44.5% accuracy**.  
   - **Feature**: Combined GRPO with Monte Carlo Tree Search (MCTS) for step generation.

2. **Logical Planner AlphaLogic**:  
   - Automated theorem proving in Coq with **68% success rate** (vs. 52% for PPO).

---

### **Conclusion**

GRPO represents a significant advancement in reinforcement learning for LLMs, particularly in tasks requiring complex reasoning. **Its application is already extending beyond mathematics—current research is testing GRPO in legal analysis and scientific hypothesis generation.** Despite limitations, the algorithm demonstrates strong potential for creating "thinking" AI systems capable of deep abstract reasoning.

### **2.3 Elimination of KL Divergence**

**Elimination of KL Divergence in DAPO for Training Models with Long Reasoning Chains**

In RLHF (Reinforcement Learning with Human Feedback) scenarios, a KL divergence-based penalty term is traditionally used to regulate deviations between the updating online policy and the frozen reference policy. Its primary purpose is to ensure that during training, the model adjusts its behavior without straying too far from the original data distribution, which is critical for preserving predictability and stability.

However, when training models that generate long reasoning chains (Chain-of-Thought, CoT), this constraint loses relevance. In such tasks, the model’s distribution naturally and significantly diverges from the original due to the complexity and multi-step nature of reasoning. Strict regulation via KL divergence becomes redundant, as it artificially limits the model’s ability to explore alternative generation strategies essential for effectively solving multi-stage problems.

The DAPO (Decoupled Adaptive Policy Optimization) algorithm proposes eliminating the KL penalty to mitigate this limitation. Removing the KL divergence term allows the model to adapt freely during training without being anchored to the initial reference policy distribution. This is especially crucial in scenarios where successful task completion requires moving beyond template solutions—for instance, when generating complex logical conclusions or creative texts. Thus, DAPO focuses on balancing exploration of new strategies with efficient policy optimization, enhancing model flexibility in long-form reasoning without compromising generation quality.

This approach underscores that in certain RLHF scenarios, strict control over deviation from the initial policy can be avoided to fully unlock the model’s adaptive potential in complex and ambiguous tasks.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">First Checkpoint:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">DAPO eliminates KL divergence penalties in RLHF for long-CoT tasks, enabling greater policy flexibility and enhanced reasoning capabilities.</p>
</div>

<details> 
    <summary><em><strong>Explanation of KL Divergence:</strong></em></summary>

#### **Explanation of KL Divergence**

In RLHF (Reinforcement Learning with Human Feedback), a key element is the use of a KL divergence-based penalty term to regularize the training process. This penalty plays a crucial role in managing deviations between the updating online policy $\pi_{\theta}$ and the frozen reference policy $\pi_{ref}$. The main goal of KL regularization is to ensure that during training, the model adjusts its behavior without deviating too far from the original data distribution represented by the reference policy. This is particularly important for preserving predictability, training stability, and preventing catastrophic forgetting of previously learned useful strategies.

Mathematically, the Kullback-Leibler (KL) divergence, denoted as $D_{KL}(P||Q)$, measures the "distance" between two probability distributions $P$ and $Q$. In RLHF, KL divergence is used to quantify the difference between the new policy $\pi_{\theta}$ and the reference policy $\pi_{ref}$.

The formula for KL divergence between two discrete distributions $P(x)$ and $Q(x)$ is:

$$D_{KL}(P||Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

For continuous distributions, the formula is analogous but uses an integral instead of a sum:

$$D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx$$

where $p(x)$ and $q(x)$ are the probability density functions for distributions $P$ and $Q$, respectively.

**Deep Explanation of Kullback-Leibler (KL) Divergence**:

1.  **Intuitive Understanding**: KL divergence can be interpreted as a measure of "information loss" incurred when using distribution $Q$ to approximate the true distribution $P$.

2.  **Components of the Formula**:
    *   **$P(x)$ (or $p(x)$)**: This is the distribution we consider "true" or "target." In RLHF, in the context of the penalty term, this is often the distribution produced by the reference policy $\pi_{ref}$.
    *   **$Q(x)$ (or $q(x)$)**: This is the distribution we use to approximate $P(x)$. In RLHF, this is the distribution produced by the current trainable policy $\pi_{\theta}$.
    *   **$\frac{P(x)}{Q(x)}$**: This is the probability ratio. If $P(x)$ is significantly larger than $Q(x)$, this ratio becomes large, and its logarithm becomes a large positive number. This indicates that using $Q$ instead of $P$ at point $x$ results in substantial "information loss."
    *   **$\log \left( \frac{P(x)}{Q(x)} \right)$**: The logarithm makes the measure additive and transforms the probability ratio into a more convenient scale. Typically, the natural logarithm (base $e$) is used, but base-2 logarithms (in information theory, where the unit is the bit) may also be used.
    *   **$P(x) \log \left( \frac{P(x)}{Q(x)} \right)$**: Each logarithmic ratio is weighted by the probability $P(x)$. This means that points $x$ with high probability under distribution $P$ contribute more to the overall KL divergence.
    *   **$\sum_{x}$ (or $\int_{-\infty}^{\infty}$)**: Summing (or integrating) over all possible values of $x$ yields the total KL divergence between distributions $P$ and $Q$.

3.  **Properties of KL Divergence**:
    *   **Non-negativity**: $D_{KL}(P||Q) \ge 0$. KL divergence is always non-negative. It equals zero if and only if $P = Q$ almost everywhere;
    *   **Asymmetry**:  $D_{KL}(P||Q) \neq D_{KL}(Q||P)$ in general. This means the "distance" from $P$ to $Q$ is not the same as the "distance" from $Q$ to $P$. It is critical to understand which distribution is the "target" ($P$) and which is the "approximation" ($Q$). In $D_{KL}(P||Q)$, we measure how much $Q$ deviates from $P$.

4.  **KL Divergence in RLHF**:
    *   **Policy Regularization**: In RLHF, KL divergence is added as a penalty term in the RL objective function. The goal is to train policy $\pi_{\theta}$ to maximize reward while not deviating too far from the reference policy $\pi_{ref}$.
    *   **Training Stabilization**: Constraining policy changes via KL divergence helps stabilize training. Without this constraint, the policy may change drastically from iteration to iteration, leading to instability and degraded performance.
    *   **Preventing "Policy Drift"**: The reference policy $\pi_{ref}$ often represents a policy trained initially or one that exhibits desired behavior. KL regularization helps prevent $\pi_{\theta}$ from drifting too far from $\pi_{ref}$, preserving important characteristics of the original policy.
    *   **Balance Between Exploration and Exploitation**: The KL penalty allows the model to explore new strategies while keeping it within reasonable bounds, preventing complete abandonment of previously learned behaviors encoded in the reference policy.

5.  **Application in RLHF Objective Function**: In a typical RLHF objective function optimized by algorithms like PPO (Proximal Policy Optimization), KL divergence is added as a penalty term:

    $J(\theta) = \mathbb{E}_{s \sim d_{\pi_{\theta}}, a \sim \pi_{\theta}} \left[ r(s, a) - \beta D_{KL}(\pi_{\theta}(. | s) || \pi_{ref}(. | s)) \right]$

    where:
    *   $J(\theta)$ — the objective function to be maximized;
    *   $r(s, a)$ — the reward function;
    *   $\beta$ — the regularization coefficient determining the strength of the KL penalty. Higher $\beta$ imposes stronger penalties for deviation from the reference policy;
    *   $D_{KL}(\pi_{\theta}(. | s) || \pi_{ref}(. | s))$ — KL divergence between the action distributions of the current policy $\pi_{\theta}$ and the reference policy $\pi_{ref}$ for state $s$.

In conclusion, KL divergence is a powerful tool for regularizing RLHF training, ensuring a balance between reward optimization and maintaining model stability and predictability by controlling deviations of the new policy from a given reference policy.

</details>

### **2.4 Rule-Based Reward Modeling**

Traditional reward models often suffer from reward hacking, where the model manipulates the reward signal to obtain high scores without genuinely improving reasoning ability. DAPO directly uses the final correctness of verifiable tasks as the reward, bypassing the complexity of reward modeling. Specifically, the reward function is defined as:

$$ R(\hat{y}, y) = \begin{cases} 1, & \text{if } \hat{y} \text{ is equivalent to } y \\ -1, & \text{otherwise} \end{cases}, \quad (7) $$

This approach has proven effective across domains including automated theorem proving, computer programming, and mathematical competitions.

> IMHO: This reward modeling works only for deterministic tasks with unambiguous answers. For tasks with open-ended answers (e.g., LLM responses based on heuristics rather than strict proofs), this approach will not work.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Second Checkpoint:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">DAPO replaces complex reward models with direct use of task final accuracy, eliminating reward hacking and simplifying training.</p>
</div>

## **3. DAPO Algorithm**

Researchers proposed the Decoupled Clip and Dynamic Sampling Strategy Optimization (DAPO) algorithm. DAPO samples a group of outputs ${o_i}_{i=1}^G$ for each question $q$ associated with answer $a$, and optimizes the policy through the following objective function:

$$J_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta,\text{old}}(·|q)}$$

$$\left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}\right)\hat{A}_{i,t}\right) \right], \quad (8)$$

$$\text{s.t.}\ 0 < |\{o_i | \text{is\_equivalent}(a, o_i)\}| < G,$$

where

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta,\text{old}}(o_{i,t} | q, o_{i,<t})}, \quad \hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}, \quad (9)$$

We now break down the key technologies underlying DAPO.

## **📌 3.1 Clip-Higher: Raising the Limit**

In reinforcement learning (RL) algorithms such as Proximal Policy Optimization (PPO) and Generalized Proximal Policy Optimization (GRPO), a common phenomenon is entropy collapse, where policy entropy rapidly decreases during training. This leads to generated responses becoming nearly identical, indicating limited exploration of the action space and premature policy determinism. This paper proposes the Clip-Higher strategy—a modification of the standard clipping mechanism in PPO—designed to address this issue by enhancing exploration capabilities for low-probability tokens.

<details> 
    <summary><em><strong>Shannon Entropy:</strong></em></summary>

### Supplement: Concept of Entropy in Reinforcement Learning Algorithms

In the context of this paper, entropy is a fundamental metric measuring the level of uncertainty or randomness in an agent’s policy. Mathematically, the entropy of policy $\pi$ is defined as:

$$H(\pi(\cdot|s)) = -\sum_{a} \pi(a|s) \log \pi(a|s)$$

where:
- $\pi(a|s)$ — probability of selecting action $a$ in state $s$ according to the current policy;
- Summation runs over all possible next tokens $a$ from the vocabulary, with $\pi(a|s)$ computed via the language model as described above;
- Thus, policy entropy is a scalar value summarizing the characteristics of the probability distribution $\pi(a|s)$, but is not itself a distribution.

The presented formula describes Shannon entropy for the action probability distribution defined by policy $\pi$ in a specific state $s$. It measures the degree of uncertainty or "randomness" in the agent’s action selection in that state. Higher entropy means the agent’s choice is more unpredictable and explores more; lower entropy means the choice is more predictable (deterministic) and exploits known good actions.

The formula computes the expected amount of information we gain by observing the agent’s action selection in state $s$.

### Role of Entropy in Reinforcement Learning

Entropy serves several critical functions in RL algorithms:

1. **Exploration-Exploitation Balance**: High entropy implies a more uniform distribution of action probabilities, promoting exploration of the action space. Low entropy indicates concentration of probability on a small subset of actions, corresponding to exploitation of known strategies.

2. **Prevention of Premature Convergence**: Maintaining sufficient entropy helps avoid getting stuck in local optima, allowing the agent to continue exploring potentially better strategies.

3. **Diversity of Generated Responses**: In generative models such as language models trained with RL, high entropy ensures diversity in generated responses.

### Problem of Entropy Collapse

When policy entropy rapidly declines during training (entropy collapse), it leads to the following negative consequences:

1. **Policy Determinism**: The agent begins selecting the same actions with very high probability, effectively turning a stochastic policy into a deterministic one.

2. **Reduced Exploration Space**: Tokens with initially low probability are virtually excluded from consideration, limiting diversity of generated sequences.

3. **Loss of Adaptability**: The agent loses the ability to adapt to changing conditions, as new strategies receive insufficient representation in training.

</details>

### Problem of Entropy Collapse

During initial experiments using standard implementations of PPO and GRPO, it was observed that policy entropy rapidly decreases during training, as shown in Figure 2b. Sampled outputs for certain groups often become nearly identical, indicating limited exploration of the action space and early policy determinism, potentially hindering expansion.

The root cause lies in the clipping mechanism for the importance sampling ratio, introduced in PPO-Clip to constrain the trust region and improve RL training stability. While this mechanism ensures training stability, it can also restrict policy exploration, particularly for low-probability tokens.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_3.png  )

### Asymmetry in Importance Ratio Clipping

The standard clipping mechanism in PPO uses a single parameter ε (typically set to 0.2) to constrain probability changes equally in both directions. However, this creates asymmetry in the capacity for probability adjustment across different tokens.

Consider an example with two actions having initial probabilities $\pi_{\text{data}}(o_i | q) = 0.01$ and $0.9$ respectively. With standard clipping using ε = 0.2, the maximum possible updated probabilities become $\pi(o_i | q) = 0.012$ and $1.08$ respectively. This means that for tokens with high initial probability (e.g., 0.9), there is less constraint on their probability growth, while for tokens with low initial probability (e.g., 0.01), the potential for significant probability increase is severely limited.

Empirical observations further confirm that the maximum clipped probability for a token typically remains below $\pi(o_i | q) < 0.2$, as shown in Figure 3a. This validates the theoretical analysis indicating that the upper clipping threshold restricts the growth of probabilities for low-probability tokens, potentially limiting system diversity.

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_4.png  )

### Clip-Higher Strategy

To address the above issue, the Clip-Higher strategy is proposed, which decouples the lower and upper clipping ranges into $\epsilon_{\text{low}}$ and $\epsilon_{\text{high}}$ respectively. Mathematically, this is expressed by the following formula:

$$\left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1 - \textcolor{red}{\epsilon_{\text{low}}}, 1 + \textcolor{red}{\epsilon_{\text{high}}}\right)\hat{A}_{i,t}\right) \right], \quad (10)$$

where $r_{i,t}(\theta)$ represents the ratio of the new policy's probability to the base policy's probability, and $\hat{A}_{i,t}$ is the advantage estimate.

Unlike the standard PPO approach where $\epsilon_{\text{low}} = \epsilon_{\text{high}} = 0.2$, the Clip-Higher strategy uses distinct values: $\epsilon_{\text{low}}$ remains at 0.2, while $\epsilon_{\text{high}}$ is increased to 0.28. This elevation of the upper clipping threshold provides greater room for the probability growth of low-initial-probability tokens, thereby encouraging exploration of "long-tail" tokens.

### Experimental Results

As shown in the plots above, the proposed adjustment to the clipping mechanism effectively improves policy entropy and promotes more diverse sampling. The researchers deliberately retained $\epsilon_{\text{low}}$ at a relatively low value (0.2), as increasing this parameter may lead to excessive suppression of certain token probabilities, ultimately causing sample space collapse.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Third Checkpoint: Clip-Higher</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">The Clip-Higher strategy combats entropy collapse in PPO/GRPO by introducing asymmetric clipping thresholds ($\epsilon_{\text{low}} < \epsilon_{\text{high}}$). Increasing the upper threshold ($\epsilon_{\text{high}}$) encourages exploration of low-probability tokens, enhancing the diversity of generated responses.</p>
</div>

## **📌 3.2 Dynamic Sampling: Enhancing Gradient Learning Efficiency**

### Problem of Gradient Vanishing

Existing reinforcement learning (RL) algorithms often suffer from gradient vanishing, which occurs when the accuracy of certain prompts reaches 1. For instance, in the GRPO algorithm, if all outputs for a specific prompt are correct and receive the same reward of 1, the resulting group advantage becomes zero. This leads to policy updates with no gradients, significantly reducing sampling efficiency.

Empirical observations (Figure 3.b — the graph above) show that the number of samples with accuracy exactly 1 continues to increase during training. Consequently, the number of valid signals per batch decreases, resulting in:

- Increased gradient variance;
- Weakened gradient signal for model learning.

### Solution: Dynamic Sampling Method

To address this issue, the dynamic sampling method is proposed. The core idea is as follows:

1. Perform oversampling of prompts;
2. Filter out prompts with accuracy exactly 0 or 1;
3. Retain only prompts with meaningful gradients in the batch;
4. Maintain a constant number of prompts in the training batch.

Sampling continues until the batch is fully populated with examples whose accuracy strictly lies within the range (0,1).

### Mathematical Formulation

Again, our DAPO objective function, focusing on the red-highlighted constraint:

$$J_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta,\text{old}}(·|q)}$$

$$\left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}\right)\hat{A}_{i,t}\right) \right], \quad (11)$$

with constraint: $$\text{s.t.}\textcolor{red}{\ 0 < |\{o_i | \text{is\_equivalent}(a, o_i)\}| < G},$$

This constraint ensures that only prompts with accuracy strictly between 0 and 1 are included in the batch. With dynamic sampling, the experiment can achieve the same performance faster, as shown in Figure 6.

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_5.png  )

The dynamic sampling method provides an effective solution to the gradient vanishing problem in RL algorithms. By selectively filtering prompts with extreme accuracy values (0 or 1) and concentrating computational resources on prompts with intermediate accuracy, this method significantly improves learning efficiency and accelerates model convergence.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fourth Checkpoint: Dynamic Sampling</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">Dynamic sampling addresses gradient vanishing in RL algorithms by excluding prompts with accuracy 0 or 1. Filtering for examples with intermediate accuracy (0 < acc < 1) preserves meaningful gradients in the batch, reduces their variance, and strengthens the learning signal. Maintaining a constant batch size with "useful" examples accelerates model convergence.</p>
</div>

## **📌 3.3 Token-Level Policy Gradient Loss: Rebalancing by Token**

The original GRPO algorithm employs sample-level loss computation, which first averages losses over tokens within each sample and then aggregates losses across all samples. In this approach, each sample is assigned equal weight in the final loss calculation. However, the authors discovered that this loss reduction method creates several problems in long-chain-of-thought RL scenarios.

<details> 
    <summary><em><strong>How Original GRPO Uses Sample-Level Loss Computation:</strong></em></summary>

### **How Original GRPO Uses Sample-Level Loss Computation**

#### 1. **Core Principle of Loss Computation**

In GRPO, losses are computed at the **sample** level — that is, for each generated output $o_i$ in group $G$, then averaged across all samples. This approach can be broken into several steps:

#### Step 1: Compute Loss for Individual Tokens

For each output $o_i$ in group $G$ and each token $t$ in the output, compute:

$$L_{i,t} = \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right)$$

where:
- $r_{i,t}(\theta) = \frac{\pi_\theta(t|q, o_{i,<t})}{\pi_{\text{old}}(t|q, o_{i,<t})}$ — ratio of probabilities between current and old policy
- $\hat{A}_{i,t}$ — advantage estimate for token $t$ in output $o_i$
- $\epsilon$ — clipping parameter to constrain policy updates

#### Step 2: Average Losses Over Tokens per Output

Losses for each output $o_i$ are averaged over all tokens:

$$L_i = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} L_{i,t}$$

where $|o_i|$ is the length of output $o_i$ in tokens.

#### Step 3: Aggregate Losses Across All Samples

Final loss function:

$$L_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G L_i = \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} L_{i,t} \right]$$

#### 2. **Why Is This Approach Used?**
- **Simplified computation**: Averaging losses over tokens within one output allows ignoring differences in output lengths during aggregation. This simplifies algorithm implementation.
- **Uniform sample influence**: Each output $o_i$ contributes equally to total loss, regardless of its length. This ensures balanced weighting across samples.

#### 3. **Problems with This Approach**
However, this method has limitations, especially in long-chain-of-thought tasks:

- **Disproportionate contribution of long outputs**:  
  In long outputs, each token has a smaller impact on total loss due to averaging over many tokens. This may cause the model to underlearn critical reasoning patterns in long sequences.

- **Low-quality long outputs**:  
  Long outputs often contain "noise" (e.g., repetitive or meaningless fragments). Since such outputs are averaged at the sample level, the model may inefficiently suppress these undesirable patterns.

- **Learning distortion**:  
  Uniform sample weighting may cause the model to favor shorter outputs, as they are easier to optimize.

---

### **How It Works in Practice**
Consider an example:

1. **Generate group of outputs**:  
   For prompt $P$, generate group of $G = 4$ outputs:
   - $o_1$: "Short answer" (3 tokens).
   - $o_2$: "Longer answer with details" (10 tokens).
   - $o_3$: "Even longer answer with repetitions" (20 tokens).
   - $o_4$: "Longest answer with noise" (30 tokens).

2. **Compute loss per output**:  
   - For $o_1$: Loss averaged over 3 tokens.
   - For $o_2$: Loss averaged over 10 tokens.
   - For $o_3$: Loss averaged over 20 tokens.
   - For $o_4$: Loss averaged over 30 tokens.

3. **Aggregate losses**:  
   All outputs contribute equally to total loss, despite vastly different lengths.

---

### **Conclusion**
Sample-level loss computation in GRPO ensures simplicity and uniformity in training but has limitations in handling long sequences. These limitations motivated the authors to introduce an alternative approach — **token-level loss computation** — which eliminates the above issues by allowing each token to contribute proportionally to the gradient update.

</details>

---

Since all samples are assigned equal weight in loss computation, tokens in longer outputs (containing more tokens) may contribute disproportionately less to total loss, leading to two undesirable effects. First, for high-quality long samples, this effect may prevent the model from learning reasoning-related patterns. Second, excessively long samples often exhibit poor patterns, such as meaningless babbling and repeated words. Thus, sample-level loss computation cannot effectively eliminate these bad patterns in long samples, leading to unhealthy increases in response entropy and length, as shown in Figures 4a and 4b.

![Figure_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_6.png  )

The authors introduce token-level policy gradient loss in long-chain-of-thought RL scenarios to address the above limitations:

$$J_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta,\text{old}}(·|q)}$$

$$\left[ \frac{1}{\textcolor{red}{\sum_{i=1}^G |o_i|}} \textcolor{red}{\sum_{i=1}^G \sum_{t=1}^{|o_i|}} \min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}\right)\hat{A}_{i,t}\right) \right], \quad (12)$$

with constraint: $$\text{s.t.}\ 0 < |\{o_i | \text{is\_equivalent}(a, o_i)\}| < G,$$

Key differences:
1. Normalization is performed over the total number of tokens across all responses: $\sum_{i=1}^G |o_i|$
2. Tokens are aggregated directly, without prior averaging per sample

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fifth Checkpoint: Token-level Policy Gradient Loss</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">Standard GRPO sample-level loss weakens gradients from tokens in long CoT responses, impairing learning and error suppression. The proposed token-level loss resolves this by direct aggregation and normalization over all batch tokens, ensuring proportional contribution of each token to the gradient.</p>
</div>

## **📌 3.4 Overlong Reward Shaping: Super-Long Reward Formulation**

In reinforcement learning, a fixed maximum generation length is typically enforced, and excessively long samples are truncated. The authors discovered that improper reward formulation for truncated samples can introduce reward noise and severely disrupt the learning process.

By default, a penalty reward is assigned to truncated samples. This approach can introduce noise into learning, as a reasonable reasoning process may be penalized merely for being too long. This penalty may confuse the model about the effectiveness of its reasoning process.

To study the impact of this reward noise, researchers first applied a very long filtering strategy to mask the loss of truncated samples. It was found that this approach significantly stabilized learning and improved results, as shown in Figure 5.

![Figure_7](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_7.png  )

Additionally, the researchers proposed a soft length penalty (Formula 13) — a length-aware penalty mechanism for truncated samples. Specifically, a penalty interval is defined when the answer length exceeds a predetermined maximum value. Within this interval, the longer the answer, the greater the penalty. This penalty is added to the initial rule-based correctness reward, signaling the model to avoid excessively long responses.

![Figure_8](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_8.png  )

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Sixth Checkpoint: Overlong Reward Shaping</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Traditional binary penalty for exceeding response length introduces noise into learning by penalizing even partially correct long solutions. The proposed Overlong Reward Shaping replaces the hard penalty with a gradual linear function over the 16–20K token range, reducing noise and enabling efficient learning on long sequences without abrupt data rejection.</p>
</div>

## 4 Experiments

### 4.1 Training Details

The researchers focused on mathematical tasks to evaluate the developed algorithm, which can be easily adapted to other tasks with clear and precise reward signals. The verl framework was used for training, with GRPO as the baseline algorithm. Advantage was estimated using group reward normalization.

The following hyperparameters were applied: AdamW optimizer with constant learning rate $1 \times 10^{-6}$ and linear warmup over 20 steps. Prompt batch size was 512, with 16 responses per prompt. For overlong reward shaping, the expected maximum length was set to 16,384 tokens with an additional soft penalty buffer of 4,906 tokens. Clipping parameters $c_{\text{low}}$ and $c_{\text{high}}$ were set to 0.2 and 0.28 respectively.

### 4.2 Main Results

In experiments on AIME 2024, the DAPO method successfully trained the base model Qwen-32B into a powerful reasoning model, surpassing DeepSeek’s results using R1 on Qwen2.5-32B. Significant performance improvement on AIME 2024 was demonstrated: accuracy increased from nearly 0% to 50% using only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B.

The researchers analyzed the contribution of each training technique in their approach. Improvements demonstrate the effectiveness of these methods in reinforcement learning. Under naive GRPO configuration, training based on the base model Qwen2.5-32B achieved only 30% accuracy.

![Figure_10](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_10.png  )

### 4.3 Training Dynamics

The DAPO training process demonstrated the complexity of RL in large language models. The researchers ensured training stability by monitoring key metrics. Experiments showed that DAPO not only enhances the model’s reasoning ability but also strengthens its exploratory capabilities.

![Figure_11](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_11.png  )

### 4.4 Case Analysis

During reinforcement learning, the DAPO model demonstrated a dynamically evolving reasoning model. As training progressed, the model not only reinforced existing reasoning patterns but gradually formed new behavioral patterns.

![Figure_12](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_12.png  )

## 5. Conclusion

The launch of the DAPO system represents, in my view, a small but meaningful breakthrough in large-scale reinforcement learning of language models. Thanks to the open-source release of algorithms, code, and datasets, the system provides valuable resources for future research.

The four core DAPO technologies — Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping — offer novel solutions for reinforcement learning. The open-source release of DAPO enables the global research community to better understand and apply reinforcement learning methods to large-scale language models.

Finally, let me add some limitations that came to mind:

- In terms of final performance, the 50% accuracy on AIME still lags behind DeepSeek-R1-Distill-Qwen-32B’s 72.6%.
- The method’s effectiveness was tested on only one training set, one test set, and one model; its generalizability is questionable.
- On the other hand, even if DAPO has only moderate generalization, we can treat the four techniques described in this paper as a toolkit from which we can select individual tools for specific scenarios, rather than treating the entire DAPO as a black box. Indeed, of the four techniques, three are designed for reward shaping — to encourage exploration, better handle long responses, and better manage length penalties — while the remaining one improves sampling efficiency. It is clear that there is no dependency among them, and any subset is rational.