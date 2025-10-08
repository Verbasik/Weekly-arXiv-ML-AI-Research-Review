# **DAPO: A RL Algorithm from ByteDance**

## **Abstract**

This paper presents **DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)**, an open platform for training large language models (LLMs) using reinforcement learning (RL) methods. Despite significant advances in modern LLMs such as OpenAI o1 and DeepSeek R1, key technical details of their RL training remain inaccessible to the scientific community, severely hindering reproducibility and further research. In response to this challenge, the authors propose the innovative DAPO algorithm, which not only demonstrates high effectiveness but also provides full openness of code, data, and methodology.

The DAPO system achieved a record score of 50 points on the **AIME 2024** mathematics competition, surpassing the previous record of DeepSeek-R1 (47 points). Remarkably, DAPO achieved this result while halving the number of training steps. The algorithm is built upon four core technologies: **Clip-Higher strategy**, **dynamic sampling**, **token-level policy gradient optimization**, and **intelligent length penalty**. These methods address fundamental challenges in RL training, including entropy collapse, reward noise, and inefficiency in learning from long texts.

The authors emphasize that large-scale reinforcement learning is critical for advancing LLMs' capacity for complex reasoning. However, unlike prior works where RL training details remained hidden (e.g., in OpenAI’s blog on o1 and DeepSeek R1’s technical report), DAPO provides full transparency. Not only are the training codes, built on the **verl** framework, openly released, but also meticulously curated datasets. This promotes reproducibility and opens new avenues for research in large-scale RL training of LLMs.

Thus, this work makes a significant contribution to the development of open and reproducible methods for training large language models, offering both theoretical innovations and practical tools for the scientific community.

## **1. Introduction**

The emergence of models with extended test-time compute and reasoning, such as OpenAI’s O1, DeepSeek’s R1, and recently Anthropic’s Claude, has marked a fundamental paradigm shift in large language models (LLMs) based on reinforcement learning (Reinforcement Learning, RL). These models have demonstrated unprecedented capabilities in complex reasoning, enabling them to successfully solve challenging mathematical and programming problems at the level of competitions like AIME and Codeforces.

The central technology enabling this breakthrough is large-scale reinforcement learning (RL), which fosters the development of sophisticated reasoning forms, including self-checking, iterative refinement, and reflection. Despite impressive results, the specific algorithms and methodological approaches to scalable RL training remain largely concealed in technical reports of existing models. As the authors note, "key technical details of modern reasoning LLMs are hidden (e.g., in OpenAI’s blog on model o1 and DeepSeek R1’s technical report)," making it difficult for the research community to reproduce their results.

During experiments, the authors used Qwen2.5-32B as the pre-trained model for reinforcement learning based on feedback. Initial runs using the baseline GRPO (Generalized Reward-weighted Policy Optimization) algorithm achieved only 30 points on the AIME test, significantly lagging behind DeepSeek-RL’s 47 points. Deeper analysis revealed that the naive implementation of GRPO faces several critical issues:

1. **Entropy collapse** — the tendency of the model to narrow the diversity of generated responses;
2. **Reward noise** — incorrect assignment of rewards for partially correct or excessively long responses;
3. **Training instability** — difficulties in scaling training to long reasoning chains.

Similar problems have been observed in the broader community attempting to reproduce DeepSeek’s results, suggesting that critical training details may be missing from the published R1 paper — details crucial for developing scalable, reproducible industrial-level RL training systems.

To bridge this gap, the authors present a modern, open-source system for large-scale RL training of LLMs that achieves 50 points on AIME 2024 using the Qwen2.5-32B model. This result surpasses the previous record of DeepSeek-RL-Zero-Qwen-32B (47 points), requiring only 50% of the training steps (see Figure 1). The system is based on the authors’ proposed algorithm, Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), which incorporates four key innovations that fundamentally improve the efficiency of RL training in long reasoning chain scenarios:

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_1.png    )

- **Clip-Higher** — a strategy promoting generation diversity by enabling adaptive sampling, separating lower and upper clipping bounds (ε-low and ε-high);
- **Dynamic Sampling** — a dynamic sampling method enhancing training efficiency and stability by excluding examples with zero gradients;
- **Token-Level Policy Gradient Loss** — computing policy gradients at the individual token level, critically important for effective learning on long sequences;
- **Overlong Reward Shaping** — an intelligent length penalty system reducing reward noise and stabilizing the training process.

A key value of this work lies in the full openness of all system components: the entire implementation code, built on the verl framework, along with the carefully curated DAPO-Math-17K dataset, are available in an open repository. This openness contrasts sharply with prior works that, despite impressive results, did not disclose critical training details.

During experiments, the authors also observed an interesting phenomenon: the model not only reinforces existing reasoning patterns but gradually develops fundamentally new capabilities, particularly behaviors related to self-checking and rethinking previous steps. This opens new perspectives for understanding the fundamental mechanisms by which LLMs learn complex reasoning.

We will now step-by-step examine the transition from PPO to GRPO, and then to DAPO, to understand how this new RL algorithm was developed.

## **2. Background**

### **2.1 Proximal Policy Optimization (PPO)**

#### Core Concept

Proximal Policy Optimization (PPO) is one of the most popular and effective reinforcement learning algorithms, developed by researchers at OpenAI in 2017. PPO is an improvement over earlier policy optimization methods such as TRPO (Trust Region Policy Optimization), offering a simpler implementation with comparable or superior performance.

#### Key Features of PPO

1. **Training Stability**: The core idea of PPO is to constrain policy updates between iterations, preventing overly abrupt changes that could destabilize learning;
2. **Clipping Mechanism**: PPO uses a clipping function to bound the importance sampling ratio, ensuring the new policy does not deviate too far from the old one;
3. **Sample Efficiency**: Compared to other algorithms, PPO makes more efficient use of collected data, achieving better results with fewer environmental interactions.

#### Mathematical Formulation

The PPO objective function is defined as follows:

![Image_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_1.png    )

![Image_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_2.png    )

**In summary, "under the hood" without mathematical details:**

* **Policy**: This is like an "instruction" for the program, telling it which action to take in each situation. In PPO, the policy is represented by a neural network.
* **Value Function**: This estimates how "good" a given state is. It helps the program understand which states to aim for. Typically also a neural network.
* **Advantage**: This is the difference between how good the chosen action was and how good actions are on average in that state. It helps identify truly successful actions.
* **PPO Objective Function (Equation 1)**: This is the main "goal" of learning. The program seeks to maximize it to improve. The most important element is the "clipping" mechanism, which constrains policy changes.
* **GAE Advantage Estimation (Equation 2)**: A method to compute how good actions were, considering not only immediate rewards but also future consequences.
* **Temporal Difference (Equation 3)**: Helps the value function learn by comparing predicted state values with actual outcomes.

#### PPO Workflow

1. **Experience Collection**: Agents interact with the environment following the current policy $\pi_{\theta_{\text{old}}}$, collecting trajectories (sequences of states, actions, and rewards).
2. **Advantage Estimation**: For each time step in collected trajectories, the advantage estimate $\hat{A}_t$ is computed using equations (2) and (3).
3. **Policy Optimization**: Policy parameters $\theta$ are updated by optimizing the objective function (1) using stochastic gradient ascent over multiple epochs.
4. **Value Function Update**: Value function parameters are updated by minimizing the mean squared error between predicted and actual state values.
5. **Iteration**: Steps 1–4 are repeated until desired performance or a set number of iterations is achieved.

#### Advantages of PPO

1. **Implementation Simplicity**: PPO is simpler to implement than TRPO and other complex RL algorithms;
2. **High Performance**: PPO achieves excellent results across diverse tasks, from Atari games to complex robotics;
3. **Compatibility with Neural Networks**: PPO works well with deep neural networks and can train complex policies;
4. **Good Scalability**: PPO can be efficiently parallelized to accelerate training.

#### Applications of PPO

PPO is widely used in various domains:

1. **Games**: From simple games to complex strategies and simulators;
2. **Robotics**: Training robots to perform complex physical tasks;
3. **Recommendation Systems**: Optimizing recommendation strategies in interactive systems;
4. **Large Language Models**: In recent years, PPO has been actively used to fine-tune language models via RLHF (Reinforcement Learning from Human Feedback).

#### Conclusion

Proximal Policy Optimization (PPO) is a powerful and flexible reinforcement learning algorithm that, due to its simplicity and effectiveness, has become the de facto standard in deep reinforcement learning. Its applications span from games and robotics to training modern language models, demonstrating the universality and power of this approach.

### **2.2 Group Relative Policy Optimization (GRPO)**

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed to optimize LLMs for tasks requiring structured reasoning, such as mathematics and logic. It was introduced in DeepSeekMath and DeepSeek-R1 **as a response to the challenges of training models with billions of parameters**. GRPO offers a more efficient approach compared to traditional methods like Proximal Policy Optimization (PPO), **by eliminating key bottlenecks associated with advantage function computation**.

#### **GRPO’s Innovative Approach to Advantage Functions**

![Figure_19](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-07_%26_08/assets/Figure_19.jpg    )

GRPO completely eliminates the need for a value network by using **group-relative normalization**:
For each prompt $P$, a group of $N$ responses $G = \{O_1, O_2, ..., O_N\}$ is generated using policy $\pi$. Each response $O_i$ receives a reward $R_i = R(O_i)$ reflecting its quality. The advantage function for the $i$-th response $O_i$ relative to group $G$ is computed as:

![Image_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_3.png    )

> Essentially, the advantage function in GRPO for each specific response is calculated as the reward of that response minus the arithmetic mean of all rewards in the group.

#### **Mathematical Formulation**

**GRPO Objective Function**:

![Image_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_4.png    )

**In summary, "under the hood" of GRPO without mathematical details:**

* **Policy**: Like in PPO, this is a neural network that determines which actions (or tokens for LLMs) to select in each state.
* **Grouped Response Generation**: Unlike PPO, GRPO generates not one but multiple diverse responses (typically 4–8) to the same prompt.
* **Relative Normalization Instead of Value Function**: GRPO does not use a separate value function. Instead, the quality of each response is evaluated relative to other responses in the group — this is the key distinction.
* **Advantage**: Calculated as the difference between a response’s reward and the average reward across the group. This identifies which responses are better than average.
* **KL Divergence**: Strictly integrated into the loss function to constrain the updated policy from deviating too far from the original, ensuring training stability.
* **Resource Efficiency**: Eliminating the value network reduces memory usage by 40–60% and accelerates training by ~35%.

#### GRPO Workflow

1. **Generate Response Group**: For each query, the model generates a group of N distinct responses using the current policy.
2. **Reward Evaluation**: Each response receives a reward from an external reward function (e.g., a critic model or predefined rules).
3. **Compute Relative Advantage**: For each response, advantage is calculated as the difference between its reward and the average reward of the group:
   
$$
A_i = R_i - (R_1 + R_2 + ... + R_N)/N
$$

4. **Policy Optimization**: Model parameters are updated to increase the probability of generating responses with positive advantage (better than average) and decrease the probability of responses with negative advantage (worse than average).
5. **KL Divergence Regularization**: Updates are constrained so the new policy does not deviate too far from the previous version or a reference model.
6. **Iteration**: Steps 1–5 are repeated until desired performance is achieved.

**Key Features of the GRPO Approach:**

*   **Group Relative Normalization**: Advantage is computed relative to a group of responses generated for the same prompt, enabling relative quality assessment;
*   **Elimination of Value Network**: The group’s average reward $\bar{R}_G$ serves as a baseline, replacing the need for a separate value network to estimate state or action values;
*   **Learning by Comparison**: GRPO focuses on training a policy that generates responses superior to the average within the group, making it effective in tasks where relative quality matters;
* **KL Divergence — Hard Integration into Loss via Relative Weights**: KL divergence is incorporated into the loss function to regularize updates, limiting the magnitude of policy change per step and preventing sharp fluctuations, thus enhancing training stability.

**Limitations and Remarks:**

*   GRPO’s effectiveness depends on the quality of the reward function $R(O)$. The reward function must be correctly designed to adequately reflect desired response properties;
*   Group size $N$ is a hyperparameter that can affect training stability and efficiency. Optimal $N$ may require experimental tuning;
*   Like other RL methods, GRPO can be sensitive to optimization hyperparameters and model architecture.

---

#### **Practical Interpretation for LLMs**

In GRPO, the advantage function becomes a **ranking tool for response variants**:
- The model learns to generate responses that are not merely "good," but **significantly better than the group average**.
- This encourages:
  - Discovery of non-obvious but effective reasoning chains;
  - Avoidance of patterned errors common within the group.

**Effect**: The model focuses on **qualitative differences between responses**, not absolute reward values — critical for complex tasks with ambiguous success criteria.

**Problem Context**:
- In reasoning tasks, LLMs often generate multiple "chain-of-thought" responses, but standard RL algorithms are poorly adapted to evaluate them.
- **Value networks in PPO require substantial resources to train and are prone to errors in multimodal reward distributions**.

---

#### **Key Differences Between GRPO and PPO**

| **Characteristic**                   | **PPO**                               | **GRPO**                                                                 |
|-------------------------------------|---------------------------------------|---------------------------------------------------------------------------|
| Presence of Value Network           | Required                              | Eliminated                                                                |
| Advantage Estimation                | Based on value network                | **Group-relative normalization within trajectories**                      |
| KL Divergence                       | Optional regularization               | **Hard-integrated into loss function via relative weights**               |
| Memory Usage                        | High (two models)                     | **Reduced by 40–60% due to elimination of value network**                 |
| Convergence                         | Depends on value network accuracy     | **More stable due to group-based gradient stabilization**                 |

---

### **2.3 Elimination of KL Divergence**

**Eliminating KL Divergence in DAPO for Training Models with Long Reasoning Chains**

In RLHF (Reinforcement Learning from Human Feedback) scenarios, a KL divergence penalty is traditionally used to regulate deviations between the updated online policy and a frozen reference policy. Its primary goal is to ensure that during training, the model adjusts its behavior without straying too far from the original data distribution — especially important for preserving predictability and stability.

However, when training models that generate long chain-of-thought (CoT) responses, this constraint becomes irrelevant. In such tasks, the model’s distribution naturally and significantly deviates from the initial one due to the complexity and multi-step nature of reasoning. Rigid regulation via KL divergence in this context becomes redundant, artificially limiting the model’s ability to explore alternative generation strategies necessary for effectively solving multi-stage tasks.

The DAPO (Decoupled Adaptive Policy Optimization) algorithm proposes eliminating the KL penalty to relax this constraint. Removing the KL term allows the model to adapt freely during training without being bound to the initial reference policy distribution. This is especially important for scenarios where successful task completion requires moving beyond template solutions — for example, generating complex logical conclusions or creative texts. Thus, DAPO focuses on balancing exploration of new strategies with efficient policy optimization, enhancing model flexibility in long-reasoning contexts without compromising generation quality.

This approach highlights that in certain RLHF scenarios, strict control over deviation from the initial policy can be avoided to fully unlock the model’s adaptive potential under complex and ambiguous tasks.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">First Checkpoint:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">DAPO eliminates KL penalties in RLHF for long-CoT tasks, enabling greater policy flexibility and enhanced reasoning capabilities.</p>
</div>

### **2.4 Rule-Based Reward Modeling**

Traditional reward models often suffer from reward hacking, where the model manipulates the reward signal to achieve high scores rather than genuinely improving reasoning ability. DAPO directly uses the final accuracy of the verified task as the reward, avoiding the complexity of reward modeling. Specifically, the reward function is defined as:

![Image_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_5.png    )

This approach has proven effective across various domains, including automated theorem proving, computer programming, and mathematical competitions.

> IMHO: In this case, reward modeling works only for deterministic tasks where the answer is unambiguous. For tasks with uncertain answers (e.g., LLM responses based on heuristics rather than strict proofs), this would not work.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Second Checkpoint:</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">DAPO replaces complex reward models with direct use of task final accuracy, eliminating reward hacking and simplifying training.</p>
</div>

## **3. The DAPO Algorithm**

The researchers proposed the Decoupled Clip and Dynamic Sampling Strategy Optimization (DAPO) algorithm. DAPO samples a group of outputs ${o_i}_{i=1}^G$ for each question $q$ associated with answer $a$, and optimizes the policy through the following objective function:

![Image_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_6.png    )

Below, we analyze the key technologies underlying DAPO.

### **📌 3.1 Clip-Higher: Increasing the Upper Bound**

In reinforcement learning (RL) algorithms such as Proximal Policy Optimization (PPO) and Generalized Proximal Policy Optimization (GRPO), a common phenomenon is entropy collapse — where policy entropy rapidly decreases during training. This leads to generated responses becoming nearly identical, indicating limited exploration of the action space and premature deterministic policy convergence. This paper proposes the Clip-Higher strategy — a modification of the standard clipping mechanism in PPO — designed to address this issue by enhancing exploration capabilities for low-probability tokens.

#### The Problem of Entropy Collapse

During initial experiments using standard implementations of PPO and GRPO, it was observed that policy entropy rapidly declines during training, as shown in Figure 2b. Sampled responses for some groups often turn out nearly identical, indicating limited exploration of the action space and early policy determinism, potentially hindering expansion.

The root of this problem lies in the importance sampling ratio clipping mechanism introduced in PPO-Clip to constrain the trust region and enhance RL training stability. While this mechanism ensures training stability, it can also restrict policy exploration, particularly for low-probability tokens.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_3.png    )

#### Asymmetry in Importance Ratio Constraints

The standard clipping mechanism in PPO uses a single parameter ε (typically set to 0.2) to constrain probability changes in both directions. However, this creates asymmetry in the allowable probability changes for different tokens.

Consider two actions with initial probabilities $\pi_{\text{data}}(o_i | q) = 0.01$ and $0.9$. With standard clipping at ε = 0.2, the maximum possible updated probabilities become $\pi(o_i | q) = 0.012$ and $1.08$, respectively. This means that for tokens with high initial probability (e.g., 0.9), there are fewer constraints on increasing their probability, whereas for tokens with low initial probability (e.g., 0.01), the potential for significant probability increases is severely restricted.

Empirical observations also confirm that the maximum clipped token probability typically remains below $\pi(o_i | q) < 0.2$, as shown in Figure 3a. This confirms theoretical analysis indicating that the upper clipping threshold restricts the growth of probabilities for low-initial-probability tokens, potentially limiting system diversity.

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_4.png    )

#### The Clip-Higher Strategy

To address the above issue, the Clip-Higher strategy is proposed, based on separating the lower and upper clipping ranges into ε_low and ε_high, respectively. Mathematically, this is expressed as:

![Image_7](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_7.png    )

where $r_{i,t}(θ)$ represents the ratio of the new policy’s probability to the base policy’s probability, and $Â_{i,t}$ is the advantage estimate.

Unlike the standard PPO approach, where ε_low = ε_high = 0.2, the Clip-Higher strategy uses different values: ε_low remains at 0.2, while ε_high is increased to 0.28. This higher upper clipping threshold provides more room for increasing the probability of low-initial-probability tokens, thereby encouraging exploration of "long-tail tokens."

#### Experimental Results

As shown in the graphs above, the proposed adjustment to the clipping mechanism effectively improves policy entropy and promotes the generation of more diverse samples. The researchers deliberately chose to keep ε_low relatively small (0.2), as increasing this parameter may lead to excessive reduction in the probability of certain tokens, ultimately causing collapse of the sampling space.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Third Checkpoint: Clip-Higher</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">The Clip-Higher strategy combats entropy collapse in PPO/GRPO by introducing asymmetric clipping thresholds (ε_low < ε_high). Increasing the upper threshold (ε_high) encourages exploration of low-probability tokens, enhancing the diversity of generated responses.</p>
</div>

### **📌 3.2 Dynamic Sampling: Enhancing Gradient Learning Efficiency**

#### The Problem of Gradient Vanishing

Existing reinforcement learning (RL) algorithms often suffer from gradient vanishing, which occurs when the accuracy of some prompts reaches 1. For instance, in the GRPO algorithm, if all outputs for a specific prompt are correct and receive the same reward of 1, the resulting group advantage becomes zero. This leads to policy updates with no gradients, significantly reducing sampling efficiency.

Empirical observations (Figure 3.b — the graph slightly above) show that the number of samples with accuracy equal to 1 continues to increase during training. As a result, the number of meaningful signals in each batch decreases, leading to:

- Increased gradient variance;
- Weakened gradient signal for model training.

#### Solution: Dynamic Sampling Method

To address this issue, the dynamic sampling method is proposed. The core idea is as follows:

1. Perform oversampling of prompts;
2. Filter out prompts with accuracy equal to 1 or 0;
3. Retain only prompts with valid gradients in the batch;
4. Maintain a constant number of prompts in the training batch.

Sampling continues until the batch is fully populated with examples whose accuracy strictly falls within the range (0,1).

#### Mathematical Formulation

Again, our DAPO objective function — here we focus on the constraint highlighted in red:

![Image_8](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_8.png  )

This constraint ensures that the batch contains only prompts with accuracy between 0 and 1. With dynamic sampling, the experiment can achieve the same performance faster. The observation is shown in Figure 6.

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_5.png  )

The dynamic sampling method provides an effective solution to the gradient vanishing problem in RL algorithms. By selectively filtering prompts with extreme accuracy values (0 or 1) and focusing computational resources on prompts with intermediate accuracy, this method significantly enhances training efficiency and accelerates model convergence.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fourth Checkpoint: Dynamic Sampling</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">Dynamic sampling resolves gradient vanishing in RL algorithms by excluding prompts with accuracy 0 or 1. Filtering examples with intermediate accuracy (0 < acc < 1) preserves meaningful gradients in the batch, reduces their variance, and strengthens the learning signal. Maintaining a constant batch size with "useful" examples accelerates model convergence.</p>
</div>

### **📌 3.3 Token-Level Policy Gradient Loss: Rebalancing Action**

The original GRPO algorithm uses sample-level loss calculation, which first averages losses over tokens within each sample and then aggregates losses across all samples. In this approach, each sample is assigned equal weight in the final loss computation. However, the authors found that this loss-reduction approach creates several problems in long-chain-of-thought RL scenarios.

Since all samples are weighted equally during loss calculation, tokens in longer responses (containing more tokens) contribute disproportionately less to the overall loss, leading to two undesirable effects. First, for high-quality long samples, this effect may hinder the model from learning reasoning-related patterns. Second, excessively long samples often exhibit poor patterns, such as meaningless gibberish and repeated words. Thus, sample-level loss calculation cannot effectively eliminate these poor patterns in long samples, resulting in unhealthy increases in response entropy and length, as shown in Figures 4a and 4b.

![Figure_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_6.png  )

The authors introduce token-level policy gradient loss into the long-chain-of-thought RL scenario to address the above limitations:

![Image_9](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Image_9.png  )

Key differences:
1. Normalization is performed over the total number of tokens across all responses: $\sum_{i=1}^G |o_i|$;
2. Tokens are aggregated directly, without prior averaging per sample.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
  <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fifth Checkpoint: Token-level Policy Gradient Loss</p>
  <p style="margin: 8px 0 0 0; color: #2c3e50;">Standard GRPO sample-level loss calculation weakens gradients from tokens in long CoT responses, degrading learning and error suppression. The proposed token-level loss resolves this by directly aggregating and normalizing over all batch tokens, ensuring each token contributes proportionally to the gradient.</p>
</div>

### **📌 3.4 Overlong Reward Shaping: Super-Long Reward Formulation**
 
In reinforcement learning, a fixed maximum generation length is typically set, and very long samples are truncated accordingly. The authors found that improper reward shaping for truncated samples can introduce reward noise and severely disrupt the training process.

By default, a penalty reward is assigned to truncated samples. This approach can introduce noise into training, as a rational reasoning process may be penalized merely because it is too long. This penalty can confuse the model about the effectiveness of its reasoning process.

To study the impact of this reward noise, researchers first applied a very long filtering strategy to mask the loss of truncated samples. It was found that this approach significantly stabilized training and improved results, as shown in Figure 5.

![Figure_7](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_7.png  )

Additionally, the researchers proposed a soft length-overrun penalty (Formula 13) — a length-aware penalty mechanism for truncated samples. Specifically, a penalty interval is defined when the response length exceeds a predetermined maximum value. Within this interval, the longer the response, the greater the penalty. This penalty is added to the initial rule-based correctness reward, signaling the model to avoid excessively long responses.

![Figure_8](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_8.png  )

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Sixth Checkpoint: Overlong Reward Shaping</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Traditional binary penalty for response length excess introduces noise into training by penalizing even partially correct long solutions. The proposed Overlong Reward Shaping replaces the hard penalty with a gradual linear function over the 16–20K token range, reducing noise and enabling the model to effectively learn from long sequences without abrupt discarding of data.</p>
</div>

## **4 Experiments**

### **4.1 Training Details**

The researchers focused on mathematical tasks to evaluate the developed algorithm, which can be easily adapted to other tasks with clear and precise reward signals. Training was conducted using the verl framework with GRPO as the base algorithm. Advantage was estimated using group reward normalization.

The following hyperparameters were applied: AdamW optimizer with a constant learning rate of $1 \times 10^{-6}$ and linear warmup over 20 steps. The prompt batch size was 512, with 16 responses sampled per prompt. For overlong reward shaping, the expected maximum length was set to 16,384 tokens, with an additional soft penalty buffer of 4,906 tokens. The clipping parameters $c_{\text{low}}$ and $c_{\text{high}}$ were set to 0.2 and 0.28, respectively.

### **4.2 Main Results**

In experiments on AIME 2024, the DAPO method successfully trained the base Qwen-32B model into a powerful reasoning model, surpassing DeepSeek’s performance using R1 on Qwen2.5-32B. A significant improvement in AIME 2024 performance was demonstrated: accuracy increased from nearly 0% to 50% using only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B.

The researchers analyzed the contribution of each training technique to their approach. The improvements demonstrate the effectiveness of these techniques in reinforcement learning. Under naive GRPO tuning, training based on the Qwen2.5-32B base model achieved only 30% accuracy.

![Figure_10](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_10.png  )

### **4.3 Training Dynamics**

The DAPO training process revealed the complexity of RL in large language models. The researchers ensured training stability by monitoring key metrics. Experiments showed that DAPO not only improves the model’s reasoning capability but also enhances its exploratory capacity.

![Figure_11](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_11.png  )

### **4.4 Case Analysis**

During RL training, the DAPO model demonstrated a dynamically evolving reasoning pattern. As training progressed, the model not only reinforced existing reasoning patterns but also gradually developed new behavioral models.

![Figure_12](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-13/assets/Figure_12.png  )

## **5 Conclusion**

The release of the DAPO system represents, in my view, a small but significant breakthrough in large-scale reinforcement learning of language models. Thanks to the open-source release of algorithms, code, and datasets, the system provides valuable resources for future research.

The four core DAPO technologies — Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping — offer novel solutions for reinforcement learning. The open-source release of DAPO enables the global research community to better understand and apply RL methods to large-scale language models.

Finally, let me add some limitations that came to mind:

- In terms of final performance, the 50% AIME accuracy still lags behind DeepSeek-R1-Distill-Qwen-32B’s 72.6%;
- The method’s effectiveness was tested on only one training set, one test set, and one model; its generalizability is questionable;
- On the other hand, even if DAPO has only moderate generalizability, we can view the four techniques described in this paper as a toolkit from which we can select individual tools for specific scenarios, rather than treating the entire DAPO as a black box. Indeed, of the four techniques, three are designed to shape reward for encouraging exploration, improving long-response handling, and better managing length penalties, while the remaining one enhances sampling efficiency. It is clear that there is no dependency among them, and any subset is rational.