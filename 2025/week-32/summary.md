# Group Sequence Policy Optimization

## Contents
1. Introduction
2. Problems with Existing Methods
3. GSPO Methodology
4. Key Algorithmic Differences
5. Experimental Results
6. Practical Applications and Infrastructure Advantages
7. Significance and Future Implications

## 1. Introduction

Reinforcement learning (RL) has become a critical tool for scaling large language models (LLMs) to solve complex reasoning tasks in mathematics and programming. However, applying RL to models with billions of parameters faces a critical challenge: training instability, which can lead to catastrophic model collapse. This paper introduces Group Sequence Policy Optimization (GSPO), a new RL algorithm designed to eliminate the fundamental shortcomings of existing methods such as Group Relative Policy Optimization (GRPO).

The core innovation of GSPO lies in shifting importance sampling from the token level to the sequence level, aligning the unit of optimization with how rewards are actually assigned. This seemingly simple change resolves severe stability issues that have plagued large-scale RL training, particularly for Mixture-of-Experts (MoE) models. The work demonstrates that GSPO not only achieves exceptional stability but also improves training efficiency and performance, contributing to the remarkable improvements observed in Alibaba's latest Qwen3 models.

!["Comparison of training performance showing GSPO's superior stability and performance over GRPO across multiple benchmarks including training reward, AIME'24, LiveCodeBench, and CodeForces"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-01.jpeg)

## 2. Problems with Existing Methods

Modern state-of-the-art RL algorithms for LLMs suffer from critical stability issues. Proximal Policy Optimization (PPO), while widely used, requires a value model of comparable size to the policy model, creating significant memory overhead. Group Relative Policy Optimization (GRPO) solved this problem by eliminating the dependency on a value model, but introduced a more fundamental issue.

The authors identify that GRPO's instability stems from incorrect application of importance sampling. GRPO applies importance weights at the token level, while rewards are assigned at the sequence level. This creates a misalignment: individual tokens within a sequence receive different importance weights, despite the reward signal being applied to the entire response.

**Let’s briefly recall how GRPO works:**

!["Group Relative Policy Optimization (GRPO)"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-02.jpg)

The most common RL algorithm for language models is Proximal Policy Optimization (PPO), and GRPO is a variant of it. The essence is:

➖ An agent has an initial policy (strategy) it follows.

➖ The agent takes actions in the environment (answers questions) according to its current policy.

➖ PPO evaluates the agent’s action using three models:
  - **reference model** — serves as a baseline to measure how much the current policy has deviated from the original,
  - **reward model** — evaluates the immediate reward the agent receives for performing the action,
  - **value model** — estimates the expected long-term benefit of the action by predicting future rewards.

➖ Based on these evaluations, the agent updates its policy. The key feature of PPO is that its loss function prevents overly abrupt policy changes. This enables the agent to gradually improve its strategy without making drastic, destabilizing steps, resulting in a more stable and efficient training process.

However, PPO has drawbacks. In particular, the value model — central to PPO — incurs massive computational costs, typically matching the size of the model being trained. This makes training expensive. GRPO eliminates the value model entirely. Instead, it uses the average reward from a group of responses to the same prompt to determine how "good" the model's actions are. In GRPO, response quality is assessed relative to other responses in the group, not by absolute reward values. If a response is better than the group average, the policy increases its likelihood; if worse, it decreases it. This compensates for the absence of a value model, making training more efficient and less resource-intensive.

Notably, GRPO works well even when skipping supervised fine-tuning. This is how R1-Zero, the younger sibling of R1, was trained: no labeled data was used at all, and GRPO alone extracted all its quality.

Mathematically, GRPO uses token-level importance weights:

!["GRPO"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-03.png)

This token-level weighting introduces high-variance noise that accumulates over long sequences. Variance is further amplified by clipping mechanisms, ultimately leading to the observed training instability and model collapse, particularly in large-scale deployments.

## 3. GSPO Methodology

GSPO resolves these issues by fundamentally restructuring importance sampling to align with the structure of rewards. The key idea is to define importance weights at the sequence level, where rewards are actually assigned.

The GSPO objective function:

!["GSPO"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-04.png)

The sequence-level importance weight is defined as:

!["sequence-level"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-05.png)

The critical element is the length-normalization term $1/|y_i|$ in the exponent. Without such normalization, sequence-level likelihood ratios could vary dramatically across responses of different lengths, requiring variable clipping ranges. Thanks to normalization, the value of $s_i(\theta)$ remains within a constant numerical range regardless of sequence length.

For advantage estimation $A_i$, the same group-based approach without a value model as in GRPO is employed.

!["advantage estimation"](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-06.png)

This preserves GRPO’s computational efficiency while resolving its fundamental stability problems.

## 4. Key Algorithmic Differences

The primary distinction between GSPO and GRPO lies in gradient weighting. GRPO applies volatile token-level importance weights to individual token gradients, creating high variance. GSPO applies a single, stable sequence-level importance weight uniformly to all tokens in the response, eliminating this problematic variance.

In gradient terms, GSPO weights all tokens in a sequence equally after applying the sequence-level importance coefficient, while GRPO creates uneven weighting between tokens within the same sequence. This alignment with the reward structure (at the sequence level) creates a more stable and theoretically sound training dynamics.

The authors also introduce GSPO-token, a variant that allows per-token advantage tuning for scenarios such as multi-step reinforcement learning, while preserving the core principle of sequence-level importance weighting via stop-gradient operations.

## 5. Experimental Results

Empirical evaluation demonstrates significant advantages of GSPO across multiple metrics:

**Training Stability:** GSPO maintains stable training throughout the process, while GRPO exhibits instability and potential collapse. This stability enables continuous performance improvement with increased training compute, regular query set updates, and extended generation lengths.

**Performance and Efficiency:** At identical computational budgets, GSPO consistently achieves superior training accuracy and performance on challenging benchmarks, including AIME'24, LiveCodeBench, and CodeForces. This demonstrates that GSPO is not only more stable but also more sample-efficient.

**Mixture-of-Experts Training:** Particularly significant is GSPO’s ability to resolve stability issues in RL training of large MoE models. GRPO required complex workarounds such as "Routing Replay" to handle expert activation volatility. The sequence-level approach of GSPO is inherently insensitive to this volatility, enabling stable MoE training without additional overhead or computational constraints.

!['Comparison of clipping rates showing GSPO clips significantly more tokens than GRPO while achieving superior performance'](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-07.jpeg)

**Counterintuitive Clipping Behavior:** Notably, GSPO clips approximately two orders of magnitude more tokens than GRPO, yet achieves superior training efficiency. This confirms the hypothesis that GRPO’s token-level gradient estimates are inherently noisy and inefficient. GSPO’s sequence-level approach provides higher-quality learning signals even when discarding more data.

!['Performance of GRPO with and without Routing Replay, demonstrating the necessity of this complex workaround for MoE stability'](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-32/assets/Image-08.jpeg)

## 6. Practical Applications and Infrastructure Advantages

Beyond performance improvements, GSPO offers practical infrastructure advantages for RL. Its dependence on sequence-level probabilities makes it more tolerant to precision discrepancies between training and inference engines. This allows direct use of probabilities from inference engines, avoiding costly recomputations and simplifying RL pipelines.

Specifically for MoE models, GSPO eliminates the need for "Routing Replay" and similar stabilization strategies, reducing memory and communication overhead and enabling models to fully leverage their architectural capacity without artificial constraints.

## 7. Significance and Future Implications

GSPO represents an evolution of the central RL algorithm for LLM giants, as it removes the core theoretical and practical limitations of existing methods. Its contribution to the "notable improvements in the latest Qwen3 models" demonstrates real-world impact beyond theoretical advances.

The algorithm’s ability to enable stable, large-scale reinforcement learning opens new possibilities for further scaling capabilities through RL. By resolving the stability bottleneck, GSPO allows researchers to invest more computational resources into reinforcement learning with confidence in convergence.

For the broader field, GSPO provides a blueprint for designing algorithms that carefully align optimization mechanisms with reward structures. This principle may underpin future RL algorithm developments, especially as models continue to scale and grow in complexity.

This work positions the field for exploring more sophisticated applications of RL for large language models, potentially enabling breakthroughs in complex reasoning, multi-step problem solving, and other capabilities that benefit from deeper exploration enabled by stable, large-scale reinforcement learning.