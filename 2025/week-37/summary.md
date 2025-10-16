# Deep Think with Confidence

## Background and Motivation

Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, but achieving high accuracy often requires generating hundreds or thousands of reasoning chains using methods such as self-consistency with majority voting. Although this "parallel thinking" approach is effective, it suffers from substantial computational costs and diminishing returns—sometimes requiring 100 million additional tokens for a modest 14% accuracy improvement on challenging tasks like AIME 2025.

![Figure 1: Accuracy comparison on AIME 2025, showing DeepConf's superior performance across various model sizes, with some achieving nearly perfect accuracy (99.9% for GPT-OSS-120B).](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-01.png)

The core issue lies in the fact that all reasoning chains are treated equally during majority voting, despite LLMs naturally producing chains of varying quality. Previous approaches attempted to use global confidence measures computed after full chain generation, but these methods cannot capture local fluctuations in reasoning or enable early termination of low-quality paths.

## Core Methodology

Deep Think with Confidence (DeepConf) addresses these limitations through a sophisticated confidence measurement system operating at multiple levels of granularity:

**Local confidence metrics**: Instead of relying on global averages, DeepConf introduces several targeted confidence measures:

- **Group confidence**: Computed over sliding token windows (typically 1024–2048 tokens) to smooth individual token fluctuations.
- **Bottom-10% group confidence**: Focuses on the most problematic segments within a chain.
- **Least group confidence**: Identifies the single least confident reasoning step.
- **Tail confidence**: Evaluates the reliability of the final reasoning steps.

The mathematical foundation uses token confidence, defined as:

$$C_i = -\frac{1}{k}\sum_{j=1}^{k} \log P_{\theta}(t_j^{(i)} | x, t_{<i})$$

where $P_{\theta}$ represents the model's predicted probability for the $j$-th most probable token at position $i$.

**Two operational modes**: DeepConf operates in both offline and online configurations:

**Offline mode**: Uses confidence-weighted majority voting with filtering, retaining only the top η percent of chains based on confidence scores prior to aggregation.

**Online mode**: Implements real-time early stopping using dynamically calibrated thresholds. The system generates an initial set of chains for "warmup" to establish stopping thresholds, then terminates new chains whose group confidence falls below this threshold.

## Key Technical Innovations

The method's effectiveness stems from several technical innovations:

**Dynamic threshold calibration**: For online generation, DeepConf employs a brief "warmup" phase with a small set of traces (typically 16) to establish problem-specific confidence thresholds. This adaptive approach ensures stopping criteria are calibrated to each problem's complexity.

**Confidence-weighted aggregation**: Rather than simple majority voting, DeepConf weights each answer by its associated chain confidence, prioritizing responses from more reliable reasoning paths.

**Adaptive consensus detection**: The system monitors consensus strength during generation and halts when sufficient agreement is reached among high-confidence traces, further reducing computational demands.

## Experimental Results and Performance

DeepConf demonstrates substantial improvements across multiple metrics:

![Figure 2: Token count comparison on AIME 2025, showing significant reduction in computational requirements—up to 84.7% for GPT-OSS-120B while maintaining superior accuracy.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-02.png)

**Improved accuracy**: On challenging mathematical reasoning benchmarks (AIME 2024/2025, BRUMO25, HMMT25), DeepConf consistently outperforms standard majority voting. Notable achievements include:

- GPT-OSS-120B achieves 99.9% accuracy on AIME 2025 (vs. 97.0% for standard majority voting)
- DeepSeek-8B improves from 82.3% to 87.4% on AIME 2025
- Consistent gains across models ranging from 8B to 120B parameters

**Computational efficiency**: Online variants achieve significant token reductions:

- DeepConf-low: 43–84.7% token reduction with aggressive early stopping
- DeepConf-high: 18–59% token reduction with conservative stopping

![Figure 3: Trade-offs between efficiency and accuracy across benchmarks, demonstrating that DeepConf variants consistently achieve higher accuracy with fewer computational resources compared to standard majority voting.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-03.png)

## Confidence Signal Analysis

The study provides a detailed analysis of why local confidence measures outperform global ones:

![Figure 4: Distribution of average confidence scores for correct (green) and incorrect (orange) reasoning traces, showing clear separation enabling effective filtering.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-04.png)

Histograms reveal that different confidence metrics provide varying degrees of separation between correct and incorrect traces. Bottom-10% Group Confidence and Tail Confidence demonstrate particularly strong discriminative power, explaining their effectiveness in filtering.

## Practical Implementation and Scalability

DeepConf was designed with practical deployment as a priority:

- **Framework integration**: The method requires no model training or hyperparameter tuning, making it immediately compatible with existing LLM serving frameworks.
- **Broad model compatibility**: Evaluation across various open-source models (DeepSeek, Qwen, GPT-OSS series) demonstrates generalizability beyond specific architectures.
- **Real-time operation**: The online mode's ability to make termination decisions during generation—rather than after completion—ensures tangible computational savings in production environments.

## Significance and Impact

This work addresses a critical bottleneck in scaling advanced LLM-based reasoning systems. By simultaneously improving accuracy and reducing computational requirements, DeepConf makes complex reasoning capabilities more accessible for practical applications.

The method's success in leveraging internal model confidence signals also advances our understanding of quantifying LLM uncertainty. The discovery that local, fine-grained confidence measures outperform global aggregations has broader implications for AI safety and reliability.

Moreover, the substantial efficiency gains (up to 84.7% token reduction) while maintaining or improving accuracy represent a significant step toward sustainable AI deployment, particularly as reasoning tasks become increasingly complex and computationally demanding.

The study demonstrates that intelligent filtering and dynamic termination strategies can transform the accuracy-efficiency trade-off in LLM reasoning, moving beyond brute-force generation of more traces toward more sophisticated, confidence-aware inference systems.