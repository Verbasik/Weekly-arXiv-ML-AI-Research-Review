# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models

## Table of Contents
0. [TL;DR](#tl;dr)  
1. [Introduction](#introduction)  
2. [Hybrid Architecture Design](#hybrid-architecture-design)  
3. [Training Methodology](#training-methodology)  
4. [Model Compression with MiniPuzzle](#model-compression-with-minipuzzle)  
5. [Performance and Efficiency Improvements](#performance-and-efficiency-improvements)  
6. [Applications and Universality](#applications-and-universality)  
7. [Conclusion](#conclusion)

## **0. TL;DR**

## Core Theme
The introduction and characterization of Nemotron-H, a family of large language models (LLMs) from NVIDIA, utilizing a hybrid architecture that combines Transformer and Mamba layers to enhance efficiency and accuracy, particularly for processing long sequences.

## **Key Ideas and Facts**

### **Hybrid Architecture**
- Nemotron-H leverages the strengths of both Transformer and Mamba.
- Mamba layers, based on State Space Models (SSMs), offer constant computational and memory complexity per token, making them highly efficient for long sequences.
- Transformer self-attention layers are retained at strategically placed positions to capture global relationships.

> "The Nemotron-H family addresses this limitation by introducing a hybrid architecture that combines the strengths of Transformer with the efficiency of Mamba layers."

> "NVIDIA's Nemotron-H models strategically replace the majority of Transformer self-attention layers with Mamba layers..."

> "Unlike self-attention, whose computational and memory complexity scales quadratically with sequence length, Mamba layers provide constant computational and memory complexity per token..."

- Approximately 8% of layers in Nemotron-H-8B/56B models are self-attention layers, uniformly distributed across the entire model.

### **Enhanced Inference Efficiency**
- The hybrid architecture leads to significantly higher inference throughput, especially for long sequences (65,536 tokens).

> "Nemotron-H-56B offers 2.4x higher throughput than Llama-3.1-70B at higher accuracy levels." (Figure 1)

> "Nemotron-H-56B achieves up to 3x higher inference throughput than Qwen-2.5-72B and Llama-3.1-70B"

> "Nemotron-H-8B provides 1.8x higher throughput than Qwen-2.5-7B at comparable accuracy levels"

### **High Accuracy**
- Despite architectural changes, Nemotron-H models demonstrate competitive or superior accuracy compared to pure Transformer models of similar size across a broad range of benchmarks.

> "Nemotron-H-56B outperforms Llama-3.1-70B in 16 out of 17 evaluated tasks"

- Models show particularly strong performance in mathematical reasoning tasks, potentially linked to the inclusion of substantial academic data (8.8%) and code data (20%) in the training set.

## **Training Methodology**

### **Data**
- Training was conducted on a diverse data mixture including web scraping (59% for 56B), code (20%), and academic content (8.8%) to develop a broad spectrum of capabilities.
- The 56B model was trained on approximately 20 trillion tokens.

### **FP8 Training**
- The use of 8-bit floating-point arithmetic (FP8) for training significantly reduces memory and computational requirements while preserving model quality.
- The method includes dynamic scaling, BF16 precision retention for specific layers, and gradual convergence from BF16 training.

> "A significant innovation in Nemotron-H development is the use of 8-bit floating-point arithmetic (FP8) for training, which reduces memory and computational requirements while preserving model quality"

> "Results show that FP8 training can match or exceed BF16 training performance across various benchmarks" (Figure 5)

### **Model Compression with MiniPuzzle**
- A novel compression framework combining pruning, neural architecture search, and knowledge distillation was developed to further improve deployment efficiency.
- MiniPuzzle includes layer importance evaluation, candidate architecture search, and distillation.
- Nemotron-H-56B was successfully compressed to Nemotron-H-47B, reducing parameters by 16%, preserving comparable accuracy, and increasing inference throughput by 20%.

## **Applications and Universality**
Nemotron-H is designed as a universal base model with potential for diverse applications:

- **Vision-Language**: Base models were extended to create vision-language models (VLMs), achieving state-of-the-art results on corresponding benchmarks (VQAv2, GQA, VizWiz).
- **Code Generation**: Models demonstrate strong capabilities in code-related tasks, attributable to the large volume of code data in the training set.
- **Long Context Processing**: The hybrid architecture is particularly well-suited for efficient long-context processing.
- **Data-Driven Capability Adaptation**: The training data distribution can be adjusted to develop specific capabilities, such as STEM competencies, without modifying the architecture.

## **Conclusion**
The Nemotron-H family represents a significant advancement in LLM development, successfully addressing the efficiency limitations of traditional Transformers while preserving high accuracy. The hybrid architecture, innovative FP8 training methodology, and MiniPuzzle compression framework make Nemotron-H an efficient and powerful solution for long-context processing and diverse applications. The availability of these models in popular frameworks is expected to accelerate their adoption and further research into hybrid architectures.

---

## **1. Introduction**
Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their computational demands during inference remain a serious challenge, especially for processing long sequences. The Nemotron-H family addresses this limitation by introducing a hybrid architecture that combines the strengths of Transformer with the efficiency of Mamba layers.

![Figure_01](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_01.jpeg  )

*Comparison of throughput and accuracy. Figure 1: Comparison of Nemotron-H models with other modern LLMs in terms of throughput (tokens/s/GPU) and accuracy on the MMLU benchmark. Nemotron-H-56B offers 2.4x higher throughput than Llama-3.1-70B at higher accuracy levels.*

NVIDIA's Nemotron-H models strategically replace the majority of self-attention layers in Transformers with Mamba layers, which are based on State Space Models (SSMs). Unlike self-attention, whose computational and memory complexity scales quadratically with sequence length, Mamba layers provide constant computational and memory complexity per token, making them especially efficient for generating long sequences.

The key innovation of Nemotron-H lies in the careful balancing of these two architectural paradigms to maintain or improve accuracy while significantly increasing inference speed. This approach addresses the critical community need for LLMs that can efficiently handle long contexts without sacrificing performance.

## **2. Hybrid Architecture Design**

The Nemotron-H architecture combines Mamba-2 layers with traditional Transformer components to create a balanced hybrid design. The model structure strategically places self-attention layers to leverage their strength in capturing global relationships, while Mamba layers are used for efficient sequence processing.

![Figure_02](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_02.png  )

*Figure 2 | Architectures of Nemotron-H-8B/56B models. Approximately 8% of total layers are self-attention layers; these are uniformly distributed across the model. The remainder consists of alternating Mamba-2 and FFN layers.*

As shown in Figure 2, both Nemotron-H-8B and Nemotron-H-56B follow a similar scheme: an initial series of Mamba-2 and FFN layer pairs, followed by a middle section containing one self-attention layer among several Mamba-2 and FFN pairs, and concluding with additional Mamba-2 and FFN layers. The key difference lies in the number of repetitions—while Nemotron-H-8B has 4 repetitions in its middle section, Nemotron-H-56B has 10.

This architecture was meticulously designed through extensive experimentation to find the optimal balance between computational efficiency and model capability. By retaining some self-attention layers, Nemotron-H preserves the model’s ability to capture certain global relationships that Mamba layers may struggle with, while leveraging Mamba’s efficiency for the majority of sequence processing operations.

Key components used in the architecture include:

1. **Mamba-2 layers**: Enable efficient sequence modeling with constant computational complexity per token;
2. **Self-attention layers**: Strategically positioned to capture global relationships;
3. **Feedforward Networks (FFN)**: Process outputs from Mamba and attention layers.

This hybrid approach allows Nemotron-H to process sequences more efficiently than pure Transformer models while maintaining comparable or superior accuracy across a wide range of tasks.

## **3. Training Methodology**

Nemotron-H models were trained using a combination of innovative approaches to ensure high performance and efficiency:

### **3.1 Data Curation and Preparation**

Training data consisted of a diverse mixture of sources, carefully balanced to develop various capabilities:

![Figure_03](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_05.jpeg  )

*Figure 3: Distribution of pre-training data sources for Nemotron-H models, showing the balance between web scraping, code, academic, and other data types.*

The 56B model was trained on approximately 20 trillion tokens, with web-scraped data comprising the largest portion (59%), followed by code (20%) and academic content (8.8%). The data mixture was designed to ensure comprehensive coverage of general knowledge while simultaneously developing strong capabilities in specialized domains such as coding and mathematics.

For post-processing stages, data distribution was adjusted to emphasize supervised fine-tuning (SFT) examples, as shown in subsequent data distribution plots.

### **3.2 FP8 Training Recipe**

A significant innovation in Nemotron-H development is the use of 8-bit floating-point arithmetic (FP8) for training, which reduces memory and computational requirements while preserving model quality:

![Relative loss difference between FP8 and BF16 training](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_06.jpeg  )

*Figure 4: Relative loss difference between FP8 and BF16 training, showing convergence over training progress.*

The FP8 training recipe includes:

- Dynamic scaling for each tensor to enhance stability
- Preservation of the first and last four GEMMs of the model in BF16 precision
- Gradual convergence from BF16 training over time

Results show that FP8 training can match or exceed BF16 training performance across various benchmarks:

![Comparison of FP8 and BF16 training](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_07.jpeg  )

*Figure 5: Comparison of FP8 and BF16 training across various benchmarks, demonstrating comparable or superior performance with FP8 training.*

This method enables more efficient training while preserving or improving model quality, as evidenced by performance metrics on MMLU, commonsense reasoning, code generation, and GSM8K benchmarks.

## **4. Model Compression with MiniPuzzle**

To further enhance deployment efficiency, researchers developed MiniPuzzle, a novel compression framework combining pruning, neural architecture search, and knowledge distillation:

![MiniPuzzle compression framework workflow](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_08.jpeg  )

*Figure 6: MiniPuzzle compression framework workflow, showing transition from a pre-trained model to a compressed model via importance evaluation, neural architecture search, and distillation.*

**The MiniPuzzle approach consists of several stages:**

1. **Importance Estimation**: Analysis of each layer’s contribution to model performance

```python
def importance_estimation(model: Any, dataset: Any) -> List[float]:
    """
    Description:
    ---------------
        Computes importance scores for each model layer based on
        the impact of temporarily disabling the layer on the loss function.
    """

    # Compute importance scores for each layer
    scores: List[float] = []
    for layer in model.layers:
        # Zero out layer outputs and measure impact on loss
        layer_score = measure_impact_on_loss(model, layer, dataset)
        scores.append(layer_score)

    return scores
```

2. **Layer Importance Analysis**: Understanding which layers contribute most to model performance

![Layer importance scores](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_09.jpeg  )

*Figure 7: Layer importance scores, showing varying contributions of different layer types.*

3. **Conditional Neural Architecture Search**: Exploration of candidate compressed architectures

![Layer selection patterns for various candidates](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_10.jpeg  )

*Figure 8: Layer selection patterns for various architecture candidates, showing which layers are retained in each potential compressed model.*

4. **Memory-Performance Trade-off**: Evaluation of models based on memory usage and accuracy

![Memory-performance trade-off search](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_11.jpeg  )

*Figure 9: Trade-off between estimated memory load and performance on benchmarks for candidate architectures.*

5. **Knowledge Distillation**: Training the compressed model to match or exceed the original model’s capabilities

Through this process, Nemotron-H-56B was successfully compressed to Nemotron-H-47B, reducing parameters by 16%, preserving comparable accuracy, and increasing inference throughput by 20%.

## **5. Performance and Efficiency Improvements**

Nemotron-H models demonstrate significant performance and efficiency improvements over comparable Transformer-based models:

### **Inference Throughput**

The hybrid architecture enables substantially faster inference, especially for long sequences:

- Nemotron-H-56B achieves up to 3x higher inference throughput than Qwen-2.5-72B and Llama-3.1-70B
- Nemotron-H-8B provides 1.8x higher throughput than Qwen-2.5-7B at comparable accuracy levels

![Comparison of Nemotron-H-8B with similarly sized models](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_12.jpeg  )

*Figure 10: Comparison of Nemotron-H-8B with similarly sized models in terms of throughput and accuracy.*

These efficiency gains are especially pronounced when processing long sequences (65,536 tokens in the presented examples), highlighting the advantage of Mamba layers’ constant per-token computational complexity.

### **Benchmark Accuracy**

Despite architectural changes, Nemotron-H models maintain high performance across a broad range of benchmarks:

- Nemotron-H-56B outperforms Llama-3.1-70B in 16 out of 17 evaluated tasks
- Models show particularly strong performance in mathematical reasoning tasks

![Comparison of Nemotron-H and other models on MMLU](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_13.jpeg  )

*Figure 11: Comparison of Nemotron-H and other models on MMLU, showing competitive performance.*

Models were evaluated using a comprehensive suite of benchmarks, including MMLU, GSM8K, MATH, HumanEval, and various reasoning tasks, consistently demonstrating competitive or superior performance compared to similarly sized Transformer models.

## **6. Applications and Universality**

Nemotron-H models were designed as universal base models adaptable to diverse applications:

### **Vision-Language Capabilities**

Base models were extended to create vision-language models (VLMs) following the NVLM-D architecture. These VLMs achieved state-of-the-art performance on benchmarks such as VQAv2, GQA, and VizWiz, demonstrating the adaptability of the hybrid architecture to multimodal tasks.

### **Code Generation**

Models demonstrate particularly strong performance in code-related tasks. The inclusion of a substantial volume of code data (20%) in the training mixture contributes to their ability to understand and generate high-quality code across multiple programming languages.

### **Long Context Processing**

One of the most significant advantages of the hybrid architecture is its ability to efficiently process long contexts. The Nemotron-H-8B model was specifically tuned for long-context capabilities, demonstrating high performance on the RULER benchmark and other long-context evaluation tasks.

### **Data Distribution for Different Capabilities**

Researchers carefully calibrated the data distribution across training stages to develop specific capabilities:

![Training data distribution](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-17_&_18/assets/Figure_14.jpeg  )

*Figure 12: Training data distribution optimized for STEM competencies, with increased emphasis on mathematical and code content.*

By adjusting the proportion of different data types (web scraping, code, mathematics, academic, etc.), researchers were able to enhance specific model capabilities without requiring architectural changes.

## **7. Conclusion**

The Nemotron-H family represents a significant leap forward in LLM development, successfully combining the strengths of Transformer and Mamba architectures to create models that are both accurate and efficient. Key achievements include:

1. A hybrid architecture that strategically integrates Mamba layers with self-attention mechanisms to balance performance and efficiency.
2. An FP8 training recipe that reduces computational and memory requirements while preserving model quality.
3. The MiniPuzzle compression framework, which further enhances efficiency through targeted pruning and distillation.
4. Demonstration of significant inference speedup (up to 3x) compared to similarly sized Transformer models.
5. Competitive or superior accuracy across a wide range of benchmark tasks.

The success of Nemotron-H models indicates that hybrid architectures represent a promising direction for the future of LLM development, particularly as applications increasingly demand efficient long-context processing. By addressing the computational bottlenecks of traditional Transformers while preserving their strengths, Nemotron-H offers a practical solution for deploying powerful language models in resource-constrained environments.

The planned release of these models with support in popular frameworks such as Hugging Face, NeMo, and Megatron-LM will enable the broader AI community to leverage these advances and further explore the potential of hybrid architectures for language modeling.