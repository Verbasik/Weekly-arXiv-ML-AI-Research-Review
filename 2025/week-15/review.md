# Computer Science > Computation and Language**
# Title: Qwen2.5-Omni Technical Report
View PDF
Abstract: In this report, we present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. To synchronize the timestamps of video inputs with audio, we organize the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE (Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose \textbf{Thinker-Talker} architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni is comparable with the similarly sized Qwen2.5-VL and outperforms Qwen2-Audio. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni's performance in end-to-end speech instruction following is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni's streaming Talker outperforms most existing streaming and non-streaming alternatives in robustness and naturalness.

### References & Citations
# Bibliographic and Citation Tools
*(What is the Explorer?)* 
*(What is Connected Papers?)* 
*(What is Litmaps?)* 
*(What are Smart Citations?)* 
# Code, Data and Media Associated with this Article
*(What is alphaXiv?)* 
*(What is CatalyzeX?)* 
*(What is DagsHub?)* 
*(What is GotitPub?)* 
*(What is Huggingface?)* 
*(What is Papers with Code?)* 
*(What is ScienceCast?)* 
# Demos
# Recommenders and Search Tools
*(What are Influence Flowers?)* 
*(What is CORE?)* 
# arXivLabs: experimental projects with community collaborators
arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.
Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.
Have an idea for a project that will add value for arXiv's community? **Learn more about arXivLabs** .

# Qwen2.5-Omni
## Overview
### Introduction
Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.
    ![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/qwen_omni.png) 
### Key Features
- **Omni and Novel Architecture**: We propose Thinker-Talker architecture, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. We propose a novel position embedding, named TMRoPE (Time-aligned Multimodal RoPE), to synchronize the timestamps of video inputs with audio.
- **Real-Time Voice and Video Chat**: Architecture designed for fully real-time interactions, supporting chunked input and immediate output.
- **Natural and Robust Speech Generation**: Surpassing many existing streaming and non-streaming alternatives, demonstrating superior robustness and naturalness in speech generation.
- **Strong Performance Across Modalities**: Exhibiting exceptional performance across all modalities when benchmarked against similarly sized single-modality models. Qwen2.5-Omni outperforms the similarly sized Qwen2-Audio in audio capabilities and achieves comparable performance to Qwen2.5-VL-7B.
- **Excellent End-to-End Speech Instruction Following**: Qwen2.5-Omni shows performance in end-to-end speech instruction following that rivals its effectiveness with text inputs, evidenced by benchmarks such as MMLU and GSM8K.
### Model Architecture
    ![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/overview.png) 
### Performance
We conducted a comprehensive evaluation of Qwen2.5-Omni, which demonstrates strong performance across all modalities when compared to similarly sized single-modality models and closed-source models like Qwen2.5-VL-7B, Qwen2-Audio, and Gemini-1.5-pro. In tasks requiring the integration of multiple modalities, such as OmniBench, Qwen2.5-Omni achieves state-of-the-art performance. Furthermore, in single-modality tasks, it excels in areas including speech recognition (Common Voice), translation (CoVoST2), audio understanding (MMAU), image reasoning (MMMU, MMStar), video understanding (MVBench), and speech generation (Seed-tts-eval and subjective naturalness).
    ![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/bar.png) 
## Multimodality -> Text
| Datasets | Model | Performance | 
|----------|-------|-------------|
| MMMU     | Qwen2.5-Omni | 75.3 |
| MathVision | Qwen2.5-Omni | 72.1 |
| MMBench  | Qwen2.5-Omni | 78.9 |
| TextVQA  | Qwen2.5-Omni | 69.4 |
| DocVQA   | Qwen2.5-Omni | 71.2 |
| ChartQA  | Qwen2.5-Omni | 73.8 |
| OmniBench | Qwen2.5-Omni | 79.6 |
| Common Voice | Qwen2.5-Omni | 92.1 WER |
| CoVoST2  | Qwen2.5-Omni | 18.4 BLEU |
| MMAU     | Qwen2.5-Omni | 84.5 |
| MMStar   | Qwen2.5-Omni | 76.2 |
| MVBench  | Qwen2.5-Omni | 68.3 |
| Seed-tts-eval | Qwen2.5-Omni | 1.42% WER (test-zh) |
|          |       | 2.33% WER (test-en) |
|          |       | 6.54% WER (test-hard) |

[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg  )](https://arxiv.org/abs/2503.20215  )
[![GitHub](https://img.shields.io/badge/GitHub-Qwen2.5-Omni-brightgreen  )](https://github.com/QwenLM/Qwen2.5-Omni  )
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow  )](https://huggingface.co/Qwen/Qwen2.5-Omni-7B  )

# How LLMs Learn Facts: Dynamics, Memorization, Hallucinations. New Research from Google DeepMind

## **1. Introduction**

### **Key Problem**: 

Although large language models (LLMs) acquire vast amounts of factual knowledge during pre-training, the internal mechanisms by which they learn, store, and apply these facts remain a "black box." Uncovering these mechanisms is critical not only for optimizing model training but also for understanding and addressing key challenges such as "hallucinations" and difficulties in updating knowledge after pre-training.

### **Research Methodology**: 

To systematically investigate this problem, this paper develops a controlled experimental methodology. Instead of using complex real-world texts, the researchers created a synthetic factual recall task based on artificial biographies. This synthetic approach allows precise control over data properties and efficient tracking of knowledge acquisition across all training stages.

### **Key Findings in Brief**:  

1. **Staged Learning**: Knowledge is not acquired linearly. The model exhibits three distinct phases of fact learning, including a critical "plateau phase" during which performance appears stagnant while internal representation mechanisms are forming. **The plateau phase occurs during pre-training!**
2. **Importance of Data Distribution**: The statistical properties of training data, particularly the frequency distribution of mentions of different "individuals," significantly impact learning speed and dynamics, including the duration of the plateau phase.
3. **Coexistence of Hallucinations and Knowledge**: The model's tendency toward "hallucinations" (generating information about non-existent entities) emerges almost simultaneously with the acquisition of real facts.
4. **Difficulties in Fine-Tuning**: Integrating new knowledge into an already-trained model via fine-tuning proves extremely difficult and often leads to rapid degradation of existing parametric memory (a phenomenon known as "catastrophic forgetting").

## **2. Tracking Knowledge Acquisition: Details of the Experimental Environment**

**Key Question**:  
When studying how LLMs acquire facts, we face two methodological challenges:

1. **Isolating Knowledge**:  
   How do we measure a model’s factual knowledge while separating it from other linguistic abilities (grammar, fluency, etc.)?

2. **Efficiency and Scalability of Evaluation**:  
   How can we continuously monitor knowledge levels during training without resorting to expensive, comprehensive evaluations (e.g., QA tests) at every step?

**Intuition and General Approach**:  
An ideal experimental environment should have the following characteristics:
- Facts are discrete and atomic;
- Task success depends directly on reproducing specific "entity-attribute" pairs;
- Data generation is controllable to register statistical properties;
- Knowledge evaluation is integrated into the standard training process.

This naturally leads to the use of structured synthetic data for factual recall tasks.

### 2.1 Knowledge vs. Memory: The Key Distinction

To accurately understand model behavior, it is essential to distinguish between **"knowledge"** and **"memory"**:

| Concept  | Definition                                                                 | Characteristics                          | Example (Knowledge "Paris is the capital of France")              |
|----------|-----------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------|
| Knowledge   | Information internalized by the model, independent of input form and flexibly applied | Abstraction, generalization, flexibility     | Answers to questions: "Capital of France?", "Which country does Paris belong to?" |
| Memory   | Reproduction of specific training examples, tied to input form              | Concreteness, fragility                | Completion of sentence: "Paris is the ___ of France"      |

### 2.2 Synthetic Biography Dataset

**Design Advantages**:
- **Atomicity**: Each fact (e.g., place of birth) is independent, separating the ability to "recall" from the ability to "reason";
- **Synthetic and Controlled**: Precise control over data distribution (e.g., character frequency) without interference from real-world corpora;
- **Realistic Statistics**: Use of common names and locations preserves natural token distributions;
- **Relevance to Prior Research**: Small models on similar data demonstrate knowledge storage mechanisms comparable to large LLMs.

**Generation Process**:
1. **Create Character Base**: Generate N virtual "characters" with unique names and six attributes;
2. **Fill Templates**: For each attribute, randomly select a template from a library (25 variants per attribute type), substituting specific information (e.g., "[Name] was born in [Place of Birth]");
   - *Key Point*: Multiple templates create textual diversity, forcing the model beyond simple memorization
3. **Assemble Biography**: Randomly order attribute sentences into a full biography;
   - *Key Point*: Random order prevents the model from relying on sequential cues
4. **Split into Training/Evaluation Sets**: For each character, 20 templates go to training, 5 to evaluation. This ensures the model encounters new formulations of known facts, testing the level of knowledge abstraction.

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_01.png  )
> Figure_1. The data generation process underlying the synthetic biography dataset on which we train models. We measure the knowledge contained in these models by the loss they achieve when predicting attribute tokens (highlighted in blue).*

### 2.3 Large-Scale Knowledge Measurement: Attribute Loss and Accuracy

**Problem**: Repeated QA evaluations during training require substantial computation  
**Solution**: The biography structure transforms attribute value prediction into a factual recall task  

### 2.4 Standardized Models and Training

To ensure reproducibility of results:

- **Architecture**: 8-layer Decoder-only Transformer (44M parameters, based on Hoffmann et al., 2022)
- **Optimizer**: AdamW with cosine learning rate decay (no warmup)
- **Learning Rate**: Custom-tuned per experiment

## **3. Dynamic Process of Language Model Knowledge Acquisition**

Main Question: Now that we have a way to measure knowledge, what is the actual dynamics of knowledge acquisition during training? Is it a smooth, gradual process, or do clear phase transitions and potential mechanism shifts occur?

Main Findings: The study revealed that, regardless of hyperparameter changes, knowledge acquisition typically follows a three-stage pattern. Among these, the seemingly "stagnant" plateau phase plays a decisive role at the mechanism level.

### **3.1 Three-Stage Model of Knowledge Acquisition**

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_02.png  )
> Figure_2. Knowledge acquisition occurs in three stages. (Left) On a very short first stage, the model learns the overall statistics of attribute values. On the second stage, performance plateaus at a level achievable by an ideal model without knowledge of individual entities (this corresponds to the "no-knowledge" baseline and nearly zero recognition accuracy). The duration of this plateau is nearly proportional to the number of individuals (right). Finally, the model learns associations between subjects and attributes: knowledge emerges as training continues (center). Results are averaged over 5 runs (± standard deviation).

Researchers consistently observed the following three stages by tracking attribute loss (Attribute Loss) and attribute accuracy (Attribute Accuracy) during training:

| Stage | Name                      | Primary Behavior                          | Explanation and Mechanism                                                                 |
|-------|---------------------------|-------------------------------------------|---------------------------------------------------------------------------------------|
| 1     | Initial Understanding / Statistical Learning | Rapid decrease in attribute loss.          | The model quickly absorbs superficial statistical data, such as frequent attribute values, biography structure, etc. By the end of this stage, performance reaches the **no-knowledge baseline**. The model understands the types of information but does not link them to specific individuals. |
| 2     | Performance Plateau ("Knowledge Acquisition Frontier") | Attribute loss remains at the **no-knowledge baseline**; attribute accuracy nears 0. | Why does stagnation occur? Two possible reasons: <br> (1) **Optimization**: The model falls into a saddle point or local minimum of the loss function. <br> (2) **Statistics (√)**: The model requires multiple observations of the same individual (despite varied descriptions) to reliably extract fact-specific information from statistical noise. <br> **Evidence**: Plateau duration scales linearly with the number of individuals ($\text{Plateau} \propto N^{0.81}$, Fig. 2 right), strongly supporting the statistical hypothesis. |
| 3     | Knowledge Emergence       | Attribute loss becomes significantly lower than the **no-knowledge baseline**; attribute accuracy consistently exceeds 0. | At this stage, the model actively forms and strengthens associations between **individual names** and their **specific attributes**. Parametrized knowledge is stored and successfully retrieved. |

Model Robustness: This three-stage model remains stable under changes in learning rate, weight decay, batch size, number of individuals, model size, and even when replacing the attention mechanism with an RNN variant.

### **Connection to the Full LLM Training Cycle**

Understanding the three-stage model of knowledge acquisition helps reframe the entire process of training modern LLMs. Let’s examine how these three stages relate to traditional LLM training phases:

<details> 
    <summary><em><strong>Full Cycle of Modern LLM Training</strong></em></summary>

---

Training modern large language models is a complex, multi-stage, and resource-intensive process. It includes several phases, each serving a distinct purpose: from forming basic language understanding to fine-tuning model behavior to align with human expectations. Let’s break down this cycle step-by-step.

**Full Cycle of LLM Training**

1.  **Data Preparation**
2.  **Pre-training**
3.  **Supervised Fine-Tuning (SFT) / Instruction Fine-Tuning**
4.  **Reinforcement Learning from Human Feedback (RLHF)**
5.  **Evaluation and Deployment**

![Этапы обучения LLM](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Image_01.jpeg  )
> Stages of LLM Training

The process of training language models typically consists of three phases shown below. First, we pre-train the language model, and this stage is by far the most computationally expensive part of training. Then we perform alignment, usually via a three-stage framework using supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF).
> Interestingly, this stage does not require human feedback. Recent research explores reinforcement learning from AI feedback (RLAIF)!

![Этапы обучения LLM](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Image_02.jpeg  )

The above stages constitute the standardized training pipeline used for most modern LLMs (e.g., ChatGPT or LLaMA-3). Compared to pre-training, SFT and RLHF are computationally inexpensive but require curation of datasets (or high-quality LLM outputs, or human feedback on LLM outputs), which can be complex and time-consuming.

![Этапы обучения LLM](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Image_03.jpeg  )

Sometimes, to solve a narrow task, we need to do more than simply apply an LLM. In particular, we may further specialize the language model (if needed) via either domain-specific fine-tuning or in-context learning (see below). Domain-specific fine-tuning simply continues model training (usually with a language modeling objective similar to pre-training/SFT) on data relevant to the narrow task, while in-context learning adds more context or examples to the LLM prompt used as context for solving the task.

![Этапы обучения LLM](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Image_04.jpeg  )

What is alignment? We have used the term above multiple times, and it is important to understand: alignment. A pre-trained language model is typically not particularly useful. If we generate outputs using this model, the results are likely repetitive and inapplicable. To create a more useful language model, we need to align this model with the desires of a human user. In other words, instead of generating the most probable text sequence, our model learns to generate the text sequence requested by the user.

This alignment, performed via the aforementioned three-stage SFT and RLHF framework, can be used to steer LLMs toward diverse behaviors and properties. Typically, they train the model to select among one or more criteria emphasized during alignment. The most common alignment criteria are: improving instruction-following ability, preventing harmful outputs, increasing usefulness of the LLM, and many others.

---

<details> 
    <summary><em><strong>Stage 1. Data Preparation</strong></em></summary>

## **Stage 1. Data Preparation**

### **1.1 Conceptual Foundations of Data Preparation**

The quality and quantity of data represent the foundational basis for the functioning of large-scale language models (Large Language Models, LLM). The process of preparing data for pre-training must be considered as a multi-faceted task, whose strategic directions are heavily influenced by recent research in neural architecture scaling. The complexity of this task stems from the need to ensure an optimal balance between data volume, quality, and available computational resources for training the model.

## **2. Systematization of Data Collection and Processing Processes**

### **2.1 Methods of Text Corpus Accumulation**

In modern LLM development, data accumulation is achieved by integrating large text corpora from diverse sources, including:
* web resources (primarily Common Crawl);
* literary sources (Project Gutenberg, Google Books collections);
* scientific publications (arXiv repository);
* program code (GitHub);
* dialogue corpora;
* media publications.

The volume of data used in modern projects amounts to terabytes of textual information, corresponding to trillions of tokenized elements. This quantitative characteristic is based on empirically established scaling laws for language models.

### **2.2 Theoretical and Practical Aspects of Scaling Laws (Chinchilla Scaling Laws)**

Data collection strategy and training planning are closely tied to empirical scaling laws. DeepMind’s "Chinchilla" study (2022) established that for optimal model performance under a fixed computational budget (FLOPs), a balanced ratio between model size (number of parameters, $N$) and training data volume (number of tokens, $D$) must be maintained.

#### **2.2.1 Statistical Patterns and Their Interpretation**
According to Chinchilla’s laws, the optimal ratio is expressed as $D \approx 20 \times N$, indicating the need for approximately 20 training tokens per model parameter. This proportion is based on empirical observations and statistical analysis of model efficiency across various configurations.

#### **2.2.2 Practical Significance of the Study in the Context of LLM Development**
The discovery made in the Chinchilla project demonstrated that prior large-scale models (including GPT-3 and Gopher) suffered from a suboptimal ratio of training data volume to model size. Specifically, the Chinchilla model (70 billion parameters), trained on 1.4 trillion tokens (ratio ~20:1), outperformed the more parameterized Gopher model (280 billion parameters), trained on 300 billion tokens (ratio ~1:1).

#### **2.2.3 Methodological Implications for Data Preparation**
The established patterns underscore the critical importance of not only increasing model parameterization but also proactively scaling the volume of high-quality training data. This insight stimulates intensified efforts by the research community to collect, filter, and process trillions of textual tokens to maximize the efficiency of computational resources during LLM training.

### **2.3 Methodology of Data Cleaning and Normalization**

Data cleaning is a crucial stage that directly impacts the quality and safety of the model. This process includes the following components:

* **Removal of duplicates** at the document and sentence level to enhance data diversity and prevent overfitting to repeating patterns.

* **Filtering of low-quality content**, including spam, templated text, and automatically generated information. This procedure aims to improve the overall corpus quality and reduce the risk of training on irrelevant or low-information data.

* **Processing of personal data** via removal or anonymization of personally identifiable information (PII) to ensure compliance with privacy and data protection regulations.

* **Filtering of undesirable content**, including toxic, biased, or potentially harmful material. This task presents a complex challenge requiring both automated algorithms and manual moderation.

* **Text normalization**, which may include standardizing case, punctuation, and whitespace. However, it should be noted that modern architectures often demonstrate higher efficiency when working with text as close as possible to its original form, preserving original case and punctuation.

### **2.4 Algorithmic Approaches to Text Tokenization**

Tokenization is the process of decomposing text into elementary units — tokens — for model processing. In the context of modern LLMs, subword tokenizers are predominantly used, among which the following can be highlighted:

* **Byte Pair Encoding (BPE)**: An algorithm that begins with individual characters (or bytes) and iteratively merges the most frequent pairs into new dictionary tokens. This method provides an effective balance between vocabulary size and the ability to model rare words.

* **WordPiece**: A method methodologically similar to BPE but differing in the criterion for merging pairs, which is maximizing the likelihood of the training data under a given tokenization model. This algorithm has been applied in BERT and other Google developments.

* **SentencePiece**: A tokenizer that processes text as a sequence of Unicode characters without prior word segmentation. This feature ensures universal applicability across diverse language systems, which is especially relevant for multilingual models. SentencePiece is actively used in modern architectures, including Llama and the GPT series.

#### **2.4.1 Illustrative Example and Functional Analysis of Tokenization**

As an example, consider the tokenization of the word "масштабирование", which may be segmented into tokens such as `[" мас", "штаб", "ирование"]` or `[" масштаб", "ирован", "ие"]`. This mechanism provides the model with the following functional capabilities:

1. **Handling Unknown Lexicon**: Ability to analyze and generate unfamiliar or rare words by composing them from known segments, significantly enhancing model flexibility with open vocabularies.

2. **Vocabulary Size Optimization**: Capability to efficiently manage vocabulary size (typically ranging from 30 to 100 thousand tokens), which would be impossible if whole words were used as base token units.

## **3. Conclusions**

The Chinchilla scaling laws represent a significant methodological guideline in the data preparation stage, defining target corpus volumes based on the planned parameter size of the model and available computational resources. These patterns emphasize that LLM efficiency is determined not only by architectural and algorithmic aspects of training but also substantially by strategic approaches to data handling.

In the context of modern AI research, and particularly in natural language processing, deep understanding of the relationship between training data characteristics and model performance is fundamental for further progress in developing increasingly sophisticated language models. Strategic planning of data collection and processing based on empirically established laws has become one of the key factors determining the success of projects in the field of large-scale language model development.

</details> 

<details> 
    <summary><em><strong>Stage 2. Pre-training</strong></em></summary>

## **2. Pre-training**

### **2.1. Conceptual Characterization of the Pre-training Stage**

Pre-training is the most computationally intensive stage in the development of large-scale language models, during which the model forms a foundational understanding of the statistical and semantic patterns of language. This process requires substantial computational resources and is decisive for the model’s functional capabilities.

The primary objective of pre-training is to enable the model to predict the next token in a sequence based on the context of preceding tokens. This approach allows the model to internalize grammatical structures, factual information, basic logical relationships, and certain aspects of reasoning presented in large text corpora. Learning is based on identifying statistical patterns in token sequences, thereby forming generalized representations of linguistic structures without requiring explicit data annotation.

### **2.2. Architectural Components of Modern Language Models**

Currently, the dominant architecture in large-scale language models is the Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017).

The figure below illustrates the Transformer architecture, which consists of two main components: the **encoder** and the **decoder**.

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-04/assets/Figure_1.png  )

### Encoder
The encoder is typically located on the left side of the architecture. It consists of multiple layers, each containing:
1. **Multi-Head Attention** — an attention mechanism that enables the model to focus on different parts of the input data.
2. **Add & Norm** — a layer that adds the input to the attention output (residual connection) and applies normalization.
3. **Feed Forward** — a fully connected layer applied independently to each element of the sequence.
4. **Add & Norm** — again adds the input to the result and normalizes.

These layers are repeated several times (typically six or more) to create a deep model.

### Decoder
The decoder is typically located on the right side of the architecture. It also consists of multiple layers but includes additional components:
1. **Masked Multi-Head Attention** — an attention mechanism that masks future tokens to prevent "peeking ahead."
2. **Add & Norm** — a layer that adds the input to the attention output and normalizes.
3. **Multi-Head Attention** — an attention mechanism that incorporates the encoder’s output.
4. **Add & Norm** — again adds the input to the result and normalizes.
5. **Feed Forward** — a fully connected layer analogous to that used in the encoder.
6. **Add & Norm** — the final add-and-norm layer.

### Inputs and Outputs
- **Input Embedding** and **Positional Encoding** correspond to the inputs fed into the encoder.
- **Output Embedding** and **Outputs (shifted right)** correspond to the outputs processed by the decoder.

### **2.3. Methodology of the Training Process**

#### **2.3.1. Formulation of the Pre-training Task**

In the context of large-scale language models, the primary pre-training task is causal language modeling (Causal Language Modeling, CLM), which involves sequential token prediction. The model receives as input a sequence of tokens $t_1, t_2, ..., t_{k-1}$ and is optimized to predict the next token $t_k$. This task formulation enables the model to learn a broad spectrum of linguistic patterns without requiring specialized data annotation.

#### **2.3.2. Loss Function and Its Justification**

The cross-entropy loss function is used as the optimization objective, quantitatively measuring the divergence between the model’s predicted probability distribution over the next token and the actual distribution represented by the true next token.

Mathematically, for a sequence $T = (t_1, ..., t_L)$, the loss function is expressed as:

$$L_{Pretrain}(\theta) = - \sum_{i=1}^{L} \log P(t_i | t_{1}, ..., t_{i-1}; \theta)$$

where: 

- $\theta$ — the model parameters
- $P(t_i | t_{1}, ..., t_{i-1}; \theta)$ — the probability of the $i$-th token predicted by the model based on preceding tokens. 
In practical implementations, computations are performed over batches of sequences.

The conceptual interpretation of this function is that the model is penalized for assigning low probability to a token that actually follows the preceding sequence in the training text. Minimizing this function encourages the model to more accurately predict text sequences, leading to the internalization of structural and semantic linguistic patterns.

#### **2.3.3. Optimization Algorithms**

In the pre-training of large-scale language models, adaptive optimization algorithms such as Adam (Adaptive Moment Estimation) or its modification AdamW, featuring an enhanced weight regularization mechanism, are predominantly used. The key hyperparameter of the optimization process is the learning rate, which determines the magnitude of parameter updates at each iteration.

#### **2.3.4. Learning Rate Scheduling Strategies**

Training typically begins with low learning rates, gradually increasing to a maximum value over the initial thousands of iterations (warmup phase). This approach stabilizes training during early stages. After the warmup phase, the learning rate is gradually reduced according to a predefined schedule (e.g., cosine decay), facilitating more precise convergence to the optimum.

The mathematical formulation of the warmup phase is:

$$lr(step) = lr_{max} \times \frac{step}{warmup\_steps}$$ 

for $step \leq warmup\_steps$

where: 

- $lr(step)$ — the learning rate at iteration $step$ 
- $lr_{max}$ — the maximum learning rate value
- $warmup\_steps$ — the number of warmup iterations. During this phase, the learning rate linearly increases from nearly zero to its maximum, ensuring a smooth start to the optimization process.

### **2.4. Computational Aspects and Scaling Strategies**

Pre-training large-scale language models demands substantial computational resources, including hundreds or thousands of GPU or TPU processors operating over extended periods. To ensure training efficiency, the following distributed learning strategies are applied:

* **Data Parallelism**: Distributes different data batches across computing devices, each holding a full copy of the model. After processing batches, the average gradient is computed and used to update parameters on all devices.

* **Tensor Parallelism**: Distributes individual tensors (weight matrices) across devices, which is especially effective for models exceeding the memory capacity of a single device. This approach enables training significantly larger models than possible on a single device.

* **Pipeline Parallelism**: Distributes different model layers across devices with pipelined processing of micro-batches, ensuring efficient utilization of computational resources when training deep architectures.

## **3. Conclusions**

The outcome of the pre-training stage is a base model characterized by developed abilities to understand and generate text. However, it should be noted that this model is not yet optimized for executing specific instructions or engaging in dialogic interaction, necessitating subsequent training stages.

It is crucial to emphasize that the quality of the base model obtained during pre-training largely determines the model’s potential for subsequent fine-tuning and reinforcement learning. Thus, optimizing the pre-training process represents a critical task in the development of highly efficient large-scale language models.

</details>

<details> 
    <summary><em><strong>Stage 3. Supervised Fine-Tuning (SFT) / Instruction Fine-Tuning</strong></em></summary>

### 3. Supervised Fine-Tuning (SFT) / Instruction Fine-Tuning

Supervised fine-tuning (SFT) is the first stage of learning within the LLM alignment process. First, we curate a dataset of high-quality LLM outputs (essentially, examples of correct LLM behavior) (see below). Then, we directly fine-tune the model on these examples. The term "Supervised" in SFT means we collect a dataset of examples of how the model should behave. The model then learns to reproduce the style of these examples during fine-tuning.

![Image_05](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Image_05.jpeg  )

Connection to Next-Token Prediction. Interestingly, SFT does not differ significantly from pre-training a language model — both use next-token prediction as the training objective! The main difference lies in how the data is applied. During pre-training, we use massive corpora of raw text data. In SFT, the dataset of high-quality LLM outputs serves as the "teacher." On each training iteration, we sample multiple examples and fine-tune the model on this data using next-token prediction as the objective. Typically, next-token prediction is applied only to the portion of each example corresponding to the LLM output (e.g., the answer in the figure above).

#### **Formalizing the SFT Process**

Formally, the Supervised Fine-Tuning (SFT) stage can be described as follows. Let $\pi_{\theta}$ denote the pre-trained language model with parameters $\theta$, obtained during pre-training. The goal of SFT is to adapt this model for better instruction-following and generation of responses in a desired style, using a curated dataset.

This dataset, $D_{SFT}$, consists of $M$ example pairs:
$$D_{SFT} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{M}$$
where:
*   $x^{(i)}$ — the input token sequence representing an instruction, question, or any other prompt (e.g., `Input` in the figure above).
*   $y^{(i)}$ — the corresponding reference output token sequence demonstrating the desired model response or behavior (e.g., `Output` in the figure above).

The SFT process involves further optimization of the model parameters, initialized with values $\theta$, to minimize the loss on dataset $D_{SFT}$. As the loss function, similar to the pre-training stage, the cross-entropy loss is typically used, with a key distinction: losses are computed **only** for the tokens of the reference answer $y^{(i)}$. This is the critical point distinguishing SFT from pre-training, where losses are computed over the entire sequence.

Let $y^{(i)} = (y_1^{(i)}, y_2^{(i)}, ..., y_{L_i}^{(i)})$ be the sequence of reference answer tokens of length $L_i$. The model $\pi_{\phi}$, with parameters $\phi$ (initialized as $\phi_{init} = \theta$), is trained to maximize the probability of generating the sequence $y^{(i)}$ given the input sequence $x^{(i)}$. This is equivalent to minimizing the following loss function $L_{SFT}$:

$$L_{SFT}(\phi) = - \sum_{i=1}^{M} \frac{1}{L_i} \sum_{j=1}^{L_i} \log P_{\pi_{\phi}}(y_j^{(i)} | x^{(i)}, y_1^{(i)}, ..., y_{j-1}^{(i)}; \phi)$$

where:
*   $\phi$ — the model parameters optimized during SFT.
*   $P_{\pi_{\phi}}(y_j^{(i)} | x^{(i)}, y_1^{(i)}, ..., y_{j-1}^{(i)}; \phi)$ — the probability of the $j$-th answer token $y_j^{(i)}$ predicted by model $\pi_{\phi}$ based on the input prompt $x^{(i)}$ and all preceding *true* answer tokens $y_1^{(i)}, ..., y_{j-1}^{(i)}$ (this mode is called "teacher forcing").
*   The summation is over all examples $i$ in dataset $D_{SFT}$ and all tokens $j$ in each reference answer $y^{(i)}$.
*   Often, averaging over sequence length $L_i$ (as shown) or total tokens in a batch is used for loss normalization.

**Practical Implementation.** In practice, this is often implemented by concatenating the prompt and answer ($c^{(i)} = x^{(i)} \oplus y^{(i)}$) and applying the standard next-token prediction loss to the entire sequence $c^{(i)}$. However, during gradient computation and weight updates, only losses corresponding to tokens in $y^{(i)}$ are considered. This is achieved via a loss masking mechanism, which nullifies (ignores) losses for tokens belonging to the input prompt $x^{(i)}$. Thus, the model learns to predict and generate only the desired answer, continuing the sequence given by the prompt.

Parameter optimization $\phi$ is performed using standard stochastic gradient descent methods (e.g., AdamW), similar to the pre-training stage, but typically with a lower learning rate and on a significantly smaller (by orders of magnitude) volume of data compared to pre-training. The goal of SFT is not so much to teach the model new knowledge (though some may be acquired), but to "teach" it to use existing knowledge to generate responses in a specific format and style matching the examples in $D_{SFT}$.

It should be noted that SFT differs slightly from general fine-tuning. Typically, fine-tuning a deep learning model aims to teach it to solve a specific task, making the model more specialized and less general — the model becomes a "niche specialist." The model will likely perform better on the fine-tuned task compared to a general model but may lose its ability to solve other tasks. SFT, however, is a fundamental component of LLM alignment, including for general-purpose base models. Since we fine-tune the model to imitate correct style or behavior, rather than to solve a specific task, it does not lose its ability to solve general tasks.

</details>

<details> 
    <summary><em><strong>Stage 4. Reinforcement Learning from Human Feedback (RLHF)</strong></em></summary>

### 4. Reinforcement Learning from Human Feedback (RLHF)

This is the most complex stage, aimed at aligning the model’s behavior with human preferences to make its responses more helpful, honest, and harmless.

*   **Goal:**  
  Fine-tuning the model using signals from human feedback to generate responses that humans rate as high-quality.

**The process consists of two main stages:**

  **Stage 4.1: Training the Reward Model (RM)**
    
  *   **Data Collection:**
      1.  A set of prompts is taken;
      2.  The SFT model generates several (often two) response variants for each prompt;
      3.  Human assessors compare these responses and select the better one (or rank them);
      4.  A preference dataset is collected: `(prompt, chosen_response, rejected_response)`.
  *   **RM Architecture:**  
  Typically, the same LLM architecture as the main model (or just its encoder) is used, with an added "head" (a linear layer) that predicts a scalar value — the "score" (reward) of response quality for a given prompt.
    *   **Loss Function (based on Bradley-Terry model):**  
  Train the RM to assign a higher score to the chosen response ($y_w$) than to the rejected one ($y_l$).

      *   **Mathematical Formalization:**

          $$L_{RM}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\phi(x, y_w) - r_\phi(x, y_l)))]$$
          
          where:

          *   $D$ — the human preference dataset.
          *   $x$ — the prompt.
          *   $y_w$ — the chosen (preferred) response.
          *   $y_l$ — the rejected response.
          *   $r_\phi(x, y)$ — the scalar score output by the RM with parameters $\phi$.
          *   $\sigma$ — the sigmoid function.

      *   **Explanation:**  
  The loss function penalizes the RM if it assigns a score to the rejected answer $y_l$ that is close to or higher than the chosen answer $y_w$. Minimizing the loss forces the RM to learn to predict which answer humans will prefer.

  **Stage 4.2: Policy Optimization via Reinforcement Learning (RL)**

*   **Goal:**  
  Adjust the SFT model (now called the "policy" $\pi^{RL}$) to generate responses that maximize the score from the trained RM, without deviating too far from the original SFT model.

  *   **Process (using PPO - Proximal Policy Optimization):**
      1.  A prompt $x$ is sampled from the prompt dataset.
      2.  The current policy $\pi_{\theta}^{RL}$ (initialized with SFT model weights) generates an answer $y$.
      3.  The reward model $r_\phi(x, y)$ evaluates the generated answer $y$.
      4.  The PPO algorithm updates the policy parameters $\theta$ to maximize the expected reward, using a specialized objective function.
  *   **Objective Function (simplified for PPO):**
      
      $$J(\theta) = \mathbb{E}_{x \sim D_{prompt}, y \sim \pi_{\theta}^{RL}(y|x)} [r_\phi(x, y) - \beta \text{KL}(\pi_{\theta}^{RL}(y|x) || \pi^{SFT}(y|x))]$$

      where:

      *   $\pi_{\theta}^{RL}$ — the optimized policy (LLM).
      *   $\pi^{SFT}$ — the original SFT model (its weights are frozen).
      *   $r_\phi(x, y)$ — the reward from the RM.
      *   $\text{KL}(\pi_{\theta}^{RL} || \pi^{SFT})$ — the KL divergence between the token probability distributions output by the current policy and the original SFT model. This is a penalty for deviation from the SFT model.
      *   $\beta$ — a coefficient controlling the strength of the KL penalty.

  *   **Explanation:**
      *   The first term $r_\phi(x, y)$ encourages the model to generate answers that the RM (and thus humans) like.
      *   The second term (KL penalty) prevents the model from deviating too far from the SFT model. This is important for two reasons: 1) It prevents "optimization collapse," where the model finds a way to get high reward from the RM by generating meaningless or undesirable answers (exploiting RM weaknesses). 2) It helps preserve the general language abilities learned during pre-training and SFT.
      *   PPO uses a more complex surrogate loss function with "clipping" of probability ratios to ensure stable updates, but the overall goal remains the same.

*   **Alternatives to RLHF:**  
  Recently, methods such as **Direct Preference Optimization (DPO)** have gained popularity, allowing direct optimization of the model based on preference data without explicitly training a separate reward model or using complex RL algorithms, potentially simplifying and stabilizing the process.

**Result of RLHF:**  
  A model whose responses better align with human preferences regarding helpfulness, honesty, and harmlessness. This is typically the final model version ready for evaluation and deployment.

</details>

<details> 
    <summary><em><strong>Stage 5. Evaluation and Deployment</strong></em></summary>

### 5. Evaluation and Deployment

After training, the model undergoes thorough evaluation before deployment.

*   **Evaluation:**
    *   **Academic Benchmarks:**  
  Sets of tasks for evaluating language understanding, question answering, and logical reasoning (e.g., MMLU, HellaSwag, ARC, TruthfulQA).
    *   **Human Evaluation:**  
  The most critical form of evaluation for dialogue models. Assessors rate response quality across various criteria (usefulness, relevance, safety, tone) or compare responses from different models on identical prompts (A/B tests, Side-by-Side comparisons).
    *   **Specialized Tests:**  
  Assessment for biases, generation of toxic content, coding ability, mathematical reasoning, and more.

*   **Deployment (Deployment):**
    *   **Inference:**  
  Running the model to generate responses to user queries. Requires significant computational resources (GPUs).
    *   **Inference Optimization:**  
  Techniques such as quantization (reducing the precision of model weights) and distillation (training a smaller model to replicate the behavior of a larger one) to reduce memory requirements and accelerate performance.
    *   **Monitoring:**  
  Continuous tracking of model performance, collection of user feedback, and identification of issues (e.g., "hallucinations" — generation of incorrect information).
    *   **Iterative Improvement:**  
  The training cycle (particularly SFT and RLHF) can be repeated with new data and feedback to continuously improve the model.

</details>

<details> 
    <summary><em><strong>Next-Token Prediction</strong></em></summary>

## Next-Token Prediction in LLM Training

Next-token prediction is the fundamental task underlying the pre-training of most modern autoregressive large language models (LLMs), such as GPT, Llama, PaLM, Mistral, and others. This strategy is a key example of **self-supervised learning**, enabling models to learn complex linguistic patterns, grammar, semantics, and even world facts by leveraging massive volumes of unlabeled text without requiring manual annotation. In this review, we examine in detail the mechanisms underpinning next-token prediction, their principles of operation, mathematical formalization, and significance for creating powerful LLMs.

### Mechanism of Next-Token Prediction

The core idea of the mechanism is autoregression: predicting the next element in a sequence based on all preceding elements. When an LLM receives as input a sequence of tokens $t_1, t_2, ..., t_{k-1}$, its task is to predict the most probable next token $t_k$.

1.  **Context Processing via Transformer:** Modern LLMs use the **Transformer** architecture. Its key component is the **Self-Attention** mechanism. It allows the model to dynamically determine which of the preceding tokens ($t_1$ to $t_{k-1}$) are most relevant for predicting the next token $t_k$. The model computes "attention scores" between the current position (where $t_k$ is expected) and all previous positions, converting them into weights that determine how strongly each preceding token influences the prediction.

    *   **Masked Self-Attention (Causal Masking):**  
  In models trained on next-token prediction (Transformer decoders), a special type of attention is used—masked attention. The mask prevents the attention mechanism from "looking ahead" in the sequence. When processing the $i$-th position, the model can only consider tokens from position $1$ to $i$ (or $i-1$ for predicting the $i$-th token), but not $i+1$, $i+2$, etc. This is critical for preserving the autoregressive property: prediction depends solely on the past, not the future.

2.  **Generation of Probability Distribution:** After the input sequence $t_1, ..., t_{k-1}$ is processed by multiple Transformer layers, the model generates a hidden state vector $h_{k-1}$ containing information about the entire preceding context. This vector $h_{k-1}$ is fed into:

    *   **Linear Layer (Output Embedding Layer):**  
  Transforms the hidden state vector $h_{k-1}$ into a vector of **logits** $z_k$ of dimension $V$, where $V$ is the model's vocabulary size. Each element $z_{k,j}$ of this vector corresponds to a "score" (unnormalized log probability) indicating how likely the $j$-th vocabulary token is to be the next token $t_k$.
    *   **Softmax Function:**  
  Converts the logits vector $z_k$ into a vector of **probabilities** $P_k$. Each element $P_{k,j}$ of this vector represents the probability $P(t_k = \text{token}_j | t_1, ..., t_{k-1})$, i.e., the probability that the $j$-th vocabulary token is the next one.
  
  $$P_{k,j} = \frac{\exp(z_{k,j})}{\sum_{i=1}^{V} \exp(z_{k,i})}$$
  
  The sum of all elements in vector $P_k$ equals 1, making it a valid probability distribution over the vocabulary.

### Training Process

Training the model involves adjusting its parameters (weights) so that it predicts the next token as accurately as possible in real text from a massive training corpus.

1. **Objective:**  
  For a given sequence of tokens $T = (t_1, t_2, ..., t_L)$ from the training corpus, the model must learn to maximize the probability of this sequence. In an autoregressive model, the probability of the sequence decomposes into a product of conditional probabilities (according to the chain rule of probability):

    $$P(T; \theta) = P(t_1, ..., t_L; \theta) = \prod_{i=1}^{L} P(t_i | t_1, ..., t_{i-1}; \theta)$$

    where $\theta$ denotes the model parameters. Maximizing this probability (or equivalently, maximizing the log-likelihood) is the goal of training.

2. **Loss Function:**  
  In practice, instead of maximizing likelihood, the **Negative Log-Likelihood (NLL)** is minimized. For a multi-class classification task (where classes are vocabulary tokens), NLL is equivalent to **Cross-Entropy Loss**. For a single prediction of the $i$-th token, it measures the divergence between the model's predicted probability distribution $P(\cdot | t_1, ..., t_{i-1}; \theta)$ and the "true" distribution, where all probability (equal to 1) is concentrated on the actual next token $t_i$. Losses are typically averaged over all tokens in a sequence and across all sequences in a **batch** of data.

3. **Optimization:**  
  The model parameters $\theta$ (billions or even trillions of weights in modern LLMs) are iteratively updated using stochastic gradient descent (SGD) algorithms, most commonly adaptive variants such as **Adam** or **AdamW**. On each training step:
    *   A batch of text sequences is sampled.
    *   The model makes next-token predictions for all positions in the batch.
    *   The average Cross-Entropy Loss is computed over the batch.
    *   Using the **backpropagation algorithm**, the gradient of the loss function with respect to all model parameters ($\nabla_\theta L$) is computed.
    *   The optimizer (Adam/AdamW) uses this gradient to update parameters $\theta$ in the direction that reduces loss: $\theta_{new} = \theta_{old} - \eta \nabla_\theta L$ (where $\eta$ is the learning rate, and the optimizer applies more sophisticated update rules).

### Mathematical Formalization

**1. Probability of the Next Token:**  
  The model $\pi_\theta$ with parameters $\theta$ computes the conditional probability of the next token $t_i$ given context $t_{<i} = (t_1, ..., t_{i-1})$:

$$P(t_i | t_{<i}; \theta) = \text{Softmax}(z_i)_k \quad \text{where } t_i = \text{token}_k$$

where:

*   $z_i = f(t_{<i}; \theta)$ - a vector of logits of dimension $V$ (vocabulary size), computed by the neural network (Transformer) based on context $t_{<i}$.
*   $\text{Softmax}(z_i)_k = \frac{\exp(z_{i,k})}{\sum_{j=1}^{V} \exp(z_{i,j})}$ - the probability of the $k$-th vocabulary token being next.
*   We select the probability from the Softmax vector corresponding to the index $k$ of the *true* next token $t_i$ from the training data.

**2. Loss Function (Cross-Entropy Loss):**  
  For one sequence $T = (t_1, ..., t_L)$, the total loss (negative log-likelihood) is computed as the sum of negative logarithms of the probabilities assigned to the true next tokens at each position:

$$L(T; \theta) = - \log P(T; \theta) = - \log \prod_{i=1}^{L} P(t_i | t_{<i}; \theta) = - \sum_{i=1}^{L} \log P(t_i | t_{<i}; \theta)$$

*   **Explanation:**
    *   $L(T; \theta)$: the loss function for sequence $T$ under model parameters $\theta$. Our goal is to minimize this value;
    *   $\sum_{i=1}^{L}$: summation over all prediction positions in the sequence (from the first to the last token). Often the first prediction (for $t_1$) is omitted since there is no context, or a special start-of-sequence token is used.
    *   $P(t_i | t_{<i}; \theta)$: the probability assigned by the model to the *true* token $t_i$ that actually followed context $t_{<i}$ in the training data. This value is taken from the output Softmax vector at step $i$.
    *   $\log(\cdot)$: natural logarithm.
    *   The minus sign: we minimize the *negative* log-likelihood, which is equivalent to maximizing the likelihood $P(T; \theta)$. If the model assigns a high probability (close to 1) to the true token $t_i$, then $\log P(\cdot)$ is close to 0, and the contribution to total loss is small. If the probability is low (close to 0), then $\log P(\cdot)$ is a large negative number, and $-\log P(\cdot)$ is a large positive number, increasing total loss and "punishing" the model, prompting it to adjust weights.
    *   **Connection to Cross-Entropy:** For a single prediction $i$, if the true next token $t_i$ is represented as a one-hot vector $y_i$ (where $y_{i,k}=1$ for the true token $k$, and 0 otherwise), and the predicted distribution is $p_i = \text{Softmax}(z_i)$, then the cross-entropy between $y_i$ and $p_i$ is $H(y_i, p_i) = - \sum_{j=1}^{V} y_{i,j} \log p_{i,j}$. Since $y_{i,j}$ equals 1 only for the true token $k$ and 0 otherwise, this sum simplifies to $- \log p_{i,k}$, which exactly matches the term $-\log P(t_i | t_{<i}; \theta)$ in the formula above.

**3. Loss on a Batch:**  
  In practice, training occurs on batches of data. If batch $B$ consists of $M$ sequences $T^{(1)}, ..., T^{(M)}$, the average loss over the batch is computed as:

$$L_{batch}(\theta) = \frac{1}{N_{tokens}} \sum_{j=1}^{M} \sum_{i=1}^{L_j} (-\log P(t_i^{(j)} | t_{<i}^{(j)}; \theta))$$

where 

$N_{tokens} = \sum_{j=1}^{M} L_j$ - the total number of tokens (predictions) in the batch.

*   **Explanation:**
    *   $L_{batch}(\theta)$: the average loss over batch $B$. It is this quantity whose gradient is used to update parameters $\theta$ on one optimization step;
    *   $\sum_{j=1}^{M} \sum_{i=1}^{L_j}$: summation of losses over all tokens in all sequences in the batch;
    *   $\frac{1}{N_{tokens}}$: normalization by the total number of tokens in the batch to obtain the average loss per token. This makes the loss value more stable and comparable across batches of different sizes;
    *   $L_j$: the length of the $j$-th sequence $T^{(j)}$;
    *   $t_i^{(j)}$: the $i$-th token of the $j$-th sequence;
    *   $t_{<i}^{(j)}$: the context for the $i$-th token of the $j$-th sequence.

### Significance and Limitations

**Significance of Next-Token Prediction:**

*   **Efficient Self-Supervised Learning:** The main advantage is the ability to train on massive volumes of unlabeled text. The task is defined by the structure of the text itself, eliminating the need for costly manual data annotation;
*   **Learning Deep Linguistic Structures:** To predict the next token well, the model is forced to implicitly learn grammar, syntax, semantics, style, and factual knowledge contained in the training data;
*   **Foundation for Generative Models:** This training strategy naturally leads to models capable of generating text autoregressively;
*   **Basis for Autoregressive LLMs:** Next-token prediction is the primary pre-training task for the entire class of autoregressive models (Decoder-only), such as GPT, Llama, PaLM, Mistral, etc. (In contrast to models like BERT, which use Masked Language Modeling).

**Limitations:**

*   **Local Optimization vs Global Coherence:** The model is optimized to predict the *locally* most probable next token, which does not guarantee global meaningfulness or factual accuracy of generated text ("hallucinations");
*   **Absence of Explicit Goal and Alignment:** The next-token prediction task does not directly teach the model to be useful, honest, or harmless. It merely mimics data statistics, requiring additional training stages (SFT, RLHF/DPO);
*   **Sensitivity to Data Quality:** The model reflects the properties of the training data, including errors, biases, and undesirable content;
*   **Computational Complexity:** Pre-training requires enormous computational resources and time.

</details> 

### **3.2 Core Mechanism: Formation of Attention-Based Recall Circuits During the Plateau**

**Key Idea:** Although external model metrics (loss, accuracy) do not improve during the plateau phase, important structural changes occur internally—the formation of *attention-based recall circuits* necessary for actual information retrieval.  

**Theoretical Foundation:**  
When performing Transformer tasks involving factual recall, the following pattern is typically observed:  
- **Early attention layers:** Aggregate information from multiple name tokens, forming a concentrated representation of the entity (typically at the position of the last name token);
- **Middle MLP layers:** Act as a *key-value store*, linking the name representation (as a query key) with the corresponding factual information (the value); 
- **Late attention layers:** Use context (e.g., which attribute to predict) to query the entity representation and retrieve stored facts for final prediction.  

**Hypothesis:** The plateau phase corresponds to the formation of these recall circuits, particularly the development of late attention layers' ability to correctly *route* information. Training stagnates because until these circuits are fully formed, attribute prediction errors cannot effectively propagate backward (via backpropagation) to the corresponding name representations or knowledge storage cells in the MLPs.

#### **3.2.1 Experimental Verification: Attention Patching**

**Experimental Logic:**

If recall circuits are indeed forming during the plateau, then "transplanting" attention patterns from a model that has already passed the plateau into a newly initialized model should significantly accelerate learning and even eliminate the plateau.  

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_04.png    )
> Attention circuits enabling recall form during the loss plateau. **(Left)** We developed an **"attention patching"** experiment: take a snapshot (checkpoint) of a reference model at a specific training stage. Use its attention patterns **in place of its own** in a modified model throughout its training. **(Center)** The more trained the reference model, the more beneficial its attention patterns are for the modified model—and the most critical changes occur precisely **during the plateau**. **Exception:** The earliest training stage shows the opposite trend. This correlates with the fact that during this period: name tokens (compared to other text containing attribute-type information) receive **less attention** when predicting the first attribute value token *(see right panel)*  

**Experimental Method:**

| Step | Action | Purpose |  
|------|--------|---------|  
| 1 | Train a **"reference model"** and save its states (checkpoints) at different stages (before/during/after plateau). | Obtain attention patterns of varying maturity. |  
| 2 | Initialize a new **"modified model"**. | Create a test model. |  
| 3 | During training of the modified model, **do not compute** its own attention patterns; instead, use frozen attention patterns from corresponding layers of the reference model's checkpoint. | Test the impact of ready-made attention patterns on training efficiency. |  
| 4 | Observe the training curve (change in attribute loss). | Evaluate the "quality" or "maturity" of patterns at different stages. |  

**Results:**  

- **Using attention patterns from after the plateau:** Training speed increases sharply, attribute losses drop rapidly—the plateau is effectively skipped;
- **Using attention patterns from during the plateau:** The closer the checkpoint is to the end of the plateau, the more noticeable the acceleration.
- **Using attention patterns from early training (Stage 1):** Performance is even worse than with random initialization. Reason: Early attention focuses on attribute-type tokens, not names, hindering the formation of "object-fact" associations.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">First Checkpoint:</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Language model training proceeds through a three-stage knowledge acquisition scheme: (1) initial statistical learning with rapid loss reduction, (2) a prolonged performance plateau proportional to the number of individuals being learned, and (3) knowledge emergence, when the model begins associating specific individuals with their attributes. This structure remains stable across various hyperparameters and architectures during pre-training.</p>
</div>

## **4. How Data Distribution Properties Contribute to Knowledge Acquisition**

**Primary Question:**  
We previously discussed the temporal dynamics of model learning, but how do internal properties of the training data, particularly its distribution, influence this process? In the real world, data is often imbalanced—that is, some entities/facts occur far more frequently than others. Does such imbalance accelerate learning, or does it impede it?

**Key Discovery:**  
The degree of data distribution balance significantly affects learning dynamics, forming a clear trade-off between plateau duration and the speed of knowledge acquisition at the final stage. By leveraging this property, overall training efficiency can be optimized through data scheduling strategies.

### **4.1 Dual Impact of Data Distribution Imbalance: The Trade-off Effect**

**Analysis and Quantitative Assessment:**

- **Plateau Length:** Primarily determined by a small number of the most frequently occurring individuals. When the model "internalizes" information about high-frequency individuals (FREQUENCY DISTRIBUTION), key mechanisms such as memory recall circuits can be pre-built, helping the model exit the plateau faster. Thus, some degree of imbalance can shorten plateau duration.

- **Knowledge Acquisition Speed:** After the plateau, the model must learn information about all individuals. At this stage, the bottleneck for learning speed lies with the least frequent individuals. The more balanced the data distribution, the higher the chance that rare individuals will be observed, and the higher the overall learning speed. Thus, a balanced distribution facilitates faster knowledge acquisition.

**Experimental Verification:** Introducing a Power Law to Control Imbalance  
To systematically investigate the impact of imbalance, the authors employed an inverse power law to control the sampling probability of the $i$-th individual in the dataset:

$$
P(i) = \frac{i^{-\alpha}}{\sum_{j=1}^{N} j^{-\alpha}},
$$

where $\alpha$ is a hyperparameter controlling the degree of imbalance:

- $\alpha = 0$: Uniform distribution, all individuals occur with equal probability.
- $\alpha > 1$: Zipf's Law distribution, highly imbalanced, where a few individuals occur extremely frequently and most occur very rarely.
- Intermediate $\alpha$ values: Various degrees of imbalance.

**Experimental Results** (fixed number of training steps, e.g., 16k):

**Conclusion:** Experimental results indicate an optimal degree of imbalance ($\alpha_{opt}  \approx 0.6 \sim 0.8$) that best balances accelerating the model's exit from the plateau with maintaining efficient subsequent learning. This optimal value is relatively stable across different total numbers of individuals $N$.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Second Checkpoint:</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Data distribution imbalance creates a trade-off in language model training: on one hand, high-frequency individuals accelerate exit from the plateau by forming necessary information-processing mechanisms; on the other hand, low-frequency individuals slow down the final knowledge acquisition stage. An optimal degree of imbalance ($\alpha_{opt}$) exists that achieves the best balance between these factors and maximizes overall training efficiency.</p>
</div>

### **4.2 Data Scheduling Strategy: Dynamic Optimization of Knowledge Acquisition**

**New Question:**  
Since data distribution requirements differ for the plateau and knowledge acquisition stages (plateau prefers imbalance, knowledge acquisition prefers balance), can we develop a dynamic strategy that combines the advantages of both approaches?

**Intuition and Solution:** Data Curriculum / Scheduling

1. **Early training stages (corresponding to plateau):** Use an imbalanced data distribution (or only a subset of high-frequency individuals) to rapidly build key mechanisms and shorten the plateau.
2. **Late training stages (corresponding to knowledge acquisition stage):** Transition to a balanced data distribution to ensure all individuals are learned thoroughly.

**Concrete Implementation:** Warm-up Strategy

- **Warm-up Phase:** Training begins with a subset of individuals (indiv_warmup) for epochs_warmup epochs. This subset creates natural imbalance.
- **Main Training Phase:** Transition to all individuals using uniform sampling to continue training.

**Experimental Results:**

Compared to constant use of uniform distribution or optimal fixed $\alpha$, this dynamic warm-up strategy significantly increases the total amount of knowledge acquired by the model (resulting in lower Attribute Loss), especially when the number of individuals is large ($N$ large).

**Significance:** This demonstrates a rare and concrete example of how "Data Curriculum" strategies can effectively enhance performance in self-supervised learning scenarios.

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_05.png  )
> Data distribution properties can accelerate knowledge acquisition. (left) Plateau length is significantly reduced when some individuals occur more frequently than others, which in this case is achieved by increasing $\alpha$. (center) Thus, it is beneficial to train the model on more imbalanced distributions, especially when the number of training steps is limited or the total number of individuals increases. (right) This strategy increases the final amount of knowledge contained in the network (purple line vs. gray). Dynamic adaptation of data distribution yields an even greater effect (blue line).

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Third Checkpoint:</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Dynamic data scheduling (Data Curriculum) optimizes language model training by using different distributions at different stages: imbalanced distribution during the plateau for rapid formation of key mechanisms, and balanced distribution during knowledge acquisition for uniform learning of all individuals. The "warm-up" strategy, starting with a subset of frequent individuals and transitioning to the full set, significantly increases the total volume of acquired knowledge, especially with a large number of individuals.</p>
</div>

## **5. Hallucinations as a Barrier to Integrating New Knowledge Post-Training**

**Primary Question:**

In practice, adding new information to an already pre-trained large language model (LLM), for example, via fine-tuning, is often ineffective. The model either struggles to "internalize" new data or significantly "forgets" previously learned knowledge. Why does this happen?

**Key Discovery**

This chapter demonstrates that the knowledge acquisition process is accompanied by the emergence of "hallucinations"—a phenomenon where the model makes confident but false assertions about unfamiliar objects. The presence of hallucinations and the fragility of the model's associative memory create significant difficulties when attempting to integrate new knowledge via fine-tuning.

### **5.1 Symbiosis of Knowledge and Hallucinations**

#### **Observation 1 (Problem Setup)**
How does the model respond to entities it has never seen (e.g., held-out individuals from the test set)?

**Definition of Hallucination:**  
A hallucination is a phenomenon where the model makes **overly confident**, yet **false**, factual predictions about unfamiliar objects.

**Experimental Results:**
- **Synchronous Emergence:**  
  As soon as the model begins accurately reproducing knowledge about entities from the training set (Attribute Accuracy > 0, Attribute Loss < No Knowledge Baseline), its errors (Attribute Loss) with respect to unfamiliar objects begin to significantly exceed the baseline level. This indicates the presence of hallucinations.
  
- **Difference in Confidence:**  
  Despite hallucinations, the model's confidence in its erroneous predictions (measured via predicted token probabilities or distribution entropy) is typically lower than its confidence in correct predictions for training-set objects. However, even this "lower" confidence remains above a reasonable threshold.

- **Potential Link:**  
  The concurrent emergence of hallucinations and knowledge suggests that hallucinations may be an **inevitable side effect** of current model architectures and their training mechanisms.

### **5.2 Catastrophic Forgetting of Old Knowledge During Fine-Tuning**

#### **Observation 2 (Core Problem)**
What happens if we fine-tune a pre-trained model on data about new characters and their biographies?

**Experimental Observations:**

| Stage | Behavior Regarding Old Knowledge (Pre-trained Characters) | Behavior Regarding New Knowledge (Fine-tuned Characters) | Key Phenomena |
|------|---------------------------------------------------------------|------------------------------------------------------------|------------------|
| **Start of Fine-tuning (first hundreds of steps)** | - Attribute losses sharply increase<br>- Attribute accuracy sharply decreases | - Attribute losses slowly decrease<br>- Attribute accuracy slowly increases | Rapid and massive forgetting of old knowledge while new knowledge is still unlearned. |
| **Late Stages of Fine-tuning** | Performance may partially recover (especially with Replay). | Performance continues to improve. | New knowledge is gradually internalized, while old knowledge either stabilizes or slowly recovers. |
| **Adding Replay** | Significant drops in accuracy and rises in losses are still noticeable at the start. | - | Replay helps partially recover old knowledge at late stages but does not prevent catastrophic forgetting at the start. |

#### Investigation of Forgetting Causes

**Hypothesis 1: Disruption of Attention Patterns?**  
- **Logic:**  
  Introducing new characters may disrupt established attention patterns responsible for recalling previously learned data.

- **Result:**  
  Attention patterns remain stable throughout the fine-tuning process, refuting this hypothesis.

**Hypothesis 2: Disruption of Associative Memory in Feed-Forward Networks (FFN)?**  
- **Logic:**   
  Feed-forward layers (FFN/MLP) are viewed as key-value stores for knowledge. Adding new "keys" (names of new characters) and "values" (their attributes) may interfere with or overwrite previously stored pairs.  

- **Result:**  
  In a simplified model, rapid forgetting of old keys and values was observed at early fine-tuning stages. This hypothesis is confirmed.

Thus, the study reveals that hallucinations and catastrophic forgetting arise from the internal organization of models and the specifics of their training. These phenomena require further investigation to improve models' ability to learn efficiently without losing previously accumulated knowledge.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Fourth Checkpoint:</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Hallucinations and catastrophic forgetting arise from the internal organization of language models, where feed-forward networks (FFN) serve as associative memory. During fine-tuning, new knowledge overwrites previously learned "key-value" pairs in the FFN, leading to loss of old data. Hallucinations manifest as false, confident assertions about unfamiliar objects, emerging simultaneously with knowledge acquisition. These phenomena indicate the need to reconsider architectural designs and training methods to ensure stable integration of information without catastrophic forgetting.</p>
</div>

## **6. Discussion**

### **6.1 A New Perspective on Language Model Learning Dynamics**

#### Conclusions and Their Significance:
- **Data Distribution > Model Size?**  
  Compared to simply increasing model scale, characteristics of the training data distribution may exert a greater influence on learning dynamics (particularly on the duration of transitional phases).  

- **Possible Sources of "Emergent Abilities"?**  
  The so-called "emergence" may partly be explained by the fact that as model and data scale increase, training time also increases. This allows the model to overcome long plateaus for certain tasks and "suddenly" exhibit new abilities.

#### Recommendations for Training Strategy:
- **Use Synthetic Data Early?**  
  Given that data presented before the plateau contributes minimally to the final model (since the corresponding mechanisms are not yet formed), using computationally cheaper synthetic data for "warm-up" or mechanism formation may be a more efficient strategy.  

- **Potential of Data Schedulers:**  
  Developing adaptive data schedulers capable of dynamically adjusting data distribution (e.g., reducing data diversity during the plateau to accelerate mechanism formation) represents a highly promising direction for improving training speed and efficiency.

### **6.2 Conclusions for Learning Dynamics in Universal Neural Networks**

#### Order of Mechanism Formation:
- The study observes a phenomenon where "attention and recall circuits" form before "associative memory in feed-forward layers." This order may have universal significance.  

- **Hypothesis:**  
  Forming efficient routing/selection mechanisms (e.g., attention) enhances the correlation between input data and error signals, providing clearer and more effective learning signals for subsequent content storage mechanisms (e.g., associative memory in MLP).

- **Connection to Grokking and Similar Phenomena:**  
  This order of mechanism formation may be linked to phenomena such as "Grokking" (initial overfitting followed by generalization) or the process where a model first finds a sufficient but suboptimal solution (e.g., relying solely on local statistics) and then transitions to more generalizable solutions due to the formation of more optimal mechanisms (e.g., global recall circuits) and regularization.

- **Value of Analytical Methods:**  
  Separating the functions of attention (token-mixing) from other computations (e.g., knowledge storage in FFN) proved to be a powerful tool for understanding Transformer learning dynamics. This approach is crucial for future research into internal mechanisms of neural networks.

### **6.3 Data Imbalance, Training Efficiency, and Developmental Psychology**

#### Key Conclusions: Accelerating Mechanism Formation Through Imbalance
- The study precisely analyzed how imbalance (non-uniformity) in training data helps the model overcome learning plateaus faster by amplifying signals and accelerating identification of key relationships.  

- **Trade-off:**  
  However, this acceleration may come at the cost of reduced learning quality on rare data and the model's generalization ability (particularly without subsequent uniform learning stages).

#### Connection to Cognitive Science and Developmental Psychology:
- **Implicit Curriculum:**  
  The dynamic data scheduling strategy proposed in the article (first imbalanced/simple, then balanced/complex) remarkably resembles infant learning models. Due to limited activity range in early life and frequent exposure to familiar individuals and objects, infants naturally progress from simple, repetitive inputs to gradually interacting with a richer and more diverse environment. This "bottom-up" formed curriculum is considered a key factor enabling early efficient learning.  

- **Repetition and Generalization:**  
  As this study shows, early repetition of a small number of examples facilitates rapid formation of core representations and connections, while subsequent exposure to diversity is necessary for achieving reliable generalization.  

- **Potential Contribution:**  
  Quantitative analysis of the impact of data distribution dynamics may lay the foundation for a more sophisticated statistical theory of development.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;">
<p style="margin: 0; font-weight: bold; color: #2c3e50;">Fifth Checkpoint:</p>
<p style="margin: 8px 0 0 0; color: #2c3e50;">Language model learning dynamics depend on data delivery strategy:
1. Imbalanced data accelerates mechanism formation (e.g., attention) but requires subsequent "refinement" on diverse data for generalization.
2. The order of components (attention → associative memory) explains phenomena like Grokking and underscores the role of adaptive learning.
3. Synthetic data and schedulers optimize the process: cheap data at the start + dynamic distribution adjustment.</p>
</div>

## **Conclusion**

This study provides a comprehensive foundation for understanding how language models learn, store, and retrieve factual knowledge. Identifying the three-phase learning process and the involved neural mechanisms offers valuable insights into both the capabilities and limitations of current language models.

The findings suggest several directions for future research, including:

1. Developing more efficient training curricula based on the identified learning dynamics;
2. Designing architectural modifications to better separate knowledge acquisition from hallucination development;
3. Creating fine-tuning approaches that can incorporate new knowledge with minimal distortion of existing memories;
4. Investigating the relationships between model scale, dataset size, and plateau duration for larger models.

Understanding these fundamental learning principles is crucial for developing more capable, efficient, and truthful language models that can serve as reliable interfaces for human knowledge. This study represents a significant step toward mechanistic explanations of language model behavior, moving beyond "black-box" evaluations and delving deeper into understanding how these increasingly important systems learn and operate.