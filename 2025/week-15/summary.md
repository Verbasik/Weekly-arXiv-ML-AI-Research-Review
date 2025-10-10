# **How Do LLMs Learn Facts and Why Do They Hallucinate?**

## Table of Contents
1. [Introduction](#introduction)
2. [The Three-Phase Learning Process](#the-three-phase-learning-process)
3. [Neural Mechanisms Underlying Factual Knowledge](#neural-mechanisms-underlying-factual-knowledge)
4. [The Impact of Data Distribution](#the-impact-of-data-distribution)
5. [Data Curriculum Strategies](#data-curriculum-strategies)
6. [Hallucinations and Knowledge Distortion](#hallucinations-and-knowledge-distortion)
7. [Fine-Tuning Challenges](#fine-tuning-challenges)
8. [Practical Implications](#practical-implications)
9. [Conclusion](#conclusion)

## Introduction
Large language models (LLMs) serve as increasingly vital interfaces to human knowledge, yet our understanding of how they acquire, represent, and recall factual information remains limited. In the paper _"How Do Language Models Learn Facts? Dynamics, Curricula, and Hallucinations,"_ researchers from Google DeepMind and ETH Zürich provide an in-depth analysis of the learning dynamics that occur as language models acquire factual knowledge.

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_02.png  )
> Figure_2. Knowledge acquisition occurs in three stages. (Left) On a very short first stage, the model learns the overall statistics of attribute values. On the second stage, performance plateaus at a level achievable by an ideal model without knowledge of individual entities (this corresponds to the "no-knowledge" baseline and nearly zero recognition accuracy). The duration of this plateau is nearly proportional to the number of individuals (right). Finally, the model learns associations between subjects and attributes: knowledge emerges as training continues (center). Results are averaged over 5 runs (± standard deviation).

The study employs synthetic biography datasets to systematically investigate how models learn to associate individual entities with their attributes. This approach provides precise control over data distribution and enables efficient measurement of knowledge acquisition throughout training. The analysis reveals a compelling three-phase learning process with significant implications for model training and reliability.

## **The Three-Phase Learning Process**

Researchers identify a distinct three-phase process through which language models acquire factual knowledge:

1. **Initial Language Understanding**: During early training stages, the model learns the overall statistics of attribute values but lacks knowledge specific to individual entities;
2. **Performance Plateau**: Model performance plateaus at a level achievable by a model without knowledge specific to individual entities. This plateau phase represents a critical period during which the model constructs the neural circuits necessary for subsequent knowledge acquisition;
3. **Knowledge Emergence**: After the plateau, the model rapidly develops the ability to link individuals with their specific attributes, leading to a sharp improvement in factual recall performance.

A key finding is that the duration of the plateau phase is nearly proportional to the number of individuals in the dataset, following this relationship:

```
Plateau_Duration ≈ 0.43 × (Number_of_Individuals)^0.81
```

This scaling law indicates that as the number of individuals increases, models require a disproportionate amount of additional training steps to transition from general language understanding to specific factual knowledge. This dependency has substantial implications for training large-scale models that must memorize vast volumes of factual information.

## **Neural Mechanisms Underlying Factual Knowledge**

To understand the neural mechanisms underlying factual recall, researchers used attention patching techniques to isolate components responsible for storing and retrieving knowledge.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_03.png  )
> **Attention circuits enabling recall form during the loss plateau.** (left) We develop an attention patching experiment where we take a snapshot of a reference model at a specific training stage and use its attention patterns in place of the modified model's own patterns throughout its training. (center) The more trained the reference model, the more beneficial its attention patterns are for the modified model, and these changes primarily occur during the plateau. However, the very beginning of training is an exception to this trend. This correlates with the fact that during this stage, name tokens (compared to other text containing attribute-type information) receive less attention when predicting the first attribute value token (see right panel).  

Results show that factual knowledge is distributed across several model components:

1. **Early Attention Layers**: Process and aggregate name tokens to form a query;
2. **Middle MLP Layers**: Act as associative memory, storing information about all attribute values;
3. **Final Attention Layers**: Retrieve the specific attribute for the queried individual.

This distributed representation of knowledge helps explain why models require substantial training before they can effectively store and retrieve information pertaining to specific individuals. The model's attention patterns evolve during training, with later attention layers becoming increasingly specialized for knowledge retrieval as the model exits the plateau phase.

Researchers discovered that retrieval-based attention circuits begin developing during the plateau phase, even before the model demonstrates improved factual recall performance. This suggests that the plateau is a critical period of circuit formation, not a period of stagnation.

## **The Impact of Data Distribution**

The study demonstrates that the distribution of individuals in training data significantly affects how quickly and effectively models internalize factual knowledge.

![Figure H](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_06.png  )
> Learning curves for "celebrity" distributions, obtained after 8k training steps and 64k individuals. On the left graph, the celebrity weight is set to 8, and on the right graph, the number of celebrities is set to 4k.

When training data contains imbalanced individual frequencies (e.g., Zipf's distribution or a "celebrity" distribution where some individuals occur more frequently), several effects emerge:

1. The plateau phase is shortened for frequently occurring individuals;
2. The model internalizes facts about high-frequency individuals earlier than those about low-frequency ones;
3. However, high-frequency individuals can lead to overfitting, where the model performs well on training data but poorly on novel examples.

This analysis reveals a fundamental trade-off in factual knowledge acquisition: accelerating learning through imbalanced data distribution may improve efficiency but potentially reduce generalization to new facts or individuals.

## **Data Curriculum Strategies**

Building on insights from the effects of data distribution, researchers investigated various data curriculum strategies to optimize factual knowledge acquisition.

![Figure I](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_07.png  )
> Learning curves for "warm-up" distributions, obtained after 8k training steps and 64k individuals. On the left graph, the warm-up steps are set to 1.5k, and on the right graph, the number of warm-up individuals is set to 8k.

A particularly effective approach is the "warm-up" curriculum, in which:

1. The model is initially trained on a small subset of high-frequency individuals;
2. After this warm-up period, training continues on the full dataset with uniform distribution.

This two-stage curriculum significantly accelerates the model's transition through the plateau phase while maintaining strong generalization performance. Researchers systematically explored various combinations of warm-up individuals and warm-up steps, identifying optimal configurations that balance training efficiency with final model performance.

Data shows that the best curricula include a moderate number of warm-up individuals (approximately 8–16 thousand individuals for a 64-thousand-individual dataset) and a moderate number of warm-up steps (approximately 1–2 thousand steps). Too few or too many warm-up individuals/steps lead to suboptimal performance.

![Figure K](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_08.png  )
> Visualization of how final performance changes as hyperparameters of the initial distribution (warm-up) are varied. The total number of individuals is fixed at 64 thousand, and these graphs correspond to Figure J (center right and right).

## **Hallucinations and Knowledge Distortion**

One of the most concerning aspects of language models is their propensity for hallucinations—generating incorrect information with high confidence. The study provides critical insights into this phenomenon.

![Figure M](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_09.png  )
> The model begins to hallucinate during training. The blue line corresponds to the model's performance on familiar individuals (as in other parts of the paper), and the purple line corresponds to its performance on 16 thousand held-out individuals. (left) Attribute loss, (middle left) knowledge accuracy, (middle right) average probability of the most likely predicted token (for attribute values), and (right) average entropy of the predictive distribution (for attribute values). Overall, the model is less confident in its hallucinations than in its justified predictions.

Researchers discovered that hallucinations emerge simultaneously with knowledge acquisition during the transition from the plateau phase. When presented with unknown individuals (not seen during training), the model follows a distinct pattern:

1. Initially, the model correctly expresses uncertainty, producing predictions with low confidence;
2. As the model learns to associate known individuals with attributes, it simultaneously develops a tendency to confidently generate incorrect attributes for unknown individuals;
3. This manifests as higher maximum predicted probabilities and lower predictive distribution entropies for unknown individuals, indicating increased (but misplaced) confidence.

This finding suggests that the neural mechanisms responsible for factual recall and hallucinations are inextricably linked, presenting a fundamental challenge for developing truthful language models.

## **Fine-Tuning Challenges**

The study also examines the challenges of incorporating new knowledge via fine-tuning, revealing serious limitations in how models adapt to new information.

![Figure N](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_10.png  )
> Evolution of model performance on the pre-training distribution (left and middle-left panels) and on the fine-tuning distribution (middle-right and right panels) as fine-tuning progresses. In the first row, pre-training data repetition is not used during fine-tuning, and we vary the number of individuals for fine-tuning. In the second row, obtained for 4 thousand fine-tuning individuals, we introduce some repetition and vary its weight (weight corresponds to how much more likely sampling an individual from pre-training is compared to one from the fine-tuning set). The data presented here are the same as in Figure 5 (middle and right panels), but are plotted as a function of time and include accuracy.

When a trained model is fine-tuned on new individuals, several problems arise:

1. **Knowledge Distortion**: Fine-tuning on new individuals rapidly distorts existing memories, causing performance on pre-trained individuals to deteriorate quickly;
2. **Vulnerability of Feed-Forward Layers**: Associative memories stored in feed-forward layers are especially susceptible to distortion during fine-tuning;
3. **Stability of Attention Patterns**: Attention patterns remain relatively stable during fine-tuning, suggesting that the knowledge retrieval mechanism is preserved while the actual memory storage is distorted.

This memory disruption presents a serious challenge for language models that must be continuously updated with new information without losing existing knowledge. The trade-off between learning new facts and preserving old ones appears fundamental to current neural network architectures.

## **Practical Implications**

The findings have several important practical implications for LLM development:

1. **Training Efficiency**: Data curriculum strategies, particularly warm-up approaches, can significantly reduce training time and computational requirements for large-scale models;
2. **Model Scaling**: The plateau scaling law provides guidance for estimating the computational resources required as models are trained on increasingly larger datasets;
3. **Mitigating Hallucinations**: Understanding the link between knowledge acquisition and the development of hallucinations can help researchers develop targeted interventions to reduce false outputs;
4. **Continual Learning**: The identified fine-tuning challenges suggest that alternative approaches—such as sparse fine-tuning methods or architectural modifications—may be needed for effective knowledge updating;
5. **Model Evaluation**: The three-phase learning process underscores the importance of evaluating models throughout training, not just at the final point, as performance can change dramatically during phase transitions.

## **Conclusion**

This study provides a foundation for understanding how language models learn, store, and retrieve factual knowledge. Identifying the three-phase learning process and the involved neural mechanisms offers valuable insights into both the capabilities and limitations of modern language models.

The findings suggest several directions for future research, including:

1. Developing more effective curricula based on the identified learning dynamics;
2. Designing architectural modifications to better separate knowledge acquisition from the development of hallucinations;
3. Creating fine-tuning approaches that can incorporate new knowledge with minimal distortion of existing memories;
4. Investigating the relationships between model scale, dataset size, and plateau duration for larger models.

Understanding these fundamental learning principles is crucial for developing more capable, efficient, and truthful language models that can serve as reliable interfaces for human knowledge. This study represents a significant step toward mechanistic explanations of language model behavior, moving beyond "black-box" evaluations and deepening our understanding of how these increasingly important systems learn and operate.