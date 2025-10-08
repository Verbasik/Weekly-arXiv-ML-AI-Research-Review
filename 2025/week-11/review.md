# Scaling Laws of Distillation

**Recommendation for Readers:**

Before diving into the details, I highly recommend reading two excellent articles by a Yandex engineer ([Article 1](https://habr.com/ru/companies/yandex/articles/801119/), [Article 2](https://habr.com/ru/companies/yandex/articles/878230/)). They provide an excellent explanation of the principles of distillation, its application in industrial tasks, and key practical aspects. This is the ideal starting point for those new to the topic.

**However**, if, like me, you seek *deep understanding*—this may not be sufficient. In this review, we go further:

1.  **Mathematical Formalization**: We delve deeper into the equations underpinning distillation, including the temperature-scaled loss function, distribution optimization, and scaling laws from Apple's work.
2.  **Code Examples**: We show how to implement distillation in practice—from simple PyTorch models to fine-tuning hyperparameters.
3.  **Research Nuances**: We answer questions left out of introductory materials. For instance, why is an "overly smart teacher" detrimental to the student, and how can we mathematically justify the optimal size ratio between them?

**Who is this for?**

If you want to not just use distillation "out of the box," but *understand how and why it works*—this breakdown is for you. We'll look "under the hood" of the methods so you can apply them consciously in your own projects.

<details>
    <summary><em><strong>Quick Overview 🎓</strong></em></summary>

## Introduction

**Goal of this review:** To provide the reader with a comprehensive understanding of the principles of Knowledge Distillation, its mathematical formalization, practical implementation (with code examples), and, most importantly, the latest research on scaling laws that determine the method's effectiveness depending on the sizes of the teacher and student models and the volume of data.

**Who is this review for?** For those who aim not merely to apply distillation as a ready-made tool, but to deeply understand its mechanisms and use them consciously in their projects.

## Part 1: Knowledge Distillation

### Core Concept

Knowledge Distillation is defined as a method for training student models (typically smaller and less complex) by transferring "knowledge" from a pre-trained teacher model (typically larger and more complex).

> "Knowledge Distillation is a method for training student models (typically smaller and less complex) by transferring 'knowledge' from a pre-trained teacher model (typically larger and more complex)."

The core idea is that the teacher, possessing greater capacity and trained on a large volume of data, can transfer not only "hard" predictions (e.g., the object class) but also richer information about the class probability distribution.

### Teacher and Student Models

In the distillation process, two primary models participate:

-   **Teacher:** A large, pre-trained model, an "expert" at solving the task. Mathematically represented as a function $p(y|x)$, outputting a probability distribution $p$ over classes $y$ for input data $x$.
-   **Student:** A smaller, simpler model to be trained to mimic the teacher's behavior. Mathematically represented as a function $q_{\theta}(y|x)$, where $\theta$ are the parameters to be optimized.

### Loss Function

The goal of distillation is to minimize the difference between the teacher's and student's predictions, formalized by a loss function $L(p(y|x), q_{\theta}(y|x))$. Training involves finding the optimal student parameters $\theta$ that minimize this loss:

$L(p(y|x), q_{\theta}(y|x)) \rightarrow \min_{\theta}$

Two main approaches are considered: hard-label distillation and soft-label distillation.

#### I. Hard-label Distillation

**Concept:** The teacher generates "hard" labels (the class with the highest probability) for the training set, and the student is trained on these synthetic labels using standard methods.

**Example for GPT models:** Describes the process where a large GPT model (teacher) generates token sequences as "hard" labels for given input texts, and a small GPT model (student) is trained to predict these sequences by minimizing cross-entropy.

**Mathematical Formalization:**

-   Generation of "hard" labels by the teacher: $y^{(n)} = \arg\max_{y} p(y|x^{(n)})$ (for classification) or generation of the sequence of most probable tokens (for language models).
-   Training the student on "hard" labels: Minimization of cross-entropy between the student's predictions and the teacher's "hard" labels.

$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log q_{\theta}(y_t^{(n)}|y_{<t}^{(n)})$

**Advantages:** Simplicity of implementation and understanding, use of standard training methods.

**Disadvantages:** Loss of information contained in the teacher's probability distribution (probabilities of other classes and "soft" relationships between classes).

#### II. Soft-label Distillation

**Concept:** Use of the full probability distribution predicted by the teacher (after applying "temperature scaling") as "soft" labels for training the student.

> "Soft-label distillation, proposed by Hinton et al. in their seminal paper 'Distilling the Knowledge in a Neural Network' (2015), is a more sophisticated knowledge distillation method. Unlike hard-label distillation, this approach uses not only the 'hard' labels but the full probability distribution predicted by the teacher as 'soft' labels."

**Temperature Scaling:** Describes how dividing the model's logits by a temperature parameter $T > 1$ makes the probability distribution more "soft" and informative.

**Example for GPT models:** A large GPT model generates probabilities for all possible next tokens, which are then "softened" using temperature. The small student is trained to reproduce this probability distribution using KL divergence (or cross-entropy) between the teacher's and student's distributions (both also "softened" with the same temperature).

**Mathematical Formalization:**

-   Teacher's "soft" labels with temperature $T$: $p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$
-   Similarly for the student: $q_i^T = \frac{\exp(z_i^q/T)}{\sum_j \exp(z_j^q/T)}$
-   Loss function for Soft-label Distillation: $L_{soft} = T^2 \cdot \text{KL}(p^T || q^T) = T^2 \cdot \sum_i p_i^T \log\frac{p_i^T}{q_i^T}$
-   Combined loss function (often used): $L = \alpha \cdot L_{soft} + (1-\alpha) \cdot L_{hard}$

**Practical Implementation:** Code snippets (borrowed from the DistillKit repository) are provided for configuring distillation, preparing models, implementing the soft-label loss function, handling different vocabulary sizes, and creating a custom trainer.

**Advantages:** More complete transfer of knowledge ("dark knowledge"), better results, improved generalization, control via temperature.

**Disadvantages:** Higher computational cost (especially for large vocabularies), implementation complexity (requires access to teacher's logits), need for hyperparameter tuning, dependence on teacher quality.

**Comparison of Hard-label and Soft-label Distillation:** A comparison table is provided across various aspects.

## Part 2: Scaling Laws of Distillation

**Motivation for Research:**

-   Lack of systematic studies on scaling laws in the context of distillation.
-   Problem of inference cost for large models.
-   Uncertainty regarding optimal distillation methods and allocation of computational resources.

**Scaling Law of Distillation:**

It is emphasized that traditional scaling laws focus on training large models from scratch, while this research seeks patterns in distillation efficiency. Experiments were conducted with models ranging from 143 million to 12.6 billion parameters and data volumes up to 512 billion tokens.

A table is provided with notation for variables used in the paper, including number of parameters ($N_S, N_T$), number of tokens ($D_S, D_T$), token-to-parameter ratio ($M$), and cross-entropy ($L, L_T, L_S, \tilde{L}_S$). A detailed explanation of the cross-entropy metric and Chinchilla's rule for the optimal token-to-parameter ratio ($M^* \approx 20$) is given.

**Formalization of the Distillation Scaling Law:**

The central formula of the law is presented:

$L_S(N_S, D_S, L_T) = L_T + \frac{1}{L_{c_0}^T} \left( 1 + \left( \frac{L_T}{\tilde{L}_S^{d_1}} \right)^{1/f_1} \right)^{-c_1f_1} \left( \frac{A}{N_S^{\alpha'}} + \frac{B}{D_S^{\beta'}} \right)^{\gamma'}$

and the variables ($L_S, L_T, N_S, D_S, \tilde{L}_S$) and empirical coefficients are explained.

**Physical Meaning of the Formula:**

-   The student cannot be better than the teacher ($L_S \geq L_T$).
-   Distillation efficiency is higher when the student's potential performance ($\tilde{L}_S$) is closer to the teacher's performance ($L_T$).
-   With a fixed teacher, the distillation scaling law does not surpass the standard scaling law.

**Practical Application:** Optimal allocation of computational resources and prediction of distillation effectiveness.

### Mixing Coefficients in Knowledge Distillation

The practical implementation of distillation and the importance of managing the balance between imitating the teacher and the student's independent learning through mixing coefficients in the loss function are discussed. Formulas are provided for KL divergence ($\mathcal{L}_{\text{KD}}$) and the combined loss function ($\mathcal{L}_S$), including next-token prediction loss ($\mathcal{L}_{\textrm{NTP}}$), distillation loss ($\mathcal{L}_{\textrm{KD}}$), and regularization Z-loss ($\mathcal{L}_Z$), along with mixing coefficients $\lambda$ and $\lambda_Z$.

### Experimental Determination of Optimal Distillation Parameters

Experiments are described to determine the impact of distillation parameters ($\lambda$ and $\tau$) on the effectiveness of the scaling law. It is noted that in "pure distillation" mode ($\lambda = 1$) with $\tau = 1$, results are often comparable to optimal values. The dependence of optimal parameters on the sizes of the teacher and student models is emphasized.

### Experiment with Fixed Teacher and Different Students

The effect of student model size and distillation data volume is studied with a fixed teacher. It is observed that, with high computational power, a larger student size leads to lower loss, especially with a larger teacher. With low computational power, a U-shaped dependency is observed. The possibility of a student outperforming the teacher in special cases (presumably with an undertrained teacher) is mentioned.

### Experiment with Fixed Student and Different Teachers

The effect of teacher model size is studied with a fixed student and distillation data volume. Results show that the larger the teacher parameters, the lower the student's cross-entropy, indicating the need for the teacher's performance to match the student's capabilities for optimal distillation.

### Distillation vs. Supervised Learning

The effectiveness of distillation and supervised learning is compared under fixed computational resources. Supervised learning outperforms distillation with sufficient computation or data for the student. Distillation has advantages with moderate data budgets but is outperformed by supervised learning with large data volumes. Overall, distillation is more efficient under limited computational resources.

### Choosing the Teacher Model

Factors influencing teacher model selection are discussed, such as signal strength (teacher's cross-entropy) and increased computational costs from using a larger teacher. The optimal teacher loss decreases as a power law with increasing student computational resources, and the optimal teacher size is almost always linearly proportional to the student size. With limited resources, choosing a smaller teacher can reduce inference costs while providing effective training signals.

### Calculate Optimal Distillation

The task of determining the optimal way to create a student model of a desired size with minimal cross-entropy under a given computational budget is considered, including choosing the optimal data volume for student training, teacher model size, and data for teacher training. It is noted that supervised learning corresponds to the best distillation option with sufficient computational budget. If the goal is to create the best model of a given size without an existing teacher, supervised learning is preferable. Smaller models are more likely to benefit from supervised pre-training, while larger models benefit from distillation under large computational budgets. The optimal teacher size first increases until it is slightly larger than the student, then stabilizes, as overtraining the teacher becomes more efficient with increasing student tokens.

### Key Research Findings

-   Ability to predict student performance using the developed scaling law.
-   Influence of teacher size ($N_T$) and training data ($D_T$) on teacher cross-entropy ($L_T$), which affects the student.
-   Discovery of the "capacity gap" phenomenon, where an overly strong teacher can lead to a worse student. The critical factor is the gap in learning capacity, not just relative size.
-   Empirical confirmation of the U-shaped dependency of student error on teacher size with a fixed student size.

### Practical Recommendations

Distillation is more effective than supervised learning if:

-   The total number of computations/tokens for the student does not exceed a threshold related to its size according to the new scaling law.
-   The teacher model already exists or its training has applications beyond a single distillation.
-   If both processes (teacher and student training) have sufficient resources, supervised learning can achieve lower cross-entropy.

### Summary:

-   The distillation scaling law (formula provided) is presented, describing the dependence of student quality ($L_S$) on its size ($N_S$), training data ($D_S$), and teacher quality ($L_T$, dependent on $N_T$ and $D_T$). The fundamental principle: $L_S \geq L_T$.
-   The key discovery is the "capacity gap" between teacher and student, encompassing differences in hypothesis space and optimization capability. This explains the U-shaped dependency of student quality on teacher size. The scaling law (Equation 8) allows estimation of how teacher and student characteristics affect final performance. Experiments statistically confirm the U-shaped dependency and help identify the optimal balance between teacher and student sizes.
-   Distillation is more effective than standard learning under limited computational resources for the student (below a certain threshold) and when a pre-trained teacher model is available. If the teacher can form representations utilizing complex multidimensional dependencies, a smaller student may be physically unable to reproduce them. The scaling law helps optimally choose teacher size, saving computational resources.

</details>

## Part 1: Knowledge Distillation:

**Knowledge Distillation** is a method for training student models (typically smaller and less complex) by transferring "knowledge" from a pre-trained teacher model (typically larger and more complex).

The core idea is that the teacher model, possessing greater capacity and trained on a large volume of data, can transfer not only its "hard" predictions (e.g., the object class) but also richer information about the class probability distribution, which the student model can use for more efficient learning.

### **Teacher and Student Models:**

In the Knowledge Distillation paradigm, two primary models participate:

*   **Teacher:** This is a large, pre-trained model considered an "expert" at solving a specific task. The teacher has already achieved high accuracy and possesses "knowledge" we wish to transfer to the student. Mathematically, the teacher is represented as a function $p(y|x)$, which for input data $x$ outputs a probability distribution $p$ over classes $y$.
*   **Student:** This is a smaller, simpler model we aim to train. The student's goal is to learn to mimic the teacher's behavior to achieve comparable performance while being more efficient in terms of computational resources, memory, or inference time. The student is represented as a function $q_{\theta}(y|x)$, where $\theta$ are the model parameters we optimize during training.

**Loss Function in Knowledge Distillation:**

The overall goal of Knowledge Distillation is to minimize the difference between the teacher's and student's predictions. This is formalized through a loss function $L$, which depends on the teacher's predictions $p(y|x)$ and the student's predictions $q_{\theta}(y|x)$.

The training process involves finding the optimal student parameters $\theta$ that minimize this loss function:

$L(p(y|x), q_{\theta}(y|x)) \rightarrow \min_{\theta}$

This is a general expression, and the specific form of the loss function and distillation approach define the different methods. We consider two main approaches: hard-label and soft-label distillation.

## I. Hard-label Distillation

**Concept:**

Hard-label distillation is the simplest and most intuitive approach. In this method, the teacher is used to generate "hard" labels (hard labels) for the training set.

A "hard" label is simply the class with the highest probability predicted by the teacher for each input example.

The student is then trained on these generated labels, as if they were true labels from a labeled dataset.

Essentially, we use the teacher to create a synthetic dataset on which we train the student using standard methods.

**Hard-label Distillation for GPT models: An intuitive explanation**

Imagine we have two models:

*   **Teacher:** A large, powerful GPT model, e.g., GPT-3 or similar. It possesses vast knowledge about language and the world and can generate very high-quality, coherent text.
*   **Student:** A small, compact GPT model, e.g., a reduced version of GPT or a smaller Transformer. It is less resource-intensive but initially inferior to the teacher in text generation quality.

Our goal is to "teach" the small student model to generate text as well as the large teacher model, using the Hard-label Distillation method.

**Steps of Hard-label Distillation in this context:**

1.  **Generation of "hard" labels by the teacher (Large GPT):**
    *   We take a large set of text data (e.g., the training set on which the teacher was originally trained, or simply a large text corpus).
    *   For each text fragment (or prompt) from this set, we ask the large teacher model to generate text. In the context of GPT, this means we feed the teacher an input text (e.g., the beginning of a sentence or a prompt) and ask it to generate a continuation.
    *   The teacher generates a sequence of tokens it considers most probable to continue the given text. These generated token sequences are our "hard" labels.

    **Example:**
    *   **Input text (prompt):** "The capital of France is"
    *   **Teacher (Large GPT) generates:** "Paris." (tokens: "Pa", "ri", "j", ".")
    *   **"Hard" label:** Token sequence: ("Pa", "ri", "j", ".")

    We repeat this process for a large number of different input texts, obtaining a set of pairs: (original input text, "hard" label—the token sequence generated by the teacher).

2.  **Training the student (Small GPT) on "hard" labels:**
    *   Now we have a synthetic dataset consisting of pairs (original input text, "hard" label). We use this dataset to train the small student model.
    *   We train the student to predict the "hard" labels generated by the teacher, using the standard language modeling task. This means that for each input text, we want the student to generate a token sequence as similar as possible to the "hard" label generated by the teacher.
    *   During training, we use the cross-entropy loss function. We compare the probability distribution of tokens predicted by the student with the "hard" label (which is essentially a distribution where the probability of the "correct" token is 1 and all others are 0). We strive to minimize this cross-entropy, forcing the student to "imitate" the teacher in token prediction.

    In our example, if the student predicts, for the input "The capital of France is", for example, "London", the loss function will be high because the teacher's "hard" label was "Paris". During training, the student will adjust its parameters to predict "Paris" or something very similar to the teacher's prediction for similar queries in the future.

**Why can the small model predict the same tokens as the large model?**

*   **Knowledge transfer via "hard" labels:** Although Hard-label Distillation loses some information from the teacher's probability distribution, it still effectively transfers **key knowledge** about which tokens are most probable in certain contexts. The large model, being well-trained, "knows" what continuations are grammatically correct, semantically appropriate, and stylistically suitable. By generating "hard" labels, it essentially "hints" to the small model which specific tokens to predict.
*   **Focus on the most important information:** "Hard" labels concentrate on the most probable tokens. In language modeling, it is often the case that for many contexts, there is one or several dominant "correct" continuations. Hard-label Distillation helps the small model quickly master these most important patterns, ignoring less significant details that may be redundant for achieving good generation quality.
*   **Simplification of the learning task:** Training on "hard" labels transforms distillation into a standard supervised learning task. This simplifies the training process and allows the use of well-known methods and optimizers. The small model does not need to try to reproduce all the nuances of the teacher's probability distribution; it only needs to learn to predict the most probable tokens, which is a simpler task.

**Important to note the limitations of Hard-label Distillation:**

*   **Loss of "soft" information:** As stated in the text, Hard-label Distillation loses information about the probabilities of other classes and "soft" relationships between classes. In the context of language models, this means the student may not capture all the nuances of style, semantics, and diversity present in the teacher's probability distribution. For example, the teacher may know that "Paris" is the most probable answer to "The capital of France is", but also understands that "Rome" or "Berlin" are less probable but still acceptable answers in certain contexts. Hard-label Distillation focuses only on "Paris", ignoring this "soft" information.
*   **Potential reduction in diversity:** Due to the focus on "hard" labels, the student may become less diverse in its generations than the teacher. It may copy the teacher's most probable answers too closely, missing the opportunity to generate alternative yet still high-quality variants.

**Mathematical Formalization:**

1.  **Generation of "hard" labels by the teacher:** For each example $x^{(n)}$ from the training set, the teacher $p(y|x)$ predicts a probability distribution over classes. The "hard" label $y^{(n)}$ is chosen as the class with the maximum probability predicted by the teacher. In the context of language models, where $y$ represents a sequence of tokens, the teacher generates a sequence of "hard" labels $y^{(1)}, \ldots y^{(N)}$ for $N$ examples. Here, $y^{(n)} = (y_1^{(n)}, \ldots, y_{T_n}^{(n)})$ represents a token sequence of length $T_n$.

    $y^{(1)}, \ldots y^{(N)} \sim p(y|x)$

    In a simpler variant for classification:
    $y^{(n)} = \arg\max_{y} p(y|x^{(n)})$. For sequences, the teacher may generate entire sequences of the most probable tokens.

2.  **Training the student on "hard" labels:** The student $q_{\theta}(y|x)$ is trained to maximize the log-probability of the "hard" labels generated by the teacher. This is a standard supervised learning task where the target labels are $y^{(1)}, \ldots y^{(N)}$. The loss function we minimize (or equivalently, maximize the negative loss) represents the expectation of the log-probability of the "hard" labels under the teacher's distribution $p(y|x)$.

    $\mathbb{E}_{p(y|x)} [\log q_{\theta}(y|x)] \rightarrow \max_{\theta}$

    In practical implementation, this expectation is approximated by the empirical average over the training set. For text sequences, the loss function is:

    $\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log q_{\theta}(y_t^{(n)}|y_{<t}^{(n)})$

    Here:
    *   $N$ — number of examples in the training set.
    *   $T_n$ — length of the sequence for the $n$-th example.
    *   $y_t^{(n)}$ — the $t$-th token in the "hard" label sequence for the $n$-th example, generated by the teacher.
    *   $y_{<t}^{(n)} = (y_1^{(n)}, \ldots, y_{t-1}^{(n)})$ — the prefix of the sequence up to the $t$-th token.
    *   $q_{\theta}(y_t^{(n)}|y_{<t}^{(n)})$ — the probability of the student predicting the $t$-th token $y_t^{(n)}$ given the previous tokens $y_{<t}^{(n)}$, parameterized by $\theta$.

    This loss function represents the **cross-entropy** between the distribution of "hard" labels generated by the teacher and the student's predictions. We strive to maximize this quantity, which is equivalent to minimizing the negative log-likelihood or cross-entropy.

**Advantages and Disadvantages of Hard-label Distillation:**

*   **Advantages:** Simplicity of implementation and understanding. Can use standard supervised learning methods.
*   **Disadvantages:** Loss of information contained in the teacher's probability distribution. "Hard" labels contain only information about the most probable class, ignoring the probabilities of other classes and "soft" relationships between classes that the teacher "knows". This can limit the effectiveness of knowledge transfer.

## **Implementation of Hard-label Distillation based on Open R1**

Below is an implementation of Hard-label Distillation using the approach applied in the Open R1 project. The process is divided into two stages: teacher data generation and student training.

``` @misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {Hugging Face},
    month = {January},
    year = {2025}
}
```

### **Stage 1: Generating "Hard" Labels with a Large Teacher Model**

```python
import argparse
from datasets import load_dataset
from typing import Optional, Dict, Any

from distilabel.pipeline import Pipeline
from distilabel.models import vLLM
from distilabel.steps.tasks import TextGeneration

def build_hard_label_pipeline(
    teacher_model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    temperature: float = 0.0,
    max_new_tokens: int = 4096,
    input_batch_size: int = 32,
) -> Pipeline:
    """
    Description:
    ---------------
        Creates a pipeline for generating "hard" labels using a teacher model.

    Args:
    ---------------
        teacher_model: Identifier of the teacher model
        base_url: URL of the vLLM server
        prompt_column: Name of the dataset column containing input texts
        prompt_template: Template for formatting prompts
        temperature: Temperature for generation (0.0 for "hard" labels)
        max_new_tokens: Maximum number of tokens to generate
        input_batch_size: Batch size for input data

    Returns:
    ---------------
        Configured Distilabel pipeline

    Raises:
    ---------------
        Exception: If pipeline configuration fails

    Examples:
    ---------------
        >>> pipeline = build_hard_label_pipeline("deepseek-ai/DeepSeek-R1")
        >>> pipeline.run(dataset)
    """
    # Configure generation parameters with temperature=0 to obtain deterministic outputs
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "do_sample": False,          # Disable sampling to obtain "hard" labels
    }

    with Pipeline(
        name="hard-label-distillation",
        description="Pipeline for generating 'hard' labels using a teacher model",
    ) as pipeline:
        # Configure the teacher model via vLLM
        teacher = vLLM(
            model=teacher_model,
            tokenizer=teacher_model,
            extra_kwargs={
                "tensor_parallel_size": 1,               # Can be increased for larger models
                "max_model_len": max_new_tokens + 2048,  # Add buffer for context
            },
            generation_kwargs=generation_kwargs,
        )

        # Configure the text generation step
        text_generation = TextGeneration(
            llm=teacher,
            template=prompt_template,
            num_generations=1,           # For "hard" labels, only one generation is needed
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
        )

    return pipeline

def generate_hard_labels(
    dataset_name: str,
    dataset_split: str = "train",
    teacher_model: str = "deepseek-ai/DeepSeek-R1",
    output_dataset: str = "my-username/hard-label-distill-dataset",
    prompt_column: str = "problem",
    prompt_template: str = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}: {{ instruction }}",
    max_examples: Optional[int] = None,
    private: bool = False,
) -> Any:
    """
    Description:
    ---------------
        Generates "hard" labels using a teacher model and saves the results as a dataset on HuggingFace Hub.

    Args:
    ---------------
        dataset_name: Name of the source dataset
        dataset_split: Name of the dataset split
        teacher_model: Teacher model for generating "hard" labels
        output_dataset: Name of the output dataset on HuggingFace Hub
        prompt_column: Name of the column containing input data
        prompt_template: Template for formatting prompts
        max_examples: Maximum number of examples to process
        private: Whether the output dataset should be private

    Returns:
    ---------------
        Dataset with "hard" labels

    Raises:
    ---------------
        Exception: If label generation fails

    Examples:
    ---------------
        >>> hard_label_dataset = generate_hard_labels("my-dataset", "train")
        >>> hard_label_dataset.push_to_hub("my-username/hard-label-dataset")
    """
    # Load the source dataset
    print(f"Loading dataset '{dataset_name}' (split: {dataset_split})...")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Limit the number of examples if specified
    if max_examples is not None and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))

    print(f"Creating pipeline for generating 'hard' labels using {teacher_model}...")
    pipeline = build_hard_label_pipeline(
        teacher_model=teacher_model,
        prompt_column=prompt_column,
        prompt_template=prompt_template,
    )

    print(f"Running pipeline to generate 'hard' labels on {len(dataset)} examples...")
    # Generate "hard" labels
    hard_label_dataset = pipeline.run(dataset=dataset)

    # Save results to HuggingFace Hub
    if output_dataset:
        print(f"Saving results to '{output_dataset}'...")
        hard_label_dataset.push_to_hub(output_dataset, private=private)
        print(f"Dataset with 'hard' labels successfully saved to '{output_dataset}'.")

    return hard_label_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating 'hard' labels using a teacher model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the source dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--teacher-model", type=str, default="deepseek-ai/DeepSeek-R1", help="Teacher model")
    parser.add_argument("--output-dataset", type=str, required=True, help="Name of the output dataset")
    parser.add_argument("--prompt-column", type=str, default="problem", help="Column containing input data")
    parser.add_argument("--prompt-template", type=str,
                       default="You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}: {{ instruction }}",
                       help="Template for formatting prompts")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples")
    parser.add_argument("--private", action="store_true", help="Make the output dataset private")

    args = parser.parse_args()

    generate_hard_labels(
        dataset_name=args.dataset,
        dataset_split=args.split,
        teacher_model=args.teacher_model,
        output_dataset=args.output_dataset,
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template,
        max_examples=args.max_examples,
        private=args.private,
    )
```

### **Stage 2: Training the Student Model on "Hard" Labels**

```python
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, ModelConfig, TrlParser, get_peft_config
from open_r1.configs import SFTConfig
from open_r1.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)

@dataclass
class HardLabelDistillConfig(SFTConfig):
    """Configuration for training a student model using Hard-label Distillation."""

    dataset_name: str = field(
        default=None, metadata={"help": "Dataset with 'hard' labels generated by the teacher"}
    )
    input_column: str = field(
        default="problem", metadata={"help": "Column containing input data"}
    )
    target_column: str = field(
        default="generation_0", metadata={"help": "Column containing teacher's outputs ('hard' labels)"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length"}
    )

def train_student_model(config: HardLabelDistillConfig, model_args: ModelConfig) -> None:
    """
    Description:
    ---------------
    Trains a student model on 'hard' labels generated by the teacher.

    Args:
    ---------------
        config: Training configuration
        model_args: Model configuration

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Exception: If model training fails

    Examples:
    ---------------
        >>> train_student_model(config, model_args)
    """
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Check for last checkpoint
    last_checkpoint: Optional[str] = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint found, resuming training from {last_checkpoint}")

    # Initialize Weights & Biases if needed
    if "wandb" in config.report_to:
        init_wandb_training(config)

    # Load dataset with 'hard' labels
    logger.info(f"Loading dataset with 'hard' labels: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    # Prepare input data and labels for training
    def prepare_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Formats data for supervised training."""
        return {
            "input_ids": examples[config.input_column],
            "labels": examples[config.target_column],
        }

    # Transform the dataset
    dataset = dataset.map(prepare_dataset, batched=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Configure chat_template if specified
    if config.chat_template is not None:
        tokenizer.chat_template = config.chat_template

    # Configure model parameters
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs: Dict[str, Any] = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if config.gradient_checkpointing else True,
    )
    config.model_init_kwargs = model_kwargs

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset and config.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Start training
    logger.info("Starting student model training...")
    checkpoint: Optional[str] = None
    if config.resume_from_checkpoint is not None:
        checkpoint = config.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)

    # Create model card and push to HuggingFace Hub if needed
    kwargs: Dict[str, Any] = {
        "dataset_name": config.dataset_name,
        "tags": ["hard-label-distillation", "open-r1"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Re-enable cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(config.output_dir)

    # Evaluate model if needed
    if config.do_eval and "validation" in dataset:
        logger.info("Evaluating model...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push model to HuggingFace Hub if needed
    if config.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    # Create argument parser
    parser = TrlParser((HardLabelDistillConfig, ModelConfig))
    config, model_args = parser.parse_args_and_config()

    # Start training
    train_student_model(config, model_args)
```

### **Usage Example**

```python
# Stage 1: Generate "hard" labels using a teacher model
python hard_label_distill.py \
  --dataset AI-MO/NuminaMath-TIR \
  --teacher-model deepseek-ai/DeepSeek-R1 \
  --output-dataset username/hard-label-math-dataset \
  --prompt-column problem

# Stage 2: Train the student model on the generated "hard" labels
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml train_student.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name username/hard-label-math-dataset \
  --input_column problem \
  --target_column generation_0 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 2 \
  --packing \
  --max_seq_length 4096 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --bf16 \
  --output_dir models/Qwen2.5-1.5B-Hard-Label-Distill
```

## II. Soft-label Distillation: Knowledge Distillation Using Soft Labels

**Concept:**

Soft-label distillation, introduced by Hinton and colleagues in their seminal paper "Distilling the Knowledge in a Neural Network" (2015), is a more sophisticated approach to knowledge distillation. Unlike hard-label distillation, this method leverages not only "hard" labels but also the full **probability distribution** predicted by the teacher as "soft labels."

Soft labels contain significantly more information than hard labels, as they reflect the teacher's confidence across different classes and the relationships between them. For example, a teacher might predict for an image of a dog the probabilities [0.8 for "dog", 0.15 for "wolf", 0.03 for "fox", 0.02 for other classes]. This information is far richer than simply the label "dog."

A key component of this method is "temperature scaling," which makes the probability distribution more "soft" and informative by dividing the model's logits by a temperature parameter T > 1.

**Soft-label Distillation for GPT Models: A Simple Explanation**

Imagine we have two models:

* **Teacher:** A large, powerful GPT model with 175 billion parameters. It possesses deep understanding of language and the world.
* **Student:** A compact GPT model with 1.5 billion parameters. Much faster and more economical, but initially inferior in quality to the teacher.

Our goal is to teach the student to generate text as well as the teacher, using soft-label distillation.

**Steps of Soft-label Distillation:**

1. **Teacher Generates Soft Labels:**

   * For the prompt "The capital of France is," the large teacher model does not simply output "Paris," but computes probabilities for all possible next tokens:
     * "Paris": 0.92
     * "city": 0.03
     * "Rome": 0.01
     * ... (and thousands of other tokens with small probabilities)

   * Problem: This distribution is too "sharp"—one token holds almost all the probability. To extract more useful knowledge, we apply **temperature scaling**:
   
   * Divide the logits by temperature T (e.g., T = 2.0) before applying softmax:
     * "Paris": 0.70 (decreased from 0.92)
     * "city": 0.08 (increased from 0.03)
     * "Rome": 0.05 (increased from 0.01)
     * ... (other tokens also receive higher probabilities)

   * These "softened" distributions preserve much more information about what the teacher model "knows."

2. **Training the Student Model:**

   * The student is trained not only to predict the correct token but also to reproduce the teacher's full probability distribution.
   * This is achieved using KL divergence (or cross-entropy) between the teacher's and student's distributions.
   * Crucially, the student's distribution is also "softened" using the same temperature T for comparability.
   * The loss function is multiplied by T² to compensate for reduced gradients.

3. **Combined Training:**

   * Typically, a combination of two loss functions is used:
     * α · (Loss from soft labels) + (1-α) · (Standard loss from hard labels)
   * Where α is a coefficient, usually between 0.5 and 0.9

**Why Does This Work Better Than Hard-label Distillation?**

* **"Dark Knowledge":** As Hinton called it, the relative probabilities of "incorrect" answers contain valuable information. For example, if a model confuses "dog" with "wolf" but not with "airplane," this is important information.

* **Transferring Uncertainty:** The student learns not only correct answers but also when to be uncertain.

* **Richer Signal:** Instead of receiving just one bit of information per example (correct/incorrect class), the student receives information about the entire probability distribution.

**Mathematical Formalization:**

1. **Teacher's Soft Labels with Temperature T:**

   If $z_i$ is the logits for class (token) $i$ from the teacher, then the soft label with temperature T is:

   $$p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

*   **Breakdown of each element in the formula:**
    *   $p_i^T$: This is the "soft" probability for token $i$, adjusted by temperature $T$. This is the probability distribution generated by the teacher that we use as the soft label.
    *   $z_i$: This is the logit for token $i$, output by the teacher model. Logits are the raw scores the model produces before applying the softmax function—they represent the model's confidence level for each token. Higher logits indicate higher confidence.
    *   $T$: This is the temperature parameter. As discussed above, temperature is used to "soften" the probability distribution.
    *   $\exp(x)$: This is the exponential function ($e^x$).
    *   $\sum_j \exp(z_j/T)$: This is the sum of the exponential values of the logits divided by temperature, across all possible tokens $j$. This sum normalizes the values so that the resulting probabilities sum to 1.

*   **Step-by-step explanation:**
    1.  **Divide logits by temperature ($z_i/T$):** When we divide logits by temperature $T > 1$, we reduce their absolute magnitudes.
    2.  **Exponentiation ($\exp(z_i/T)$):** The exponential function transforms logits into positive values.
    3.  **Normalization ($\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$):** Dividing by the sum of all exponential logits ensures that the resulting values $p_i^T$ form a valid probability distribution—non-negative and summing to 1. This is the standard softmax operation, but with temperature applied.

*   **Intuition and effect of temperature:**
    *   At high temperature (e.g., $T = 2.0$), the probability distribution becomes more "soft" or "smooth." Probabilities for less likely tokens increase, while the probability of the most likely token decreases. This allows us to "extract" more information from the distribution, including "dark knowledge" about less probable but still relevant alternatives.
    *   At low temperature (approaching $T = 1.0$, or even lower), the distribution becomes more "sharp." The probability of the most likely token approaches 1, while others approach 0. At $T=1$, this is standard softmax. As $T \rightarrow 0$, the distribution becomes a delta function, selecting only the token with the highest logit.

![Figure_1.jpg](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_1.png  )

2. **Similarly for the Student:**

   $$q_i^T = \frac{\exp(z_i^q/T)}{\sum_j \exp(z_j^q/T)}$$

   where $z_i^q$ is the student's logits for class $i$.

*   **Analogy with teacher's formula:** This formula is identical to the teacher's, except that it uses logits output by the student model.
    *   $q_i^T$: The "soft" probability for token $i$, generated by the student with temperature $T$.
    *   $z_i^q$: The logits for token $i$, output by the student model.

*   **Goal:** We apply the same temperature $T$ to the student's distribution to make it comparable to the teacher's soft labels. This is necessary for correctly computing the distillation loss function.

3. **Loss Function for Soft-label Distillation:**

   $$L_{soft} = T^2 \cdot \text{KL}(p^T || q^T) = T^2 \cdot \sum_i p_i^T \log\frac{p_i^T}{q_i^T}$$

   The multiplier $T^2$ compensates for the reduction in gradients due to temperature scaling.

*   **Breakdown of components:**
    *   $L_{soft}$: The soft-label distillation loss function. This is the value we aim to minimize during student training.
    *   $T^2$: The square of the temperature. This multiplier scales the loss function to compensate for gradient reduction caused by temperature.
    *   $\text{KL}(p^T || q^T)$: The Kullback-Leibler (KL) divergence between the teacher's distribution $p^T$ and the student's distribution $q^T$.
    *   $\sum_i p_i^T \log\frac{p_i^T}{q_i^T}$: The expanded formula for KL divergence over discrete distributions.

*   **Step-by-step explanation of KL divergence:**
    1.  **$\frac{p_i^T}{q_i^T}$:** The ratio of the teacher's probability to the student's probability for each token $i$. If the student predicts $q_i^T$ close to the teacher's $p_i^T$, this ratio will be close to 1.
    2.  **$\log\frac{p_i^T}{q_i^T}$:** The logarithm of this ratio. If the ratio is close to 1, the logarithm is near 0. If $q_i^T$ deviates significantly from $p_i^T$, the logarithm becomes large in absolute value (negative if $q_i^T > p_i^T$, positive if $q_i^T < p_i^T$).
    3.  **$p_i^T \log\frac{p_i^T}{q_i^T}$:** Multiplying by $p_i^T$ weights each token's contribution to the total divergence. Tokens the teacher considers more probable (high $p_i^T$) contribute more to the loss.
    4.  **$\sum_i p_i^T \log\frac{p_i^T}{q_i^T}$:** Summing over all tokens $i$ gives the total KL divergence. KL divergence measures the "distance" between two probability distributions. In distillation, it quantifies how much the student's distribution $q^T$ differs from the teacher's distribution $p^T$.

*   **Role of $T^2$:**
    *   Applying temperature $T$ "softens" the distributions, which may reduce the magnitude of gradients during training. Multiplying by $T^2$ scales the loss function to compensate for this reduction and make gradients more significant, especially early in training. This is an empirical correction that helps stabilize and accelerate learning.

*   **Goal of $L_{soft}$:** By minimizing $L_{soft}$, we force the student's probability distribution $q^T$ to closely match the teacher's distribution $p^T$. The student learns not only to predict the "correct" token but also to imitate the teacher's entire "thinking style," expressed in the probability distribution.

4. **Combined Loss Function:**

   $$L = \alpha \cdot L_{soft} + (1-\alpha) \cdot L_{hard}$$

   where $L_{hard}$ is the standard cross-entropy with ground-truth labels, and $\alpha$ is the balancing coefficient.

*   **Breakdown of components:**
    *   $L$: The total loss function used to train the student.
    *   $\alpha$: The balancing coefficient (typically between 0.5 and 0.9). It determines how strongly we rely on the teacher's soft labels versus standard hard labels.
    *   $L_{soft}$: The soft-label distillation loss function, explained above.
    *   $L_{hard}$: The standard hard-label loss function, usually cross-entropy between the student's predictions and the true (one-hot) labels.

*   **$L_{hard}$ (Standard Hard-label Loss):**
    *   In a standard language model training task, we have "hard" labels—the ground-truth next tokens in the training data. For example, for the phrase "The capital of France is Paris," "Paris" is the hard label.
    *   $L_{hard}$ is computed as cross-entropy between the student's predicted probability distribution (usually with $T=1$, i.e., standard softmax) and a one-hot vector representing the true token. This loss forces the student to predict the exact "correct" token.

*   **Combining $L_{soft}$ and $L_{hard}$:**
    *   Combining soft and hard losses allows the student to learn both from the teacher (via $L_{soft}$) and from the original data (via $L_{hard}$).
    *   The coefficient $\alpha$ allows tuning the balance.
        *   High $\alpha$ (e.g., 0.9) means we rely more on the teacher's knowledge transmitted through soft labels. This is useful when the teacher has significantly better knowledge than can be extracted from hard labels alone.
        *   Low $\alpha$ (e.g., 0.5) means we equally consider both the teacher's knowledge and hard labels. This is useful when we want the student to retain its ability to perform well on original data, not just mimic the teacher.

**Practical Implementation of Soft-label Distillation for GPT Models**

> Code was adapted from: https://github.com/arcee-ai/DistillKit  

**1. Distillation Configuration**

The first step is to configure distillation parameters, including temperature and the balance coefficient between soft and hard labels:

```python
"""
Here temperature: 2.0 corresponds to parameter T in the formulas, which "softens" the probability distribution, and alpha: 0.5 is the coefficient α, determining the ratio between soft and hard label losses.
"""

config = {
    "project_name": "distil-multilayer",    # Project name
    "dataset": {
        "name": "mlabonne/FineTome-100k",   # Dataset name
        "split": "train",                   # Dataset split for training
        "num_samples": 1000,                # Number of training samples (can be limited)
        "seed": 42                          # Random seed
    },
    "models": {
        "teacher": "arcee-ai/Arcee-Spark",  # Teacher model
        "student": "Qwen/Qwen2-1.5B"        # Student model
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}\
    {% if loop.first and messages[0]['role'] != 'system' %}\
    {{ ' <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\
    {% endif %}\
    {{ ' <|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}\
    {% endfor %}\
    {% if add_generation_prompt %}\
    {{ ' <|im_start|>assistant\\n' }}\
    {% endif %}"
    },
    "training": {
        "output_dir": "./results",           # Directory to save results
        "num_train_epochs": 3,               # Number of training epochs
        "per_device_train_batch_size": 1,    # Batch size per device
        "gradient_accumulation_steps": 8,    # Number of steps for gradient accumulation
        "save_steps": 1000,                  # Steps between model saves
        "logging_steps": 2,                  # Steps between logging
        "save_total_limit": 2,               # Maximum number of saved models
        "learning_rate": 2e-5,               # Learning rate
        "weight_decay": 0.01,                # Regularization coefficient
        "warmup_ratio": 0.2,                 # Fraction of steps for learning rate warmup
        "lr_scheduler_type": "linear",       # Learning rate scheduler type
        "resume_from_checkpoint": None,      # Path to checkpoint for resuming training (if any)
        "fp16": False,                       # Use 16-bit floating point
        "bf16": True,                        # Use BFloat16
        "max_grad_norm": 1.0,                # Maximum gradient norm
        "group_by_length": False             # Group batches by length
    },
    "distillation": {
        "temperature": 2.0,                  # Temperature for distillation
        "alpha": 0.5                         # Alpha coefficient for distillation
    },
    "model_config": {
        "use_flash_attention": True          # Use Flash Attention
    }
}
```

**2. Preparing Teacher and Student Models**

For distillation, both the teacher model (larger) and the student model (more compact) must be loaded:

```python
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM

def load_models_with_flash_attention(config: Dict[str, Any]) -> Dict[str, AutoModelForCausalLM]:
    """
    Description:
    ---------------
        Loads models with flash attention enabled for acceleration.

    Args:
    ---------------
        config: Model and parameter configuration

    Returns:
    ---------------
        Dictionary containing loaded models

    Raises:
    ---------------
        KeyError: If required keys are missing from the configuration

    Examples:
    ---------------
        >>> config = {
        ...     "model_config": {"use_flash_attention": True},
        ...     "models": {"teacher": "teacher_model_path", "student": "student_model_path"}
        ... }
        >>> load_models_with_flash_attention(config)
        {'teacher_model': <transformers.models.model_name.model.ModelName object>,
         'student_model': <transformers.models.model_name.model.ModelName object>}
    """
    # Model loading settings
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}

    # Check for flash attention usage
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load models
    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

    return {"teacher_model": teacher_model, "student_model": student_model}

# Function call
models = load_models_with_flash_attention(config)

# Now models contains the loaded models
teacher_model = models["teacher_model"]
student_model = models["student_model"]
```

**3. Implementation of Soft-label Loss Function**

The key component is the soft-label distillation loss function. Below is its implementation from `distil_logits.py`:

```python
"""
This is a direct implementation of the KL divergence formula. Note the following key points:

1. Logits are scaled by temperature T before applying softmax/log_softmax.
2. Losses are multiplied by T² to compensate for gradient reduction, as described in the theory.
3. The final loss combines soft labels (KL divergence) and hard labels (original_loss) with coefficient α.
"""

from typing import Any
import torch
import torch.nn.functional as F

def distillation_loss(
    self,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    inputs: Any,
    original_loss: torch.Tensor,
    config: Dict[str, Any]
) -> torch.Tensor:
    """
    Description:
    ---------------
        Computes distillation loss between student and teacher logits.

    Args:
    ---------------
        student_logits: Student logits.
        teacher_logits: Teacher logits.
        inputs: Input data.
        original_loss: Original loss.
        config: Model and parameter configuration.

    Returns:
    ---------------
        Total loss combining distillation loss and original loss.

    Raises:
    ---------------
        KeyError: If required keys are missing from the configuration.

    Examples:
    ---------------
        >>> config = {
        ...     "distillation": {"temperature": 2.0, "alpha": 0.5},
        ...     "tokenizer": {"max_length": 512}
        ... }
        >>> student_logits = torch.randn(3, 512)
        >>> teacher_logits = torch.randn(3, 512)
        >>> inputs = ...
        >>> original_loss = torch.tensor(0.5)
        >>> distillation_loss(self, student_logits, teacher_logits, inputs, original_loss, config)
        tensor(0.25)
    """
    # Align dimensions of teacher and student logits
    student_logits, teacher_logits = pad_logits(
        student_logits.to(self.model.device),
        teacher_logits.to(self.model.device)
    )

    # Scale logits with temperature T
    temperature = config["distillation"]["temperature"]
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature

    # Compute KL divergence between teacher and student distributions
    loss_kd = F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),  # log(q_i^T)
        F.softmax(teacher_logits_scaled, dim=-1),      # p_i^T
        reduction='batchmean'
    ) * (temperature ** 2) / config["tokenizer"]["max_length"]

    # Combine soft and hard label losses
    alpha = config["distillation"]["alpha"]
    total_loss = alpha * loss_kd + (1 - alpha) * original_loss

    return total_loss
```

**4. Handling Different Vocabulary Sizes**

Since teacher and student models may have different token vocabulary sizes, an additional function is required to align the dimensions of their logits:

```python
"""
This function adds zero logits to the smaller distribution to ensure equal dimensions for comparison.
"""

from typing import Tuple
import torch

def pad_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    ---------------
        Aligns the dimensions of student and teacher logits to be identical.

    Args:
    ---------------
        student_logits: Student logits.
        teacher_logits: Teacher logits.

    Returns:
    ---------------
        Tuple of student and teacher logits with matching dimensions.

    Raises:
    ---------------
        ValueError: If logits dimensions do not match and cannot be aligned.

    Examples:
    ---------------
        >>> student_logits = torch.randn(3, 512)
        >>> teacher_logits = torch.randn(3, 510)
        >>> pad_logits(student_logits, teacher_logits)
        (tensor([...]), tensor([...]))
    """
    # Determine logits sizes
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)

    # If sizes differ, apply padding
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros(
            (*teacher_logits.shape[:-1], pad_size),
            dtype=teacher_logits.dtype,
            device=teacher_logits.device
        )

        # Return logits with added padding
        if student_size < teacher_size:
            return torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits
        else:
            return student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1)

    # Return logits unchanged if sizes match
    return student_logits, teacher_logits
```

**5. Custom Trainer for Distillation**

To integrate distillation into the training process, a custom trainer class is created that overrides the loss computation function:

```python
"""
This class:
1. Obtains outputs (logits) from both student and teacher models
2. Freezes teacher weights using `torch.no_grad()`
3. Computes combined loss using soft and hard label losses
"""

from typing import Dict, Any, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import SFTTrainer

class LogitsTrainer(SFTTrainer):
    """
    Description:
    ---------------
        Class for training a model using logits distillation.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Description:
        ---------------
            Computes combined loss for student and teacher models.

        Args:
        ---------------
            model: Student model.
            inputs: Input data.
            return_outputs: Flag to return model outputs.

        Returns:
        ---------------
            Combined loss and, if specified, model outputs.

        Raises:
        ---------------
            ValueError: If input data does not meet expectations.

        Examples:
        ---------------
            >>> model = ...
            >>> inputs = ...
            >>> trainer = LogitsTrainer()
            >>> trainer.compute_loss(model, inputs, return_outputs=True)
            (tensor(0.5), ...)
        """
        # Move inputs to model device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Move teacher model to device
        self.teacher_model = self.teacher_model.to(model.device)

        # Get model modules if they exist
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        # Obtain model outputs
        student_outputs = student_model(**inputs)
        with torch.no_grad():  # Teacher is not trained
            teacher_outputs = teacher_model(**inputs)

        # Compute combined loss
        custom_loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            inputs,
            student_outputs.loss
        )

        # Return loss and outputs if requested
        if return_outputs:
            return custom_loss, student_outputs
        return custom_loss

    def pad_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
        ---------------
            Aligns the dimensions of student and teacher logits to be identical.

        Args:
        ---------------
            student_logits: Student logits.
            teacher_logits: Teacher logits.

        Returns:
        ---------------
            Tuple of student and teacher logits with matching dimensions.

        Raises:
        ---------------
            ValueError: If logits dimensions do not match and cannot be aligned.

        Examples:
        ---------------
            >>> student_logits = torch.randn(3, 512)
            >>> teacher_logits = torch.randn(3, 510)
            >>> trainer = LogitsTrainer()
            >>> trainer.pad_logits(student_logits, teacher_logits)
            (tensor([...]), tensor([...]))
        """
        # Determine logits sizes
        student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)

        # If sizes differ, apply padding
        if student_size != teacher_size:
            pad_size = abs(student_size - teacher_size)
            pad_tensor = torch.zeros(
                (*teacher_logits.shape[:-1], pad_size),
                dtype=teacher_logits.dtype,
                device=teacher_logits.device
            )

            # Return logits with added padding
            if student_size < teacher_size:
                return torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits
            else:
                return student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1)

        # Return logits unchanged if sizes match
        return student_logits, teacher_logits

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        inputs: Any,
        original_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Computes distillation loss between student and teacher logits.

        Args:
        ---------------
            student_logits: Student logits.
            teacher_logits: Teacher logits.
            inputs: Input data.
            original_loss: Original loss.

        Returns:
        ---------------
            Total loss combining distillation loss and original loss.

        Raises:
        ---------------
            KeyError: If required keys are missing from the configuration.

        Examples:
        ---------------
            >>> config = {
            ...     "distillation": {"temperature": 2.0, "alpha": 0.5},
            ...     "tokenizer": {"max_length": 512}
            ... }
            >>> student_logits = torch.randn(3, 512)
            >>> teacher_logits = torch.randn(3, 512)
            >>> inputs = ...
            >>> original_loss = torch.tensor(0.5)
            >>> trainer = LogitsTrainer()
            >>> trainer.distillation_loss(student_logits, teacher_logits, inputs, original_loss)
            tensor(0.25)
        """
        # Align dimensions of teacher and student logits
        student_logits, teacher_logits = self.pad_logits(
            student_logits.to(self.model.device),
            teacher_logits.to(self.model.device)
        )

        # Scale logits with temperature T
        temperature = config["distillation"]["temperature"]
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature

        # Compute KL divergence between teacher and student distributions
        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),  # log(q_i^T)
            F.softmax(teacher_logits_scaled, dim=-1),      # p_i^T
            reduction='batchmean'
        ) * (temperature ** 2) / config["tokenizer"]["max_length"]

        # Combine soft and hard label losses
        alpha = config["distillation"]["alpha"]
        total_loss = alpha * loss_kd + (1 - alpha) * original_loss

        return total_loss
```

**6. Initializing the Trainer and Starting Training**

After defining all components, initialize the trainer and launch the distillation process:

```python
"""
Note: The teacher model is added to the trainer as an attribute so it is accessible within the `compute_loss` function.
"""

# Import required libraries
from transformers import TrainingArguments
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Check for preprocessed dataset
if 'tokenized_dataset' not in locals():
    # If dataset is not preprocessed, perform necessary preprocessing
    # Dataset preprocessing code should be here...
    print("Dataset preprocessing must be performed first!")

# Create custom SFT trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=student_tokenizer,
    args=training_arguments,
    max_seq_length=config["tokenizer"]["max_length"],
    dataset_text_field="text",
)

# Add teacher model to trainer
trainer.teacher_model = teacher_model

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Start training
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save final model
trainer.save_model(config["training"]["output_dir"])

print(f"Training completed. Model saved to {config['training']['output_dir']}")
```

**Advantages of Soft-label Distillation:**

* **More Complete Knowledge Transfer:** The student gains access to the teacher’s "dark knowledge"—information about complex cases, subtle distinctions between classes, and degrees of uncertainty.
* **Better Performance:** Students trained with this method typically achieve performance closer to the teacher compared to hard-label distillation.
* **Improved Generalization:** Models perform better on unseen data because they learn not only "what" to predict but also "with what confidence."
* **Control via Temperature:** The parameter T allows tuning the degree of "softness" in distillation. Higher T values produce more uniform distributions, helping convey more information about low-probability classes.
* **Compatibility with Other Methods:** Easily combined with other model enhancement techniques.

**Disadvantages of Soft-label Distillation:**

* **Computational Cost:** For language models with large vocabularies (50,000+ tokens), storing and transmitting full probability distributions requires significant resources.
* **Implementation Complexity:** Requires access to the teacher’s logits/probabilities, not just final predictions.
* **Hyperparameter Tuning:** Temperature T and coefficient α must be carefully tuned for optimal results.
* **Dependence on Teacher Quality:** If the teacher has systematic errors, they may be transferred to the student.

**Comparison of Hard-label and Soft-label Distillation:**

| Aspect | Hard-label Distillation | Soft-label Distillation |
|--------|-------------------------|-------------------------|
| Information Transferred | Only final classes/tokens | Full probability distributions |
| Temperature | Not used | Used to "soften" distributions |
| Implementation Complexity | Simple | Moderate |
| Computational Requirements | Low | Medium–High |
| Data Storage Volume | Small | Large (especially for language models) |
| Model Quality Achieved | Good | Better |
| Ability to Transfer Uncertainty | Low | High |
| Effectiveness for Language Models | Moderate | High |

In conclusion, Soft-label Distillation offers a more powerful method for transferring knowledge from teacher to student, particularly for complex tasks where fine distinctions between classes and understanding uncertainty matter. The key distinction from Hard-label Distillation lies in the use of full probability distributions and temperature scaling, enabling the extraction of "dark knowledge" and teaching the student not only to produce correct answers but also to replicate the teacher’s nuanced reasoning.

## **Part 2: Scaling Laws of Distillation**

After DeepSeek open-sourced its knowledge distillation method for R1, researchers from Apple and the University of Oxford quickly proposed a scaling law for distillation and completed all experiments by February 28, uploading a 67-page paper to arXiv.

The motivation behind this research can be summarized as follows:

1. **Current State of Scaling Laws for Models**: In recent years, research has revealed relationships between language model performance, model size, and training data volume. However, systematic studies of scaling laws in the context of distillation have not yet been conducted.

2. **Inference Cost Problem**: As language model sizes grow, inference cost increases significantly. Understanding how to reduce inference cost without sacrificing performance has become a critical challenge.

3. **Efficiency and Performance of Distillation**: Theoretically, distillation can reduce inference cost; however, there is no consensus in academia on optimal distillation methods—particularly regarding how to rationally allocate computational resources to build the most powerful models—leaving significant uncertainty.

![Extrapolation of Distillation Scaling Law](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_2.webp  )

Figure 1. Extrapolations of the distillation scaling law. The distillation scaling law (Equation 8) is approximated on weak students $( L_S > 2.3 )$ for a range of teachers with losses $( L_T )$. Solid lines represent the model’s predicted behavior for unseen teachers under a fixed student configuration (interpolation), while dashed lines represent predicted behavior beyond observed teachers and for strong students $( L_S \leq 2.3 )$.

### **Distillation Scaling Law**

The traditional scaling law for large models demonstrates that a language model’s (LM) performance can improve with increased computational resources, provided the model follows an optimal training computational paradigm. However, the continuous rise in inference cost makes this approach increasingly impractical, prompting researchers to seek alternatives such as retraining and distillation to create small yet powerful models.

Researchers conducted extensive experiments using student and teacher models ranging from 143 million to 12.6 billion parameters and training data up to 512 billion tokens. The goal was to study the relationship between model performance and computational resources during distillation, and to find ways to optimize the allocation of these resources.

The following table shows the symbols used in this paper:

Table 1. Expressions related to scaling laws used in this work. In every case, $S$ refers to the student, not to teacher training.

| Expression | Meaning |
|---|---|
| $N / N_S / N_T$ | Number of model/student/teacher parameters excluding embeddings. In the text, when we refer to parameters, we always mean non-embedding parameters unless otherwise specified. See Appendix H.2 for details. |
| $D / D_T$ | Number of tokens on which the model/teacher was pre-trained. |
| $D_S$ | Number of tokens on which the student was distilled. |
| $M \equiv D / N$ | Tokens-per-parameter ratio, or $M$-ratio. In Hoffmann et al. (2022), $M$ achieves an optimal value $M^* \approx 20$, which is the empirical Chinchilla rule. |
| $L \approx L(N, D)$ | Model cross-entropy, representing the validation cross-entropy of a model with $N$ parameters trained on $D$ tokens, evaluated according to the teacher scaling law. (Equation 1). |
| $L_T \approx L(N_T, D_T)$ | Teacher cross-entropy, representing the validation cross-entropy of a teacher with $N_T$ parameters trained on $D_T$ tokens, evaluated according to the teacher scaling law. |
| $L_S \approx L_S(N_S, D_S, L_T)$ | Student cross-entropy, representing the validation cross-entropy of a student with $N_S$ parameters distilled on $D_S$ tokens using a teacher with pre-training loss $L_T$, evaluated according to our distillation scaling law (Equation 8). |
| $\tilde{L}_S \approx L(N_S, D_S)$ | Teacher-trained student cross-entropy, representing the validation cross-entropy of a student with $N_S$ parameters trained on $D_S$ tokens *without* distillation, evaluated according to the teacher scaling law. |

> **Explanation**: Cross-entropy is a metric measuring the divergence between the model’s predicted probability distribution and the true distribution. Lower cross-entropy indicates better prediction of correct tokens—it is the primary metric for language model quality.

<details> 
    <summary><em><strong>Mathematical Formalization of Cross-Entropy</strong></em></summary>

Cross-entropy $H(p, q)$ between two probability distributions $p$ (true distribution) and $q$ (predicted distribution) is defined as:

$$H(p, q) = - \sum_{x} p(x) \log_2(q(x))$$  

(Usually, base-2 or natural logarithm is used; here base-2 is shown for illustration.)

In the context of language models, for evaluating the quality of predicting the next token in a sequence, the cross-entropy formula is adapted as follows:

$$H(p, q) = - \frac{1}{N} \sum_{i=1}^{N} \log_2(q(w_i | w_{<i}))$$

where:
*   $p$ is the **true probability distribution**. Ideally, this is the distribution of the real language. In practice, for each token $w_i$ in the training sequence, the true distribution $p(w)$ is a **one-hot vector**: $p(w_i) = 1$ for the true token $w_i$, and $p(w) = 0$ for all other tokens $w \neq w_i$ in the vocabulary.
*   $q$ is the **predicted probability distribution** by the model. The model predicts the probability for *every* token in the vocabulary to be the next one, given the context.
*   $w_i$ is the $i$-th token in the sequence.
*   $w_{<i}$ is the sequence of tokens preceding the $i$-th token (the context).
*   $q(w_i | w_{<i})$ is the probability predicted by the model for token $w_i$ given preceding tokens $w_{<i}$. This is the probability that the next token will be $w_i$, according to the model.
*   $N$ is the total number of tokens in the dataset over which cross-entropy is computed.

![Figure_3.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_3.png  )

**Detailed Explanation:**

The goal of a language model is to predict the next token in a word sequence. For each token in the training dataset, we want the model to assign high probability to the *actual next* token. The "true" probability distribution in this case can be represented as a distribution where the probability of the true next token is 1, and the probability of all other tokens is 0.

The model, in turn, predicts a probability distribution $q(w | w_{<i})$ for *all* possible tokens $w$ in the vocabulary, given the context $w_{<i}$. Cross-entropy measures how "far" the predicted distribution $q$ is from the "true" distribution $p$.

The cross-entropy formula in the context of language models computes the **average negative log-probability assigned to each true token**.

*   **Logarithm ($\log_2$ or natural $\ln$)**: Used to transform probabilities (values between 0 and 1) into values that are convenient to sum. The logarithm of a probability is always negative (or zero if the probability is 1). The use of logarithms is also tied to information theory and measuring information quantity (bits or nats).
*   **Negative Sign (-)**: Added to convert minimization of cross-entropy into maximization of probability. Minimizing the negative log-probability is equivalent to maximizing the probability itself.

**Interpretation of Cross-Entropy Value:**

*   **Lower cross-entropy means better model performance**. Low cross-entropy indicates that, on average, the model assigns high probabilities to the correct next tokens, indicating good prediction quality.
*   Cross-entropy is measured in bits (if using $\log_2$) or nats (if using natural logarithm $\ln$). In language modeling, **perplexity** is often discussed, which is exponentially related to cross-entropy (Perplexity = $2^{H(p,q)}$ for $\log_2$). Perplexity is also a popular quality metric, and lower perplexity indicates better performance.

</details>   

---

> **Explanation of the Chinchilla Rule**: The study by Hoffmann et al. (2022) established an empirical rule for the optimal ratio between model parameters and training tokens—approximately 20 tokens per parameter. This rule enables efficient allocation of computational resources during training of large language models.

<details> 
    <summary><em><strong>Explanation of the Chinchilla Rule</strong></em></summary>

The Chinchilla Rule can be expressed as the following **empirical** relationship:

$T_{optimal} \approx 20 \times P$

where:
*   $T_{optimal}$ is the **optimal** number of training tokens required to achieve the best performance given a fixed number of parameters and computational resources.
*   $P$ is the number of parameters in the model.

**Explanation:**

The Chinchilla Rule, proposed in Hoffmann et al. (2022), is an **empirical observation** derived from extensive experiments with large language models. Researchers sought to find the optimal balance between model size (number of parameters) and training data volume (number of tokens) to **maximize computational efficiency**.

The ratio $T_{optimal} \approx 20 \times P$ suggests that to achieve optimal performance when training a model with a given number of parameters, it is **optimal to use approximately 20 training tokens per parameter**.

**Intuitive Explanation:**

*   **Insufficient Tokens (T << 20P):** If a model is trained on significantly fewer tokens than recommended by the Chinchilla Rule, it may **underfit**. Even with many parameters, the model cannot fully extract knowledge from limited data, resulting in suboptimal performance. In this case, increasing training tokens yields greater benefit than increasing model size.
*   **Excessive Tokens (T >> 20P):** If a model is trained on excessively large data with relatively few parameters, computational resources may be **wasted inefficiently**. The model saturates on the data, and further increases in data volume yield negligible performance gains. In this case, increasing model size (number of parameters) is a more efficient way to improve performance.

**Practical Application and Limitations of the Chinchilla Rule:**

The Chinchilla Rule is a valuable **guideline** for planning training of large language models, especially under limited computational resources. It helps determine a reasonable balance between model size and training data volume to **optimize training and achieve the best possible performance**.

For example, if you have a fixed computational budget, the Chinchilla Rule can help decide whether to train a smaller model on more data or a larger model on less data.

**Important Notes:**

*   The Chinchilla Rule is **empirical**, not a strict mathematical law. The optimal ratio may vary slightly depending on model architecture, data quality, training methods, and other factors.
*   The Chinchilla Rule is **approximate**. It provides a good initial estimate but may require additional tuning and experimentation to find the true optimum for a specific task.
*   The Chinchilla Rule primarily targets **computational resource optimization** and achieving **maximum performance** under constraints.

</details> 

---

## Formalization of the Distillation Scaling Law

The central contribution of the study is the formulation of the distillation scaling law:

$$L_S(N_S, D_S, L_T) = L_T + \frac{1}{L_{c_0}^T} \left( 1 + \left( \frac{L_T}{\tilde{L}_S^{d_1}} \right)^{1/f_1} \right)^{-c_1f_1} \left( \frac{A}{N_S^{\alpha'}} + \frac{B}{D_S^{\beta'}} \right)^{\gamma'}$$

### Explanation of Variables:

*   $L_S(N_S, D_S, L_T)$ — **Student cross-entropy** (prediction error metric; lower is better).
*   $L_T$ — **Teacher cross-entropy** (prediction error metric of the large model).
*   $N_S$ — **Number of non-embedding parameters of the student** (core trainable model parameters).
*   $D_S$ — **Number of tokens** used to train the student during distillation.
*   $\tilde{L}_S = L(N_S, D_S)$ — **Potential student cross-entropy under standard training without distillation**, determined by the classical scaling law:

$$L(N, D) = E - \frac{A}{N^\alpha} - \frac{B}{D^\beta}$$

*   $\{c_0, c_1, d_1, f_1, \alpha', \beta', \gamma'\}$ — **Coefficients** determined empirically.
*   $A$ and $B$ — **Positive coefficients** dependent on model architecture and dataset characteristics.

### Physical Meaning of the Formula:

1. **Base Term**: $L_T$ — The student cannot outperform the teacher.
2. **Modifying Term**: The remaining part of the formula describes how effectively the student can approach the teacher, depending on its size, data volume, and teacher quality.

### Key Conclusions:

1. The student cannot surpass the teacher (always $L_S \geq L_T$). **Cross-entropy (L) is a measure of model error**. The **lower** the value of L, the **better** the model predicts data.
2. The closer the student’s potential performance is to the teacher’s, the more effective distillation becomes.
3. With a fixed teacher, the distillation scaling law does not exceed the standard scaling law.

### Practical Application:

This law enables optimal allocation of computational resources between teacher and student and predicts distillation effectiveness.

> **In other words**: This law describes how a small model’s quality depends on three factors: its own size, amount of training data, and the quality of the large teacher model. The key insight: a student can never be better than its teacher, but how closely it approaches the teacher depends on its own capacity and training volume.

## Mixing Coefficients in Knowledge Distillation

Having examined the general distillation scaling law, it is essential to understand practical implementation aspects, particularly how to balance imitation of the teacher against independent learning by the student model.

The core idea of knowledge distillation is transferring information from a large teacher model to a compact student model. In this process, the teacher’s predicted probability distribution serves as the target for the student. Training minimizes the Kullback-Leibler (KL) divergence between the student’s and teacher’s distributions:

$$
\mathcal{L}_{\text{KD}} \left( z_T^{(i)}, z_S^{(i)} \right) = -\tau^2 \sum_{a=1}^V \sigma_a \left( \frac{z_T^{(i)}}{\tau} \right) \log \sigma_a \left( \frac{z_S^{(i)}}{\tau} \right)
$$

where:
- $z_T^{(i)}$ and $z_S^{(i)}$ are the output logits of the teacher and student models, respectively
- $\tau$ is the distillation temperature, controlling the "smoothness" of the teacher’s probability distribution
- $\sigma_a$ is the softmax function converting logits to probabilities
- $V$ is the vocabulary size

The combined loss function for the student model integrates multiple components:

$$
\mathcal{L}_S\big(x^{(i)}, \boldsymbol{z}_T^{(i)},\boldsymbol{z}_S^{(i)}\big) = (1-\lambda)\,\mathcal{L}_{\textrm{NTP}}(x^{(i)},\boldsymbol{z}_S^{(i)}) + \lambda\,\mathcal{L}_{\textrm{KD}}(\boldsymbol{z}_T^{(i)},\boldsymbol{z}_S^{(i)}) + \lambda_Z\,\mathcal{L}_Z(\boldsymbol{z}_S^{(i)}).
$$

where:
- $\mathcal{L}_{\textrm{NTP}}$ — Next-token prediction loss (standard cross-entropy)
- $\mathcal{L}_{\textrm{KD}}$ — Knowledge distillation loss (KL divergence)
- $\mathcal{L}_Z$ — Regularization Z-loss, stabilizing training by normalizing logits
- $\lambda$ — Mixing coefficient, determining the balance between learning from "clean" data and imitating the teacher
- $\lambda_Z$ — Weight coefficient for Z-loss

## Experimental Determination of Optimal Distillation Parameters

To assess the impact of distillation parameters on scaling law effectiveness, researchers conducted a series of experiments. To isolate the role of the teacher model and exclude data effects, experiments were performed in "pure distillation" mode with $\lambda = 1$. Results showed this choice of $\lambda$ yields performance statistically comparable to using optimal $\lambda^*$ values.

In all experiments, a **fixed distillation temperature $\tau = 1$** was used, which empirically demonstrated the highest efficiency for training the student model.

![Mixing Coefficients λ](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_4.webp  )

> **Mixing coefficients $\lambda$.** (a) Six student models of sizes $N_S \in \{198M, 266M, \ldots, 2.72B\}$, trained with ratio $M = D_S/N_S = \text{Number of tokens used to distill the student} / \text{Number of student model parameters} = 20$, distilled from teacher models of sizes $N_T \in \{546M, 975M, \ldots, 7.75B\}$, trained with ratio $M = D_T/N_T = \text{Number of tokens used to pre-train the teacher} / \text{Number of teacher model parameters} = 20$, using various mixing coefficients $\lambda \in [0, 1]$. Values $\lambda = 0$ and $\lambda = 1$ correspond to standard training and pure distillation, respectively.  
(b) Optimal mixing coefficients $\lambda^* = \arg \min_{\lambda} \mathcal{L}(\lambda)$ yielding the lowest validation loss for each teacher-student pair.

These experiments confirm that distillation parameters significantly affect the final student model performance, and their optimal selection directly correlates with the sizes of the teacher and student models, consistent with the general distillation scaling law.

### Conclusion

Knowledge distillation is a method for transferring the capabilities of a large neural model (teacher) to a smaller, computationally efficient model (student). The process is based on training the student to mimic the teacher’s probability distribution by minimizing the Kullback-Leibler divergence between their predictions.

Distillation effectiveness is determined by the balance among several components in the loss function:
- Standard cross-entropy for next-token prediction
- KL divergence for teacher imitation
- Regularization Z-loss for training stabilization

Two key parameters control this process:
- Mixing coefficient $\lambda$, regulating the balance between independent learning and teacher imitation
- Distillation temperature $\tau$, influencing the "smoothness" of the probability distribution

<u>Experimental studies demonstrate that "pure distillation" mode ($\lambda = 1$) with temperature $\tau = 1$ often yields results comparable to optimally tuned parameters. However, the most important discovery is that ideal values of these parameters systematically depend on the size ratio of the specific teacher-student model pair.</u>

This discovery aligns with the general distillation scaling law and has direct practical implications: to achieve maximum efficiency in practical distillation, parameters must be individually tuned based on the sizes of the models used, significantly improving the final performance of the compact model while preserving its computational efficiency.

## Experiment with Fixed Teacher and Varying Students

The teacher model size and the volume of data on which the teacher was trained are fixed, while the student model size and distillation data volume vary. The goal is to study how the student model’s performance changes with its size and the volume of distillation data under a fixed teacher. Thus, optimal student performance under varying scales and data volumes can be determined.

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_5.webp  )

![Figure_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_6.webp  )

**From the experimental results, it can be observed that:**

- At high computational power, the larger the student model’s parameter scale, the lower its loss, and the more evident this trend becomes with larger teacher models.
- When student and teacher model sizes are fixed, it becomes clear that higher computational power leads to better student model performance.
- At low computational power, model performance first improves then deteriorates with increasing model size. Here, it is evident that larger models do not fully train under limited computational power.
- In special cases, the student model may surpass the teacher model and demonstrate superior generalization. I personally hypothesize that the teacher model may be undertrained in such scenarios.

## Experiment with Fixed Student and Varying Teachers

The student model size and distillation data volume are fixed, while the teacher model size and training data volume vary. The goal is to study how the teacher model’s effectiveness influences the final student model performance. Thus, the optimal teacher model size and training data volume for maximizing student performance can be determined.

![Figure_7](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_7.webp  )

As shown in the results, the larger the teacher model’s parameters, the lower the student’s cross-entropy. This indicates that for optimal distillation, the teacher model’s performance must match the student model’s capabilities.

## Distillation vs. Supervised Learning

To understand when distillation provides benefits, the following figure compares distillation and supervised learning performance under fixed computational resources. Results show that supervised learning always outperforms distillation when sufficient computation or data is available to the student. With moderate data budgets, distillation has advantages; however, with large data volumes, supervised learning surpasses distillation.

In summary, under limited computational resources, distillation is typically more efficient than supervised learning. This is because distillation can rapidly absorb efficient feature representations under the teacher’s guidance, achieving higher performance with fewer computational resources.

![Figure_8](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_8.webp  )

## Teacher Model Selection

- **Strength of Training Signal**: Teacher models of different sizes may provide different strengths of training signals, typically measured by cross-entropy loss. Larger teacher models can provide stronger training signals (lower cross-entropy), helping the student learn better.
- **Increased Cost**: Using a larger teacher model incurs higher costs due to the need to compute the teacher’s logits. This means larger teacher models are not only more expensive to train but also consume more computational resources during distillation.

The figure below shows the change in student cross-entropy loss under varying distillation data budgets. Results show that the optimal teacher loss (red line) decreases following a power law as student size increases until the student’s loss matches the optimal teacher loss.

![Figure_9](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_9.webp  )

As shown in the figure below, as the distillation data volume increases, the cross-entropy of the optimal teacher model gradually decreases. Thus, we conclude: when computational resources are limited, selecting a smaller teacher model can reduce inference cost while still providing effective training signals to the student model.

![Figure_10](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_10.webp  )

## Compute Optimal Distillation

The goal of computationally optimal distillation is to determine how to create a student model of desired size with the lowest possible cross-entropy under a given computational budget. Specifically, we must find the optimal distillation data volume, teacher model size, and teacher training data to minimize student cross-entropy while satisfying computational budget constraints.

In the figure below we see:

- Supervised learning always corresponds to the best distillation configuration when computational budget is sufficient: Supervised learning always matches the best distillation configuration under a fixed total computational budget. This means supervised learning can achieve the same performance as distillation if the computational budget is large enough.

- If teacher training is included in computations, student cross-entropy is always higher than in supervised settings: This means if your sole goal is to create the best possible model with a target size and you have no access to a teacher, you should choose supervised learning instead of training a teacher followed by distillation. Conversely, if the goal is to produce a family of models or use the teacher as a serving model, distillation may be more computationally advantageous than supervised learning.

- Smaller models are more likely to benefit from supervised pre-training, while larger models are more likely to benefit from distillation: Smaller models are more likely to benefit from supervised learning under large computational budgets, while larger models are more likely to benefit from distillation under large computational budgets.

![Figure_11](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_11.webp  )

The figure below shows trends in optimal teacher size and training data volume as computational budget changes. Student and teacher model tokens scale according to power laws, with student tokens growing faster. The optimal teacher model size first increases until it becomes slightly larger than the student, then stabilizes. This occurs because **using a large teacher model for inference is expensive, and as student token count increases, retraining the teacher becomes more efficient**.

![Figure_12](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-11/assets/Figure_12.webp  )

### **Key Research Findings**

Based on their research, the authors reached the following conclusions:

1. **Predictability of Performance via Scaling Law**: The performance of a student model of size $N_S$, obtained by distillation from a teacher model of size $N_T$ using $D_S$ tokens, can be predicted using the developed distillation scaling law.

   > **Practical Value**: This allows early estimation of distillation outcomes without costly experiments. Companies can plan resources and decide whether to invest in distillation or choose an alternative approach to building an efficient model.

2. **Impact of Teacher Parameters on Student**: The teacher model size $N_T$ and its training token count $D_T$ determine the teacher’s cross-entropy $L_T = L_T(N_T, D_T)$, which in turn affects the student’s cross-entropy.

   > **Illustrative Example**: Imagine the teacher as a source of knowledge for the student. If the teacher itself is inadequately trained (high cross-entropy), it cannot effectively teach the student, regardless of the student’s capabilities.

3. **The "Capacity Gap" Phenomenon**: The study revealed an interesting effect—stronger teachers can lead to worse students, explained by a "capacity gap." The effect of teacher cross-entropy on student loss follows a power law that switches between two regimes depending on the relative learning capacity of student and teacher. The study showed that the critical factor is the *gap in learning capacity* (hypothesis space and optimization capability) between teacher and student, not merely their relative size.

   > **Analogy for Understanding**: Imagine a quantum physics professor trying to teach a first-grader. Despite the professor’s expertise, the child cannot absorb complex material due to the gap in learning capacity. Similarly, if the teacher model is too complex and "thinks" at a level inaccessible to the student model, training efficiency decreases.

4. **U-shaped Student Error Dependency**: Empirically confirmed is a U-shaped dependence of student error on teacher size for a fixed student size, theoretically justified by the capacity gap between them.

   > **Visual Representation**: If student error is plotted against teacher size on a graph, a U-shaped curve emerges. This means there exists an optimal teacher size for a given student—neither too small (insufficient knowledge) nor too large (overly complex knowledge representation).

### **Practical Recommendations**

The study results show that distillation becomes more effective than teacher training under the following conditions:

1. The total number of computations or tokens for the student does not exceed a threshold tied to the student’s size, according to the new scaling law.

   > **Practical Scenario**: For a company with limited computational budget seeking to build a 1-billion-parameter model, distillation may be optimal if fewer than 20 billion training tokens are available (per the Chinchilla Rule).

2. The teacher model already exists, or training the teacher model has applications beyond a single distillation.

   > **Business Case**: If a company has already trained a large model for its core tasks, it makes sense to use it for distilling smaller, specialized models for deployment on mobile devices or resource-constrained environments.

If both training processes (teacher and student) have sufficient data or computation, distillation cannot achieve lower cross-entropy than supervised training.