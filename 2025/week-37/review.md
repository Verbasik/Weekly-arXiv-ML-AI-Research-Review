# Mathematical Foundations of DeepConf: Enhancing Reasoning through Confidence Estimation

## Conceptual Basis of DeepConf

Deep Think with Confidence (DeepConf) is a simple yet effective method that **eliminates the need for repeated generation of full reasoning chains** through an elegant use of the model's internal confidence signals. The method significantly improves both reasoning efficiency and computational performance of large language models during inference.

The fundamental distinction of DeepConf from classical parallel thinking approaches lies in its ability to **dynamically filter low-quality reasoning traces** both during (online) and after (offline) generation, without requiring additional model training or hyperparameter tuning.

## Confidence Metrics as Reasoning Quality Signals

![Figure 5: DeepConf with parallel thinking rejects low-confidence reasoning traces during generation to achieve higher reasoning performance while using significantly fewer generated tokens.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-05.png)

![Figure 6: Measuring confidence and confident reasoning in offline mode.](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-37/assets/Image-06.png)

The foundation of DeepConf lies in the use of various confidence metrics extracted from the model's next-token probability distribution. Formally, the following key metrics can be identified:

### 1. Token Entropy

$$H_i = -\sum_{j} P_i(j) \log P_i(j)$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$H_i$** — entropy of the token probability distribution at position $i$.
- **$P_i(j)$** — probability of the $j$-th token from the vocabulary at position $i$.

---

  📐 Logarithm Base

  Variants and interpretation:

  1. Natural logarithm (ln):
    - Formula: $H_i = -\sum_{j} P_i(j) \ln P_i(j)$
    - Units: nats
    - Context: Thermodynamics/physical interpretation
  2. Logarithm base 2 (log₂):
    - Formula: $H_i = -\sum_{j} P_i(j) \log_2 P_i(j)$
    - Units: bits
    - Advantage: Direct comparability with Shannon information measures

  Recommendation: log₂ — for compatibility with bit-based metrics and confidence comparisons.

  🧮 Vocabulary Coverage

  Calculation variants:
  - Full vocabulary: exact entropy, accounts for the entire distribution tail.
  - Top-K approximation: compute using top-K with normalization $\tilde{P}_i(j) = \frac{P_i(j)}{\sum_{k \in \text{top-K}} P_i(k)}$.

  Trade-offs:
| **Full vocabulary**              | **Top-K approximation**            |
|----------------------------------|------------------------------------|
| 🎯 Accurate, sensitive to tail   | ⚡ Faster, no full softmax required |
| 🧵 Higher sensitivity to noise   | 🛡️ More robust to low probabilities |
| ⏳ Computationally expensive     | 📦 Simple production implementation |

  ⚠️ Epsilon (Numerical Stability)

  Problem: $\log(0) = -\infty$ for zero probability.

  Solution: $\log(\max(P_i(j), \epsilon))$ with small $\epsilon$.

  Typical values:
  - eps=1e-12: Minimal impact
  - eps=1e-8: Standard for PyTorch
  - eps=1e-6: Slightly more aggressive, better for coarse approximations

  ★ Insight ─────────────────────────────────────

  Entropy measures the "overall uncertainty" of the distribution; useful as a global indicator of reasoning step complexity. On tasks with long tails, the difference between full and top-K entropy can be substantial — account for this when comparing models.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. log₂ (bits) for compatibility with other metrics
  2. Full vocabulary if available; otherwise top-50 with normalization
  3. eps=1e-8 for stability

---

</details>

---

The entropy of distribution $P$ is a measure of its **uncertainty**. Lower entropy indicates a more "concentrated" probability distribution and higher model confidence. High entropy means the probability mass is "spread out" across many tokens, indicating low model confidence.

For example, consider predicting the next token in the context:
- **Context:** `"The Pythagorean theorem states that in a right triangle, the sum of the squares of ___"`
- **Possible tokens:** `"legs"` (0.9), `"hypotenuse"` (0.05), `"sides"` (0.05)

Entropy will be low ($H_i \approx 0.47$), indicating high model confidence. If the distribution is close to uniform, e.g., `"legs"` (0.4), `"hypotenuse"` (0.3), `"sides"` (0.3), entropy will be high ($H_i \approx 1.57$), indicating low model confidence.

### 2. Token Confidence

$$C_i = -\frac{1}{k}\sum_{j=1}^{k} \log P_i(j)$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$C_i$** — model confidence at generating the token at position $i$.
- **$k$** — number of top tokens considered.
- **$P_i(j)$** — probability of the $j$-th top token from the vocabulary.

---

  🔢 Parameter $k$ (number of top tokens)

  What it does: Determines how many of the most probable tokens are included in the formula $C_i = -\frac{1}{k}\sum_{j=1}^{k} \log P_i(j)$.

  Impact on results:
  - k=5: Focus only on top-5 tokens → high sensitivity to dominant candidates
  - k=10: Balance between accuracy and robustness → recommended for starters
  - k=20: Includes more alternatives → less sensitive to noise, but may include insignificant tokens

  Trade-offs:
| **Small `k` (e.g., 5)**         | **Large `k` (e.g., 20)**          |
|----------------------------------|-----------------------------------|
| 🔍 High sensitivity              | 🛡️ Robust to noise                |
| 🎯 Focused on top choices        | 🌐 Accounts for more alternatives |
| ⚡ Fast computation              | ⏳ Slower computation             |

  📐 Logarithm Base

  Variants and interpretation:

  1. Natural logarithm (ln):
    - Formula: $C_i = -\frac{1}{k}\sum_{j=1}^{k} \ln P_i(j)$
    - Units: nats (natural units)
    - Connection to information theory: Energy/thermodynamics
  2. Logarithm base 2 (log₂):
    - Formula: $C_i = -\frac{1}{k}\sum_{j=1}^{k} \log_2 P_i(j)$
    - Units: bits
    - Advantage: Direct comparability with Shannon entropy!

  Recommendation: log₂ for better comparability with entropy.

  ⚠️ Epsilon (Numerical Stability)

  Problem: $\log(0) = -\infty$ when a token has zero probability.

  Solution: $\log(\max(P_i, \epsilon))$ where $\epsilon$ is a small number.

  Typical values:
  - eps=1e-12: Very conservative, minimal impact
  - eps=1e-8: Standard for PyTorch computations
  - eps=1e-6: More aggressive protection

  ★ Insight ─────────────────────────────────────
  
  Relationship to entropy: If using log₂, both confidence and entropy will be in the same units (bits), simplifying comparative analysis. Entropy shows "overall uncertainty," while confidence reflects the model's "decisiveness" among top alternatives.

  k=10 — the sweet spot: sufficient to capture main alternatives without including noise from the long tail.
  ─────────────────────────────────────────────────

  🎯 My starter recommendations:

  1. k=10 (balance of accuracy and performance)
  2. log₂ (comparability with entropy in bits)
  3. eps=1e-8 (standard PyTorch protection)

  Rationale: These parameters yield interpretable results comparable to entropy and cover core cases without excessive complexity.

---

</details>

Token confidence is the negative average of the logarithmic probabilities of the top-$k$ tokens. This metric quantitatively defines how confident the model is in its prediction. A high value of $C_i$ corresponds to peaked distributions and greater model confidence, while a low value indicates uncertainty in the token prediction.

A key distinction from entropy is that token confidence considers only the top-$k$ most probable tokens, ignoring the distribution tail. This makes the metric more robust to noise in low-probability tokens and better reflects the model's "decisiveness" when choosing among the most likely alternatives.

### 3. Group Confidence

$$C_{G_i} = \frac{1}{|G_i|} \sum_{t \in G_i} C_t$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$C_{G_i}$** — confidence of token group $G_i$.
- **$G_i$** — group of tokens consisting of $n$ previous tokens with overlapping sliding windows.
- **$|G_i|$** — number of tokens in group $G_i$.
- **$C_t$** — confidence of token $t$.

---

  🪟 Window length `n` and step `s`

  What it does: Defines locality and degree of signal smoothing.

  Impact on results:
  - Small `n` (e.g., 32–128): High local sensitivity, more fluctuations
  - Medium `n` (512–2048): Balance of locality and robustness → recommended
  - Large `n` (4096+): Strong smoothing, worse at detecting brief confidence dips

  Window step `s`:
  - Fine step (e.g., 1–16): Precise tracking, computationally expensive
  - Coarse step (e.g., n/2): Faster, but coarser estimation

  Trade-offs (window):
| **Small `n`**                   | **Large `n`**                     |
|---------------------------------|-----------------------------------|
| 🔍 Responds to short dips       | 🛡️ Resistant to noise             |
| 🎯 Better for early stopping    | 🌐 Loses local details            |
| ⚡ Fast to recalculate           | ⏳ More expensive, more overlap   |

  🧮 Window Aggregator

  Options: mean (default), median, trimmed-mean (e.g., 10%).

  - Mean: Sensitive to outliers, good for early problem detection
  - Median: More robust, less sensitive
  - Trimmed-mean: Compromise between mean and median

  🔗 Inherited parameters

  - `k`, log base, `eps` — same as for $C_t$ (Section 2)

  ★ Insight ─────────────────────────────────────

  Group confidence translates the model's "pointwise decisiveness" into localized reasoning context and is especially useful for online stopping: a brief "stall — stop" signal instead of waiting for the trace to complete.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. Window n=2048, step s=128–256
  2. Aggregator: mean; for noisy traces — trimmed-mean 10%
  3. Inherit k=10, log₂, eps=1e-8

---

</details>

Group confidence provides a more localized and smoothed signal by averaging token confidence across overlapping reasoning intervals. This approach enables the identification of problematic segments in the reasoning chain where the model becomes less confident.

For example, if the model begins to hesitate and generates phrases like "wait, let me check," or "no, I made a mistake," group confidence in this segment drops sharply. This is a more reliable indicator of reasoning issues than the average confidence across the entire trace, which may be diluted by highly confident segments elsewhere.

### 4. Bottom-10% Group Confidence

$$C_{\text{bottom-10}}(t) = \frac{1}{|G_b|} \sum_{G_j \in G_b} C_{G_j}$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$C_{\text{bottom-10}}(t)$** — metric considering only the least confident token groups.
- **$G_b$** — set of groups with the lowest 10% confidence values in the trace.
- **$|G_b|$** — number of groups in $G_b$.

---

  📉 Bottom group fraction `p`

  What it does: Determines what proportion of "weak" windows to consider (typically 10%).

  Impact on results:
  - Small `p` (5%): Strong focus on extreme dips, high sensitivity
  - Medium `p` (10%): Balance between sensitivity and robustness → recommended
  - Large `p` (20–30%): More robust, but may blur problem signals

  🧮 Aggregator over bottom groups

  Options: mean (default), median over bottom `p%`.
  - Mean: better captures overall "weakness" of problematic segment
  - Median: more robust to single outliers

  Trade-offs:
| **Small `p`**                   | **Large `p`**                     |
|---------------------------------|-------------------------------------|
| 🔍 Captures critical dips       | 🛡️ Resistant to random noise       |
| 🎯 Good for early cutoff        | 🌐 May "average out" weakness      |
| ⚠️ Higher risk of false alarms  | ⏳ Poorer at detecting brief dips  |

  ★ Insight ─────────────────────────────────────

  "A chain is as strong as its weakest link." Averaging over the bottom `p%` provides a stable compromise between "min" and "mean over all."
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. p=10%
  2. Aggregator: mean over bottom groups
  3. Group window: n=2048, step s=128–256

---

</details>

---

Bottom-10% group confidence focuses on the most problematic segments of reasoning. Researchers found that reasoning quality is often determined by its weakest links — small segments where the model loses confidence or makes errors.

This metric computes the average confidence only for the 10% least confident groups in the trace. Thus, even if most of the reasoning appears confident, a critical moment where the model begins to "wander" or vacillate will be captured by this metric.

### 5. Least Confident Group Confidence

$$C_{\text{least}}(t) = \min_{G_j \in G} C_{G_j}$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$C_{\text{least}}(t)$** — confidence of the least confident token group in the trace.
- **$G$** — set of all token groups in the reasoning trace.

---

  🔻 Min selection and smoothing

  Variants:
  - Hard min: maximum sensitivity to local failures
  - Soft-min: mean over bottom `k_min` windows (e.g., 3–5) — reduces impact of single outliers
  - Quantile: use `q`-th percentile (e.g., 2–5%) instead of strict minimum

  Trade-offs:
| **Hard min**                    | **Soft-min / Quantile**             |
|---------------------------------|-------------------------------------|
| 🔍 Maximum sensitivity          | 🛡️ Robust to single spikes         |
| 🎯 Earlier stopping trigger     | 🌐 More stable trace ranking       |
| ⚠️ More false stops             | ⏳ Slightly more computationally expensive |

  🔗 Inherited parameters
  - Group window n, step s, k/log₂/eps for $C_t$

  ★ Insight ─────────────────────────────────────

  For online stopping, "group minimum" achieves the best correlation with trace quality, but soft variants (soft-min) reduce false stops without notable loss of sensitivity.
  
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. Soft-min over bottom k_min=3 windows
  2. Group window n=2048, step s=128–256
  3. Inherit k=10, log₂, eps=1e-8

---

</details>

---

Least confident group confidence is the extreme case of the "bottom-10%" metric, considering only the absolute minimum confidence across the entire reasoning trace. This metric assumes that reasoning quality cannot exceed the quality of its weakest link.

Using the minimum instead of the mean makes this metric especially sensitive to local confidence drops, enabling effective identification of traces with critical reasoning errors. This metric proved most effective for early stopping in DeepConf's online mode.

### 6. Tail Confidence

$$C_{\text{tail}}(t) = \frac{1}{|T_{\text{tail}}|} \sum_{t \in T_{\text{tail}}} C_t$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$C_{\text{tail}}(t)$** — average confidence of the final tokens in the trace.
- **$T_{\text{tail}}$** — fixed number of tokens at the end of the sequence (e.g., 2048).

---

  🦺 Tail length `L`

  What it does: Determines how much the "end of reasoning" influences the metric.

  Impact on results:
  - Small `L` (128–512): More sensitive to final failures, higher variance
  - Medium `L` (1024–2048): Balance of sensitivity and robustness → recommended
  - Large `L` (4096+): Smoothes tail, but dilutes final signals

  🧾 Special token handling

  - Exclude `eos`/stop tokens from tail
  - For incomplete traces: use available `min(L, len)` tokens

  Trade-offs:
| **Small `L`**                   | **Large `L`**                     |
|---------------------------------|-------------------------------------|
| 🔍 Captures final errors        | 🛡️ More robust to noise            |
| 🎯 Useful for online stopping   | 🌐 Less sensitive to endings       |
| ⚠️ Higher risk of fluctuations  | ⏳ More expensive for large L      |

  ★ Insight ─────────────────────────────────────

  Final steps are often critical in mathematics. "Tail confidence" is a targeted metric for controlling the quality of reasoning completion.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. L=2048
  2. Exclude `eos`/stop
  3. Inherit k=10, log₂, eps=1e-8

---

</details>

---

Tail confidence evaluates reasoning reliability by focusing on its concluding portion. This metric is motivated by the observation that reasoning quality often degrades toward the end of long thought chains, and final steps are crucial for correct conclusions.

In mathematical reasoning, the final answer and concluding steps are especially critical: traces that start strongly but end weakly may lead to incorrect results despite promising intermediate reasoning.

## DeepConf Algorithm: Offline and Online Modes

DeepConf operates in two primary modes: offline and online. In offline mode, all reasoning traces are already generated; in online mode, DeepConf can dynamically interrupt the generation of low-quality traces.

### Offline Mode (Offline Thinking with Confidence)

1. **Confidence-Weighted Majority Voting:**

Instead of equal weighting of all traces, each final answer is weighted by the confidence of its corresponding trace:

$$V(a) = \sum_{t \in T} C_t \cdot I(\text{answer}(t) = a)$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$V(a)$** — weighted count of votes for answer $a$.
- **$T$** — set of all generated traces.
- **$C_t$** — trace confidence, computed using one of the confidence metrics.
- **$I(\text{answer}(t) = a)$** — indicator function, equal to 1 if trace $t$'s answer matches $a$, and 0 otherwise.

---

  ⚖️ Weight function `w = f(C_t)`

  Transformation options:
  - Linear: `w = C_t` (baseline)
  - Temperature: `w = exp(C_t / T)` with `T∈[2,8]` — amplifies differences between confident/unconfident traces
  - Clipping: `w = clip(C_t, a, b)` — truncates extreme values

  🧮 Weight normalization

  - Per-trace: `w ← w / mean(w)` to stabilize scale
  - Per-answer: normalize sums within answer space for comparability

  🤝 Tie-breaking

  - When weights are equal: choose answer with lower tail entropy or higher median group confidence

  Trade-offs:
| **Linear weights**              | **Temperature amplification**       |
|---------------------------------|-------------------------------------|
| 🛡️ More stable on noise         | 🎯 Better isolates strong traces   |
| 🌐 Lower risk of over-amplification | ⚠️ Sensitive to choice of `T`      |

  ★ Insight ─────────────────────────────────────

  Transforming weights via soft amplification (exp/temperature) can significantly improve consensus on adjacent answers, but requires calibration on validation data.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. `w = C_t` (no transformation)
  2. Ties — resolve by lower tail entropy
  3. With strong spread: `clip(C_t, p5, p95)`

---

</details>

---

В отличие от стандартного мажоритарного голосования, где каждая трасса имеет равный вес, взвешенное голосование учитывает уверенность модели в каждой трассе. Это позволяет отдавать предпочтение ответам, которые поддерживаются более уверенными рассуждениями, даже если таких трасс меньше.

Например, если 6 трасс с низкой уверенностью (по 0.4) дают ответ "103", а 4 трассы с высокой уверенностью (по 0.9) дают ответ "109", то взвешенное голосование выберет "109" с общим весом 3.6 против 2.4 для "103", в то время как простое мажоритарное голосование выбрало бы "103".

2. **Фильтрация по уверенности (Confidence Filtering):**

Отбираются только трассы с наибольшей уверенностью (например, топ-10% или топ-90%):

$$\hat{a} = \arg\max_a V(a)$$

<details> 
    <summary><em><strong>пояснение переменных</strong></em></summary>

где:
- **$\hat{a}$** — итоговый выбранный ответ с наибольшим весом.

---

  ✂️ Стратегии фильтрации по уверенности

  Варианты:
  - Quantile: оставить top-`η%` по уверенности (DeepConf-low/high)
  - Threshold: абсолютный порог `C_t ≥ s`

  Порядок операций:
  - Сначала фильтрация, затем взвешенное голосование → меньше шума

  Trade-offs:
| **Top-η%**                      | **Абсолютный порог**                |
|---------------------------------|-------------------------------------|
| 🎯 Контролируемая доля трасс    | 🛡️ Стабилен межзадачно              |
| 🌐 Адаптивно к задаче           | ⚠️ Требует подбора `s`              |

  ★ Insight ─────────────────────────────────────

  На разнородных задачах квантили обгоняют фиксированные пороги; в стабильных доменах абсолютный порог упрощает эксплуатацию.
  ─────────────────────────────────────────────────

  🎯 Рекомендации для старта:

  1. DeepConf-low: top-10%; DeepConf-high: top-90%
  2. Фильтрация → взвешенное голосование
  3. При дефиците трасс: переход на абсолютный порог

---

</details>

Confidence filtering enables focusing solely on the most reliable traces while excluding potentially erroneous or convoluted reasoning. DeepConf incorporates two primary filtering strategies:

- **DeepConf-high (top-90%)**: A conservative approach that discards only the 10% least confident traces. This ensures a good balance between accuracy and reasoning diversity.

- **DeepConf-low (top-10%)**: An aggressive approach that retains only the 10% most confident traces. This often yields the highest accuracy gains but may occasionally reduce precision due to excessive concentration on a limited set of reasoning traces.

### Online Mode (Online Thinking with Confidence)

In online mode, DeepConf uses least-confident group confidence to dynamically interrupt generation:

1. **Warmup (Offline Warmup):**
   - Generate $N_{init}$ (e.g., 16) full reasoning traces.
   - Set the stopping threshold $s$ based on the percentile of confidence:
   
   $$s = \text{Percentile}_{100-\eta}(\{C_t : t \in T_{warmup}\})$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$s$** — threshold value for stopping generation.
- **$\eta$** — percentage of traces we wish to retain (e.g., 10% or 90%).
- **$T_{warmup}$** — set of all warmup traces.
- **$C_t$** — confidence of trace $t$.

---

  🔧 Warmup parameters

  - `N_init` (number of traces): 8–32; more → more stable threshold, longer warmup
  - `η` (selection percentile): 10% for DeepConf-low, 90% for DeepConf-high
  - `s` update: static (based on warmup) or adaptive (sliding window during generation)

  Trade-offs:
| **Small `N_init`**              | **Large `N_init`**                |
|---------------------------------|-------------------------------------|
| ⚡ Fast startup                 | 🛡️ Stable threshold               |
| ⚠️ Risk of threshold bias      | ⏳ More time-consuming             |

  ★ Insight ─────────────────────────────────────

  Adaptive `s` updates via sliding window reduce the risk of "overfitting" to initial traces on long tasks.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. `N_init=16`, `η=10%/90%`
  2. Fix `s` from warmup; enable adaptation for long prompts
  3. Compute threshold using "soft-min group" metric

---

</details>

---

The warmup phase is necessary to determine the stopping threshold based on the confidence distribution for a specific task. For each new query, DeepConf first generates a small number of full reasoning traces to "understand" what level of confidence is typical for that task.

For DeepConf-low, the threshold is set at the 90th percentile (traces below this level are discarded); for DeepConf-high, it is set at the 10th percentile (nearly all traces are retained, except the least confident).

2. **Adaptive Sampling:**
   - During generation of a new trace, if group confidence $C_{G_i}$ falls below threshold $s$, generation is halted.
   - DeepConf dynamically adjusts the number of generated traces based on task complexity, assessed via the consensus coefficient:
   
   $$\beta = \frac{V(\hat{a})}{\sum_a V(a)}$$

<details> 
    <summary><em><strong>Variable explanations</strong></em></summary>

where:
- **$\beta$** — consensus coefficient.
- **$V(\hat{a})$** — weighted count of votes for the most popular answer.
- **$\sum_a V(a)$** — total weighted votes.

---

  🚦 Online generation control

  Parameters:
  - Window for $C_{G_i}$: n=2048, step s=128–256
  - "Patience": halt after `m` consecutive windows below `s` (e.g., m=2–3)
  - Consensus threshold `τ`: 0.9–0.98; higher → earlier stop on agreement
  - Budget `B`: maximum traces (e.g., 256/512)

  Trade-offs:
| **Low `τ`/small `m`**           | **High `τ`/large `m`**            |
|---------------------------------|-------------------------------------|
| 🎯 Early termination            | 🛡️ Higher answer confidence       |
| ⚡ Saves tokens                  | ⏳ Longer to reach consensus       |

  ★ Insight ─────────────────────────────────────

  Combining "soft-min over windows + patience" significantly reduces false stops without sacrificing token savings.
  ─────────────────────────────────────────────────

  🎯 Starter recommendations:

  1. `n=2048, s=128–256, m=2`
  2. `τ=0.95`, `B=512`
  3. Stop on confidence drop below `s` and sufficient consensus

---

</details>

---

During online generation, DeepConf monitors the model's group confidence within a sliding window (typically 2048 tokens). If confidence falls below the established threshold, it indicates the model has begun to "hesitate" in its reasoning, and generation of that trace is prematurely halted.

This significantly conserves computational resources by avoiding the generation of full traces that would likely be filtered out during weighted voting.

3. **Generation Termination:**
   - If $\beta \geq \tau$ (where $\tau$ is the consensus threshold, e.g., 0.95), generation of new traces ceases.
   - Otherwise, generation continues until the fixed budget $B$ is reached.

Adaptive termination based on consensus allows DeepConf to conclude early if the model achieves high agreement on the final answer. This is especially beneficial for simple tasks where generating all $B$ traces is unnecessary.

## Concrete Example of DeepConf in Action

Consider how DeepConf operates on a mathematical problem:

### Scenario for Example

* **Task:** "Find the number of integer solutions (x, y) with 1 ≤ x, y ≤ 100, where x² + y² = z² for some positive integer z."
* **Group size:** $G = 512$ reasoning traces.
* **Confidence metric:** Group confidence with a 2048-token window.
* **Model:** Large language model (e.g., GPT-OSS-120B).

<details> 
    <summary><em><strong>Example: How DeepConf works "under the hood"</strong></em></summary>

### Example: How DeepConf Operates Internally

#### **Step 1: Generate initial traces and compute threshold (Offline Warmup)**

DeepConf generates $N_{init} = 16$ full reasoning traces. For each trace, group confidence with a 2048-token window is computed. Assume confidence values range from 11 to 18.

For DeepConf-low (top-10%):
- The 90th percentile is computed: $s_{low} = 16.5$

For DeepConf-high (top-90%):
- The 10th percentile is computed: $s_{high} = 12.8$

#### **Step 2: Online generation with early stopping**

Generation of new traces begins. Consider three examples:

**Trace 1:**
```
Consider the problem step-by-step. Step 1: The equation x² + y² = z² represents Pythagorean triples.
All primitive triples can be generated using the formula x = m² - n², y = 2mn, z = m² + n²...
```

As confidence remains high (e.g., $C_{G_i} \approx 17.3 > s_{low}$), generation continues to completion.

**Trace 2:**
```
Consider the problem step-by-step. Step 1: The equation x² + y² = z² represents Pythagorean triples.
All primitive... Wait, I need to verify the results... I must recheck step 1... 
```

At this point, group confidence drops ($C_{G_i} \approx 11.5 < s_{low}$), and DeepConf-low halts generation of this trace.

**Trace 3:**
```
Consider the problem step-by-step. Step 1: We are seeking solutions to the equation x² + y² = z²...
```

The confidence of this trace ($C_{G_i} \approx 13.5$) exceeds DeepConf-high’s threshold ($s_{high} = 12.8$) but falls below DeepConf-low’s threshold ($s_{low} = 16.5$). Thus, DeepConf-high continues, while DeepConf-low stops.

#### **Step 3: Confidence-weighted voting with filtering**

After generating sufficient traces (or achieving high consensus), DeepConf performs confidence-weighted voting with filtering:

1. Traces are filtered based on the chosen strategy (top-10% or top-90% by confidence).
2. For each unique answer (e.g., "109"), a weighted vote sum is computed.
3. The answer with the highest weight is selected.

For example, if answer "109" receives the highest weight $V(109) = 17$, it is selected as the final answer.

#### **Step 4: Comparison with baseline methods**

Compared to standard majority voting, DeepConf achieves:

1. Higher accuracy: 99.9% vs. 97.0% (cons@512) on AIME 2025 with GPT-OSS-120B.
2. Fewer tokens: 84.7% reduction for DeepConf-low and 56.0% for DeepConf-high.

This occurs because DeepConf effectively filters low-quality reasoning traces and concentrates on the most confident ones.

</details>

---

## Key Advantages of DeepConf

1. **Computational Efficiency**: DeepConf reduces generated tokens by 43–85% (DeepConf-low) and 18–59% (DeepConf-high) while maintaining or improving accuracy compared to majority voting.

2. **Improved Accuracy**: On complex tasks like AIME 2025, DeepConf@512 achieves up to 99.9% accuracy versus 97.0% for standard majority voting and 91.8% for single-pass generation.

3. **Simple Integration**: DeepConf requires no additional model training or hyperparameter tuning and can be integrated into existing LLM serving frameworks.

4. **Flexibility**: The method offers two operational modes (offline and online) and multiple filtering strategies (DeepConf-low and DeepConf-high), enabling customization of the accuracy-efficiency trade-off.

## Practical Implementation

DeepConf can be implemented with minimal modifications to standard LLM serving libraries such as vLLM. Required changes include:

1. Extending the log-probability handler to compute and maintain a sliding window of confidence.
2. Adding an early-stopping condition based on confidence.
3. Implementing filtering and weighted voting for the final answer.

These changes can be integrated into model serving APIs, enabling efficient deployment of DeepConf in production systems.