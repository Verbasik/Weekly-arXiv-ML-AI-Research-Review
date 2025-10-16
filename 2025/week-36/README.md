# Schema Guided Reasoning (SGR) and SG²: Structured Reasoning and Multi-Agent Scenes 🚀

[![arXiv](https://img.shields.io/badge/arXiv-2502.03450-b31b1b.svg)](https://arxiv.org/abs/2502.03450)

## 📝 Description

This week we examine **Schema Guided Reasoning (SGR)** — a structured prompting method that guides LLM reasoning through typed schemas and explicit workflows. We also dissect its extension for spatial reasoning — **SG² (Schema-Guided Scene-Graph Reasoning)**, a multi-agent "reason-while-retrieve" framework for scene graph tasks. These approaches demonstrate a 5–10% accuracy gain and achieve 95%+ reproducibility while reducing hallucinations through schema validation and programmatic fact extraction.

## 🔍 Key Features

- **Structured outputs**: Typed schemas (JSON Schema / Pydantic) enforce format and semantic integrity of responses.
- **Three reasoning patterns**: Cascade, Routing, Cycle — for different task types and control over reasoning steps.
- **Constrained decoding**: CFG/grammar restrictions for safe generation, automatic retries upon validation.
- **Multi-agent SG² architecture**: Separation into a Reasoning module and a Retrieval module with programmatic graph access.
- **Programmatic retrieval**: Generation of Python code for scene-graph traversal instead of rigid APIs.
- **Reduced hallucinations**: Context separation and schema-guided navigation minimize distractions and erroneous conclusions.
- **Compatibility**: OpenAI Structured Outputs, Instructor, LangChain, Pydantic AI, local backends (xgrammar/Outlines/etc.).

## 📈 Results and Comparisons

| Characteristic | SGR | CoT | ReAct | ToT | Plan-and-Solve |
|---|---|---|---|---|---|
| Reproducibility | 95%+ | 70–85% | 60–80% | 50–70% | 75–85% |
| Structuredness | Strict (schemas) | Freeform (prompt) | Cyclical | Trees | Two-phase |
| GSM8K (benchmark) | 85–92% | 40–58% | 65–75% | ~74% | 78–82% |
| Technical complexity | 5/10 | 2/10 | 6/10 | 9/10 | 3/10 |

In experiments, SG² outperforms baselines: achieving up to **98%** in numerical tasks in BabyAI (vs. ~86% for ReAct) and **96–97%** in planning. Even with small models (e.g., Phi4‑14B), it retains an advantage: **~60%** vs. <30% for baseline approaches.

## 🧠 SG² Architecture Briefly

- **Task Planner**: Formulates retrieval queries and coordinates solution flow.
- **Tool Caller / Code Generator**: Generates executable Python code for scene graph traversal.
- **Verifier**: Ensures retrieved facts satisfy the query/schema requirements.
- **Scene as schema**: Node types, attributes, and edges guide both reasoning and retrieval.

![](assets/Image-01.png)
![](assets/Image-02.png)

## 🌟 Practical Applications

- Spatial planning and navigation (robotics, VirtualHome, simulations).
- Extraction of structured facts from complex environments and large graphs.
- Enterprise use cases: Auditable reasoning pipelines with quality control and compliance adherence.

## 📜 Citation

```bibtex
@misc{sgr_scene_graph_2025,
  title={Schema-Guided Scene-Graph Reasoning based on Multi-Agent Large Language Model System},
  year={2025},
  eprint={2502.03450},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>