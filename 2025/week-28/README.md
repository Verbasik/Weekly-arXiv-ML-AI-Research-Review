[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue  )](https://t.me/TheWeeklyBrief  )

# OpenAI Agents SDK: Quick Start or Expensive “Black Box”?

> A deep dive into the architecture of OpenAI’s agent tools by The Weekly Brief team. Learn when to choose this tool—and when to opt for alternatives like LangChain.

🚀 **New release in the [#AgentsUnderHood](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/tree/develop/agents-under-hood/openai-cs-agents-demo  ) series!**

---

## 🔍 Key Takeaways

* **Two-level architecture**: Assistants API (execution) + Agents SDK (orchestration)
* **Zero-shot multi-agent capability**: Task handoff between agents via `Handoffs`
* **Rapid development**: A working prototype with RAG and tools in under 2 hours
* **Cost trade-off**: Convenience vs. high cost and vendor lock-in

---

## 🏗️ Architectural Components

```python
from agents import Agent, Runner, function_tool

# Create an agent with a tool
@function_tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

agent = Agent(
    name="WeatherBot",
    instructions="You provide weather forecasts",
    tools=[get_weather]
)

# Run a multi-agent workflow
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
```

---

## ⚖️ Comparison with LangChain

| Criterion       | Agents SDK                     | LangChain                      |
|-----------------|--------------------------------|--------------------------------|
| Learning curve  | ★★★★★ (very low)               | ★★★☆☆ (moderate)               |
| Flexibility     | ★★☆☆☆ (limited)                | ★★★★★ (full)                   |
| Cost            | ★★☆☆☆ (high)                   | ★★★★☆ (controllable)           |
| Multi-agent     | Built-in via Handoffs          | Via LangGraph                  |

---

## 💡 When to Use?

**✅ Ideal for:**
- Rapid MVP prototyping
- Projects within the OpenAI ecosystem
- Getting started with agent concepts

**❌ Problematic scenarios:**
- Production systems requiring cost control
- Hybrid-architecture systems (multiple LLMs)
- Projects requiring custom RAG pipelines

---

## 📌 Key Features

* **Automatic state management** via Threads
* **Built-in tools**: Code Interpreter, File Search
* **Execution tracing** for debugging complex workflows
* **`@function_tool` decorator** for integrating arbitrary functions

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>