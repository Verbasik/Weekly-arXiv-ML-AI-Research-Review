# OpenAI Agents SDK: Tools for Building Intelligent Agents

---

### **TWRB_FM 📻**

<audio controls>
  <source src="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/raw/refs/heads/develop/agents-under-hood/openai-cs-agents-demo/TWRB.wav  " type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

---

## 1. Introduction: A New Era in AI Agent Development

In recent years, the development of AI agents—autonomous systems capable of reasoning, planning, and executing complex tasks—has become one of the key frontiers in artificial intelligence. OpenAI, at the forefront of this revolution, has introduced two powerful tools designed to simplify and standardize the creation of such systems: the **Assistants API** and the newer **Agents SDK**.

This review provides a deep dive into the architecture, capabilities, advantages, and limitations of these tools. We’ll explore how they work, which tasks they are best suited for, and how they compare to popular alternatives such as LangChain and Microsoft AutoGen. Our goal is to give developers and technical specialists a clear understanding of how to effectively leverage these technologies to build the next generation of intelligent applications.

## 2. Architecture: Two Levels of One System

To understand OpenAI’s agent tool ecosystem, it’s essential to distinguish between two core components: the low-level **Assistants API** and the high-level **Agents SDK**. They do not replace each other—they complement one another.

```
┌───────────────────────────┐
│        Agents SDK         │  <- Orchestration layer (multi-agent interaction)
│  (Agent, Runner, Handoff) │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      Assistants API       │  <- Execution layer (state and tool management)
│ (Assistant, Thread, Run)  │
└───────────────────────────┘
```

### 2.1. Assistants API: The Foundation for Task Execution

The **Assistants API** is essentially the engine that enables you to create a **single** intelligent agent, manage its state, attach tools, and conduct conversations.

**Key objects:**
*   **Assistant**: The AI agent itself, defined by instructions (e.g., "You are a financial analyst"), a model (e.g., `gpt-4o`), and a set of tools.
*   **Thread**: The conversation context or session between the user and the assistant. It stores message history and automatically manages the model’s context window size.
*   **Message**: A message within a `Thread`, sent by either the user or the assistant.
*   **Run**: The process of the assistant executing a task within a thread. It is during a `Run` that the assistant invokes models and tools to generate a response.

### 2.2. Agents SDK: Orchestrator for Collaborative Work

The **Agents SDK** is a newer, high-level Python framework designed for building **multi-agent systems**. It uses the Assistants API under the hood but adds key abstractions for coordinating multiple agents.

**Key components:**
*   **Agent**: Similar to `Assistant` from the API, but with enhanced capabilities for interaction.
*   **Runner**: A component that manages the execution cycle of an agent or group of agents.
*   **Handoffs (handover)**: The SDK’s key innovation. This mechanism allows one agent (the "dispatcher agent") to delegate a task to another, more specialized agent by invoking the latter as a tool (tool calling).
*   **Guardrails**: Customizable checks to validate agent inputs and outputs, ensuring safe and predictable behavior.
*   **Tracing**: Built-in system for debugging and visualizing workflow flows—critical in complex multi-agent systems.

## 3. Key Capabilities: What Makes the SDK Powerful?

OpenAI’s ecosystem provides developers with a range of powerful out-of-the-box features.

*   **Multi-agent workflows**: With `Handoffs`, you can easily create complex chains—for example, one agent receives a client request, another searches a database, and a third generates a response based on the findings.
*   **Built-in Tools**:
    *   **Code Interpreter**: Provides the agent with an isolated Python execution environment for computations, data analysis, and file generation.
    *   **File Search (RAG)**: Enables the agent to search through documents you provide, implementing Retrieval-Augmented Generation.
*   **Function Calling**: A simple way to "teach" the agent to interact with any external API or Python function. The SDK automatically generates the required JSON schemas using the `@function_tool` decorator and Pydantic.
*   **Effortless State Management**: The Assistants API handles all complexity of preserving conversation context in `Threads`, freeing developers from manually managing history.
*   **Observability and Debugging**: Built-in tracing in Agents SDK, compatible with Logfire, AgentOps, and others, provides full visibility into the request lifecycle.

## 4. Practical Examples: From Theory to Code

Let’s examine several examples demonstrating the SDK’s ease of use.

### Example 1: "Hello, World!" — Single Agent

Creating and running a simple assistant agent.

```python
from agents import Agent, Runner

# 1. Define the agent with instructions
agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# 2. Run it with a specific task
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")

# 3. Print the final output
print(result.final_output)

# Output:
# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

### Example 2: Multi-agent System with Handoff

Create a dispatcher agent that detects language and delegates the task to the appropriate specialist.

```python
from agents import Agent, Runner
import asyncio

# Specialist agents
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)
english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

# Triage agent (dispatcher)
triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent], # Specify which agents can receive tasks
)

async def main():
    # Run the dispatcher with a Spanish request
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)

asyncio.run(main())

# Output:
# ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?
```

### Example 3: Using an External Tool (Function Tool)

Teach the agent to fetch weather information.

```python
import asyncio
from agents import Agent, Runner, function_tool

# Define our Python function as a tool
@function_tool
def get_weather(city: str) -> str:
    """Gets the weather for a given city."""
    return f"The weather in {city} is sunny."

# Create the agent and pass it our new tool
agent = Agent(
    name="Weather Bot",
    instructions="You are a helpful agent that can provide weather forecasts.",
    tools=[get_weather],
)

async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)

asyncio.run(main())

# Output:
# The weather in Tokyo is sunny.
```

## 5. Comparison with LangChain

OpenAI’s tools do not exist in a vacuum—let’s compare them with popular frameworks.

| Criterion | Agents SDK | LangChain |
| :--- | :--- | :--- |
| **Primary Focus** | Simplicity in building agents within the OpenAI ecosystem | Universal framework for building LLM applications |
| **Flexibility** | Low. Strongly tied to OpenAI, less control. | Very high. Supports hundreds of integrations, models, and vector databases. |
| **Multi-agent Support** | Built into Agents SDK (Handoffs), but conceptually simple. | Possible (via LangGraph), but requires more code and understanding. |
| **Learning Curve** | **Very low.** Ideal for rapid prototyping. | Moderate. Requires learning its own abstractions (LCEL). |
| **State Management** | Automatic ("out of the box"). | Manual or via additional libraries. |

## 6. Advantages and Limitations: A Honest Look

### ✅ Advantages

*   **Simplicity and speed of development**: You can build a powerful agent with RAG and tools in just a few hours, not days.
*   **Powerful built-in tools**: The `Code Interpreter` is a unique capability that is extremely difficult and expensive to replicate independently.
*   **Automatic state management**: `Threads` eliminate the headache of managing conversation history.
*   **Built-in observability**: Integrated tracing in Agents SDK is a major aid in debugging.

### ❌ Limitations

*   **Cost**: The main drawback. Using the Assistants API, especially with RAG, can be very expensive—and worse, unpredictable.
*   **"Black box"**: Simplicity comes at the cost of abstraction. Developers have little control over how RAG works (e.g., how text is chunked) or how context is processed.
*   **Stability and Beta Status**: The API is still evolving; bugs and changes are possible. Developer communities have noted the "roughness" of some components.
*   **Vendor Lock-in**: Despite claims of third-party LLM support in Agents SDK, reports indicate poor compatibility. You become heavily locked into the OpenAI ecosystem.

## 7. Practical Recommendations: When and What to Use?

Tool choice depends on your use case, budget, and flexibility requirements.

*   **Use OpenAI Assistants API / Agents SDK if:**
    *   You need a quick prototype or MVP.
    *   Your project is already tightly integrated into the OpenAI ecosystem.
    *   You’re new to agent concepts and taking your first steps.

*   **Use LangChain if:**
    *   You need full control over every aspect of your application (RAG, prompts, models).
    *   You plan to use diverse LLMs (including open-source) or vector databases.
    *   You need a mature, battle-tested ecosystem with a massive community.

## 8. Conclusion: Summary and Outlook

OpenAI’s Assistants API and Agents SDK are powerful, yet compromise-laden tools. They democratize AI agent creation, making it accessible to a broad range of developers and significantly accelerating prototyping.

However, this simplicity comes at a cost—cost, flexibility, and control. For serious production systems requiring customization, multi-model support, and predictable costs, mature frameworks like LangChain remain the more reliable choice.