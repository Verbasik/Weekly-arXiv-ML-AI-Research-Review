![Figure_0](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-16/assets/Figure_0.png)

# **The Age of Multi-Agents? LangChain, but on Steroids: Google’s Agent2Agent (A2A) Protocol + MCP**

## **0. Formalization**

Let’s begin by defining the term "protocol":

**Formal Definition:**

**Protocol** (in the context of computer networks and technologies) is a set of predefined rules and standards that govern interactions between devices, programs, or systems. It specifies:
- **Data format** (how information is structured and encoded);
- **Sequence of actions** (e.g., connection establishment, data transfer, session termination);
- **Error handling methods** and delivery guarantees.

Protocols ensure **compatibility** between different devices and systems, enabling them to understand each other even if produced by different manufacturers.

**Examples:**
- **HTTP** — for transferring web pages;
- **TCP/IP** — for breaking down and reassembling data over the internet;
- **FTP** — for file exchange.

**Definition of protocol in this context:**

A protocol is a standardized set of rules defining:
- **Data format** (e.g., JSON, HTTP requests);
- **Interaction sequence** (agent discovery, task execution, message exchange);
- **Security** (authentication, encryption);
- **Support for diverse scenarios** (long-running tasks, multimodality).

Both protocols adhere to this definition but address different challenges within the AI ecosystem.

## **1. Background**

### **Growth of AI Agents and Interaction Challenges**

In recent years, there has been explosive development and adoption of agent technologies in the field of artificial intelligence (AI). AI agents are designed as software entities capable of perceiving their environment, making reasoned decisions, and autonomously performing tasks. They have demonstrated significant potential across numerous domains—including customer service, enterprise automation, and scientific research—and are viewed as the next frontier in business operations. However, as the number of agents created by different vendors and based on diverse frameworks increases, a serious problem emerges: compatibility.

The current AI agent ecosystem is fragmented. Each agent often operates in an isolated environment, like an information island, making efficient communication, collaboration, or information exchange between them difficult. This lack of interoperability severely limits the potential of multi-agent systems to solve complex enterprise tasks and automate cross-system processes. To fully unlock the potential of AI agents, it is critical to break down these barriers and enable seamless interaction between agents.

### **Background and Goal of Google’s A2A Agreement**

To address this challenge, Google officially unveiled a new open protocol named Agent2Agent (A2A) at the Google Cloud Next '25 conference held on April 9, 2025. The primary goal of the A2A protocol is to provide a common, open standard enabling AI agents to securely communicate, exchange information, and coordinate actions across different platforms, frameworks, and vendors. Google claims that the development of A2A is grounded in the company’s experience scaling intelligent systems internally and is designed to solve practical problems encountered when deploying large-scale multi-agent systems for customers.

With A2A, Google hopes to empower developers to build applications capable of connecting to any other agent adhering to the protocol, allowing users to combine agents from different vendors to create solutions tailored to their specific needs.

## **2. Overview of the A2A Protocol**

### **Basic Definition and Core Value Proposition**

A2A is defined as an open protocol designed to enable AI agents to securely communicate, exchange information, and coordinate actions across different platforms and vendors. Its core value proposition lies in opening a "new era of agent interaction," increasing agent autonomy, significantly boosting productivity, and potentially reducing long-term costs by advancing innovation and enabling more powerful, universal agent systems. The protocol allows users to flexibly combine agents from different vendors to construct solutions meeting their specific requirements.

The timing and market positioning of the A2A protocol are noteworthy. It was launched after the Model Context Protocol (MCP) by Anthropic (discussed here: https://habr.com/ru/articles/893482/) attracted significant market attention and gained industry adoption. MCP primarily addresses the challenge of how intelligent agents connect to and utilize external tools and data sources. Its success confirmed the value of standardization in the intelligent agent domain. Google has closely observed this trend and identified the next key area requiring standardization: interaction between intelligent agents. Google explicitly positions A2A as a complement to MCP, not a competitor. This positioning strategy allows A2A to leverage the existing momentum of MCP, lowering the adoption barrier for developers and enterprises and avoiding market resistance that might arise from direct competition.

Thus, Google can establish standards in a key adjacent area of the agent ecosystem (inter-agent interaction) while tightly integrating it with its broader cloud AI strategy (e.g., Vertex AI, Agentspace, Agent Development Kit, etc.), simultaneously stimulating enterprise adoption by emphasizing a strong partner ecosystem. This suggests Google is attempting to shape and direct the evolution of the agent ecosystem through what appears to be a collaborative approach (built upon existing standards), helping to channel agent interaction through or via cloud infrastructure.

## **3. Goals and Core Design Principles**

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-16/assets/Figure_01.png)

In developing the A2A protocol, Google followed several core principles to ensure its effectiveness, usability, and security—particularly for enterprise-level application scenarios.

**Leverage Agent Capabilities:** A2A is designed to support natural agent-to-agent interaction, even when agents do not share a common internal memory, tools, or context. The protocol’s goal is to enable genuine multi-agent interaction scenarios—not merely treating one agent as a simple "tool" or API endpoint for another. This contrasts with simple API integrations or extended function calls, which often limit interaction flexibility and depth.

**Built on Existing Standards:** To simplify integration and learning, the A2A protocol is built upon widely adopted existing technological standards, including HTTP, Server-Sent Events (SSE), and JSON-RPC. The selection of these mature technologies means enterprises can more easily integrate A2A into their existing IT infrastructure used daily.

**Security by Default:** Security is a cornerstone of the A2A protocol architecture. The protocol is designed to support enterprise-grade authentication and authorization mechanisms. At release, its security model was aligned with the OpenAPI authentication scheme, ensuring compatibility with industry-standard security practices.

**Support for Long-Running Tasks and Real-Time Feedback:** Collaborative agent work often involves executing complex tasks that may take hours or even days, sometimes requiring human intervention. The A2A protocol is designed to be sufficiently flexible to handle both rapid-response tasks and long-running, in-depth investigations or complex processes. Throughout task execution, the protocol enables real-time feedback, notifications, and status updates—likely achieved through Server-Sent Events (SSE).

**Modality Independence:** Recognizing that the agent world extends beyond text-based interactions, the A2A protocol is designed to be modality-agnostic. It supports various interaction modes, including audio and video streaming, to adapt to diverse future needs of intelligent agent interactions.

Collectively, these design principles embody a pragmatic strategy aimed at broad adoption within enterprise environments. The choice of mature web standards (HTTP, JSON-RPC, SSE) and adherence to the OpenAPI security model significantly reduces the technical barrier and integration costs for enterprises. Instead of investing massive resources in learning an entirely new technology stack or paradigm, companies can directly leverage existing infrastructure and technical expertise.

Moreover, support for "opaque agents"—agents that can interact without revealing their internal state or reasoning—directly addresses key enterprise concerns related to security, intellectual property protection, and system modularity when working with agents from different vendors or business units. This pragmatic, demand-driven design increases the likelihood that A2A will become the de facto industry standard by reducing resistance to adoption.

## **4. Technical Architecture and Key Features**

### **Communication Mechanism**

The A2A protocol’s communication layer is built on the following widely adopted standards:

- **HTTP:** The foundational request/response protocol used for basic communication interactions between agents. HTTP provides the fundamental infrastructure for all A2A interactions and makes the protocol easily integrable with existing web services and enterprise systems. Using HTTP ensures practicality and simplifies implementation, which is critical for rapid adoption in enterprise environments.

- **SSE (Server-Sent Events):** Used to implement one-way real-time data streaming from server to client. This is especially important for long-running tasks, as it allows a remote agent to send status updates, notifications, or intermediate results to the client agent without requiring constant polling. SSE enables remote agents to broadcast updates to clients as work progresses, ensuring efficiency during long operations. For long-running tasks, servers supporting streaming can use the `tasks/sendSubscribe` method, where the client receives event notifications containing `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` messages to track real-time progress.

- **JSON-RPC:** A lightweight remote procedure call (RPC) protocol using JSON as its data format. A2A uses JSON-RPC to standardize structured method calls and responses between agents. A2A employs JSON-RPC 2.0 for message exchange, providing a simple, language-independent way to perform remote procedure calls using JSON data format. The basic JSON-RPC request structure includes a `method` field (string identifying the operation, e.g., "tasks/send"), `params` (an object or array containing parameters for the method), and `id` (a unique identifier correlating the request and response).

- **Client-Server Interaction Model:** A2A implements a clean client-server model, where client and server agents can operate and be hosted remotely. Agent configuration is simple and requires only specifying a base URL; the "Agent Card" handles context exchange. This approach simplifies integration and provides flexibility when adding new agents to the system. The protocol clearly defines interaction types: discovery (retrieving the agent card), initiation (sending a task), interaction (when input is required), and completion (when the task reaches a terminal state).

### **Data Format and Structure**

Data exchange in the protocol primarily occurs in JSON format. Key data structures include:

- **Agent Card:** A JSON object used by an agent to publish and describe its capabilities, identity information, etc. This is the foundation for intelligent agent discovery. The method by which an A2A server announces its capabilities is implemented via an "Agent Card" in JSON format. The card is typically hosted at the standard path `/.well-known/agent.json`, making discovery predictable—similar to how web browsers locate robots.txt files. The card contains information about the agent’s capabilities, required authentication schemes, supported content types, and API endpoints.

- **Task:** The primary object representing a work request and serving as the central unit of communication between agents. The protocol defines a lifecycle management mechanism for tasks. A task is the central unit of work. The client initiates a task by sending a message (`tasks/send` or `tasks/sendSubscribe`). Tasks have unique identifiers and progress through various states (submitted, working, input-required, completed, failed, canceled). This enables tracking of progress and facilitates asynchronous agent interaction.

- **Message:** Represents conversational exchanges between a client (role: "user") and an agent (role: "agent"). Messages contain Parts and are used to convey context, responses, artifacts, or user instructions. Message structure enables agents to engage in multi-step dialogues within a single task and exchange various content types.

- **Artifact:** Output data or results obtained after successful task completion. Artifacts also contain Parts and may represent generated files, structured data, or other agent outputs. This is a standardized method for returning task execution results.

- **Part:** An autonomous content block included in a message, such as text, generated images, etc. A Part can be textual (TextPart), file-based (FilePart) with embedded bytes or a URI, or structured JSON (DataPart), e.g., for forms. Each part has a clearly defined content type, enabling client and remote agent to agree on required data formats. This is a UI coordination mechanism where each message includes "parts" as fully formed content fragments.

- **Streaming:** For long-running tasks, A2A provides a mechanism to receive real-time updates. Servers supporting streaming can use `tasks/sendSubscribe`. Clients receive Server-Sent Events (SSE) containing `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` messages, providing real-time progress tracking.

- **Push Notifications:** Servers supporting the `pushNotifications` feature can proactively send task updates to a webhook URL provided by the client via `tasks/pushNotification/set`. This feature allows clients to receive updates without constant server polling.

The A2A architecture implements a complete agent interaction cycle: discovery (client retrieves the agent card from a known server URL), initiation (client sends a task with an initial user message and unique task ID), interaction (if additional input is needed, the client sends follow-up messages), and completion (the task reaches a terminal state: completed, failed, or canceled).

The full JSON specification is available in the official repository: [GitHub](https://github.com/syntax-syndicate/Agent-2-agent/blob/main/specification/json/a2a.json).

### **Detailed Explanation of Key Features**

> Full code implementation is available in the A2A directory.

The A2A protocol defines four core functions for agent interaction, each supported by specific mechanisms and data structures. Below, we detail each function.

1. **Capability Discovery:**
   *   **Mechanism:** This is the process by which agents find each other and learn about each other's capabilities. Agents "broadcast" their capabilities via [**Agent Cards**](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/hosts/cli/__main__.py#L61-L66) in JSON format. Typically, this card is accessible at a standard path (`/.well-known/agent.json`), making discovery predictable. Client agents (using components such as [`A2ACardResolver`](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/client/card_resolver.py#L23-L104)) can retrieve this card to discover and select a remote agent best suited for a specific task.

       *Example of a client retrieving an agent card:*

       ```python
       # From A2A/hosts/cli/__main__.py
       card_resolver = A2ACardResolver(agent_url)
       card = card_resolver.get_agent_card()
       print("======= Agent Card ========")
       print(card.model_dump_json(exclude_none=True))

       # Class A2ACardResolver uses httpx for a GET request
       # to self.base_url + "/" + self.agent_card_path
       # and parses the JSON response into an AgentCard model
       class A2ACardResolver:
           # ...
           def get_agent_card(self) -> AgentCard:
               with httpx.Client() as client:
                   response = client.get(self.base_url + "/" + self.agent_card_path)
                   response.raise_for_status()
                   try:
                       # Uses Pydantic model AgentCard for parsing
                       return AgentCard(**response.json())
                   except json.JSONDecodeError as e:
                       raise A2AClientJSONError(str(e)) from e
       ```
   *   **Details:** `AgentCard` contains comprehensive information defined using Pydantic models: identity data (`name`, `description`, `url`, `provider`, `version`), key **capabilities** (`capabilities`), required **authentication schemes** (`authentication`), supported default and skill-specific **input/output modes** (`inputModes`, `outputModes`), and a list of **skills** (`skills`).

       *[Example structure of AgentCard and nested models](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py):*

       ```python
       # From A2A/common/types.py
       class AgentCapabilities(BaseModel):
           streaming: bool = False
           pushNotifications: bool = False
           stateTransitionHistory: bool = False

       class AgentAuthentication(BaseModel):
           schemes: List[str]
           credentials: str | None = None

       class AgentSkill(BaseModel):
           id: str
           name: str
           description: str | None = None
           # ... other fields ...
           inputModes:  List[str] | None = None
           outputModes: List[str] | None = None

       class AgentCard(BaseModel):
           name: str
           description: str | None = None
           url: str # URL for JSON-RPC endpoint
           provider: AgentProvider | None = None
           version: str
           capabilities: AgentCapabilities
           authentication: AgentAuthentication | None = None
           defaultInputModes:  List[str] = ["text"]
           defaultOutputModes: List[str] = ["text"]
           skills: List[AgentSkill]
       ```

   *   **Value:** Capability discovery allows the client to dynamically adapt its interaction strategy. For example, the client will use the `tasks/sendSubscribe` method for streaming only if the `AgentCard` specifies `streaming: true`. This is critical for building dynamic, adaptive multi-agent systems.

       *[Example of capability usage in CLI client:](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/hosts/cli/__main__.py#L193-L201)*

       ```python
       # From A2A/hosts/cli/__main__.py
       async def completeTask(client: A2AClient, streaming, ...):
           # ...
           if streaming: # Value obtained from card.capabilities.streaming
               response_stream = client.send_task_streaming(payload)
               async for result in response_stream:
                   print(f"stream event => {result.model_dump_json(exclude_none=True)}")
               taskResult = await client.get_task({"id": taskId})
           else:
               taskResult = await client.send_task(payload)
           # ...
       ```

2.  **Task Management:**
    *   **Mechanism:** This is the core interaction model of the A2A protocol. All communication revolves around the creation, execution, and completion of **Tasks**. The client initiates a task by sending a JSON-RPC request (e.g., `tasks/send` or `tasks/sendSubscribe`).

        *[Examples of JSON-RPC requests for task management:](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py)*

        ```python
        # From A2A/common/types.py
        class SendTaskRequest(JSONRPCRequest):
            method: Literal["tasks/send"] = "tasks/send"
            params: TaskSendParams  # Contains id, sessionId, message, etc.

        class SendTaskStreamingRequest(JSONRPCRequest):
            method: Literal["tasks/sendSubscribe"] = "tasks/sendSubscribe"
            params: TaskSendParams

        class GetTaskRequest(JSONRPCRequest):
            method: Literal["tasks/get"] = "tasks/get"
            params: TaskQueryParams # Contains id, historyLength

        class CancelTaskRequest(JSONRPCRequest):
            method: Literal["tasks/cancel",] = "tasks/cancel"
            params: TaskIdParams    # Contains id
        ```
    *   **Details:** Each `Task` has a unique `id`, an optional `sessionId` for grouping related tasks, a current `status`, a list of received `artifacts`, and a `history` of messages. The protocol defines clear lifecycle states for tasks via the `TaskState` enumeration. The current task status (`TaskStatus`) includes the state itself (`state`), an optional agent message (`message`), and a timestamp (`timestamp`).

        *Example structure of [Task](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L286-L314), [TaskStatus](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L208-L232), and [TaskState](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L36-L55):*

        ```python
        # From A2A/common/types.py
        class TaskState(str, Enum):
            SUBMITTED = "submitted"
            WORKING = "working"
            INPUT_REQUIRED = "input-required"
            COMPLETED = "completed"
            CANCELED = "canceled"
            FAILED = "failed"
            UNKNOWN = "unknown"

        class TaskStatus(BaseModel):
            state: TaskState
            message: Message | None = None # Agent status message
            timestamp: datetime = Field(default_factory=datetime.now)
            # ... timestamp serializer ...

        class Task(BaseModel):
            id: str
            sessionId: str | None = None
            status: TaskStatus
            artifacts: List[Artifact] | None = None
            history:   List[Message]  | None = None # History of user/agent messages
            metadata:  dict[str, Any] | None = None
        ```
    *   **Long-running tasks:** The protocol supports both immediate tasks and long-running tasks. For the latter, it provides:
        *   **SSE (Server-Sent Events):** The `tasks/sendSubscribe` method allows the client to receive asynchronous real-time updates ([`TaskStatusUpdateEvent`](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L317-L341), [`TaskArtifactUpdateEvent`](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L344-L368)) without polling.

            *[Example of client SSE handling and event structure:](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/client/client.py#L123-L172)*
            ```python
            # From A2A/common/client/client.py
            async def send_task_streaming(
                self, payload: dict[str, Any]
            ) -> AsyncIterable[SendTaskStreamingResponse]:
                request = SendTaskStreamingRequest(params=payload)
                with httpx.Client(timeout=None) as client:
                    with connect_sse( # Uses httpx_sse
                        client, "POST", self.url, json=request.model_dump()
                    ) as event_source:
                        for sse in event_source.iter_sse():
                            # Parses event data into SendTaskStreamingResponse
                            yield SendTaskStreamingResponse(**json.loads(sse.data))
                            
            # From A2A/common/types.py
            class TaskStatusUpdateEvent(BaseModel):
                id: str             # Task ID
                status: TaskStatus  # New status
                final: bool = False # Is this the final status?
                metadata: dict[str, Any] | None = None

            class TaskArtifactUpdateEvent(BaseModel):
                id: str            # Task ID
                artifact: Artifact # New artifact
                metadata: dict[str, Any] | None = None
            ```
        *   **Push Notifications:** If the agent supports it (`pushNotifications: true` in `AgentCard`), the client can configure a URL to receive proactive task status notifications via `tasks/pushNotification/set` and `tasks/pushNotification/get` methods.

            *Example of client setup for [Push Notifications](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/hosts/cli/__main__.py#L182-L189) and configuration structure:*
            ```python
            # From A2A/hosts/cli/__main__.py - forming payload for send_task
            if use_push_notifications:
                payload["pushNotification"] = {
                    "url": f"http://{notification_receiver_host}:{notification_receiver_port}/notify",
                    "authentication": {
                        "schemes": ["bearer"], # Indicates JWT Bearer authentication
                    },
                }

            # From A2A/common/types.py - structures for requests and configuration
            class SetTaskPushNotificationRequest(JSONRPCRequest):
                method: Literal["tasks/pushNotification/set",] = "tasks/pushNotification/set"
                params: TaskPushNotificationConfig

            class TaskPushNotificationConfig(BaseModel):
                id: str # Task ID
                pushNotificationConfig: PushNotificationConfig

            class PushNotificationConfig(BaseModel):
                url: str # Client URL to receive notifications
                token: str | None = None # May be used for simple authentication
                authentication: AuthenticationInfo | None = None # For complex authentication (e.g., JWT)
            ```
    *   **Result:** The final task outcome is represented as [**Artifacts**](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L253-L283), which contain one or more **Parts** with the results. Artifacts may also have metadata, a name, a description, and flags for managing large or streaming results (`index`, `append`, `lastChunk`).

        *Example Artifact structure:*
        ```python
        # From A2A/common/types.py
        class Artifact(BaseModel):
            name: str | None = None
            description: str | None = None
            parts: List[Part]             # Contains actual result data
            metadata: dict[str, Any] | None = None
            index: int = 0                # For ordering artifact parts
            append: bool | None = None    # Indicates whether to append to previous artifact
            lastChunk: bool | None = None # Indicates whether this is the last part
        ```

3.  **Collaboration:**
    *   **Mechanism:** Defines how agents exchange information during task execution. The primary unit of exchange is the **Message**.
    *   **Details:** Each `Message` has a `role` ("user" for the client, "agent" for the server), a list of **Parts** containing the actual content, and optional `metadata`. Agents can exchange messages to convey context, send responses, transfer artifacts, or deliver user instructions. The message history (`history` in `Task`) preserves the dialogue context.

        *Example [Message](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/common/types.py#L184-L205) structure:*
        ```python
        # From A2A/common/types.py
        class Message(BaseModel):
            role: Literal["user", "agent"] # Defines sender
            parts: List[Part]              # List of message parts (text, files, data)
            metadata: dict[str, Any] | None = None
        ```
        *Example of client message construction:*
        ```python
        # From A2A/hosts/cli/__main__.py
        payload = {
            # ...
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text", # Indicates part type
                        "text": prompt, # User-provided text
                    }
                ],
            },
            # ...
        }
        ```
    *   **Multi-step interaction:** The `TaskState.INPUT_REQUIRED` state explicitly indicates that the agent awaits additional input from the client to continue the task, enabling multi-step dialogue within a single task.

        *Example of handling [INPUT_REQUIRED](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-16/A2A/hosts/cli/__main__.py#L203-L215) in the CLI client:*
        ```python
        # From A2A/hosts/cli/__main__.py
        async def completeTask(client: A2AClient, ...):
            # ... (task submission, receiving taskResult) ...
            state = TaskState(taskResult.result.status.state)
            if state.name == TaskState.INPUT_REQUIRED.name:
                # If agent requires input, recursively call the same function
                # to obtain next user input within the same task (taskId)
                return await completeTask(
                    client,
                    streaming,
                    use_push_notifications,
                    notification_receiver_host,
                    notification_receiver_port,
                    taskId, # Use same task ID
                    sessionId
                )
            else:
                # Task completed (COMPLETED, FAILED, CANCELED)
                return True
        ```
    *   **Opaque agents:** The protocol enables agents to dynamically collaborate (e.g., requesting clarifications from each other) without requiring disclosure of their internal state or reasoning logic—critical for enterprise scenarios involving agents from different vendors.

4.  **User Experience Negotiation:**
    *   **Mechanism:** Allows an agent and client to negotiate and adapt the format and presentation of information based on each other's capabilities and user needs. The key element here is the **Part**.
    *   **Details:** The protocol defines various `Part` types using Pydantic Union with a `type` discriminator.

        *Example Part definition and variants:*
        ```python
        # From A2A/common/types.py
        class TextPart(BaseModel):
            type: Literal["text"] = "text"
            text: str
            metadata: dict[str, Any] | None = None

        class FileContent(BaseModel):
            name: str | None = None
            mimeType: str | None = None
            bytes: str | None = None # base64-encoded data
            uri: str | None = None   # Link to external resource
            # Validator requires either bytes OR uri

        class FilePart(BaseModel):
            type: Literal["file"] = "file"
            file: FileContent
            metadata: dict[str, Any] | None = None

        class DataPart(BaseModel):
            type: Literal["data"] = "data"
            data: dict[str, Any] # Arbitrary JSON data
            metadata: dict[str, Any] | None = None

        # Union with discriminator for automatic parsing of correct type
        Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator="type")]
        ```
        When submitting a task, the client can specify preferred output formats in the `acceptedOutputModes` field of the `TaskSendParams` request. The server agent, aware of its own capabilities (from `AgentCard` and `AgentSkill`) and client preferences, can select the most suitable format for its response.

        *Example of client specifying accepted formats:*
        ```python
        # From A2A/hosts/cli/__main__.py
        payload = {
            # ...
            "acceptedOutputModes": ["text", "image/png"],   # Client accepts text or PNG images
            # ...
        }

        # From A2A/common/types.py
        class TaskSendParams(BaseModel):
            # ...
            acceptedOutputModes: Optional[List[str]] = None # List of MIME types or other identifiers
            # ...
        ```
    *   **Flexibility:** This system explicitly supports negotiation of diverse user interfaces (text, images, files, forms, potentially streaming audio/video), making interactions richer and context-adaptive. The `metadata` fields at various levels (Task, Message, Part, Artifact) provide an additional channel for conveying UI- or context-specific information.


### **A2A Protocol Security**

The A2A protocol specification includes enterprise security considerations:

- Support for enterprise-grade authentication (AuthN) and authorization (AuthZ) mechanisms;
- The security model aligns with the OpenAPI authentication scheme at release;
- Official documentation includes a dedicated discussion on "Enterprise Ready";
- The protocol's goal is to ensure secure information exchange and coordination.

However, a potential contradiction exists between protocol-level standardization and overall security of a multi-agent system. While A2A aims to provide a secure communication foundation, experts quickly noted risks associated with inter-agent interaction, such as rapid injection attacks.

**A2A Protocol Characteristics**

1. **Interaction between "Opaque" Agents**  
   A2A is designed to enable interaction between agents whose internal implementation details may be hidden. This underscores the importance of additional security measures.

2. **Complement to MCP**  
   A2A serves as a complement to MCP, which itself proved vulnerable to rapid injection attacks when granting tools to agents.

3. **Reasons for Vulnerabilities**  
   - Standardization of communication facilitates both legitimate interaction and malicious actors exploiting the same standard to launch attacks.
   - "Prompt injection" or "social engineering" attacks target the ability of Large Language Models (LLMs) to understand and follow instructions—capabilities operating above the communication protocol level.
   - Connecting multiple agents via A2A while granting them tool capabilities in combination with MCP creates a complex interaction chain, amplifying the potential impact of compromising one agent.
   - The "opaque" nature of enterprise applications means the client agent may not fully understand the inner workings or reliability of a remote agent with which it interacts.

**Conclusion**

Although A2A provides baseline security mechanisms (such as authentication), the overall security of multi-agent systems built using A2A (and MCP) will depend on:

- Robust agent design;
- Sophisticated permission control;
- Rigorous input validation and filtering;
- Continuous monitoring;
- Development of new security paradigms for agent systems.

> The protocol itself is necessary but not sufficient. This remains a significant research and engineering challenge in the current field of AI security.

## **5. Industry History and Ecosystem**

The launch of the A2A protocol is not an isolated event. It occurs against the backdrop of rapid advancement in AI agent technologies and growing industry demand for standardization. Understanding A2A requires situating it within the broader industry and ecosystem.

### Positioning Relationships
Google representatives have repeatedly emphasized that A2A is an addition to MCP. These two solutions address different problems but can work together.

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-16/assets/Figure_02.png)

### Primary Focus
- **MCP**: MCP's primary goal is connecting intelligent agents to external tools, APIs, and data resources, emphasizing structured input/output capabilities to enable agents to leverage external functionalities.
- **A2A**: A2A's essence lies in connecting agents to each other, focusing on collaboration, communication, and task coordination between agents, while supporting more dynamic and potentially unstructured interactions.

### Requirements for Joint Operation
Documentation and discussions indicate that complex agent applications may require both MCP (for accessing tools and data) and A2A (for collaborating with other agents). For example, Google's Agent Development Kit (ADK) initially supports the MCP tooling, and agents built with ADK can interact via A2A.

### Other Intelligent Agent Communication Protocols

Beyond MCP, other intelligent agent communication protocols may exist or emerge in the industry. For instance, Cisco's Agent Connect Protocol (ACP) is also mentioned as an addition to MCP. This indicates that the industry broadly recognizes the need for standardizing intelligent agent interaction.

### Comparative Table of A2A and MCP Protocol Features

| Features/Aspects              | A2A (Agent2Agent)                                                                 | MCP (Model Context Protocol)                                |
|-------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------|
| **Primary Goals**             | Enable interaction and collaboration between intelligent agents                  | Connect agents to external tools, APIs, and data resources     |
| **Type of Interaction**       | Agent-to-agent                                                                   | Agent-to-tool/resource                                        |
| **Key Entities**              | Client agent, remote agent                                                       | Agent, MCP server, tool                                       |
| **Communication Style**       | Dynamic, consultative, task-oriented, supporting unstructured and multimodal interactions | Structured, request-response, tool-calling oriented           |
| **Data Structure Focus**      | Agent Card, Task, Artifact, Part                                                 | Tool definition, function call/response, resource schema      |
| **Security Focus**            | Inter-agent authentication/authorization                                         | - |
| **Typical Use Cases**         | Cross-system workflow automation, multi-agent task decomposition and collaboration | Agents use external APIs to retrieve information, perform operations, access databases |
| **Additional Roles**          | Provides a "linguistic and network layer" for agent communication                | A "pluggable system" providing agents access to external capabilities |

### Advantages of A2A
A2A advantages include:
- Enabling cross-platform communication;
- Facilitating collaboration;
- Enterprise-focused design (authentication, long-running tasks, opaque agents);
- Use of standard formats;
- Strong partner ecosystem support;
- Reduced integration barriers.

### Potential Drawbacks or Concerns Regarding A2A
Potential drawbacks or concerns include:
- Questionable necessity (compared to REST+ conventions);
- Potential for redundant development;
- Concerns regarding Google's motivations and control;
- Security risks (especially rapid injection);
- Potential complexity and fragility;
- Possible overlap with MCP functionality;
- Lack of clear early-stage examples.

### Protocol Intersections

Google recommends modeling A2A agents as MCP resources (described by AgentCard). Thus, a framework can not only invoke tools via MCP but also interact with users, remote agents, and other agents via A2A to achieve seamless interaction.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-16/assets/Figure_03.png)

## **Conclusion**

This analysis thoroughly examines **Google’s Agent2Agent (A2A) protocol**, developed to **standardize interaction between artificial intelligence (AI) agents**. Amidst the rapid proliferation of agents built on diverse platforms and frameworks, **A2A proposes a common open standard for secure information exchange and coordination among them**.

Key aspects of the A2A protocol include its **focus on interaction between "opaque" agents**, **use of existing web standards** (HTTP, SSE, JSON-RPC) to simplify integration, **support for long-running tasks and real-time feedback**, and **modality-independent interaction**. A2A’s technical architecture includes a **mechanism for agent capability discovery via the "Agent Card"**, **task lifecycle management**, **support for multi-step collaboration through "Message" and "Part" exchange**, and **user experience negotiation via defined data formats**.

Importantly, **Google positions A2A as a complement to Anthropic’s Model Context Protocol (MCP)**, where MCP focuses on connecting agents to external tools and data, while **A2A focuses on interaction between agents themselves**. Thus, building complex multi-agent systems may require simultaneous use of both protocols.

Despite A2A’s potential advantages—such as **enabling cross-platform communication and fostering collaboration**—there are also potential drawbacks and concerns, including **security risks, particularly in the context of rapid injection attacks**. **The security of multi-agent systems built on A2A will depend not only on the protocol itself but also on the robustness of agent implementation, permission control, and continuous monitoring**.

Overall, the **Agent2Agent protocol represents a significant step toward standardizing interaction between AI agents**, potentially **greatly increasing agent autonomy, boosting productivity, and reducing long-term costs** by enabling more powerful and universal systems. However, for its broad and secure adoption, **further development and focused attention on security in inter-agent interaction are required**.