# MCP (Model Context Protocol)

Recently, the acronym MCP has appeared increasingly frequently in some articles and comment sections on arXiv or Daily Papers Hugging Face that I browse. Realizing my understanding of it was only approximate, I decided to investigate it in detail and share my findings with you.

## Single Agent

Let's first examine the single-agent architecture.

![Figure](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-12/assets/Figure.png  )

1. **Tools** are functions defined and invoked within the current program. The function definitions of tools are included in the system prompt to enable the LLM to understand available tools.

2. **Memory** is divided into two parts: the current session data stream, including actions performed at each step and their results, is stored in session memory and can be fully fed into the LLM at any time to help the LLM determine its next action. Long-term data and user knowledge bases—such as user preferences on a platform, domain content, multi-round conversation context, etc.—are retrieved from a vector database.

3. The **Router** centralizes the planning of the entire process by passing user prompts/system prompts/memory to the LLM; the LLM performs deep reasoning and outputs specific execution tasks, and the router invokes the corresponding action function (function calling).

This is a simple, general single-agent architecture implementing the Thought–Plan–Action–Reflect (Thought) cycle within an Agent, with one model responsible for everything.

## MCP

The above architecture has minor issues with the Tools module: limited support and scalability for tool functions. Management becomes difficult when there are too many. Adding functions requires updating the main program. Additionally, the function call specification must be defined manually. External tool services to be used must be encapsulated independently.

To resolve these minor issues, this architecture can be optimized: the tools module is separated from the agent and uniformly managed and implemented using the MCP protocol.

## Model Context Protocol (MCP): A New Standard for AI Ecosystem Integration

The Model Context Protocol (MCP) is an open standard developed and introduced by Anthropic on November 25, 2024. The primary goal of MCP is to create a unified communication protocol between large language models (LLMs) and external data sources and tools. In my view, MCP emerged as a natural evolution of the Function Calling approach, overcoming its limitations and expanding the capabilities of AI models to interact with the external world. If Function Calling can be seen as a point solution for specific interaction tasks, MCP represents a comprehensive approach to integration, providing a more flexible, scalable, and standardized ecosystem.

### Essence of MCP

MCP is not a framework or tool, but a protocol—similar to:
- HTTP for the internet
- SMTP for messaging
- LSP (Language Server Protocol) for programming language support

Anthropic accurately characterizes MCP as the "USB-C port equivalent for agent systems"—a universal interface enabling standardized interaction between different components of the AI ecosystem, regardless of vendor.

> As the saying goes, a picture is worth a thousand words.

![Figure_1](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-12/assets/Figure_1.jpeg  )

MCP unifies interface call definitions for accessing capabilities of various tools. Previously, a service (e.g., Slack) had to connect to function call formats defined by multiple user products (e.g., Cursor). Now, both the service and client need to connect only to a single format, and each side needs to implement it only once.

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-12/assets/Figure_2.png  )

The MCP Server operates independently on any server and can have its own independent database of information/resources. It is not bound to the Agent server and can be reused and easily connected or disconnected.

![Figure_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-12/assets/Figure_3.png  )

Original tool function calls are encapsulated via the MCP Server, and the architecture becomes:

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-12/assets/Figure_4.png  )

The difference from the original pure function call lies in a more flexible architecture, including:

1. **Clustering**: Dispersed functions can be grouped into a single service for easier management.
2. **Decoupling**: The actual call occurs on the corresponding MCP server side, not directly invoked by the Agent service. Tool extension deployment is decoupled from the Agent project.
3. **Cohesion**: The MCP server itself can perform coordinated actions, including independent resource management, independent context, etc.
4. **Reusability**: Universal protocols and tool capabilities facilitate reuse across multiple agents. Many existing MCP servers exist in the external ecosystem that can be accessed directly.
5. **Unification**: Calls to both client-side and cloud-based tools can be implemented using the unified MCP protocol.

### Architecture and Operation Principle

MCP defines:
1. Ways clients interact with servers
2. Methods servers use to handle tools (APIs, functions)
3. Rules for accessing resources (files, databases)

In this architecture:
- AI models act as clients
- External services and data sources are peripheral devices (tools)
- MCP is the standardized interface (port) between them

### Example

Let's consider an example implementation of an MCP server and MCP client below. I will attempt to answer two key questions:

1. How does an LLM model interact with an MCP server?
2. How does an LLM model invoke tools on the MCP server side?

### Implementing a Custom MCP Server

Creating a basic MCP server is straightforward. Here is an example server for working with local Git repositories using FastMCP: [GitHub 🐙](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-12/MCP/MCP_Server.py  )

### Implementing a Client to Work with the MCP Server

An example of a minimal client capable of interacting with the MCP server: [GitHub 🐙](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-12/MCP/MCP_Client.py  )

### Client Configuration File

Example implementation of the config file: [GitHub 🐙](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/develop/2025/week-12/MCP/config.json  )

## Interaction Between LLM Models and MCP Servers

### Technical Implementation of LLM-MCP Server Interaction

In this analysis, I will examine the technical implementation of interactions between large language models (LLMs) and Model Context Protocol (MCP) servers at the FastMCP library level, focusing on specific mechanisms and the programming interface.

### How the LLM Model Interacts with the MCP Server

### Declaration and Registration of Tools

On the server side, tools are declared using the `@mcp.tool()` decorator, which registers the function in the tool manager:

```python
@mcp.tool()
async def list_repositories() -> str:
    """
    Description:
    ---------------
        Returns a list of registered local Git repositories.
    """
    # Function implementation
    return result
```

Internally, the FastMCP `tool()` decorator adds the function to the tool manager:

```python
def tool(self, name: str | None = None, description: str | None = None) -> Callable:
    def decorator(fn: AnyFunction) -> AnyFunction:
        self.add_tool(fn, name=name, description=description)
        return fn
    return decorator

def add_tool(self, fn: AnyFunction, name: str | None = None, description: str | None = None) -> None:
    self._tool_manager.add_tool(fn, name=name, description=description)
```

Upon initializing the MCP server, it configures handlers for the protocol's core requests:

```python
def _setup_handlers(self) -> None:
    """Set up core MCP protocol handlers."""
    self._mcp_server.list_tools()(self.list_tools)
    self._mcp_server.call_tool()(self.call_tool)
    # ... other handlers ...
```

### Connection Establishment and Tool Discovery

When the MCP server starts, it waits for a connection:

```python
def run(self, transport: Literal["stdio", "sse"] = "stdio") -> None:
    if transport == "stdio":
        anyio.run(self.run_stdio_async)
    else:  # transport == "sse"
        anyio.run(self.run_sse_async)
```

When a client (containing the LLM) connects to the server, the first step is for the client to request the list of available tools via the `list_tools` method:

```python
async def list_tools(self) -> list[MCPTool]:
    """List all available tools."""
    tools = self._tool_manager.list_tools()
    return [
        MCPTool(
            name=info.name,
            description=info.description,
            inputSchema=info.parameters,
        )
        for info in tools
    ]
```

This method converts all registered tools into the `MCPTool` format, containing:
- Tool name
- Description (obtained from docstring)
- Input parameter schema (obtained from type annotations)

### Interaction Protocol

The MCP protocol supports two primary communication mechanisms: local communication via standard input/output and remote communication via SSE (Server-Sent Events).

Both mechanisms use JSON-RPC 2.0 for message transmission, ensuring standardized and scalable communication.

- **Local communication**: Data is transmitted via stdio, suitable for communication between clients and servers running on the same machine.
- **Remote communication**: SSE is combined with HTTP to enable real-time data transfer over networks, suitable for scenarios requiring access to remote resources or distributed deployment.

1. **stdio** - Communication via standard input/output:
```python
async def run_stdio_async(self) -> None:
    """Run the server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await self._mcp_server.run(
            read_stream,
            write_stream,
            self._mcp_server.create_initialization_options(),
        )
```

2. **SSE** (Server-Sent Events) - Communication via HTTP:
```python
async def run_sse_async(self) -> None:
    """Run the server using SSE transport."""
    # ... HTTP server setup ...
    server = uvicorn.Server(config)
    await server.serve()
```

JSON-RPC 2.0 is a lightweight protocol for remote procedure calls (RPC) using JSON (JavaScript Object Notation) to encode data. It allows a client to invoke methods on a server by passing parameters in JSON format and receiving responses also in JSON format.

Key characteristics of JSON-RPC 2.0:

1. **Simplicity**: The protocol is minimalist and easy to implement.
2. **Transport independence**: Can operate over various transport protocols such as HTTP, WebSocket, and others.
3. **Notifications**: Supports notifications, which do not require a server response.
4. **Batch requests**: Allows sending multiple requests in a single batch.
5. **Error handling**: Defines a standard format for error messages.

## How the LLM Model Invokes Tools on the MCP Server Side

### Tool Invocation Mechanism

When the LLM decides to invoke a tool, it forms a specific structure in its response:

```json
{
  "tool_calls": [
    {
      "id": "call_uniqueID",
      "type": "function",
      "function": {
        "name": "list_repositories",
        "arguments": "{}"
      }
    }
  ]
}
```

This call is converted by the client into a JSON-RPC request to the server:

```json
{
  "jsonrpc": "2.0",
  "method": "tool/call",
  "params": {
    "name": "list_repositories",
    "arguments": {}
  },
  "id": 1
}
```

On the server side, this request is handled by the `call_tool` method:

```python
async def call_tool(
    self, name: str, arguments: dict[str, Any]
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Call a tool by name with arguments."""
    context = self.get_context()
    result = await self._tool_manager.call_tool(name, arguments, context=context)
    converted_result = _convert_to_content(result)
    return converted_result
```

### Tool Execution Process

Here is the detailed process occurring on the server side:

1. **Obtaining execution context**:
```python
context = self.get_context()
```
The context contains information about the current request and session, allowing tools to interact with the client (e.g., sending intermediate results).

2. **Invoking the tool via the tool manager**:
```python
result = await self._tool_manager.call_tool(name, arguments, context=context)
```

Inside `ToolManager`, the following occurs:
- Tool lookup by name
- Argument validation against schema
- Function invocation with passed arguments
- Exception handling

3. **Converting the result into a standard format**:
```python
converted_result = _convert_to_content(result)
```

The `_convert_to_content` function converts the result (which may be a string, object, or other data type) into a standard representation:

```python
def _convert_to_content(
    result: Any,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert a result to a sequence of content objects."""
    if result is None:
        return []

    if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
        return [result]

    # ... handle other types ...

    # Convert to text if not a string
    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            result = str(result)

    return [TextContent(type="text", text=result)]
```

### Example of Full Execution Flow

Consider the full execution flow for invoking the `list_repositories` tool:

1. **LLM forms a tool call in its response**:
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_unique123",
      "type": "function",
      "function": {
        "name": "list_repositories",
        "arguments": "{}"
      }
    }
  ]
}
```

2. **Client converts the call to JSON-RPC and sends it to the server**:
```
2025-03-19 13:25:06,244 - mcp.server.lowlevel.server - INFO - Processing request of type CallToolRequest
```

3. **Server processes the request via the `call_tool` method**:
```python
async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[...]:
    context = self.get_context()
    result = await self._tool_manager.call_tool(name, arguments, context=context)
    converted_result = _convert_to_content(result)
    return converted_result
```

4. **ToolManager finds the function and invokes it**:
```python
# Inside ToolManager
tool_info = self._find_tool(name)
result = await self._invoke_tool(tool_info, arguments, context)
```

5. **The function decorated with `@mcp.tool()` executes**:
```python
@mcp.tool()
async def list_repositories() -> str:
    logger.info("Requesting repository list")
    repos = repo_manager.list_repositories()
    # ... form result ...
    return result
```

6. **Result is converted to standard format and returned to the client**:
```
2025-03-19 13:25:35,227 - __main__ - INFO - Repository registered: /path/to/repo
```

7. **Client converts the response and passes it to the LLM**:
```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": tool_result
})
```

8. **LLM generates the final response to the user**:
```
⭐ Iteration 2/5 ⭐
2025-03-19 13:29:39,274 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions   "HTTP/1.1 200 OK"
✅ Received response from model: {'role': 'assistant', 'content': '...interpretation of results...'}
```

### Execution Context and Additional Capabilities

FastMCP provides tools with an execution context via the `Context` class, enabling:

```python
@server.tool()
def tool_with_context(x: int, ctx: Context) -> str:
    # Logging
    ctx.info(f"Processing {x}")
    
    # Progress reporting
    ctx.report_progress(50, 100)
    
    # Resource access
    data = ctx.read_resource("resource://data")
    
    # Request information retrieval
    request_id = ctx.request_id
    
    return str(x)
```

This extends tool capabilities, allowing them to interact with the client during execution.

## Conclusion

The FastMCP library provides an elegant interface for creating MCP servers, abstracting protocol complexities and offering a simple path for tool registration via decorators.

LLMs interact with MCP servers through the standardized JSON-RPC protocol, using a specific response structure to invoke tools. The server converts these calls into executions of corresponding functions and returns results in a format interpretable by the LLM.

Key components of this interaction:
1. Decorators `@mcp.tool()` for tool registration
2. Tool discovery mechanism via `list_tools`
3. Tool invocation via `call_tool` method
4. Conversion of results into standard content format

This approach ensures modularity, extensibility, and standardization of interactions between language models and external tools, making MCP a powerful protocol for building integrated AI systems.

### Problems Solved

MCP solves a key problem of modern AI models—their potential being limited by data isolation. Before MCP:
- Data transfer occurred via manual copy/paste or upload/download
- Each new data source required individual configuration and implementation
- "Information islands" formed, limiting the capabilities of even the most powerful models

### Capabilities and Prospects

MCP enables building a direct "bridge" between AI and various data sources and tools, including:
- Local file systems
- Internet resources
- Development tools
- Web and browser automation tools
- Productivity and communication systems

With widespread adoption of the MCP standard, the possibility arises to realize the concept of an "Internet of Everything" in AI, enabling powerful collaborative capabilities across diverse systems and components.

MCP is designed to become an intermediary protocol layer that simplifies and standardizes the development and integration of AI applications, making the ecosystem more open, flexible, and functional.