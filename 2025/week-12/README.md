# Model Context Protocol (MCP)

## Description
This repository contains an overview of the **Model Context Protocol (MCP)** developed by Anthropic. MCP is a new protocol that standardizes interactions between large language models (LLMs) and external tools and data, making the AI ecosystem more modular, extensible, and universal.

## Overview Contents
1. **What is MCP?** 🤔
   - MCP is an open standard introduced by Anthropic on November 25, 2024.
   - Its primary goal is to create a unified interface for LLMs to interact with external data sources and tools.

2. **MCP Architecture** 🏗️
   - **Host**: Coordinates the system and manages LLM interactions.
   - **Clients**: Enable connections between hosts and servers.
   - **Servers**: Provide tools, resources, and prompt templates.
   - **Base Protocol**: Defines interactions between components.

3. **Advantages of MCP** ✨
   - **Clustering**: Grouping functions into a single service.
   - **Decoupling**: Independent deployment of tools.
   - **Reusability**: Universal protocols enabling tool reuse.
   - **Unification**: A single protocol for both client-side and cloud-based tools.

4. **How MCP Works** ⚙️
   - LLMs interact with MCP servers via JSON-RPC.
   - Tools are registered on the server using the `@mcp.tool()` decorator.
   - Clients request a list of tools and invoke them through a standardized protocol.

5. **Use Cases** 🛠️
   - Integration with local file systems.
   - Interaction with internet resources and development tools.
   - Web and browser automation.

6. **Future Prospects of MCP** 🚀
   - MCP has the potential to become a foundational protocol for realizing the "Internet of Everything" in AI.
   - Simplification of AI application development and integration.

## How to Use This Overview?
- Review the file [review.md](review.md) for a detailed analysis of MCP.
- Examine examples of MCP server and client implementations in the [MCP](MCP) folder.

## Links
- [Anthropic's Publication](https://www.anthropic.com/news/model-context-protocol  )

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>