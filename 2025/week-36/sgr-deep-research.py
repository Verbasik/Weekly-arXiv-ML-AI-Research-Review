#!/usr/bin/env python3
"""
SGR Research Agent - Schema-Guided Reasoning with Adaptive Planning
Clean implementation following SGR principles with clarification-first approach

🧠 Adaptive planning meets structured reasoning in perfect harmony
📎 Automatic citation management for academic excellence
🌍 Multi-language support with LLM-based detection
🔬 Production-ready research automation system
"""

import json
import os
import yaml
from datetime import datetime
from typing import List, Union, Literal, Optional, Dict, Any
try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8
from pydantic import BaseModel, Field
from annotated_types import MinLen, MaxLen
from openai import OpenAI
from tavily import TavilyClient
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config():
    """Load configuration from config.yaml and environment variables"""
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'openai_base_url': os.getenv('OPENAI_BASE_URL', ''),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'max_tokens': int(os.getenv('MAX_TOKENS', '8000')),
        'temperature': float(os.getenv('TEMPERATURE', '0.4')),
        'tavily_api_key': os.getenv('TAVILY_API_KEY', ''),
        'max_search_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
        'max_execution_steps': int(os.getenv('MAX_EXECUTION_STEPS', '6')),
        'reports_directory': os.getenv('REPORTS_DIRECTORY', 'reports'),

    }

    if os.path.exists('config.yaml'):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                if 'openai' in yaml_config:
                    openai_cfg = yaml_config['openai']
                    config['openai_api_key'] = openai_cfg.get('api_key', config['openai_api_key'])
                    config['openai_base_url'] = openai_cfg.get('base_url', config['openai_base_url'])
                    config['openai_model'] = openai_cfg.get('model', config['openai_model'])
                    config['max_tokens'] = openai_cfg.get('max_tokens', config['max_tokens'])
                    config['temperature'] = openai_cfg.get('temperature', config['temperature'])

                if 'tavily' in yaml_config:
                    config['tavily_api_key'] = yaml_config['tavily'].get('api_key', config['tavily_api_key'])

                if 'search' in yaml_config:
                    config['max_search_results'] = yaml_config['search'].get('max_results', config['max_search_results'])

                if 'execution' in yaml_config:
                    config['max_execution_steps'] = yaml_config['execution'].get('max_steps', config['max_execution_steps'])
                    config['reports_directory'] = yaml_config['execution'].get('reports_dir', config['reports_directory'])

        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")

    return config

CONFIG = load_config()

# Check required parameters
if not CONFIG['openai_api_key']:
    print("ERROR: OPENAI_API_KEY not set in config.yaml or environment")
    exit(1)

if not CONFIG['tavily_api_key']:
    print("ERROR: TAVILY_API_KEY not set in config.yaml or environment")
    exit(1)

# =============================================================================
# SGR SCHEMAS - Core Schema-Guided Reasoning Definitions
# =============================================================================

class Clarification(BaseModel):
    """Ask clarifying questions when facing ambiguous requests"""
    tool: Literal["clarification"]
    reasoning: str = Field(description="Why clarification is needed")
    unclear_terms: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(description="List of unclear terms or concepts")
    assumptions: Annotated[List[str], MinLen(2), MaxLen(4)] = Field(description="Possible interpretations to verify")
    questions: Annotated[List[str], MinLen(3), MaxLen(5)] = Field(description="3-5 specific clarifying questions")

class GeneratePlan(BaseModel):
    """Generate research plan based on clear user request"""
    tool: Literal["generate_plan"]
    reasoning: str = Field(description="Justification for research approach")
    research_goal: str = Field(description="Primary research objective")
    planned_steps: Annotated[List[str], MinLen(3), MaxLen(4)] = Field(description="List of 3-4 planned steps")
    search_strategies: Annotated[List[str], MinLen(2), MaxLen(3)] = Field(description="Information search strategies")

class WebSearch(BaseModel):
    """Search for information with credibility focus"""
    tool: Literal["web_search"]
    reasoning: str = Field(description="Why this search is needed and what to expect")
    query: str = Field(description="Search query in same language as user request")
    max_results: int = Field(default=10, description="Maximum results (1-15)")
    plan_adapted: bool = Field(default=False, description="Is this search after plan adaptation?")

class AdaptPlan(BaseModel):
    """Adapt research plan based on new findings"""
    tool: Literal["adapt_plan"]
    reasoning: str = Field(description="Why plan needs adaptation based on new data")
    original_goal: str = Field(description="Original research goal")
    new_goal: str = Field(description="Updated research goal")
    plan_changes: Annotated[List[str], MinLen(1), MaxLen(3)] = Field(description="Specific changes made to plan")
    next_steps: Annotated[List[str], MinLen(2), MaxLen(4)] = Field(description="Updated remaining steps")

class CreateReport(BaseModel):
    """Create comprehensive research report with citations"""
    tool: Literal["create_report"]
    reasoning: str = Field(description="Why ready to create report now")
    title: str = Field(description="Report title")
    user_request_language_reference: str = Field(
        description="Copy of original user request to ensure language consistency"
    )
    content: str = Field(description="""
    DETAILED technical content (800+ words) with in-text citations.

    🚨 CRITICAL LANGUAGE REQUIREMENT 🚨:
    - WRITE ENTIRELY IN THE SAME LANGUAGE AS user_request_language_reference
    - If user_request_language_reference is in Russian → WRITE IN RUSSIAN
    - If user_request_language_reference is in English → WRITE IN ENGLISH
    - DO NOT mix languages - use ONLY the language from user_request_language_reference

    OTHER REQUIREMENTS:
    - Include in-text citations for EVERY fact using [1], [2], [3] etc.
    - Citations must be integrated into sentences, not separate
    - Example Russian: "Apple M5 использует 3нм процесс [1], что улучшает производительность [2]."
    - Example English: "Apple M5 uses 3nm process [1] which improves performance [2]."

    Structure:
    1. Executive Summary / Исполнительное резюме
    2. Technical Analysis / Технический анализ (with citations)
    3. Key Findings / Ключевые выводы
    4. Conclusions / Заключения

    🚨 LANGUAGE COMPLIANCE: Text MUST match user_request_language_reference language 100%
    """)
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in findings")

class ReportCompletion(BaseModel):
    """Complete research task"""
    tool: Literal["report_completion"]
    reasoning: str = Field(description="Why research is now complete")
    completed_steps: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(description="Summary of completed steps")
    status: Literal["completed", "failed"] = Field(description="Task completion status")

# =============================================================================
# MAIN SGR SCHEMA - Adaptive Reasoning Core
# =============================================================================

class NextStep(BaseModel):
    """SGR Core - Determines next reasoning step with adaptive planning"""

    # Reasoning chain - step-by-step thinking process (helps stabilize model)
    reasoning_steps: Annotated[List[str], MinLen(2), MaxLen(4)] = Field(
        description="Step-by-step reasoning process leading to decision"
    )

    # Reasoning and state assessment
    current_situation: str = Field(description="Current research situation analysis")
    plan_status: str = Field(description="Status of current plan execution")

    # Progress tracking (IMPORTANT: Use these to avoid infinite loops)
    searches_done: int = Field(default=0, description="Number of searches completed (MAX 3-4 searches)")
    enough_data: bool = Field(default=False, description="Sufficient data for report? (True after 2-3 searches)")

    # Next step planning
    remaining_steps: Annotated[List[str], MinLen(1), MaxLen(3)] = Field(description="1-3 remaining steps to complete task")
    task_completed: bool = Field(description="Is the research task finished?")

    # Tool routing with clarification-first bias
    function: Union[
        Clarification,      # FIRST PRIORITY: When uncertain
        GeneratePlan,       # SECOND: When request is clear
        WebSearch,          # Core research tool
        AdaptPlan,          # When findings conflict with plan
        CreateReport,       # When sufficient data collected
        ReportCompletion    # Task completion
    ] = Field(description="""
    DECISION PRIORITY (BIAS TOWARD CLARIFICATION):

    1. If ANY uncertainty about user request → Clarification
    2. If no plan exists and request is clear → GeneratePlan
    3. If need to adapt research approach → AdaptPlan
    4. If need more information AND searches_done < 3 → WebSearch
    5. If searches_done >= 2 OR enough_data = True → CreateReport
    6. If report created → ReportCompletion

    CLARIFICATION TRIGGERS:
    - Unknown terms, acronyms, abbreviations
    - Ambiguous requests with multiple interpretations
    - Missing context for specialized domains
    - Any request requiring assumptions

    ANTI-CYCLING RULES:
    - Max 1 clarification per session
    - Max 3-4 searches per session
    - Create report after 2-3 searches regardless of completeness
    """)

# =============================================================================
# PROMPTS - System Instructions
# =============================================================================

def get_system_prompt(user_request: str) -> str:
    """Generate system prompt with user request for language detection"""

    return f"""
You are an expert researcher with adaptive planning and Schema-Guided Reasoning capabilities.

USER REQUEST EXAMPLE: "{user_request}"
↑ IMPORTANT: Detect the language from this request and use THE SAME LANGUAGE for all responses, searches, and reports.

CORE PRINCIPLES:
1. CLARIFICATION FIRST: For ANY uncertainty - ask clarifying questions
2. DO NOT make assumptions - better ask than guess wrong
3. Adapt plan when new data conflicts with initial assumptions
4. Search queries in SAME LANGUAGE as user request
5. REPORT ENTIRELY in SAME LANGUAGE as user request
6. Every fact in report MUST have inline citation [1], [2], [3] integrated into sentences

WORKFLOW:
0. clarification (HIGHEST PRIORITY) - when request unclear
1. generate_plan - create research plan
2. web_search - gather information (2-3 searches MAX)
   - Use SPECIFIC terms and context in search queries
   - For acronyms like "SGR", add context: "SGR Schema-Guided Reasoning"
   - Use quotes for exact phrases: "Structured Output OpenAI"
   - SEARCH QUERIES in SAME LANGUAGE as user request
   - STOP after 2-3 searches and create report
3. adapt_plan - adapt when conflicts found
4. create_report - create detailed report with citations
5. report_completion - complete task

ANTI-CYCLING: Maximum 1 clarification request per session.

ADAPTIVITY: Actively change plan when discovering new data.

LANGUAGE ADAPTATION: Always respond and create reports in the SAME LANGUAGE as the user's request. If user writes in Russian - respond in Russian, if in English - respond in English.
        """.strip()

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize OpenAI client with base_url if provided
openai_kwargs = {'api_key': CONFIG['openai_api_key']}
if CONFIG['openai_base_url']:
    openai_kwargs['base_url'] = CONFIG['openai_base_url']

client = OpenAI(**openai_kwargs)
tavily = TavilyClient(CONFIG['tavily_api_key'])
console = Console()
print = console.print

# Simple in-memory context
CONTEXT = {
    "plan": None,
    "searches": [],
    "sources": {},  # url -> citation_number mapping
    "citation_counter": 0,
    "clarification_used": False  # Anti-cycling mechanism
}

# =============================================================================
# UTILITIES
# =============================================================================



def add_citation(url: str, title: str = "") -> int:
    """Add source and return citation number"""
    if url in CONTEXT["sources"]:
        return CONTEXT["sources"][url]["number"]

    CONTEXT["citation_counter"] += 1
    number = CONTEXT["citation_counter"]

    CONTEXT["sources"][url] = {
        "number": number,
        "title": title,
        "url": url
    }

    return number

def format_sources() -> str:
    """Format sources for report"""
    if not CONTEXT["sources"]:
        return ""

    sources_text = "\n\n## Sources\n"

    for url, data in CONTEXT["sources"].items():
        number = data["number"]
        title = data["title"]
        if title:
            sources_text += f"- [{number}] {title} - {url}\n"
        else:
            sources_text += f"- [{number}] {url}\n"

    return sources_text

# =============================================================================
# DISPATCH - Tool Execution
# =============================================================================

def dispatch(cmd: BaseModel, context: Dict[str, Any]) -> Any:
    """Execute SGR commands"""

    if isinstance(cmd, Clarification):
        # Mark clarification as used to prevent cycling
        context["clarification_used"] = True

        print(f"\n🤔 [bold yellow]CLARIFICATION NEEDED[/bold yellow]")
        print(f"💭 Reason: {cmd.reasoning}\n")

        if cmd.unclear_terms:
            print(f"❓ [bold]Unclear terms:[/bold] {', '.join(cmd.unclear_terms)}")

        print(f"\n[bold cyan]CLARIFYING QUESTIONS:[/bold cyan]")
        for i, question in enumerate(cmd.questions, 1):
            print(f"   {i}. {question}")

        if cmd.assumptions:
            print(f"\n[bold green]Possible interpretations:[/bold green]")
            for assumption in cmd.assumptions:
                print(f"   • {assumption}")

        print(f"\n[bold yellow]⏸️  Research paused - please answer questions above[/bold yellow]")

        return {
            "tool": "clarification",
            "questions": cmd.questions,
            "status": "waiting_for_user"
        }

    elif isinstance(cmd, GeneratePlan):
        plan = {
            "research_goal": cmd.research_goal,
            "planned_steps": cmd.planned_steps,
            "search_strategies": cmd.search_strategies,
            "created_at": datetime.now().isoformat()
        }

        context["plan"] = plan

        print(f"📋 [bold]Research Plan Created:[/bold]")
        print(f"🎯 Goal: {cmd.research_goal}")
        print(f"📝 Steps: {len(cmd.planned_steps)}")
        for i, step in enumerate(cmd.planned_steps, 1):
            print(f"   {i}. {step}")

        return plan

    elif isinstance(cmd, WebSearch):
        print(f"🔍 [bold cyan]Search query:[/bold cyan] [white]'{cmd.query}'[/white]")

        try:
            response = tavily.search(
                query=cmd.query,
                max_results=cmd.max_results
            )

            # Add citations
            citation_numbers = []
            for result in response.get('results', []):
                url = result.get('url', '')
                title = result.get('title', '')
                if url:
                    citation_num = add_citation(url, title)
                    citation_numbers.append(citation_num)

            search_result = {
                "query": cmd.query,
                "answer": response.get('answer', ''),
                "results": response.get('results', []),
                "citation_numbers": citation_numbers,
                "timestamp": datetime.now().isoformat()
            }

            context["searches"].append(search_result)

            print(f"🔍 Found {len(citation_numbers)} sources")
            for i, (result, citation_num) in enumerate(zip(response.get('results', [])[:5], citation_numbers), 1):
                print(f"   {i}. [{citation_num}] {result.get('url', '')}")

            return search_result

        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    elif isinstance(cmd, AdaptPlan):
        if context.get("plan"):
            context["plan"]["research_goal"] = cmd.new_goal
            context["plan"]["planned_steps"] = cmd.next_steps
            context["plan"]["adapted"] = True
            context["plan"]["adaptations"] = context["plan"].get("adaptations", []) + [cmd.plan_changes]

        print(f"\n🔄 [bold yellow]PLAN ADAPTED![/bold yellow]")
        print(f"📝 [bold]Changes:[/bold]")
        for change in cmd.plan_changes:
            print(f"   • [yellow]{change}[/yellow]")
        print(f"🎯 [bold green]New goal:[/bold green] {cmd.new_goal}")

        return {
            "tool": "adapt_plan",
            "original_goal": cmd.original_goal,
            "new_goal": cmd.new_goal,
            "changes": cmd.plan_changes
        }

    elif isinstance(cmd, CreateReport):
        # Debug: Log CreateReport fields
        print(f"📝 [bold cyan]CREATE REPORT FULL DEBUG:[/bold cyan]")
        print(f"   🌍 Language Reference: '{cmd.user_request_language_reference}'")
        print(f"   📊 Title: '{cmd.title}'")
        print(f"   🔍 Reasoning: '{cmd.reasoning[:150]}...'")
        print(f"   📈 Confidence: {cmd.confidence}")
        print(f"   📄 Content Preview: '{cmd.content[:200]}...'")
        print(f"   🌐 Content Language Detected: {'Russian' if 'Apple' in cmd.content and ('характеристик' in cmd.content or 'цен' in cmd.content) else 'English'}")

        # Save report
        os.makedirs(CONFIG['reports_directory'], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in cmd.title if c.isalnum() or c in (' ', '-', '_'))[:50]
        filename = f"{timestamp}_{safe_title}.md"
        filepath = os.path.join(CONFIG['reports_directory'], filename)

        # Format full report with sources
        full_content = f"# {cmd.title}\n\n"
        full_content += f"*Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        full_content += cmd.content
        full_content += format_sources()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)

        report = {
            "title": cmd.title,
            "content": cmd.content,
            "confidence": cmd.confidence,
            "sources_count": len(context["sources"]),
            "word_count": len(cmd.content.split()),
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }

        print(f"📄 [bold blue]Report Created:[/bold blue] {cmd.title}")
        print(f"📊 Words: {report['word_count']}, Sources: {report['sources_count']}")
        print(f"💾 Saved: {filepath}")
        print(f"📈 Confidence: {cmd.confidence}")

        return report

    elif isinstance(cmd, ReportCompletion):
        print(f"\n✅ [bold green]RESEARCH COMPLETED[/bold green]")
        print(f"📋 Status: {cmd.status}")

        if cmd.completed_steps:
            print(f"📝 [bold]Completed steps:[/bold]")
            for step in cmd.completed_steps:
                print(f"   • {step}")

        return {
            "tool": "report_completion",
            "status": cmd.status,
            "completed_steps": cmd.completed_steps
        }

    else:
        return f"Unknown command: {type(cmd)}"

# =============================================================================
# MAIN EXECUTION ENGINE
# =============================================================================

def execute_research_task(task: str) -> str:
    """Execute research task using SGR"""

    print(Panel(task, title="🔍 Research Task", title_align="left"))

    # Use universal system prompt with user request for language detection
    system_prompt = get_system_prompt(task)

    print(f"\n[bold green]🚀 SGR RESEARCH STARTED[/bold green]")
    print(f"[dim]🤖 Model: {CONFIG['openai_model']}[/dim]")
    print(f"[dim]🔗 Base URL: {CONFIG['openai_base_url'] or 'default'}[/dim]")
    print(f"[dim]🔑 API Key: {'✓ Configured' if CONFIG['openai_api_key'] else '✗ Missing'}[/dim]")
    print(f"[dim]📊 Max tokens: {CONFIG['max_tokens']}, Temperature: {CONFIG['temperature']}[/dim]")

    # Initialize conversation log
    log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    # Execute reasoning steps
    for i in range(CONFIG['max_execution_steps']):
        step_id = f"step_{i+1}"
        print(f"\n🧠 {step_id}: Planning next action...")

        # Add context about clarification usage and available sources
        context_msg = ""
        if CONTEXT["clarification_used"]:
            context_msg = "IMPORTANT: Clarification already used. Do not request clarification again - proceed with available information."

        # Add original user request for language reference and search count
        searches_count = len(CONTEXT.get("searches", []))
        user_request_info = f"\nORIGINAL USER REQUEST: '{task}'\n(Use this for language consistency in reports)"
        search_count_info = f"\nSEARCHES COMPLETED: {searches_count} (MAX 3-4 searches before creating report)"
        context_msg = context_msg + "\n" + user_request_info + search_count_info if context_msg else user_request_info + search_count_info

        # Add available sources information
        if CONTEXT["sources"]:
            sources_info = "\nAVAILABLE SOURCES FOR CITATIONS:\n"
            for url, data in CONTEXT["sources"].items():
                number = data["number"]
                title = data["title"] or "Untitled"
                sources_info += f"[{number}] {title} - {url}\n"
            sources_info += "\nUSE THESE EXACT NUMBERS [1], [2], [3] etc. in your report citations."
            context_msg = context_msg + "\n" + sources_info if context_msg else sources_info

        if context_msg:
            log.append({"role": "system", "content": context_msg})
            # Debug: Show context being sent
            print(f"[dim]🔧 Context: {context_msg[:150]}...[/dim]")

        try:
            completion = client.beta.chat.completions.parse(
                model=CONFIG['openai_model'],
                response_format=NextStep,
                messages=log,
                max_tokens=CONFIG['max_tokens'],
                temperature=CONFIG['temperature']
            )

            job = completion.choices[0].message.parsed

            if job is None:
                print("[bold red]❌ Failed to parse LLM response[/bold red]")
                break

            # Debug: Log ALL NextStep fields
            print(f"🤖 [bold magenta]LLM RESPONSE DEBUG:[/bold magenta]")
            print(f"   🧠 Reasoning Steps: {job.reasoning_steps}")
            print(f"   📊 Current Situation: '{job.current_situation[:100]}...'")
            print(f"   📋 Plan Status: '{job.plan_status[:100]}...'")
            print(f"   🔍 Searches Done: {job.searches_done}")
            print(f"   ✅ Enough Data: {job.enough_data}")
            print(f"   📝 Remaining Steps: {job.remaining_steps}")
            print(f"   🏁 Task Completed: {job.task_completed}")
            print(f"   🔧 Tool: {job.function.tool}")

        except Exception as e:
            print(f"[bold red]❌ LLM request error: {str(e)}[/bold red]")
            break

        # Check for task completion
        if job.task_completed or isinstance(job.function, ReportCompletion):
            print(f"[bold green]✅ Task completed[/bold green]")
            dispatch(job.function, CONTEXT)
            break

        # Check for clarification cycling
        if isinstance(job.function, Clarification) and CONTEXT["clarification_used"]:
            print(f"[bold red]❌ Clarification cycling detected - forcing continuation[/bold red]")
            log.append({
                "role": "user",
                "content": "ANTI-CYCLING: Clarification already used. Continue with generate_plan based on available information."
            })
            continue

        # Display current step
        next_step = job.remaining_steps[0] if job.remaining_steps else "Completing"
        print(f"[blue]{next_step}[/blue]")
        print(f"[dim]💭 Reasoning: {job.function.reasoning[:100]}...[/dim]")
        print(f"  Tool: {job.function.tool}")

        # Handle clarification specially
        if isinstance(job.function, Clarification):
            result = dispatch(job.function, CONTEXT)
            return "CLARIFICATION_NEEDED"

        # Add to conversation log
        log.append({
            "role": "assistant",
            "content": next_step,
            "tool_calls": [{
                "type": "function",
                "id": step_id,
                "function": {
                    "name": job.function.tool,
                    "arguments": job.function.model_dump_json()
                }
            }]
        })

        # Execute tool
        result = dispatch(job.function, CONTEXT)

        # Add result to log - format search results better
        if isinstance(job.function, WebSearch) and isinstance(result, dict):
            # Format search results for better LLM understanding
            formatted_result = f"Search Query: {result.get('query', '')}\n\n"

            # Include answer only if it exists (with include_answer=True)
            if result.get('answer'):
                formatted_result += f"AI Answer: {result.get('answer')}\n\n"

            formatted_result += "Search Results:\n"
            for i, source_result in enumerate(result.get('results', [])[:5], 1):
                citation_num = result.get('citation_numbers', [])[i-1] if i-1 < len(result.get('citation_numbers', [])) else i
                title = source_result.get('title', 'Untitled')
                url = source_result.get('url', '')
                content = source_result.get('content', '')[:300] + "..." if source_result.get('content', '') else ""
                formatted_result += f"[{citation_num}] {title}\n{url}\n{content}\n\n"

            result_text = formatted_result
        else:
            result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        log.append({"role": "tool", "content": result_text, "tool_call_id": step_id})

        print(f"  Result: {result_text[:100]}..." if len(result_text) > 100 else f"  Result: {result_text}")

        # Auto-complete after report creation
        if isinstance(job.function, CreateReport):
            print(f"\n✅ [bold green]Auto-completing after report creation[/bold green]")
            break

    return "COMPLETED"

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    """Main application interface"""
    print("[bold]🧠 SGR Research Agent - Adaptive Planning & Clarification[/bold]")
    print("Schema-Guided Reasoning with plan adaptation capabilities")
    print()
    print("Core features:")
    print("  🤔 Clarification-first approach")
    print("  🔄 Adaptive plan modification")
    print("  📎 Automatic citation management")
    print("  🌍 Multi-language support")
    print()

    awaiting_clarification = False
    original_task = ""

    while True:
        try:
            print("=" * 60)
            if awaiting_clarification:
                response = input("💬 Your clarification response (or 'quit'): ").strip()
                awaiting_clarification = False

                if response.lower() in ['quit', 'exit']:
                    break

                # Combine original task with clarification
                task = f"Original request: '{original_task}'\nClarification: {response}\n\nProceed with research based on clarification."

                # Reset clarification flag for new combined task
                CONTEXT["clarification_used"] = False
            else:
                task = input("🔍 Enter research task (or 'quit'): ").strip()

            if task.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break

            if not task:
                print("❌ Empty task. Try again.")
                continue

            # Reset context for new task (except during clarification)
            if not awaiting_clarification:
                CONTEXT.clear()
                CONTEXT.update({
                    "plan": None,
                    "searches": [],
                    "sources": {},
                    "citation_counter": 0,
                    "clarification_used": False
                })
                original_task = task

            result = execute_research_task(task)

            if result == "CLARIFICATION_NEEDED":
                awaiting_clarification = True
                continue

            # Show statistics
            searches_count = len(CONTEXT.get("searches", []))
            sources_count = len(CONTEXT.get("sources", {}))
            print(f"\n📊 Session stats: 🔍 {searches_count} searches, 📎 {sources_count} sources")
            print(f"📁 Reports saved to: ./{CONFIG['reports_directory']}/")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()