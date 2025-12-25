#!/usr/bin/env python3
"""
RAG Agent CLI Client - Simple HTTP client for the RAG server.

Pure argparse-based CLI for scripting and one-off commands.
For interactive use with autocomplete, use: python -m src.cli

Usage:
    python -m src.cli_client query "What is momentum trading?"
    python -m src.cli_client research "vectorbt Portfolio signals"
    python -m src.cli_client coder "write a momentum strategy"
    python -m src.cli_client debugger "fix this error: IndexError..."
    python -m src.cli_client search "backtest" --type hybrid --limit 10
    python -m src.cli_client agents
    python -m src.cli_client health

Server must be running:
    python -m src.server
"""

import argparse
import sys
import json
import os
from pathlib import Path
import httpx

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:8000"


def get_server_url() -> str:
    """Get server URL from environment or use default."""
    return os.getenv("RAG_SERVER_URL", DEFAULT_SERVER_URL)


def query_agent(question: str, agent: str = "orchestrator", output_json: bool = False) -> str:
    """Send a query to a specific agent via the server."""
    server_url = get_server_url()
    
    try:
        response = httpx.post(
            f"{server_url}/query",
            json={"query": question, "agent": agent},
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        
        if output_json:
            return json.dumps(data, indent=2)
        return data.get("response", str(data))
        
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {server_url}. Start with: python -m src.server"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error: {e}"


def search_documents(
    query: str,
    search_type: str = "hybrid",
    limit: int = 10,
    knowledge_base: str = "all",
    output_json: bool = False
) -> str:
    """Search the knowledge base via the server."""
    server_url = get_server_url()
    
    try:
        response = httpx.post(
            f"{server_url}/search",
            json={
                "query": query,
                "search_type": search_type,
                "match_count": limit,
                "knowledge_base": knowledge_base
            },
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        
        if output_json:
            return json.dumps(data, indent=2)
        
        results = data.get("results", [])
        if not results:
            return "No results found."
        
        output = [f"Found {len(results)} results for '{query}' ({search_type}, KB: {knowledge_base}):\n"]
        for i, r in enumerate(results, 1):
            score = r.get("score", "N/A")
            if isinstance(score, float):
                score = f"{score:.4f}"
            content = r.get("content", "")[:500]
            output.append(f"--- Result {i} (Score: {score}) ---")
            output.append(content + ("..." if len(r.get("content", "")) > 500 else ""))
            output.append("")
        
        return "\n".join(output)
        
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {server_url}"
    except Exception as e:
        return f"Error: {e}"


def review_files(
    file_paths: list[str],
    instructions: str = "",
    agent: str = "coder",
    output_json: bool = False
) -> str:
    """Read files and send them for review to an agent."""
    # Validate and read files
    file_contents = []
    errors = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            errors.append(f"File not found: {file_path}")
            continue
        if not path.is_file():
            errors.append(f"Not a file: {file_path}")
            continue
        
        try:
            content = path.read_text(encoding="utf-8")
            file_contents.append({
                "path": str(path.resolve()),
                "name": path.name,
                "content": content
            })
        except Exception as e:
            errors.append(f"Cannot read {file_path}: {e}")
    
    if errors:
        for error in errors:
            print(f"Warning: {error}", file=sys.stderr)
    
    if not file_contents:
        return "Error: No valid files to review."
    
    # Build review prompt
    prompt_parts = ["Please review the following file(s):\n"]
    
    for f in file_contents:
        prompt_parts.append(f"--- {f['name']} ({f['path']}) ---")
        prompt_parts.append(f"```\n{f['content']}\n```\n")
    
    if instructions:
        prompt_parts.append(f"\nReview instructions: {instructions}")
    else:
        prompt_parts.append("\nProvide a thorough code review including:")
        prompt_parts.append("- Code quality and readability")
        prompt_parts.append("- Potential bugs or issues")
        prompt_parts.append("- Suggestions for improvement")
        prompt_parts.append("- Best practices adherence")
    
    review_query = "\n".join(prompt_parts)
    
    return query_agent(review_query, agent, output_json)


def list_agents(output_json: bool = False) -> str:
    """List available agents (static list, agents are defined in code)."""
    # Agents are hardcoded in the codebase, not exposed via API
    agents = [
        {"name": "orchestrator", "description": "Routes requests to specialized agents"},
        {"name": "research", "description": "Searches knowledge bases for documentation"},
        {"name": "coder", "description": "Writes and generates trading code"},
        {"name": "debugger", "description": "Debugs and fixes code issues"},
    ]
    
    if output_json:
        return json.dumps({"agents": agents}, indent=2)
    
    output = ["Available Agents:"]
    for agent in agents:
        name = agent.get("name", "unknown")
        desc = agent.get("description", "No description")
        output.append(f"  {name:12} - {desc}")
    
    return "\n".join(output)


def get_server_info(output_json: bool = False) -> str:
    """Get server information."""
    server_url = get_server_url()
    
    try:
        response = httpx.get(f"{server_url}/info", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        if output_json:
            return json.dumps(data, indent=2)
        
        lines = ["Server Information:"]
        for key, value in data.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
        
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {server_url}"
    except Exception as e:
        return f"Error: {e}"


def health_check() -> bool:
    """Check if server is running."""
    try:
        response = httpx.get(f"{get_server_url()}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RAG Multi-Agent CLI Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli_client query "What is momentum trading?"
  python -m src.cli_client research "Portfolio.from_signals"
  python -m src.cli_client coder "write RSI strategy"
  python -m src.cli_client review src/agent.py --instructions "check for bugs"
  python -m src.cli_client review file1.py file2.py --agent debugger
  python -m src.cli_client search "backtest" --limit 5 --kb vectorbt
  python -m src.cli_client agents
  python -m src.cli_client health

For interactive mode with autocomplete: python -m src.cli
"""
    )
    
    parser.add_argument("--server", "-s", default=None, help="Server URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # query
    p = subparsers.add_parser("query", help="Query an agent")
    p.add_argument("question", help="Your question")
    p.add_argument("--agent", "-a", default="orchestrator", help="Agent to use")
    p.add_argument("--json", action="store_true", help="JSON output")
    
    # research
    p = subparsers.add_parser("research", help="Research Agent")
    p.add_argument("query", help="Research query")
    p.add_argument("--json", action="store_true")
    
    # coder
    p = subparsers.add_parser("coder", help="Coder Agent")
    p.add_argument("request", help="Coding request")
    p.add_argument("--json", action="store_true")
    
    # debugger
    p = subparsers.add_parser("debugger", help="Debugger Agent")
    p.add_argument("issue", help="Issue to debug")
    p.add_argument("--json", action="store_true")
    
    # review
    p = subparsers.add_parser("review", help="Review files for code quality")
    p.add_argument("files", nargs="+", help="File paths to review")
    p.add_argument("--instructions", "-i", default="", help="Custom review instructions")
    p.add_argument("--agent", "-a", default="coder", choices=["coder", "debugger"], help="Agent to use")
    p.add_argument("--json", action="store_true")
    
    # search
    p = subparsers.add_parser("search", help="Search documents")
    p.add_argument("query", help="Search query")
    p.add_argument("--type", "-t", choices=["semantic", "text", "hybrid"], default="hybrid")
    p.add_argument("--limit", "-l", type=int, default=10)
    p.add_argument("--kb", default="all", help="Knowledge base")
    p.add_argument("--json", action="store_true")
    
    # agents
    p = subparsers.add_parser("agents", help="List agents")
    p.add_argument("--json", action="store_true")
    
    # info
    p = subparsers.add_parser("info", help="Server info")
    p.add_argument("--json", action="store_true")
    
    # health
    subparsers.add_parser("health", help="Health check")
    
    args = parser.parse_args()
    
    if args.server:
        os.environ["RAG_SERVER_URL"] = args.server
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "query":
        print(query_agent(args.question, args.agent, args.json))
    elif args.command == "research":
        print(query_agent(args.query, "research", args.json))
    elif args.command == "coder":
        print(query_agent(args.request, "coder", args.json))
    elif args.command == "debugger":
        print(query_agent(args.issue, "debugger", args.json))
    elif args.command == "review":
        print(review_files(args.files, args.instructions, args.agent, args.json))
    elif args.command == "search":
        print(search_documents(args.query, args.type, args.limit, args.kb, args.json))
    elif args.command == "agents":
        print(list_agents(args.json))
    elif args.command == "info":
        print(get_server_info(args.json))
    elif args.command == "health":
        if health_check():
            print(f"OK: Server at {get_server_url()} is healthy")
            sys.exit(0)
        else:
            print(f"FAIL: Server at {get_server_url()} is not responding")
            sys.exit(1)


if __name__ == "__main__":
    main()
