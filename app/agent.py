#!/usr/bin/env python3
"""
Plasmid Design Agent — Claude Agent SDK

A CLI agent for expression plasmid design powered by the Claude Agent SDK.
Uses the same tools and system prompt as the eval harness.

Usage:
    python app/agent.py "Design an EGFP expression plasmid using pcDNA3.1(+)"
    python app/agent.py --model opus "Put mCherry into a mammalian expression vector"
"""

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Add project root to path so src/ is importable as a package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionResultAllow,
)
from src.tools import create_plasmid_tools, ALL_TOOL_NAMES

# Load system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""


async def _auto_approve(tool_name, tool_input, context):
    """Auto-approve all tool calls (MCP tools are safe, in-process)."""
    return PermissionResultAllow()


async def run_plasmid_agent(
    user_prompt: str,
    model: str = "claude-opus-4-5-20251101",
    max_turns: int = 15,
    verbose: bool = False,
):
    """Run the plasmid design agent on a single prompt."""
    server_config = create_plasmid_tools()

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"plasmid-library": server_config},
        allowed_tools=ALL_TOOL_NAMES,
        permission_mode="acceptEdits",
        model=model,
        max_turns=max_turns,
        cwd=str(PROJECT_ROOT),
        can_use_tool=_auto_approve,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
                    elif isinstance(block, ToolUseBlock) and verbose:
                        print(f"\n  [tool] {block.name}({str(block.input)[:100]}...)")
                    elif isinstance(block, ToolResultBlock) and verbose:
                        preview = str(block.content)[:200]
                        print(f"  [result] {preview}...")
            elif isinstance(message, ResultMessage):
                if verbose:
                    print(f"\n\nDone. Cost: ${message.total_cost_usd:.4f}")
                break


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plasmid Design Agent (Claude Agent SDK)")
    parser.add_argument("prompt", nargs="?", help="Design prompt")
    parser.add_argument("--model", default="claude-opus-4-5-20251101", help="Model to use (default: claude-opus-4-5-20251101)")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if not args.prompt:
        print("Usage: python app/agent.py \"Your plasmid design prompt\"")
        sys.exit(1)

    await run_plasmid_agent(
        user_prompt=args.prompt,
        model=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )
    print()  # Final newline


if __name__ == "__main__":
    asyncio.run(main())
