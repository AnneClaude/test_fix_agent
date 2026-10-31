"""Test-and-fix agent: reads a Python file, writes tests, runs them,
and reports real bugs. Usage:  python agent.py examples/stack.py"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from tools import make_workspace, read_file, write_test_file, run_tests

load_dotenv()

MAX_STEPS = 20  # guardrail: the agent can't loop forever

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=120,
)

# ---------------------------------------------------------------------------
# 1. System prompt: the agent's job description
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful software test engineer.
You are given one Python module. Your job is to find REAL bugs in it.

How to work:
1. Read the module. Understand its intended behavior from names, comments and docstrings.
2. Write pytest tests for normal cases AND edge cases (empty, full, zero, boundaries).
3. Run the tests.
4. For every failing test, decide: is the TEST wrong, or is the CODE wrong?
   - If the test is wrong, fix the test and run again.
   - If the code is wrong, keep the test: it proves a real bug.
5. When you are confident, call `finish` with the real bugs and the test that proves each one.

Rules:
- You can NOT change the module. You can only write files in tests/.
- Tests run from the workspace root, so import the module directly,
  e.g. `from {module} import ...`.
- Only report bugs that a failing test proves. Do not guess."""

# ---------------------------------------------------------------------------
# 2. Tool schemas: what the MODEL reads (same format as `add` in the playground)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the workspace, for example the module under test.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_test_file",
            "description": "Create or overwrite a pytest file in tests/. Filename must look like test_something.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string", "description": "The full Python source of the test file."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run all tests in tests/ with pytest and return the output.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when you are done. Report every real bug found, or an empty list if none.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "bugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One entry per real bug, naming the failing test that proves it.",
                    },
                },
                "required": ["summary", "bugs"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 3. execute_tool: turn the model's request into a real function call
# ---------------------------------------------------------------------------
def execute_tool(ws: Path, name: str, args: dict) -> str:
    try:
        if name == "read_file":
            return read_file(ws, **args)
        if name == "write_test_file":
            return write_test_file(ws, **args)
        if name == "run_tests":
            return run_tests(ws)
        return f"ERROR: unknown tool '{name}'."
    except TypeError as e:
        # The model sent wrong or missing arguments. Tell it, instead of crashing.
        return f"ERROR: bad arguments for {name}: {e}"

# ---------------------------------------------------------------------------
# 4. The agent loop
# ---------------------------------------------------------------------------
def run_agent(target: Path) -> dict:
    ws = make_workspace(target)
    print(f"Workspace: {ws}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(module=target.stem)},
        {"role": "user", "content": f"The module under test is `{target.name}`. Begin."},
    ]

    for step in range(1, MAX_STEPS + 1):
        # (a) Ask the model what to do next, sending the WHOLE history
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message

        # (b) Save the model's reply in the history (its memory)
        messages.append(message)

        # (c) The model replied with text instead of a tool call: nudge it to keep going
        if not message.tool_calls:
            print(f"[step {step}] model said: {message.content}")
            messages.append({
                "role": "user",
                "content": "Continue using the tools. Call `finish` when you are done.",
            })
            continue

        # (d) Run EVERY tool the model asked for, and report each result back
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None

            if name == "finish" and args is not None:
                print(f"[step {step}] finish")
                return args

            if args is None:
                result = "ERROR: your arguments were not valid JSON."
            else:
                result = execute_tool(ws, name, args)

            preview = result.replace("\n", " ")[:120]
            print(f"[step {step}] {name}({', '.join(args or {})}) -> {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return {"summary": f"Stopped after {MAX_STEPS} steps without calling finish.", "bugs": []}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python agent.py path/to/module.py")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"File not found: {target}")
        sys.exit(1)

    result = run_agent(target)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))