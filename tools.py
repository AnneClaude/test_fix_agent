"""The agent's tools. Plain Python functions: no AI in this file.
Every tool returns a string, because that string is what the model sees."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_OUTPUT_CHARS = 4000  # long outputs waste tokens, so we cut them


def truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"
    return text


def make_workspace(target: Path) -> Path:
    """Copy the file under test into a fresh temporary folder with an empty tests/ folder.
    The agent works ONLY in this copy, so your real files are never touched."""
    ws = Path(tempfile.mkdtemp(prefix="agent_ws_"))
    shutil.copy(target, ws / target.name)
    (ws / "tests").mkdir()
    return ws


def read_file(ws: Path, path: str) -> str:
    """Return the contents of a file inside the workspace."""
    full_path = (ws / path).resolve()
    # Guardrail: block paths like "../../secret.txt" that escape the workspace
    if not full_path.is_relative_to(ws.resolve()):
        return "ERROR: you can only read files inside the workspace."
    if not full_path.exists():
        return f"ERROR: {path} does not exist."
    return truncate(full_path.read_text(encoding="utf-8"))


def write_test_file(ws: Path, filename: str, content: str) -> str:
    """Save a test file into tests/. The agent can write ONLY here."""
    # Guardrail: the agent must not edit the code under test to make failures disappear
    if "/" in filename or "\\" in filename:
        return "ERROR: give a plain filename like test_stack.py, without folders."
    if not (filename.startswith("test_") and filename.endswith(".py")):
        return "ERROR: test filenames must look like test_something.py"
    (ws / "tests" / filename).write_text(content, encoding="utf-8")
    return f"Saved tests/{filename}"


def run_tests(ws: Path) -> str:
    """Run pytest on the tests/ folder and return its output."""
    try:
        result = subprocess.run(
            # "python -m pytest" (instead of just "pytest") adds the workspace folder
            # to Python's import path, so tests can do "from stack import BoundedStack"
            [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
            cwd=ws,               # run inside the workspace
            capture_output=True,  # catch the output instead of printing it
            text=True,
            timeout=30,           # guardrail: buggy code could loop forever
        )
    except subprocess.TimeoutExpired:
        return "ERROR: tests took longer than 30 seconds (an infinite loop?)."
    return truncate(f"exit code: {result.returncode}\n{result.stdout}{result.stderr}")


# ---------------------------------------------------------------------------
# Manual check: run `python tools.py` to test every tool by hand, with no AI.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ws = make_workspace(Path("examples/stack.py"))
    print("1. Workspace created at:", ws)

    print("\n2. read_file:")
    print(read_file(ws, "stack.py")[:200], "...")

    print("\n3. read_file outside the workspace (should be blocked):")
    print(read_file(ws, "../../secret.txt"))

    print("\n4. write_test_file with a bad name (should be blocked):")
    print(write_test_file(ws, "stack.py", "oops"))

    print("\n5. write_test_file with a real test:")
    test_code = """
from stack import BoundedStack

def test_pop_returns_last_pushed():
    s = BoundedStack(3)
    s.push(1)
    s.push(2)
    assert s.pop() == 2
"""
    print(write_test_file(ws, "test_manual.py", test_code))

    print("\n6. run_tests:")
    print(run_tests(ws))