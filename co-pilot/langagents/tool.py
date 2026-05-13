import sys
import shlex
import subprocess
from typing import Any
from langchain_core.tools import tool


@tool
def run_pytest(pytest_args: str = "", timeout_seconds: int = 120) -> dict[str, Any]:
	"""Run pytest with optional arguments and return captured output.
	Args:
		pytest_args: Extra pytest CLI arguments, e.g. "-q tests/test_api.py".
		timeout_seconds: Maximum time to wait for pytest.
	"""
	cmd = [sys.executable, "-m", "pytest", *shlex.split(pytest_args)]
	try:
		completed = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
			check=False,
		)
		return {
			"ok": completed.returncode == 0,
			"command": " ".join(cmd),
			"returncode": completed.returncode,
			"stdout": completed.stdout,
			"stderr": completed.stderr,
		}
	except subprocess.TimeoutExpired as exc:
		return {
			"ok": False,
			"command": " ".join(cmd),
			"returncode": None,
			"stdout": exc.stdout or "",
			"stderr": (exc.stderr or "") + f"\nTimed out after {timeout_seconds} seconds.",
		}


@tool
def run_shell_command(command: str, timeout_seconds: int = 60) -> dict[str, Any]:
	"""Run a shell command and return captured stdout/stderr.
	This executes through zsh ("zsh -lc") so pipes and shell syntax are supported.
	Use only with trusted command input.
	Args:
		command: Shell command to execute.
		timeout_seconds: Maximum time to wait for command completion.
	"""
	cmd = ["zsh", "-lc", command]
	try:
		completed = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
			check=False,
		)
		return {
			"ok": completed.returncode == 0,
			"command": command,
			"returncode": completed.returncode,
			"stdout": completed.stdout,
			"stderr": completed.stderr,
		}
	except subprocess.TimeoutExpired as exc:
		return {
			"ok": False,
			"command": command,
			"returncode": None,
			"stdout": exc.stdout or "",
			"stderr": (exc.stderr or "") + f"\nTimed out after {timeout_seconds} seconds.",
		}


TOOLS = [run_pytest, run_shell_command]
