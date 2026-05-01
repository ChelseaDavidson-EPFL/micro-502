#!/usr/bin/env python3
"""
run_webots_headless.py

Runs a Webots simulation in headless (no-GUI) mode a specified number of times.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_once(
    webots_exec: str,
    world_path: Path,
    run_index: int,
    timeout: float | None,
    suppress_stdout: bool,
    suppress_stderr: bool,
) -> int:
    """
    Launch Webots once in headless mode and wait for it to finish.

    Returns the process exit code.
    """
    # Quote any argument that contains spaces so the shell parses it correctly
    def quote(s: str) -> str:
        return f'"{s}"' if " " in s else s

    args = [
        webots_exec,
        #"--headless",       # No GUI window
        "--no-rendering",   # Skip rendering (faster for logic-only sims)
        "--stdout",         # Redirect robot/supervisor stdout to this process
        "--stderr",         # Redirect robot/supervisor stderr to this process
        "--batch",          # Disable dialog boxes (non-interactive)
        "--mode=fast",      # Run as fast as possible
        str(world_path),
    ]
    cmd = " ".join(quote(a) for a in args)

    stdout = subprocess.DEVNULL if suppress_stdout else None
    stderr = subprocess.DEVNULL if suppress_stderr else None

    print(f"[Run {run_index}] Starting: {cmd}")
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            shell=True,
        )
        elapsed = time.monotonic() - start
        print(f"[Run {run_index}] Finished in {elapsed:.1f}s — exit code {result.returncode}")
        return result.returncode

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        print(
            f"[Run {run_index}] TIMEOUT after {elapsed:.1f}s — process killed.",
            file=sys.stderr,
        )
        return -1


if __name__ == "__main__":
    # ── Configure your simulation here ──────────────────────────────────────
    WORLD_PATH   = Path("worlds/crazyflie_world_assignment.wbt")                          # Path to your .wbt file
    RUNS         = 100                                                        # Number of simulation runs
    WEBOTS_EXEC  = "C:/Program Files/Webots/msys64/mingw64/bin/webotsw.exe"       # Full path to Webots executable
    TIMEOUT      = None                                                      # Seconds per run, or None for no limit
    NO_STDOUT    = False                                                     # True to suppress Webots stdout
    NO_STDERR    = False                                                     # True to suppress Webots stderr
    # ────────────────────────────────────────────────────────────────────────

    if not WORLD_PATH.exists():
        print(f"ERROR: World file not found: {WORLD_PATH}", file=sys.stderr)
        sys.exit(1)

    if RUNS < 1:
        print("ERROR: RUNS must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    print(f"Webots executable : {WEBOTS_EXEC}")
    print(f"World file        : {WORLD_PATH}")
    print(f"Runs requested    : {RUNS}")
    print(f"Timeout per run   : {TIMEOUT if TIMEOUT else 'none'}")
    print("-" * 50)

    results: list[int] = []

    for i in range(1, RUNS + 1):
        exit_code = run_once(
            webots_exec=WEBOTS_EXEC,
            world_path=WORLD_PATH,
            run_index=i,
            timeout=TIMEOUT,
            suppress_stdout=NO_STDOUT,
            suppress_stderr=NO_STDERR,
        )
        results.append(exit_code)

    print("-" * 50)
    successes = sum(1 for c in results if c == 0)
    failures  = sum(1 for c in results if c != 0)
    print(f"Summary: {successes}/{RUNS} runs succeeded, {failures} failed.")

    if failures:
        sys.exit(1)
