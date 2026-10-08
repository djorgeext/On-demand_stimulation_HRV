#!/usr/bin/env python3
"""PreToolUse guard for every agent (main session and subagents).

The test split is opened once: for the final evaluation, after the user has created
.claude/FINAL_TEST_APPROVED by hand. Until then any tool call that reads the test files or
runs evaluate.py without `--split val` is blocked. Agents can never create or edit the
approval file. Exit code 2 blocks the call; the stderr text is shown to Claude.
This catches accidents, not determined circumvention - the rule also lives in CLAUDE.md.
"""
import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
APPROVAL = PROJECT / ".claude" / "FINAL_TEST_APPROVED"
TEST_FILES = [r"subjects_test_split\.json", r"hrv_test\.h5", r"--split[ =]+test\b"]
RUNS_EVALUATE = r"python[0-9.]*\s+(\S*[/\\])?evaluate\.py"   # executing it, not grepping it


def main():
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    tool = event.get("tool_name", "")
    text = " ".join(str(v) for v in (event.get("tool_input") or {}).values())

    if "FINAL_TEST_APPROVED" in text and tool != "Read":
        print("Blocked: only the user creates or removes .claude/FINAL_TEST_APPROVED.", file=sys.stderr)
        return 2

    touches_test = any(re.search(p, text) for p in TEST_FILES)
    if tool == "Bash" and re.search(RUNS_EVALUATE, text) and not re.search(r"--split[ =]+val\b", text) \
            and not re.search(r"(^|\s)(-h|--help)\b", text):
        touches_test = True  # evaluate.py defaults to the test split

    if touches_test and not APPROVAL.exists():
        print("Blocked: the test split is reserved for the single final evaluation. Use "
              "`--split val` for model selection. The user enables the final run by creating "
              ".claude/FINAL_TEST_APPROVED.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
