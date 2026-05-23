"""SWE-bench prompts for MainAgent."""
from typing import Any, Dict, List

from aorchestra.main_agent import build_model_pricing_table


# SWE-bench SubAgent uses ACI commands, not explicit tool objects
SWEBENCH_TOOLS_DESCRIPTION = """
SubAgent uses ACI (Agentic Code Interface) commands to navigate and edit code.

=== FILE VIEWING ===
- open <file> [line]: Open file at specified line
- scroll_down/scroll_up: Navigate within open file
- goto <line>: Jump to specific line

=== FILE EDITING ===
- str_replace <file>: Replace exact text match (preferred for edits)
- edit <start>:<end>: Edit line range with syntax check
- insert <file> <line>: Insert content after line
- create <file>: Create new file

=== SEARCHING ===
- search_file <pattern> [file]: Search content in file
- search_dir <pattern> [dir]: Search in directory
- find_file <name> [dir]: Find files by name

=== BASH ===
- Any bash command (ls, cat, grep, python, pytest, etc.)

=== SUBMIT ===
- submit: Submit fix and run official tests
""".strip()


class SWEBenchMainAgentPrompt:
    """Build prompts for SWE-bench tasks (GitHub issue fixing)."""
    
    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        tools: List[Any] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index + 1
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        
        # Extract repository info from metadata
        repo = meta.get("repo", "unknown")
        instance_id = meta.get("instance_id", "unknown")
        
        # Budget warning
        if remaining_attempts <= 2:
            budget_warning = f"🚨 CRITICAL: Only {remaining_attempts} attempt(s) left! Submit now if fix is verified, or make final attempt."
        elif remaining_attempts <= 4:
            budget_warning = f"⚠️ Warning: {remaining_attempts} attempts remaining. Plan carefully."
        else:
            budget_warning = ""
        
        return f"""You are the MainAgent (Orchestrator) for a SWE-bench task. Your goal is to fix a GitHub issue by delegating work to SubAgents.


==== TASK ====
{instruction}

REPOSITORY: {repo}
INSTANCE: {instance_id}

==== DECISION PROCESS ====
1. READ the TASK carefully - understand the GitHub issue and what needs to be fixed
2. REVIEW SUBTASK HISTORY - check SubAgent's progress, completed steps, and test results
3. VERIFY against TASK requirements:
   - Did SubAgent locate the buggy code?
   - Did SubAgent make appropriate code changes?
   - Did SubAgent run tests and confirm the fix works?
4. CHECK REPRODUCTION EVIDENCE before considering 'submit':
   A 'submit' is only valid when SUBTASK HISTORY (or SubAgent's finish message)
   shows ALL three of the following — quote the lines to yourself:
     (a) a reproduction script exists at /testbed/reproduce_issue.py,
     (b) it was run on the original code and DID exhibit the bug (the
         expected error / wrong output appeared),
     (c) it was re-run AFTER the fix and now exits cleanly (the bug is gone).
   If any of (a)(b)(c) is missing or unclear, DO NOT submit — delegate one
   more round explicitly asking the SubAgent to fill the gap.
5. DECIDE:
   - ✅ status="done" AND reproduction evidence (a)(b)(c) all present → Use 'submit'
   - ⚠️ status="done" BUT reproduction evidence incomplete → Use 'delegate_task'
     asking SubAgent to produce/run the reproduce script
   - ⚠️ status="done" BUT tests fail or other gaps → Use 'delegate_task' to fix
   - ⚠️ status="partial" → Use 'delegate_task' with guidance on next steps

CRITICAL: SWE-BENCH CONTAINER BEHAVIOR
- The container is REUSED across rounds — code changes from prior SubAgent rounds
  are still on disk. New SubAgents should `git status` inside /testbed to see
  prior progress instead of redoing it.
- 'submit' runs the official test suite to determine success. A premature
  submit (before reproduction is verified) wastes the only graded attempt
  for this instance.


==== MODEL SELECTION ====
{model_pricing_table}

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts
{budget_warning}


==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS ====
{SWEBENCH_TOOLS_DESCRIPTION}

==== OUTPUT ====
Return JSON. The 'reasoning' field MUST quote the reproduction evidence
(steps 4(a)(b)(c) above) when choosing 'submit'.

If SubAgent status="done" AND reproduction evidence (a)(b)(c) all present:
{{
  "action": "submit",
  "reasoning": "Verified end-to-end: (a) reproduce_issue.py at /testbed exists; (b) before fix it produced: <quote bug output>; (c) after fix it produced: <quote success output>. Submitting for grading.",
  "params": {{ "reason": "Fix verified by reproduction: [specific fix description]" }}
}}

If SubAgent status="done" BUT reproduction evidence incomplete:
{{
  "action": "delegate_task",
  "reasoning": "SubAgent claims done but reproduction evidence is incomplete: missing [(a)|(b)|(c)]. Cannot trust 'tests pass' claim without it.",
  "params": {{
    "task_instruction": "Do NOT modify code further. Your only job this round: (1) ensure /testbed/reproduce_issue.py exists and demonstrates the bug from the issue, (2) run it to confirm the bug status pre-fix is reproducible (if the patched code is already in place, you may need git stash to check pre-fix behavior), (3) run it on the patched code and quote the exact output. Finish with done ONLY if all three are demonstrated.",
    "context": "⚠️ Prior round reported done without reproduction evidence.\\n- Code changes claimed: [what SubAgent said it fixed]\\n- Missing: reproduction script / before-fix run / after-fix run",
    "model": "one of {sub_models}"
  }}
}}

If SubAgent status="done" BUT tests fail or other gaps:
{{
  "action": "delegate_task",
  "reasoning": "SubAgent reported done but [specific issue]: tests show [failure details]",
  "params": {{
    "task_instruction": "CRITICAL: Previous fix incomplete. [specific next steps needed]",
    "context": "⚠️ ISSUE: [what failed]\\n- ✅ DONE: [completed work]\\n- ❌ TODO: [remaining work]",
    "model": "one of {sub_models}"
  }}
}}

If SubAgent status="partial":
{{
  "action": "delegate_task",
  "reasoning": "SubAgent made partial progress: [summary]. Need to [next steps]",
  "params": {{
    "task_instruction": "Continue: [specific next steps based on SUBTASK HISTORY]",
    "context": "From previous attempt:\\n- ✅ WORKED: [keep these]\\n- ❌ FAILED: [avoid these]",
    "model": "one of {sub_models}"
  }}
}}
""".strip()
