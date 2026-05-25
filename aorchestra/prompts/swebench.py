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
4. CHECK BEFORE/AFTER EVIDENCE before considering 'submit':
   A 'submit' is only valid when SUBTASK HISTORY (or SubAgent's finish message)
   shows a BEFORE/AFTER evidence pair in ONE of the three shapes below — quote
   the lines to yourself.

   Shape R (Runtime reproduction) — when the issue describes a crash / exception /
   wrong output that triggers at runtime:
     - BEFORE: a script (typically /testbed/reproduce_issue.py) was run on the
       base code and reproduced the error/wrong-output described in the issue.
     - AFTER:  the same script was re-run after the fix and exits cleanly /
       prints the correct output.

   Shape S (Spec conformance) — when the issue is a version bump, new
   constraint, default-behavior change, or deprecation that does NOT crash at
   runtime in the current env:
     - BEFORE: a minimal demonstration (mock.patch on an edge value, or a
       targeted assertion) showed the base code accepts an input the new spec
       forbids, or rejects an input the new spec requires.
     - AFTER:  the same demonstration shows the new behavior matches the spec.

   Shape T (Test-suite delta) — when there is an existing repo test that maps
   to the issue, or SubAgent wrote a targeted pytest case:
     - BEFORE: running `pytest path/to/test.py::test_X` on the base code shows
       the test failing with a quoted assertion / error.
     - AFTER:  same command shows it passing.

   ANY ONE of R / S / T is sufficient. What is NOT sufficient: a SubAgent
   claiming "tests passed" without a BEFORE state, or running only a
   self-authored sanity script that never demonstrates the base-code failure.
   If BEFORE or AFTER is missing/unclear, DO NOT submit — delegate one more
   round asking SubAgent to fill the missing half.
5. DECIDE:
   - ✅ status="done" AND a valid BEFORE/AFTER pair (R, S, or T) is present
        → Use 'submit'
   - ⚠️ status="done" BUT evidence is one-sided or absent
        → Use 'delegate_task' asking SubAgent to produce the missing half
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
Return JSON. The 'reasoning' field MUST identify the evidence shape (R / S / T)
and quote the BEFORE and AFTER lines from SUBTASK HISTORY when choosing 'submit'.

If SubAgent status="done" AND a valid BEFORE/AFTER pair is present:
{{
  "action": "submit",
  "reasoning": "Evidence shape <R|S|T>. BEFORE: <quote base-state failure>. AFTER: <quote post-fix success>. Submitting for grading.",
  "params": {{ "reason": "Fix verified by <shape> evidence: [specific fix description]" }}
}}

If SubAgent status="done" BUT evidence is one-sided or absent:
{{
  "action": "delegate_task",
  "reasoning": "SubAgent claims done but BEFORE/AFTER evidence is incomplete: missing <BEFORE state | AFTER state>. Cannot trust 'tests pass' claim without it.",
  "params": {{
    "task_instruction": "Do NOT modify code further. Your only job this round: produce the missing half of the BEFORE/AFTER pair. Pick the shape that fits the issue: (R) run reproduce_issue.py on the unfixed code to show the runtime error from the issue; (S) for spec-change issues, write a minimal demonstration (mock or assertion) that shows the base code violates the new spec; (T) identify a repo test or write a targeted pytest case, run it on base to show failure. Then re-run after the fix to show the matching AFTER state. Quote both outputs verbatim. Finish with done ONLY when both halves are demonstrated.",
    "context": "⚠️ Prior round reported done without a complete BEFORE/AFTER pair.\\n- Code changes claimed: [what SubAgent said it fixed]\\n- Missing: [BEFORE state | AFTER state | both]",
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
