"""
SWE-bench SubAgent for aorchestra framework.

Uses DISCUSSION + COMMAND format (same as baseline SWEAgent),
with 'finish' command for reporting back to MainAgent.
"""
import re
from typing import Any, Dict, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo


# =============================================================================
# SWEBENCH SUBAGENT PROMPT
# =============================================================================
SWEBENCH_SUBAGENT_PROMPT = """
You are an autonomous software engineering agent tasked with solving GitHub issues.
You have access to a specialized command interface (ACI) for navigating, viewing, editing, and testing code.
You will work in a Docker container with the repository already cloned and checked out to the correct commit.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== REQUIRED WORKFLOW (Evidence-First) ====
You MUST produce a BEFORE/AFTER evidence pair before declaring `done`. Pick the
shape that fits the issue — at least one shape is always applicable.

1. EXPLORE: Find the file(s) the issue points at. If the issue mentions a
   specific module/function, `find_file` or `search_dir` for it first.
2. CHOOSE THE EVIDENCE SHAPE based on the issue:
   - Shape R (Runtime reproduction): the issue describes a crash / exception /
     wrong output that you can trigger by running code in this container.
   - Shape S (Spec conformance): the issue is a version bump, new constraint,
     default-behavior change, or deprecation — there is no runtime crash to
     trigger, only a spec the code does not yet match.
   - Shape T (Test-suite delta): the issue refers to an existing repo test, or
     you can write a targeted pytest case that maps cleanly to the issue.
3. PRODUCE THE 'BEFORE' STATE on the base code, before any fix:
   - R: write `/testbed/reproduce_issue.py` (10-30 lines) and run it; it MUST
     emit the error / wrong output described in the issue.
   - S: write a minimal demonstration (using `mock.patch` on an edge value, or
     a targeted assertion) showing the base code accepts an input the new spec
     forbids, or rejects an input the new spec requires.
   - T: run `python -m pytest path/to/test.py::test_X` on the unfixed code and
     confirm it fails with a quoted assertion / error.
   If the BEFORE state does not show the expected failure, your evidence is
   wrong — adjust it before proceeding.
4. FIX: Apply the minimal code change to fix the issue. Do not modify any test
   files under `tests/`, `test_*.py`, or `*_test.py` — the grader uses the
   original tests and your changes will be wiped before grading.
5. PRODUCE THE 'AFTER' STATE by re-running the same script / demonstration /
   pytest command from step 3. It MUST now succeed (correct output, spec
   satisfied, test passing). If it still fails, your fix is incomplete —
   iterate on step 4.
6. REGRESSION CHECK (recommended if steps remain): run the repo's existing
   test suite for the touched module, e.g. `cd /testbed && python -m pytest
   path/to/test_<module>.py -x -q`. Surface any new failures in your finish
   message rather than ignoring them.

CONTAINER PERSISTENCE: This container is reused across MainAgent rounds. If
you were re-delegated, a `reproduce_issue.py` and partial fixes from prior
rounds may already exist — check first with `ls /testbed/` and `git status`
inside `/testbed` before redoing work.

==== Context (from previous attempts) ====
{context}

==== Current State ====
{state_info}

==== Command Reference ====
{command_docs}

=== FINISH (Report to MainAgent) ===
finish <status> <message>
    Report your progress back to MainAgent. Status MUST be one of:
    - done:    A complete BEFORE/AFTER evidence pair (shape R, S, or T) is in
               place. In your <message>, identify the shape and cite the exact
               BEFORE output (failure / wrong-output / failing assertion) and
               the matching AFTER output (success / spec satisfied / test
               passing). No BEFORE/AFTER pair = do NOT use status=done.
    - partial: Made progress but the BEFORE/AFTER pair is not yet complete
               (e.g., found the bug location, made tentative edits, but the
               AFTER state still shows failure, or the BEFORE state was never
               demonstrated).

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{observation}

==== OUTPUT FORMAT (STRICT) ====
You MUST output EXACTLY two sections in this order. No other text allowed.

DISCUSSION
<your reasoning here>

COMMAND
<single command here>

RULES:
- DISCUSSION must contain your step-by-step reasoning
- COMMAND must contain exactly ONE command on a single line
- After COMMAND line, do NOT add any explanation, examples, or comments
- Do NOT output anything after the command
"""


def build_subagent_prompt(
    task_instruction: str,
    context: str,
    command_docs: str,
    state_info: str,
    memory: str,
    observation: str,
    current_step: int,
    max_steps: int,
) -> str:
    """Build the complete prompt for SubAgent."""
    remaining_steps = max_steps - current_step
    
    # Budget warning
    if remaining_steps <= 3:
        budget_warning = "🚨 CRITICAL: Only {} steps left! Use 'finish' NOW to report your progress!".format(remaining_steps)
    elif remaining_steps <= 5:
        budget_warning = "⚠️ Warning: {} steps remaining. Plan to finish soon.".format(remaining_steps)
    else:
        budget_warning = ""
    
    return SWEBENCH_SUBAGENT_PROMPT.format(
        task_instruction=task_instruction,
        context=context if context else "No additional context provided.",
        command_docs=command_docs,
        state_info=state_info,
        memory=memory,
        observation=observation,
        current_step=current_step,
        max_steps=max_steps,
        remaining_steps=remaining_steps,
        budget_warning=budget_warning,
    )


def parse_subagent_response(response: str) -> Dict[str, Any]:
    """
    Parse SubAgent response to extract DISCUSSION and COMMAND.
    Handles 'finish' command for reporting back to MainAgent.
    """
    discussion = ""
    command = ""
    
    # Pattern for DISCUSSION section
    discussion_match = re.search(
        r'DISCUSSION\s*\n(.*?)(?=\nCOMMAND|\Z)', 
        response, 
        re.DOTALL | re.IGNORECASE
    )
    if discussion_match:
        discussion = discussion_match.group(1).strip()
    
    # Pattern for COMMAND section - extract all content after COMMAND
    command_match = re.search(
        r'COMMAND\s*\n(.*?)(?=\n(?:DISCUSSION|OUTPUT)|\Z)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if command_match:
        raw_command = command_match.group(1).strip()
        
        # Only take the first meaningful line as the command
        # This prevents LLM's extra text/thoughts from being included
        lines = raw_command.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines and comment-like lines
            if line and not line.startswith('#') and not line.startswith('('):
                # For multi-line commands like str_replace, create, edit, etc.
                # we need to capture the full block
                if any(line.startswith(cmd) for cmd in ['str_replace', 'create', 'edit', 'insert', 'delete_lines']):
                    # These commands may span multiple lines - use the full raw content
                    # but stop at obvious non-command text (parenthetical notes, etc.)
                    clean_lines = []
                    for l in lines:
                        # Stop if we hit a line that looks like commentary
                        if l.strip().startswith('(') or l.strip().startswith('**') or l.strip().startswith('Wait'):
                            break
                        clean_lines.append(l)
                    command = '\n'.join(clean_lines).strip()
                else:
                    # Single-line commands: just take the first line
                    command = line
                break
        
        # If no command found, use the first line anyway
        if not command and lines:
            command = lines[0].strip()
    
    # Handle finish command (SubAgent specific)
    # Format: finish <status> <message>
    # status: done | partial
    if command.lower().startswith('finish'):
        # Extract content after 'finish'
        finish_content = command[6:].strip() if len(command) > 6 else ""
        
        # Parse status and message
        status = "done"  # default
        message = finish_content or discussion
        
        # Check if content starts with a valid status (done or partial)
        content_lower = finish_content.lower()
        if content_lower.startswith('done'):
            status = "done"
            message = finish_content[4:].strip() or discussion
        elif content_lower.startswith('partial'):
            status = "partial"
            message = finish_content[7:].strip() or discussion
        
        return {
            "action": "finish",
            "params": {
                "status": status,
                "message": message,
                "completed": [],
                "issues": [],
            },
            "reasoning": discussion
        }
    
    # Handle submit command (should not be used by SubAgent, convert to finish)
    if command.lower() == 'submit':
        return {
            "action": "finish",
            "params": {
                "status": "done",
                "message": "SubAgent attempted to submit - converting to finish",
                "completed": [],
                "issues": [],
            },
            "reasoning": discussion
        }
    
    # Regular ACI or bash command - use aci_command like Baseline
    if command:
        return {
            "action": "aci_command",
            "params": {"command": command},
            "reasoning": discussion
        }
    
    # Fallback: extract any command-like content
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('DISCUSSION'):
            return {
                "action": "aci_command",
                "params": {"command": line},
                "reasoning": response
            }
    
    return {
        "action": "error",
        "params": {"message": "Could not parse response"},
        "reasoning": response
    }


class SWEBenchSubAgent(BaseAgent):
    """
    SubAgent for SWE-bench in aorchestra framework.
    
    Uses DISCUSSION + COMMAND format (same as baseline SWEAgent),
    with 'finish' command for reporting back to MainAgent.
    """
    name: str = Field(default="SWEBenchSubAgent")
    description: str = Field(default="SubAgent for SWE-bench with ACI tools")
    
    # Task context from MainAgent
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    
    # Internal state
    current_instruction: str = Field(default="")
    command_docs: str = Field(default="")
    state_info: str = Field(default="(Open file: n/a) (Current directory: /testbed)")
    memory: Optional[Memory] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def reset(self, env_info: BasicInfo) -> None:
        """Reset agent state for new task."""
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=20)
        else:
            self.memory.clear()
        
        # Use task_instruction if set by MainAgent, otherwise use env instruction
        self.current_instruction = self.task_instruction or env_info.instruction
        self.command_docs = env_info.meta_data.get("command_docs", env_info.action_space)
        self.state_info = "(Open file: n/a) (Current directory: /testbed)"
        
        # Store original question for reference
        if not self.original_question:
            self.original_question = env_info.instruction
        
        logger.info(f"[SWEBenchSubAgent] Reset for task")

    def parse_action(self, resp: str) -> Dict[str, Any]:
        """Parse LLM response using DISCUSSION + COMMAND format."""
        return parse_subagent_response(resp)

    def _get_memory(self) -> str:
        """Get formatted memory context."""
        if self.memory:
            return self.memory.as_text()
        return "No previous observations."

    async def step(self, observation: Any, history: Any, current_step: int = 1, max_steps: int = 50) -> tuple:
        """Execute one step of the agent loop.
        
        Returns:
            tuple: (action, raw_response, raw_input_prompt)
        """
        # Format observation
        if isinstance(observation, dict):
            obs_str = observation.get("output", str(observation))
            if "state_info" in observation:
                self.state_info = observation["state_info"]
        else:
            obs_str = str(observation)
        
        # Build prompt
        prompt = build_subagent_prompt(
            task_instruction=self.current_instruction,
            context=self.context,
            command_docs=self.command_docs,
            state_info=self.state_info,
            memory=self._get_memory(),
            observation=obs_str,
            current_step=current_step,
            max_steps=max_steps,
        )
        
        logger.log_to_file(LogLevel.INFO, f"[SWEBenchSubAgent] Step {current_step} Input:\n{prompt}\n")
        
        # Query LLM
        try:
            resp = await self.llm(prompt)
        except Exception as e:
            logger.error(f"[SWEBenchSubAgent] LLM call failed: {e}")
            resp = "DISCUSSION\nLLM call failed, reporting back.\nCOMMAND\nfinish LLM call failed"
        
        # Parse response
        action = self.parse_action(resp)
        logger.agent_action(f"[SWEBenchSubAgent] Action: {action}")
        
        # Extract reasoning for memory
        reasoning = action.get("reasoning", "")
        
        # Update memory
        if self.memory:
            await self.memory.add_memory(
                obs=obs_str,
                action=action,
                thinking=reasoning,
                raw_response=resp
            )
        
        return action, resp, prompt

    async def run(self, request: Optional[str] = None) -> str:
        """Main run method (not used in benchmark loop)."""
        return ""
