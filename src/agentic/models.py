"""
agent/models.py

Shared dataclasses for the agentic loop.

  ActionStep       — a single tool call with expected outcome
  Plan             — ordered list of ActionSteps with metadata
  ExecutionResult  — outcome of running one ActionStep
  ReflectionResult — outcome of the Reflector's post-execution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING   = auto()
    SUCCESS   = auto()
    FAILED    = auto()
    SKIPPED   = auto()


class PlanStatus(Enum):
    PENDING    = auto()
    PARTIAL    = auto()   # some steps done, some failed
    COMPLETE   = auto()
    ABANDONED  = auto()   # exceeded MAX_REPLAN_ATTEMPTS


# ---------------------------------------------------------------------------
# ActionStep
# ---------------------------------------------------------------------------

@dataclass
class ActionStep:
    """
    One discrete unit of work the Executor will carry out.

    tool             : logical name matching a key in AgentOrchestrator._tools
    args             : keyword arguments forwarded to the tool callable
    expected_outcome : plain-English description used by the Reflector
                       to verify success (e.g. "task 'Buy milk' exists in DB")
    status           : updated by Executor / Reflector in-place
    result           : raw return value from the tool call
    error            : exception message if the step raised
    attempts         : how many times this step has been tried
    """
    tool:             str
    args:             dict[str, Any]        = field(default_factory=dict)
    expected_outcome: str                   = ""
    status:           StepStatus            = StepStatus.PENDING
    result:           Any                   = None
    error:            str | None            = None
    attempts:         int                   = 0


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    Ordered sequence of ActionSteps produced by the Planner.

    goal             : natural-language description of what the user wants
    steps            : mutable list; Replanner may append / replace entries
    status           : updated as execution proceeds
    replan_count     : incremented each time the Replanner mutates this plan
    created_at       : timestamp for log_event
    """
    goal:          str
    steps:         list[ActionStep]   = field(default_factory=list)
    status:        PlanStatus         = PlanStatus.PENDING
    replan_count:  int                = 0
    created_at:    datetime           = field(default_factory=datetime.now)

    # Convenience helpers

    @property
    def pending_steps(self) -> list[ActionStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def failed_steps(self) -> list[ActionStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    @property
    def succeeded_steps(self) -> list[ActionStep]:
        return [s for s in self.steps if s.status == StepStatus.SUCCESS]

    @property
    def is_fully_complete(self) -> bool:
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
            for s in self.steps
        )


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Thin wrapper returned by Executor.run_step()."""
    step:     ActionStep
    response: str          = ""   # human-readable output from the tool
    ok:       bool         = True


# ---------------------------------------------------------------------------
# ReflectionResult
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """
    Produced by Reflector.reflect() after each execution pass.

    success      : True if every required step succeeded
    score        : 0.0–1.0  (succeeded / total non-skipped steps)
    unmet_goals  : plain-English list of what still needs doing
    summary      : one-paragraph human-facing summary of the whole attempt
    should_replan: True when unmet goals exist AND replan budget remains
    """
    success:       bool
    score:         float
    unmet_goals:   list[str]  = field(default_factory=list)
    summary:       str        = ""
    should_replan: bool       = False