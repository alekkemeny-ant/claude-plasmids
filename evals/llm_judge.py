#!/usr/bin/env python3
"""
LLM-as-Judge Grader for Agent Evals

Uses Claude (Sonnet by default) to evaluate the full agent transcript
against a structured rubric covering qualitative dimensions that the
deterministic rubric cannot assess: disambiguation quality, scientific
accuracy, tool routing, and communication quality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from evals.rubric import RubricResult


@dataclass
class JudgeScore:
    """A single dimension score from the LLM judge."""
    dimension: str
    score: int         # 1-5
    explanation: str


@dataclass
class JudgeResult:
    """Complete LLM judge evaluation result."""
    scores: list[JudgeScore] = field(default_factory=list)
    overall_score: float = 0.0  # average of all dimension scores
    raw_response: str = ""      # full LLM response for debugging

    def summary(self) -> str:
        if not self.scores:
            return "No judge scores"
        parts = [f"{s.dimension}={s.score}/5" for s in self.scores]
        return f"Judge: {self.overall_score:.1f}/5 ({', '.join(parts)})"


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a plasmid design AI agent. You assess the \
quality of the agent's responses across multiple dimensions.

You will be given:
1. The test case description and expected outcomes
2. The full agent transcript (user prompts, tool calls, and agent responses)
3. The deterministic rubric results (if available)

Score each dimension on a 1-5 scale:
- 5: Excellent — exceeds expectations
- 4: Good — meets expectations with minor issues
- 3: Adequate — acceptable but with notable gaps
- 2: Poor — significant issues
- 1: Failing — fundamentally wrong or missing

Return your evaluation as a JSON object with this exact structure:
{
  "scores": [
    {
      "dimension": "<dimension_name>",
      "score": <1-5>,
      "explanation": "<brief explanation>"
    }
  ]
}

Return ONLY the JSON object, no other text."""


JUDGE_USER_PROMPT = """\
## Test Case

**ID:** {case_id}
**Name:** {case_name}
**Description:** {case_description}
**Prompt:** {case_prompt}
**Expected backbone:** {expected_backbone}
**Expected insert:** {expected_insert}
{transcript_assertions_section}

## Scoring Dimensions

{dimensions_section}

## Agent Transcript

{transcript}

## Tool Call Sequence

{tool_calls}

{rubric_section}

Evaluate the agent's performance on each dimension listed above. Return \
your scores as JSON."""


# Dimension definitions
DIMENSION_DISAMBIGUATION = {
    "name": "disambiguation_quality",
    "description": (
        "Did the agent identify the right ambiguities? Were clarifying "
        "questions clear, informative, and scientifically appropriate? "
        "Did the agent present relevant options (e.g., species, gene family "
        "members, RFP variants) without overwhelming the user?"
    ),
}

DIMENSION_SCIENTIFIC = {
    "name": "scientific_accuracy",
    "description": (
        "Are biological explanations correct? Was the right gene/protein "
        "identified? Was an appropriate backbone chosen for the expression "
        "system? Are any biological claims accurate?"
    ),
}

DIMENSION_TOOL_ROUTING = {
    "name": "tool_routing",
    "description": (
        "Did the agent use the right tools in the right order? Did it avoid "
        "unnecessary calls (e.g., NCBI for library inserts, fuse_inserts "
        "when no fusion was requested)? Was the workflow efficient?"
    ),
}

DIMENSION_COMMUNICATION = {
    "name": "communication_quality",
    "description": (
        "Was the response clear, professional, and well-organized? Did the "
        "agent provide a useful construct summary? Was the explanation "
        "appropriate for a molecular biologist audience?"
    ),
}

ALL_DIMENSIONS = [
    DIMENSION_DISAMBIGUATION,
    DIMENSION_SCIENTIFIC,
    DIMENSION_TOOL_ROUTING,
    DIMENSION_COMMUNICATION,
]


class LLMJudge:
    """Evaluates agent transcripts using an LLM judge."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.model = model
        self.client = anthropic.Anthropic()

    def evaluate(
        self,
        case_id: str,
        case_name: str,
        case_description: str,
        case_prompt: str,
        expected_backbone: str,
        expected_insert: str,
        transcript: str,
        tool_calls: list[dict],
        transcript_assertions: list[str] | None = None,
        rubric_result: Optional["RubricResult"] = None,
    ) -> JudgeResult:
        """Evaluate an agent transcript against the structured rubric.

        Args:
            case_id: Test case ID (e.g., "A6-002").
            case_name: Test case name.
            case_description: Test case description.
            case_prompt: The original user prompt.
            expected_backbone: Expected backbone ID.
            expected_insert: Expected insert ID.
            transcript: Full agent transcript text.
            tool_calls: List of tool call dicts with "tool" and "input" keys.
            transcript_assertions: Expected strings in the transcript
                (for disambiguation cases).
            rubric_result: Deterministic rubric result (if available).

        Returns:
            JudgeResult with per-dimension scores and overall average.
        """
        # Select which dimensions to score
        has_disambiguation = bool(transcript_assertions)
        dimensions = []
        if has_disambiguation:
            dimensions.append(DIMENSION_DISAMBIGUATION)
        dimensions.append(DIMENSION_SCIENTIFIC)
        dimensions.append(DIMENSION_TOOL_ROUTING)
        dimensions.append(DIMENSION_COMMUNICATION)

        dimensions_section = "\n".join(
            f"- **{d['name']}**: {d['description']}" for d in dimensions
        )

        # Format transcript assertions section
        if transcript_assertions:
            assertions_str = ", ".join(f'"{a}"' for a in transcript_assertions)
            transcript_assertions_section = (
                f"**Transcript assertions:** Agent should mention: {assertions_str}"
            )
        else:
            transcript_assertions_section = ""

        # Format tool calls
        tool_calls_str = "\n".join(
            f"  {i+1}. {tc['tool']}({json.dumps(tc.get('input', {}))[:150]})"
            for i, tc in enumerate(tool_calls)
        ) or "  (no tool calls)"

        # Format rubric results
        if rubric_result:
            rubric_section = (
                f"## Deterministic Rubric Results\n\n"
                f"{rubric_result.summary()}\n\n"
                f"{rubric_result.report()}"
            )
        else:
            rubric_section = (
                "## Deterministic Rubric Results\n\n"
                "No rubric results available (agent may not have produced "
                "an assembled sequence)."
            )

        user_prompt = JUDGE_USER_PROMPT.format(
            case_id=case_id,
            case_name=case_name,
            case_description=case_description,
            case_prompt=case_prompt,
            expected_backbone=expected_backbone,
            expected_insert=expected_insert,
            transcript_assertions_section=transcript_assertions_section,
            dimensions_section=dimensions_section,
            transcript=transcript[:8000],  # Truncate very long transcripts
            tool_calls=tool_calls_str,
            rubric_section=rubric_section,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text  # type: ignore[union-attr]
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> JudgeResult:
        """Parse the JSON response from the judge LLM."""
        result = JudgeResult(raw_response=raw)

        # Extract JSON from the response (handle markdown code blocks)
        json_str = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find a JSON object in the text
            brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    return result
            else:
                return result

        scores_data = data.get("scores", [])
        for s in scores_data:
            score_val = s.get("score", 0)
            # Clamp to 1-5 range
            score_val = max(1, min(5, int(score_val)))
            result.scores.append(JudgeScore(
                dimension=s.get("dimension", "unknown"),
                score=score_val,
                explanation=s.get("explanation", ""),
            ))

        if result.scores:
            result.overall_score = sum(s.score for s in result.scores) / len(result.scores)

        return result
