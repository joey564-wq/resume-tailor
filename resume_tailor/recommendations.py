"""LLM-generated suggestions for *closing* a skill gap — kept strictly separate
from resume tailoring.

The integrity line this module respects
---------------------------------------
Tailoring (``tailoring.py``) re-presents experience the candidate ALREADY has,
and the grounding validator stops it inventing anything. This module does the
opposite kind of work: for requirements the candidate does NOT yet cover, it
suggests concrete ways to *acquire* that experience — certifications, courses,
project ideas, and the kind of role/internship to seek.

Two rules keep that honest:

1. These suggestions are about the FUTURE (things to go get), never claims about
   the present. They must be surfaced in their own clearly-labelled section and
   MUST NOT feed back into tailored bullets. Suggesting "earn the AWS SAA cert"
   is fine; writing a bullet that implies the candidate already holds it is the
   exact fabrication the grounding validator exists to prevent.

2. PRIVACY BY CONSTRUCTION. The prompt is built from job-requirement text only —
   which describes the employer, not the candidate. No resume, no bullets, no
   name/contact ever enters this prompt path. There is therefore no channel for
   candidate PII to leak, which is why this module needs no resume-PII guard:
   it never receives a resume in the first place.

Suggestions are LLM-generated from the model's own knowledge (no web lookups),
so treat specific names/prices as directional rather than authoritative — the
value is the *direction* to grow, not a live catalogue.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from .gap_analysis import GapReport
from .models import JobDescription
from .tailoring import LLMClient


class SuggestionKind(StrEnum):
    """The four ways to close a gap, matching what the UI groups by."""

    certification = "certification"
    course = "course"
    project = "project"
    experience = "experience"  # role / internship / volunteer work to seek


class Suggestion(BaseModel):
    """One concrete, actionable way to close a specific gap."""

    kind: SuggestionKind
    title: str
    detail: str = ""
    # Rough effort marker so the UI can sort quick wins vs long plays.
    effort: str = ""  # e.g. "weekend", "weeks", "months"


class RequirementSuggestions(BaseModel):
    """All suggestions for a single uncovered/partial requirement."""

    requirement_id: str
    requirement_text: str
    suggestions: list[Suggestion] = Field(default_factory=list)


class RecommendationReport(BaseModel):
    """The full set of gap-closing recommendations returned by /recommend."""

    items: list[RequirementSuggestions] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompting — requirement text only (no resume, no PII channel)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a career-development advisor. Given job requirements a
candidate does NOT yet meet, suggest concrete, realistic ways to CLOSE each gap.

For every requirement, suggest a mix across these kinds where they make sense:
- "certification": a recognised credential that demonstrates the skill
- "course": specific learning (a course, specialization, or focused topic)
- "project": something buildable that would create real, demonstrable experience
- "experience": a kind of role, internship, or volunteer work to seek out

HARD RULES — obey every one:
- Suggest things to ACQUIRE in the future. Never phrase a suggestion as if the
  candidate already has it.
- Be realistic and specific. Prefer widely-recognised, accessible options.
- You are working from general knowledge, not a live catalogue: do not invent
  exact prices, dates, or URLs. Name well-known options and keep details
  directional.
- Give 2-4 suggestions per requirement, spread across kinds (not four certs).
- Return ONLY valid JSON, no prose, no markdown fences.

JSON schema:
{"items":[{"requirement_id":"<id>","requirement_text":"<text>","suggestions":[
  {"kind":"certification|course|project|experience","title":"<short>",
   "detail":"<one sentence on what it is / why it helps>",
   "effort":"weekend|weeks|months"}]}]}
"""


def build_recommend_prompt(
    gaps: Sequence[tuple[str, str]],
    jd: JobDescription,
) -> str:
    """Build the user prompt from (requirement_id, requirement_text) pairs only.

    Like `tailoring.build_user_prompt`, this has no parameter for any resume or
    candidate field by design — there is no path for identity data to enter.
    The job title is included for context (it describes the employer).
    """
    gap_lines = "\n".join(f'- id={rid}: "{text}"' for rid, text in gaps)
    return (
        f"TARGET ROLE: {jd.title}\n\n"
        "The candidate does not yet meet these requirements. Suggest concrete "
        "ways to close each gap.\n\n"
        f"UNMET REQUIREMENTS:\n{gap_lines}\n"
    )


def parse_recommendations(raw: str) -> RecommendationReport:
    """Parse the model's JSON into a validated RecommendationReport.

    Fence-tolerant, mirroring tailoring.parse_tailored. Raises on malformed
    input so the caller can re-prompt or fail cleanly.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
        text = text.strip()
    data = json.loads(text)
    items = [RequirementSuggestions(**item) for item in data["items"]]
    return RecommendationReport(items=items)


def recommend_for_gaps(
    gap_report: GapReport,
    jd: JobDescription,
    llm: LLMClient,
    *,
    include_partial: bool = True,
    max_attempts: int = 2,
) -> RecommendationReport:
    """Suggest ways to close every uncovered (and optionally partial) gap.

    Sends only requirement ids + text to the LLM — no resume, no bullets, no PII.
    Re-prompts once on malformed JSON, then fails cleanly with an empty report
    rather than raising into the request handler.
    """
    statuses = {"uncovered"} | ({"partial"} if include_partial else set())
    gaps: list[tuple[str, str]] = [
        (c.requirement_id, c.requirement_text)
        for c in gap_report.coverages
        if c.status.value in statuses
    ]
    if not gaps:
        return RecommendationReport(items=[])

    user = build_recommend_prompt(gaps, jd)
    for _ in range(max_attempts):
        raw = llm.complete(SYSTEM_PROMPT, user)
        try:
            return parse_recommendations(raw)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            user += (
                "\nYour previous reply was not valid JSON matching the schema. "
                "Return ONLY the JSON object, no prose, no fences."
            )
    return RecommendationReport(items=[])
