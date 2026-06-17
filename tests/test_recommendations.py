"""Tests for gap-closing recommendations (the /recommend endpoint and module).

Mirrors the existing API test style: a FakeLLM is injected via
app.dependency_overrides so nothing touches the network, and the recommendation
path is asserted to be PII-free and cleanly separated from tailoring.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app, get_llm, get_provider
from resume_tailor import JobDescription, Resume, analyze_gap
from resume_tailor.embeddings import TfidfEmbeddingProvider
from resume_tailor.models import (
    Bullet,
    ContactInfo,
    Entry,
    EntryKind,
    JobRequirement,
)
from resume_tailor.recommendations import recommend_for_gaps
from resume_tailor.tailoring import LLMClient


class FakeLLM:
    """Records the last prompt and returns schema-valid recommendation JSON."""

    def __init__(self) -> None:
        self.last_user: str | None = None
        self.last_system: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return json.dumps(
            {
                "items": [
                    {
                        "requirement_id": "r1",
                        "requirement_text": "Writing automated tests",
                        "suggestions": [
                            {
                                "kind": "certification",
                                "title": "ISTQB Foundation Level",
                                "detail": "Entry-level software testing credential.",
                                "effort": "weeks",
                            },
                            {
                                "kind": "course",
                                "title": "Test-Driven Development with pytest",
                                "detail": "Learn to write tests first.",
                                "effort": "weeks",
                            },
                            {
                                "kind": "project",
                                "title": "Add a test suite to an existing repo",
                                "detail": "Cover a real project with pytest + CI.",
                                "effort": "weekend",
                            },
                            {
                                "kind": "experience",
                                "title": "QA-focused internship",
                                "detail": "Seek a role that owns test coverage.",
                                "effort": "months",
                            },
                        ],
                    }
                ]
            }
        )


def _resume() -> Resume:
    return Resume(
        name="Jane Doe",
        contact=ContactInfo(email="jane@example.com", links=["github.com/jane"]),
        entries=[
            Entry(
                kind=EntryKind.experience,
                title="Support Agent",
                organization="ShopCo",
                bullets=[Bullet(text="Helped customers with device issues.")],
            )
        ],
    )


def _jd() -> JobDescription:
    return JobDescription(
        title="Software Engineering Intern",
        requirements=[
            JobRequirement(text="Writing automated tests"),
            JobRequirement(text="Experience with Python"),
        ],
    )


def test_recommend_for_gaps_is_pii_free() -> None:
    """The prompt must carry requirement text only — never resume identity."""
    resume = _resume()
    jd = _jd()
    provider = TfidfEmbeddingProvider()
    gap = analyze_gap(resume, jd, provider)
    llm = FakeLLM()

    recommend_for_gaps(gap, jd, llm)

    assert llm.last_user is not None
    # Requirement text and the job title are allowed (employer-side, not PII).
    assert "Writing automated tests" in llm.last_user
    assert "Software Engineering Intern" in llm.last_user
    # No candidate identity may appear anywhere in the prompt.
    for leak in ["Jane Doe", "jane@example.com", "github.com/jane", "ShopCo"]:
        assert leak not in llm.last_user
    # And no resume bullet text either — recommendations are about the gap,
    # not the candidate's existing experience.
    assert "Helped customers" not in llm.last_user


def test_recommend_skips_covered_requirements() -> None:
    """Covered requirements must not be sent for recommendations."""
    resume = _resume()
    jd = _jd()
    provider = TfidfEmbeddingProvider()
    gap = analyze_gap(resume, jd, provider)
    # Force one requirement to 'covered' so we can assert it's excluded.
    if gap.coverages:
        gap.coverages[0].status = gap.coverages[0].status.__class__("covered")
    llm = FakeLLM()

    recommend_for_gaps(gap, jd, llm)
    assert llm.last_user is not None
    assert f"id={gap.coverages[0].requirement_id}" not in llm.last_user


def test_recommend_endpoint_returns_grouped_suggestions() -> None:
    """POST /recommend returns suggestions grouped per requirement."""
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_provider] = TfidfEmbeddingProvider
    client = TestClient(app)

    resume = _resume()
    jd = _jd()
    provider = TfidfEmbeddingProvider()
    gap = analyze_gap(resume, jd, provider)

    resp = client.post(
        "/recommend",
        json={"gap": gap.model_dump(), "job_description": jd.model_dump()},
    )
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert body["items"][0]["requirement_id"] == "r1"
    kinds = {s["kind"] for s in body["items"][0]["suggestions"]}
    assert kinds == {"certification", "course", "project", "experience"}


def test_fake_llm_satisfies_protocol() -> None:
    """Guard: the test double really implements the LLMClient protocol."""
    assert isinstance(FakeLLM(), LLMClient)
