"""
Copy gate regression tests.

Every case here is a defect that actually shipped and was caught by reading
real generated output, not a hypothetical. They run offline - no model, no
GPU - so they are cheap enough to run on every change.
"""

from __future__ import annotations

import pytest

from adforge import reference
from adforge.brands import get_brand
from adforge.llm import rules
from adforge.platforms.spec import spec


@pytest.fixture(scope="module")
def inferix():
    return get_brand("inferix")


@pytest.fixture(scope="module")
def vallorix():
    return get_brand("vallorix")


def codes(text, brand, platform):
    return [v.code for v in rules.blocking(rules.check(text, brand, spec(platform)))]


# --- fabricated metrics -----------------------------------------------------
# The first real generation asserted "boosts serving throughput by 25%" with no
# benchmark behind it.

@pytest.mark.parametrize(
    "text",
    [
        "5-minute TTL prompt caching boosts serving throughput by 25%. https://inferix.co",
        "Continuous batching gives 3x faster inference on any card. https://inferix.co",
        "Cuts your training time by 40 percent. https://inferix.co",
        "Saves you 10 hours a week on deployment. https://inferix.co",
    ],
)
def test_invented_benchmarks_are_blocked(text, inferix):
    assert "fabricated_metric" in codes(text, inferix, "x")


def test_arithmetic_the_reader_can_check_is_allowed(inferix):
    text = (
        "A 7B model in fp16 is roughly 14GB of weights before any KV cache. "
        "Benchmark it on your own hardware first. https://inferix.co"
    )
    assert codes(text, inferix, "x") == []


# --- Vallorix legal rules ---------------------------------------------------
# Social copy gets screenshotted, so these are enforced in code rather than
# left to the prompt.

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Our platform is SOC 2 certified and ISO 27001 compliant.",
         "claims_own_certification"),
        ("We hold ISO 27001 and maintain SOC 2 across the business.",
         "claims_own_certification"),
        ("Get SOC 2 in just 30 days, guaranteed to pass your audit.",
         "promises_audit_timeline"),
        ("Vallorix replaces your auditor entirely.", "replaces_auditor"),
    ],
)
def test_vallorix_never_claims_to_hold_a_certification(text, expected, vallorix):
    assert expected in codes(text, vallorix, "linkedin")


def test_vallorix_may_say_it_automates_the_programme(vallorix):
    text = (
        "Vallorix automates SOC 2 evidence collection inside your own boundary, "
        "so audit artefacts never leave your network. #soc2 #grc"
    )
    assert codes(text, vallorix, "linkedin") == []


# --- framework control citations -------------------------------------------
# The model mapped CC7.1 to vendor risk (that is CC9.2) and emitted retired
# ISO 27001:2013 numbering.

def test_retired_2013_iso_numbering_is_blocked(vallorix):
    text = "Document access rights per A.9.2.3 before your next audit cycle."
    assert "unknown_control" in codes(text, vallorix, "linkedin")


def test_real_2022_control_passes(vallorix):
    text = (
        "ISO 27001:2022 moved access rights to A.5.18. Worth grepping your "
        "policy library before the next surveillance visit. #iso27001 #grc"
    )
    assert codes(text, vallorix, "linkedin") == []


def test_control_lookup_is_accurate():
    assert "vendor" in reference.lookup("CC9.2").lower()
    assert "vulnerabilit" in reference.lookup("CC7.1").lower()
    assert reference.lookup("A.7.10") == "Storage media"
    assert reference.lookup("A.9.2.3") is None  # 2013, retired
    assert reference.lookup("A8.24") == reference.lookup("A.8.24")  # both spellings


# --- unverifiable claims ----------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Trusted by 5000+ developers worldwide. https://inferix.co", "social_proof"),
        ("Join 10,000 teams already building. https://inferix.co", "user_counts"),
        ("The #1 GPU cloud for AI teams. https://inferix.co", "superlative_claim"),
    ],
)
def test_unverifiable_social_proof_is_blocked(text, expected, inferix):
    assert expected in codes(text, inferix, "x")


# --- model leakage ----------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Here is a great LinkedIn post for you: we automate evidence collection "
        "across many compliance frameworks for regulated teams.",
        "Sure! Here is your post about GPU rental and per-second billing for "
        "teams who need flexible compute capacity.",
        "Deploy endpoints at [INSERT URL HERE] and start building something "
        "useful with your existing SDK code today.",
    ],
)
def test_model_preamble_and_placeholders_are_blocked(text, inferix):
    assert codes(text, inferix, "linkedin")


# --- platform format --------------------------------------------------------

def test_copy_over_the_limit_is_blocked(inferix):
    assert "too_long" in codes("x" * 400, inferix, "x")


def test_links_are_blocked_where_they_are_unclickable(inferix):
    text = (
        "Real detail about per-second billing and verified GPUs, plus how to "
        "size your context window properly. https://inferix.co"
    )
    assert "link_not_allowed" in codes(text, inferix, "instagram")


def test_linkedin_body_link_warns_but_does_not_block(inferix):
    text = (
        "Concrete detail about GPU economics and why depreciation dominates "
        "the total cost of ownership here. https://inferix.co #gpu #ml"
    )
    violations = rules.check(text, inferix, spec("linkedin"))
    assert not rules.blocking(violations)
    assert any(v.code == "link_demoted" for v in violations)


def test_hashtag_ceiling_is_enforced(inferix):
    text = "Solid technical point about VRAM here. #a #b #c #d #e https://inferix.co"
    assert "too_many_hashtags" in codes(text, inferix, "x")


# --- AI tells ---------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase",
    ["delve into", "unlock the power of", "in today's fast-paced world",
     "game-changer", "seamless", "excited to announce"],
)
def test_cliche_phrases_are_blocked(phrase, inferix):
    text = f"We {phrase} GPU infrastructure for teams that ship. https://inferix.co"
    assert "ai_tell" in codes(text, inferix, "x")


def test_antithesis_cliche_is_flagged_in_both_forms(inferix):
    for text in [
        "It's not just a GPU cloud, it's your whole pipeline in one account.",
        "It is not just a GPU cloud, it is your whole pipeline in one account.",
    ]:
        violations = rules.check(text + " https://inferix.co", inferix, spec("x"))
        assert any(v.code == "antithesis_cliche" for v in violations), text
