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


# --- caught only by running it for real -------------------------------------
# All three shipped past the gate at score 8.0 in a live batch.

@pytest.mark.parametrize(
    "text",
    [
        "40% of our users batch inference requests to minimize their GPU costs.",
        "Most of our customers batch requests to cut spend on rented hardware.",
        "3 in 4 of our teams hit this ceiling before they change hardware.",
    ],
)
def test_invented_user_statistics_are_blocked(text, inferix):
    """A proportion-of-customers claim is a survey nobody ran.

    "40% of our users batch inference requests" scored 8.0 and passed. The old
    rule required the number to sit immediately before the noun. On a
    commercial account this is a false-advertising problem, not just a
    credibility one.
    """
    assert "invented_user_statistic" in codes(text, inferix, "linkedin")


@pytest.mark.parametrize(
    "text",
    [
        "Batching 10 requests saves $0.06 per hour on a mid-range card.",
        "Inference runs about $0.40 per hour on that tier for most workloads.",
        "This cuts $200 a month off a typical training budget for small teams.",
    ],
)
def test_invented_costs_are_blocked(text, inferix):
    """A costing claim invites someone to hold you to it."""
    assert "fabricated_metric" in codes(text, inferix, "linkedin")


@pytest.mark.parametrize(
    "text",
    [
        "I once spent hours tracking a mystery daemon crash before finding it.",
        "Last week I traced an auth failure through three layers of journals.",
        "I learned this the hard way after an access review went sideways.",
        "A client of ours hit this during their first surveillance audit.",
    ],
)
def test_first_person_anecdotes_are_blocked_on_a_brand_account(text, vallorix):
    """Nobody at the company did this - it reads as a lie because it is one.

    The radar's reply policy has blocked invented personal stories since it was
    written; posts never did, and a live run opened a Vallorix tweet with
    "I once spent hours tracking a mystery daemon crash".
    """
    assert "fabricated_anecdote" in codes(text, vallorix, "x")


def test_real_technical_content_still_passes(inferix, vallorix):
    """The new rules must not suppress the posts that are actually good."""
    assert codes(
        "A 7B model in fp16 is about 14GB of weights before any KV cache is "
        "allocated, which is why a 16GB card dies at long context.",
        inferix, "linkedin",
    ) == []
    assert codes(
        "For A.8.5 secure authentication auditors want logs of every privileged "
        "access attempt, not a checkbox confirming MFA is enabled. #grc #iso27001",
        vallorix, "linkedin",
    ) == []
