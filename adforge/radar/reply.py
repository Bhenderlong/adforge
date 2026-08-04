"""
Drafting replies to radar threads.

Every draft is written to be useful on its own merits, carries a disclosure,
and requires human approval before it is sent. See policy.py for why the
"unaffiliated happy user" framing in the original brief is not implemented.

The prompt is deliberately built around answering the question first. A reply
that leads with the product is removed by mods and downvoted by readers, so the
constraint is a performance requirement as much as an ethical one.
"""

from __future__ import annotations

import logging

from ..brands import get_brand
from ..config import settings
from ..db import RadarThread, ReplyDraft, PostStatus
from ..gpu import TEXT, gpu
from ..llm.client import LLMError, chat_with_fallback
from ..llm.copywriter import _clean
from . import policy

log = logging.getLogger("adforge.radar")

SYSTEM = """\
You are an experienced engineer replying in a technical community. You work on
the product in question, you say so plainly, and you are here because you can
answer the question.

How you reply:
- Answer the actual question FIRST, concretely and completely. If the best
  answer does not involve your product, give that answer anyway.
- Mention your product at most once, only where it is genuinely the relevant
  option, and describe what it does rather than why it is good.
- Say what it does NOT do, or where another tool fits better, whenever that is
  true. This is what makes the reply credible.
- Match the register of a forum comment: plain, direct, no marketing cadence,
  no bullet-point pitch deck.
- Length: two to four short paragraphs. Long enough to be substantive, short
  enough to read.

NEVER do any of these - they get the comment removed and the account banned:
- Pretend to be an unaffiliated user. Never write "I just used X", "I tried X
  and it was great", "X has been a lifesaver", or any personal testimonial.
- "You should check out", "highly recommend", "game-changer", or any sales
  phrasing.
- Post a link when you have been told links are not permitted here.
- Reply at all if you have nothing useful to say. In that case output exactly:
  SKIP

Output ONLY the reply text. No preamble, no signature, no subject line.
"""


def draft_reply(thread: RadarThread, links_allowed: bool = False) -> ReplyDraft | None:
    """Draft one reply. Returns None when the model declines the thread."""
    brand = get_brand(thread.brand)

    link_rule = (
        f"You may include the single link {brand.url} IF and only if it "
        "directly answers what they asked."
        if links_allowed
        else "Links are NOT permitted in this community. Do not include any URL."
    )

    user = f"""YOUR PRODUCT: {brand.name} - {brand.tagline}
WHAT IT DOES: {brand.about.strip()}

WHAT IT CAN HONESTLY CLAIM (nothing beyond this):
{chr(10).join(f'  - {p}' for p in brand.proof_points)}
{_hard_rules(brand)}
WHERE YOU ARE REPLYING: {thread.where}
{link_rule}

THREAD TITLE: {thread.title}
THREAD BODY:
{thread.excerpt[:2000]}

WHY THIS THREAD WAS SURFACED: {thread.score_reason}

Write the reply, or output SKIP if you cannot add real value here."""

    try:
        with gpu(TEXT):
            raw, _ = chat_with_fallback(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
                settings.writer_chain,
                temperature=0.8,
                max_tokens=900,
            )
    except LLMError as e:
        log.warning("reply drafting failed for thread %s: %s", thread.id, e)
        return None

    body = _clean(raw)
    if not body or body.strip().upper().startswith("SKIP"):
        log.info("model declined thread %s (%s)", thread.id, thread.where)
        return None

    problems = policy.check_reply(body, thread.brand, links_allowed)
    blocking = policy.blocking(problems)

    # One retry against the specific objections. Shill phrasing is usually a
    # single sentence and the second attempt normally clears it.
    if blocking:
        log.info("draft for thread %s rejected: %s", thread.id, "; ".join(blocking[:3]))
        try:
            with gpu(TEXT):
                raw, _ = chat_with_fallback(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": body},
                        {
                            "role": "user",
                            "content": "That reply was rejected for:\n"
                            + "\n".join(f"  - {p}" for p in blocking)
                            + "\n\nRewrite it. Answer the question and drop "
                            "every promotional phrasing. Output only the reply.",
                        },
                    ],
                    settings.writer_chain,
                    temperature=0.6,
                    max_tokens=900,
                )
            body = _clean(raw)
            problems = policy.check_reply(body, thread.brand, links_allowed)
            blocking = policy.blocking(problems)
        except LLMError as e:
            log.warning("reply revision failed: %s", e)

    # The disclosure is attached separately from the body so that editing the
    # text in the UI cannot remove it; it is re-applied at send time too.
    draft = ReplyDraft(
        thread_id=thread.id,
        brand=thread.brand,
        body=body,
        disclosure=policy.disclosure_for(thread.brand),
        status=PostStatus.REVIEW,
        quality_score=0.0 if blocking else float(thread.relevance or 0.0) * 10,
        critic_notes="; ".join(problems[:6]),
    )
    if blocking:
        draft.critic_notes = "STILL FAILING POLICY: " + draft.critic_notes
    return draft


def _hard_rules(brand) -> str:
    if not brand.hard_rules:
        return ""
    return "\n!! ABSOLUTE RULES:\n" + "\n".join(
        f"  - {r.strip()}" for r in brand.hard_rules
    )


def send_reply(draft: ReplyDraft, thread: RadarThread, account, dry_run: bool) -> str:
    """Post an approved reply. Returns the resulting URL, or '' on dry run.

    The disclosure is appended here, after any human edits, so a reply can
    never go out without it.
    """
    body = policy.ensure_disclosure(draft.body, draft.brand)

    if dry_run:
        log.info(
            "[DRY RUN] would reply to %s (%s):\n%s", thread.url, thread.where, body
        )
        return ""

    if thread.source == "reddit":
        import praw

        creds = account.creds()
        reddit = praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            username=creds["username"],
            password=creds["password"],
            user_agent=creds.get("user_agent") or "adforge-radar",
        )
        submission = reddit.submission(id=thread.external_id.removeprefix("t3_"))
        comment = submission.reply(body)
        return f"https://reddit.com{comment.permalink}"

    if thread.source == "discord":
        import httpx

        creds = account.creds()
        channel_id = thread.where.split("/")[-1]
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                headers={"Authorization": f"Bot {creds['bot_token']}"},
                json={
                    "content": body,
                    "message_reference": {
                        "message_id": thread.external_id.removeprefix("dm_")
                    },
                },
            )
        if r.status_code >= 400:
            raise RuntimeError(f"discord reply failed: {r.text[:300]}")
        return thread.url

    raise RuntimeError(f"cannot reply on source {thread.source!r}")
