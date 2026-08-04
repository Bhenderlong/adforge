"""
Per-platform hard constraints.

These are format rules enforced BEFORE anything reaches an API: character
limits, media dimensions, how many hashtags read as native vs desperate.
Getting these wrong is the single most common way automated posting looks
automated, so they are checked in code rather than left to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    # Hard API limit. Copy longer than this is regenerated, never truncated -
    # a truncated post is worse than a shorter one.
    max_chars: int
    supports_image: bool = True
    supports_video: bool = False
    # (width, height) for generated feed imagery.
    image_size: tuple[int, int] = (1200, 675)
    video_size: tuple[int, int] | None = None
    video_seconds: int = 0
    # Hashtag count that reads as native on this platform. LinkedIn tolerates
    # 3; Instagram expects many; X punishes more than 2 with reach throttling.
    hashtags: tuple[int, int] = (0, 2)
    # How the platform treats a URL in the post body:
    #   "ok"          - normal, include it
    #   "demoted"     - allowed but costs reach (LinkedIn); prefer first comment
    #   "unclickable" - renders as plain text (Instagram, TikTok); never include
    link_policy: str = "ok"
    notes: str = ""
    # Tone shaping passed to the copywriter.
    register: str = ""


SPECS: dict[str, PlatformSpec] = {
    "x": PlatformSpec(
        key="x",
        label="X",
        max_chars=280,
        supports_image=True,
        supports_video=True,
        image_size=(1600, 900),
        video_size=(1280, 720),
        video_seconds=45,
        hashtags=(0, 2),
        register="Terse and declarative. Lead with the claim, not the setup. "
        "No 'excited to announce'. Threads only when the idea genuinely needs "
        "more than one beat.",
    ),
    "linkedin": PlatformSpec(
        key="linkedin",
        label="LinkedIn",
        max_chars=3000,
        image_size=(1200, 627),
        supports_video=True,
        video_size=(1920, 1080),
        video_seconds=60,
        hashtags=(2, 3),
        link_policy="demoted",
        notes="Links in the body are demoted; put the URL in the first comment.",
        register="Professional but human. A concrete observation or number in "
        "the first two lines, because everything after that is behind 'see "
        "more'. No motivational-poster cadence, no single-sentence paragraphs "
        "stacked for drama.",
    ),
    "facebook": PlatformSpec(
        key="facebook",
        label="Facebook",
        max_chars=2000,
        image_size=(1200, 630),
        supports_video=True,
        video_size=(1280, 720),
        video_seconds=60,
        hashtags=(0, 2),
        register="Plain and conversational. Assume a mixed audience that may "
        "not know the jargon; explain the value before the mechanism.",
    ),
    "instagram": PlatformSpec(
        key="instagram",
        label="Instagram",
        max_chars=2200,
        image_size=(1080, 1350),
        supports_video=True,
        video_size=(1080, 1920),
        video_seconds=30,
        hashtags=(5, 10),
        link_policy="unclickable",
        notes="Links are not clickable in captions; CTA must say 'link in bio'.",
        register="Visual-first. The caption supports the image, it does not "
        "repeat it. Hook in the first line before the 'more' cutoff.",
    ),
    "tiktok": PlatformSpec(
        key="tiktok",
        label="TikTok",
        max_chars=2200,
        supports_image=False,
        supports_video=True,
        video_size=(1080, 1920),
        video_seconds=35,
        hashtags=(3, 5),
        link_policy="unclickable",
        notes="Content Posting API publishes to DRAFTS only until the app "
        "passes TikTok's audit. Unaudited apps cannot post directly live.",
        register="Fast. The first 1.5 seconds decide everything - open on the "
        "payoff or a sharp problem statement, never on a logo or a greeting.",
    ),
    "youtube": PlatformSpec(
        key="youtube",
        label="YouTube",
        max_chars=5000,
        supports_image=False,
        supports_video=True,
        video_size=(1080, 1920),
        video_seconds=50,
        hashtags=(2, 3),
        notes="Vertical <=60s is published as a Short automatically.",
        register="Descriptive and searchable. The description is an SEO "
        "surface - state plainly what the video shows.",
    ),
    "reddit": PlatformSpec(
        key="reddit",
        label="Reddit",
        max_chars=40000,
        image_size=(1200, 675),
        supports_video=True,
        video_size=(1280, 720),
        video_seconds=60,
        hashtags=(0, 0),
        notes="Every target subreddit has its own self-promotion rule; the "
        "adapter refuses to post to a sub not explicitly allowlisted.",
        register="Write like a practitioner posting to peers. Marketing "
        "cadence is detected instantly and downvoted. Lead with the technical "
        "substance; the product is context, not the subject. Hashtags never.",
    ),
    "discord": PlatformSpec(
        key="discord",
        label="Discord",
        max_chars=2000,
        image_size=(1200, 675),
        supports_video=True,
        video_size=(1280, 720),
        video_seconds=60,
        hashtags=(0, 0),
        register="Peer-to-peer, low ceremony. Announcements are short and "
        "factual; no press-release voice.",
    ),
    "slack": PlatformSpec(
        key="slack",
        label="Slack",
        max_chars=3000,
        image_size=(1200, 675),
        hashtags=(0, 0),
        register="Internal-update voice. Direct, skimmable, no marketing gloss.",
    ),
}

# Platforms where a generated video is the primary artifact.
VIDEO_FIRST = ("tiktok", "youtube")


def spec(key: str) -> PlatformSpec:
    try:
        return SPECS[key]
    except KeyError:
        raise ValueError(f"unknown platform {key!r}; known: {sorted(SPECS)}") from None
