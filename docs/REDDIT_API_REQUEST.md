# Reddit Data API access request — prepared answers

Everything below is verifiable against this repository. Do not inflate it. If
access is granted on these representations and the tool later behaves
differently, that is a revocation, so the numbers are deliberately the real
ones from the code, not comfortable ones.

Source of each figure:
- `adforge/radar/scan.py`  — `SEARCH_LIMIT = 120`, `PREFILTER_LIMIT = 60`
- `adforge/config.py`      — `radar_interval_minutes = 30`, `radar_max_replies_per_day = 8`
- `adforge/platforms/community.py` — the only write path
- `adforge/radar/policy.py` — disclosure enforcement

---

## Use case

Owner-operated marketing automation for two of my own companies — Inferix
(inferix.co, GPU cloud for AI workloads) and Vallorix (vallorix.ai, self-hosted
compliance automation). Single operator, one Reddit account, self-hosted on my
own hardware. Not a product, not resold, no third-party users, no public
instance.

The tool does two things on Reddit:

1. **Scheduled self-posts** to subreddits I have individually allowlisted in
   the tool after reading their self-promotion rules. It refuses to post to a
   subreddit that has not been explicitly confirmed.
2. **Search for threads where my product is a genuine answer**, draft a reply,
   and hold it in a queue until I personally approve it. Every reply carries a
   material-connection disclosure ("I work on Inferix") that is appended at
   send time and cannot be edited out. Replies posing as an unaffiliated happy
   user are blocked in code, not merely discouraged by a prompt.

Cap: **8 replies per day maximum**, each individually human-approved.

## Endpoints and target data

| Call | Endpoint | Volume |
|---|---|---|
| Sitewide search | `/r/all/search` — `sort=new`, `time_filter=week`, `limit=120` | 2 requests per target per scan |
| Single-subreddit listing | `/r/{sub}/new` — `limit=60` | 1 request per target per scan |
| Submit | `/api/submit` | ≤ a few per day, scheduled |
| Reply | `/api/comment` | ≤ 8 per day, each human-approved |

Scan interval is 30 minutes. With a handful of targets that is on the order of
**100–300 requests per day**, i.e. well under 1 request/minute against a
per-client limit measured in the hundreds. There is no bulk collection, no
historical backfill, no firehose consumption, and no crawling of user profiles.

Retained per matched thread: **id, title, excerpt, author name, permalink,
timestamp, relevance score**. Retained so I can review a drafted reply before
sending it, and to avoid re-surfacing the same thread. Nothing is republished,
resold, or shared.

## Model use — stated explicitly

Thread titles and bodies are passed to a **locally-hosted LLM running on my own
GPUs** to score whether a reply would be relevant. This is **inference only**.
Reddit content is **not used to train or fine-tune any model**, is not retained
as a training corpus, and never leaves my machine — the model is local, so no
Reddit content is sent to any third-party AI provider.

## Why this cannot be built on Devvit

I have shipped a Devvit app, so this is not unfamiliarity with the platform.
Four things make it the wrong runtime here, in descending order of importance:

1. **Reddit is one of nine destinations.** The same post and campaign schedule
   also goes to X, LinkedIn, Facebook, Instagram, TikTok, YouTube, Discord and
   Slack. A Devvit app cannot call those APIs, and the scheduling, review
   window and approval queue are shared across all of them. Reddit is a
   destination in an existing system, not the system.

2. **The generation runs on my own hardware.** Copy is written by a local
   Llama 3.3 70B via Ollama, and every image and video is rendered by a local
   ComfyUI instance on two RTX 5090s. Devvit's sandbox cannot reach a private
   LAN, cannot invoke local GPUs, and has no execution budget for a multi-
   minute diffusion render.

3. **Devvit apps are installed per-subreddit by moderators.** The whole point of
   the search step is to find threads in communities I am not a moderator of
   and that have not installed anything. An app can only act where it is
   installed, which is precisely the set of places this does not need to
   search.

4. **It acts as me.** These are disclosed replies from a founder, posted from a
   personal account. A subreddit-installed app bot is a different actor with
   different norms, and posting founder disclosures through one would be more
   confusing, not less.

## Compliance posture

- Official API via PRAW throughout. No scraping, no unauthenticated JSON
  endpoints, no browser automation, no fingerprint or proxy rotation.
- One account. No vote manipulation, no sockpuppets, no coordinated posting.
- Undisclosed endorsement is blocked in code (`adforge/radar/policy.py`) — the
  FTC Endorsement Guides (16 CFR Part 255) apply to a company's own principals,
  and disclosure is appended after human editing so it cannot be removed.
- Posting is refused to any subreddit not individually allowlisted after its
  rules were read.
- Rate limiting and backoff handled by PRAW defaults; no retry storms.

## "What subreddits do you intend to use the bot/app in?"

Answer honestly, because the source is linked and the two scopes differ:

> **Search is sitewide.** It uses Reddit's own search endpoint over r/all for a
> short keyword list. That is the only way to find the threads worth answering,
> and it is a read-only use of the official search API.
>
> **Posting is restricted to an allowlist** I confirm per-subreddit after reading
> that subreddit's self-promotion rules. Currently: r/LocalLLaMA, r/comfyui,
> r/selfhosted, r/homelab (Inferix); r/cybersecurity, r/AskNetsec (Vallorix).
> The adapter refuses any subreddit not on it.
>
> **Replies can occur wherever a matching thread is**, since that is inherent to
> sitewide search. Three constraints apply to those: I approve each one
> individually, they are capped at 8/day across all of Reddit, and a reply to a
> thread found sitewide **never carries a link** — links require me to have
> allowlisted that specific subreddit first. So the default posture in a
> community whose rules I have not read is a disclosed, link-free answer.

## Contact

(email on the form) inferix.co, vallorix.ai
