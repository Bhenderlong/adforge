# AdForge

Self-hosted, fully open-source social marketing automation for **Inferix**
(inferix.co) and **Vallorix** (vallorix.ai). Two separate brands with separate
voices, claims and content — never a shared post.

Everything runs on your own hardware: copy from Ollama or vLLM, images and video
from ComfyUI, publishing through each platform's official API.

```
┌── brands/*.yaml ──────── voice, approved claims, pillars, hard rules
│
├── copywriter ─────────── angle → draft → rules gate → critic → rewrite
│      ↓
├── media ──────────────── SDXL still per post; Wan 2.2 i2v shorts for TikTok/YT
│      ↓
├── scheduler ──────────── slots per brand/platform → review window → publish
│      ↓
└── platforms ──────────── X, LinkedIn, Facebook, Instagram, TikTok,
                           YouTube, Reddit, Discord, Slack

    radar ──────────────── scan Reddit/Discord → score → draft disclosed reply
                           → human approves → send
```

## Quick start

```bash
cd ~/adforge
./venv/bin/python run.py --check     # preflight: models, engine, ffmpeg, GPUs
./venv/bin/python run.py             # UI + scheduler on http://127.0.0.1:8770
```

Everything ships **disabled and dry-run**. Nothing reaches a platform until you
enable an account *and* turn dry-run off.

Recommended first hour:

1. **Settings** — confirm the writer model. Leave global dry run **on**.
2. **Compose** — generate one post per brand. Read them. Adjust
   `brands/*.yaml` on the **Brands** page until the voice is right.
3. **Accounts** — add credentials for one platform (see *Getting credentials*
   below), press **Test connection**, then keep dry-run on and watch the log
   show you the exact payload it would send.
4. **Schedules** — set a cadence. Let it generate for a day with nothing live.
5. Only then turn a single account live.

## The UI

| Page | What it does |
|---|---|
| **Dashboard** | LLM/ComfyUI/GPU health, live destinations, queue, running jobs |
| **Queue** | Everything generated. Filter, open, edit, approve, reject, publish |
| **Compose** | One-off post outside the schedule, with an optional angle |
| **Schedules** | Posts/day, days, pinned times, window, jitter, pillars, media |
| **Radar** | Targets and keywords; scored threads; drafted replies to approve |
| **Accounts** | Credentials, enable, AUTO/MANUAL, per-account dry run |
| **Brands** | Edit brand YAML with validation and automatic rollback |
| **Settings** | Models, ComfyUI, Wan, review window, caps, timezone, dry run |

The UI has **no authentication**. Keep it on `127.0.0.1` unless you put an
authenticating reverse proxy in front — it holds every credential you enter.

## How copy quality is enforced

A draft must pass a deterministic gate *and* an LLM critic before it can be
published. The gate is code, not prompting, because some of it is a legal matter:

- **AI tells** — ~50 phrases and structures that mark text as machine-written
  ("delve into", "unlock the power", "It's not X, it's Y", rhetorical openers,
  emoji bullets).
- **Fabricated metrics** — the writer will happily assert "boosts throughput by
  25%" with no benchmark behind it. Unsourced performance numbers are blocked
  and the copy is pushed toward the mechanism instead, which is better marketing
  anyway. Figures traceable to the brand's `proof_points` are allowed through.
- **Unverifiable claims** — user counts, "trusted by", "#1", funding.
- **Vallorix hard rules** — the product may never be described as *holding* SOC 2
  or ISO 27001; it automates those programmes for its users. Also no promised
  audit timelines, no guaranteed outcomes, no "replaces your auditor".
- **Framework controls** — a post citing SOC 2 or ISO 27001 is checked against
  real control data (154 controls). The model confidently mismaps criteria (it
  produced a CC7.1 post about vendor risk, which is CC9.2) and emits retired
  ISO 27001:2013 numbering. Both are blocked, and the real control text is
  injected into the prompt so it writes about a control that exists.
- **Platform format** — length, hashtag count, link policy. Copy over the limit
  is regenerated, never truncated.

Anything that fails the bar is routed to a human regardless of the account mode.

### What the gates cannot do

**The technical-claim checker is noisy, deliberately.** Measured on real output:
whole-post checking caught a genuinely wrong claim 0 times in 4 runs; checking
in chunks caught it 3/3 but also flagged 2 of 4 correct posts. A second pass
re-judging each flag against the full text made both numbers worse, so it was
removed.

Chunking was kept because the asymmetry favours it - a wrong claim in front of
engineers is expensive, a false flag costs a click. Treat a **check facts**
badge as "read this post", not "this post is wrong". On long Inferix technical
posts it will fire often.


They catch invented product claims, invented benchmarks, invented control
numbers, banned phrasing and incoherent structure. They **cannot verify
arbitrary technical assertions**. A local model will state that an RTX 4090
hits NVLink bottlenecks; it has no NVLink. The factcheck flags claims it
believes are wrong, but that is one model's opinion about another's output and
it is neither complete nor always right.

The practical consequence, worth knowing before you set a review window to 0:

- **Vallorix content is the more reliable of the two.** Its posts are grounded
  in real control data, so the failure mode is caught rather than plausible.
- **Inferix engineering content needs you to read it.** The claims are about
  hardware and inference behaviour, which nothing here can check. Treat it as a
  strong first draft, not finished copy.

Keep a review window on the Inferix technical pillars. `tips`, `cost` and
`ai_news` are where a confident wrong statement is most likely and most
expensive - a technical audience tests what you tell them.

## Getting credentials

Four platforms need an OAuth flow, and **none of them shows you a usable token
in a settings page**. YouTube Studio's Channel ID and User ID are not
credentials; Meta's console shows a user token that expires in an hour and is
the wrong kind anyway; TikTok shows nothing at all. In each case the credential
is the *output* of an authorisation you perform once:

```bash
./venv/bin/python run.py --auth youtube  --client-id ... --client-secret ...
./venv/bin/python run.py --auth linkedin --client-id ... --client-secret ...
./venv/bin/python run.py --auth meta     --client-id ... --client-secret ...   # facebook + instagram
./venv/bin/python run.py --auth tiktok   --client-id ... --client-secret ...
```

Run any of them **with no arguments** and it prints that platform's setup steps
first. Register `http://localhost:8791/callback` as the redirect URI (YouTube
uses its own loopback port and needs a **Desktop app** client, not "Web").

The rest you copy directly — `--auth <name>` tells you where:

| Platform | Where |
|---|---|
| **Reddit** | `old.reddit.com/prefs/apps` → create app → type **script**. The `client_id` is the *unlabelled* string under the app name. |
| **X** | `developer.x.com` → your app → Keys and tokens. Set the app to **Read + Write before** generating the access token, or it 403s. |
| **Discord** | A channel webhook URL (Edit Channel → Integrations), or a bot token for the radar. |
| **Slack** | `api.slack.com/apps` → OAuth & Permissions → Bot token. |

**Then press "Test connection" on the Accounts page.** It makes a read-only
identity call to the platform and reports who you are authenticated as — which
also catches valid credentials pointing at the wrong account. Field-presence
checking cannot tell you that; asking the platform can.

## Per-platform notes that actually bite

| Platform | What to know |
|---|---|
| **X** | Media still uploads through v1.1; the post itself is v2. Both credential sets required. |
| **LinkedIn** | Body links are demoted, so the URL is posted as the **first comment** automatically. `author_urn` is `urn:li:organization:<id>` for a company page. |
| **Facebook** | Needs a **page** token, not a user token. |
| **Instagram** | The API cannot accept a file upload — Meta fetches the media from a public HTTPS URL, so `public_media_base` must be internet-reachable. |
| **TikTok** | Until your app passes TikTok's audit it can **only reach drafts**. You finish posting in the app. Not a limitation of this tool. |
| **YouTube** | Vertical ≤60s is classified as a Short automatically; there is no API flag. If the consent screen is left in **Testing**, the refresh token expires after 7 days — click PUBLISH APP. |
| **Reddit** | Refuses to post to any subreddit you have not explicitly allowlisted, and omits links unless you have confirmed they are permitted. The allowlist is bound to the subreddit *name* it was granted for, so retyping the field revokes it. **2FA on the account breaks script auth entirely** — the 401 looks exactly like a wrong password. |
| **Discord** | A webhook is enough to post. A bot token is only needed for the radar. |

## The radar, and what it deliberately does not do

The radar finds threads where your product genuinely answers the question,
drafts a reply that is useful on its own merits, attaches a disclosure, and
**requires you to approve it**.

It does **not** generate replies posing as an unaffiliated user who "just tried
Inferix". That was in the original brief and is not implemented, because:

- Undisclosed endorsements by a company's own people breach the **FTC
  Endorsement Guides (16 CFR Part 255)**, and the FTC's 2024 rule on fake
  reviews and testimonials attaches civil penalties per violation. That is a
  legal exposure, not a platform-rules technicality.
- Reddit escalates undisclosed promotion to a **sitewide domain ban**, after
  which nobody can link inferix.co anywhere on Reddit. That is effectively
  permanent and costs far more traffic than astroturfing could produce.

Disclosed, substantive replies are welcome in most technical communities and
outperform astroturf because they survive instead of being removed. The policy
check in `radar/policy.py` blocks testimonial phrasing, hard-sell language,
referral spam and link-stuffing even if a model produces them, and re-checks at
send time in case the text was edited.

## GPU arbitration

A 70B writer needs essentially both RTX 5090s, and it OOMs while ComfyUI holds a
checkpoint. The box therefore runs **one mode at a time** behind a re-entrant
file lock: entering text mode tells ComfyUI to unload, entering media mode tells
Ollama to unload. Switching costs a model reload, so generation batches copy
first, then media.

This only works on Ollama — vLLM owns its memory for the process lifetime. With
vLLM, run ComfyUI on another machine.

## Configuration

Everything is env-overridable with an `ADFORGE_` prefix, in `.env` (chmod 600,
gitignored) or the environment. The Settings page writes this file.

```bash
ADFORGE_DRY_RUN=true
ADFORGE_MODEL_WRITER=llama3.3:70b
ADFORGE_MODEL_FALLBACKS=qwen2.5:14b-instruct,hermes3:latest
ADFORGE_REVIEW_WINDOW_MINUTES=60
ADFORGE_DAILY_POST_CAP=6
ADFORGE_COMFY_URL=http://127.0.0.1:8189
ADFORGE_WAN_VAE=wan_2.1_vae.safetensors
ADFORGE_TIMEZONE=America/New_York
```

**Do not set a reasoning model** (qwen3.x, glm-4.x, deepseek-r1) as writer or
critic. They spend the budget on a chain of thought and return an empty answer,
which surfaces as blank posts.

**`wan_vae` must be the 2.1 VAE** despite the 2.2 model names. `wan2.2_vae`
belongs to the TI2V-5B variant and has a different latent channel count; pairing
it with the 14B i2v experts fails with a 36-vs-64 channel mismatch.

## Running as a service

```bash
python run.py            # UI + scheduler (one process, fine for one machine)
python run.py --worker   # scheduler only, if you want the UI separate
```

## Licence and dependencies

All components are open source and commercially usable: Ollama/vLLM, the model
weights you choose, ComfyUI, SDXL/RealVisXL, Wan 2.2 (Apache-2.0), FastAPI,
SQLAlchemy, APScheduler, Pillow, ffmpeg.
