#!/usr/bin/env python
"""
AdForge entrypoint.

    python run.py            # web UI + scheduler
    python run.py --check    # environment preflight, exits non-zero on a problem
    python run.py --worker   # scheduler only, no UI
"""

from __future__ import annotations

import argparse
import logging
import sys

from adforge.config import DATA, settings


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(DATA / "logs" / "adforge.log"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors").setLevel(logging.WARNING)


def check() -> int:
    """Preflight. Prints what is and is not usable, without changing anything."""
    from adforge.brands import all_brands
    from adforge.gpu import vram
    from adforge.llm.client import available_models, health
    from adforge.media import comfy

    problems = 0
    print(f"AdForge preflight\n{'=' * 60}")

    try:
        brands = all_brands()
        print(f"[ ok ] brands: {', '.join(b.name for b in brands.values())}")
        for b in brands.values():
            print(f"       {b.name}: {len(b.proof_points)} claims, "
                  f"{len(b.pillars)} pillars, {len(b.hard_rules)} hard rules")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] brands: {e}")
        problems += 1

    ok, msg = health()
    print(f"[{' ok ' if ok else 'FAIL'}] llm ({settings.llm_backend} at "
          f"{settings.llm_base}): {msg}")
    if not ok:
        problems += 1
        models = available_models()
        if models:
            print(f"       available: {', '.join(models[:12])}"
                  f"{' ...' if len(models) > 12 else ''}")

    if comfy.is_up():
        cks = comfy.checkpoints()
        print(f"[ ok ] comfyui at {settings.comfy_url}: {len(cks)} usable checkpoints")
        if settings.comfy_checkpoint not in cks:
            print(f"[warn] configured checkpoint {settings.comfy_checkpoint!r} "
                  f"is not in the engine's list")
            problems += 1
        if settings.enable_video:
            unets = comfy.unet_models()
            for name, label in [(settings.wan_high_noise, "wan high-noise"),
                                (settings.wan_low_noise, "wan low-noise")]:
                if name not in unets:
                    print(f"[warn] {label} model {name!r} not found")
                    problems += 1
            vaes = comfy.list_options("VAELoader", "vae_name")
            if settings.wan_vae not in vaes:
                print(f"[warn] wan vae {settings.wan_vae!r} not found")
                problems += 1
            elif "2.2" in settings.wan_vae:
                print(f"[warn] wan_vae is {settings.wan_vae!r} - the 14B i2v "
                      f"experts need the 2.1 VAE; 2.2 fails with a channel "
                      f"mismatch")
                problems += 1
    else:
        print(f"[warn] comfyui unreachable at {settings.comfy_url} - "
              f"posts will generate without media")

    import subprocess
    try:
        out = subprocess.run([settings.ffmpeg, "-version"], capture_output=True,
                             text=True, timeout=30)
        print(f"[ ok ] ffmpeg: {out.stdout.splitlines()[0][:70]}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] ffmpeg not usable: {e}")
        problems += 1

    for g in vram():
        print(f"[ info ] GPU{g['index']} {g['name']}: "
              f"{g['used_mb'] // 1024}/{g['total_mb'] // 1024} GB ({g['pct']}%)")

    print(f"{'=' * 60}")
    print(f"dry_run={settings.dry_run}  review_window={settings.review_window_minutes}m"
          f"  daily_cap={settings.daily_post_cap}  tz={settings.timezone}")
    if settings.dry_run:
        print("Global DRY RUN is on: nothing will be transmitted to any platform.")
    else:
        print("Global DRY RUN is OFF: enabled accounts will publish for real.")
    print(f"\n{problems} problem(s).")
    return 1 if problems else 0


MUTATIONS = [
    # (label, file, find, replace) - each disables one guard. The suite must
    # fail for every one of them, otherwise that guard is untested and a later
    # refactor can remove it silently. Written after finding a test that passed
    # on the exact bug it was meant to catch.
    ("dry run defaults to live with no account", "adforge/platforms/registry.py",
     "    if account is None:\n        return True",
     "    if account is None:\n        return False"),
    ("anti-astroturf policy disabled", "adforge/radar/policy.py",
     "    for pat, code in SHILL_PATTERNS:", "    for pat, code in []:"),
    ("reddit allowlist disabled", "adforge/platforms/community.py",
     '        if not options.get("allowed") or granted.lower() != sub.lower():',
     "        if False:"),
    ("vallorix legal rules disabled", "adforge/llm/rules.py",
     '    if brand.key == "vallorix":', "    if False:"),
    ("republish guard removed", "adforge/platforms/registry.py",
     "    if post.remote_id and not dry:", "    if False:"),
    ("score uses the average not the worst", "adforge/llm/copywriter.py",
     "        return min(vals)", "        return sum(vals)/len(vals)"),
    ("fabricated metrics allowed", "adforge/llm/rules.py",
     "    for pat in FABRICATED_METRIC:", "    for pat in []:"),
    ("fabricated anecdotes allowed", "adforge/llm/rules.py",
     "    for pat, code in FABRICATED_ANECDOTE:", "    for pat, code in []:"),
]


def selftest() -> int:
    """Preflight, the test suite, then check the suite can actually fail.

    A green suite proves nothing on its own - one of these tests originally
    passed on the exact routing bug it was written to catch. The mutation pass
    breaks each safety guard in turn and requires the suite to notice.
    """
    import pathlib
    import subprocess

    print("=" * 62)
    print("1. environment")
    print("=" * 62)
    env = check()

    print("\n" + "=" * 62)
    print("2. test suite")
    print("=" * 62)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else "no output")
    if r.returncode:
        print("\nTests are failing; fix those before trusting anything below.")
        return 1

    print("\n" + "=" * 62)
    print("3. mutation spot-checks - can the suite actually fail?")
    print("=" * 62)
    survived = []
    for label, path, find, repl in MUTATIONS:
        f = pathlib.Path(path)
        orig = f.read_text()
        if find not in orig:
            print(f"  SKIP     {label} (anchor moved - update MUTATIONS)")
            survived.append(label + " [anchor moved]")
            continue
        f.write_text(orig.replace(find, repl, 1))
        try:
            m = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                               capture_output=True, text=True)
        finally:
            f.write_text(orig)  # always restore, even on ctrl-c
        if m.returncode:
            print(f"  caught   {label}")
        else:
            print(f"  SURVIVED {label}  <- that guard is untested")
            survived.append(label)

    print("\n" + "=" * 62)
    if survived:
        print(f"{len(survived)} guard(s) not covered by any test:")
        for s in survived:
            print(f"  - {s}")
        return 1
    print("environment ok, tests pass, and every safety guard is covered.")
    return 0 if not env else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="AdForge")
    ap.add_argument("--check", action="store_true", help="preflight and exit")
    ap.add_argument("--auth", metavar="PLATFORM",
                    help="run a one-time OAuth flow (youtube) and print the "
                         "credentials to paste into Accounts")
    ap.add_argument("--client-id", default="", help="OAuth client id for --auth")
    ap.add_argument("--client-secret", default="", help="OAuth client secret for --auth")
    ap.add_argument("--secrets-file", default="",
                    help="path to the client_secret JSON downloaded from Google")
    ap.add_argument("--selftest", action="store_true",
                    help="preflight + test suite + mutation spot-checks")
    ap.add_argument("--worker", action="store_true", help="scheduler only, no UI")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    if args.check:
        return check()

    if args.selftest:
        return selftest()

    if args.auth:
        from adforge.platforms.oauth_helper import (SETUP_STEPS, print_result,
                                                    youtube_auth)

        from adforge.platforms.oauth_flows import FLOWS

        want = args.auth.lower()

        if want == "youtube":
            if not (args.client_id or args.secrets_file):
                print(SETUP_STEPS)
                return 1
            creds = youtube_auth(args.client_id, args.client_secret,
                                 args.secrets_file)
            print_result("YouTube", creds)
            return 0

        if want in FLOWS:
            fn, setup = FLOWS[want]
            if not (args.client_id and args.client_secret):
                print(setup)
                return 1
            creds = fn(args.client_id, args.client_secret)
            print_result(want.title(), creds)
            return 0

        print(f"no OAuth flow for {args.auth!r}.")
        print("\nWith a helper:  youtube, linkedin, meta (facebook+instagram), tiktok")
        print("\nCopy directly from the platform:\n"
              "  reddit    old.reddit.com/prefs/apps -> create app -> type SCRIPT\n"
              "            the client_id is the unlabelled string under the app name\n"
              "  x         developer.x.com -> your app -> Keys and tokens\n"
              "            set the app to Read+Write BEFORE generating the token\n"
              "  discord   a channel webhook URL (Edit Channel -> Integrations),\n"
              "            or a bot token from discord.com/developers\n"
              "  slack     api.slack.com/apps -> OAuth & Permissions -> Bot token")
        return 1

    if args.worker:
        import time

        from adforge.db import init_db
        from adforge.scheduler.engine import build_scheduler

        init_db()
        sched = build_scheduler()
        sched.start()
        logging.info("worker running; ctrl-c to stop")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            sched.shutdown()
        return 0

    import os

    import uvicorn

    # PORT wins over the configured value so a supervisor or dev harness can
    # place the server wherever it needs to without editing settings.
    port = int(os.environ.get("PORT") or settings.port)
    uvicorn.run("adforge.ui.app:app", host=settings.host, port=port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
