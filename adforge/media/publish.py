"""Make a local asset publicly fetchable, for APIs that refuse uploads.

Instagram's Graph API takes a URL, not bytes: you create a media container
pointing at an https:// address and Meta's servers fetch it themselves. So a
file sitting in data/assets on this machine cannot be posted at all until it
exists somewhere on the public internet.

This pushes the one file being published to the host serving
`public_media_base`, then CONFIRMS it is fetchable before returning. Skipping
that check would trade a clear local error for Meta's opaque one - the Graph
API reports a failed fetch as a generic media-creation error that says nothing
about the URL.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger("adforge.media.publish")


class PublishMediaError(RuntimeError):
    pass


def _already_there(url: str) -> bool:
    try:
        r = httpx.head(url, timeout=15, follow_redirects=True)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_public(local: Path, base_url: str) -> str:
    """Return a public https URL for `local`, uploading it if needed.

    Raises PublishMediaError with something actionable rather than letting the
    platform fail on a URL it could not read.
    """
    base = base_url.rstrip("/")
    if not base.startswith("https://"):
        raise PublishMediaError(
            f"public_media_base must be https:// for Meta to fetch it, got {base!r}"
        )
    local = Path(local)
    if not local.exists():
        raise PublishMediaError(f"no such asset: {local}")

    url = f"{base}/{local.name}"

    target = settings.media_sync_target.strip()
    if not target:
        # No sync configured: the file may still be reachable if something else
        # publishes that directory. Verify rather than assume.
        if _already_there(url):
            return url
        raise PublishMediaError(
            f"{url} is not reachable and media_sync_target is unset, so there "
            f"is nothing to upload it with. Set media_sync_target on Settings "
            f"(e.g. 'user@host:/var/www/adforge-media/') or publish "
            f"{local.name} to {base} by other means."
        )

    dest = target if target.endswith("/") else target + "/"
    cmd = ["rsync", "-q", "--chmod=F644", "-e",
           "ssh -o BatchMode=yes -o ConnectTimeout=10", str(local), dest]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise PublishMediaError(f"timed out uploading {local.name} to {dest}") from None
    if r.returncode:
        raise PublishMediaError(
            f"upload of {local.name} to {dest} failed: "
            f"{(r.stderr or r.stdout).strip()[:300]}"
        )

    # Confirm from the outside. rsync succeeding only proves the bytes reached
    # the host - it says nothing about whether the web server will serve them,
    # which is the thing Meta actually depends on.
    if not _already_there(url):
        raise PublishMediaError(
            f"uploaded {local.name} to {dest} but {url} is still not fetchable. "
            f"Check that the web server serves that directory and that the "
            f"file extension is one it allows."
        )
    log.info("published %s -> %s", local.name, url)
    return url
