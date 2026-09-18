# Music beds for shorts (data/music/)

Drop licensed audio files here (`.mp3 .m4a .wav .aac .flac .ogg`) and every
rendered short gets one mixed in at `music_gain_db` (default -20dB) with a
fade in and out. A track shorter than the short is looped; the video length is
unchanged.

Selection is **deterministic** — the same post always gets the same track, so
re-rendering to review a copy change does not also change the music.

**With this directory empty, shorts still get a silent AAC stream.** That is
deliberate: a file with no audio stream at all is treated as malformed by some
platform encoders, which is a different problem from being quiet.

## Where to get tracks you may actually use

Nothing is bundled, because nothing here is licensed to be redistributed.

| Source | Terms |
|---|---|
| YouTube Audio Library | Free. Some tracks require attribution; **cleared for YouTube, not necessarily elsewhere** |
| Pixabay Music | Free, broad licence, no attribution required |
| Free Music Archive | Per-track licence — check each one, several are non-commercial only |
| Epidemic Sound / Artlist / Musicbed | Subscription, cleared for all platforms including ads |

If you are posting the same short to YouTube *and* TikTok *and* Instagram, a
YouTube-only licence is not enough. A paid subscription is the only
uncomplicated answer for cross-platform commercial use.

## What NOT to put here

Anything from a streaming service, anything ripped, and anything whose licence
you have not read. Platform audio fingerprinting is automatic and immediate:
the consequence is a muted or blocked video, and on a business account it can
escalate further.
