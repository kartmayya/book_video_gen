import os
import re
from typing import AsyncIterator

import httpx
import ormsgpack

from .schema import SynthesizeRequest

# Fish Speech 1.5.1 native API server (tools/api_server.py). Despite the legacy
# env var name, this is NOT an OpenAI-compatible endpoint: it serves POST /v1/tts
# with an ormsgpack-encoded ServeTTSRequest body and streams raw 16-bit PCM.
FISH_TTS_URL = os.getenv("SGLANG_TTS_URL", "http://localhost:30001")

TONE_MAP = {
    "grimly": "serious",
    "whispered": "soft",
    "shouting": "loud",
    "ominously": "dramatic",
    "softly": "gentle",
    "frantically": "anxious",
}

_TONE_PATTERN = re.compile(r"\[[\w\s]+\]")


def _extract_tone_and_clean(text: str) -> tuple[str, str | None]:
    markers = _TONE_PATTERN.findall(text)
    clean = _TONE_PATTERN.sub("", text).strip()
    style = None
    if markers:
        raw = markers[0][1:-1].strip().lower()
        style = TONE_MAP.get(raw)
    return clean, style


def _pcm_offset(buffer: bytes | bytearray) -> int | None:
    """Locate the first PCM sample in a streamed WAV body.

    Fish Speech streams a canonical RIFF/WAVE header (44 bytes) followed by raw
    int16 PCM. The mixer feeds our output straight into ffmpeg as ``-f s16le``,
    so the header must be removed. We scan for the ``data`` sub-chunk marker and
    skip it plus its 4-byte size field; everything after that is PCM.
    """
    idx = buffer.find(b"data")
    if idx == -1:
        return None
    pcm_start = idx + 8  # len(b"data") + 4-byte chunk size
    if len(buffer) < pcm_start:
        return None
    return pcm_start


async def synthesize_stream(
    req: SynthesizeRequest, client: httpx.AsyncClient
) -> AsyncIterator[bytes]:
    # Fish Speech has no tone/style parameter; we still strip the [tone] marker
    # so it is never spoken aloud.
    clean_text, _ = _extract_tone_and_clean(req.dialogue)

    # speaker_id maps to a Fish Speech reference voice. An unknown id is safe:
    # the server creates an empty references/<id>/ folder and falls back to the
    # default voice. Populate that folder with a sample .wav + .lab to pin a
    # timbre per speaker.
    payload: dict = {
        "text": clean_text,
        "format": "wav",
        "streaming": True,
        "reference_id": req.speaker_id or None,
        "chunk_length": 200,
        "normalize": True,
        "use_memory_cache": "on",
    }

    body = ormsgpack.packb(payload)

    header_stripped = False
    prefix = bytearray()
    async with client.stream(
        "POST",
        f"{FISH_TTS_URL}/v1/tts",
        content=body,
        headers={"content-type": "application/msgpack"},
        timeout=None,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            if header_stripped:
                yield chunk
                continue
            prefix.extend(chunk)
            offset = _pcm_offset(prefix)
            if offset is not None:
                header_stripped = True
                pcm = bytes(prefix[offset:])
                prefix.clear()
                if pcm:
                    yield pcm
