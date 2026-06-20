"""Pydantic response models for the on-demand context-compile API.

These shapes are the contract handed directly to the downstream generation
pipelines: `character.visual_description` / `location.visual_description`
go to the video diffusion model, `voice_description` / `voice_reference_audio_uri`
go to XTTS, and `sfx_prompts` go to the SFX (Stable Audio) model.
"""
from __future__ import annotations

from pydantic import BaseModel


class DialogueLinePayload(BaseModel):
    character_id: int
    character_name: str
    line: str
    emotion: str
    delivery: str


class CharacterContextPayload(BaseModel):
    character_id: int
    name: str
    visual_description: str
    voice_description: str
    voice_reference_audio_uri: str | None
    emotional_state: str | None
    profile: dict


class LocationContextPayload(BaseModel):
    location_id: int
    name: str
    visual_description: str
    lighting_state: str | None
    ambient_sfx_prompt: str
    profile: dict


class GenerationContextPayload(BaseModel):
    """The fully pre-compiled payload for one paragraph beat."""

    paragraph_id: int
    book_id: int
    sequence_index: int
    chapter_number: int
    raw_text: str
    camera_framing: str
    action_summary: str
    characters: list[CharacterContextPayload]
    location: LocationContextPayload | None
    dialogue_script: list[DialogueLinePayload]
    sfx_prompts: list[str]
    narrative_context: str


# ---------------------------------------------------------------------------
# Library / reader endpoints (app/routers/books.py)
# ---------------------------------------------------------------------------
class BookSummaryPayload(BaseModel):
    book_id: int
    title: str
    author: str | None
    ingestion_status: str
    paragraph_count: int


class ParagraphPayload(BaseModel):
    """A bare paragraph for the reader view -- text + identity only, no
    resolved state. State is fetched on demand once the reader selects a
    span of text and presses "query"."""

    paragraph_id: int
    sequence_index: int
    chapter_number: int
    raw_text: str


class BatchContextRequest(BaseModel):
    paragraph_ids: list[int]


class ComposeSceneRequest(BaseModel):
    paragraph_ids: list[int]


class ComposedScenePayload(BaseModel):
    """Output of the scene-consolidation step: every paragraph the reader's
    selection touched, merged into one self-contained scene description plus
    a flattened text prompt ready for a video/audio generation model."""

    book_id: int
    paragraph_ids: list[int]
    sequence_index_range: tuple[int, int]
    selected_text: str
    characters: list[CharacterContextPayload]
    location: LocationContextPayload | None
    dialogue_script: list[DialogueLinePayload]
    sfx_prompts: list[str]
    camera_framing: str
    video_prompt: str
    audio_prompt: str
