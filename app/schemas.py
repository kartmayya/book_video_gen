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


class LocationContextPayload(BaseModel):
    location_id: int
    name: str
    visual_description: str
    lighting_state: str | None
    ambient_sfx_prompt: str


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
