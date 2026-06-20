"""Scene-consolidation step: merges one or more per-paragraph generation
contexts (as resolved by app/context_compiler.py) into a single
self-contained scene description, plus flattened text prompts for a
downstream video model and audio (XTTS dialogue / Stable Audio SFX) model.

A reader's highlighted span can cross paragraph boundaries, so this is where
multiple `GenerationContextPayload`s -- each potentially naming overlapping
characters/locations with slightly different deltas -- get deduplicated
into one coherent "current state of the scene" view.

This is deterministic merge logic today. It is intentionally isolated
behind `compose_scene()` so it can be swapped for an LLM-driven consolidator
later (e.g. to rewrite the merged action summary in natural prose) without
touching the API layer or the callers in app/routers/books.py.
"""
from __future__ import annotations

from app.schemas import (
    CharacterContextPayload,
    ComposedScenePayload,
    DialogueLinePayload,
    GenerationContextPayload,
    LocationContextPayload,
)


def _dedupe_characters(payloads: list[GenerationContextPayload]) -> list[CharacterContextPayload]:
    """Last paragraph in sequence order wins for a given character_id, since
    that reflects their most recent state within the selected span."""
    by_id: dict[int, CharacterContextPayload] = {}
    for payload in payloads:
        for character in payload.characters:
            by_id[character.character_id] = character
    return list(by_id.values())


def _resolve_location(
    payloads: list[GenerationContextPayload],
) -> tuple[LocationContextPayload | None, list[str]]:
    """Returns the most recent (last) active location plus a transition note
    list describing any earlier distinct locations the span also touched."""
    locations = [p.location for p in payloads if p.location is not None]
    if not locations:
        return None, []

    current = locations[-1]
    seen_names: list[str] = []
    transitions: list[str] = []
    for location in locations:
        if location.name not in seen_names:
            seen_names.append(location.name)
    if len(seen_names) > 1:
        transitions = [name for name in seen_names if name != current.name]
    return current, transitions


def _merge_dialogue(payloads: list[GenerationContextPayload]) -> list[DialogueLinePayload]:
    merged: list[DialogueLinePayload] = []
    for payload in payloads:
        merged.extend(payload.dialogue_script)
    return merged


def _merge_sfx(payloads: list[GenerationContextPayload]) -> list[str]:
    seen: list[str] = []
    for payload in payloads:
        for prompt in payload.sfx_prompts:
            if prompt not in seen:
                seen.append(prompt)
    return seen


def _build_video_prompt(
    characters: list[CharacterContextPayload],
    location_name: str | None,
    location_description: str | None,
    location_lighting: str | None,
    location_transitions: list[str],
    camera_framing: str,
    action_summary: str,
) -> str:
    parts: list[str] = [f"Shot type: {camera_framing.replace('_', ' ')}."]

    if location_name:
        setting = f"Setting: {location_name} -- {location_description}."
        if location_lighting:
            setting += f" Lighting: {location_lighting}."
        parts.append(setting)
    if location_transitions:
        parts.append(f"Scene transitions through: {', '.join(location_transitions)}.")

    if characters:
        character_bits = []
        for character in characters:
            bit = f"{character.name} ({character.visual_description}"
            if character.emotional_state:
                bit += f", {character.emotional_state}"
            bit += ")"
            character_bits.append(bit)
        parts.append("Characters on screen: " + "; ".join(character_bits) + ".")

    parts.append(f"Action: {action_summary}")
    return " ".join(parts)


def _build_audio_prompt(
    characters: list[CharacterContextPayload],
    dialogue_script: list[DialogueLinePayload],
    sfx_prompts: list[str],
    ambient_sfx_prompt: str | None,
) -> str:
    parts: list[str] = []

    if ambient_sfx_prompt:
        parts.append(f"Ambient bed: {ambient_sfx_prompt}.")

    if dialogue_script:
        voice_by_name = {c.name: c.voice_description for c in characters}
        dialogue_bits = []
        for line in dialogue_script:
            voice = voice_by_name.get(line.character_name, "")
            voice_note = f" [voice: {voice}]" if voice else ""
            dialogue_bits.append(
                f'{line.character_name} ({line.emotion}, {line.delivery}){voice_note}: "{line.line}"'
            )
        parts.append("Dialogue: " + " | ".join(dialogue_bits))

    if sfx_prompts:
        parts.append("SFX cues: " + "; ".join(sfx_prompts) + ".")

    return " ".join(parts) if parts else "No dialogue or SFX cues in this span."


def compose_scene(payloads: list[GenerationContextPayload]) -> ComposedScenePayload:
    """Consolidates an ordered (by sequence_index) list of per-paragraph
    contexts -- the raw output of compile_contexts() for the reader's
    highlighted span -- into one merged scene payload."""
    if not payloads:
        raise ValueError("compose_scene requires at least one paragraph context")

    payloads = sorted(payloads, key=lambda p: p.sequence_index)
    book_id = payloads[0].book_id

    characters = _dedupe_characters(payloads)
    location, location_transitions = _resolve_location(payloads)
    dialogue_script = _merge_dialogue(payloads)
    sfx_prompts = _merge_sfx(payloads)

    selected_text = "\n\n".join(p.raw_text for p in payloads)
    camera_framing = payloads[-1].camera_framing
    action_summary = " Then, ".join(p.action_summary for p in payloads)

    video_prompt = _build_video_prompt(
        characters=characters,
        location_name=location.name if location else None,
        location_description=location.visual_description if location else None,
        location_lighting=location.lighting_state if location else None,
        location_transitions=location_transitions,
        camera_framing=camera_framing,
        action_summary=action_summary,
    )
    audio_prompt = _build_audio_prompt(
        characters=characters,
        dialogue_script=dialogue_script,
        sfx_prompts=sfx_prompts,
        ambient_sfx_prompt=location.ambient_sfx_prompt if location else None,
    )

    return ComposedScenePayload(
        book_id=book_id,
        paragraph_ids=[p.paragraph_id for p in payloads],
        sequence_index_range=(payloads[0].sequence_index, payloads[-1].sequence_index),
        selected_text=selected_text,
        characters=characters,
        location=location,
        dialogue_script=dialogue_script,
        sfx_prompts=sfx_prompts,
        camera_framing=camera_framing,
        video_prompt=video_prompt,
        audio_prompt=audio_prompt,
    )
